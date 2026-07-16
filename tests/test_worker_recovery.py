from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from lexi_service.jobs.effects import JobEffects
from lexi_service.jobs.leases import JobLeaseRepository
from lexi_service.jobs.models import JobRow, JobStatus, ServiceBase
from lexi_service.jobs.outbox import OutboxEnvelope
from lexi_service.worker.main import Worker


@pytest.fixture
async def worker_parts():
    engine = create_async_engine("sqlite+aiosqlite://", poolclass=StaticPool)
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    async with engine.begin() as connection:
        await connection.run_sync(ServiceBase.metadata.create_all)
    now = datetime.now()
    async with sessions.begin() as session:
        session.add(
            JobRow(
                job_id="job-1",
                owner_subject="svc",
                owner_tenant="",
                operation="generate",
                idempotency_key="key",
                status="queued",
                payload_version=1,
                payload_json="{}",
                payload_hash="0" * 64,
                reference_dataset_fingerprint="d",
                accepted_at=now,
                expires_at=now + timedelta(hours=1),
                max_retries=2,
            )
        )
    try:
        yield JobLeaseRepository(sessions), sessions
    finally:
        await engine.dispose()


async def test_crashed_worker_lease_is_reclaimed_and_next_worker_completes(worker_parts):
    leases, sessions = worker_parts
    now = datetime.now(timezone.utc)
    old_claim = await leases.claim("job-1", now, timedelta(seconds=1))
    assert old_claim
    assert await leases.recover_expired(now + timedelta(seconds=2)) == 1

    seen = []
    worker = Worker(leases, lambda event, token: _execute(seen, event, token), timedelta(minutes=1))
    assert await worker.handle(OutboxEnvelope("event", "job-1", "generate", 1))
    async with sessions() as session:
        row = await session.get(JobRow, "job-1")
    assert row.status == JobStatus.SUCCEEDED.value
    assert seen[0][0].job_id == "job-1"
    assert not await leases.complete("job-1", old_claim[0], old_claim[1])


async def test_expired_queued_job_is_never_claimed(worker_parts):
    leases, sessions = worker_parts
    async with sessions.begin() as session:
        row = await session.get(JobRow, "job-1")
        row.expires_at = datetime.now() - timedelta(days=1)

    now = datetime.now(timezone.utc)
    assert await leases.expire_queued(now) == 1
    assert await leases.claim("job-1", now, timedelta(minutes=1)) is None


async def test_stale_completion_cannot_overwrite_newer_success(worker_parts):
    leases, sessions = worker_parts
    now = datetime.now(timezone.utc)
    first = await leases.claim("job-1", now, timedelta(seconds=1))
    assert first
    await leases.recover_expired(now + timedelta(seconds=2))
    second = await leases.claim("job-1", now + timedelta(seconds=2), timedelta(minutes=1))
    assert second

    assert await leases.complete("job-1", second[0], second[1])
    assert not await leases.complete("job-1", first[0], first[1])
    async with sessions() as session:
        assert (await session.get(JobRow, "job-1")).status == JobStatus.SUCCEEDED.value


async def test_stale_failure_cannot_turn_newer_success_into_error(worker_parts):
    leases, sessions = worker_parts
    now = datetime.now(timezone.utc)
    first = await leases.claim("job-1", now, timedelta(seconds=1))
    assert first
    await leases.recover_expired(now + timedelta(seconds=2))
    second = await leases.claim("job-1", now + timedelta(seconds=2), timedelta(minutes=1))
    assert second
    assert await leases.complete("job-1", second[0], second[1])
    assert not await leases.fail("job-1", first[0], first[1], "internal")
    async with sessions() as session:
        row = await session.get(JobRow, "job-1")
    assert row.status == JobStatus.SUCCEEDED.value
    assert row.public_error_code is None


async def test_retryable_worker_failure_is_released_then_dead_lettered_at_cap(worker_parts):
    leases, sessions = worker_parts

    async def fail(_event, _token):
        raise RuntimeError("provider unavailable")

    worker = Worker(leases, fail, timedelta(minutes=1))
    envelope = OutboxEnvelope("event", "job-1", "generate", 1)
    assert not await worker.handle(envelope)
    assert not await worker.handle(envelope)
    assert await worker.handle(envelope)
    async with sessions() as session:
        row = await session.get(JobRow, "job-1")
    assert row.status == JobStatus.DEAD_LETTER.value
    assert row.attempt == 3


async def test_replay_after_effect_commit_completes_without_repeating_provider_work(worker_parts):
    leases, sessions = worker_parts
    effects = JobEffects(sessions)
    now = datetime.now(timezone.utc)
    first = await leases.claim("job-1", now, timedelta(seconds=1))
    assert first
    await effects.adopt_or_record("job-1", "generate", {"word_id": 7})
    assert await leases.recover_expired(now + timedelta(seconds=2)) == 1

    calls = []

    async def execute(event, token):
        calls.append((event, token))
        return {"word_id": 8}

    worker = Worker(leases, execute, timedelta(minutes=1), effects)
    assert await worker.handle(OutboxEnvelope("event", "job-1", "generate", 1))
    assert calls == []
    async with sessions() as session:
        assert (await session.get(JobRow, "job-1")).status == JobStatus.SUCCEEDED.value


async def test_unsupported_stream_payload_is_dead_lettered_without_provider_execution(worker_parts):
    leases, sessions = worker_parts
    calls = []

    async def execute(event, token):
        calls.append((event, token))
        return {"word_id": 7}

    worker = Worker(leases, execute, timedelta(minutes=1))
    assert await worker.handle(OutboxEnvelope("event", "job-1", "generate", 2))
    assert calls == []
    async with sessions() as session:
        row = await session.get(JobRow, "job-1")
    assert row.status == JobStatus.DEAD_LETTER.value
    assert row.public_error_code == "unsupported_payload_version"


async def test_duplicate_success_delivery_is_acknowledged_without_reexecution(worker_parts):
    leases, _ = worker_parts
    calls = []

    async def execute(event, token):
        calls.append((event, token))
        return {"word_id": 7}

    worker = Worker(leases, execute, timedelta(minutes=1))
    envelope = OutboxEnvelope("event", "job-1", "generate", 1)
    assert await worker.handle(envelope)
    assert await worker.handle(envelope)
    assert len(calls) == 1


async def test_reclaim_creates_a_consumer_group_before_claiming_pending_messages(worker_parts):
    leases, _ = worker_parts

    class Consumer:
        def __init__(self):
            self.ensured = 0

        async def ensure_group(self):
            self.ensured += 1

        async def reclaim(self, _min_idle_ms):
            if False:
                yield None

    async def execute(event, token):
        return {"word_id": 7}

    consumer = Consumer()
    worker = Worker(leases, execute, timedelta(minutes=1))
    assert await worker.reclaim_once(consumer, 60_000) == 0
    assert consumer.ensured == 1


async def _execute(seen, event, token):
    seen.append((event, token))
