from dataclasses import replace
from datetime import datetime, timezone

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from lexi_service.application.commands import JobSubmission
from lexi_service.application.errors import ApplicationError, ErrorCode
from lexi_service.identity import Principal
from lexi_service.jobs.models import JobRow, OutboxEventRow, ServiceBase
from lexi_service.jobs.repository import SqlJobRepository


@pytest.fixture
async def jobs():
    engine = create_async_engine("sqlite+aiosqlite://", poolclass=StaticPool)
    async with engine.begin() as connection:
        await connection.run_sync(ServiceBase.metadata.create_all)
    try:
        yield (
            SqlJobRepository(async_sessionmaker(engine, expire_on_commit=False)),
            async_sessionmaker(engine),
        )
    finally:
        await engine.dispose()


def submission(key="key-1"):
    return JobSubmission(
        operation="generate",
        request_id="request-1",
        owner=Principal("svc", "tenant"),
        idempotency_key=key,
        payload_version=1,
        reference_dataset_fingerprint="dataset-1",
        accepted_at=datetime.now(timezone.utc),
        payload={"display": "cat"},
        maximum_age_seconds=60,
        max_retries=2,
    )


async def test_publish_writes_job_and_outbox_in_one_transaction(jobs):
    repository, sessions = jobs
    reference = await repository.publish(submission())

    async with sessions() as session:
        assert await session.scalar(select(func.count()).select_from(JobRow)) == 1
        event = await session.scalar(select(OutboxEventRow))
    assert event.job_id == reference.job_id
    assert event.operation == "generate"
    async with sessions() as session:
        assert (await session.get(JobRow, reference.job_id)).request_id == "request-1"


async def test_idempotent_publish_reuses_existing_owner_operation_key(jobs):
    repository, _ = jobs
    first = await repository.publish(submission())
    second = await repository.publish(submission())

    assert second.job_id == first.job_id
    assert second.deduplicated


async def test_idempotency_key_rejects_a_different_canonical_payload(jobs):
    repository, _ = jobs
    await repository.publish(submission())
    altered = submission()
    object.__setattr__(altered, "payload", {"display": "dog"})

    with pytest.raises(ValueError, match="different payload"):
        await repository.publish(altered)


async def test_idempotency_key_isolated_by_authenticated_owner(jobs):
    repository, _ = jobs
    first = await repository.publish(submission())
    other_owner = replace(submission(), owner=Principal("another-service", "tenant"))
    second = await repository.publish(other_owner)

    assert first.job_id != second.job_id
    assert not second.deduplicated


async def test_outstanding_job_quota_is_scoped_to_authenticated_owner(jobs):
    _, sessions = jobs
    repository = SqlJobRepository(sessions, max_outstanding_jobs_per_owner=1)
    first = await repository.publish(submission())
    assert (await repository.publish(submission())).deduplicated

    with pytest.raises(ApplicationError) as raised:
        await repository.publish(submission("key-2"))
    assert raised.value.error.code == ErrorCode.CONFLICT
    assert raised.value.error.message == "Outstanding job quota exceeded."

    other = replace(submission("key-2"), owner=Principal("another-service", "tenant"))
    assert (await repository.publish(other)).job_id != first.job_id
