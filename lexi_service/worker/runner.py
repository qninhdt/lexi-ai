"""Worker process entrypoint; runtime-specific handler wiring is injected at startup."""

import asyncio
import logging
import os
from datetime import timedelta

from redis.asyncio import Redis

from lexi_ai.config import get_settings
from lexi_ai.normalize import match_key
from lexi_ai.read_models import SearchResult
from lexi_service.application.commands import ExecuteGenerate, ExecuteTranslation
from lexi_service.application.errors import ErrorCode, public_error
from lexi_service.bootstrap import compose_runtime
from lexi_service.coordination.limits import RedisProviderGate
from lexi_service.coordination.redis_locks import RedisLock
from lexi_service.jobs.dispatcher import OutboxDispatcher
from lexi_service.jobs.effects import JobEffects
from lexi_service.jobs.leases import JobLeaseRepository
from lexi_service.jobs.outbox import SqlOutbox
from lexi_service.jobs.redis_stream import RedisStreamConsumer, RedisStreamPublisher
from lexi_service.observability.logging import log_event
from lexi_service.settings import ServiceSettings
from lexi_service.worker.main import Worker

logger = logging.getLogger(__name__)


async def run() -> None:
    settings = ServiceSettings()
    redis = Redis.from_url(settings.redis_url)
    runtime = compose_runtime(
        settings,
        get_settings(),
        provider_gate=RedisProviderGate(
            redis,
            settings.max_provider_concurrency,
            settings.max_provider_concurrency_per_tenant,
            settings.provider_attempt_timeout_seconds * 2_000 + 1_000,
        ),
    )
    dataset = runtime.dataset
    if dataset is not None and hasattr(dataset, "ready") and not await dataset.ready():
        await redis.aclose()
        await runtime.close()
        raise RuntimeError(
            "reference dataset is missing or does not match its configured fingerprint"
        )
    repository = runtime.submissions._publisher
    sessions = repository.sessions
    consumer = RedisStreamConsumer(
        redis,
        os.environ.get("LEXI_SERVICE_REDIS_STREAM", "lexi.jobs"),
        os.environ.get("LEXI_SERVICE_REDIS_GROUP", "lexi-workers"),
        os.environ.get("HOSTNAME", "worker"),
    )
    dispatcher = OutboxDispatcher(
        SqlOutbox(sessions),
        RedisStreamPublisher(redis, os.environ.get("LEXI_SERVICE_REDIS_STREAM", "lexi.jobs")),
    )

    async def execute(envelope, token):
        loaded = await repository.load_submission(envelope.job_id)
        if loaded is None:
            raise RuntimeError("job no longer exists")
        job, attempt = loaded
        log_event(
            logger,
            "provider_execution_started",
            request_id=job.request_id,
            job_id=envelope.job_id,
            operation=job.operation,
            attempt=attempt,
        )
        if job.operation == "generate":
            target = SearchResult(
                display=str(job.payload["display"]),
                entry_type=None,
                lexi_word_id=job.payload.get("lexi_word_id"),
                cambridge_id=job.payload.get("cambridge_id"),
            )
            lock = RedisLock(
                redis,
                f"lexi:generation:{match_key(target.display)}",
                int(runtime.policy.provider_attempt_timeout.total_seconds() * 2_000) + 1_000,
            )
            if not await lock.acquire():
                raise public_error(
                    ErrorCode.INTERNAL, "Generation is already being processed.", retryable=True
                )
            try:
                entry = await runtime.executions.execute_generate(
                    ExecuteGenerate(job, target, 0, attempt)
                )
            finally:
                await lock.release()
            return {
                "word_id": entry.word_id,
                "sense_ids": [
                    sense.sense_id for sense in entry.senses if sense.sense_id is not None
                ],
            }
        if job.operation == "translate":
            translated = await runtime.executions.execute_translation(
                ExecuteTranslation(
                    job,
                    job.payload["source_kind"],
                    job.payload["source_id"],
                    job.payload["language"],
                    job.payload["source_hash"],
                    attempt,
                )
            )
            return {"text": translated}
        raise RuntimeError(f"unsupported job operation: {job.operation}")

    worker = Worker(
        JobLeaseRepository(sessions), execute, timedelta(seconds=60), effects=JobEffects(sessions)
    )
    try:
        while True:
            await dispatcher.dispatch_once()
            await worker.reclaim_once(consumer, min_idle_ms=60_000)
            await worker.consume_once(consumer)
    finally:
        await redis.aclose()
        await runtime.close()


if __name__ == "__main__":
    asyncio.run(run())
