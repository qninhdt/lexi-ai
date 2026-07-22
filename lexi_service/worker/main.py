"""Small worker loop: Redis delivery is acknowledged only after leased completion."""

import logging
from collections.abc import Awaitable, Callable
from datetime import datetime, timedelta, timezone

from lexi_service.application.errors import to_public_error
from lexi_service.jobs.effects import JobEffects
from lexi_service.jobs.leases import JobLeaseRepository
from lexi_service.jobs.models import JobStatus
from lexi_service.jobs.outbox import OutboxEnvelope
from lexi_service.jobs.redis_stream import RedisStreamConsumer
from lexi_service.observability.logging import log_event

logger = logging.getLogger(__name__)


class Worker:
    def __init__(
        self,
        leases: JobLeaseRepository,
        execute: Callable[[OutboxEnvelope, str], Awaitable[object]],
        lease_for: timedelta,
        effects: JobEffects | None = None,
    ):
        self._leases = leases
        self._execute = execute
        self._lease_for = lease_for
        self._effects = effects

    async def handle(self, envelope: OutboxEnvelope) -> bool:
        now = datetime.now(timezone.utc)
        claim = await self._leases.claim(envelope.job_id, now, self._lease_for)
        if claim is None:
            return await self._leases.should_ack(envelope.job_id)
        token, epoch = claim
        if envelope.payload_version != 1:
            return await self._leases.dead_letter(
                envelope.job_id, token, epoch, "unsupported_payload_version"
            )
        try:
            if self._effects is not None:
                adopted = await self._effects.get(envelope.job_id, envelope.operation)
                if adopted is not None:
                    completed = await self._leases.complete(envelope.job_id, token, epoch)
                    if completed:
                        log_event(
                            logger,
                            "job_effect_adopted",
                            job_id=envelope.job_id,
                            operation=envelope.operation,
                        )
                    return completed
            result = await self._execute(envelope, token)
            if self._effects is not None:
                await self._effects.adopt_or_record(envelope.job_id, envelope.operation, result)
        except Exception as exc:
            error = to_public_error(exc)
            logger.exception(
                "job_execution_failed",
                extra={"job_id": envelope.job_id, "operation": envelope.operation},
            )
            if error.retryable:
                status = await self._leases.retry_or_dead_letter(
                    envelope.job_id, token, epoch, error.code.value
                )
                # A queued retry remains pending in Redis and is reclaimed by a
                # worker; a dead letter is terminal and may be acknowledged.
                return status == JobStatus.DEAD_LETTER.value
            return await self._leases.fail(envelope.job_id, token, epoch, error.code.value)
        completed = await self._leases.complete(envelope.job_id, token, epoch)
        if completed:
            log_event(logger, "job_completed", job_id=envelope.job_id, operation=envelope.operation)
        return completed

    async def consume_once(self, consumer: RedisStreamConsumer) -> int:
        """Ack only after a handler owns and completes the current lease."""
        handled = 0
        await consumer.ensure_group()
        async for message_id, envelope in consumer.read():
            if await self.handle(envelope):
                await consumer.acknowledge(message_id)
                handled += 1
        return handled

    async def reclaim_once(self, consumer: RedisStreamConsumer, min_idle_ms: int) -> int:
        handled = 0
        await consumer.ensure_group()
        async for message_id, envelope in consumer.reclaim(min_idle_ms):
            if await self.handle(envelope):
                await consumer.acknowledge(message_id)
                handled += 1
        return handled
