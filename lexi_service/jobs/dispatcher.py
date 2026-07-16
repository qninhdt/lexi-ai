"""Dispatch committed outbox envelopes to Redis Streams without payload leakage."""

from typing import Protocol

from lexi_service.jobs.outbox import OutboxEnvelope, SqlOutbox


class StreamPublisher(Protocol):
    async def publish(self, envelope: OutboxEnvelope) -> None: ...


class OutboxDispatcher:
    def __init__(self, outbox: SqlOutbox, publisher: StreamPublisher):
        self._outbox = outbox
        self._publisher = publisher

    async def dispatch_once(self, limit: int = 100) -> int:
        dispatched = 0
        for envelope in await self._outbox.pending(limit):
            await self._publisher.publish(envelope)
            await self._outbox.acknowledge(envelope.event_id)
            dispatched += 1
        return dispatched
