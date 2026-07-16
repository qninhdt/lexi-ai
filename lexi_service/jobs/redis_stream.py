"""Redis Streams envelope publisher; never serializes durable job payloads."""

from redis.asyncio import Redis

from lexi_service.jobs.outbox import OutboxEnvelope


class RedisStreamPublisher:
    def __init__(self, client: Redis, stream: str, maxlen: int | None = None):
        self._client, self._stream, self._maxlen = client, stream, maxlen

    async def publish(self, envelope: OutboxEnvelope) -> None:
        await self._client.xadd(
            self._stream,
            {
                "event_id": envelope.event_id,
                "job_id": envelope.job_id,
                "operation": envelope.operation,
                "payload_version": str(envelope.payload_version),
            },
            maxlen=self._maxlen,
            approximate=True if self._maxlen else False,
        )


class RedisStreamConsumer:
    def __init__(self, client: Redis, stream: str, group: str, consumer: str):
        self._client, self._stream, self._group, self._consumer = client, stream, group, consumer

    async def ensure_group(self) -> None:
        try:
            await self._client.xgroup_create(self._stream, self._group, id="0", mkstream=True)
        except Exception as exc:
            if "BUSYGROUP" not in str(exc):
                raise

    async def read(self, block_ms: int = 1000):
        rows = await self._client.xreadgroup(
            self._group, self._consumer, {self._stream: ">"}, count=1, block=block_ms
        )
        for _, messages in rows:
            for message_id, values in messages:
                yield (
                    message_id,
                    OutboxEnvelope(
                        values[b"event_id"].decode(),
                        values[b"job_id"].decode(),
                        values[b"operation"].decode(),
                        int(values[b"payload_version"]),
                    ),
                )

    async def acknowledge(self, message_id: str) -> None:
        await self._client.xack(self._stream, self._group, message_id)

    async def reclaim(self, min_idle_ms: int, count: int = 100):
        """Claim abandoned pending messages after a crashed consumer."""
        _, messages, _ = await self._client.xautoclaim(
            self._stream, self._group, self._consumer, min_idle_ms, "0-0", count=count
        )
        for message_id, values in messages:
            yield (
                message_id,
                OutboxEnvelope(
                    values[b"event_id"].decode(),
                    values[b"job_id"].decode(),
                    values[b"operation"].decode(),
                    int(values[b"payload_version"]),
                ),
            )
