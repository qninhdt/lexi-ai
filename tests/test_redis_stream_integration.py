"""Opt-in Redis Streams delivery contract against a real Redis server."""

import asyncio
import os
from uuid import uuid4

import pytest
from redis.asyncio import Redis

from lexi_service.coordination.limits import RedisProviderGate
from lexi_service.jobs.outbox import OutboxEnvelope
from lexi_service.jobs.redis_stream import RedisStreamConsumer, RedisStreamPublisher

REDIS_URL = os.environ.get("LEXI_TEST_REDIS_URL")
pytestmark = pytest.mark.skipif(
    not REDIS_URL, reason="Redis tier: set LEXI_TEST_REDIS_URL to a disposable Redis instance"
)


async def test_real_redis_stream_delivers_and_acknowledges_a_durable_envelope():
    redis = Redis.from_url(REDIS_URL)
    suffix = uuid4().hex
    stream, group = f"lexi-test-{suffix}", f"workers-{suffix}"
    envelope = OutboxEnvelope("event-1", "job-1", "generate", 1)
    try:
        await RedisStreamPublisher(redis, stream).publish(envelope)
        consumer = RedisStreamConsumer(redis, stream, group, "worker-a")
        await consumer.ensure_group()
        messages = [message async for message in consumer.read(block_ms=10)]
        assert len(messages) == 1
        message_id, delivered = messages[0]
        assert delivered == envelope
        await consumer.acknowledge(message_id)
        pending = await redis.xpending(stream, group)
        assert pending["pending"] == 0
    finally:
        await redis.delete(stream)
        await redis.aclose()


async def test_real_redis_provider_gate_limits_multiple_instances():
    redis = Redis.from_url(REDIS_URL)
    prefix = f"lexi-test-provider-{uuid4().hex}"
    first = RedisProviderGate(redis, global_limit=1, tenant_limit=1, lease_ms=1000, prefix=prefix)
    second = RedisProviderGate(redis, global_limit=1, tenant_limit=1, lease_ms=1000, prefix=prefix)
    try:
        async with first.acquire("tenant-a"):
            with pytest.raises(RuntimeError, match="quota"):
                async with second.acquire("tenant-b"):
                    pass
        async with second.acquire("tenant-b"):
            pass
    finally:
        keys = [key async for key in redis.scan_iter(match=f"{prefix}:*")]
        if keys:
            await redis.delete(*keys)
        await redis.aclose()


async def test_expired_provider_lease_cannot_release_a_newer_holder_slot():
    redis = Redis.from_url(REDIS_URL)
    prefix = f"lexi-test-provider-expiry-{uuid4().hex}"
    first = RedisProviderGate(redis, global_limit=1, tenant_limit=1, lease_ms=10, prefix=prefix)
    second = RedisProviderGate(redis, global_limit=1, tenant_limit=1, lease_ms=1000, prefix=prefix)
    third = RedisProviderGate(redis, global_limit=1, tenant_limit=1, lease_ms=1000, prefix=prefix)
    first_context = first.acquire("tenant-a")
    second_context = second.acquire("tenant-b")
    try:
        await first_context.__aenter__()
        await asyncio.sleep(0.02)
        await second_context.__aenter__()
        await first_context.__aexit__(None, None, None)
        with pytest.raises(RuntimeError, match="quota"):
            async with third.acquire("tenant-c"):
                pass
    finally:
        await second_context.__aexit__(None, None, None)
        keys = [key async for key in redis.scan_iter(match=f"{prefix}:*")]
        if keys:
            await redis.delete(*keys)
        await redis.aclose()
