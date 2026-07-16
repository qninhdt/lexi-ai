import asyncio

import pytest

from lexi_service.coordination.limits import ProviderLimits, RedisProviderGate


async def test_tenant_limit_serializes_same_tenant_provider_work():
    limits, active, maximum = ProviderLimits(2, 1), 0, 0

    async def run():
        nonlocal active, maximum
        async with limits.acquire("tenant-a"):
            active += 1
            maximum = max(maximum, active)
            await asyncio.sleep(0)
            active -= 1

    await asyncio.gather(run(), run())
    assert maximum == 1


class FakeRedis:
    def __init__(self):
        self.values = {}

    async def eval(self, script, _numkeys, *args):
        keys = args[:2]
        if "zcard" in script:
            global_limit, tenant_limit = int(args[2]), int(args[3])
            if (
                len(self.values.get(keys[0], set())) >= global_limit
                or len(self.values.get(keys[1], set())) >= tenant_limit
            ):
                return 0
            for key in keys:
                self.values.setdefault(key, set()).add(args[5])
            return 1
        for key in keys:
            value = self.values.get(key, set())
            value.discard(args[2])
            if not value:
                self.values.pop(key, None)
        return 1


async def test_redis_gate_enforces_global_slots_across_gate_instances():
    redis = FakeRedis()
    first = RedisProviderGate(redis, global_limit=1, tenant_limit=1, lease_ms=1000)
    second = RedisProviderGate(redis, global_limit=1, tenant_limit=1, lease_ms=1000)
    async with first.acquire("tenant-a"):
        with pytest.raises(RuntimeError, match="quota"):
            async with second.acquire("tenant-b"):
                pass
    async with second.acquire("tenant-b"):
        pass
