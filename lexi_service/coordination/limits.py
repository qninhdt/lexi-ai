"""Fail-closed provider concurrency limits; read paths never acquire these gates."""

import asyncio
from contextlib import asynccontextmanager
from hashlib import sha256
from uuid import uuid4


class ProviderLimits:
    def __init__(self, global_limit: int, tenant_limit: int):
        self._global = asyncio.Semaphore(global_limit)
        self._tenant_limit = tenant_limit
        self._tenants: dict[str, asyncio.Semaphore] = {}

    @asynccontextmanager
    async def acquire(self, tenant: str):
        semaphore = self._tenants.setdefault(tenant, asyncio.Semaphore(self._tenant_limit))
        async with self._global:
            async with semaphore:
                yield


class RedisProviderGate:
    """Distributed global/per-tenant provider slots with crash-safe TTLs."""

    def __init__(
        self,
        client,
        global_limit: int,
        tenant_limit: int,
        lease_ms: int,
        *,
        prefix: str = "lexi:provider-slots",
    ):
        self._client = client
        self._global_limit = global_limit
        self._tenant_limit = tenant_limit
        self._lease_ms = lease_ms
        self._prefix = prefix

    @asynccontextmanager
    async def acquire(self, tenant: str):
        token = uuid4().hex
        global_key = f"{self._prefix}:global"
        tenant_key = f"{self._prefix}:tenant:{sha256(tenant.encode()).hexdigest()}"
        acquired = await self._client.eval(
            """
            local now = redis.call('TIME')
            local now_ms = now[1] * 1000 + math.floor(now[2] / 1000)
            redis.call('zremrangebyscore', KEYS[1], '-inf', now_ms)
            redis.call('zremrangebyscore', KEYS[2], '-inf', now_ms)
            if redis.call('zcard', KEYS[1]) >= tonumber(ARGV[1])
                or redis.call('zcard', KEYS[2]) >= tonumber(ARGV[2]) then
              return 0
            end
            local expires = now_ms + tonumber(ARGV[3])
            redis.call('zadd', KEYS[1], expires, ARGV[4])
            redis.call('zadd', KEYS[2], expires, ARGV[4])
            redis.call('pexpire', KEYS[1], ARGV[3] * 2)
            redis.call('pexpire', KEYS[2], ARGV[3] * 2)
            return 1
            """,
            2,
            global_key,
            tenant_key,
            self._global_limit,
            self._tenant_limit,
            self._lease_ms,
            token,
        )
        if not acquired:
            raise RuntimeError("provider concurrency quota is exhausted")
        try:
            yield
        finally:
            await self._client.eval(
                """
                redis.call('zrem', KEYS[1], ARGV[1])
                redis.call('zrem', KEYS[2], ARGV[1])
                return 1
                """,
                2,
                global_key,
                tenant_key,
                token,
            )
