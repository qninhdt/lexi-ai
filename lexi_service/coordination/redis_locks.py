"""Token-owned Redis locks used only to avoid duplicate provider work."""

from uuid import uuid4


class RedisLock:
    def __init__(self, client, key: str, ttl_ms: int):
        self._client, self._key, self._ttl, self.token = client, key, ttl_ms, str(uuid4())

    async def acquire(self) -> bool:
        return bool(await self._client.set(self._key, self.token, nx=True, px=self._ttl))

    async def release(self) -> bool:
        script = (
            "if redis.call('get', KEYS[1]) == ARGV[1] "
            "then return redis.call('del', KEYS[1]) else return 0 end"
        )
        return bool(await self._client.eval(script, 1, self._key, self.token))

    async def renew(self) -> bool:
        """Extend this lock only when this instance still owns its token."""
        script = (
            "if redis.call('get', KEYS[1]) == ARGV[1 "
            "then return redis.call('pexpire', KEYS[1], ARGV[2]) else return 0 end"
        )
        return bool(await self._client.eval(script, 1, self._key, self.token, self._ttl))
