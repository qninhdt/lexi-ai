from lexi_service.coordination.redis_locks import RedisLock


class FakeRedis:
    def __init__(self):
        self.values = {}

    async def set(self, key, value, nx, px):
        if nx and key in self.values:
            return False
        self.values[key] = value
        return True

    async def eval(self, script, _, key, token, *args):
        if self.values.get(key) != token:
            return 0
        if "pexpire" in script:
            return 1
        del self.values[key]
        return 1


async def test_only_lock_owner_can_release_token_lock():
    redis = FakeRedis()
    first, second = RedisLock(redis, "word:cat", 1000), RedisLock(redis, "word:cat", 1000)
    assert await first.acquire()
    assert not await second.acquire()
    assert not await second.release()
    assert await first.release()


async def test_only_lock_owner_can_renew_token_lock():
    redis = FakeRedis()
    first, second = RedisLock(redis, "word:cat", 1000), RedisLock(redis, "word:cat", 1000)
    assert await first.acquire()
    assert not await second.renew()
