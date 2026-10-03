"""Byte-bounded JSON values; each hit owns its nested mutable data.

Access stays on the owning event loop, never in search worker threads. Cache
budgets count serialized payload bytes, not total Python allocator overhead.
"""

import orjson
from cachetools import LRUCache, TTLCache


class Cache:
    def __init__(self, max_bytes: int, *, ttl: float | None = None):
        if type(max_bytes) is not int or max_bytes <= 0:
            raise ValueError("cache budget must be a positive integer")
        if ttl is not None and ttl <= 0:
            raise ValueError("cache TTL must be positive")
        self._values = (
            LRUCache(max_bytes, getsizeof=len)
            if ttl is None
            else TTLCache(max_bytes, ttl, getsizeof=len)
        )

    def get(self, key, model=None):
        payload = self._values.get(key)
        if payload is None:
            return None
        value = orjson.loads(payload)
        return model(**value) if model is not None else value

    def get_many(self, keys, model=None):
        return {key: value for key in keys if (value := self.get(key, model)) is not None}

    def put(self, key, value):
        if value is None:
            return
        payload = orjson.dumps(value)
        if len(payload) <= self._values.maxsize:
            self._values[key] = payload
        else:
            self.discard(key)

    def discard(self, key):
        self._values.pop(key, None)

    def clear(self):
        self._values.clear()
