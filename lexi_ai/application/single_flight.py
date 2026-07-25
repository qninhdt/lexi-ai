"""Per-key locks that collapse concurrent work on the same target into one.

These locks must live for the life of the PROCESS, not the life of a service
instance. A lock rebuilt per call locks nothing: two callers would each take their
own lock, both see no result, and both call the provider — which is the exact
duplication this exists to prevent.

Generation and theming keep SEPARATE registries. The generation lock is always
released before the theming step begins, and keeping the two key spaces apart is
what guarantees they can never nest and therefore never form a cycle.
"""

import asyncio
from collections.abc import AsyncIterator, Hashable
from contextlib import asynccontextmanager


class SingleFlight:
    """A registry of per-key locks, evicted once idle."""

    def __init__(self) -> None:
        self._locks: dict[Hashable, asyncio.Lock] = {}

    @asynccontextmanager
    async def hold(self, key: Hashable) -> AsyncIterator[None]:
        """Hold the lock for ``key``, dropping it afterwards if nobody else waits.

        Eviction keeps the registry from growing without bound over a long-lived
        process, and only happens when the lock is genuinely unheld.
        """
        lock = self._locks.setdefault(key, asyncio.Lock())
        try:
            async with lock:
                yield
        finally:
            if not lock.locked() and self._locks.get(key) is lock:
                del self._locks[key]

    def __len__(self) -> int:
        return len(self._locks)
