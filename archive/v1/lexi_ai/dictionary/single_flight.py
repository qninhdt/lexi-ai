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
from dataclasses import dataclass, field


@dataclass
class _Entry:
    """One key's lock plus the number of callers currently interested in it.

    ``waiters`` counts everyone between entering :meth:`SingleFlight.hold` and
    leaving it — the holder included — rather than only those blocked on acquire.
    That is what makes eviction safe: the count is incremented before the lock is
    awaited, so there is no window in which an arriving caller is invisible.
    """

    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    waiters: int = 0


class SingleFlight:
    """A registry of per-key locks, evicted once genuinely idle."""

    def __init__(self) -> None:
        self._entries: dict[Hashable, _Entry] = {}

    @asynccontextmanager
    async def hold(self, key: Hashable) -> AsyncIterator[None]:
        """Hold the lock for ``key``, dropping it only when nobody else wants it.

        Eviction is refcounted rather than keyed on ``lock.locked()``. That
        predicate is false the instant the holder releases — while queued waiters
        are still parked on the very same lock object — so the entry was deleted
        out from under them. The next arrival then found an empty registry, built
        a second lock for the same key, and ran concurrently with a waiter that
        was about to wake: two provider calls for the work this exists to
        deduplicate, which for the generation path means paying an LLM twice.

        The counter is incremented before the acquire and decremented in a
        ``finally``, so an exception or a cancellation cannot strand an entry.
        """
        entry = self._entries.get(key)
        if entry is None:
            entry = self._entries[key] = _Entry()
        entry.waiters += 1
        try:
            async with entry.lock:
                yield
        finally:
            entry.waiters -= 1
            # Re-check identity: a pathological interleaving could have replaced
            # the entry, and deleting someone else's would reopen the same hole.
            if entry.waiters <= 0 and self._entries.get(key) is entry:
                del self._entries[key]

    def __len__(self) -> int:
        return len(self._entries)
