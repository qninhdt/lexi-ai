"""SingleFlight must not evict a lock that still has waiters queued on it.

The registry evicts an idle key so a long-lived process does not accumulate one
lock per word ever generated. The eviction predicate used to be
``not lock.locked()``, which is false the moment the holder releases — while
callers awaiting that exact lock object are still parked on it and have not yet
been scheduled.

The consequence is a duplicate provider call. The entry is deleted, the next
arrival finds nothing in the registry, builds a *second* lock for the same key,
and proceeds concurrently with the waiter that is about to wake on the first. For
the generation path that means paying an LLM twice for one word, which is exactly
what SingleFlight exists to prevent.

These tests drive the interleaving directly rather than through a service, so a
failure points at the primitive instead of at whatever was layered on top.
"""

import asyncio

import pytest

from lexi_ai.application.single_flight import SingleFlight

pytestmark = pytest.mark.asyncio


async def test_only_one_caller_runs_at_a_time_under_contention():
    flight = SingleFlight()
    concurrent = 0
    peak = 0
    calls = 0

    async def work() -> None:
        nonlocal concurrent, peak, calls
        async with flight.hold("same-key"):
            calls += 1
            concurrent += 1
            peak = max(peak, concurrent)
            # Yield inside the critical section: this is the window in which a
            # second holder would become observable.
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            concurrent -= 1

    await asyncio.gather(*[work() for _ in range(8)])

    assert calls == 8
    assert peak == 1, f"{peak} callers held the same key at once"


async def test_the_lock_is_not_evicted_while_a_waiter_is_queued():
    """The precise defect: eviction keyed on `locked()` drops a live entry."""
    flight = SingleFlight()
    holder_inside = asyncio.Event()
    release_holder = asyncio.Event()
    observed: list[int] = []

    async def holder() -> None:
        async with flight.hold("k"):
            holder_inside.set()
            await release_holder.wait()

    async def waiter() -> None:
        # Enters `hold` while the holder is inside, so it must queue on the very
        # same lock object rather than create a second one.
        async with flight.hold("k"):
            observed.append(len(flight))

    holder_task = asyncio.create_task(holder())
    await holder_inside.wait()

    waiter_task = asyncio.create_task(waiter())
    # Let the waiter reach the acquire and park there.
    for _ in range(5):
        await asyncio.sleep(0)

    # Two interested callers, exactly one registry entry between them.
    assert len(flight) == 1

    release_holder.set()
    await asyncio.gather(holder_task, waiter_task)

    assert observed == [1], "the waiter ran against a rebuilt second lock"
    # Both callers are done, so the entry is finally collectable.
    assert len(flight) == 0


async def test_an_idle_key_is_still_evicted():
    """The refcount must not turn into a leak: eviction is the reason it exists."""
    flight = SingleFlight()

    for key in ("a", "b", "c"):
        async with flight.hold(key):
            pass

    assert len(flight) == 0


async def test_an_exception_inside_the_critical_section_releases_and_evicts():
    flight = SingleFlight()

    with pytest.raises(RuntimeError):
        async with flight.hold("boom"):
            raise RuntimeError("failure inside the held section")

    assert len(flight) == 0
    # The key is reusable rather than poisoned by the failure.
    async with flight.hold("boom"):
        pass
    assert len(flight) == 0


async def test_a_cancelled_waiter_does_not_strand_the_entry():
    flight = SingleFlight()
    holder_inside = asyncio.Event()
    release_holder = asyncio.Event()

    async def holder() -> None:
        async with flight.hold("k"):
            holder_inside.set()
            await release_holder.wait()

    async def waiter() -> None:
        async with flight.hold("k"):
            pass

    holder_task = asyncio.create_task(holder())
    await holder_inside.wait()
    waiter_task = asyncio.create_task(waiter())
    for _ in range(5):
        await asyncio.sleep(0)

    waiter_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter_task

    release_holder.set()
    await holder_task

    assert len(flight) == 0, "a cancelled waiter left its entry behind"


async def test_distinct_keys_do_not_serialize_against_each_other():
    """Collapsing work on one key must not become a global bottleneck."""
    flight = SingleFlight()
    inside = asyncio.Event()

    async def first() -> None:
        async with flight.hold("one"):
            inside.set()
            await asyncio.sleep(0.05)

    async def second() -> None:
        await inside.wait()

        async def take_other_key() -> None:
            async with flight.hold("two"):
                pass

        # Would time out if the registry served both keys from one lock.
        # `wait_for` rather than `asyncio.timeout`, which needs Python 3.11.
        await asyncio.wait_for(take_other_key(), timeout=0.02)

    await asyncio.gather(first(), second())
