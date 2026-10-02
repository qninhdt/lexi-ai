import asyncio

import pytest

from benchmarks.workers import map_workers


@pytest.mark.parametrize("workers", [0, -1, True, 1.5, "2"])
async def test_invalid_worker_count_does_not_start_jobs(workers):
    async def unused(_item):
        pytest.fail("invalid worker count started a job")

    with pytest.raises(ValueError, match="positive integer"):
        await map_workers([1], unused, workers=workers, progress=False)


async def test_bounded_workers_keep_input_order_and_report_completion(monkeypatch):
    active = peak = 0
    updates = []
    completed = []

    class Bar:
        def __init__(self, **kwargs):
            assert kwargs == dict(total=7, desc="Test", unit="item", disable=False)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            assert active == 0

        def update(self, amount):
            updates.append(amount)

    monkeypatch.setattr("benchmarks.workers.tqdm", Bar)

    async def one(item):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.001 * (7 - item))
        active -= 1
        return item * 2

    results = await map_workers(
        range(7),
        one,
        workers=3,
        desc="Test",
        on_result=lambda index, result: completed.append((index, result)),
    )
    assert peak == 3
    assert results == [item * 2 for item in range(7)]
    assert sorted(completed) == list(enumerate(results))
    assert updates == [1] * 7


async def test_failure_cancels_and_drains_other_workers():
    running = asyncio.Event()
    cleaned = asyncio.Event()

    async def one(item):
        if item == 0:
            await running.wait()
            raise ValueError("failed job")
        running.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaned.set()

    with pytest.raises(ValueError, match="failed job"):
        await map_workers([0, 1], one, workers=2, progress=False)
    assert cleaned.is_set()


async def test_empty_jobs():
    async def unused(_item):
        pytest.fail("no job expected")

    assert await map_workers([], unused, workers=4, progress=False) == []
