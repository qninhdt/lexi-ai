"""Order-aligned batch execution shared by every ``*_many`` use case."""

import asyncio
from collections.abc import Awaitable, Callable, Sequence

from lexi_ai.read_models import BatchResult


async def gather_batch(
    items: Sequence, run: Callable[..., Awaitable], concurrency: int | None = None
) -> list[BatchResult]:
    """Run ``run(item)`` for every item and report each outcome separately.

    One item's failure is captured rather than raised, so a bad id never cancels
    its siblings — that is the whole point of a batch surface.

    ``concurrency`` bounds in-flight calls, which matters for provider-backed
    batches; leave it unset for cheap database-only work.
    """
    if not items:
        return []
    if concurrency is None:
        raw = await asyncio.gather(*(run(item) for item in items), return_exceptions=True)
    else:
        limit = asyncio.Semaphore(concurrency)

        async def _guarded(item):
            async with limit:
                return await run(item)

        raw = await asyncio.gather(*(_guarded(item) for item in items), return_exceptions=True)
    return [
        BatchResult(key=item, error=str(outcome))
        if isinstance(outcome, Exception)
        else BatchResult(key=item, value=outcome)
        for item, outcome in zip(items, raw, strict=True)
    ]
