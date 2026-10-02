"""Bounded async workers for Lexi's async clients, with completion progress."""

import asyncio

from tqdm import tqdm


def validate_workers(workers):
    if type(workers) is not int or workers < 1:
        raise ValueError("workers must be a positive integer")


async def map_workers(items, function, *, workers=4, progress=True, desc="", on_result=None):
    validate_workers(workers)
    items = list(items)
    results = [None] * len(items)
    pending = iter(enumerate(items))
    with tqdm(total=len(items), desc=desc, unit="item", disable=not progress) as bar:

        async def worker():
            for index, item in pending:
                result = await function(item)
                results[index] = result
                if on_result is not None:
                    on_result(index, result)
                bar.update(1)

        tasks = [asyncio.create_task(worker()) for _ in range(min(workers, len(items)))]
        try:
            await asyncio.gather(*tasks)
        finally:
            # Never close shared clients while other jobs are still using them.
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
    return results
