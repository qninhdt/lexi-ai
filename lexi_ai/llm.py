import asyncio
from langchain_core.runnables import Runnable
from pydantic import BaseModel


async def ainvoke_structured(
    runnable: Runnable,
    messages: list,
    expect: type[BaseModel],
    *,
    max_retries: int = 3,
    base_delay: float = 0.5,
) -> BaseModel:
    """Invoke a structured runnable, retrying transient failures, and validate.

    ``runnable`` is already bound to ``expect`` (``with_structured_output``); we
    still ``model_validate`` because some providers return a dict rather than the
    model instance. Raises the last exception after ``max_retries`` attempts.
    """
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            result = await runnable.ainvoke(messages)
            if isinstance(result, expect):
                return result
            # Some providers return a dict when include_raw or json_mode.
            return expect.model_validate(result)
        except Exception as exc:  # noqa: BLE001 - retried, then re-raised
            last_exc = exc
            if attempt < max_retries - 1:
                await asyncio.sleep(base_delay * (2**attempt))
    assert last_exc is not None
    raise last_exc
