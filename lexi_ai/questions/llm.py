"""Structured-LLM invocation with retry — shared by the llm plugin and judge.

Both the contextual-MCQ generator and the rubric scorer invoke a runnable that is
already bound to a Pydantic schema (via ``with_structured_output``) and want the
same transient-error retry with exponential backoff as
``generation/generator.py``. Factored here so the two sites cannot drift.
"""

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
    model instance. Raises the last exception after ``max_retries`` attempts —
    grading/generation surface the error rather than silently degrade.
    """
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            result = await runnable.ainvoke(messages)
            if isinstance(result, expect):
                return result
            return expect.model_validate(result)
        except Exception as exc:  # noqa: BLE001 - retried, then re-raised
            last_exc = exc
            if attempt < max_retries - 1:
                await asyncio.sleep(base_delay * (2**attempt))
    assert last_exc is not None
    raise last_exc
