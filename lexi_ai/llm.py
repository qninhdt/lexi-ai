"""Structured-LLM seam (openai client).

A narrow project-owned Protocol — ``StructuredLLM.parse(messages, schema)`` —
replaces LangChain's ``Runnable`` + ``with_structured_output``. The real impl
wraps ``openai.AsyncOpenAI`` and calls ``chat.completions.parse`` with the
existing Pydantic schemas (native structured output). Tests inject a fake that
implements ``parse`` directly, so the suite stays hermetic (no network).

Messages are plain dicts (``{"role": "system"|"user", "content": str}``) — the
OpenAI wire format — built via :func:`sys_msg`/:func:`user_msg`.
"""

import asyncio
from typing import Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel

ChatMsg = dict[str, str]
_T = TypeVar("_T", bound=BaseModel)


def sys_msg(content: str) -> ChatMsg:
    return {"role": "system", "content": content}


def user_msg(content: str) -> ChatMsg:
    return {"role": "user", "content": content}


@runtime_checkable
class StructuredLLM(Protocol):
    """The injectable LLM seam: parse messages into a validated schema instance."""

    async def parse(self, messages: list[ChatMsg], schema: type[_T]) -> _T: ...


class OpenAIStructuredLLM:
    """Real :class:`StructuredLLM` over an OpenAI-compatible ``/chat/completions``
    endpoint, using the SDK's native structured-output ``parse`` helper.

    Constructed lazily (the client is only built when first needed) so importing
    a module that builds one costs no creds/network.
    """

    def __init__(self, *, base_url: str, api_key: str, model: str, temperature: float):
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self._model = model
        self._temperature = temperature

    async def parse(self, messages: list[ChatMsg], schema: type[_T]) -> _T:
        completion = await self._client.chat.completions.parse(
            model=self._model,
            messages=messages,  # type: ignore[arg-type]
            response_format=schema,
            temperature=self._temperature,
        )
        parsed = completion.choices[0].message.parsed
        if parsed is None:
            raise ValueError("model returned no parsed structured output")
        return parsed


def build_structured_llm(settings, model: str | None = None) -> StructuredLLM:
    """Build the real openai-backed :class:`StructuredLLM` from settings.

    ``model`` overrides ``settings.llm_model`` (e.g. a per-task translate model);
    base_url/api_key/temperature always come from the shared LLM settings.
    """
    return OpenAIStructuredLLM(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=model or settings.llm_model,
        temperature=settings.llm_temperature,
    )


async def ainvoke_structured(
    llm: StructuredLLM,
    messages: list[ChatMsg],
    expect: type[_T],
    *,
    max_retries: int = 3,
    base_delay: float = 0.5,
) -> _T:
    """Call ``llm.parse``, retrying transient failures, and validate the result.

    ``parse`` should already return an ``expect`` instance; we still
    ``model_validate`` defensively in case a fake/provider returns a dict. Raises
    the last exception after ``max_retries`` attempts.
    """
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            result = await llm.parse(messages, expect)
            if isinstance(result, expect):
                return result
            return expect.model_validate(result)
        except Exception as exc:  # noqa: BLE001 - retried, then re-raised
            last_exc = exc
            if attempt < max_retries - 1:
                await asyncio.sleep(base_delay * (2**attempt))
    assert last_exc is not None
    raise last_exc
