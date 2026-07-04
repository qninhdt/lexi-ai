"""LLM themed-generation pipeline (Phase 2).

Mirrors :mod:`lexi_ai.generation.generator`: wraps an OpenAI-compatible chat
model bound to :class:`ThemedResult`, injectable for hermetic tests, retried
with exponential backoff.
"""

import asyncio
from collections.abc import Sequence

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable

from lexi_ai.config import Settings, get_settings
from lexi_ai.theming.prompts import SYSTEM_PROMPT, format_themed
from lexi_ai.theming.schemas import ThemedResult


def build_themed_llm(settings: Settings) -> Runnable:
    """Build the ChatOpenAI runnable bound to ``ThemedResult``.

    Uses ``settings.translate_model or settings.llm_model`` — themed generation
    is not the primary generation model, so it may point elsewhere while defaulting
    to the main model. Imported lazily so fakes need no creds.
    """
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=settings.llm_model,
        temperature=settings.llm_temperature,
    )
    return llm.with_structured_output(ThemedResult)


class ThemedGenerator:
    """Turn a style prompt + neutral sense facts into a validated ThemedResult."""

    def __init__(
        self,
        structured_llm: Runnable | None = None,
        settings: Settings | None = None,
        max_retries: int = 3,
        base_delay: float = 0.5,
    ):
        self._settings = settings or get_settings()
        self._llm = structured_llm
        self._max_retries = max_retries
        self._base_delay = base_delay

    @property
    def llm(self) -> Runnable:
        if self._llm is None:
            self._llm = build_themed_llm(self._settings)
        return self._llm

    async def generate(
        self,
        style_prompt: str,
        neutral_senses: Sequence[tuple[str, str | None, str | None, str]],
    ) -> ThemedResult:
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=format_themed(style_prompt, neutral_senses)),
        ]
        last_exc: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                result = await self.llm.ainvoke(messages)
                if isinstance(result, ThemedResult):
                    return result
                return ThemedResult.model_validate(result)
            except Exception as exc:  # noqa: BLE001 - retried, then re-raised
                last_exc = exc
                if attempt < self._max_retries - 1:
                    await asyncio.sleep(self._base_delay * (2**attempt))
        assert last_exc is not None
        raise last_exc
