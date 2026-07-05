"""LLM themed-generation pipeline (Phase 2).

Mirrors :mod:`lexi_ai.generation.generator`: wraps an OpenAI-compatible chat
model bound to :class:`ThemedResult`, injectable for hermetic tests, retried
with exponential backoff.
"""

import asyncio
from collections.abc import Sequence

from collections.abc import Sequence

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable

from lexi_ai.config import Settings, get_settings
from lexi_ai.llm import ainvoke_structured
from lexi_ai.prompts import PromptLoader
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
        mapped_senses = [
            {"definition": d, "pos": pos, "guideword": gw, "tier": tier}
            for d, pos, gw, tier in neutral_senses
        ]
        system_content = PromptLoader.render("themed_restyling_system")
        user_content = PromptLoader.render(
            "themed_restyling_user",
            style_prompt=style_prompt,
            neutral_senses=mapped_senses,
        )
        messages = [
            SystemMessage(content=system_content),
            HumanMessage(content=user_content),
        ]
        return await ainvoke_structured(
            self.llm,
            messages,
            ThemedResult,
            max_retries=self._max_retries,
            base_delay=self._base_delay,
        )


class ThemeMetadataGenerator:
    """Turn a name/key and a style concept prompt into a detailed theme."""

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
            from lexi_ai.theming.schemas import GeneratedTheme
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(
                base_url=self._settings.llm_base_url,
                api_key=self._settings.llm_api_key,
                model=self._settings.llm_model,
                temperature=self._settings.llm_temperature,
            )
            self._llm = llm.with_structured_output(GeneratedTheme)
        return self._llm

    async def generate(self, key: str, prompt: str) -> object:
        from lexi_ai.theming.schemas import GeneratedTheme

        system_content = PromptLoader.render("theme_metadata_system")
        user_content = PromptLoader.render(
            "theme_metadata_user",
            key=key,
            prompt=prompt,
        )
        messages = [
            SystemMessage(content=system_content),
            HumanMessage(content=user_content),
        ]
        return await ainvoke_structured(
            self.llm,
            messages,
            GeneratedTheme,
            max_retries=self._max_retries,
            base_delay=self._base_delay,
        )
