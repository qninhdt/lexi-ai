"""LLM themed-generation pipeline (Phase 2).

Mirrors :mod:`lexi_ai.generation.generator`: wraps an OpenAI-compatible chat
model bound to :class:`ThemedResult`, injectable for hermetic tests, retried
with exponential backoff.
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING

from lexi_ai.config import Settings, get_settings
from lexi_ai.llm import StructuredLLM, ainvoke_structured, build_structured_llm, sys_msg, user_msg
from lexi_ai.prompts import PromptLoader
from lexi_ai.theming.schemas import ThemedResult

if TYPE_CHECKING:
    from lexi_ai.theming.schemas import GeneratedTheme


def build_themed_llm(settings: Settings) -> StructuredLLM:
    """openai-backed structured LLM bound to ``ThemedResult`` at call time.

    Uses ``settings.llm_model``. Imported lazily so fakes need no creds.
    """
    return build_structured_llm(settings)


class ThemedGenerator:
    """Turn a style prompt + neutral sense facts into a validated ThemedResult."""

    def __init__(
        self,
        structured_llm: StructuredLLM | None = None,
        settings: Settings | None = None,
        max_retries: int = 3,
        base_delay: float = 0.5,
    ):
        self._settings = settings or get_settings()
        self._llm = structured_llm
        self._max_retries = max_retries
        self._base_delay = base_delay

    @property
    def llm(self) -> StructuredLLM:
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
        messages = [sys_msg(system_content), user_msg(user_content)]
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
        structured_llm: StructuredLLM | None = None,
        settings: Settings | None = None,
        max_retries: int = 3,
        base_delay: float = 0.5,
    ):
        self._settings = settings or get_settings()
        self._llm = structured_llm
        self._max_retries = max_retries
        self._base_delay = base_delay

    @property
    def llm(self) -> StructuredLLM:
        if self._llm is None:
            self._llm = build_structured_llm(self._settings)
        return self._llm

    async def generate(self, key: str, prompt: str) -> "GeneratedTheme":
        from lexi_ai.theming.schemas import GeneratedTheme

        system_content = PromptLoader.render("theme_metadata_system")
        user_content = PromptLoader.render(
            "theme_metadata_user",
            key=key,
            prompt=prompt,
        )
        messages = [sys_msg(system_content), user_msg(user_content)]
        return await ainvoke_structured(
            self.llm,
            messages,
            GeneratedTheme,
            max_retries=self._max_retries,
            base_delay=self._base_delay,
        )
