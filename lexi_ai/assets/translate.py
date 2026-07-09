"""LLM translation provider (Phase 5).

Mirrors :mod:`lexi_ai.generation.generator`: an OpenAI-compatible chat model
bound to a tiny structured-output schema, injectable for hermetic tests. The
source text is placed in a delimited user turn (defense-in-depth against
instruction injection — it is dictionary content, but treated as data).
"""

from pydantic import BaseModel, Field

from lexi_ai.config import Settings, get_settings
from lexi_ai.llm import (
    StructuredLLM,
    ainvoke_structured,
    build_structured_llm,
    guarded_messages,
)
from lexi_ai.prompts import PromptLoader


class TranslatedText(BaseModel):
    text: str = Field(description="The translation of the source text into the target language.")


def build_translate_llm(settings: Settings) -> StructuredLLM:
    """openai-backed structured LLM for translation. Model = ``translate_model``
    or ``llm_model``. Imported lazily so fakes need no creds."""
    return build_structured_llm(settings, model=settings.translate_model or settings.llm_model)


class Translator:
    """Translate a string into a target language via a structured LLM call."""

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
            self._llm = build_translate_llm(self._settings)
        return self._llm

    async def translate(self, text: str, lang: str) -> str:
        system_content = PromptLoader.render("translate_system")
        user_content = PromptLoader.render("translate_user", lang=lang, text=text)
        messages = guarded_messages(system_content, user_content)
        result = await ainvoke_structured(
            self.llm,
            messages,
            TranslatedText,
            max_retries=self._max_retries,
            base_delay=self._base_delay,
        )
        return result.text
