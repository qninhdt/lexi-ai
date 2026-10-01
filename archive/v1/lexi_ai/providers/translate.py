"""LLM translation provider.

An OpenAI-compatible chat model bound to a tiny structured-output schema,
injectable for hermetic tests. The source text is placed in a delimited user turn
(defense-in-depth against instruction injection — it is dictionary content, but
treated as data).
"""

from pydantic import BaseModel, Field

from lexi_ai.prompts import PromptLoader
from lexi_ai.providers.base import StructuredCall
from lexi_ai.providers.seam import (
    StructuredLLM,
    ainvoke_structured,
    build_structured_llm,
    guarded_messages,
)


class TranslatedText(BaseModel):
    text: str = Field(description="The translation of the source text into the target language.")


class Translator(StructuredCall):
    def build_llm(self) -> StructuredLLM:
        """Translation uses its own model when one is configured."""
        return build_structured_llm(
            self._settings, model=self._settings.translate_model or self._settings.llm_model
        )

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
