"""Themed generation: restyle a done word's senses in a named voice.

The neutral entry stays canonical; themed definitions and fresh in-voice examples overlay
it. Generation is anchored to neutral sense FACTS (definition, pos, guideword, tier), never
neutral examples, which are re-authored in-voice.
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING

from lexi_ai.prompts import PromptLoader
from lexi_ai.providers.base import StructuredCall
from lexi_ai.providers.seam import ainvoke_structured, guarded_messages
from lexi_ai.schemas import ThemedResult

if TYPE_CHECKING:
    from lexi_ai.schemas import ExampleBatch, ExampleGenContext, GeneratedTheme


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lexi_ai.schemas import ExampleBatch, ExampleGenContext, GeneratedTheme


class ThemedGenerator(StructuredCall):
    """Turn a style prompt + neutral sense facts into a ThemedResult."""

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
        # style_prompt is user-authored (theme creation), so the user turn is untrusted;
        # route it through guarded_messages like every other LM caller.
        messages = guarded_messages(system_content, user_content)
        return await ainvoke_structured(
            self.llm,
            messages,
            ThemedResult,
            max_retries=self._max_retries,
            base_delay=self._base_delay,
        )

    async def generate_examples(
        self,
        style_prompt: str,
        sense: "ExampleGenContext",
        existing: Sequence[str],
        n: int,
    ) -> "ExampleBatch":
        """Author up to ``n`` fresh in-voice tagged examples for ONE themed sense.

        Themed counterpart to :meth:`Generator.generate_examples`: same
        :class:`ExampleBatch` schema and ``<t inf>`` tag contract, plus the voice
        from ``style_prompt``. ``n`` is a best-effort max.
        """
        from lexi_ai.schemas import ExampleBatch

        system_content = PromptLoader.render("themed_example_augment_system")
        user_content = PromptLoader.render(
            "themed_example_augment_user",
            style_prompt=style_prompt,
            definition=sense.definition,
            pos=sense.pos,
            guideword=sense.guideword,
            tier=sense.tier,
            forms=sense.forms,
            existing=list(existing),
            n=n,
        )
        messages = guarded_messages(system_content, user_content)
        return await ainvoke_structured(
            self.llm,
            messages,
            ExampleBatch,
            max_retries=self._max_retries,
            base_delay=self._base_delay,
        )


class ThemeMetadataGenerator(StructuredCall):
    """Turn a name/key and a style concept prompt into a detailed theme."""

    async def generate(self, key: str, prompt: str) -> "GeneratedTheme":
        from lexi_ai.schemas import GeneratedTheme

        system_content = PromptLoader.render("theme_metadata_system")
        user_content = PromptLoader.render(
            "theme_metadata_user",
            key=key,
            prompt=prompt,
        )
        messages = guarded_messages(system_content, user_content)
        return await ainvoke_structured(
            self.llm,
            messages,
            GeneratedTheme,
            max_retries=self._max_retries,
            base_delay=self._base_delay,
        )
