"""LLM generation pipeline (Phase 4).

Wraps an OpenAI-compatible chat model (via the ``openai`` SDK) bound to the
:class:`GeneratedResult` structured-output schema. The model is injectable so
unit tests pass a fake :class:`StructuredLLM` and never touch the network.

Retry: transient errors are retried with exponential backoff up to a small cap;
low temperature keeps output roughly deterministic.
"""

from collections.abc import Sequence

from lexi_ai.config import Settings, get_settings
from lexi_ai.generation.schemas import GeneratedResult
from lexi_ai.llm import StructuredLLM, ainvoke_structured, build_structured_llm, sys_msg, user_msg
from lexi_ai.prompts import PromptLoader
from lexi_ai.references.loader import ReferenceBundle


class Generator:
    """Turn a ReferenceBundle into a validated GeneratedResult."""

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
        # Lazily build the real model only when first needed.
        if self._llm is None:
            self._llm = build_structured_llm(self._settings)
        return self._llm

    async def generate(
        self, bundle: ReferenceBundle, existing_tags: Sequence[tuple[str, str]] = ()
    ) -> GeneratedResult:
        system_content = PromptLoader.render("senses_generation_system")
        alts = (
            ", ".join(f"{a}({t})" for a, t in bundle.cambridge_alternatives)
            if bundle.cambridge_alternatives
            else None
        )
        user_content = PromptLoader.render(
            "senses_generation_user",
            word=bundle.word_raw,
            cambridge_word_raw=bundle.word_raw,
            cambridge_entry_type=bundle.entry_type,
            cambridge_senses=bundle.cambridge_senses,
            cambridge_alternatives=alts,
            wordnet_synsets=bundle.wordnet_synsets,
            existing_tags=existing_tags,
        )
        messages = [sys_msg(system_content), user_msg(user_content)]
        return await ainvoke_structured(
            self.llm,
            messages,
            GeneratedResult,
            max_retries=self._max_retries,
            base_delay=self._base_delay,
        )
