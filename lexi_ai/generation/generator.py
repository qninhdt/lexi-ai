"""LLM generation pipeline (Phase 4).

Wraps an OpenAI-compatible chat model (via the ``openai`` SDK) bound to the
:class:`GeneratedResult` structured-output schema. The model is injectable so
unit tests pass a fake :class:`StructuredLLM` and never touch the network.

Retry: transient errors are retried with exponential backoff up to a small cap;
low temperature keeps output roughly deterministic.
"""

from collections.abc import Sequence

from lexi_ai.config import Settings, get_settings
from lexi_ai.generation.schemas import ExampleBatch, ExampleGenContext, GeneratedResult
from lexi_ai.llm import StructuredLLM, ainvoke_structured, build_structured_llm, guarded_messages
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
        self._injected_llm = structured_llm is not None
        self._llms_by_method: dict[str, StructuredLLM] = {}
        self._max_retries = max_retries
        self._base_delay = base_delay

    @property
    def llm(self) -> StructuredLLM:
        # Lazily build the real model only when first needed.
        if self._llm is None:
            self._llm = build_structured_llm(self._settings)
        return self._llm

    async def generate(
        self,
        bundle: ReferenceBundle,
        existing_tags: Sequence[tuple[str, str]] = (),
        *,
        structured_method: str | None = None,
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
        messages = guarded_messages(system_content, user_content)
        return await ainvoke_structured(
            self._llm_for_method(structured_method),
            messages,
            GeneratedResult,
            max_retries=self._max_retries,
            base_delay=self._base_delay,
        )

    def _llm_for_method(self, structured_method: str | None) -> StructuredLLM:
        """Select a per-job output mode without mutating shared generator state."""
        if structured_method is None:
            return self.llm
        if structured_method not in {"json_schema", "function_calling"}:
            raise ValueError("unsupported structured LLM method")
        # An injected LLM is a caller-provided test/integration seam, not a
        # production client that may be replaced according to settings.
        if self._injected_llm:
            return self.llm
        if self._llm is not None and structured_method == self._settings.llm_structured_method:
            return self._llm
        if structured_method not in self._llms_by_method:
            self._llms_by_method[structured_method] = build_structured_llm(
                self._settings.model_copy(update={"llm_structured_method": structured_method})
            )
        return self._llms_by_method[structured_method]

    async def generate_examples(
        self, sense: ExampleGenContext, existing: Sequence[str], n: int
    ) -> ExampleBatch:
        """Author up to ``n`` fresh tagged example sentences for ONE sense.

        Targeted counterpart to :meth:`generate` (which produces a whole word):
        feeds the sense's facts + its existing examples (soft dedup) to the model
        and returns a validated :class:`ExampleBatch`. ``n`` is a best-effort max.
        """
        system_content = PromptLoader.render("example_augment_system")
        user_content = PromptLoader.render(
            "example_augment_user",
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
