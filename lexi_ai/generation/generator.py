"""LLM generation pipeline (Phase 4).

Wraps an OpenAI-compatible chat model (via LangChain 1.x) bound to the
:class:`GeneratedResult` structured-output schema. The model is injectable so
unit tests pass a fake runnable and never touch the network.

Retry: transient errors are retried with exponential backoff up to a small cap;
low temperature keeps output roughly deterministic.
"""

import asyncio
from collections.abc import Sequence

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable

from lexi_ai.config import Settings, get_settings
from lexi_ai.generation.schemas import GeneratedResult
from lexi_ai.llm import ainvoke_structured
from lexi_ai.prompts import PromptLoader
from lexi_ai.references.loader import ReferenceBundle


def build_structured_llm(settings: Settings) -> Runnable:
    """Build the ChatOpenAI runnable bound to the structured-output schema.

    Uses the LangChain 1.x ``with_structured_output`` API (default json_schema
    method). Imported lazily so tests that inject a fake model need no creds.
    """
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=settings.llm_model,
        temperature=settings.llm_temperature,
    )
    return llm.with_structured_output(GeneratedResult)


class Generator:
    """Turn a ReferenceBundle into a validated GeneratedResult."""

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
        messages = [
            SystemMessage(content=system_content),
            HumanMessage(content=user_content),
        ]
        return await ainvoke_structured(
            self.llm,
            messages,
            GeneratedResult,
            max_retries=self._max_retries,
            base_delay=self._base_delay,
        )
