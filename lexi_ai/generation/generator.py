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
from lexi_ai.generation.prompts import SYSTEM_PROMPT, format_bundle
from lexi_ai.generation.schemas import GeneratedResult
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
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=format_bundle(bundle, existing_tags)),
        ]
        last_exc: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                result = await self.llm.ainvoke(messages)
                # with_structured_output returns the validated model instance.
                if isinstance(result, GeneratedResult):
                    return result
                # Some providers return a dict when include_raw or json_mode.
                return GeneratedResult.model_validate(result)
            except Exception as exc:  # noqa: BLE001 - retried, then re-raised
                last_exc = exc
                if attempt < self._max_retries - 1:
                    await asyncio.sleep(self._base_delay * (2**attempt))
        assert last_exc is not None
        raise last_exc
