"""The shared shape of a task-bound LLM caller.

One task bound to one model call: a settings snapshot, an injectable
``StructuredLLM``, a lazily built real client, and a retry policy.

A subclass supplies :meth:`build_llm` and inherits everything else. Injection is the
seam that keeps the suite hermetic: passing a fake means no credential and no network,
which is why ``_llm`` is stored rather than built eagerly.
"""

from lexi_ai.config import Settings, get_settings
from lexi_ai.providers.seam import StructuredLLM, build_structured_llm


class StructuredCall:
    def __init__(
        self,
        structured_llm: StructuredLLM | None = None,
        settings: Settings | None = None,
        max_retries: int = 3,
        base_delay: float = 0.5,
    ) -> None:
        self._settings = settings or get_settings()
        self._llm = structured_llm
        self._max_retries = max_retries
        self._base_delay = base_delay

    @property
    def llm(self) -> StructuredLLM:
        """The model, built on first access when none was injected."""
        if self._llm is None:
            self._llm = self.build_llm()
        return self._llm

    def build_llm(self) -> StructuredLLM:
        """Build the real client from settings. Override to change the model."""
        return build_structured_llm(self._settings)
