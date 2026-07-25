"""Lazy factories for the optional external providers.

Everything in here answers one question: *is this capability configured, and if so
what object implements it?* Each factory returns ``None`` (or, for TTS, a loudly
failing stub) when the corresponding settings are absent, so an install without an
LLM/TTS key degrades instead of exploding at import time.

The registry is the single home for that "configured?" branching. It holds the
built instances so a provider is constructed at most once per process, and every
field is writable, which is how tests inject fakes without patching settings.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from lexi_ai.config import Settings, get_settings

if TYPE_CHECKING:
    from lexi_ai.generation.generator import Generator
    from lexi_ai.generation.wsd import WsdJudge
    from lexi_ai.llm import StructuredLLM


class ProviderRegistry:
    """Builds the LLM / TTS / translation providers on first use.

    ``generator`` and ``wsd_judge`` are pre-built collaborators the owner may inject;
    the rest are constructed here from :class:`Settings`. Settings are read per call
    (not captured) so a test that swaps the settings singleton is honoured by the
    next build.
    """

    def __init__(
        self,
        *,
        generator: Generator | None = None,
        wsd_judge: WsdJudge | None = None,
    ) -> None:
        self.generator = generator
        # Cached instances. ``None`` means not-yet-built for everything except the
        # WSD judge, whose "not configured" answer is also ``None`` — hence the
        # separate _wsd_built flag.
        self.translator: Any | None = None
        self.tts: Any | None = None
        self.themed_generator: Any | None = None
        self.theme_metadata_generator: Any | None = None
        self._wsd_judge = wsd_judge
        self._wsd_built = wsd_judge is not None

    # --- settings ---------------------------------------------------------

    @staticmethod
    def _settings() -> Settings:
        return get_settings()

    def llm_configured(self) -> bool:
        """Whether a structured-output LLM can be built."""
        return bool(self._settings().llm_api_key)

    def tts_configured(self) -> bool:
        """Whether a real speech provider is reachable (key or self-hosted URL)."""
        settings = self._settings()
        return bool(settings.tts_api_key or settings.tts_base_url)

    # --- language models --------------------------------------------------

    def structured_llm(self) -> StructuredLLM | None:
        """The openai-backed structured LLM, or ``None`` when none is configured.

        Built fresh per call: the schema is supplied per-call by ``parse``, so one
        client type serves both the MCQ generator and the rubric judge, and holding
        it would only pin a stale settings snapshot.
        """
        if not self.llm_configured():
            return None
        from lexi_ai.llm import build_structured_llm

        return build_structured_llm(self._settings())

    def questions_llm(self) -> StructuredLLM | None:
        """Structured LLM for the contextual-MCQ plugin (bound to ``GeneratedMCQ``
        at the call site via :func:`ainvoke_structured`)."""
        return self.structured_llm()

    def judge_llm(self) -> StructuredLLM | None:
        """Structured LLM for the rubric scorer (bound to ``Judgment`` at the call
        site via :func:`ainvoke_structured`)."""
        return self.structured_llm()

    def wsd(self) -> WsdJudge | None:
        """The WSD judge for sense-relation reconciliation, built once.

        ``None`` when no LLM is configured — resolve then degrades to a no-op, like
        the other llm-dependent formats.
        """
        if not self._wsd_built:
            llm = self.structured_llm()
            if llm is None:
                self._wsd_judge = None
            else:
                from lexi_ai.generation.wsd import WsdJudge

                self._wsd_judge = WsdJudge(llm)
            self._wsd_built = True
        return self._wsd_judge

    def set_wsd(self, judge: WsdJudge | None) -> None:
        """Inject a WSD judge, marking it as already built."""
        self._wsd_judge = judge
        self._wsd_built = True

    # --- generation -------------------------------------------------------

    def example_generator(self) -> Generator:
        """The neutral generator, used for targeted example augmentation.

        Raises ``ValueError`` when none is wired, mirroring ``translate_field``'s
        no-LLM posture — a caller asking for new sentences cannot be served with
        silence.
        """
        if self.generator is None:
            raise ValueError("no LLM configured for example generation")
        return self.generator

    def themed(self):
        """Lazy themed generator; uses settings/OpenAI proxy by default."""
        if self.themed_generator is None:
            from lexi_ai.theming.generator import ThemedGenerator

            self.themed_generator = ThemedGenerator(settings=self._settings())
        return self.themed_generator

    def theme_metadata(self):
        """Lazy theme-metadata generator (expands a theme name into a profile)."""
        if self.theme_metadata_generator is None:
            from lexi_ai.theming.generator import ThemeMetadataGenerator

            self.theme_metadata_generator = ThemeMetadataGenerator(settings=self._settings())
        return self.theme_metadata_generator

    # --- assets -----------------------------------------------------------

    def translator_provider(self):
        """The translator, or ``None`` when no LLM is configured."""
        if self.translator is not None:
            return self.translator
        if not self.llm_configured():
            return None
        from lexi_ai.assets.translate import Translator

        self.translator = Translator(settings=self._settings())
        return self.translator

    def tts_provider(self):
        """The speech provider: the real one when configured, else the stub.

        The stub raises instead of returning audio, so an unconfigured install fails
        loudly rather than caching something fake.
        """
        if self.tts is not None:
            return self.tts
        settings = self._settings()
        if self.tts_configured():
            from lexi_ai.assets.tts import OpenAICompatibleTTSProvider

            self.tts = OpenAICompatibleTTSProvider(
                base_url=settings.tts_base_url,
                api_key=settings.tts_api_key,
                model=settings.tts_model,
            )
        else:
            from lexi_ai.assets.tts import StubTTSProvider

            self.tts = StubTTSProvider()
        return self.tts
