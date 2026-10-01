"""Lazy factories for the optional external providers.

Each factory answers *is this capability configured, and if so what object implements
it?*, returning ``None`` (or, for TTS, a loudly failing stub) when the settings are absent
— so an install without an LLM/TTS key degrades instead of exploding at import time. This
is the single home for that branching and holds the built instances, so a provider is
constructed at most once per process. Fields are writable, which is how tests inject fakes
without patching settings.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from lexi_ai.config import Settings, get_settings

if TYPE_CHECKING:
    from lexi_ai.providers.generate import Generator
    from lexi_ai.providers.seam import StructuredLLM
    from lexi_ai.providers.wsd import WsdJudge


class ProviderRegistry:
    """Builds the LLM / TTS / translation providers on first use.

    ``generator`` and ``wsd_judge`` are pre-built collaborators the owner may
    inject; the rest come from one :class:`Settings` snapshot, or from the ambient
    settings factory when none is supplied.
    """

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        generator: Generator | None = None,
        wsd_judge: WsdJudge | None = None,
    ) -> None:
        self._settings_value = settings
        self.generator = generator
        # ``None`` means not-yet-built, except the WSD judge whose "not configured"
        # answer is also ``None`` — hence _wsd_built.
        self.translator: Any | None = None
        self.tts: Any | None = None
        self.themed_generator: Any | None = None
        self.theme_metadata_generator: Any | None = None
        self._wsd_judge = wsd_judge
        self._wsd_built = wsd_judge is not None

    def _settings(self) -> Settings:
        return self._settings_value or get_settings()

    def llm_configured(self) -> bool:
        return bool(self._settings().llm_api_key)

    def tts_configured(self) -> bool:
        """Whether a real speech provider is reachable (key or self-hosted URL)."""
        settings = self._settings()
        return bool(settings.tts_api_key or settings.tts_base_url)

    def structured_llm(self) -> StructuredLLM | None:
        """The openai-backed structured LLM, or ``None`` when none is configured.
        Built fresh per call: the schema comes from ``parse``, so nothing pins a
        stale snapshot.
        """
        if not self.llm_configured():
            return None
        from lexi_ai.providers.seam import build_structured_llm

        return build_structured_llm(self._settings())

    def questions_llm(self) -> StructuredLLM | None:
        return self.structured_llm()

    def judge_llm(self) -> StructuredLLM | None:
        return self.structured_llm()

    def wsd(self) -> WsdJudge | None:
        """The WSD judge for sense-relation reconciliation, built once.

        ``None`` when no LLM is configured — resolve then degrades to a no-op.
        """
        if not self._wsd_built:
            llm = self.structured_llm()
            if llm is None:
                self._wsd_judge = None
            else:
                from lexi_ai.providers.wsd import WsdJudge

                self._wsd_judge = WsdJudge(llm)
            self._wsd_built = True
        return self._wsd_judge

    def set_wsd(self, judge: WsdJudge | None) -> None:
        self._wsd_judge = judge
        self._wsd_built = True

    def example_generator(self) -> Generator:
        """The neutral generator, used for targeted example augmentation. Raises
        ``ValueError`` when none is wired: a caller asking for new sentences cannot
        be served with silence.
        """
        if self.generator is None:
            raise ValueError("no LLM configured for example generation")
        return self.generator

    def themed(self):
        if self.themed_generator is None:
            from lexi_ai.providers.theme import ThemedGenerator

            self.themed_generator = ThemedGenerator(settings=self._settings())
        return self.themed_generator

    def theme_metadata(self):
        if self.theme_metadata_generator is None:
            from lexi_ai.providers.theme import ThemeMetadataGenerator

            self.theme_metadata_generator = ThemeMetadataGenerator(settings=self._settings())
        return self.theme_metadata_generator

    def translator_provider(self):
        """The translator, or ``None`` when no LLM is configured."""
        if self.translator is not None:
            return self.translator
        if not self.llm_configured():
            return None
        from lexi_ai.providers.translate import Translator

        self.translator = Translator(settings=self._settings())
        return self.translator

    def tts_provider(self):
        """The real provider when configured, else the stub — which raises instead
        of returning audio, so an unconfigured install fails loudly.
        """
        if self.tts is not None:
            return self.tts
        settings = self._settings()
        if self.tts_configured():
            from lexi_ai.providers.tts import OpenAICompatibleTTSProvider

            self.tts = OpenAICompatibleTTSProvider(
                base_url=settings.tts_base_url,
                api_key=settings.tts_api_key,
                model=settings.tts_model,
            )
        else:
            from lexi_ai.providers.tts import StubTTSProvider

            self.tts = StubTTSProvider()
        return self.tts
