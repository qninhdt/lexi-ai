"""A TypeSafe-backed decision model with an optional whole-request fallback."""

import math
from collections.abc import Mapping

from typesafe_sdk import AsyncTypeSafeClient, Choice, Noul

from ..errors import InvalidOutputError, MissingProviderError
from .config import DecisionConfig, DecisionMode, LLMConfig
from .usage import UsageRecorder, decision_usage


class DecisionModel:
    def __init__(
        self,
        config: DecisionConfig,
        primary=None,
        fallback=None,
        *,
        llm_config: LLMConfig | None = None,
        fallback_model: str | None = None,
    ):
        self.config = config
        self.primary = primary
        self.fallback = fallback
        self.llm_config = llm_config or LLMConfig()
        self.fallback_model = fallback_model
        self._own_primary = False
        self._own_fallback = False
        self._fallback_provider = None

    def _primary(self):
        if self.primary is None:
            if not self.config.api_key or not self.config.api_key.strip():
                raise MissingProviderError("decision model credentials not configured")
            self.primary = AsyncTypeSafeClient(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
                model=self.config.model,
            )
            self._own_primary = True
        return self.primary

    def _fallback(self, *, mode: DecisionMode = DecisionMode.LLM_FALLBACK):
        if self.fallback is None:
            model = self.fallback_model or self.llm_config.model
            if not model:
                raise MissingProviderError("LLM decision fallback model not configured")
            if not self.llm_config.api_key or not self.llm_config.api_key.strip():
                raise MissingProviderError("LLM decision fallback credentials not configured")
            from system_one_adapter import AsyncSystemOneAdapterClient
            from system_one_adapter.providers.openai import AsyncOpenAIProvider

            # An explicit provider prevents the adapter from reading SDK environment defaults.
            options = {}
            if not self.llm_config.structured_outputs:
                # The adapter's Responses text path still requires JSON mode.
                # Chat Completions supports prompted text without JSON enforcement.
                options["api"] = "chat_completions"
            provider_type = AsyncOpenAIProvider
            if self.llm_config.temperature is not None:
                from .adapter_provider import TemperatureOpenAIProvider

                provider_type = TemperatureOpenAIProvider
                options["temperature"] = self.llm_config.temperature
            provider = provider_type(
                model,
                api_key=self.llm_config.api_key,
                base_url=self.llm_config.base_url,
                **options,
            )
            self.fallback = AsyncSystemOneAdapterClient(
                structured_outputs=self.llm_config.structured_outputs,
                llm_answer_mode="discrete",
                n_retry_malformed_structure=0,
                model=provider,
            )
            self._fallback_provider = provider
            self._own_fallback = True
        return self.fallback

    async def decide(
        self,
        state: dict,
        questions: Mapping[str, Noul | Choice],
        *,
        mode: DecisionMode = DecisionMode.LLM_FALLBACK,
        with_usage: bool = False,
    ):
        """Choose Decision, LLM, or the default Decision-to-LLM fallback path."""
        with UsageRecorder(with_usage) as usage:
            mode = DecisionMode(mode)

            async def request(client):
                try:
                    response = await client.system_one(state=state, questions=questions)
                except Exception as error:
                    if with_usage:
                        usage.records.extend(decision_usage(error))
                    raise
                if with_usage:
                    usage.records.extend(decision_usage(response))
                return response

            use_llm = mode == DecisionMode.LLM_ONLY or (
                mode == DecisionMode.LLM_FALLBACK
                and self.primary is None
                and not (self.config.api_key and self.config.api_key.strip())
            )
            client = self._fallback(mode=mode) if use_llm else self._primary()
            response = await request(client)
            self._validate(response, questions)
            if (
                mode == DecisionMode.LLM_FALLBACK
                and not use_llm
                and any(
                    not self.config.accepts(response.choices[name].confidence)
                    for name, question in questions.items()
                    if isinstance(question, Choice)
                )
            ):
                response = await request(self._fallback())
                self._validate(response, questions)
            return usage.finish(response)

    @staticmethod
    def _validate(response, questions):
        for name, question in questions.items():
            if isinstance(question, Choice):
                answer = response.choices.get(name)
                if answer is None or answer.choice not in question.criteria:
                    raise InvalidOutputError("invalid decision option")
                value = answer.confidence
            else:
                answer = response.nouls.get(name)
                if answer is None:
                    raise InvalidOutputError("missing decision boolean")
                value = answer.noul
            if (
                not isinstance(value, (float, int))
                or not math.isfinite(value)
                or not 0 <= value <= 1
            ):
                raise InvalidOutputError("invalid decision probability")

    async def close(self) -> None:
        if self._own_primary and self.primary is not None:
            await self.primary.aclose()
            self.primary = None
            self._own_primary = False
        if self._own_fallback and self.fallback is not None:
            await self.fallback.aclose()
            self.fallback = None
            self._own_fallback = False
        if self._fallback_provider is not None:
            await self._fallback_provider.aclose()
            self._fallback_provider = None
