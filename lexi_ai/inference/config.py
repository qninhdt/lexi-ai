"""Configuration for model-backed inference operations."""

import math
from dataclasses import dataclass, field
from enum import StrEnum


class DecisionMode(StrEnum):
    LLM_FALLBACK = "LLM_FALLBACK"
    DECISION_ONLY = "DECISION_ONLY"
    LLM_ONLY = "LLM_ONLY"


@dataclass(frozen=True)
class DecisionConfig:
    threshold: float
    api_key: str | None = field(default=None, kw_only=True, repr=False)
    base_url: str = field(default="https://api.typesafe.ai", kw_only=True)
    model: str = field(default="jev-latest", kw_only=True)

    def __post_init__(self) -> None:
        if type(self.threshold) not in (float, int) or not 0 < self.threshold <= 1:
            raise ValueError("decision threshold must be in (0, 1]")
        if not self.model.strip() or not self.base_url.strip():
            raise ValueError("decision model and base URL must not be blank")

    def accepts(self, value: float) -> bool:
        """Use the same inclusive decision boundary for probability and confidence."""
        return value >= self.threshold


@dataclass(frozen=True)
class LLMConfig:
    model: str = "gpt-4o"
    api_key: str | None = field(default=None, repr=False)
    base_url: str = "https://api.openai.com/v1"
    timeout: float = 30.0
    max_completion_tokens: int = 4096
    reasoning_effort: str | None = None
    structured_outputs: bool = True
    temperature: float | None = None
    max_retries: int = 2

    def __post_init__(self) -> None:
        if type(self.structured_outputs) is not bool:
            raise TypeError("structured_outputs must be a boolean")
        if type(self.max_retries) is not int or self.max_retries < 0:
            raise ValueError("max_retries must be a non-negative integer")
        if self.reasoning_effort is not None and (
            not isinstance(self.reasoning_effort, str) or not self.reasoning_effort.strip()
        ):
            raise ValueError("reasoning_effort must be a nonblank string or None")
        if self.temperature is not None and (
            type(self.temperature) not in (int, float)
            or not math.isfinite(self.temperature)
            or not 0 <= self.temperature <= 2
        ):
            raise ValueError("temperature must be a finite number in [0, 2] or None")
        if not self.model or not 1 <= self.timeout <= 120:
            raise ValueError("invalid model or timeout")
        if not self.model.strip() or not self.base_url.strip():
            raise ValueError("LLM model and base URL must not be blank")
        if not 1 <= self.max_completion_tokens <= 32768:
            raise ValueError("invalid completion token ceiling")
