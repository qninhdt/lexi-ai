"""Configuration for model-backed inference operations."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DecisionConfig:
    threshold: float
    api_key: str | None = field(default=None, kw_only=True, repr=False)
    base_url: str = field(default="https://api.typesafe.ai", kw_only=True)
    model: str = field(default="jev-latest", kw_only=True)

    def __post_init__(self) -> None:
        if not 0 < self.threshold <= 1:
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

    def __post_init__(self) -> None:
        if not self.model or not 1 <= self.timeout <= 120:
            raise ValueError("invalid model or timeout")
        if not self.model.strip() or not self.base_url.strip():
            raise ValueError("LLM model and base URL must not be blank")
        if not 1 <= self.max_completion_tokens <= 16384:
            raise ValueError("invalid completion token ceiling")
