"""Shared model transports, configuration and rendering; prompts belong to domains."""

from .config import DecisionConfig, LLMConfig

__all__ = [
    "DecisionConfig",
    "DecisionModel",
    "LLMConfig",
    "OpenAIStructuredLLM",
    "StructuredLLM",
]


def __getattr__(name: str):
    if name == "DecisionModel":
        from .decision import DecisionModel

        return DecisionModel
    if name in {"OpenAIStructuredLLM", "StructuredLLM"}:
        from . import llm

        return getattr(llm, name)
    raise AttributeError(name)
