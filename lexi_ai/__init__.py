"""On-demand learner dictionary.

Importing this package does not initialize clients or database connections.
"""

from .inference.config import DecisionConfig, DecisionMode, LLMConfig
from .inference.llm import StructuredLLM
from .models import TokenUsage

__all__ = [
    "DecisionConfig",
    "DecisionMode",
    "LLMConfig",
    "Lexicon",
    "StructuredLLM",
    "TokenUsage",
]


def __getattr__(name: str):
    if name == "Lexicon":
        from .api import Lexicon

        return Lexicon
    raise AttributeError(name)
