"""On-demand learner dictionary.

Importing this package does not initialize clients or database connections.
"""

from .inference.config import DecisionConfig, LLMConfig
from .models import TokenUsage

__all__ = ["DecisionConfig", "LLMConfig", "Lexicon", "TokenUsage"]


def __getattr__(name: str):
    if name == "Lexicon":
        from .api import Lexicon

        return Lexicon
    raise AttributeError(name)
