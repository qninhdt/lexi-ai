"""On-demand learner dictionary.

Importing this package does not initialize clients or database connections.
"""

import importlib

from .inference.config import DecisionConfig, DecisionMode, LLMConfig
from .inference.llm import StructuredLLM
from .models import TokenUsage
from .vocab import (
    ALLOWED_PAIRS,
    FIXED_STEM_QUESTION_TYPES,
    QUESTION_FORMATS,
    CEFRLevel,
    DefinitionAccuracy,
    DefinitionCoverage,
    EntryType,
    GenerationState,
    Inflection,
    MatchKind,
    PartOfSpeech,
    QuestionType,
    Register,
    ResolutionState,
    ResponseFormat,
    SenseRelationType,
    TargetPlacement,
    Tier,
    UsageAppropriacy,
    UsageCollocation,
    UsageForm,
    UsageMeaning,
    WordRelationType,
)

__all__ = [
    "ALLOWED_PAIRS",
    "FIXED_STEM_QUESTION_TYPES",
    "QUESTION_FORMATS",
    "CEFRLevel",
    "EntryType",
    "GenerationState",
    "Inflection",
    "MatchKind",
    "PartOfSpeech",
    "QuestionType",
    "Register",
    "ResolutionState",
    "ResponseFormat",
    "SenseRelationType",
    "TargetPlacement",
    "Tier",
    "WordRelationType",
    "DefinitionAccuracy",
    "DefinitionCoverage",
    "UsageMeaning",
    "UsageForm",
    "UsageCollocation",
    "UsageAppropriacy",
    "DecisionConfig",
    "DecisionMode",
    "LEXI_SCHEMA",
    "LLMConfig",
    "Lexicon",
    "SENSE_PRIMARY_KEY",
    "Sense",
    "StructuredLLM",
    "TokenUsage",
    "metadata",
    "migrations",
]


def __getattr__(name: str):
    if name == "Lexicon":
        from .api import Lexicon

        globals()["Lexicon"] = Lexicon
        return Lexicon
    if name in {"LEXI_SCHEMA", "SENSE_PRIMARY_KEY", "Sense", "metadata"}:
        from . import contract

        for k in ("LEXI_SCHEMA", "SENSE_PRIMARY_KEY", "Sense", "metadata"):
            globals()[k] = getattr(contract, k)
        return globals()[name]
    if name == "migrations":
        mod = importlib.import_module(".migrations", __name__)
        globals()["migrations"] = mod
        return mod
    raise AttributeError(name)
