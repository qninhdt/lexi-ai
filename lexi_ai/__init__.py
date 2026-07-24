"""Lexi-AI: lazy LLM dictionary library.

Synthesizes an English learner's dictionary with an LLM, anchored to Cambridge
and WordNet for hallucination control. See ``plans/`` for the design.
"""

from lexi_ai.api import Lexicon
from lexi_ai.facades import LexiconEngine, LexiconReader
from lexi_ai.markup import parse_marked_example, strip_markup
from lexi_ai.normalize import match_key, render
from lexi_ai.questions.base import PrepareReport, QuestionDemand, QuestionTypeDescriptor
from lexi_ai.read_models import (
    Asset,
    BatchResult,
    Entry,
    Evaluation,
    Question,
    SearchResult,
    SemanticHit,
    TagCount,
    Theme,
    TopicView,
)

__all__ = [
    "Lexicon",
    "LexiconEngine",
    "LexiconReader",
    "Asset",
    "BatchResult",
    "Entry",
    "Evaluation",
    "PrepareReport",
    "Question",
    "QuestionDemand",
    "QuestionTypeDescriptor",
    "SearchResult",
    "SemanticHit",
    "TagCount",
    "Theme",
    "TopicView",
    "match_key",
    "render",
    "parse_marked_example",
    "strip_markup",
]
