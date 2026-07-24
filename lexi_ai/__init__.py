"""Lexi-AI: lazy LLM dictionary library.

Synthesizes an English learner's dictionary with an LLM, anchored to Cambridge
and WordNet for hallucination control. See ``plans/`` for the design.
"""

from lexi_ai.api import Lexicon
from lexi_ai.contracts.questions import (
    AnswerSubmission,
    ChoiceResponse,
    ChoiceReveal,
    Evaluation,
    Flashcard,
    FreeText,
    PrepareDemand,
    PresentedQuestion,
    QuestionTypeInfo,
    RenderContract,
    RenderKind,
    Response,
    Reveal,
    RubricReveal,
    SingleChoice,
    SpanReveal,
    TextResponse,
    TextSpan,
)
from lexi_ai.facades import LexiconEngine, LexiconReader
from lexi_ai.markup import parse_marked_example, strip_markup
from lexi_ai.normalize import match_key, render
from lexi_ai.questions.base import PrepareReport
from lexi_ai.read_models import (
    Asset,
    BatchResult,
    Entry,
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
    # Answer-safe question contract surface.
    "AnswerSubmission",
    "ChoiceResponse",
    "ChoiceReveal",
    "Evaluation",
    "Flashcard",
    "FreeText",
    "PrepareDemand",
    "PrepareReport",
    "PresentedQuestion",
    "QuestionTypeInfo",
    "RenderContract",
    "RenderKind",
    "Response",
    "Reveal",
    "RubricReveal",
    "SingleChoice",
    "SpanReveal",
    "TextResponse",
    "TextSpan",
    # Dictionary read models.
    "Asset",
    "BatchResult",
    "Entry",
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
