"""Lexi-AI: lazy LLM dictionary library.

The package root is a compatibility-friendly facade over the public API. Its
exports are resolved lazily so importing ``lexi_ai.contracts`` remains genuinely
dependency-free and does not initialize the database or provider stack.
"""

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
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


_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "Lexicon": ("lexi_ai.api", "Lexicon"),
    "LexiconEngine": ("lexi_ai.facades", "LexiconEngine"),
    "LexiconReader": ("lexi_ai.facades", "LexiconReader"),
    "AnswerSubmission": ("lexi_ai.contracts.questions", "AnswerSubmission"),
    "ChoiceResponse": ("lexi_ai.contracts.questions", "ChoiceResponse"),
    "ChoiceReveal": ("lexi_ai.contracts.questions", "ChoiceReveal"),
    "Evaluation": ("lexi_ai.contracts.questions", "Evaluation"),
    "Flashcard": ("lexi_ai.contracts.questions", "Flashcard"),
    "FreeText": ("lexi_ai.contracts.questions", "FreeText"),
    "PrepareDemand": ("lexi_ai.contracts.questions", "PrepareDemand"),
    "PresentedQuestion": ("lexi_ai.contracts.questions", "PresentedQuestion"),
    "QuestionTypeInfo": ("lexi_ai.contracts.questions", "QuestionTypeInfo"),
    "RenderContract": ("lexi_ai.contracts.questions", "RenderContract"),
    "RenderKind": ("lexi_ai.contracts.questions", "RenderKind"),
    "Response": ("lexi_ai.contracts.questions", "Response"),
    "Reveal": ("lexi_ai.contracts.questions", "Reveal"),
    "RubricReveal": ("lexi_ai.contracts.questions", "RubricReveal"),
    "SingleChoice": ("lexi_ai.contracts.questions", "SingleChoice"),
    "SpanReveal": ("lexi_ai.contracts.questions", "SpanReveal"),
    "TextResponse": ("lexi_ai.contracts.questions", "TextResponse"),
    "TextSpan": ("lexi_ai.contracts.questions", "TextSpan"),
    "PrepareReport": ("lexi_ai.questions.base", "PrepareReport"),
    "Asset": ("lexi_ai.read_models", "Asset"),
    "BatchResult": ("lexi_ai.read_models", "BatchResult"),
    "Entry": ("lexi_ai.read_models", "Entry"),
    "SearchResult": ("lexi_ai.read_models", "SearchResult"),
    "SemanticHit": ("lexi_ai.read_models", "SemanticHit"),
    "TagCount": ("lexi_ai.read_models", "TagCount"),
    "Theme": ("lexi_ai.read_models", "Theme"),
    "TopicView": ("lexi_ai.read_models", "TopicView"),
    "match_key": ("lexi_ai.normalize", "match_key"),
    "render": ("lexi_ai.normalize", "render"),
    "parse_marked_example": ("lexi_ai.markup", "parse_marked_example"),
    "strip_markup": ("lexi_ai.markup", "strip_markup"),
}


def __getattr__(name: str) -> Any:
    """Resolve a legacy root export on first use."""
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = target
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


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
