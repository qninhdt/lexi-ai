"""Lexi-AI: lazy LLM dictionary library.

Exports resolve lazily so importing a submodule — ``lexi_ai.questions.types`` in
particular — stays dependency-free and does not initialize the database or
provider stack.
"""

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from lexi_ai.lexicon import Lexicon
    from lexi_ai.models import (
        Asset,
        BatchResult,
        Entry,
        SearchResult,
        TagCount,
        Theme,
        TopicView,
    )
    from lexi_ai.questions.types import (
        AnswerSubmission,
        ChoiceResponse,
        ChoiceReveal,
        Evaluation,
        Flashcard,
        FreeText,
        PrepareDemand,
        PrepareReport,
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
    from lexi_ai.text import match_key, parse_marked_example, render, strip_markup


_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "Lexicon": ("lexi_ai.lexicon", "Lexicon"),
    "AnswerSubmission": ("lexi_ai.questions.types", "AnswerSubmission"),
    "ChoiceResponse": ("lexi_ai.questions.types", "ChoiceResponse"),
    "ChoiceReveal": ("lexi_ai.questions.types", "ChoiceReveal"),
    "Evaluation": ("lexi_ai.questions.types", "Evaluation"),
    "Flashcard": ("lexi_ai.questions.types", "Flashcard"),
    "FreeText": ("lexi_ai.questions.types", "FreeText"),
    "PrepareDemand": ("lexi_ai.questions.types", "PrepareDemand"),
    "PrepareReport": ("lexi_ai.questions.types", "PrepareReport"),
    "PresentedQuestion": ("lexi_ai.questions.types", "PresentedQuestion"),
    "QuestionTypeInfo": ("lexi_ai.questions.types", "QuestionTypeInfo"),
    "RenderContract": ("lexi_ai.questions.types", "RenderContract"),
    "RenderKind": ("lexi_ai.questions.types", "RenderKind"),
    "Response": ("lexi_ai.questions.types", "Response"),
    "Reveal": ("lexi_ai.questions.types", "Reveal"),
    "RubricReveal": ("lexi_ai.questions.types", "RubricReveal"),
    "SingleChoice": ("lexi_ai.questions.types", "SingleChoice"),
    "SpanReveal": ("lexi_ai.questions.types", "SpanReveal"),
    "TextResponse": ("lexi_ai.questions.types", "TextResponse"),
    "TextSpan": ("lexi_ai.questions.types", "TextSpan"),
    "Asset": ("lexi_ai.models", "Asset"),
    "BatchResult": ("lexi_ai.models", "BatchResult"),
    "Entry": ("lexi_ai.models", "Entry"),
    "SearchResult": ("lexi_ai.models", "SearchResult"),
    "TagCount": ("lexi_ai.models", "TagCount"),
    "Theme": ("lexi_ai.models", "Theme"),
    "TopicView": ("lexi_ai.models", "TopicView"),
    "match_key": ("lexi_ai.text", "match_key"),
    "render": ("lexi_ai.text", "render"),
    "parse_marked_example": ("lexi_ai.text", "parse_marked_example"),
    "strip_markup": ("lexi_ai.text", "strip_markup"),
}


def __getattr__(name: str) -> Any:
    """Resolve a root export on first use."""
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = target
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


__all__ = [
    "Lexicon",
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
    "TagCount",
    "Theme",
    "TopicView",
    # Text helpers.
    "match_key",
    "parse_marked_example",
    "render",
    "strip_markup",
]
