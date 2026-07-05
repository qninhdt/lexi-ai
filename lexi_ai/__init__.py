"""Lexi-AI: lazy LLM dictionary library.

Synthesizes an English learner's dictionary with an LLM, anchored to Cambridge
and WordNet for hallucination control. See ``plans/`` for the design.
"""

from lexi_ai.api import Lexicon
from lexi_ai.normalize import match_key, render
from lexi_ai.read_models import (
    Asset,
    BatchResult,
    Entry,
    Question,
    Score,
    SearchResult,
    SemanticHit,
    TagCount,
    Theme,
    TopicView,
)

__all__ = [
    "Lexicon",
    "Asset",
    "BatchResult",
    "Entry",
    "Question",
    "Score",
    "SearchResult",
    "SemanticHit",
    "TagCount",
    "Theme",
    "TopicView",
    "match_key",
    "render",
]
