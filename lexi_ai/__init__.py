"""Lexi-AI: lazy LLM dictionary library.

Synthesizes an English learner's dictionary with an LLM, anchored to Cambridge
and WordNet for hallucination control. See ``plans/`` for the design.
"""

from lexi_ai.api import Lexicon
from lexi_ai.normalize import match_key, render
from lexi_ai.read_models import Entry, SearchResult, SemanticHit, TagCount, TopicView

__all__ = [
    "Lexicon",
    "Entry",
    "SearchResult",
    "SemanticHit",
    "TagCount",
    "TopicView",
    "match_key",
    "render",
]
