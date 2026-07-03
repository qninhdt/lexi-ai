"""Read-only access to anchor sources (Cambridge + WordNet).

Produces a :class:`ReferenceBundle` that the LLM prompt (Phase 4) consumes for
hallucination control.
"""

from lexi_ai.references.cambridge import (
    CambridgeEntry,
    CambridgeSource,
    CamSense,
)
from lexi_ai.references.loader import ReferenceBundle, ReferenceLoader
from lexi_ai.references.wordnet import WnSense, WordNetSource

__all__ = [
    "CambridgeSource",
    "CambridgeEntry",
    "CamSense",
    "WordNetSource",
    "WnSense",
    "ReferenceLoader",
    "ReferenceBundle",
]
