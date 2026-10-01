"""Read-only access to anchor sources (Cambridge + WordNet), producing the
:class:`ReferenceBundle` the LLM prompt consumes for hallucination control.
"""

from lexi_ai.providers.references.bundle import ReferenceBundle, ReferenceLoader
from lexi_ai.providers.references.cambridge import (
    CambridgeEntry,
    CambridgeSource,
    CamSense,
)
from lexi_ai.providers.references.wordnet import WnSense, WordNetSource

__all__ = [
    "CambridgeSource",
    "CambridgeEntry",
    "CamSense",
    "WordNetSource",
    "WnSense",
    "ReferenceLoader",
    "ReferenceBundle",
]
