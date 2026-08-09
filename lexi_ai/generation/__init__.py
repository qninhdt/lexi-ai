"""Generation package: ReferenceBundle -> validated GeneratedResult."""

from lexi_ai.generation.generator import Generator
from lexi_ai.generation.schemas import (
    GeneratedAlias,
    GeneratedEntry,
    GeneratedReference,
    GeneratedResult,
    GeneratedSense,
    RelatedWord,
)

__all__ = [
    "Generator",
    "GeneratedResult",
    "GeneratedEntry",
    "GeneratedSense",
    "GeneratedAlias",
    "GeneratedReference",
    "RelatedWord",
]
