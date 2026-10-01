"""Optional generation-only WordNet evidence."""

import asyncio
from dataclasses import dataclass


@dataclass(frozen=True)
class Synset:
    key: str
    pos: str
    definition: str
    examples: list[str]


def _lookup(citation: str) -> list[Synset]:
    from nltk.corpus import wordnet

    try:
        return [
            Synset(item.name(), item.pos(), item.definition(), list(item.examples()))
            for item in wordnet.synsets(citation.replace(" ", "_"))[:20]
        ]
    except LookupError:
        return []


async def lookup(citation: str) -> list[Synset]:
    """Skip uncertain Cambridge displays and absent WordNet data, never guess a target."""
    if not citation or any(marker in citation for marker in ("/", "{", "}", "(", ")")):
        return []
    return await asyncio.to_thread(_lookup, citation)
