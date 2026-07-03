"""Read-only WordNet source access (Phase 3).

nltk WordNet lookups are synchronous, so they run in ``asyncio.to_thread``.
Lemmas with spaces are mapped to underscore form (``make up`` -> ``make_up``).
Zero synsets is a normal outcome for idioms/expressions and must not raise.
"""

import asyncio
from dataclasses import dataclass, field

from nltk.corpus import wordnet as wn

# WordNet single-letter POS -> Cambridge-style POS name.
_WN_POS = {"n": "noun", "v": "verb", "a": "adjective", "s": "adjective", "r": "adverb"}

# Cambridge-style POS name -> WordNet POS tag, for optional filtering.
_POS_TO_WN = {"noun": "n", "verb": "v", "adjective": "a", "adverb": "r"}


@dataclass
class WnSense:
    name: str
    pos: str | None
    definition: str
    lemmas: list[str] = field(default_factory=list)
    examples: list[str] = field(default_factory=list)


def _to_wn_lemma(lemma: str) -> str:
    return lemma.strip().replace(" ", "_")


class WordNetSource:
    """Thin async wrapper over ``nltk.corpus.wordnet``."""

    def _lookup_sync(self, lemma: str, pos: str | None) -> list[WnSense]:
        wn_lemma = _to_wn_lemma(lemma)
        wn_pos = _POS_TO_WN.get(pos) if pos else None
        synsets = wn.synsets(wn_lemma, pos=wn_pos) if wn_pos else wn.synsets(wn_lemma)
        result: list[WnSense] = []
        for s in synsets:
            result.append(
                WnSense(
                    name=s.name(),
                    pos=_WN_POS.get(s.pos()),
                    definition=s.definition(),
                    lemmas=[lm.name().replace("_", " ") for lm in s.lemmas()],
                    examples=list(s.examples()),
                )
            )
        return result

    async def lookup(self, lemma: str, pos: str | None = None) -> list[WnSense]:
        """Return synsets for ``lemma`` (optionally POS-filtered); may be empty."""
        return await asyncio.to_thread(self._lookup_sync, lemma, pos)
