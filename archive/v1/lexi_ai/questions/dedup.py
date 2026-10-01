"""Shared distractor exclude-and-dedup helper.

Both :class:`DistractorProvider` and ``ContextualMCQ._merge_distractors`` enforce the SAME
correctness-sensitive rule: a candidate whose ``match_key`` collides with the target word, one of
its aliases, or an already-taken option must never appear as a distractor — otherwise the "wrong"
option could be the right answer.
"""

from lexi_ai.models import Entry
from lexi_ai.text import match_key


def exclude_keys(entry: Entry) -> set[str]:
    """The ``match_key`` set a distractor must never collide with: the target word plus every alias
    (a surface variant of the answer is still the answer).
    """
    keys = {match_key(entry.norm)}
    keys.update(match_key(a.alias_norm) for a in entry.aliases)
    return keys


class DistractorDedup:
    """Accumulate distractor displays, skipping empty/duplicate/excluded keys.

    Seeded with the target's exclude set; ``take`` appends a display only when its ``match_key`` is
    non-empty and unseen, so the answer or a variant can never slip in.
    """

    def __init__(self, entry: Entry):
        self._seen = exclude_keys(entry)
        self.items: list[str] = []

    def take(self, display: str) -> bool:
        """Append ``display`` if its ``match_key`` is new; return whether taken."""
        key = match_key(display)
        if not key or key in self._seen:
            return False
        self._seen.add(key)
        self.items.append(display)
        return True
