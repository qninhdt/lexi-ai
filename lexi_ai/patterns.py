"""Fully anchored matching of explicitly licensed, bounded expression slots."""

import re
from functools import cache, lru_cache

from .config import MAX_PATTERN_LENGTH
from .text import answer_key, validate_lemma
from .vocab import SLOTS

_TOKEN = re.compile(r"\{[^{}]+\}")
_SLASH_ALTERNATIVE = re.compile(r"\b[a-z]+/[a-z]+\b", re.IGNORECASE)
_WORD = r"[^\W_]+(?:['’-][^\W_]+)*"
_BOUNDS = {
    "{sb}": (1, 4),
    "{sth}": (1, 5),
    "{one's}": (1, 2),
    "{oneself}": (1, 1),
    "{place}": (1, 4),
    "{doing}": (1, 4),
    "{do}": (1, 4),
    "{num}": (1, 1),
    "{clause}": (2, 12),
}
_PRONOUNS = {
    "{sb}": (
        r"(?:someone|somebody|[a-z]+|"
        r"(?:the|a|an|my|your|his|her|our|their)\s+[a-z]+(?:\s+[a-z]+){0,2})"
    ),
    "{one's}": r"(?:my|your|his|her|its|our|their|one's|[a-z]+['’]s)",
    "{oneself}": r"(?:myself|yourself|himself|herself|itself|ourselves|themselves|oneself)",
    "{num}": r"(?:\d+(?:[.,]\d+)?|one|two|three|four|five|six|seven|eight|nine|ten)",
}


def validate_pattern(pattern: str) -> str:
    if not isinstance(pattern, str) or len(pattern) > MAX_PATTERN_LENGTH:
        raise ValueError("invalid pattern length")
    pattern = validate_lemma(pattern)
    if any(match.group() not in SLOTS for match in _TOKEN.finditer(pattern)):
        raise ValueError("unknown pattern slot")
    if "/" in pattern and _SLASH_ALTERNATIVE.search(pattern):
        raise ValueError("expand lexical alternatives into separate patterns")
    return pattern


def _slot_regex(slot: str) -> str:
    if slot in _PRONOUNS:
        return _PRONOUNS[slot]
    minimum, maximum = _BOUNDS[slot]
    return rf"(?:{_WORD}(?:\s+{_WORD}){{{minimum - 1},{maximum - 1}}})"


def matches_pattern(
    pattern: str,
    surface: str,
    *,
    forms: dict[str, list[str]] | None = None,
) -> bool:
    """Match the entire expression; verified forms can replace only its initial head."""
    pattern = validate_pattern(pattern)
    if not isinstance(surface, str) or len(surface) > MAX_PATTERN_LENGTH:
        return False
    if not surface.strip():
        return False
    candidate = answer_key(surface)
    licensed = tuple(sorted((head, tuple(values)) for head, values in (forms or {}).items()))
    parts = _compiled_parts(pattern, licensed)

    @cache
    def slot_ends(slot, start):
        # Each individual slot is unambiguous internally. Memoized transitions
        # prevent adjacent slots from exploring exponentially many partitions.
        regex = _compiled_slot(slot)
        return tuple(
            end
            for end in range(start + 1, len(candidate) + 1)
            if regex.fullmatch(candidate, start, end) is not None
        )

    positions = {0}
    for slot, literals in parts:
        following = set()
        for start in positions:
            if slot is not None:
                following.update(slot_ends(slot, start))
            else:
                for literal in literals:
                    if match := literal.match(candidate, start):
                        following.add(match.end())
        if not following:
            return False
        positions = following
    return len(candidate) in positions


def _literal(text: str, *, first: bool, forms) -> list[str]:
    text = text.casefold()
    if first and forms and text.strip():
        head, separator, tail = text.partition(" ")
        licensed = [head]
        for replacement in forms.get(head, []):
            replacement = answer_key(replacement)
            if " " not in replacement:
                licensed.append(replacement)
        suffix = re.escape(separator + tail).replace(r"\ ", r"\s+")
        return [re.escape(item) + suffix for item in dict.fromkeys(licensed)]
    return [re.escape(text).replace(r"\ ", r"\s+")]


@lru_cache(maxsize=len(SLOTS))
def _compiled_slot(slot):
    return re.compile(_slot_regex(slot), flags=re.IGNORECASE)


@lru_cache(maxsize=1024)
def _compiled_parts(pattern, licensed):
    forms = dict(licensed)
    parts = []
    cursor = 0
    for match in _TOKEN.finditer(pattern):
        literals = _literal(pattern[cursor : match.start()], first=cursor == 0, forms=forms)
        parts.append((None, tuple(re.compile(part, re.I) for part in literals)))
        parts.append((match.group(), ()))
        cursor = match.end()
    literals = _literal(pattern[cursor:], first=cursor == 0, forms=forms)
    parts.append((None, tuple(re.compile(part, re.I) for part in literals)))
    return tuple(parts)


def pattern_head_key(pattern: str) -> str | None:
    """Index only a complete fixed first token; all other patterns stay in fallback."""
    pattern = validate_pattern(pattern)
    slot = _TOKEN.search(pattern)
    literal = pattern[: slot.start()] if slot else pattern
    if not literal.strip() or (slot is not None and " " not in literal):
        return None
    return surface_head_key(literal)


def surface_head_key(surface: str) -> str:
    # Python regex IGNORECASE treats i and dotless ı alike, even though casefold
    # alone does not. The narrowing key must remain a superset of regex matches.
    return answer_key(surface).partition(" ")[0].replace("ı", "i")
