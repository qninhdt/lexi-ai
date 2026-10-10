"""Identity, exact-text cache keys, and validated target markup."""

import hashlib
import re
import unicodedata
from dataclasses import dataclass

from .config import MAX_LEMMA_LENGTH, MAX_TEXT_LENGTH
from .vocab import INFLECTION_CODES, SLOTS, Inflection

_SPACES = re.compile(r"\s+")
_BRACES = re.compile(r"\{[^{}]*\}")
_TAG = re.compile(r"\[([^\[\]|]+)(?:\|([a-z0-9]+))?\]", re.IGNORECASE)
_ANY_TAG = re.compile(r"[\[\]]")
_ETC_OR_PAREN = re.compile(r"(?:\b[Ee][Tt][Cc]\.(?:\s|$)|[()])")


def validate_lemma(lemma: str) -> str:
    """Check a citation without deleting lexical characters or unknown tokens."""
    if not isinstance(lemma, str) or not lemma.strip() or len(lemma) > MAX_LEMMA_LENGTH:
        raise ValueError("invalid lemma length")
    if any(ord(char) < 32 or unicodedata.category(char) == "Cf" for char in lemma):
        raise ValueError("control or invisible character in lemma")
    if "{" in lemma or "}" in lemma:
        remaining = _BRACES.sub("", lemma)
        if "{" in remaining or "}" in remaining:
            raise ValueError("malformed slot")
        if any(match.group() not in SLOTS for match in _BRACES.finditer(lemma)):
            raise ValueError("unknown slot")
    if _ETC_OR_PAREN.search(lemma):
        raise ValueError("citation contains alternative/optional notation")
    return _SPACES.sub(" ", unicodedata.normalize("NFKC", lemma)).strip()


def match_key(lemma: str) -> str:
    """Stable Unicode/case/whitespace identity; literal someone is not {sb}."""
    return validate_lemma(lemma).casefold()


def answer_key(answer: str) -> str:
    """Only harmless case/Unicode/spacing differences are ignored."""
    if not isinstance(answer, str) or not answer.strip() or len(answer) > MAX_TEXT_LENGTH:
        raise ValueError("invalid answer")
    return _SPACES.sub(" ", unicodedata.normalize("NFKC", answer)).strip().casefold()


def content_hash(content: str) -> str:
    """Hash the exact UTF-8 bytes passed to a translation provider."""
    if not isinstance(content, str) or not content or len(content) > MAX_TEXT_LENGTH:
        raise ValueError("invalid text")
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Span:
    surface: str
    inf: Inflection
    start: int
    end: int


def parse_marked_example(content: str) -> tuple[str, list[Span]]:
    """Return unwrapped text and target spans; reject partial/invalid target tags."""
    if not isinstance(content, str) or len(content) > MAX_TEXT_LENGTH:
        raise ValueError("invalid marked content")
    result: list[str] = []
    spans: list[Span] = []
    cursor = 0
    clean_length = 0
    for match in _TAG.finditer(content):
        prefix = content[cursor : match.start()]
        if _ANY_TAG.search(prefix):
            raise ValueError("invalid target markup")
        result.append(prefix)
        clean_length += len(prefix)
        surface, code = match.groups()
        try:
            inf = INFLECTION_CODES[(code or "").lower()]
        except KeyError as exc:
            raise ValueError("invalid target inflection") from exc
        if not surface.strip():
            raise ValueError("invalid target inflection")
        spans.append(Span(surface, inf, clean_length, clean_length + len(surface)))
        result.append(surface)
        clean_length += len(surface)
        cursor = match.end()
    tail = content[cursor:]
    if _ANY_TAG.search(tail):
        raise ValueError("invalid target markup")
    result.append(tail)
    return "".join(result), spans


def strip_markup(content: str) -> str:
    return parse_marked_example(content)[0]


def canonical_markup(content: str) -> str:
    """Validate before normalizing only known tag/inflection labels, never learner text."""
    parse_marked_example(content)
    return _TAG.sub(lambda m: f"[{m[1]}{'|' + m[2].lower() if m[2] else ''}]", content)


def parse_form(content: str) -> tuple[str, Inflection]:
    """Decode a compact surface/inflection pair without guessing morphology."""
    if not isinstance(content, str):
        raise ValueError("form must be a string")
    surface, separator, code = content.partition("|")
    if separator and (not code or code.lower() not in INFLECTION_CODES):
        raise ValueError("invalid form inflection")
    if any(char in surface for char in "[]|"):
        raise ValueError("invalid form surface")
    return validate_lemma(surface), INFLECTION_CODES[code.lower()]
