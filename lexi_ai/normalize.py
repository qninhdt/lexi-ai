"""Normalize core — the single most critical module.

``match_key`` and ``render`` are pure functions used on BOTH the write path
(indexing generated aliases) and the read path (resolving user input). If the
two paths disagree, lookups miss forever. Every other module imports these; no
module reimplements key logic.

Canonical ``norm`` convention: placeholders are stored as brace tokens
``{sb}`` ``{sth}`` ``{one's}`` ``{oneself}`` — braces never occur in real
lemmas, so the token regex is unambiguous.

- ``match_key(s)`` — deterministic *lossy* key: lowercase, strip diacritics,
  fold placeholders to stable sentinels, collapse whitespace. Placeholder
  surface variants collapse together so a user typing ``act on behalf of
  somebody`` and the stored ``act on behalf of {sb}`` land on the same key.
- ``render(norm)`` — human display form: expand brace tokens to words.

A ``/`` is kept literal (decision #8): ``match_key`` never splits on it.
"""

import re
import unicodedata

__all__ = ["match_key", "render", "tag_key", "theme_key", "PLACEHOLDER_RE"]

# --- placeholder canonical map (single source of truth) -------------------
#
# Match sentinels use Private Use Area code points (U+E000+). Like NUL they
# never occur in real dictionary text, but — unlike NUL — they are valid in
# both SQLite and PostgreSQL text columns (Postgres rejects 0x00 in text), so
# a placeholder word's match_key persists identically on both backends.

_SB = ""
_STH = ""
_POS = ""
_SELF = ""

# Brace token -> display word. Public/read-time expansion.
_RENDER_MAP = {
    "{sb}": "somebody",
    "{sth}": "something",
    "{one's}": "your",
    "{oneself}": "yourself",
}

# Every fold-able surface form (brace token, natural word, Cambridge citation
# form) -> match sentinel.
_FOLD_MAP = {
    "{sb}": _SB,
    "{sth}": _STH,
    "{one's}": _POS,
    "{oneself}": _SELF,
    "somebody": _SB,
    "someone": _SB,
    "sb": _SB,
    "something": _STH,
    "sth": _STH,
    "oneself": _SELF,
    "yourself": _SELF,
    "one's": _POS,
    "your": _POS,
}

# Public: matches ONLY brace tokens. Safe because braces never occur in lemmas.
PLACEHOLDER_RE = re.compile(r"\{[^}]+\}")

# Fold pattern for match_key: known brace tokens, then any (unknown) brace
# token, then natural surface forms anchored by word boundaries so substrings
# like the "one's" inside "someone's"/"stone's" are never folded. Longer
# alternatives are listed first so the greediest form wins.
_FOLD_RE = re.compile(
    r"\{sb\}|\{sth\}|\{one's\}|\{oneself\}"
    r"|\{[^}]+\}"
    r"|\b(?:something|somebody|yourself|someone|oneself|one's|your|sth|sb)\b"
)

_WS_RE = re.compile(r"\s+")

# Control chars (incl NUL) that must never reach a Postgres text column — it
# rejects 0x00. Unlike _WS_RE (\s+), this matches \x00..\x1f and \x7f. Both
# tag_key (below) and the repository's tag sanitizer route through this.
_CTRL_RE = re.compile(r"[\x00-\x1f\x7f]")


def _strip_diacritics(s: str) -> str:
    """NFKD-decompose and drop combining marks (café -> cafe, naïve -> naive)."""
    decomposed = unicodedata.normalize("NFKD", s)
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def _fold_placeholders(s: str) -> str:
    def _repl(m: re.Match) -> str:
        tok = m.group(0)
        mapped = _FOLD_MAP.get(tok)
        if mapped is not None:
            return mapped
        # Unknown brace token: strip braces to its literal content so the key
        # matches render()'s degraded form. (Known tokens handled above.)
        return tok[1:-1]

    return _FOLD_RE.sub(_repl, s)


def match_key(s: str) -> str:
    """Deterministic lossy lookup key. Same input surface variants -> same key.

    Pipeline: lowercase -> strip diacritics -> fold placeholders -> collapse
    whitespace. Never splits on ``/``.
    """
    s = s.lower()
    s = _strip_diacritics(s)
    s = _fold_placeholders(s)
    s = _WS_RE.sub(" ", s).strip()
    return s


def render(norm: str) -> str:
    """Render the human display form from a canonical ``norm`` string.

    Expands known brace tokens to words; an unknown ``{token}`` degrades to its
    literal content (braces stripped) rather than being left in the output.
    """

    def _repl(m: re.Match) -> str:
        tok = m.group(0)
        return _RENDER_MAP.get(tok, tok[1:-1])

    return PLACEHOLDER_RE.sub(_repl, norm)


# --- tag_key (topic dedup) -------------------------------------------------
#
# tag_key is to topic tags what match_key is to words: a deterministic, lossy,
# repository-only dedup key. Same pipeline as match_key (lowercase, strip
# diacritics, control-char safe, collapse whitespace) PLUS a conservative
# singularize so "cars"/"car" collapse. It is deliberately UNDER-stemming: a
# missed singularization is a recoverable near-dup; a wrong merge is data loss.

# Irregular plurals worth handling (kept tiny). Only the LAST token is checked.
_IRREGULAR_PLURALS = {
    "people": "person",
    "children": "child",
    "men": "man",
    "women": "woman",
    "feet": "foot",
    "teeth": "tooth",
}

# Endings that look plural but are NOT — never strip these (guard against
# over-stemming the broad subject-area topics the tag rubric asks for):
#   -ss  business, address        -ics physics, politics, economics, ethics
#   -sis analysis, crisis, thesis -us  status, virus, bias(-as)  -as atlas
# Whole words that end in -s but are singular / invariant.
_SINGULAR_S_SUFFIXES = ("ss", "ics", "sis", "us", "as", "is", "ous")
_INVARIANT_S_WORDS = {"series", "species", "news"}


def _singularize_token(token: str) -> str:
    """Singularize ONE word, conservatively (under-stem on doubt)."""
    if token in _IRREGULAR_PLURALS:
        return _IRREGULAR_PLURALS[token]
    if token in _INVARIANT_S_WORDS:
        return token
    if not token.endswith("s") or len(token) <= 3:
        return token
    if token.endswith(_SINGULAR_S_SUFFIXES):
        return token  # not a regular plural — leave it
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"  # bodies -> body
    if token.endswith(("ses", "xes", "zes", "ches", "shes")):
        return token[:-2]  # boxes -> box, dishes -> dish
    return token[:-1]  # regular: cars -> car


def _singularize(phrase: str) -> str:
    """Singularize only the LAST token of a topic phrase.

    Topics are noun phrases: "social media" keeps "media"; "domestic animals"
    -> "domestic animal". Splitting on space and touching only the final token
    avoids mangling non-final words.
    """
    if " " not in phrase:
        return _singularize_token(phrase)
    head, _, last = phrase.rpartition(" ")
    return f"{head} {_singularize_token(last)}"


def tag_key(s: str) -> str:
    """Deterministic lossy dedup key for a topic tag.

    Pipeline: lowercase -> strip diacritics -> strip control chars (incl NUL,
    which _WS_RE does not match and Postgres text rejects) -> collapse
    whitespace -> strip -> singularize (last token). Returns "" when nothing
    survives; callers skip empty-key tags (best-effort, never persist one).
    """
    s = s.lower()
    s = _strip_diacritics(s)
    s = _CTRL_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    if not s:
        return ""
    return _singularize(s)


# --- theme_key (theme dedup) ----------------------------------------------
#
# theme_key is to themes what tag_key is to topic tags: a deterministic, lossy,
# repository dedup key. Shares tag_key's pipeline (lowercase, strip diacritics,
# control-char safe, collapse whitespace) but deliberately does NOT singularize:
# theme names are proper voices ("Harry Potter", "The Witches"), not noun
# phrases, so folding "witches" -> "witch" would corrupt a name. The ~5 shared
# lines are duplicated rather than factored so the two keys can evolve freely.


def theme_key(s: str) -> str:
    """Deterministic lossy dedup key for a theme name.

    Pipeline: lowercase -> strip diacritics -> strip control chars (incl NUL,
    Postgres-rejected) -> collapse whitespace -> strip. NO singularization
    (unlike ``tag_key``): a theme name is a proper voice, not a noun phrase.
    Returns "" when nothing survives; the create API rejects an empty key.
    """
    s = s.lower()
    s = _strip_diacritics(s)
    s = _CTRL_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s
