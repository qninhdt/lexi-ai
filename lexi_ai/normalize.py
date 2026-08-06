"""Normalize core — the single most critical module.

``match_key`` and ``render`` are pure functions used on BOTH the write path
(indexing generated aliases) and the read path (resolving user input). If the
two paths disagree, lookups miss forever. Every other module imports these; no
module reimplements key logic.

Canonical ``norm`` convention: placeholders are stored as brace tokens
``{sb}`` ``{sth}`` ``{one's}`` ``{oneself}`` — braces never occur in real
lemmas, so the token regex is unambiguous.

- ``match_key(s)`` — deterministic *lossy* key: lowercase, NFKC-canonicalize,
  fold placeholders to stable sentinels, collapse whitespace. Placeholder
  surface variants collapse together so a user typing ``act on behalf of
  somebody`` and the stored ``act on behalf of {sb}`` land on the same key.
  Diacritics are PRESERVED — ``résumé`` and ``resume`` are different words, and
  this key is UNIQUE on ``words``.
- ``fold_diacritics(s)`` — accent folding, separated out of ``match_key``. Use it
  where merging accents is the intent: alias generation, and comparing a
  learner's typed answer against the expected one.
- ``render(norm)`` — human display form: expand brace tokens to words.

A ``/`` is kept literal (decision #8): ``match_key`` never splits on it.
"""

import re
import unicodedata

__all__ = [
    "match_key",
    "answer_key",
    "fold_diacritics",
    "render",
    "tag_key",
    "theme_key",
    "PLACEHOLDER_RE",
]

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


def _canonicalize(s: str) -> str:
    """NFKC-normalize: canonical composition plus compatibility folding.

    ``match_key`` used to run NFKD and drop combining marks, which did two
    unrelated jobs at once — canonicalizing how a character is *encoded*, and
    merging characters that are *different letters*. Only the second was a bug.
    This keeps the first.

    Without it the key is raw code points, so ``café`` typed as U+00E9 and ``café``
    typed as ``e`` + U+0301 produce different keys while rendering identically —
    two dictionary entries for one word, which is the same class of failure as the
    collision, just inverted. macOS and several IMEs emit the decomposed form.

    The K (compatibility) folding additionally maps presentational variants onto
    their plain letters: ``ﬁle`` -> ``file``, fullwidth ``ａbc`` -> ``abc``. Those
    are encodings of the same word, not distinct headwords, so folding them is
    correct.
    """
    return unicodedata.normalize("NFKC", s)


def _strip_diacritics(s: str) -> str:
    """NFKD-decompose and drop combining marks (café -> cafe, naïve -> naive)."""
    decomposed = unicodedata.normalize("NFKD", s)
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def fold_diacritics(s: str) -> str:
    """The accent-folded surface of ``s`` (``résumé`` -> ``resume``).

    Public because it is no longer part of :func:`match_key`. Folding inside the
    identity key made distinct headwords collide under a UNIQUE constraint; the
    folded form is now registered as a ``diacritic`` alias instead, which keeps
    accent-insensitive lookup working without claiming the two words are one.

    Returns the input unchanged when it carries no diacritics, so a caller can
    test ``fold_diacritics(x) != x`` to decide whether an alias is warranted.
    """
    return _strip_diacritics(s)


def _drop_format_chars(s: str) -> str:
    """Drop Unicode format / zero-width chars (category ``Cf``): ZWSP (U+200B),
    BOM/ZWNBSP (U+FEFF), soft hyphen (U+00AD), ZWJ/ZWNJ, word-joiner, etc.

    These are INVISIBLE, so an input carrying one must key identically to the
    clean form — otherwise the write key and the read key diverge and the lookup
    misses forever (core B3). ``_CTRL_RE`` (``\\x00-\\x1f\\x7f``) does NOT cover
    this range, so the sibling keys don't strip these either; ``match_key`` must
    EXCEED sibling behavior here. The placeholder sentinels are private-use
    (category ``Co``), NOT ``Cf``, so they are preserved."""
    return "".join(c for c in s if unicodedata.category(c) != "Cf")


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

    Pipeline: lowercase -> NFKC-canonicalize -> strip control chars (incl NUL,
    Postgres-rejected) -> drop zero-width/format chars -> fold placeholders ->
    collapse whitespace. Never splits on ``/``.

    **Diacritics are preserved; encoding differences are not.** The key used to run
    NFKD and drop combining marks, which conflated two jobs. Dropping the marks
    made this key fold pairs that are different words: ``résumé``/``resume``,
    ``pâté``/``pate``, ``exposé``/``expose``, ``rosé``/``rose``. Since
    ``words.match_key`` is UNIQUE, the second of each pair could not be inserted
    at all — generating ``pâté`` after ``pate`` either failed or overwrote the
    first entry and took its senses with it. No amount of downstream care can
    recover a headword the key says does not exist.

    The canonicalization half is kept as NFKC, because ``café`` composed and
    ``café`` decomposed are one word spelled one way, and a key that separates them
    produces two indistinguishable entries.

    Accent-insensitive *lookup* is preserved without the collision: the
    generation path registers the folded spelling as a ``diacritic`` alias, and
    ``resolve_key`` searches aliases as well as headwords. So typing ``resume``
    still finds ``résumé`` — it just no longer means they are the same row. Use
    :func:`fold_diacritics` where merging accents IS the intent, such as comparing
    a learner's typed answer.

    The control-strip + format-drop make ``match_key`` EXCEED its sibling keys
    (``tag_key``/``theme_key`` only ``_CTRL_RE``): an embedded NUL crashes the
    ``words.match_key`` INSERT on Postgres (core B2), and an invisible zero-width
    char (ZWSP/BOM/soft-hyphen) keys differently from the clean form so the
    write key and read key diverge forever (core B3). Both are stripped BEFORE
    placeholder folding so the PUA sentinels (category ``Co``) survive.

    Scope note (core B1/H4): this is a lossy read+write key; changing its output
    for a given input orphans any row already keyed under the old output, so this
    fix is pre-population-only against the regenerable DB the architecture doc
    specifies. A non-regenerable DB needs a one-time backfill first.
    """
    s = s.lower()
    s = _canonicalize(s)
    s = _CTRL_RE.sub(" ", s)
    s = _drop_format_chars(s)
    s = _fold_placeholders(s)
    s = _WS_RE.sub(" ", s).strip()
    # Cap to the words.match_key / word_aliases.alias_match_key String(512) width.
    # NFKD diacritic-stripping can EXPAND length (ligatures/compat chars decompose
    # to multiple code points), so a schema-legal input (<=128) can overflow 512.
    # Postgres enforces VARCHAR(512) (INSERT fails -> word to status="error");
    # SQLite ignores the declared width and would silently store the over-length
    # key, diverging the two backends. Cap here so both behave identically.
    return s[:512]


def answer_key(s: str) -> str:
    """Comparison key for a learner's typed answer: ``match_key`` plus accent folding.

    Identity and comparison want opposite things from a diacritic.
    ``words.match_key`` must keep ``résumé`` and ``resume`` apart, because they
    are different words and the column is UNIQUE. Grading must put them together,
    because a learner typing ``cafe`` for ``café`` has recalled the word and most
    phone keyboards do not offer the accent.

    Folding is applied AFTER ``match_key`` so this key inherits the whole pipeline
    — placeholder folding, control/format stripping, NFKC — and can only ever be
    more permissive than the identity key, never differently shaped.
    """
    return _strip_diacritics(match_key(s))


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
