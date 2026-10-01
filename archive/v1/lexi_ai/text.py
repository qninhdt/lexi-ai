"""Pure rules over strings: keys, rendering, markup, and content fingerprints.

Nothing here touches a session, a table, the filesystem, or settings — every
function is a pure transformation, which is what lets the write path and the
read path agree by construction rather than by convention. ``match_key`` and
friends are used on BOTH paths (indexing generated aliases, resolving user
input); if the two disagree, lookups miss forever.
"""

import hashlib
import re
import unicodedata
from dataclasses import dataclass

from lexi_ai.vocab import (
    ASSET_KINDS,
    DEFAULT_TTS_FORMAT,
    DEFAULT_TTS_VOICE,
    TRANSLATION_LANGUAGES,
    TTS_FORMATS,
    TTS_VOICES,
)

__all__ = [
    # lookup keys and display
    "match_key",
    "answer_key",
    "fold_diacritics",
    "render",
    "tag_key",
    "theme_key",
    "strip_control_chars",
    "PLACEHOLDER_RE",
    # markup
    "Span",
    "parse_marked_example",
    "strip_markup",
    # content fingerprints
    "sense_content_hash",
    # asset identity
    "content_hash",
    "normalize_asset_params",
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


def strip_control_chars(s: str) -> str:
    """Replace control characters with spaces (the public form of ``_CTRL_RE``)."""
    return _CTRL_RE.sub(" ", s)


def _canonicalize(s: str) -> str:
    """NFKC-normalize: canonical composition plus compatibility folding.

    Canonicalizing how a character is *encoded* is not the same job as merging
    characters that are *different letters*, and this does the first only:
    ``café`` typed as U+00E9 and ``café`` typed as ``e`` + U+0301 are one word
    spelled one way (macOS and several IMEs emit the decomposed form), and
    without this they key differently while rendering identically — two
    dictionary entries for one word. The K folding additionally maps
    presentational variants onto their plain letters (``ﬁle`` -> ``file``,
    fullwidth ``ａbc`` -> ``abc``); those are encodings of the same word, not
    distinct headwords.
    """
    return unicodedata.normalize("NFKC", s)


def _strip_diacritics(s: str) -> str:
    """NFKD-decompose and drop combining marks (café -> cafe, naïve -> naive)."""
    decomposed = unicodedata.normalize("NFKD", s)
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def fold_diacritics(s: str) -> str:
    """The accent-folded surface of ``s`` (``résumé`` -> ``resume``).

    Separate from :func:`match_key` on purpose: folding accents inside the
    identity key makes distinct headwords collide under a UNIQUE constraint. The
    folded form belongs on a ``diacritic`` alias instead, which keeps
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
    misses forever. ``_CTRL_RE`` (``\\x00-\\x1f\\x7f``) does NOT cover
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
    """Deterministic lossy lookup key. Surface variants of one word share a key.

    Pipeline: lowercase -> NFKC-canonicalize -> strip control chars (incl NUL,
    which Postgres rejects) -> drop zero-width/format chars -> fold placeholders
    -> collapse whitespace. Never splits on ``/``.

    **Diacritics are preserved; encoding differences are not.** Dropping the
    combining marks would fold pairs that are different words — ``résumé`` and
    ``resume``, ``pâté`` and ``pate`` — and since ``words.match_key`` is UNIQUE
    the second of each pair could not be inserted at all: generating ``pâté``
    after ``pate`` either fails or overwrites the first entry and takes its senses
    with it. Canonicalization stays NFKC, because composed and decomposed ``café``
    are one word spelled one way and separating them yields two indistinguishable
    entries. Accent-insensitive *lookup* survives without the collision: the
    generation path registers the folded spelling as a ``diacritic`` alias, so
    typing ``resume`` still finds ``résumé`` without claiming they are one row.
    Use :func:`fold_diacritics` where merging accents IS the intent.

    This key EXCEEDS its siblings (``tag_key``/``theme_key`` apply only
    ``_CTRL_RE``): an embedded NUL crashes the ``words.match_key`` INSERT on
    Postgres, and an invisible zero-width char keys differently from the clean
    form so the write key and read key diverge forever. Both are stripped BEFORE
    placeholder folding, so the PUA sentinels (category ``Co``) survive.

    Scope: lossy read+write key. Changing its output for a given input orphans
    rows already keyed under the old output, so a change is safe only against a
    regenerable DB, or with a one-time backfill first.
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

    Identity and comparison want opposite things from a diacritic. The UNIQUE
    identity key must keep ``résumé`` and ``resume`` apart; grading must put them
    together, because a learner typing ``cafe`` for ``café`` has recalled the word
    and most phone keyboards offer no accent. Folding runs AFTER ``match_key``, so
    this key inherits the whole pipeline and can only be more permissive than the
    identity key, never differently shaped.
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
# The topic-tag dedup key: same shape as match_key but repository-only, PLUS a
# conservative singularize so "cars"/"car" collapse. It is deliberately
# UNDER-stemming: a missed singularization is a recoverable near-dup; a wrong
# merge is data loss.

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

    Topics are noun phrases, so "social media" keeps "media" while "domestic
    animals" -> "domestic animal"; touching only the final token avoids mangling
    non-final words.
    """
    if " " not in phrase:
        return _singularize_token(phrase)
    head, _, last = phrase.rpartition(" ")
    return f"{head} {_singularize_token(last)}"


def tag_key(s: str) -> str:
    """Deterministic lossy dedup key for a topic tag.

    Returns "" when nothing survives; callers skip empty-key tags (best-effort,
    never persist one).
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
# Shares tag_key's pipeline but deliberately does NOT singularize: theme names
# are proper voices ("Harry Potter", "The Witches"), not noun phrases, so folding
# "witches" -> "witch" would corrupt a name. The ~5 shared lines are duplicated
# rather than factored so the two keys can evolve freely.


def theme_key(s: str) -> str:
    """Deterministic lossy dedup key for a theme name.

    NO singularization (unlike ``tag_key``): a theme name is a proper voice, not
    a noun phrase. Returns "" when nothing survives; the create API rejects an
    empty key.
    """
    s = s.lower()
    s = _strip_diacritics(s)
    s = _CTRL_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s





# One target tag: <t inf="past">glistened</t>. ``inf`` is bounded to word chars
# (the closed INFLECTION_LABELS vocab); the inner surface is captured lazily so a
# sentence with several tags does not span across them.
_TAG_RE = re.compile(r'<t\s+inf="([^"]*)"\s*>(.*?)</t>', re.IGNORECASE | re.DOTALL)


@dataclass(frozen=True)
class Span:
    """A marked target occurrence, positioned in the RENDERED (tag-free) text.

    ``surface`` is the inflected form as it appears in the sentence; ``inf`` is
    its inflection label (∈ INFLECTION_LABELS, though this module does not
    validate the vocab — it only reports what the tag carried). ``start``/``end``
    index into the clean text returned alongside, so a caller can blank exactly
    ``clean[start:end]``.
    """
    surface: str
    inf: str
    start: int
    end: int


def parse_marked_example(text: str) -> tuple[str, list[Span]]:
    """Split a marked example into (clean_text, spans).

    ``clean_text`` is the sentence with every ``<t>`` tag unwrapped to its inner
    surface (display form). ``spans`` locate each target within ``clean_text``.
    Un-tagged input returns unchanged with no spans.
    """
    spans: list[Span] = []
    out: list[str] = []
    pos = 0  # cursor into the ORIGINAL text
    clean_len = 0  # running length of the clean text built so far
    for m in _TAG_RE.finditer(text):
        out.append(text[pos : m.start()])
        clean_len += m.start() - pos
        inf = m.group(1)
        surface = m.group(2)
        out.append(surface)
        spans.append(Span(surface=surface, inf=inf, start=clean_len, end=clean_len + len(surface)))
        clean_len += len(surface)
        pos = m.end()
    out.append(text[pos:])
    return "".join(out), spans


def strip_markup(text: str) -> str:
    """The example's display form: tags unwrapped to their inner surface."""
    return parse_marked_example(text)[0]





def sense_content_hash(definition: str) -> str:
    """Stable content fingerprint of a target sense.

    Stamped on ``sense_relation.target_hash`` at resolve time and re-checked on
    read: if the target sense's definition later changes (regenerate), the stored
    hash no longer matches and the edge is treated as unresolved rather than
    silently pointing at a mutated meaning.
    """
    return hashlib.sha256(definition.encode("utf-8")).hexdigest()






def content_hash(text: str) -> str:
    """sha256 hex of the NORMALIZED source text (VERIFY function, not identity).

    Normalization (strip control chars, collapse whitespace, strip) runs before
    hashing so trailing/interior-whitespace variants of the same text collapse to
    one hash. The SAME normalization on every call — this is the verify contract.
    """
    s = strip_control_chars(text)
    s = " ".join(s.split()).strip()
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def normalize_asset_params(kind: str, **kw: str | None) -> str:
    """Stable param token for an asset identity, normalized on read AND write.

    ``translate`` → a normalized lang code (``lang``); ``tts`` → ``voice|fmt``.
    Unknown kind → ``ValueError``.

    Every free param is validated against a closed vocab at this one choke point
    (like ``lang`` against ``TRANSLATION_LANGUAGES``): ``voice``/``fmt`` against
    ``TTS_VOICES``/``TTS_FORMATS``. This closes the filename-collision bug where
    two distinct DB rows (``en-US`` vs ``en_US``) squashed to the SAME on-disk
    path and served each other's bytes. A ``None`` voice/fmt resolves to
    ``DEFAULT_TTS_VOICE``/``DEFAULT_TTS_FORMAT`` BEFORE validation, so a default
    TTS call never hard-rejects on the happy path.

    Those defaults come from :mod:`lexi_ai.vocab`, never from runtime settings: an
    asset's identity must not depend on the configuration of whichever process
    computed it, or two processes with different ``tts_voice`` would derive two
    paths for one logical asset and a cache written by one would miss for the
    other. ``AssetService`` resolves its own configured voice/fmt and passes them
    explicitly, so the configurable path is unchanged.
    """
    if kind not in ASSET_KINDS:
        raise ValueError(f"unknown asset kind: {kind!r}")
    if kind == "translate":
        lang = _norm_token(kw.get("lang"))
        if lang not in TRANSLATION_LANGUAGES:
            raise ValueError(f"invalid/unsupported language code: {lang!r}")
        return lang
    # tts — resolve None to the vocabulary default, then validate both params.
    voice = _norm_token(kw.get("voice") if kw.get("voice") is not None else DEFAULT_TTS_VOICE)
    fmt = _norm_token(kw.get("fmt") if kw.get("fmt") is not None else DEFAULT_TTS_FORMAT)
    if voice not in TTS_VOICES:
        raise ValueError(f"invalid/unsupported TTS voice: {voice!r}")
    if fmt not in TTS_FORMATS:
        raise ValueError(f"invalid/unsupported TTS format: {fmt!r}")
    return f"{voice}|{fmt}"


def _norm_token(value: str | None) -> str:
    return (value or "").strip().lower()
