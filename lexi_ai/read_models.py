"""Public read models for the lazy lookup API (Phase 6).

Plain dataclasses assembled inside the DB session (never lazy-loaded after the
session closes). ``display`` is always ``render(norm)`` — never a stored column.
"""

from dataclasses import dataclass, field


@dataclass
class ReferenceView:
    source: str
    source_ref: str


@dataclass
class FormView:
    """One inflected form of a sense's headword: ``surface`` (ran) tagged with its
    ``inf`` label (past). A label may repeat across rows when a form has variants
    (dreamed / dreamt), so this is a flat list, not a dict."""

    inf: str
    surface: str


@dataclass
class SenseView:
    definition: str
    tier: str
    pos: str | None
    cefr_level: str | None
    # IPA pronunciation per sense (POS-grouped). None when unavailable.
    ipa_uk: str | None = None
    ipa_us: str | None = None
    examples: list[str] = field(default_factory=list)
    references: list[ReferenceView] = field(default_factory=list)
    # Inflection paradigm (best-effort — empty for invariant words). LLM-emitted
    # per POS, not scraped from examples, so it is complete.
    forms: list[FormView] = field(default_factory=list)
    # Learner-dictionary enrichments (best-effort — empty/None when unmarked).
    # ``grammar``/``collocations`` are lists; the others are single labels.
    guideword: str | None = None
    grammar: list[str] = field(default_factory=list)
    register: str | None = None
    connotation: str | None = None
    collocations: list[str] = field(default_factory=list)
    # ``domain`` = subject-area label (computing, medicine, law); ``usage_note`` =
    # one-line usage/confusable hint. Both free-text, best-effort.
    domain: str | None = None
    usage_note: str | None = None
    # DB id of the sense, when this view was assembled from a persisted sense
    # (None for views synthesized without a row). Lets the questions engine record
    # which sense a generated question targets, for provenance.
    sense_id: int | None = None
    # Sense-level semantic relations THIS sense emits (synonym/antonym/hypernym/
    # ...). Distinct from ``Entry.links`` (word-level): these are anchored to the
    # specific meaning that emitted them. Additive — a consumer reading only
    # ``Entry.links`` is unaffected. Empty when the sense emits none.
    relations: list["SenseRelationView"] = field(default_factory=list)


@dataclass
class SenseRelationView:
    """One sense-level relation surfaced from a :class:`SenseView` (parallel to
    :class:`LinkView` for word-level links).

    Always carries the sense->word half: ``to_word_id`` + ``to_word_display`` +
    ``rel_type`` + ``to_word_status`` (a ``pending`` target is a stub awaiting
    lazy generation). The resolved target sense is additive: ``to_sense_id`` +
    ``to_sense_gloss`` are set only once WSD reconciled it AND the read-time hash
    still matches (F5 — a stale target is surfaced as unresolved).

    ``wsd_state`` is DERIVED (Q1 — there is no ``wsd_state`` DB column): the read
    model computes ``resolved`` / ``unresolvable`` / ``pending`` from
    ``to_sense_id`` + ``resolve_attempted_at`` + the hash-verify result, exposed
    as a string for consumers to filter on.
    """

    rel_type: str
    to_word_display: str
    to_word_id: int
    to_word_status: str
    wsd_state: str
    to_sense_id: int | None = None
    to_sense_gloss: str | None = None


@dataclass
class AliasView:
    display: str
    alias_norm: str
    type: str
    dialect: str | None


@dataclass
class LinkView:
    display: str
    norm: str
    rel_type: str
    # DB id + status of the linked word, so a consumer discovering an idiom /
    # phrasal verb / related word from its host has a generatable handle without a
    # second lookup. A pending link target is a stub awaiting lazy generation.
    word_id: int
    status: str


@dataclass
class TopicView:
    """A topic tag on an entry: ``name`` short slug, ``title`` human display."""

    name: str
    title: str


@dataclass
class Entry:
    display: str
    norm: str
    entry_type: str | None
    pos: str | None
    status: str
    # DB id of the underlying word. Lets the questions engine stamp/list questions
    # by word without a second lookup. Populated in ``api._build_entry``.
    word_id: int
    senses: list[SenseView] = field(default_factory=list)
    aliases: list[AliasView] = field(default_factory=list)
    links: list[LinkView] = field(default_factory=list)
    topics: list[TopicView] = field(default_factory=list)


@dataclass
class Theme:
    """A style voice (public read view). ``key`` is the normalized dedup key and
    is intentionally exposed — callers pass it back to ``get_entry``/``generate``
    (and to ``get_theme``/``update_theme``/``delete_theme``) to address a theme
    (unlike ``tag_key``, which stays repository-internal)."""

    key: str
    name: str
    style_prompt: str
    description: str | None = None
    tone: str | None = None


@dataclass
class Asset:
    """A cached derived asset (public read view). ``text_value`` holds inline
    results (translation); ``file_path`` points at a binary clip (TTS) relative
    to the asset cache dir. ``ready`` tells a ready asset from a placeholder.

    Identity is the reference tuple ``(source_kind, source_id, kind, params)``
    (Phase 1). ``id`` is the DB id when this view was assembled from a persisted
    row (the handle passed to ``get_asset``/``delete_asset``); ``None`` for a
    placeholder synthesized without a row (e.g. an empty-text short-circuit).
    NOTE: a durable consumer (e.g. a frozen question payload) must bind to the
    reference tuple, NOT ``id`` — a purge/regenerate deletes the row."""

    kind: str
    params: str
    source_kind: str | None = None
    source_id: int | None = None
    text_value: str | None = None
    file_path: str | None = None
    meta: str | None = None
    id: int | None = None

    @property
    def ready(self) -> bool:
        """True when the asset carries usable content (inline text or a file)."""
        return self.text_value is not None or self.file_path is not None


@dataclass
class SearchResult:
    """One hit from :meth:`Lexicon.search` — a single ranked list mixes two kinds.

    ``generated`` (``lexi_word_id`` set): an entry that already exists in the
    dictionary; pass the id to :meth:`Lexicon.get_entry`. ``suggestion``
    (``cambridge_id`` set, ``lexi_word_id`` is ``None``): a reference word that
    can be generated; pass the result to :meth:`Lexicon.generate`.
    """

    display: str
    entry_type: str | None
    score: float = 0.0
    lexi_word_id: int | None = None
    cambridge_id: int | None = None
    gloss: str | None = None

    @property
    def generated(self) -> bool:
        """True if this word already exists (has a ``lexi_word_id``)."""
        return self.lexi_word_id is not None


@dataclass
class TagCount:
    """One topic in the browse index: display ``name``/``title`` + live member
    ``count`` (over ``status="done"`` words). The internal ``tag_key`` is NOT
    exposed — normalized keys stay repository-internal, like ``match_key``."""

    name: str
    title: str
    count: int


@dataclass
class SemanticHit:
    """One hit from :meth:`Lexicon.semantic_search` — a generated sense ranked by
    cosine similarity of its embedding to the query. ``score`` is in ``[-1, 1]``
    (1 = identical direction). ``sense`` is the matched sense; ``lexi_word_id``
    points at its owning word (pass to :meth:`Lexicon.get_entry` for the full entry)."""

    lexi_word_id: int
    display: str
    entry_type: str | None
    score: float
    sense: SenseView


@dataclass
class Stats:
    """A point-in-time snapshot of dictionary counts (from :meth:`Lexicon.stats`).

    Counts are read in one session but are NOT transaction-isolated against
    concurrent writes across calls — acceptable for a stats surface.
    ``themed_words`` counts distinct words with at least one themed overlay.
    """

    words_by_status: dict[str, int]  # {"done": n, "pending": m, "error": k, ...}
    senses: int
    examples: int
    tags: int
    themes: int
    themed_words: int  # words with >=1 themed overlay
    assets_by_kind: dict[str, int]  # {"translate": n, "tts": m}
    questions: int


@dataclass
class BatchResult:
    """One item's outcome in a batch call (``*_many``).

    Results are order-aligned with the inputs (``results[i]`` for ``inputs[i]``);
    ``key`` additionally echoes the input identity for convenience. Exactly one
    side is meaningful: ``ok`` True → read ``value``; ``ok`` False → read
    ``error`` (a short message). A batch never aborts on one item — a failed item
    is reported here while its siblings still complete.
    """

    key: object
    value: object | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        """True when this item succeeded (no error captured)."""
        return self.error is None
