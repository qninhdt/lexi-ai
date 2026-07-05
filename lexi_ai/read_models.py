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
class SenseView:
    definition: str
    tier: str
    pos: str | None
    cefr_level: str | None
    examples: list[str] = field(default_factory=list)
    references: list[ReferenceView] = field(default_factory=list)
    # Learner-dictionary enrichments (best-effort — empty/None when unmarked).
    # ``grammar``/``collocations`` are lists; the others are single labels.
    guideword: str | None = None
    grammar: list[str] = field(default_factory=list)
    register: str | None = None
    connotation: str | None = None
    collocations: list[str] = field(default_factory=list)
    # DB id of the sense, when this view was assembled from a persisted sense
    # (None for views synthesized without a row). Lets the questions engine record
    # which sense a generated question targets, for provenance.
    sense_id: int | None = None


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

    ``id`` is the DB id when this view was assembled from a persisted row (the
    handle passed to ``get_asset``/``delete_asset``); ``None`` for a placeholder
    synthesized without a row (e.g. an empty-text short-circuit)."""

    kind: str
    params: str
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
class Question:
    """A vocabulary question about a word (public read view).

    ``id`` is ``None`` for an ephemeral question a plugin chose not to persist;
    a persisted question carries its DB id. ``payload`` is the parsed per-format
    dict (prompt, options, correct index, rubric, ... — shape depends on
    ``format``); the DB stores it as a JSON string, the read layer parses it.
    """

    id: int | None
    word_id: int
    sense_id: int | None
    format: str
    answer_kind: str
    payload: dict


@dataclass
class Score:
    """The result of grading an answer to a :class:`Question`.

    ``kind`` records how the verdict was reached (``'rule'`` deterministic vs
    ``'llm'`` judge) for the caller's information only. ``feedback`` is populated
    by the llm-judge path and is ``None`` for rule grading.
    """

    correct: bool
    score: float  # 0.0..1.0
    kind: str  # ∈ SCORE_KINDS
    feedback: str | None = None


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
