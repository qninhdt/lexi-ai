"""SQLAlchemy 2.0 async ORM models (Phase 2).

Portable types only (``Text``/``String``/``Integer``/``Boolean``) so the schema
runs identically on SQLite and Postgres — no JSONB, no ARRAY, no native ENUM
(controlled vocabularies live in :mod:`lexi_ai.constants` and are validated at
the application layer, in the repository).

Two-DB topology (decision #14): these tables are the *generated* dictionary,
separate from the read-only Cambridge source. ``words.match_key`` is UNIQUE —
the durable dedup key and the concurrency safety net. Keys are computed only by
the repository (Phase 5) via :mod:`lexi_ai.normalize`; models never compute
them.
"""

from datetime import datetime, timezone

from sqlalchemy import (
    DateTime,
    ForeignKey,
    Integer,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
)


class Base(DeclarativeBase):
    pass


def _utcnow() -> datetime:
    # Naive UTC to match the naive DateTime columns (and server_default now()),
    # so aware/naive values never mix on Postgres TIMESTAMP WITHOUT TIME ZONE.
    return datetime.now(timezone.utc).replace(tzinfo=None)


class Word(Base):
    __tablename__ = "words"

    id: Mapped[int] = mapped_column(primary_key=True)
    norm: Mapped[str] = mapped_column(Text, nullable=False)
    # Lossy, CODE-computed (Phase 1). UNIQUE = durable dedup + concurrency guard.
    match_key: Mapped[str] = mapped_column(String(512), nullable=False, unique=True, index=True)
    entry_type: Mapped[str | None] = mapped_column(String(32))
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="pending")
    pos: Mapped[str | None] = mapped_column(String(32))
    # Provenance if this entry matched a Cambridge row (no cross-DB FK).
    cambridge_word_id: Mapped[int | None] = mapped_column(Integer)
    error_msg: Mapped[str | None] = mapped_column(Text)

    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=_utcnow, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=_utcnow, onupdate=_utcnow, server_default=func.now()
    )

    aliases: Mapped[list["WordAlias"]] = relationship(
        back_populates="word",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    senses: Mapped[list["Sense"]] = relationship(
        back_populates="word",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    # Outgoing links (this word -> others).
    links_out: Mapped[list["EntryLink"]] = relationship(
        back_populates="from_word",
        foreign_keys="EntryLink.from_word_id",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    tags: Mapped[list["WordTag"]] = relationship(
        back_populates="word",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    questions: Mapped[list["Question"]] = relationship(
        back_populates="word",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class WordAlias(Base):
    __tablename__ = "word_aliases"
    __table_args__ = (UniqueConstraint("word_id", "alias_match_key", name="uq_alias_word_key"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    word_id: Mapped[int] = mapped_column(ForeignKey("words.id", ondelete="CASCADE"), nullable=False)
    alias_norm: Mapped[str] = mapped_column(Text, nullable=False)
    alias_match_key: Mapped[str] = mapped_column(String(512), nullable=False, index=True)
    type: Mapped[str] = mapped_column(String(32), nullable=False)
    dialect: Mapped[str | None] = mapped_column(String(8))

    word: Mapped["Word"] = relationship(back_populates="aliases")


class EntryLink(Base):
    __tablename__ = "entry_links"
    __table_args__ = (
        UniqueConstraint("from_word_id", "to_word_id", "rel_type", name="uq_link_triple"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    from_word_id: Mapped[int] = mapped_column(
        ForeignKey("words.id", ondelete="CASCADE"), nullable=False
    )
    # Always a real id (stub-row pattern, decision #11) — never a dangling string.
    to_word_id: Mapped[int] = mapped_column(
        ForeignKey("words.id", ondelete="CASCADE"), nullable=False
    )
    rel_type: Mapped[str] = mapped_column(String(32), nullable=False)

    from_word: Mapped["Word"] = relationship(
        back_populates="links_out", foreign_keys=[from_word_id]
    )
    to_word: Mapped["Word"] = relationship(foreign_keys=[to_word_id])


class Sense(Base):
    __tablename__ = "senses"

    id: Mapped[int] = mapped_column(primary_key=True)
    word_id: Mapped[int] = mapped_column(ForeignKey("words.id", ondelete="CASCADE"), nullable=False)
    definition: Mapped[str] = mapped_column(Text, nullable=False)
    tier: Mapped[str] = mapped_column(String(16), nullable=False)
    sense_order: Mapped[int] = mapped_column(Integer, default=0)
    pos: Mapped[str | None] = mapped_column(String(32))
    # Cambridge-first, LLM fallback (decision #13).
    cefr_level: Mapped[str | None] = mapped_column(String(8))

    # Learner-dictionary enrichments (best-effort, null when unmarked). ``grammar``
    # holds a comma-joined set of schema-validated GRAMMAR_LABELS tokens (never
    # queried alone), split back to a list on read; ``register``/``connotation``
    # are single closed-vocab enum tokens; ``guideword`` is a short free label.
    guideword: Mapped[str | None] = mapped_column(String(64))
    grammar: Mapped[str | None] = mapped_column(String(128))
    register: Mapped[str | None] = mapped_column(String(32))
    connotation: Mapped[str | None] = mapped_column(String(16))

    # Semantic-search vector: float32 little-endian BLOB (portable — SQLite BLOB /
    # Postgres BYTEA, no pgvector). Best-effort: null until an embedder runs.
    # model + dim are stored per row so switching embedding models is safe
    # (search filters to the current model; backfill re-embeds the rest).
    embedding: Mapped[bytes | None] = mapped_column(LargeBinary)
    embedding_model: Mapped[str | None] = mapped_column(String(128))
    embedding_dim: Mapped[int | None] = mapped_column(Integer)

    word: Mapped["Word"] = relationship(back_populates="senses")
    references: Mapped[list["SenseReference"]] = relationship(
        back_populates="sense",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    examples: Mapped[list["Example"]] = relationship(
        back_populates="sense",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    collocations: Mapped[list["Collocation"]] = relationship(
        back_populates="sense",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class SenseReference(Base):
    """N-N provenance from a sense to a source sense/synset (may be empty)."""

    __tablename__ = "sense_reference"

    id: Mapped[int] = mapped_column(primary_key=True)
    sense_id: Mapped[int] = mapped_column(
        ForeignKey("senses.id", ondelete="CASCADE"), nullable=False
    )
    source: Mapped[str] = mapped_column(String(16), nullable=False)
    # cambridge sense id | wordnet synset key
    source_ref: Mapped[str] = mapped_column(String(255), nullable=False)

    sense: Mapped["Sense"] = relationship(back_populates="references")


class Example(Base):
    __tablename__ = "examples"

    id: Mapped[int] = mapped_column(primary_key=True)
    sense_id: Mapped[int] = mapped_column(
        ForeignKey("senses.id", ondelete="CASCADE"), nullable=False
    )
    text: Mapped[str] = mapped_column(Text, nullable=False)
    example_order: Mapped[int] = mapped_column(Integer, default=0)

    sense: Mapped["Sense"] = relationship(back_populates="examples")


class Collocation(Base):
    """A high-frequency partner phrase illustrating a sense in use (make a
    decision, heavy rain). Open-ended free text of arbitrary count, so it mirrors
    ``examples`` as a child table rather than a column."""

    __tablename__ = "collocations"

    id: Mapped[int] = mapped_column(primary_key=True)
    sense_id: Mapped[int] = mapped_column(
        ForeignKey("senses.id", ondelete="CASCADE"), nullable=False
    )
    text: Mapped[str] = mapped_column(Text, nullable=False)
    collocation_order: Mapped[int] = mapped_column(Integer, default=0)

    sense: Mapped["Sense"] = relationship(back_populates="collocations")


class Tag(Base):
    """An open-vocabulary topic tag (LLM-authored). ``tag_key`` is the lossy
    dedup key (repository-computed via ``normalize.tag_key``, like ``match_key``);
    ``name`` is the short display slug and ``title`` the human display, both set
    once when the row is first created (first-seen wins)."""

    __tablename__ = "tags"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    # Lossy, CODE-computed. UNIQUE = durable dedup + concurrency guard (like words).
    tag_key: Mapped[str] = mapped_column(String(255), nullable=False, unique=True, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=_utcnow, server_default=func.now()
    )

    words: Mapped[list["WordTag"]] = relationship(
        back_populates="tag",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class WordTag(Base):
    """Word <-> Tag join. UNIQUE(word_id, tag_id) so a word links a tag once."""

    __tablename__ = "word_tags"
    __table_args__ = (UniqueConstraint("word_id", "tag_id", name="uq_word_tag"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    word_id: Mapped[int] = mapped_column(ForeignKey("words.id", ondelete="CASCADE"), nullable=False)
    tag_id: Mapped[int] = mapped_column(ForeignKey("tags.id", ondelete="CASCADE"), nullable=False)

    word: Mapped["Word"] = relationship(back_populates="tags")
    tag: Mapped["Tag"] = relationship(back_populates="words")


class Theme(Base):
    """A user-authored style voice ("Harry Potter", "humorous"). ``theme_key`` is
    the lossy dedup key (repository-computed via ``normalize.theme_key``, like
    ``tag_key`` but WITHOUT singularization — a name is a proper voice, not a noun
    phrase). ``name``/``style_prompt`` are set once on create (first-seen wins).

    Themes are addressed BY KEY at the API (``get``/``generate_theme`` take a
    ``theme_key`` string), so — unlike ``tag_key`` — the key is intentionally
    exposed in the read model.
    """

    __tablename__ = "themes"

    id: Mapped[int] = mapped_column(primary_key=True)
    # Lossy, CODE-computed. UNIQUE = durable dedup + concurrency guard (like tags).
    theme_key: Mapped[str] = mapped_column(String(255), nullable=False, unique=True, index=True)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    style_prompt: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=_utcnow, server_default=func.now()
    )

    themed_senses: Mapped[list["ThemedSense"]] = relationship(
        back_populates="theme",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class ThemedSense(Base):
    """A neutral sense restyled in a theme's voice. One row per (sense, theme).
    Deleting either the neutral sense or the theme cascades this away."""

    __tablename__ = "themed_senses"
    __table_args__ = (UniqueConstraint("sense_id", "theme_id", name="uq_themed_sense"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    sense_id: Mapped[int] = mapped_column(
        ForeignKey("senses.id", ondelete="CASCADE"), nullable=False
    )
    theme_id: Mapped[int] = mapped_column(
        ForeignKey("themes.id", ondelete="CASCADE"), nullable=False
    )
    definition: Mapped[str] = mapped_column(Text, nullable=False)

    theme: Mapped["Theme"] = relationship(back_populates="themed_senses")
    examples: Mapped[list["ThemedExample"]] = relationship(
        back_populates="themed_sense",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class ThemedExample(Base):
    """A fresh in-voice example authored for a themed sense (mirrors ``Example``)."""

    __tablename__ = "themed_examples"

    id: Mapped[int] = mapped_column(primary_key=True)
    themed_sense_id: Mapped[int] = mapped_column(
        ForeignKey("themed_senses.id", ondelete="CASCADE"), nullable=False
    )
    text: Mapped[str] = mapped_column(Text, nullable=False)
    example_order: Mapped[int] = mapped_column(Integer, default=0)

    themed_sense: Mapped["ThemedSense"] = relationship(back_populates="examples")


class Asset(Base):
    """A content-addressed derived asset (translation text, TTS clip).

    Identity is ``(content_hash, kind, params)`` — the hash of the EXACT source
    text plus a normalized param token. No FK to senses/words: the source is
    identified by its CONTENT, not its location, so themed vs neutral text hash
    differently (distinct rows) and identical text dedups for free. ``text_value``
    holds inline results (translation); ``file_path`` points at a binary clip
    (TTS) relative to ``LEXI_ASSET_CACHE_DIR``.
    """

    __tablename__ = "assets"
    __table_args__ = (UniqueConstraint("content_hash", "kind", "params", name="uq_asset_identity"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    kind: Mapped[str] = mapped_column(String(16), nullable=False)  # ∈ ASSET_KINDS
    params: Mapped[str] = mapped_column(String(64), nullable=False)
    text_value: Mapped[str | None] = mapped_column(Text)
    file_path: Mapped[str | None] = mapped_column(Text)
    meta: Mapped[str | None] = mapped_column(Text)  # optional app-JSON
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=_utcnow, server_default=func.now()
    )


class Question(Base):
    """A generated vocabulary question about a word (optionally a specific sense).

    The polymorphic per-format content lives in ``payload`` as an app-serialized
    JSON string (portable ``Text``, never native JSONB), so a new format needs no
    new table — only a new plugin. Rows exist only for questions a plugin chose to
    persist (it calls the question store itself); ephemeral questions never reach
    here. No UNIQUE key: questions are content, not identity — the app decides dup
    tolerance (contrast ``words.match_key``).
    """

    __tablename__ = "questions"

    id: Mapped[int] = mapped_column(primary_key=True)
    word_id: Mapped[int] = mapped_column(
        ForeignKey("words.id", ondelete="CASCADE"), nullable=False, index=True
    )
    # Nullable: a whole-word question targets no single sense.
    sense_id: Mapped[int | None] = mapped_column(ForeignKey("senses.id", ondelete="CASCADE"))
    format: Mapped[str] = mapped_column(String(32), nullable=False)  # ∈ QUESTION_FORMATS
    answer_kind: Mapped[str] = mapped_column(String(16), nullable=False)  # ∈ ANSWER_KINDS
    payload: Mapped[str] = mapped_column(Text, nullable=False)  # app-level JSON string
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=_utcnow, server_default=func.now()
    )

    word: Mapped["Word"] = relationship(back_populates="questions")
