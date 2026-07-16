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
    # Monotonic service-generation fence.  A worker may publish a replacement
    # graph only while it still owns this value; ordinary library calls do not
    # need to provide a fence.
    generation_epoch: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
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
    # Outgoing word-level links (this word -> others). Table renamed
    # EntryLink -> WordRelation (Phase 2); the ``links_out`` attribute name is
    # kept so the read model / consumers do not churn.
    links_out: Mapped[list["WordRelation"]] = relationship(
        back_populates="from_word",
        foreign_keys="WordRelation.from_word_id",
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


class WordRelation(Base):
    """A WORD-level relation (this word -> another word), Phase 2 rename of the
    former ``EntryLink`` / ``entry_links`` table. Shape is unchanged: no sense on
    either end, no WSD. ``word_family``/``confused_with``/``variant_of``/
    ``arrow_redirect``/``another_word``/``part_of_phrasal_family`` ride this path.

    Sense-DEPENDENT relations (synonym/antonym/hypernym/...) live in the separate
    :class:`SenseRelation` table (sense-level).
    """

    __tablename__ = "word_relation"
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


class SenseRelation(Base):
    """A SENSE-level semantic relation: it is BOTH the edge AND the WSD work-queue
    row (Phase 2). Emitted at generation time as a half-edge
    ``from_sense -> (to_word, gloss)``; a later WSD pass fills ``to_sense_id``.

    State is DERIVED (Q1 — there is deliberately NO ``wsd_state`` column):

    - ``resolved``     ⟺ ``to_sense_id IS NOT NULL``
    - ``unresolvable`` ⟺ ``to_sense_id IS NULL AND resolve_attempted_at IS NOT NULL``
    - ``pending``      ⟺ ``to_sense_id IS NULL AND resolve_attempted_at IS NULL``

    FK ondelete is load-bearing:

    - ``from_sense_id`` CASCADE — source sense gone ⇒ the edge is meaningless, drop
      it (Case 6; re-emitted when the source regenerates).
    - ``to_sense_id``   SET NULL — target sense gone ⇒ keep the edge at sense->word
      level (``to_word_id`` still valid), only the resolved target is cleared. This
      auto-demotes ``resolved`` -> (``to_sense_id`` NULL) via a single FK path, so
      there is no hand-maintained column to fall out of sync (the F4 bug class).
      A ``_demote_edges_for_senses`` helper still resets ``resolve_attempted_at``
      so the edge lands back in ``pending`` rather than ``unresolvable`` (Phase 5).
    """

    __tablename__ = "sense_relation"
    __table_args__ = (
        UniqueConstraint("from_sense_id", "to_word_id", "rel_type", name="uq_sense_rel"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    # Source is ALWAYS a concrete sense (the meaning that emits the relation).
    from_sense_id: Mapped[int] = mapped_column(
        ForeignKey("senses.id", ondelete="CASCADE"), nullable=False
    )
    # Target WORD is always present (stub-row pattern) — the consumer always gets
    # at least sense->word, enough to display.
    to_word_id: Mapped[int] = mapped_column(
        ForeignKey("words.id", ondelete="CASCADE"), nullable=False
    )
    # Target SENSE: filled by WSD (resolved); SET NULL when the target sense is
    # deleted/regenerated (Case 2). NULL = not-yet / no-longer resolved.
    to_sense_id: Mapped[int | None] = mapped_column(ForeignKey("senses.id", ondelete="SET NULL"))
    rel_type: Mapped[str] = mapped_column(String(32), nullable=False)
    # LM description of the TARGET's intended meaning — the load-bearing signal WSD
    # uses to pick the right target sense. Non-empty (Phase 3 skips empty edges).
    gloss: Mapped[str] = mapped_column(Text, nullable=False)
    # sha256 of the resolved to_sense content, stamped at resolve; VERIFIED on read
    # (Phase 6) so a mutated/regenerated target is treated as unresolved.
    target_hash: Mapped[str | None] = mapped_column(String(64))
    # Marks "WSD tried and the judge returned none" — the ONLY thing distinguishing
    # derived ``unresolvable`` from ``pending`` (Q1). NO ``wsd_state`` column.
    resolve_attempted_at: Mapped[datetime | None] = mapped_column(DateTime)

    from_sense: Mapped["Sense"] = relationship(
        back_populates="relations_out", foreign_keys=[from_sense_id]
    )
    to_word: Mapped["Word"] = relationship(foreign_keys=[to_word_id])
    to_sense: Mapped["Sense | None"] = relationship(foreign_keys=[to_sense_id])


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
    # IPA pronunciation per POS (senses carry pos; Cambridge entries are POS-grouped,
    # so per-sense IPA folds heteronyms in naturally). Cambridge-anchored, LLM
    # fallback for out-of-Cambridge words. Sanitized through _clean on write.
    ipa_uk: Mapped[str | None] = mapped_column(String(64))
    ipa_us: Mapped[str | None] = mapped_column(String(64))

    # Learner-dictionary enrichments (best-effort, null when unmarked). ``grammar``
    # holds a comma-joined set of schema-validated GRAMMAR_LABELS tokens (never
    # queried alone), split back to a list on read; ``register``/``connotation``
    # are single closed-vocab enum tokens; ``guideword`` is a short free label.
    guideword: Mapped[str | None] = mapped_column(String(64))
    grammar: Mapped[str | None] = mapped_column(String(128))
    register: Mapped[str | None] = mapped_column(String(32))
    connotation: Mapped[str | None] = mapped_column(String(16))
    # ``domain`` = open-ended subject-area label (computing, medicine, law) — free
    # text, not an enum (the field set is unbounded). ``usage_note`` = one-line
    # usage / confusable hint ("don't confuse with affect"). Both sanitized on write.
    domain: Mapped[str | None] = mapped_column(String(64))
    usage_note: Mapped[str | None] = mapped_column(String(255))

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
    forms: Mapped[list["SenseForm"]] = relationship(
        back_populates="sense",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    # Sense-level relations THIS sense emits (synonym/antonym/hypernym/...). The
    # edge rows carry from_sense_id -> to_word (+ optional resolved to_sense).
    # CASCADE on from_sense_id: regenerating/deleting this sense drops its edges
    # (Case 6). Edges targeting THIS sense (to_sense_id) are NOT in this collection
    # and are SET NULL, not cascaded (Case 2).
    relations_out: Mapped[list["SenseRelation"]] = relationship(
        back_populates="from_sense",
        foreign_keys="SenseRelation.from_sense_id",
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


class SenseForm(Base):
    """An inflected form of a sense's headword (run -> ran/running/runs).

    The model emits the FULL paradigm per POS directly (verb: base/past/
    past_participle/present_3sg/ing; noun: plural; adjective: comparative/
    superlative), so this is not scraped from examples — an example uses only one
    form, but the paradigm must be complete. One row per (inf, surface); a label
    may repeat when a form has variants (dreamed / dreamt). Mirrors ``examples``/
    ``collocations`` as an ordered child table rather than a column."""

    __tablename__ = "sense_forms"

    id: Mapped[int] = mapped_column(primary_key=True)
    sense_id: Mapped[int] = mapped_column(
        ForeignKey("senses.id", ondelete="CASCADE"), nullable=False
    )
    inf: Mapped[str] = mapped_column(String(24), nullable=False)  # ∈ INFLECTION_LABELS
    surface: Mapped[str] = mapped_column(Text, nullable=False)
    form_order: Mapped[int] = mapped_column(Integer, default=0)

    sense: Mapped["Sense"] = relationship(back_populates="forms")


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

    Themes are addressed BY KEY at the API (``get_entry``/``generate``/
    ``get_theme``/``update_theme``/``delete_theme`` take a ``theme_key`` string),
    so — unlike ``tag_key`` — the key is intentionally exposed in the read model.
    """

    __tablename__ = "themes"

    id: Mapped[int] = mapped_column(primary_key=True)
    # Lossy, CODE-computed. UNIQUE = durable dedup + concurrency guard (like tags).
    theme_key: Mapped[str] = mapped_column(String(255), nullable=False, unique=True, index=True)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    style_prompt: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    tone: Mapped[str | None] = mapped_column(String(255), nullable=True)
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
    """A reference-addressed derived asset (translation text, TTS clip).

    Identity is ``(source_kind, source_id, kind, params)`` — the source row this
    asset derives from (e.g. ``sense_def``/``42``) plus the asset kind and a
    normalized param token. A consumer holding a ``sense_id`` can look its
    translation/audio up directly. ``content_hash`` is NOT part of the identity;
    it stores the sha256 of the source text AT WRITE TIME and is VERIFIED on read
    so a reused/regenerated ``source_id`` yields a miss (regenerate), never
    poisoned content. ``text_value`` holds inline results (translation);
    ``file_path`` points at a binary clip (TTS) relative to ``LEXI_ASSET_CACHE_DIR``.

    No cross-table FK on ``source_id`` (the source may be a sense, example, or
    collocation depending on ``source_kind``); the read-time hash verify — not a
    FK cascade — is the correctness guarantee. Cascade-on-delete is best-effort GC.
    """

    __tablename__ = "assets"
    __table_args__ = (
        UniqueConstraint("source_kind", "source_id", "kind", "params", name="uq_asset_identity"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    source_kind: Mapped[str] = mapped_column(String(32), nullable=False)  # ∈ SOURCE_KINDS
    source_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    kind: Mapped[str] = mapped_column(String(16), nullable=False)  # ∈ ASSET_KINDS
    params: Mapped[str] = mapped_column(String(64), nullable=False)
    # sha256 of the source text at write time — VERIFIED on read, never identity.
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
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
