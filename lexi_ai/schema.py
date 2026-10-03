"""Generated-dictionary schema. All child deletion and namespace rules live in FKs."""

from sqlalchemy import (
    CheckConstraint,
    ForeignKey,
    Index,
    String,
    Text,
    UniqueConstraint,
    event,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, validates

from lexi_ai.patterns import pattern_head_key, surface_head_key
from lexi_ai.text import answer_key
from lexi_ai.vocab import (
    ENTRY_TYPES,
    GENERATION_STATES,
    INFLECTIONS,
    POS_TAGS,
    QUESTION_TYPES,
    SENSE_REL_TYPES,
    TIERS,
    WORD_REL_TYPES,
    GenerationState,
)

from .questions.positions import install_positions, remove_positions
from .relations.invalidation import install_relation_triggers, remove_relation_triggers


def _choice(column: str, values: set[str] | frozenset[str] | tuple[str, ...]) -> CheckConstraint:
    choices = ", ".join(f"'{item}'" for item in sorted(values))
    return CheckConstraint(f"{column} IN ({choices})", name=f"ck_{column}_vocab")


class Base(DeclarativeBase):
    pass


class Word(Base):
    __tablename__ = "words"
    __table_args__ = (
        _choice("entry_type", ENTRY_TYPES),
        _choice("generation_state", GENERATION_STATES),
        Index(
            "ix_words_match_key_prefix",
            "match_key",
            postgresql_ops={"match_key": "varchar_pattern_ops"},
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    lemma: Mapped[str] = mapped_column(Text, nullable=False)
    match_key: Mapped[str] = mapped_column(String(512), unique=True, nullable=False)
    entry_type: Mapped[str | None] = mapped_column(String(32))
    generation_state: Mapped[str] = mapped_column(
        String(16), nullable=False, default=GenerationState.PENDING
    )


class WordAlias(Base):
    __tablename__ = "word_aliases"
    __table_args__ = (
        UniqueConstraint("word_id", "match_key"),
        Index(
            "ix_word_aliases_match_key",
            "match_key",
            postgresql_ops={"match_key": "varchar_pattern_ops"},
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    word_id: Mapped[int] = mapped_column(ForeignKey("words.id", ondelete="CASCADE"), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    match_key: Mapped[str] = mapped_column(String(512), nullable=False)


class WordSource(Base):
    __tablename__ = "word_sources"

    id: Mapped[int] = mapped_column(primary_key=True)
    word_id: Mapped[int] = mapped_column(
        ForeignKey("words.id", ondelete="CASCADE"), nullable=False, index=True
    )
    source_id: Mapped[int] = mapped_column(unique=True, nullable=False)


class Sense(Base):
    __tablename__ = "senses"
    __table_args__ = (
        _choice("pos", POS_TAGS),
        _choice("tier", TIERS),
        Index("ix_senses_word_pos_order", "word_id", "pos", "id"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    word_id: Mapped[int] = mapped_column(
        ForeignKey("words.id", ondelete="CASCADE"), nullable=False, index=True
    )
    pos: Mapped[str] = mapped_column(String(32), nullable=False)
    tier: Mapped[str] = mapped_column(String(16), nullable=False)
    cefr_level: Mapped[str | None] = mapped_column(String(8))
    register: Mapped[str | None] = mapped_column(String(32))
    usage_note: Mapped[str | None] = mapped_column(Text)
    ipa_uk: Mapped[str | None] = mapped_column(String(80))
    ipa_us: Mapped[str | None] = mapped_column(String(80))


class Theme(Base):
    __tablename__ = "themes"

    id: Mapped[int] = mapped_column(primary_key=True)
    key: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    voice: Mapped[str] = mapped_column(Text, nullable=False)
    diction: Mapped[str] = mapped_column(Text, nullable=False)


class Definition(Base):
    __tablename__ = "definitions"
    __table_args__ = (
        Index(
            "ux_definitions_neutral",
            "sense_id",
            unique=True,
            sqlite_where=text("theme_id IS NULL"),
            postgresql_where=text("theme_id IS NULL"),
        ),
        Index(
            "ux_definitions_themed",
            "sense_id",
            "theme_id",
            unique=True,
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    sense_id: Mapped[int] = mapped_column(
        ForeignKey("senses.id", ondelete="CASCADE"), nullable=False
    )
    theme_id: Mapped[int | None] = mapped_column(
        ForeignKey("themes.id", ondelete="CASCADE"), index=True
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)


class Example(Base):
    __tablename__ = "examples"
    __table_args__ = (Index("ix_examples_scope", "sense_id", "theme_id"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    sense_id: Mapped[int] = mapped_column(
        ForeignKey("senses.id", ondelete="CASCADE"), nullable=False
    )
    theme_id: Mapped[int | None] = mapped_column(
        ForeignKey("themes.id", ondelete="CASCADE"), index=True
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)


class SenseForm(Base):
    __tablename__ = "sense_forms"
    __table_args__ = (
        _choice("inf", INFLECTIONS),
        Index(
            "ix_sense_forms_match_key",
            "match_key",
            postgresql_ops={"match_key": "varchar_pattern_ops"},
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    sense_id: Mapped[int] = mapped_column(
        ForeignKey("senses.id", ondelete="CASCADE"), nullable=False, index=True
    )
    surface: Mapped[str] = mapped_column(Text, nullable=False)
    match_key: Mapped[str] = mapped_column(
        String(512),
        nullable=False,
        default=lambda context: answer_key(context.get_current_parameters()["surface"]),
    )
    head_key: Mapped[str] = mapped_column(
        String(512),
        nullable=False,
        index=True,
        default=lambda context: surface_head_key(context.get_current_parameters()["surface"]),
    )
    inf: Mapped[str] = mapped_column(String(24), nullable=False)

    @validates("surface")
    def normalize_surface(self, _key, surface):
        self.match_key = answer_key(surface)
        self.head_key = surface_head_key(surface)
        return surface


class SensePattern(Base):
    __tablename__ = "sense_patterns"

    id: Mapped[int] = mapped_column(primary_key=True)
    sense_id: Mapped[int] = mapped_column(
        ForeignKey("senses.id", ondelete="CASCADE"), nullable=False, index=True
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    head_key: Mapped[str | None] = mapped_column(
        String(512),
        index=True,
        default=lambda context: pattern_head_key(context.get_current_parameters()["content"]),
    )

    @validates("content")
    def normalize_head(self, _key, content):
        self.head_key = pattern_head_key(content)
        return content


class Collocation(Base):
    __tablename__ = "collocations"

    id: Mapped[int] = mapped_column(primary_key=True)
    sense_id: Mapped[int] = mapped_column(
        ForeignKey("senses.id", ondelete="CASCADE"), nullable=False, index=True
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)


class SenseReference(Base):
    __tablename__ = "sense_references"

    id: Mapped[int] = mapped_column(primary_key=True)
    sense_id: Mapped[int] = mapped_column(
        ForeignKey("senses.id", ondelete="CASCADE"), nullable=False, index=True
    )
    source: Mapped[str] = mapped_column(String(16), nullable=False)
    source_ref: Mapped[str] = mapped_column(String(255), nullable=False)


class WordRelation(Base):
    __tablename__ = "word_relations"
    __table_args__ = (
        UniqueConstraint("from_word_id", "to_word_id", "rel_type"),
        _choice("rel_type", WORD_REL_TYPES),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    from_word_id: Mapped[int] = mapped_column(
        ForeignKey("words.id", ondelete="CASCADE"), nullable=False
    )
    to_word_id: Mapped[int] = mapped_column(
        ForeignKey("words.id", ondelete="CASCADE"), nullable=False, index=True
    )
    rel_type: Mapped[str] = mapped_column(String(32), nullable=False)


class SenseRelation(Base):
    __tablename__ = "sense_relations"
    __table_args__ = (
        UniqueConstraint("from_sense_id", "to_word_id", "rel_type"),
        _choice("rel_type", SENSE_REL_TYPES),
        Index(
            "ix_sense_relations_target_pending",
            "to_word_id",
            "id",
            sqlite_where=text("resolve_attempted_at IS NULL"),
            postgresql_where=text("resolve_attempted_at IS NULL"),
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    from_sense_id: Mapped[int] = mapped_column(
        ForeignKey("senses.id", ondelete="CASCADE"), nullable=False
    )
    to_word_id: Mapped[int] = mapped_column(
        ForeignKey("words.id", ondelete="CASCADE"), nullable=False, index=True
    )
    to_sense_id: Mapped[int | None] = mapped_column(
        ForeignKey("senses.id", ondelete="SET NULL"), index=True
    )
    rel_type: Mapped[str] = mapped_column(String(32), nullable=False)
    gloss: Mapped[str] = mapped_column(Text, nullable=False)
    target_hash: Mapped[str | None] = mapped_column(String(64))
    resolve_attempted_at: Mapped[str | None] = mapped_column(String(32))


class Question(Base):
    __tablename__ = "questions"
    __table_args__ = (
        Index("ix_questions_scope", "sense_id", "theme_id", "position"),
        Index("ix_questions_type_scope", "sense_id", "theme_id", "question_type", "type_position"),
        Index("ix_questions_list_scope", "sense_id", "theme_id", "id"),
        Index("ix_questions_type_list_scope", "sense_id", "theme_id", "question_type", "id"),
        _choice("question_type", QUESTION_TYPES),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    sense_id: Mapped[int] = mapped_column(
        ForeignKey("senses.id", ondelete="CASCADE"), nullable=False
    )
    theme_id: Mapped[int | None] = mapped_column(
        ForeignKey("themes.id", ondelete="CASCADE"), index=True
    )
    question_type: Mapped[str] = mapped_column(String(32), nullable=False)
    payload: Mapped[str] = mapped_column(Text, nullable=False)
    position: Mapped[int] = mapped_column(nullable=False, server_default=text("0"))
    type_position: Mapped[int] = mapped_column(nullable=False, server_default=text("0"))


class Translation(Base):
    __tablename__ = "translations"
    __table_args__ = (UniqueConstraint("input_hash", "target_language"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    input_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    target_language: Mapped[str] = mapped_column(String(16), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)


@event.listens_for(Base.metadata, "after_create")
def _install_domain_triggers(_metadata, connection, **_kwargs):
    install_positions(connection)
    install_relation_triggers(connection)


@event.listens_for(Base.metadata, "before_drop")
def _remove_domain_triggers(_metadata, connection, **_kwargs):
    remove_positions(connection)
    remove_relation_triggers(connection)
