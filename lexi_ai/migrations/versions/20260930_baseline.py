"""Initial Lexicon schema: relational content, dense Questions and lexical indexes.

Revision ID: 20260930_base
Revises: None

Fresh generated dictionaries only; never target the read-only Cambridge database.
Table definitions are frozen here, not imported from live ORM metadata.
"""

import sqlalchemy as sa
from alembic import context, op

from lexi_ai.questions.positions import install_positions, remove_positions
from lexi_ai.relations.invalidation import install_relation_triggers, remove_relation_triggers
from lexi_ai.words.indexes import install_search_index, remove_search_index

revision = "20260930_base"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    if context.is_offline_mode():
        raise RuntimeError("Dictionary bootstrap requires an online migration")
    connection = op.get_bind()
    existing = set(sa.inspect(connection).get_table_names()) - {"alembic_version"}
    if existing:
        raise RuntimeError(
            "Baseline requires an empty generated dictionary schema; "
            "do not overwrite existing tables"
        )
    op.create_table(
        "themes",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("key", sa.String(length=255), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("voice", sa.Text(), nullable=False),
        sa.Column("diction", sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("key"),
    )
    op.create_table(
        "translations",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("input_hash", sa.String(length=64), nullable=False),
        sa.Column("target_language", sa.String(length=16), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("input_hash", "target_language"),
    )
    op.create_table(
        "words",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("lemma", sa.Text(), nullable=False),
        sa.Column("match_key", sa.String(length=512), nullable=False),
        sa.Column("entry_type", sa.String(length=32), nullable=True),
        sa.Column("generation_state", sa.String(length=16), nullable=False),
        sa.CheckConstraint(
            "entry_type IN ('expression', 'idiom', 'phrasal_verb', 'phrase', 'word')",
            name="ck_entry_type_vocab",
        ),
        sa.CheckConstraint(
            "generation_state IN ('done', 'error', 'pending')", name="ck_generation_state_vocab"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("match_key"),
    )
    op.create_index(
        "ix_words_match_key_prefix",
        "words",
        ["match_key"],
        postgresql_ops={"match_key": "varchar_pattern_ops"},
    )
    op.create_table(
        "senses",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("word_id", sa.Integer(), nullable=False),
        sa.Column("pos", sa.String(length=32), nullable=False),
        sa.Column("tier", sa.String(length=16), nullable=False),
        sa.Column("cefr_level", sa.String(length=8)),
        sa.Column("register", sa.String(length=32)),
        sa.Column("usage_note", sa.Text()),
        sa.Column("ipa_uk", sa.String(length=80)),
        sa.Column("ipa_us", sa.String(length=80)),
        sa.CheckConstraint(
            "pos IN ('adjective', 'adverb', 'article', 'auxiliary', 'conjunction', "
            "'determiner', 'interjection', 'noun', 'numeral', 'preposition', 'pronoun', 'verb')",
            name="ck_pos_vocab",
        ),
        sa.CheckConstraint(
            "tier IN ('common', 'core', 'less_common', 'rare')", name="ck_tier_vocab"
        ),
        sa.ForeignKeyConstraint(["word_id"], ["words.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_senses_word_id", "senses", ["word_id"])
    op.create_index("ix_senses_word_pos_order", "senses", ["word_id", "pos", "id"])
    op.create_table(
        "word_aliases",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("word_id", sa.Integer(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("match_key", sa.String(length=512), nullable=False),
        sa.ForeignKeyConstraint(["word_id"], ["words.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("word_id", "match_key"),
    )
    op.create_index(
        "ix_word_aliases_match_key",
        "word_aliases",
        ["match_key"],
        postgresql_ops={"match_key": "varchar_pattern_ops"},
    )
    op.create_table(
        "word_relations",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("from_word_id", sa.Integer(), nullable=False),
        sa.Column("to_word_id", sa.Integer(), nullable=False),
        sa.Column("rel_type", sa.String(length=32), nullable=False),
        sa.CheckConstraint(
            "rel_type IN ('confused_with', 'part_of_phrasal_family', 'word_family')",
            name="ck_rel_type_vocab",
        ),
        sa.ForeignKeyConstraint(["from_word_id"], ["words.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["to_word_id"], ["words.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("from_word_id", "to_word_id", "rel_type"),
    )
    op.create_index("ix_word_relations_to_word_id", "word_relations", ["to_word_id"])
    op.create_table(
        "word_sources",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("word_id", sa.Integer(), nullable=False),
        sa.Column("source_id", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(["word_id"], ["words.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("source_id"),
    )
    op.create_index("ix_word_sources_word_id", "word_sources", ["word_id"])
    op.create_table(
        "collocations",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("sense_id", sa.Integer(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.ForeignKeyConstraint(["sense_id"], ["senses.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_collocations_sense_id", "collocations", ["sense_id"])
    op.create_table(
        "definitions",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("sense_id", sa.Integer(), nullable=False),
        sa.Column("theme_id", sa.Integer()),
        sa.Column("content", sa.Text(), nullable=False),
        sa.ForeignKeyConstraint(["sense_id"], ["senses.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["theme_id"], ["themes.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_definitions_theme_id", "definitions", ["theme_id"])
    op.create_index(
        "ux_definitions_neutral",
        "definitions",
        ["sense_id"],
        unique=True,
        sqlite_where=sa.text("theme_id IS NULL"),
        postgresql_where=sa.text("theme_id IS NULL"),
    )
    op.create_index("ux_definitions_themed", "definitions", ["sense_id", "theme_id"], unique=True)
    op.create_table(
        "examples",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("sense_id", sa.Integer(), nullable=False),
        sa.Column("theme_id", sa.Integer()),
        sa.Column("content", sa.Text(), nullable=False),
        sa.ForeignKeyConstraint(["sense_id"], ["senses.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["theme_id"], ["themes.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_examples_scope", "examples", ["sense_id", "theme_id"])
    op.create_index("ix_examples_theme_id", "examples", ["theme_id"])
    op.create_table(
        "questions",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("sense_id", sa.Integer(), nullable=False),
        sa.Column("theme_id", sa.Integer()),
        sa.Column("question_type", sa.String(length=32), nullable=False),
        sa.Column("payload", sa.Text(), nullable=False),
        sa.Column("position", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("type_position", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.CheckConstraint(
            "question_type IN ('cloze_to_word', 'context_to_word', 'definition_to_word', "
            "'dialogue_completion', 'meaning_in_context', 'word_to_definition', 'word_to_usage')",
            name="ck_question_type_vocab",
        ),
        sa.ForeignKeyConstraint(["sense_id"], ["senses.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["theme_id"], ["themes.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_questions_list_scope", "questions", ["sense_id", "theme_id", "id"])
    op.create_index("ix_questions_scope", "questions", ["sense_id", "theme_id", "position"])
    op.create_index("ix_questions_theme_id", "questions", ["theme_id"])
    op.create_index(
        "ix_questions_type_list_scope", "questions", ["sense_id", "theme_id", "question_type", "id"]
    )
    op.create_index(
        "ix_questions_type_scope",
        "questions",
        ["sense_id", "theme_id", "question_type", "type_position"],
    )
    op.create_table(
        "sense_forms",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("sense_id", sa.Integer(), nullable=False),
        sa.Column("surface", sa.Text(), nullable=False),
        sa.Column("match_key", sa.String(length=512), nullable=False),
        sa.Column("head_key", sa.String(length=512), nullable=False),
        sa.Column("inf", sa.String(length=24), nullable=False),
        sa.CheckConstraint(
            "inf IN ('base', 'comparative', 'ing', 'past', 'past_participle', "
            "'plural', 'present_3sg', 'superlative')",
            name="ck_inf_vocab",
        ),
        sa.ForeignKeyConstraint(["sense_id"], ["senses.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_sense_forms_head_key", "sense_forms", ["head_key"])
    op.create_index(
        "ix_sense_forms_match_key",
        "sense_forms",
        ["match_key"],
        postgresql_ops={"match_key": "varchar_pattern_ops"},
    )
    op.create_index("ix_sense_forms_sense_id", "sense_forms", ["sense_id"])
    op.create_table(
        "sense_patterns",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("sense_id", sa.Integer(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("head_key", sa.String(length=512)),
        sa.ForeignKeyConstraint(["sense_id"], ["senses.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_sense_patterns_head_key", "sense_patterns", ["head_key"])
    op.create_index("ix_sense_patterns_sense_id", "sense_patterns", ["sense_id"])
    op.create_table(
        "sense_references",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("sense_id", sa.Integer(), nullable=False),
        sa.Column("source", sa.String(length=16), nullable=False),
        sa.Column("source_ref", sa.String(length=255), nullable=False),
        sa.ForeignKeyConstraint(["sense_id"], ["senses.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_sense_references_sense_id", "sense_references", ["sense_id"])
    op.create_table(
        "sense_relations",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("from_sense_id", sa.Integer(), nullable=False),
        sa.Column("to_word_id", sa.Integer(), nullable=False),
        sa.Column("to_sense_id", sa.Integer()),
        sa.Column("rel_type", sa.String(length=32), nullable=False),
        sa.Column("gloss", sa.Text(), nullable=False),
        sa.Column("target_hash", sa.String(length=64)),
        sa.Column("resolve_attempted_at", sa.String(length=32)),
        sa.CheckConstraint(
            "rel_type IN ('antonym', 'holonym', 'hypernym', 'hyponym', 'meronym', 'synonym')",
            name="ck_rel_type_vocab",
        ),
        sa.ForeignKeyConstraint(["from_sense_id"], ["senses.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["to_sense_id"], ["senses.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["to_word_id"], ["words.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("from_sense_id", "to_word_id", "rel_type"),
    )
    op.create_index(
        "ix_sense_relations_target_pending",
        "sense_relations",
        ["to_word_id", "id"],
        sqlite_where=sa.text("resolve_attempted_at IS NULL"),
        postgresql_where=sa.text("resolve_attempted_at IS NULL"),
    )
    op.create_index("ix_sense_relations_to_sense_id", "sense_relations", ["to_sense_id"])
    op.create_index("ix_sense_relations_to_word_id", "sense_relations", ["to_word_id"])
    install_positions(connection)
    install_relation_triggers(connection)
    install_search_index(connection)


def downgrade():
    connection = op.get_bind()
    remove_positions(connection)
    remove_search_index(connection)
    remove_relation_triggers(connection)
    for table in (
        "sense_relations",
        "sense_references",
        "sense_patterns",
        "sense_forms",
        "questions",
        "examples",
        "definitions",
        "collocations",
        "word_sources",
        "word_relations",
        "word_aliases",
        "senses",
        "words",
        "translations",
        "themes",
    ):
        op.drop_table(table)
