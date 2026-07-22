"""Immutable PostgreSQL schema captured by the initial Lexi migration."""

import sqlalchemy as sa


def metadata(schema: str) -> sa.MetaData:
    """Return the schema frozen at revision 20260722_01, independent of ORM models."""
    m = sa.MetaData(schema=schema)
    words = sa.Table(
        "words",
        m,
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("norm", sa.Text, nullable=False),
        sa.Column("match_key", sa.String(512), nullable=False, unique=True, index=True),
        sa.Column("entry_type", sa.String(32)),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("generation_epoch", sa.Integer, nullable=False),
        sa.Column("pos", sa.String(32)),
        sa.Column("cambridge_word_id", sa.Integer),
        sa.Column("error_msg", sa.Text),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime, nullable=False, server_default=sa.text("now()")),
    )
    senses = sa.Table(
        "senses",
        m,
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column(
            "word_id", sa.Integer, sa.ForeignKey(words.c.id, ondelete="CASCADE"), nullable=False
        ),
        sa.Column("definition", sa.Text, nullable=False),
        sa.Column("tier", sa.String(16), nullable=False),
        sa.Column("sense_order", sa.Integer, nullable=False),
        sa.Column("pos", sa.String(32)),
        sa.Column("cefr_level", sa.String(8)),
        sa.Column("ipa_uk", sa.String(64)),
        sa.Column("ipa_us", sa.String(64)),
        sa.Column("guideword", sa.String(64)),
        sa.Column("grammar", sa.String(128)),
        sa.Column("register", sa.String(32)),
        sa.Column("connotation", sa.String(16)),
        sa.Column("domain", sa.String(64)),
        sa.Column("usage_note", sa.String(255)),
        sa.Column("embedding", sa.LargeBinary),
        sa.Column("embedding_model", sa.String(128)),
        sa.Column("embedding_dim", sa.Integer),
    )
    tags = sa.Table(
        "tags",
        m,
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("name", sa.Text, nullable=False),
        sa.Column("title", sa.Text, nullable=False),
        sa.Column("tag_key", sa.String(255), nullable=False, unique=True, index=True),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.text("now()")),
    )
    themes = sa.Table(
        "themes",
        m,
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("theme_key", sa.String(255), nullable=False, unique=True, index=True),
        sa.Column("name", sa.Text, nullable=False),
        sa.Column("style_prompt", sa.Text, nullable=False),
        sa.Column("description", sa.Text),
        sa.Column("tone", sa.String(255)),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.text("now()")),
    )
    sa.Table(
        "word_aliases",
        m,
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column(
            "word_id", sa.Integer, sa.ForeignKey(words.c.id, ondelete="CASCADE"), nullable=False
        ),
        sa.Column("alias_norm", sa.Text, nullable=False),
        sa.Column("alias_match_key", sa.String(512), nullable=False, index=True),
        sa.Column("type", sa.String(32), nullable=False),
        sa.Column("dialect", sa.String(8)),
        sa.UniqueConstraint("word_id", "alias_match_key", name="uq_alias_word_key"),
    )
    sa.Table(
        "word_relation",
        m,
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column(
            "from_word_id",
            sa.Integer,
            sa.ForeignKey(words.c.id, ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "to_word_id", sa.Integer, sa.ForeignKey(words.c.id, ondelete="CASCADE"), nullable=False
        ),
        sa.Column("rel_type", sa.String(32), nullable=False),
        sa.UniqueConstraint("from_word_id", "to_word_id", "rel_type", name="uq_link_triple"),
    )
    sa.Table(
        "sense_relation",
        m,
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column(
            "from_sense_id",
            sa.Integer,
            sa.ForeignKey(senses.c.id, ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "to_word_id", sa.Integer, sa.ForeignKey(words.c.id, ondelete="CASCADE"), nullable=False
        ),
        sa.Column("to_sense_id", sa.Integer, sa.ForeignKey(senses.c.id, ondelete="SET NULL")),
        sa.Column("rel_type", sa.String(32), nullable=False),
        sa.Column("gloss", sa.Text, nullable=False),
        sa.Column("target_hash", sa.String(64)),
        sa.Column("resolve_attempted_at", sa.DateTime),
        sa.UniqueConstraint("from_sense_id", "to_word_id", "rel_type", name="uq_sense_rel"),
    )
    for name, _column, columns in (
        (
            "sense_reference",
            "source_ref",
            (("source", sa.String(16)), ("source_ref", sa.String(255))),
        ),
        ("examples", "text", (("text", sa.Text), ("example_order", sa.Integer))),
        ("collocations", "text", (("text", sa.Text), ("collocation_order", sa.Integer))),
        (
            "sense_forms",
            "inf",
            (("inf", sa.String(24)), ("surface", sa.Text), ("form_order", sa.Integer)),
        ),
    ):
        sa.Table(
            name,
            m,
            sa.Column("id", sa.Integer, primary_key=True),
            sa.Column(
                "sense_id",
                sa.Integer,
                sa.ForeignKey(senses.c.id, ondelete="CASCADE"),
                nullable=False,
            ),
            *(
                sa.Column(column_name, column_type, nullable=False)
                for column_name, column_type in columns
            ),
        )
    sa.Table(
        "word_tags",
        m,
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column(
            "word_id", sa.Integer, sa.ForeignKey(words.c.id, ondelete="CASCADE"), nullable=False
        ),
        sa.Column(
            "tag_id", sa.Integer, sa.ForeignKey(tags.c.id, ondelete="CASCADE"), nullable=False
        ),
        sa.UniqueConstraint("word_id", "tag_id", name="uq_word_tag"),
    )
    themed_senses = sa.Table(
        "themed_senses",
        m,
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column(
            "sense_id", sa.Integer, sa.ForeignKey(senses.c.id, ondelete="CASCADE"), nullable=False
        ),
        sa.Column(
            "theme_id", sa.Integer, sa.ForeignKey(themes.c.id, ondelete="CASCADE"), nullable=False
        ),
        sa.Column("definition", sa.Text, nullable=False),
        sa.UniqueConstraint("sense_id", "theme_id", name="uq_themed_sense"),
    )
    sa.Table(
        "themed_examples",
        m,
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column(
            "themed_sense_id",
            sa.Integer,
            sa.ForeignKey(themed_senses.c.id, ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("text", sa.Text, nullable=False),
        sa.Column("example_order", sa.Integer, nullable=False),
    )
    sa.Table(
        "assets",
        m,
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("source_kind", sa.String(32), nullable=False),
        sa.Column("source_id", sa.Integer, nullable=False, index=True),
        sa.Column("kind", sa.String(16), nullable=False),
        sa.Column("params", sa.String(64), nullable=False),
        sa.Column("content_hash", sa.String(64), nullable=False),
        sa.Column("text_value", sa.Text),
        sa.Column("file_path", sa.Text),
        sa.Column("meta", sa.Text),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.text("now()")),
        sa.UniqueConstraint("source_kind", "source_id", "kind", "params", name="uq_asset_identity"),
    )
    sa.Table(
        "questions",
        m,
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column(
            "word_id",
            sa.Integer,
            sa.ForeignKey(words.c.id, ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column("sense_id", sa.Integer, sa.ForeignKey(senses.c.id, ondelete="CASCADE")),
        sa.Column("format", sa.String(32), nullable=False),
        sa.Column("answer_kind", sa.String(16), nullable=False),
        sa.Column("payload", sa.Text, nullable=False),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.text("now()")),
    )
    return m
