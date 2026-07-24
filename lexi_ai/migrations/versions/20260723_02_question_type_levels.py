"""Add question type, render, difficulty, and interaction contracts.

Revision ID: 20260723_02
Revises: 20260722_01
Create Date: 2026-07-23
"""

import sqlalchemy as sa

from alembic import context, op

revision = "20260723_02"
down_revision = "20260722_01"
branch_labels = None
depends_on = None


def _schema() -> str:
    return context.config.attributes["lexi_schema"]


def upgrade() -> None:
    schema = _schema()
    op.drop_column("questions", "answer_kind", schema=schema)
    op.drop_column("questions", "format", schema=schema)
    op.add_column(
        "questions", sa.Column("type_id", sa.String(length=32), nullable=False), schema=schema
    )
    op.add_column(
        "questions",
        sa.Column("render_format", sa.String(length=16), nullable=False),
        schema=schema,
    )
    op.add_column(
        "questions", sa.Column("difficulty_level", sa.Integer(), nullable=False), schema=schema
    )
    op.add_column(
        "questions",
        sa.Column("interaction_mode", sa.String(length=16), nullable=False),
        schema=schema,
    )
    op.add_column(
        "questions",
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        schema=schema,
    )
    op.create_unique_constraint(
        "uq_question_content",
        "questions",
        ["sense_id", "type_id", "difficulty_level", "content_hash"],
        schema=schema,
    )


def downgrade() -> None:
    schema = _schema()
    op.drop_constraint("uq_question_content", "questions", type_="unique", schema=schema)
    op.drop_column("questions", "content_hash", schema=schema)
    op.drop_column("questions", "interaction_mode", schema=schema)
    op.drop_column("questions", "difficulty_level", schema=schema)
    op.drop_column("questions", "render_format", schema=schema)
    op.drop_column("questions", "type_id", schema=schema)
    op.add_column(
        "questions", sa.Column("format", sa.String(length=32), nullable=False), schema=schema
    )
    op.add_column(
        "questions",
        sa.Column("answer_kind", sa.String(length=16), nullable=False),
        schema=schema,
    )
