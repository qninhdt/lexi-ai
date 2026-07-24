"""Create the Lexi domain tables in the configured PostgreSQL schema.

Revision ID: 20260722_01
Revises:
Create Date: 2026-07-22
"""

from alembic import context, op

from lexi_ai.migrations.frozen_schema import metadata

revision = "20260722_01"
down_revision = None
branch_labels = ("lexi",)
depends_on = None


def _schema() -> str:
    return context.config.attributes["lexi_schema"]


def upgrade() -> None:
    metadata(_schema()).create_all(op.get_bind())


def downgrade() -> None:
    metadata(_schema()).drop_all(op.get_bind())
