"""sense vectors leave the primary db

Sense embeddings move to a dedicated vector index keyed by sense id. They could
never join this table's transaction, and embedding is a post-commit best-effort
step, so storing them here only pretended the two were consistent.

No data is carried across: the vectors are derived content, and a backfill
regenerates them into the new store. Semantic search returns nothing (never an
error) until that backfill runs.

Revision ID: 20260724_02
Revises: 20260724_01
Create Date: 2026-07-24
"""

import sqlalchemy as sa
from alembic import op

revision = "20260724_02"
down_revision = "20260724_01"
branch_labels = None
depends_on = None

_COLUMNS = ("embedding", "embedding_model", "embedding_dim")


def _schema() -> str:
    """The configured domain schema, resolved by the alembic environment."""
    from alembic import context

    return context.config.attributes["lexi_schema"]


def upgrade() -> None:
    # batch_alter_table so SQLite gets a table rebuild instead of an unsupported
    # DROP COLUMN; on Postgres it compiles to plain ALTER TABLE statements.
    with op.batch_alter_table("senses", schema=_schema()) as batch:
        for column in _COLUMNS:
            batch.drop_column(column)


def downgrade() -> None:
    with op.batch_alter_table("senses", schema=_schema()) as batch:
        batch.add_column(sa.Column("embedding", sa.LargeBinary(), nullable=True))
        batch.add_column(sa.Column("embedding_model", sa.String(length=128), nullable=True))
        batch.add_column(sa.Column("embedding_dim", sa.Integer(), nullable=True))
