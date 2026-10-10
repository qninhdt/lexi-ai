"""Remove usage notes from stored Senses."""

import sqlalchemy as sa
from alembic import op

revision = "20261010_no_notes"
down_revision = "20261010_compact"
branch_labels = None
depends_on = None


def upgrade():
    op.drop_column("senses", "usage_note")


def downgrade():
    # Removed notes cannot be reconstructed.
    op.add_column("senses", sa.Column("usage_note", sa.Text, nullable=True))
