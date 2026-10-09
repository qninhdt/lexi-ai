"""Import reference inflections for the shared search index."""

import sqlalchemy as sa
from alembic import op

revision = "20261010_ref_forms"
down_revision = "20261009_base"
branch_labels = None
depends_on = None


def upgrade():
    connection = op.get_bind()
    if connection.dialect.name != "postgresql" or sa.inspect(connection).has_table(
        "entry_inflections", schema="lexi_reference"
    ):
        return
    op.create_table(
        "entry_inflections",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=False),
        sa.Column("entry_id", sa.Integer(), nullable=False),
        sa.Column("form_type", sa.Text()),
        sa.Column("inflected_form", sa.Text(), nullable=False),
        schema="lexi_reference",
    )
    op.create_index(
        "ix_entry_inflections_entry_id", "entry_inflections", ["entry_id"], schema="lexi_reference"
    )
    # Reimport the pinned artifact once to fill the newly added reference table.
    op.execute("DELETE FROM lexi_reference.datasets WHERE name='reference'")


def downgrade():
    # Shared reference storage outlives individual generated-dictionary schemas.
    pass
