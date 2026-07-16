"""Add a fencing epoch to generated dictionary words.

Revision ID: 20260716_02
Revises: 20260716_01
Create Date: 2026-07-16
"""

from alembic import op

revision = "20260716_02"
down_revision = "20260716_01"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Library bootstrap may already have applied this additive compatibility
    # column before a service is introduced to the same generated DB.
    op.execute(
        "ALTER TABLE words ADD COLUMN IF NOT EXISTS generation_epoch INTEGER NOT NULL DEFAULT 0"
    )
    op.alter_column("words", "generation_epoch", server_default=None)


def downgrade() -> None:
    # This is an expand-only compatibility column shared with the library
    # bootstrap. Rolling back a service image must leave it in place so a newer
    # library process can still read the same generated dictionary.
    pass
