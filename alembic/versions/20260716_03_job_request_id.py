"""Persist request correlation for worker replay logs.

Revision ID: 20260716_03
Revises: 20260716_02
Create Date: 2026-07-16
"""

from alembic import op

revision = "20260716_03"
down_revision = "20260716_02"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE service_jobs ADD COLUMN IF NOT EXISTS request_id VARCHAR(255)")


def downgrade() -> None:
    # Retained for N/N-1 workers and correlation during a service rollback.
    pass
