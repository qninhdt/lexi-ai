"""Create service-owned jobs and outbox tables.

Revision ID: 20260716_01
Revises:
Create Date: 2026-07-16
"""

import sqlalchemy as sa

from alembic import op

revision = "20260716_01"
down_revision = None
branch_labels = ("service",)
depends_on = None


def upgrade() -> None:
    op.create_table(
        "service_jobs",
        sa.Column("job_id", sa.String(36), primary_key=True),
        sa.Column("request_id", sa.String(255)),
        sa.Column("owner_subject", sa.String(255), nullable=False),
        sa.Column("owner_tenant", sa.String(255), nullable=False, server_default=""),
        sa.Column("operation", sa.String(32), nullable=False),
        sa.Column("idempotency_key", sa.String(255), nullable=False),
        sa.Column("status", sa.String(32), nullable=False, server_default="queued"),
        sa.Column("payload_version", sa.Integer, nullable=False),
        sa.Column("payload_json", sa.Text, nullable=False),
        sa.Column("payload_hash", sa.String(64), nullable=False),
        sa.Column("reference_dataset_fingerprint", sa.String(255), nullable=False),
        sa.Column("accepted_at", sa.DateTime, nullable=False),
        sa.Column("expires_at", sa.DateTime, nullable=False),
        sa.Column("max_retries", sa.Integer, nullable=False),
        sa.Column("attempt", sa.Integer, nullable=False, server_default="0"),
        sa.Column("generation_epoch", sa.Integer, nullable=False, server_default="0"),
        sa.Column("lease_token", sa.String(36)),
        sa.Column("lease_expires_at", sa.DateTime),
        sa.Column("public_error_code", sa.String(64)),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint(
            "owner_subject",
            "owner_tenant",
            "operation",
            "idempotency_key",
            name="uq_job_idempotency",
        ),
    )
    op.create_table(
        "service_outbox_events",
        sa.Column("event_id", sa.String(36), primary_key=True),
        sa.Column("job_id", sa.String(36), nullable=False, unique=True),
        sa.Column("operation", sa.String(32), nullable=False),
        sa.Column("payload_version", sa.Integer, nullable=False),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.Column("dispatched_at", sa.DateTime),
    )
    op.create_index("ix_service_outbox_events_job_id", "service_outbox_events", ["job_id"])
    op.create_table(
        "service_job_effects",
        sa.Column("effect_id", sa.String(36), primary_key=True),
        sa.Column("job_id", sa.String(36), nullable=False),
        sa.Column("effect_kind", sa.String(64), nullable=False),
        sa.Column("ordinal", sa.Integer, nullable=False, server_default="0"),
        sa.Column("result_json", sa.Text, nullable=False),
        sa.Column("committed_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("job_id", "effect_kind", "ordinal", name="uq_job_effect"),
    )


def downgrade() -> None:
    op.drop_table("service_job_effects")
    op.drop_table("service_outbox_events")
    op.drop_table("service_jobs")
