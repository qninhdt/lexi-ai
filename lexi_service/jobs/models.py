"""Transport-independent job lifecycle and service-only SQL schema."""

from datetime import datetime
from enum import Enum

from sqlalchemy import DateTime, Integer, String, Text, UniqueConstraint, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class ServiceBase(DeclarativeBase):
    """Metadata owned by Alembic service migrations, never library bootstrap."""


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    EXPIRED = "expired"
    SUPERSEDED = "superseded"
    DEAD_LETTER = "dead_letter"


TERMINAL_JOB_STATUSES = frozenset(
    {
        JobStatus.SUCCEEDED,
        JobStatus.FAILED,
        JobStatus.EXPIRED,
        JobStatus.SUPERSEDED,
        JobStatus.DEAD_LETTER,
    }
)


def may_transition(current: JobStatus, target: JobStatus) -> bool:
    """Return whether a repository may conditionally commit this lifecycle edge."""
    return target in {
        JobStatus.QUEUED: {
            JobStatus.RUNNING,
            JobStatus.EXPIRED,
            JobStatus.SUPERSEDED,
            JobStatus.DEAD_LETTER,
        },
        JobStatus.RUNNING: {
            JobStatus.QUEUED,
            JobStatus.SUCCEEDED,
            JobStatus.FAILED,
            JobStatus.EXPIRED,
            JobStatus.SUPERSEDED,
            JobStatus.DEAD_LETTER,
        },
    }.get(current, set())


class JobRow(ServiceBase):
    __tablename__ = "service_jobs"
    __table_args__ = (
        UniqueConstraint(
            "owner_subject",
            "owner_tenant",
            "operation",
            "idempotency_key",
            name="uq_job_idempotency",
        ),
    )

    job_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    request_id: Mapped[str | None] = mapped_column(String(255))
    owner_subject: Mapped[str] = mapped_column(String(255), nullable=False)
    owner_tenant: Mapped[str] = mapped_column(String(255), nullable=False, default="")
    operation: Mapped[str] = mapped_column(String(32), nullable=False)
    idempotency_key: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default=JobStatus.QUEUED.value)
    payload_version: Mapped[int] = mapped_column(Integer, nullable=False)
    payload_json: Mapped[str] = mapped_column(Text, nullable=False)
    payload_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    reference_dataset_fingerprint: Mapped[str] = mapped_column(String(255), nullable=False)
    accepted_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    max_retries: Mapped[int] = mapped_column(Integer, nullable=False)
    attempt: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    generation_epoch: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    lease_token: Mapped[str | None] = mapped_column(String(36))
    lease_expires_at: Mapped[datetime | None] = mapped_column(DateTime)
    public_error_code: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )


class OutboxEventRow(ServiceBase):
    __tablename__ = "service_outbox_events"

    event_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    job_id: Mapped[str] = mapped_column(String(36), nullable=False, unique=True, index=True)
    operation: Mapped[str] = mapped_column(String(32), nullable=False)
    payload_version: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )
    dispatched_at: Mapped[datetime | None] = mapped_column(DateTime)


class JobEffectRow(ServiceBase):
    __tablename__ = "service_job_effects"
    __table_args__ = (UniqueConstraint("job_id", "effect_kind", "ordinal", name="uq_job_effect"),)

    effect_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    job_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    effect_kind: Mapped[str] = mapped_column(String(64), nullable=False)
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    result_json: Mapped[str] = mapped_column(Text, nullable=False)
    committed_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )
