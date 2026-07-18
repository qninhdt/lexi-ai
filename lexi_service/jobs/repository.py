"""SQLAlchemy job/outbox repository; Redis dispatch is intentionally separate."""

import json
from datetime import timedelta, timezone
from hashlib import sha256
from uuid import uuid4

from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from lexi_service.application.commands import JobReference, JobSubmission
from lexi_service.application.errors import ErrorCode, public_error
from lexi_service.identity import Principal
from lexi_service.jobs.models import JobEffectRow, JobRow, OutboxEventRow
from lexi_service.ports import JobRecord


def canonical_payload_hash(submission: JobSubmission) -> str:
    value = json.dumps(
        {
            "operation": submission.operation,
            "version": submission.payload_version,
            "payload": submission.payload,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256(value.encode()).hexdigest()


class SqlJobRepository:
    def __init__(
        self,
        sessions: async_sessionmaker[AsyncSession],
        max_outstanding_jobs_per_owner: int | None = None,
    ):
        self._sessions = sessions
        self._max_outstanding_jobs_per_owner = max_outstanding_jobs_per_owner

    @property
    def sessions(self) -> async_sessionmaker[AsyncSession]:
        return self._sessions

    async def publish(self, submission: JobSubmission) -> JobReference:
        tenant = submission.owner.tenant or ""
        payload_hash = canonical_payload_hash(submission)
        async with self._sessions.begin() as session:
            # PostgreSQL service deployments serialize publish decisions for one
            # authenticated owner. This makes the idempotency lookup and
            # outstanding-job quota one atomic decision across API replicas.
            # Hash collisions merely serialize unrelated owners; they cannot
            # merge ownership or weaken the database predicates below.
            bind = session.get_bind()
            if bind.dialect.name == "postgresql":
                await session.execute(
                    text("SELECT pg_advisory_xact_lock(hashtext(:owner_key))"),
                    {
                        "owner_key": sha256(
                            json.dumps([submission.owner.subject, tenant]).encode()
                        ).hexdigest()
                    },
                )
            existing = await session.scalar(
                select(JobRow).where(
                    JobRow.owner_subject == submission.owner.subject,
                    JobRow.owner_tenant == tenant,
                    JobRow.operation == submission.operation,
                    JobRow.idempotency_key == submission.idempotency_key,
                )
            )
            if existing is not None:
                if existing.payload_hash != payload_hash:
                    raise ValueError("idempotency key is already bound to a different payload")
                return JobReference(existing.job_id, existing.status, deduplicated=True)
            if self._max_outstanding_jobs_per_owner is not None:
                outstanding = await session.scalar(
                    select(func.count())
                    .select_from(JobRow)
                    .where(
                        JobRow.owner_subject == submission.owner.subject,
                        JobRow.owner_tenant == tenant,
                        JobRow.status.in_(("queued", "running")),
                    )
                )
                if outstanding >= self._max_outstanding_jobs_per_owner:
                    raise public_error(ErrorCode.CONFLICT, "Outstanding job quota exceeded.")
            job_id = str(uuid4())
            session.add(
                JobRow(
                    job_id=job_id,
                    request_id=submission.request_id,
                    owner_subject=submission.owner.subject,
                    owner_tenant=tenant,
                    operation=submission.operation,
                    idempotency_key=submission.idempotency_key,
                    payload_version=submission.payload_version,
                    payload_json=json.dumps(submission.payload, sort_keys=True),
                    payload_hash=payload_hash,
                    reference_dataset_fingerprint=submission.reference_dataset_fingerprint,
                    accepted_at=submission.accepted_at.replace(tzinfo=None),
                    expires_at=(
                        submission.accepted_at + timedelta(seconds=submission.maximum_age_seconds)
                    ).replace(tzinfo=None),
                    max_retries=submission.max_retries,
                )
            )
            session.add(
                OutboxEventRow(
                    event_id=str(uuid4()),
                    job_id=job_id,
                    operation=submission.operation,
                    payload_version=submission.payload_version,
                )
            )
            return JobReference(job_id)

    async def get(self, job_id: str) -> JobRecord | None:
        async with self._sessions() as session:
            row = await session.get(JobRow, job_id)
            if row is None:
                return None
            effect = await session.scalar(
                select(JobEffectRow)
                .where(JobEffectRow.job_id == job_id, JobEffectRow.effect_kind == row.operation)
                .order_by(JobEffectRow.ordinal)
            )
            return JobRecord(
                JobReference(row.job_id, row.status),
                Principal(row.owner_subject, row.owner_tenant or None),
                row.operation,
                None if effect is None else json.loads(effect.result_json),
                row.public_error_code,
            )

    async def load_submission(self, job_id: str) -> tuple[JobSubmission, int] | None:
        async with self._sessions() as session:
            row = await session.get(JobRow, job_id)
            if row is None:
                return None
            submission = JobSubmission(
                operation=row.operation,
                request_id=row.request_id or "worker-replay",
                owner=Principal(row.owner_subject, row.owner_tenant or None),
                idempotency_key=row.idempotency_key,
                payload_version=row.payload_version,
                reference_dataset_fingerprint=row.reference_dataset_fingerprint,
                accepted_at=row.accepted_at.replace(tzinfo=timezone.utc),
                payload=json.loads(row.payload_json),
                maximum_age_seconds=max(0, int((row.expires_at - row.accepted_at).total_seconds())),
                max_retries=row.max_retries,
            )
            return submission, row.attempt
