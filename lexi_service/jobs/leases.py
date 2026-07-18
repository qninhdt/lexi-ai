"""Conditional job leasing and recovery for Redis's at-least-once delivery."""

from datetime import datetime, timedelta
from uuid import uuid4

from sqlalchemy import and_, case, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from lexi_service.jobs.models import TERMINAL_JOB_STATUSES, JobRow, JobStatus


class JobLeaseRepository:
    def __init__(self, sessions: async_sessionmaker[AsyncSession]):
        self._sessions = sessions

    async def claim(
        self, job_id: str, now: datetime, lease_for: timedelta
    ) -> tuple[str, int] | None:
        token = str(uuid4())
        async with self._sessions.begin() as session:
            changed = await session.execute(
                update(JobRow)
                .where(
                    JobRow.job_id == job_id,
                    JobRow.status == JobStatus.QUEUED.value,
                    JobRow.expires_at > now.replace(tzinfo=None),
                )
                .values(
                    status=JobStatus.RUNNING.value,
                    attempt=JobRow.attempt + 1,
                    generation_epoch=JobRow.generation_epoch + 1,
                    lease_token=token,
                    lease_expires_at=(now + lease_for).replace(tzinfo=None),
                )
            )
            row = await session.get(JobRow, job_id) if changed.rowcount == 1 else None
        return (token, row.generation_epoch) if row is not None else None

    async def should_ack(self, job_id: str) -> bool:
        """Acknowledge duplicate/missing stream deliveries only after terminal state."""
        async with self._sessions() as session:
            row = await session.get(JobRow, job_id)
        if row is None:
            return True
        return row.status in {status.value for status in TERMINAL_JOB_STATUSES}

    async def expire_queued(self, now: datetime) -> int:
        async with self._sessions.begin() as session:
            changed = await session.execute(
                update(JobRow)
                .where(
                    JobRow.status == JobStatus.QUEUED.value,
                    JobRow.expires_at <= now.replace(tzinfo=None),
                )
                .values(status=JobStatus.EXPIRED.value)
            )
        return changed.rowcount

    async def recover_expired(self, now: datetime) -> int:
        async with self._sessions.begin() as session:
            changed = await session.execute(
                update(JobRow)
                .where(
                    JobRow.status == JobStatus.RUNNING.value,
                    JobRow.lease_expires_at < now.replace(tzinfo=None),
                )
                .values(status=JobStatus.QUEUED.value, lease_token=None, lease_expires_at=None)
            )
        return changed.rowcount

    async def complete(self, job_id: str, token: str, epoch: int) -> bool:
        async with self._sessions.begin() as session:
            changed = await session.execute(
                update(JobRow)
                .where(
                    and_(
                        JobRow.job_id == job_id,
                        JobRow.status == JobStatus.RUNNING.value,
                        JobRow.lease_token == token,
                        JobRow.generation_epoch == epoch,
                    )
                )
                .values(
                    status=JobStatus.SUCCEEDED.value,
                    public_error_code=None,
                    lease_token=None,
                    lease_expires_at=None,
                )
            )
        return changed.rowcount == 1

    async def fail(self, job_id: str, token: str, epoch: int, error_code: str) -> bool:
        async with self._sessions.begin() as session:
            changed = await session.execute(
                update(JobRow)
                .where(
                    and_(
                        JobRow.job_id == job_id,
                        JobRow.status == JobStatus.RUNNING.value,
                        JobRow.lease_token == token,
                        JobRow.generation_epoch == epoch,
                    )
                )
                .values(
                    status=JobStatus.FAILED.value,
                    public_error_code=error_code,
                    lease_token=None,
                    lease_expires_at=None,
                )
            )
        return changed.rowcount == 1

    async def dead_letter(self, job_id: str, token: str, epoch: int, error_code: str) -> bool:
        """Terminally retain a job that no deployed worker can interpret."""
        async with self._sessions.begin() as session:
            changed = await session.execute(
                update(JobRow)
                .where(
                    and_(
                        JobRow.job_id == job_id,
                        JobRow.status == JobStatus.RUNNING.value,
                        JobRow.lease_token == token,
                        JobRow.generation_epoch == epoch,
                    )
                )
                .values(
                    status=JobStatus.DEAD_LETTER.value,
                    public_error_code=error_code,
                    lease_token=None,
                    lease_expires_at=None,
                )
            )
        return changed.rowcount == 1

    async def retry_or_dead_letter(
        self, job_id: str, token: str, epoch: int, error_code: str
    ) -> str | None:
        """Release a retryable lease, or terminate it once its retry budget is spent."""
        async with self._sessions.begin() as session:
            changed = await session.execute(
                update(JobRow)
                .where(
                    and_(
                        JobRow.job_id == job_id,
                        JobRow.status == JobStatus.RUNNING.value,
                        JobRow.lease_token == token,
                        JobRow.generation_epoch == epoch,
                    )
                )
                .values(
                    status=case(
                        (JobRow.attempt <= JobRow.max_retries, JobStatus.QUEUED.value),
                        else_=JobStatus.DEAD_LETTER.value,
                    ),
                    public_error_code=case(
                        (JobRow.attempt <= JobRow.max_retries, None), else_=error_code
                    ),
                    lease_token=None,
                    lease_expires_at=None,
                )
            )
            if changed.rowcount != 1:
                return None
            row = await session.get(JobRow, job_id)
            return row.status if row is not None else None
