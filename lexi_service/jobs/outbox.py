"""Outbox polling and acknowledgement; database remains delivery authority."""

from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from lexi_service.jobs.models import OutboxEventRow


@dataclass(frozen=True)
class OutboxEnvelope:
    event_id: str
    job_id: str
    operation: str
    payload_version: int


class SqlOutbox:
    def __init__(self, sessions: async_sessionmaker[AsyncSession]):
        self._sessions = sessions

    async def pending(self, limit: int) -> list[OutboxEnvelope]:
        async with self._sessions() as session:
            rows = list(
                await session.scalars(
                    select(OutboxEventRow)
                    .where(OutboxEventRow.dispatched_at.is_(None))
                    .order_by(OutboxEventRow.created_at)
                    .limit(limit)
                )
            )
        return [
            OutboxEnvelope(row.event_id, row.job_id, row.operation, row.payload_version)
            for row in rows
        ]

    async def acknowledge(self, event_id: str) -> None:
        async with self._sessions.begin() as session:
            row = await session.get(OutboxEventRow, event_id)
            if row is not None and row.dispatched_at is None:
                row.dispatched_at = datetime.now(timezone.utc).replace(tzinfo=None)
