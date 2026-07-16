"""Durable effect checkpoints prevent replay from repeating committed work."""

import json
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from lexi_service.jobs.models import JobEffectRow


class JobEffects:
    def __init__(self, sessions: async_sessionmaker[AsyncSession]):
        self._sessions = sessions

    async def adopt_or_record(
        self, job_id: str, kind: str, result: object, ordinal: int = 0
    ) -> object:
        async with self._sessions.begin() as session:
            existing = await session.scalar(
                select(JobEffectRow).where(
                    JobEffectRow.job_id == job_id,
                    JobEffectRow.effect_kind == kind,
                    JobEffectRow.ordinal == ordinal,
                )
            )
            if existing is not None:
                return json.loads(existing.result_json)
            try:
                async with session.begin_nested():
                    session.add(
                        JobEffectRow(
                            effect_id=str(uuid4()),
                            job_id=job_id,
                            effect_kind=kind,
                            ordinal=ordinal,
                            result_json=json.dumps(result, sort_keys=True),
                        )
                    )
                    await session.flush()
            except IntegrityError:
                adopted = await session.scalar(
                    select(JobEffectRow).where(
                        JobEffectRow.job_id == job_id,
                        JobEffectRow.effect_kind == kind,
                        JobEffectRow.ordinal == ordinal,
                    )
                )
                if adopted is None:
                    raise
                return json.loads(adopted.result_json)
            return result

    async def get(self, job_id: str, kind: str, ordinal: int = 0) -> object | None:
        async with self._sessions() as session:
            existing = await session.scalar(
                select(JobEffectRow).where(
                    JobEffectRow.job_id == job_id,
                    JobEffectRow.effect_kind == kind,
                    JobEffectRow.ordinal == ordinal,
                )
            )
        return None if existing is None else json.loads(existing.result_json)
