"""Cross-aggregate dictionary counts.

Read through one session so the numbers are mutually consistent: per-aggregate calls would let
concurrent writes land between the reads and produce a snapshot that never existed.
"""

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from lexi_ai.db.schema import (
    Asset,
    Example,
    Question,
    Sense,
    Tag,
    Theme,
    ThemedSense,
    Word,
)
from lexi_ai.models import Stats


class StatsQueries:
    """Cross-aggregate counts, deliberately read in one snapshot."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def snapshot(self) -> Stats:
        """Point-in-time counts with no N+1 and no LLM call.

        ``themed_words`` counts distinct words carrying at least one themed overlay.
        """
        status_rows = await self._session.execute(
            select(Word.status, func.count(Word.id)).group_by(Word.status)
        )
        words_by_status = dict(status_rows.all())
        asset_rows = await self._session.execute(
            select(Asset.kind, func.count(Asset.id)).group_by(Asset.kind)
        )
        assets_by_kind = dict(asset_rows.all())
        return Stats(
            words_by_status=words_by_status,
            senses=await self._count(Sense.id),
            examples=await self._count(Example.id),
            tags=await self._count(Tag.id),
            themes=await self._count(Theme.id),
            themed_words=(
                await self._session.execute(
                    select(func.count(func.distinct(Sense.word_id)))
                    .select_from(ThemedSense)
                    .join(Sense, Sense.id == ThemedSense.sense_id)
                )
            ).scalar_one(),
            assets_by_kind=assets_by_kind,
            questions=await self._count(Question.id),
        )

    async def _count(self, column) -> int:  # noqa: ANN001 - any mapped id column
        return (await self._session.execute(select(func.count(column)))).scalar_one()
