"""Garbage-collect cached assets for rows that are about to disappear.

Cached translations and audio clips have no foreign key to the sense, example, or
collocation they describe, so deleting content leaves their cache rows behind.
This runs on the CALLER's session so it rolls back together with the delete that
prompted it.

Bulk deletes never report which child rows they removed, so the ids are collected
BEFORE the delete. Missing one is inert: the read path verifies a content hash
before serving, so a stale row can never be mis-served. This is housekeeping to
reclaim rows and on-disk clips, not a correctness mechanism.
"""

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from lexi_ai.infrastructure.db.repositories.asset_repo import AssetRepository
from lexi_ai.infrastructure.db.models import Collocation, Example, Sense


async def collect_word_assets(
    session: AsyncSession, assets: AssetRepository | None, word_id: int
) -> None:
    """Delete cached assets for every sense, example, and collocation of a word."""
    if assets is None:
        return
    sense_ids = list(
        (await session.execute(select(Sense.id).where(Sense.word_id == word_id))).scalars().all()
    )
    if not sense_ids:
        return
    example_ids = list(
        (await session.execute(select(Example.id).where(Example.sense_id.in_(sense_ids))))
        .scalars()
        .all()
    )
    collocation_ids = list(
        (await session.execute(select(Collocation.id).where(Collocation.sense_id.in_(sense_ids))))
        .scalars()
        .all()
    )
    # Table-driven over source_kind so adding a kind adds a branch for free, and one
    # bulk delete per kind rather than per id to bound round trips.
    for source_kind, ids in (
        ("sense_def", sense_ids),
        ("example", example_ids),
        ("collocation", collocation_ids),
    ):
        await assets.delete_by_source_ids(session, source_kind, ids)
