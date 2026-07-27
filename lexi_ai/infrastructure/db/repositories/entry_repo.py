"""Read-model reads: whole entries and sense views.

These are the queries whose only job is to build a public read model, so they need
eager loading rather than the column projections the aggregate repositories use.
They live in their own module because the loading strategy IS the implementation:
every relationship a mapper touches must be loaded here, or the mapper would
lazy-load after the session closed and raise outside greenlet context.

Keeping them behind a port is what removes the application layer's second
persistence path — it previously issued these selects itself.
"""

from collections.abc import Sequence

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from lexi_ai.infrastructure.db.mappers import ThemedOverlay, entry_view, sense_view
from lexi_ai.infrastructure.db.models import Sense, SenseRelation, Word, WordRelation, WordTag
from lexi_ai.read_models import Entry, SenseView


def _sense_loads(load):  # noqa: ANN001 - a loader factory (selectinload or a nested one)
    """Eager-load everything a sense view reads, including relation targets.

    ``load`` is the loader to apply, so the same list serves a standalone sense
    query and the nested load under a word.
    """
    return [
        load(Sense.references),
        load(Sense.examples),
        load(Sense.collocations),
        load(Sense.forms),
        # The edge plus its target word (always present) and target sense (present
        # once resolved); the view reads all three.
        load(Sense.relations_out).selectinload(SenseRelation.to_word),
        load(Sense.relations_out).selectinload(SenseRelation.to_sense),
    ]


class SqlEntryRepo:
    """Session-bound implementation of :class:`lexi_ai.domain.ports.EntryRepo`."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def entry(self, word_id: int, overlay: ThemedOverlay | None = None) -> Entry:
        """The full entry for one word. Raises when the id is unknown."""
        word = (
            await self._session.execute(
                select(Word)
                .options(
                    *_sense_loads(selectinload(Word.senses).selectinload),
                    selectinload(Word.aliases),
                    selectinload(Word.links_out).selectinload(WordRelation.to_word),
                    selectinload(Word.tags).selectinload(WordTag.tag),
                )
                .where(Word.id == word_id)
            )
        ).scalar_one()
        return entry_view(word, overlay)

    async def sense_views(self, sense_ids: Sequence[int]) -> list[SenseView]:
        """Views for the given senses, in the order requested.

        An unknown id is skipped rather than raising: callers tolerate a sense that
        was regenerated away between reads.
        """
        if not sense_ids:
            return []
        rows = (
            await self._session.execute(
                select(Sense)
                # The parent word, for the headword the view carries. Added here
                # and NOT in ``_sense_loads``, because that list is also applied
                # nested under ``Word.senses`` in ``entry``, where the word is
                # already loaded and this would be redundant work.
                .options(*_sense_loads(selectinload), selectinload(Sense.word))
                .where(Sense.id.in_(sense_ids))
            )
        ).scalars()
        by_id = {sense.id: sense_view(sense) for sense in rows}
        return [by_id[sense_id] for sense_id in sense_ids if sense_id in by_id]
