"""Reading the dictionary: entries, senses, browse listings, and counts.

Everything here is free — no provider is ever called, so these are the surfaces a
reader process can serve. Generation lives in its own service precisely so a read
path cannot accidentally reach a language model.
"""

from collections.abc import Callable, Sequence

from lexi_ai.application.batching import gather_batch
from lexi_ai.domain.ports import UnitOfWork
from lexi_ai.normalize import render, tag_key
from lexi_ai.read_models import BatchResult, Entry, SearchResult, SenseView, Stats, TagCount


class DictionaryService:
    """Read use cases over the unit of work."""

    def __init__(
        self,
        uow_factory: Callable[[], UnitOfWork],
        resolve_theme: Callable[[str | int], object],
    ) -> None:
        self._uow = uow_factory
        # Themed reads need a theme resolved to its id first. Taking the resolver
        # keeps this service independent of the theme service.
        self._resolve_theme = resolve_theme

    async def entry(self, word_id: int, theme: str | int | None = None) -> Entry:
        """One entry by id, optionally overlaid with a theme.

        An unknown theme raises rather than quietly returning the neutral entry,
        which would hide a caller's mistake.
        """
        theme_id = None
        if theme is not None:
            theme_id, _style = await self._resolve_theme(theme)
        return await self.entry_by_theme_id(word_id, theme_id)

    async def entry_by_theme_id(self, word_id: int, theme_id: int | None = None) -> Entry:
        """One entry by id, overlaid with an ALREADY RESOLVED theme id.

        The generation path resolves the theme itself (it needs the style prompt
        anyway), so it enters here and skips a redundant resolve round trip.
        """
        async with self._uow() as uow:
            overlay = (
                await uow.themes.overlay_for_word(word_id, theme_id)
                if theme_id is not None
                else None
            )
            return await uow.entries.entry(word_id, overlay)

    async def entries(
        self, word_ids: Sequence[int], theme: str | int | None = None
    ) -> list[BatchResult]:
        """Batch entry reads; an unknown id is reported, not raised."""

        async def _one(word_id: int) -> Entry:
            return await self.entry(word_id, theme=theme)

        return await gather_batch(list(word_ids), _one)

    async def senses(self, sense_ids: Sequence[int]) -> list[SenseView]:
        """Views for the given senses, in order. Unknown ids are skipped."""
        if not sense_ids:
            return []
        async with self._uow() as uow:
            return await uow.entries.sense_views(list(sense_ids))

    async def status(self, word_id: int) -> str | None:
        """Lifecycle status of a word, or ``None`` when the id is unknown."""
        async with self._uow() as uow:
            return await uow.words.status(word_id)

    async def statuses(self, word_ids: Sequence[int]) -> list[BatchResult]:
        """Batch status reads. ``None`` is a valid answer, not a failure."""

        async def _one(word_id: int) -> str | None:
            return await self.status(word_id)

        return await gather_batch(list(word_ids), _one)

    async def list_entries(
        self, *, status: str = "done", limit: int | None = None, offset: int = 0
    ) -> list[SearchResult]:
        """Paginated browse of the whole dictionary, norm-sorted."""
        async with self._uow() as uow:
            rows = await uow.words.listing(status=status, limit=limit, offset=offset)
        return [self._generated_hit(row) for row in rows]

    async def list_entries_by_tag(
        self, tag: str, *, limit: int | None = None
    ) -> list[SearchResult]:
        """Generated words carrying a tag.

        The query is normalized with the same function the write path uses, so
        casing and plural variants all resolve to the same tag.
        """
        async with self._uow() as uow:
            rows = await uow.tags.words_for_key(tag_key(tag), limit=limit)
        return [self._generated_hit(row) for row in rows]

    async def list_tags(self) -> list[TagCount]:
        """Every topic with its live member count, busiest first."""
        async with self._uow() as uow:
            rows = await uow.tags.usage()
        return [TagCount(name=row.name, title=row.title, count=row.count) for row in rows]

    async def delete_entry(self, word_id: int) -> bool:
        """Delete a word and everything under it; cascades handle the children."""
        async with self._uow() as uow:
            deleted = await uow.words.delete(word_id)
            await uow.commit()
            return deleted

    async def stats(self) -> Stats:
        """Point-in-time dictionary counts, read as one snapshot."""
        async with self._uow() as uow:
            return await uow.stats.snapshot()

    @staticmethod
    def _generated_hit(row) -> SearchResult:  # noqa: ANN001 - a WordListing
        """A browse row as an already-generated search hit.

        Display is always rendered from the lemma; there is no display column.
        """
        return SearchResult(
            display=render(row.norm), entry_type=row.entry_type, lexi_word_id=row.word_id
        )
