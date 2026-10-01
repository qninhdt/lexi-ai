"""Reading the dictionary: entries, senses, browse listings, and counts.

Nothing here calls a provider. Generation lives in its own service precisely so a read path cannot
accidentally reach a language model.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

from lexi_ai.dictionary.batch import gather_batch
from lexi_ai.errors import UnknownTheme
from lexi_ai.models import BatchResult, Entry, SearchResult, SenseView, Stats, TagCount
from lexi_ai.text import render, tag_key
from lexi_ai.text import theme_key as normalize_theme_key

if TYPE_CHECKING:
    from lexi_ai.db.session import SqlAlchemyUnitOfWork

# Ceilings on the batch reads, which a caller supplies the size of.
#
# `entries` and `statuses` issue one query per id, so an unbounded list is an
# unbounded fan-out from a single call. `senses` is one query, but its `IN` list
# grows with the caller's input and Postgres plans a large one poorly. Both exist
# so one malformed request cannot become a thousand concurrent statements.
MAX_BATCH_IDS = 500
# Concurrent in-flight queries within one batch: 500 simultaneous reads exhaust the
# pool and starve every other caller.
BATCH_CONCURRENCY = 16


def _bounded(ids: Sequence[int], surface: str) -> list[int]:
    """The caller's ids, refused rather than truncated when there are too many.

    Truncating would answer a 600-id request with 500 results in order, which the caller reads as
    "the last hundred do not exist".
    """
    if len(ids) > MAX_BATCH_IDS:
        raise ValueError(
            f"{surface} takes at most {MAX_BATCH_IDS} ids, got {len(ids)}; "
            "page the request rather than widening this bound"
        )
    return list(ids)


class DictionaryService:
    """Read use cases over the unit of work."""

    def __init__(self, uow_factory: Callable[[], SqlAlchemyUnitOfWork]) -> None:
        self._uow = uow_factory

    async def entry(self, word_id: int, theme: str | int | None = None) -> Entry | None:
        """One entry by id, or ``None`` when the id is unknown.

        An unknown theme still raises rather than quietly returning the neutral entry: a missing
        word is ordinary, a theme the caller named that does not exist is a bug in the call.
        """
        theme_id = None
        if theme is not None:
            theme_id, _style = await self._resolve_theme_id(theme)
        return await self.entry_by_theme_id(word_id, theme_id)

    async def _resolve_theme_id(self, theme: str | int) -> tuple[int, str]:
        """Resolve a theme key or id, raising for an unknown one.

        Reads through this service's own unit of work: calling the theme service would close an
        import cycle at construction.
        """
        async with self._uow() as uow:
            resolved = await uow.themes.resolve(theme)
            if resolved is None and isinstance(theme, str):
                resolved = await uow.themes.resolve(normalize_theme_key(theme))
        if resolved is None:
            raise UnknownTheme(f"unknown theme: {theme!r}")
        return resolved

    async def entry_by_theme_id(self, word_id: int, theme_id: int | None = None) -> Entry | None:
        """One entry by id, overlaid with an ALREADY RESOLVED theme id, so the generation path
        (which resolves the theme for its style prompt anyway) skips a redundant resolve round
        trip.
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
        """Batch entry reads; an unknown id is reported, not raised. The miss is raised here so
        `gather_batch` can classify it — returning `None` would be reported as a *successful*
        result.
        """

        async def _one(word_id: int) -> Entry:
            found = await self.entry(word_id, theme=theme)
            if found is None:
                raise KeyError(f"no entry for word id {word_id}")
            return found

        return await gather_batch(
            _bounded(word_ids, "entries"), _one, concurrency=BATCH_CONCURRENCY
        )

    async def senses(self, sense_ids: Sequence[int]) -> list[SenseView]:
        """Views for the given senses, in order. Unknown ids are skipped."""
        if not sense_ids:
            return []
        async with self._uow() as uow:
            return await uow.entries.sense_views(_bounded(sense_ids, "senses"))

    async def word_id_for(self, sense_id: int) -> int | None:
        """The owning word id, or ``None`` when the sense is gone."""
        async with self._uow() as uow:
            return await uow.senses.word_id_for(sense_id)

    async def status(self, word_id: int) -> str | None:
        """Lifecycle status of a word, or ``None`` when the id is unknown."""
        async with self._uow() as uow:
            return await uow.words.status(word_id)

    async def statuses(self, word_ids: Sequence[int]) -> list[BatchResult]:
        """Batch status reads. ``None`` is a valid answer, not a failure."""

        async def _one(word_id: int) -> str | None:
            return await self.status(word_id)

        return await gather_batch(
            _bounded(word_ids, "statuses"), _one, concurrency=BATCH_CONCURRENCY
        )

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
        """Generated words carrying a tag, normalized with the same function the write path uses so
        casing and plural variants resolve to the same tag.
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
        """A browse row as an already-generated search hit. Display is always rendered from the
        lemma; there is no display column.
        """
        return SearchResult(
            display=render(row.norm), entry_type=row.entry_type, lexi_word_id=row.word_id
        )
