"""Generating entries on demand, once per word.

Three guards stack, each covering what the others cannot: a per-key single-flight lock collapses
concurrent callers in THIS process; a database re-check inside that lock lets a waiter adopt the
winner's result; and for independently deployed workers a database fence claims the word before any
provider call, since separate processes share no lock.

The fenced entry point deliberately has no force flag: a remote caller must not be able to use a
delayed job to overwrite an entry a newer claim already owns.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

from lexi_ai.dictionary.batch import gather_batch
from lexi_ai.dictionary.single_flight import SingleFlight
from lexi_ai.dictionary.writer import GenerationWriter
from lexi_ai.models import BatchResult, Entry, SearchResult
from lexi_ai.text import match_key
from lexi_ai.vocab import canonical_cambridge_ref

if TYPE_CHECKING:
    from lexi_ai.db.session import SqlAlchemyUnitOfWork


class GenerationService:
    """Generation use cases over the writer, the loader, and the enrichment hooks."""

    def __init__(
        self,
        uow_factory: Callable[[], SqlAlchemyUnitOfWork],
        writer: GenerationWriter,
        loader,  # noqa: ANN001 - the reference loader
        generator,  # noqa: ANN001 - the LLM-backed entry generator
        read_entry: Callable[..., object],
        locks: SingleFlight,
        theme_locks: SingleFlight,
        resolve_theme: Callable[[str | int], object],
        restyle_word: Callable[[int, int, str], object],
        resolve_inbound: Callable[[Sequence[int]], object],
    ) -> None:
        self._uow = uow_factory
        self._writer = writer
        self._loader = loader
        self._generator = generator
        self._read_entry = read_entry
        self._locks = locks
        self._theme_locks = theme_locks
        self._resolve_theme = resolve_theme
        self._restyle_word = restyle_word
        self._resolve_inbound = resolve_inbound

    async def generate(
        self,
        source: SearchResult | str,
        *,
        force: bool = False,
        theme: str | int | None = None,
        structured_method: str | None = None,
    ) -> Entry:
        """Generate or return the entry for a search hit or a custom string. A suggestion whose word
        already exists converges on that entry; ``force`` regenerates and overwrites in place.
        """
        entry = await self._resolve_source(source, force, structured_method)
        if theme is None:
            return entry
        return await self._apply_theme(entry, theme, force)

    async def generate_many(
        self,
        sources: Sequence[SearchResult | str],
        *,
        force: bool = False,
        theme: str | int | None = None,
        concurrency: int = 5,
    ) -> list[BatchResult]:
        """Batch generation. Items route through the same single-flight path, so two inputs
        resolving to one word still generate exactly once.
        """

        async def _one(source: SearchResult | str) -> Entry:
            return await self.generate(source, force=force, theme=theme)

        return await gather_batch(list(sources), _one, concurrency=concurrency)

    async def generate_fenced(
        self, source: SearchResult | str, *, structured_method: str | None = None
    ) -> Entry:
        """Generate once under a database fence, for workers in another process."""
        key, word, cambridge_id = await self._anchor(source)
        async with self._locks.hold(key):
            done = await self._done_ids(key)
            if done:
                return await self._read_entry(done[0])
            # The claim commits before the provider call; that is what makes it
            # visible to a competing worker.
            fence = await self._writer.claim(word)
            result = await self._run(word, cambridge_id, fence=fence, method=structured_method)
        return await self._entry_for_key(key, result)

    async def _resolve_source(
        self, source: SearchResult | str, force: bool, method: str | None
    ) -> Entry:
        """Route one input to an existing entry or a fresh generation. A forced run skips the
        existence shortcuts and always reaches the lock.
        """
        if not force and isinstance(source, SearchResult):
            existing = await self._existing_entry(source)
            if existing is not None:
                return existing
        key, word, cambridge_id = await self._anchor(source)
        return await self._locked(key, word, cambridge_id, force, method)

    async def _existing_entry(self, source: SearchResult) -> Entry | None:
        """An entry already persisted for this input; ``None`` means generate it."""
        if source.lexi_word_id is not None:
            return await self._read_entry(source.lexi_word_id)
        if source.cambridge_id is None:
            return None
        async with self._uow() as uow:
            hits = await uow.words.generated_by_cambridge([source.cambridge_id])
        existing = hits.get(source.cambridge_id)
        if existing is None:
            return None
        return await self._read_entry(existing.word_id)

    async def _locked(
        self, key: str, word: str, cambridge_id: int | None, force: bool, method: str | None
    ) -> Entry:
        """Generate under the per-key lock; the re-check inside lets a waiter adopt the winner's
        entry instead of duplicating the work.
        """
        async with self._locks.hold(key):
            if not force:
                done = await self._done_ids(key)
                if done:
                    return await self._read_entry(done[0])
            result = await self._run(word, cambridge_id, fence=None, method=method)
        return await self._entry_for_key(key, result)

    async def _apply_theme(self, entry: Entry, theme: str | int, force: bool) -> Entry:
        """Restyle the entry in a theme, at most once per (word, theme).

        The provider call happens before the overlay is written, so without the lock and its
        re-check two concurrent callers would both see no overlay and both call the model. Separate
        from the generation locks, so the two cannot nest.
        """
        theme_id, style_prompt = await self._resolve_theme(theme)
        async with self._theme_locks.hold((entry.word_id, theme_id)):
            async with self._uow() as uow:
                overlay = await uow.themes.overlay_for_word(entry.word_id, theme_id)
            if not overlay or force:
                await self._restyle_word(entry.word_id, theme_id, style_prompt)
        return await self._read_entry(entry.word_id, theme_id)

    async def _anchor(self, source: SearchResult | str) -> tuple[str, str, int | None]:
        """The lookup key, lemma, and reference id for one input — the one dispatch over the two
        input shapes, shared by the locked and fenced paths.
        """
        if isinstance(source, str):
            return match_key(source), source, None
        if source.lexi_word_id is not None:
            async with self._uow() as uow:
                norm, cambridge_id = await uow.words.norm_and_cambridge(source.lexi_word_id)
            return match_key(norm), norm, cambridge_id
        if source.cambridge_id is not None:
            return match_key(source.display), source.display, source.cambridge_id
        raise ValueError("SearchResult has neither lexi_word_id nor cambridge_id")

    async def _run(self, word: str, cambridge_id: int | None, *, fence, method: str | None):  # noqa: ANN001, ANN202 - returns the provider's GeneratedResult
        """Build the reference bundle, generate, publish, then enrich.

        Enrichment runs AFTER the publish commits: relation resolution is best-effort, and inside
        the write transaction a failure in it would roll back a good entry.
        """
        bundle, cefr_map = await self._bundle(word, cambridge_id)
        existing_tags = await self._existing_tags()
        if method is None:
            result = await self._generator.generate(bundle, existing_tags=existing_tags)
        else:
            result = await self._generator.generate(
                bundle, existing_tags=existing_tags, structured_method=method
            )
        words = await self._writer.publish(
            result, cambridge_word_id=cambridge_id, cambridge_cefr=cefr_map, fence=fence
        )
        word_ids = [record.id for record in words]
        await self._resolve_inbound(word_ids)
        return result

    async def _bundle(self, word: str, cambridge_id: int | None):  # noqa: ANN202
        if cambridge_id is None:
            return await self._loader.bundle_custom(word), {}
        bundle = await self._loader.bundle_by_id(cambridge_id)
        if bundle is None:
            raise ValueError(f"Cambridge word_id {cambridge_id} not found")
        return bundle, self._cefr_map(bundle)

    async def _existing_tags(self) -> list:
        """The topic vocabulary to show the model; best-effort by design: failing to read it costs
        tag reuse, not the generation itself.
        """
        try:
            async with self._uow() as uow:
                return list(await uow.tags.names())
        except Exception:  # noqa: BLE001 - vocab is best-effort; empty on failure
            return []

    async def _done_ids(self, key: str) -> list[int]:
        """Word ids already generated for this key, via headword or alias."""
        async with self._uow() as uow:
            matches = await uow.words.resolve_key(key)
        return [match.word_id for match in matches if match.status == "done"]

    async def _entry_for_key(self, key: str, result) -> Entry:  # noqa: ANN001
        """The entry for the key just generated, falling back to the first unit.

        The model may normalize a lemma differently from the query, so the queried key can miss even
        though the generation succeeded.
        """
        done = await self._done_ids(key)
        if done:
            return await self._read_entry(done[0])
        fallback = await self._done_ids(match_key(result.units[0].norm))
        if not fallback:
            async with self._uow() as uow:
                matches = await uow.words.resolve_key(match_key(result.units[0].norm))
            if not matches:
                raise ValueError(
                    f"no persisted entry found for key {key!r} or "
                    f"first-unit norm {result.units[0].norm!r} — generation may have errored"
                )
            return await self._read_entry(matches[0].word_id)
        return await self._read_entry(fallback[0])

    @staticmethod
    def _cefr_map(bundle) -> dict[str, str]:  # noqa: ANN001 - a ReferenceBundle
        """Reference sense id to CEFR level, for the Cambridge-first rule. Keys are canonicalized so
        they match whatever form the model echoes back.
        """
        return {
            canonical_cambridge_ref(str(sense.cambridge_sense_id)): sense.cefr_level
            for sense in bundle.cambridge_senses
            if sense.cefr_level
        }
