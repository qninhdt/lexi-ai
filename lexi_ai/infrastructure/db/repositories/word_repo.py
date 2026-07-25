"""The ``words`` aggregate: identity, lifecycle status, aliases, word-level links.

This module and the sense repository are the only writers of ``match_key``, always
through :func:`lexi_ai.normalize.match_key`, so the write path and the read path
derive keys identically. If they diverge, lookups miss forever.
"""

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, cast

from sqlalchemy import CursorResult, delete, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from lexi_ai.assets.repository import AssetRepository
from lexi_ai.domain.models import GenerationFence, WordListing, WordRecord
from lexi_ai.infrastructure.db.asset_gc import collect_word_assets
from lexi_ai.infrastructure.db.mappers import word_record
from lexi_ai.infrastructure.db.models import Word, WordAlias, WordRelation
from lexi_ai.infrastructure.db.sanitize import MAX_NORM, clean
from lexi_ai.normalize import match_key

if TYPE_CHECKING:
    from lexi_ai.generation.schemas import GeneratedAlias, GeneratedEntry, RelatedWord


class SqlWordRepo:
    """Session-bound implementation of :class:`lexi_ai.domain.ports.WordRepo`."""

    def __init__(self, session: AsyncSession, assets: AssetRepository | None = None) -> None:
        self._session = session
        self._assets = assets

    async def get_or_create(self, norm: str) -> int:
        """Resolve a lemma to its word id, creating a pending stub when absent."""
        word = await self._get_or_create_row(norm)
        return word.id

    async def claim_next_epoch(self, norm: str) -> GenerationFence:
        """Bump the generation epoch and return the resulting ownership token.

        The caller MUST commit before starting provider work. The epoch bump is what
        competing workers check, so while it is uncommitted the fence is invisible
        and two workers can both believe they own the word.
        """
        key = match_key(norm)
        word = await self._get_or_create_row(norm, key=key)
        await self._session.execute(
            update(Word)
            .where(Word.id == word.id)
            .values(
                generation_epoch=Word.generation_epoch + 1,
                status="pending",
                error_msg=None,
            )
        )
        await self._session.flush()
        await self._session.refresh(word)
        return GenerationFence(key, word.generation_epoch)

    async def fence_is_current(self, fence: GenerationFence) -> bool:
        """Whether ``fence`` still owns its word, locking the row for the publish.

        ``FOR UPDATE`` serializes two publishes racing on the same epoch, so the
        loser blocks until the winner commits instead of overwriting it.
        """
        claimed = await self._session.execute(
            select(Word.id)
            .where(Word.match_key == fence.match_key, Word.generation_epoch == fence.epoch)
            .with_for_update()
        )
        return claimed.scalar_one_or_none() is not None

    async def upsert_core(self, entry: "GeneratedEntry", cambridge_word_id: int | None) -> int:
        """Insert or update one unit's own columns and return its word id.

        ``norm`` is untrusted model text landing in a ``Text`` column, so it is
        cleaned like every other free field. The key is already control-safe, so
        cleaning ``norm`` cannot desync the two.
        """
        key = match_key(entry.norm)
        norm = clean(entry.norm, MAX_NORM)
        word = await self._get_or_create_row(norm, key=key)
        word.norm = norm
        word.entry_type = entry.entry_type
        word.pos = entry.pos
        word.status = "done"
        word.error_msg = None
        if cambridge_word_id is not None:
            word.cambridge_word_id = cambridge_word_id
        await self._session.flush()
        return word.id

    async def sync_aliases(self, word_id: int, aliases: Iterable["GeneratedAlias"]) -> None:
        """Replace a word's aliases, deduped by key. Fully derived from generation."""
        await self._session.execute(delete(WordAlias).where(WordAlias.word_id == word_id))
        seen: set[str] = set()
        for alias in aliases:
            alias_key = match_key(alias.alias_norm)
            if alias_key in seen:
                continue
            seen.add(alias_key)
            self._session.add(
                WordAlias(
                    word_id=word_id,
                    alias_norm=clean(alias.alias_norm, MAX_NORM),
                    alias_match_key=alias_key,
                    type=alias.type,
                    dialect=alias.dialect,
                )
            )
        await self._session.flush()

    async def link_related(self, word_id: int, related: Iterable["RelatedWord"]) -> None:
        """Ensure the word-level relation edges for one unit, skipping self-links."""
        for relation in related:
            target_id = await self.get_or_create(relation.norm)
            if target_id == word_id:
                continue
            await self._ensure_link(word_id, target_id, relation.rel_type)
        await self._session.flush()

    async def mark_done(self, word_id: int) -> None:
        await self._session.execute(
            update(Word).where(Word.id == word_id).values(status="done", error_msg=None)
        )

    async def mark_error(
        self, norms: Sequence[str], message: str, fence: GenerationFence | None = None
    ) -> None:
        """Stamp every unit of a failed generation with ``status='error'``.

        This deliberately overwrites a previously ``done`` word. A transient
        regeneration failure therefore hides healthy content until it is
        regenerated or cleared. The alternative, skipping the flip for a done word,
        would keep it visible but could leave stale content live after a failed
        update. Fail-closed is the chosen trade-off; changing it is an owner
        decision.

        The fenced branch only writes while the epoch still matches, so a newer
        claim's word is left untouched.
        """
        truncated = message[:2000]
        for norm in norms:
            key = match_key(norm)
            word = await self._get(key)
            if fence is not None and key == fence.match_key:
                if word is None:
                    continue
                await self._session.execute(
                    update(Word)
                    .where(Word.id == word.id, Word.generation_epoch == fence.epoch)
                    .values(status="error", error_msg=truncated)
                )
                continue
            if word is None:
                word = Word(norm=clean(norm, MAX_NORM), match_key=key, status="error")
                self._session.add(word)
            word.status = "error"
            word.error_msg = truncated
            await self._session.flush()

    async def record(self, word_id: int) -> WordRecord:
        word = (await self._session.execute(select(Word).where(Word.id == word_id))).scalar_one()
        return word_record(word)

    async def records(self, word_ids: Sequence[int]) -> list[WordRecord]:
        """Detached records for ``word_ids``, in the order requested."""
        if not word_ids:
            return []
        rows = await self._session.execute(select(Word).where(Word.id.in_(word_ids)))
        by_id = {word.id: word_record(word) for word in rows.scalars()}
        return [by_id[word_id] for word_id in word_ids if word_id in by_id]

    async def done_keys(self) -> set[str]:
        """Every ``match_key`` already generated — the candidate diff."""
        rows = await self._session.execute(select(Word.match_key).where(Word.status == "done"))
        return set(rows.scalars().all())

    async def delete(self, word_id: int) -> bool:
        """Delete a word; return whether a row was removed.

        Senses, aliases, links, tags, and questions go through ``ON DELETE
        CASCADE``, so one Core delete suffices and no relationship walk is needed.
        Cached assets have no such foreign key, so they are collected first on this
        same session.

        Inbound sense-relation edges need no demotion here: ``to_word_id`` cascades,
        so deleting this word removes every edge pointing at it outright. Demotion
        is only for the regenerate path, where the word survives but its senses
        churn.
        """
        await collect_word_assets(self._session, self._assets, word_id)
        result = await self._session.execute(delete(Word).where(Word.id == word_id))
        return (cast("CursorResult", result).rowcount or 0) > 0

    async def listing(
        self, status: str = "done", limit: int | None = None, offset: int = 0
    ) -> list[WordListing]:
        """Paginated dictionary browse, norm-sorted, filtered by status."""
        stmt = (
            select(Word.id, Word.norm, Word.entry_type)
            .where(Word.status == status)
            .order_by(Word.norm)
            .offset(offset)
        )
        if limit is not None:
            stmt = stmt.limit(limit)
        rows = await self._session.execute(stmt)
        return [WordListing(wid, norm, entry_type) for wid, norm, entry_type in rows]

    async def seed_phrase_unit(
        self,
        phrase_title: str,
        host_display: str | None,
        entry_type: str | None,
        is_overlap: bool,
    ) -> None:
        """Seed one phrase title, reusing the stub and link paths.

        An orphan phrase gets a pending stub so it enters the lazy generation queue,
        which it would otherwise never join. An overlapping phrase additionally
        links host to unit. Idempotent: stubs dedup by key, links by their unique
        triple.
        """
        unit = await self._get_or_create_row(phrase_title)
        if unit.entry_type is None and entry_type is not None:
            unit.entry_type = entry_type
        if is_overlap and host_display:
            host = await self._get_or_create_row(host_display)
            if host.id != unit.id:
                await self._ensure_link(host.id, unit.id, "part_of_phrasal_family")
        await self._session.flush()

    async def _ensure_link(self, from_id: int, to_id: int, rel_type: str) -> None:
        """Insert a word relation when the (from, to, rel_type) triple is absent."""
        exists = await self._session.execute(
            select(WordRelation.id).where(
                WordRelation.from_word_id == from_id,
                WordRelation.to_word_id == to_id,
                WordRelation.rel_type == rel_type,
            )
        )
        if exists.first() is None:
            self._session.add(
                WordRelation(from_word_id=from_id, to_word_id=to_id, rel_type=rel_type)
            )

    async def _get_or_create_row(self, norm: str, key: str | None = None) -> Word:
        resolved = key if key is not None else match_key(norm)
        word = await self._get(resolved)
        if word is not None:
            return word
        return await self._insert(resolved, norm)

    async def _get(self, key: str) -> Word | None:
        result = await self._session.execute(select(Word).where(Word.match_key == key))
        return result.scalar_one_or_none()

    async def _insert(self, key: str, norm: str) -> Word:
        """Insert a pending stub, adopting a concurrent winner on key collision.

        The object is added INSIDE the savepoint: if the flush trips the unique
        constraint, the savepoint rollback discards it cleanly. Adding it first
        would leave it attached after the failed flush and poison the outer
        transaction, so the recovery SELECT would raise.
        """
        word = Word(norm=clean(norm, MAX_NORM), match_key=key, status="pending")
        try:
            async with self._session.begin_nested():
                self._session.add(word)
                await self._session.flush()
        except IntegrityError:
            existing = await self._get(key)
            if existing is None:
                raise
            return existing
        return word
