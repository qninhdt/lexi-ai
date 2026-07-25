"""The ``tags`` aggregate: the LLM-authored topic vocabulary and its curation."""

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, cast

from sqlalchemy import CursorResult, delete, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from lexi_ai.domain.models import TagName, TagUsage, WordListing
from lexi_ai.infrastructure.db.models import Tag, Word, WordTag
from lexi_ai.infrastructure.db.sanitize import MAX_TAG, MAX_TAG_KEY, MAX_TITLE, clean
from lexi_ai.normalize import tag_key

if TYPE_CHECKING:
    from lexi_ai.generation.schemas import GeneratedTopic


class SqlTagRepo:
    """Session-bound implementation of :class:`lexi_ai.domain.ports.TagRepo`.

    Every read joins through to ``words`` and filters ``status="done"`` so the
    vocabulary injected into prompts, the browse counts, and the browse itself all
    describe the same population. A tag whose only members are not done is absent
    from all three rather than from some of them.
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def sync(self, word_id: int, topics: Iterable["GeneratedTopic"]) -> None:
        """Clear and rebuild a word's tag links, resolving each topic to a tag.

        Topics are resolved in ``tag_key`` order so two concurrent words proposing
        the same new tags take the UNIQUE keys in one global order. Without that,
        two transactions can each hold one key and wait for the other; on
        PostgreSQL the resulting deadlock abort is not an ``IntegrityError`` and
        would roll back the entire word.
        """
        await self._session.execute(delete(WordTag).where(WordTag.word_id == word_id))
        seen: set[int] = set()
        for topic in sorted(topics, key=lambda t: tag_key(t.tag)):
            tag = await self._get_or_create(topic.tag, topic.title)
            if tag is None or tag.id in seen:
                continue  # skip empty-key/oversized and intra-entry duplicates
            seen.add(tag.id)
            self._session.add(WordTag(word_id=word_id, tag_id=tag.id))
        await self._session.flush()

    async def names(self) -> list[TagName]:
        """Existing topic vocab for prompt injection, name-sorted."""
        rows = await self._session.execute(
            select(Tag.name, Tag.title)
            .join(WordTag, WordTag.tag_id == Tag.id)
            .join(Word, Word.id == WordTag.word_id)
            .where(Word.status == "done")
            .group_by(Tag.id, Tag.name, Tag.title)
            .order_by(Tag.name)
        )
        return [TagName(name, title) for name, title in rows]

    async def usage(self) -> list[TagUsage]:
        """Every topic with its live member count, count-desc then name.

        The inner join drops a zero-member tag, which is effectively dead for
        browsing.
        """
        count = func.count(WordTag.id)
        rows = await self._session.execute(
            select(Tag.name, Tag.title, count)
            .join(WordTag, WordTag.tag_id == Tag.id)
            .join(Word, Word.id == WordTag.word_id)
            .where(Word.status == "done")
            .group_by(Tag.id, Tag.name, Tag.title)
            .order_by(count.desc(), Tag.name)
        )
        return [TagUsage(name, title, total) for name, title, total in rows]

    async def words_for_key(self, key: str, limit: int | None = None) -> list[WordListing]:
        """Done words carrying the tag with this ``tag_key``, norm-sorted."""
        stmt = (
            select(Word.id, Word.norm, Word.entry_type)
            .join(WordTag, WordTag.word_id == Word.id)
            .join(Tag, Tag.id == WordTag.tag_id)
            .where(Tag.tag_key == key, Word.status == "done")
            .order_by(Word.norm)
        )
        if limit is not None:
            stmt = stmt.limit(limit)
        rows = await self._session.execute(stmt)
        return [WordListing(wid, norm, entry_type) for wid, norm, entry_type in rows]

    async def rename(self, tag: str, name: str | None = None, title: str | None = None) -> bool:
        """Update a tag's display text in place; ``tag_key`` is immutable.

        The key is the dedup identity, set once at creation, so only display text
        changes and callers keep addressing the tag by the same key.
        """
        if name is None and title is None:
            raise ValueError("rename requires name and/or title")
        existing = await self._get(tag_key(tag))
        if existing is None:
            return False
        if name is not None:
            existing.name = clean(name, MAX_TAG)
        if title is not None:
            existing.title = clean(title, MAX_TITLE)
        return True

    async def delete(self, tag: str) -> bool:
        """Delete a tag by resolved key. Tagged words keep everything but this topic."""
        result = await self._session.execute(delete(Tag).where(Tag.tag_key == tag_key(tag)))
        return (cast("CursorResult", result).rowcount or 0) > 0

    async def merge(self, sources: Sequence[str], into: str) -> int:
        """Fold every source tag into ``into``, then delete the sources.

        ``into`` must already exist; merging never invents the destination. A word
        already carrying both tags would collide on ``UNIQUE(word_id, tag_id)``, so
        that duplicate link is dropped rather than re-pointed. Returns the number
        of links actually re-pointed.
        """
        target = await self._get(tag_key(into))
        if target is None:
            raise ValueError(f"unknown destination tag: {into!r}")
        moved = 0
        for source in sources:
            src_tag = await self._get(tag_key(source))
            if src_tag is None or src_tag.id == target.id:
                continue
            links = (
                await self._session.execute(select(WordTag).where(WordTag.tag_id == src_tag.id))
            ).scalars()
            for link in links:
                duplicate = await self._session.execute(
                    select(WordTag.id).where(
                        WordTag.word_id == link.word_id, WordTag.tag_id == target.id
                    )
                )
                if duplicate.first() is not None:
                    await self._session.delete(link)
                else:
                    link.tag_id = target.id
                    moved += 1
            await self._session.flush()
            await self._session.execute(delete(Tag).where(Tag.id == src_tag.id))
        return moved

    async def _get_or_create(self, name: str, title: str) -> Tag | None:
        """Resolve a topic to its tag row by key, or create it.

        Best-effort by design: a control/whitespace-only tag yields an empty key
        and is skipped rather than failing the word, as is a key that outgrows the
        column (NFKD normalization can expand a within-bound input). Title is set
        once on create; a later proposal for an existing tag is ignored. A
        concurrent create of the same key is recovered by re-fetching inside a
        SAVEPOINT so the outer transaction survives.
        """
        key = tag_key(name)
        if not key or len(key) > MAX_TAG_KEY:
            return None
        clean_name = clean(name, MAX_TAG)
        clean_title = clean(title, MAX_TITLE) or clean_name
        existing = await self._get(key)
        if existing is not None:
            return existing
        tag = Tag(name=clean_name, title=clean_title, tag_key=key)
        try:
            async with self._session.begin_nested():
                self._session.add(tag)
                await self._session.flush()
        except IntegrityError:
            existing = await self._get(key)
            if existing is None:
                raise
            return existing
        return tag

    async def _get(self, key: str) -> Tag | None:
        result = await self._session.execute(select(Tag).where(Tag.tag_key == key))
        return result.scalar_one_or_none()
