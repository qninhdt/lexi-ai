"""Repository: persist a GeneratedResult into the generated DB (Phase 5).

This is the ONLY module that writes ``match_key`` values — always via
:func:`lexi_ai.normalize.match_key`, so the write path and the read path (Phase
6) stay identical. If they diverge, lookups miss forever.

Async-safety: relationship collections are never touched on a *persistent*
object (that would trigger a lazy-load outside greenlet context). Children are
cleared with Core ``delete()`` and re-inserted with explicit foreign-key ids.

Idempotency & dedup: words/aliases/links are keyed so re-persisting the same
result creates no duplicates. ``words.match_key`` UNIQUE (Phase 2) is the
durable backstop; a concurrent insert that trips it is recovered via SAVEPOINT
re-fetch (decision #18 — single-process library, so this is a rare edge).
"""

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, NamedTuple, cast

from sqlalchemy import CursorResult, delete, func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from lexi_ai.constants import canonical_cambridge_ref
from lexi_ai.db import session_scope
from lexi_ai.generation.schemas import (
    GeneratedAlias,
    GeneratedEntry,
    GeneratedResult,
    GeneratedSense,
    GeneratedTopic,
    RelatedWord,
)
from lexi_ai.models import (
    Collocation,
    EntryLink,
    Example,
    Sense,
    SenseReference,
    Tag,
    Theme,
    ThemedExample,
    ThemedSense,
    Word,
    WordAlias,
    WordTag,
)
from lexi_ai.normalize import _CTRL_RE, match_key, tag_key, theme_key

if TYPE_CHECKING:
    from lexi_ai.theming.schemas import ThemedResult


class EmbeddedSenseRow(NamedTuple):
    """A done sense + its current-model vector, for semantic ranking."""

    sense_id: int
    word_id: int
    norm: str
    entry_type: str | None
    definition: str
    tier: str
    embedding: bytes


class Repository:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]):
        self._session_factory = session_factory

    def session(self):
        """Transactional session scope (commit on success, rollback on error)."""
        return session_scope(self._session_factory)

    # --- public API -------------------------------------------------------

    async def persist_result(
        self,
        result: GeneratedResult,
        cambridge_word_id: int | None = None,
        cambridge_cefr: dict[str, str] | None = None,
    ) -> list[Word]:
        """Persist every unit of one generation call in a single transaction.

        On any failure the transaction rolls back and each unit's word is
        marked ``status='error'`` in a separate transaction, then the error is
        re-raised.
        """
        cefr_map = cambridge_cefr or {}
        try:
            async with session_scope(self._session_factory) as session:
                words: list[Word] = []
                # Pass 1: upsert each unit word + its aliases/senses.
                for entry in result.units:
                    word = await self._upsert_entry(session, entry, cambridge_word_id, cefr_map)
                    words.append(word)
                await session.flush()
                # Pass 2: links — sibling units now have real ids to link to.
                for word, entry in zip(words, result.units, strict=True):
                    await self._link_related(session, word, entry.related)
                    word.status = "done"
                # Detach ids so callers can read them after the session closes.
                await session.flush()
                ids = [w.id for w in words]
            return await self._reload(ids)
        except Exception as exc:  # noqa: BLE001 - recorded then re-raised
            await self._record_error(result, str(exc))
            raise

    async def get_done_keys(self) -> set[str]:
        """All match_keys already generated (status='done') — candidate diff."""
        async with session_scope(self._session_factory) as session:
            rows = await session.execute(select(Word.match_key).where(Word.status == "done"))
            return {r[0] for r in rows}

    # --- word management (delete / paginate) ------------------------------

    async def delete_word(self, word_id: int) -> bool:
        """Delete a word by id; return whether a row was removed.

        Children (senses, aliases, links, tags, questions) are removed by the
        DB-level ``ON DELETE CASCADE`` FKs (SQLite pragma enabled in ``db.py``),
        so a single Core ``delete`` suffices — no relationship walk."""
        async with session_scope(self._session_factory) as session:
            result = await session.execute(delete(Word).where(Word.id == word_id))
            return (cast("CursorResult", result).rowcount or 0) > 0

    async def list_words(
        self, status: str = "done", limit: int | None = None, offset: int = 0
    ) -> list[tuple[int, str, str | None]]:
        """Paginated dictionary browse as ``(id, norm, entry_type)``, norm-sorted.

        Filters by ``status`` (default ``done`` — the same population as the tag
        browse). The api layer turns rows into lightweight ``SearchResult``s."""
        async with session_scope(self._session_factory) as session:
            stmt = (
                select(Word.id, Word.norm, Word.entry_type)
                .where(Word.status == status)
                .order_by(Word.norm)
                .offset(offset)
            )
            if limit is not None:
                stmt = stmt.limit(limit)
            rows = await session.execute(stmt)
            return [(wid, norm, etype) for wid, norm, etype in rows]

    # --- topic tags (reads — never calls an LLM) --------------------------

    async def all_tags(self) -> list[tuple[str, str]]:
        """Existing topic vocab (name, title) for prompt injection, name-sorted.

        Joins through to ``words`` and filters ``status="done"`` — the SAME
        population as :meth:`count_tags`/:meth:`words_for_tag_key` — so a tag whose
        only members are non-done (e.g. a word force-regenerated away then flipped
        to ``error``) is NOT re-injected into future prompts and the injected vocab
        never diverges from the browse surface.
        """
        async with session_scope(self._session_factory) as session:
            rows = await session.execute(
                select(Tag.name, Tag.title)
                .join(WordTag, WordTag.tag_id == Tag.id)
                .join(Word, Word.id == WordTag.word_id)
                .where(Word.status == "done")
                .group_by(Tag.id, Tag.name, Tag.title)
                .order_by(Tag.name)
            )
            return [(name, title) for name, title in rows]

    async def count_tags(self) -> list[tuple[str, str, int]]:
        """Every topic with its live member count, sorted count-desc then name.

        Joins through to ``words`` and filters ``status="done"`` so the count
        reflects the SAME population as :meth:`words_for_tag_key` — the two never
        diverge. Inner join → a 0-member tag is omitted (effectively dead for
        browse).
        """
        async with session_scope(self._session_factory) as session:
            cnt = func.count(WordTag.id)
            rows = await session.execute(
                select(Tag.name, Tag.title, cnt)
                .join(WordTag, WordTag.tag_id == Tag.id)
                .join(Word, Word.id == WordTag.word_id)
                .where(Word.status == "done")
                .group_by(Tag.id, Tag.name, Tag.title)
                .order_by(cnt.desc(), Tag.name)
            )
            return [(name, title, count) for name, title, count in rows]

    async def words_for_tag_key(
        self, key: str, limit: int | None = None
    ) -> list[tuple[int, str, str | None]]:
        """Done words carrying the tag with this ``tag_key``, as (id, norm, type).

        The api layer resolves the raw query to ``key`` via ``tag_key`` (same
        function as the write path) so casing/plural variants all hit.
        """
        async with session_scope(self._session_factory) as session:
            stmt = (
                select(Word.id, Word.norm, Word.entry_type)
                .join(WordTag, WordTag.word_id == Word.id)
                .join(Tag, Tag.id == WordTag.tag_id)
                .where(Tag.tag_key == key, Word.status == "done")
                .order_by(Word.norm)
            )
            if limit is not None:
                stmt = stmt.limit(limit)
            rows = await session.execute(stmt)
            return [(wid, norm, etype) for wid, norm, etype in rows]

    # --- embeddings (bytes in/out only — never calls an embedder) ----------

    async def senses_needing_embedding(
        self,
        model_name: str,
        word_ids: list[int] | None = None,
        limit: int | None = None,
    ) -> list[tuple[int, str, str]]:
        """Senses missing a current-model vector, as ``(sense_id, norm, definition)``.

        A sense needs embedding if its ``embedding`` is null OR was produced by a
        different ``embedding_model`` (model drift). Restrict to ``word_ids`` for
        the post-generation path; omit them for a global backfill. The api layer
        turns ``(norm, definition)`` into the embed text and calls the embedder —
        this method stays purely a DB read.
        """
        async with session_scope(self._session_factory) as session:
            stmt = (
                select(Sense.id, Word.norm, Sense.definition)
                .join(Word, Word.id == Sense.word_id)
                .where(
                    Word.status == "done",
                    (Sense.embedding.is_(None)) | (Sense.embedding_model != model_name),
                )
                .order_by(Sense.id)
            )
            if word_ids is not None:
                if not word_ids:
                    return []
                stmt = stmt.where(Sense.word_id.in_(word_ids))
            if limit is not None:
                stmt = stmt.limit(limit)
            rows = await session.execute(stmt)
            return [(sid, norm, definition) for sid, norm, definition in rows]

    async def store_embeddings(
        self, vectors: list[tuple[int, bytes]], model_name: str, dim: int
    ) -> int:
        """Write packed vectors onto senses by id. Returns the number of rows
        actually updated (a sense deleted by a concurrent force-regen between the
        read and this write matches 0 rows and is not counted; the next backfill
        re-embeds its replacement).

        Receives ready bytes (the api layer packed them) so this module never
        touches an embedder. Idempotent per row: re-writing the same sense just
        overwrites its vector/model/dim.
        """
        if not vectors:
            return 0
        written = 0
        async with session_scope(self._session_factory) as session:
            for sense_id, blob in vectors:
                result = await session.execute(
                    update(Sense)
                    .where(Sense.id == sense_id)
                    .values(embedding=blob, embedding_model=model_name, embedding_dim=dim)
                )
                written += cast("CursorResult", result).rowcount or 0
        return written

    async def embedded_senses(self, model_name: str) -> list["EmbeddedSenseRow"]:
        """Every done sense with a current-model vector, for semantic search.

        Returns rows carrying enough to build a :class:`SemanticHit` without a
        second query. Only rows whose ``embedding_model`` matches are returned, so
        vectors from a previous model are ignored until re-embedded. ``embedding``
        is non-null (filtered in SQL).
        """
        async with session_scope(self._session_factory) as session:
            rows = await session.execute(
                select(
                    Sense.id,
                    Sense.word_id,
                    Word.norm,
                    Word.entry_type,
                    Sense.definition,
                    Sense.tier,
                    Sense.embedding,
                )
                .join(Word, Word.id == Sense.word_id)
                .where(
                    Word.status == "done",
                    Sense.embedding.is_not(None),
                    Sense.embedding_model == model_name,
                )
            )
            return [
                EmbeddedSenseRow(sid, wid, norm, etype, definition, tier, blob)
                for sid, wid, norm, etype, definition, tier, blob in rows
            ]

    # --- one unit ---------------------------------------------------------

    async def _upsert_entry(
        self,
        session: AsyncSession,
        entry: GeneratedEntry,
        cambridge_word_id: int | None,
        cefr_map: dict[str, str],
    ) -> Word:
        key = match_key(entry.norm)
        word = await self._get_or_create_word(session, key, entry.norm)
        word.norm = entry.norm
        word.entry_type = entry.entry_type
        word.pos = entry.pos
        word.status = "done"
        word.error_msg = None
        if cambridge_word_id is not None:
            word.cambridge_word_id = cambridge_word_id
        await session.flush()

        await self._sync_aliases(session, word.id, entry.aliases)
        await self._sync_senses(session, word.id, entry.senses, cefr_map)
        await self._sync_tags(session, word.id, entry.topics)
        return word

    async def _sync_aliases(
        self, session: AsyncSession, word_id: int, aliases: Iterable[GeneratedAlias]
    ) -> None:
        # Fully derived from generation: clear and rebuild, deduping by key.
        await session.execute(delete(WordAlias).where(WordAlias.word_id == word_id))
        seen: set[str] = set()
        for alias in aliases:
            akey = match_key(alias.alias_norm)
            if akey in seen:
                continue
            seen.add(akey)
            session.add(
                WordAlias(
                    word_id=word_id,
                    alias_norm=alias.alias_norm,
                    alias_match_key=akey,
                    type=alias.type,
                    dialect=alias.dialect,
                )
            )
        await session.flush()

    async def _sync_senses(
        self,
        session: AsyncSession,
        word_id: int,
        senses: Iterable[GeneratedSense],
        cefr_map: dict[str, str],
    ) -> None:
        # Deleting senses cascades to references+examples+collocations (ON DELETE CASCADE).
        await session.execute(delete(Sense).where(Sense.word_id == word_id))
        for order, gen_sense in enumerate(senses):
            cefr = self._resolve_cefr(gen_sense, cefr_map)
            sense = Sense(
                word_id=word_id,
                definition=gen_sense.definition,
                tier=gen_sense.tier,
                sense_order=order,
                pos=gen_sense.pos,
                cefr_level=cefr,
                # Enrichments: guideword is free LLM text -> sanitize; grammar is a
                # set of schema-validated tokens joined with ',' (no token contains
                # ',' — guarded by test); register/connotation are validated enum
                # values or None. Empty -> None so the read side yields [] / None.
                guideword=self._clean(gen_sense.guideword, self._MAX_GUIDEWORD) or None
                if gen_sense.guideword
                else None,
                grammar=",".join(gen_sense.grammar) or None,
                register=gen_sense.register,
                connotation=gen_sense.connotation,
            )
            session.add(sense)
            await session.flush()
            for ref in gen_sense.references:
                session.add(
                    SenseReference(
                        sense_id=sense.id,
                        source=ref.source,
                        source_ref=ref.source_ref,
                    )
                )
            for ex_order, ex in enumerate(gen_sense.examples):
                session.add(Example(sense_id=sense.id, text=ex, example_order=ex_order))
            # Collocations mirror examples: ordered child rows. text is a Text
            # column (unbounded) -> control-strip via _clean (Postgres NUL-safety)
            # with a GENEROUS sanity cap, not the tag's 255 (no DB need to truncate
            # a legit long collocation). Empty/whitespace-only is skipped (best-effort).
            for c_order, colloc in enumerate(gen_sense.collocations):
                text = self._clean(colloc, self._MAX_COLLOCATION)
                if not text:
                    continue
                session.add(Collocation(sense_id=sense.id, text=text, collocation_order=c_order))
        await session.flush()

    # --- topic tags (resolve-or-create, deterministic dedup) --------------

    _MAX_TAG = 64
    _MAX_TITLE = 128
    _MAX_TAG_KEY = 255  # must match Tag.tag_key String(255) — NFKD can expand a key
    _MAX_GUIDEWORD = 64  # must match Sense.guideword String(64)
    _MAX_COLLOCATION = 512  # Collocation.text is Text (unbounded) — generous sanity cap only
    _MAX_THEME_NAME = 128  # Theme.name is Text (unbounded) — generous sanity cap only
    _MAX_THEME_KEY = 255  # must match Theme.theme_key String(255)
    _MAX_STYLE_PROMPT = 4000  # Theme.style_prompt is Text (unbounded) — generous sanity cap
    _MAX_THEMED_TEXT = 4000  # ThemedSense.definition / ThemedExample.text (Text) — generous cap

    @staticmethod
    def _clean(s: str, cap: int) -> str:
        """Single-line, control-free, trimmed, length-capped — never trust LLM text.

        Guarantees a stored ``name``/``title`` cannot carry an embedded newline
        (which would break the vocab block re-injected into every future prompt)
        or exceed the column width (truncation-merge on Postgres).
        """
        s = _CTRL_RE.sub(" ", s)
        s = " ".join(s.split()).strip()
        return s[:cap]

    async def _sync_tags(
        self, session: AsyncSession, word_id: int, topics: Iterable[GeneratedTopic]
    ) -> None:
        """Clear + rebuild a word's ``word_tags``, resolving each topic to a tag.

        Topics are resolved in ``tag_key``-sorted order so two concurrent words
        proposing the same new tags acquire the UNIQUE keys in one global order —
        no cross-transaction lock cycle (Postgres deadlock, whose abort is NOT an
        ``IntegrityError`` and would roll back the whole word). Dedup by resolved
        ``tag.id`` so proposals that normalize to the same tag link once.
        """
        await session.execute(delete(WordTag).where(WordTag.word_id == word_id))
        ordered = sorted(topics, key=lambda t: tag_key(t.tag))
        seen: set[int] = set()
        for topic in ordered:
            tag = await self._get_or_create_tag(session, topic.tag, topic.title)
            if tag is None or tag.id in seen:
                continue  # skip empty-key/oversized + intra-entry duplicate tags
            seen.add(tag.id)
            session.add(WordTag(word_id=word_id, tag_id=tag.id))
        await session.flush()

    async def _get_or_create_tag(self, session: AsyncSession, name: str, title: str) -> Tag | None:
        """Resolve a topic to its ``Tag`` row by ``tag_key``, or create it.

        Best-effort: a NUL/control/whitespace-only tag yields an empty key and is
        skipped (``None``), never fatal — as is a key that exceeds the ``tag_key``
        column width (NFKD normalization can expand a within-bound input past 255).
        Title is set ONCE on create (first-seen wins); on resolve the stored title
        is kept and the proposal ignored. Concurrent create of the same key is
        recovered via SAVEPOINT re-fetch, exactly like ``_insert_word`` (the
        Postgres path; on SQLite the word-level write lock serializes first).
        """
        key = tag_key(name)
        if not key or len(key) > self._MAX_TAG_KEY:
            return None
        clean_name = self._clean(name, self._MAX_TAG)
        clean_title = self._clean(title, self._MAX_TITLE) or clean_name
        existing = await self._get_tag(session, key)
        if existing is not None:
            return existing  # title set-once: keep stored, ignore proposed
        tag = Tag(name=clean_name, title=clean_title, tag_key=key)
        try:
            async with session.begin_nested():
                session.add(tag)
                await session.flush()
        except IntegrityError:
            existing = await self._get_tag(session, key)
            if existing is None:
                raise
            return existing
        return tag

    async def _get_tag(self, session: AsyncSession, key: str) -> Tag | None:
        result = await session.execute(select(Tag).where(Tag.tag_key == key))
        return result.scalar_one_or_none()

    # --- tag management (curate the LLM-authored vocab) --------------------

    async def rename_tag(self, tag: str, name: str | None = None, title: str | None = None) -> bool:
        """Update an existing tag's display ``name``/``title`` in place.

        Resolved via ``tag_key(tag)`` — the SAME normalizer as the write path, so
        casing/plural variants all hit. ``tag_key`` itself is immutable (it is the
        dedup identity, set once at creation); only display text changes. Returns
        whether a matching tag was found. At least one of ``name``/``title`` must
        be given.
        """
        if name is None and title is None:
            raise ValueError("rename_tag requires name and/or title")
        async with session_scope(self._session_factory) as session:
            existing = await self._get_tag(session, tag_key(tag))
            if existing is None:
                return False
            if name is not None:
                existing.name = self._clean(name, self._MAX_TAG)
            if title is not None:
                existing.title = self._clean(title, self._MAX_TITLE)
            return True

    async def delete_tag(self, tag: str) -> bool:
        """Delete a tag by (resolved) key; return whether one was removed.

        Cascades ``word_tags`` (the DB-level FK); the tagged words themselves are
        untouched — they simply lose this one topic."""
        async with session_scope(self._session_factory) as session:
            result = await session.execute(delete(Tag).where(Tag.tag_key == tag_key(tag)))
            return (cast("CursorResult", result).rowcount or 0) > 0

    async def merge_tags(self, sources: Sequence[str], into: str) -> int:
        """Fold every ``sources`` tag into ``into``, then delete the sources.

        ``into`` must already exist (``ValueError`` otherwise) — merging never
        invents the destination tag. Each source's ``word_tags`` rows are
        re-pointed to ``into``'s id; a word already carrying both tags would
        collide on ``UNIQUE(word_id, tag_id)``, so that duplicate association is
        dropped instead of re-pointed. Returns the number of associations
        actually re-pointed (dropped duplicates are not counted).
        """
        async with session_scope(self._session_factory) as session:
            target = await self._get_tag(session, tag_key(into))
            if target is None:
                raise ValueError(f"unknown destination tag: {into!r}")
            moved = 0
            for source in sources:
                src_tag = await self._get_tag(session, tag_key(source))
                if src_tag is None or src_tag.id == target.id:
                    continue
                links = (
                    await session.execute(select(WordTag).where(WordTag.tag_id == src_tag.id))
                ).scalars()
                for link in links:
                    dup = await session.execute(
                        select(WordTag.id).where(
                            WordTag.word_id == link.word_id, WordTag.tag_id == target.id
                        )
                    )
                    if dup.first() is not None:
                        await session.delete(link)
                    else:
                        link.tag_id = target.id
                        moved += 1
                await session.flush()
                await session.execute(delete(Tag).where(Tag.id == src_tag.id))
            return moved

    # --- themes (resolve-or-create by theme_key) --------------------------

    async def create_theme(
        self,
        name: str,
        style_prompt: str,
        description: str | None = None,
        tone: str | None = None,
        key: str | None = None,
        overwrite: bool = False,
    ) -> Theme:
        """Resolve-or-create a theme by ``theme_key``.

        If ``overwrite`` is True, existing theme fields are updated.
        """
        final_key = theme_key(key) if key is not None else theme_key(name)
        if not final_key or len(final_key) > self._MAX_THEME_KEY:
            raise ValueError(f"theme key/name yields no valid key: {key or name!r}")
        clean_name = self._clean(name, self._MAX_THEME_NAME)
        clean_prompt = self._clean(style_prompt, self._MAX_STYLE_PROMPT)
        clean_desc = self._clean(description, 1000) if description else None
        clean_tone = self._clean(tone, 255) if tone else None
        async with session_scope(self._session_factory) as session:
            existing = await self._get_theme(session, final_key)
            if existing is not None:
                if overwrite:
                    existing.name = clean_name
                    existing.style_prompt = clean_prompt
                    existing.description = clean_desc
                    existing.tone = clean_tone
                return existing
            theme = Theme(
                theme_key=final_key,
                name=clean_name,
                style_prompt=clean_prompt,
                description=clean_desc,
                tone=clean_tone,
            )
            try:
                async with session.begin_nested():
                    session.add(theme)
                    await session.flush()
            except IntegrityError:
                existing = await self._get_theme(session, final_key)
                if existing is None:
                    raise
                if overwrite:
                    existing.name = clean_name
                    existing.style_prompt = clean_prompt
                    existing.description = clean_desc
                    existing.tone = clean_tone
                return existing
            return theme

    async def list_themes(self) -> list[Theme]:
        """Every theme, name-sorted. FREE — never calls an LLM."""
        async with session_scope(self._session_factory) as session:
            rows = await session.execute(select(Theme).order_by(Theme.name))
            return list(rows.scalars())

    async def _get_theme(self, session: AsyncSession, key: str) -> Theme | None:
        result = await session.execute(select(Theme).where(Theme.theme_key == key))
        return result.scalar_one_or_none()

    # --- theme management (get one / update / delete) ----------------------

    async def get_theme(self, key: str) -> Theme | None:
        """A theme by ``theme_key`` (raw display name resolved via ``theme_key``
        by the api layer), or ``None`` if absent. FREE."""
        async with session_scope(self._session_factory) as session:
            return await self._get_theme(session, key)

    async def update_theme(
        self,
        key: str,
        name: str | None = None,
        style_prompt: str | None = None,
        description: str | None = None,
        tone: str | None = None,
    ) -> Theme | None:
        """Partially update an EXISTING theme's fields; ``theme_key`` is immutable
        (renaming ``name`` never re-keys the theme, so callers keep addressing it
        by the same key). Returns the updated row, or ``None`` if ``key`` is
        unknown (the api layer raises for that case — unlike ``create_theme``,
        this never creates)."""
        async with session_scope(self._session_factory) as session:
            existing = await self._get_theme(session, key)
            if existing is None:
                return None
            if name is not None:
                existing.name = self._clean(name, self._MAX_THEME_NAME)
            if style_prompt is not None:
                existing.style_prompt = self._clean(style_prompt, self._MAX_STYLE_PROMPT)
            if description is not None:
                existing.description = self._clean(description, 1000)
            if tone is not None:
                existing.tone = self._clean(tone, 255)
            return existing

    async def delete_theme(self, key: str) -> bool:
        """Delete a theme by ``theme_key``; return whether one was removed.

        Cascades ``themed_senses``/``themed_examples`` (DB-level FKs); the
        neutral entries themselves are untouched."""
        async with session_scope(self._session_factory) as session:
            result = await session.execute(delete(Theme).where(Theme.theme_key == key))
            return (cast("CursorResult", result).rowcount or 0) > 0

    async def persist_themed(
        self, theme_id: int, result: "ThemedResult", sense_ids: Sequence[int]
    ) -> None:
        """Overwrite the themed rows for ``(sense_ids, theme_id)`` in place.

        ``sense_ids`` is the ordered list of neutral sense ids (the api layer
        supplies it in the SAME order it numbered the senses in the prompt), so
        ``result.senses[i]`` maps to ``sense_ids[i]``. A count mismatch is a hard
        error (never a silent zip): the model returned the wrong number of senses.
        Per sense: delete the existing themed row (cascades themed_examples), then
        insert fresh + its ordered themed examples. Core delete + explicit-FK insert
        only (never touch relationship collections on a persistent object).
        """
        if len(result.senses) != len(sense_ids):
            raise ValueError(
                f"themed sense count {len(result.senses)} != neutral sense count {len(sense_ids)}"
            )
        async with session_scope(self._session_factory) as session:
            for sense_id, themed in zip(sense_ids, result.senses, strict=True):
                await session.execute(
                    delete(ThemedSense).where(
                        ThemedSense.sense_id == sense_id,
                        ThemedSense.theme_id == theme_id,
                    )
                )
                definition = self._clean(themed.definition, self._MAX_THEMED_TEXT)
                row = ThemedSense(sense_id=sense_id, theme_id=theme_id, definition=definition)
                session.add(row)
                await session.flush()
                order = 0
                for ex in themed.examples:
                    text = self._clean(ex, self._MAX_THEMED_TEXT)
                    if not text:
                        continue
                    session.add(
                        ThemedExample(themed_sense_id=row.id, text=text, example_order=order)
                    )
                    order += 1
            await session.flush()

    async def themed_for_word(
        self, word_id: int, theme_id: int
    ) -> dict[int, tuple[str, list[str]]]:
        """Overlay map ``{sense_id: (themed_definition, [themed_examples])}`` for a
        word under one theme. One companion query per word (no N+1); senses without
        a themed row are simply absent (the read layer falls back to neutral)."""
        async with session_scope(self._session_factory) as session:
            rows = await session.execute(
                select(ThemedSense.id, ThemedSense.sense_id, ThemedSense.definition)
                .join(Sense, Sense.id == ThemedSense.sense_id)
                .where(Sense.word_id == word_id, ThemedSense.theme_id == theme_id)
            )
            themed = [(tsid, sid, definition) for tsid, sid, definition in rows]
            if not themed:
                return {}
            ex_rows = await session.execute(
                select(ThemedExample.themed_sense_id, ThemedExample.text)
                .where(ThemedExample.themed_sense_id.in_([t[0] for t in themed]))
                .order_by(ThemedExample.example_order)
            )
            examples: dict[int, list[str]] = {}
            for tsid, text in ex_rows:
                examples.setdefault(tsid, []).append(text)
            return {sid: (definition, examples.get(tsid, [])) for tsid, sid, definition in themed}

    async def resolve_theme(self, key_or_id: str | int) -> tuple[int, str] | None:
        """``(theme_id, style_prompt)`` for a ``theme_key`` or ``theme_id``, or ``None`` if unknown."""
        async with session_scope(self._session_factory) as session:
            if isinstance(key_or_id, int):
                row = await session.execute(
                    select(Theme.id, Theme.style_prompt).where(Theme.id == key_or_id)
                )
            else:
                try:
                    theme_id = int(key_or_id)
                    row = await session.execute(
                        select(Theme.id, Theme.style_prompt).where(Theme.id == theme_id)
                    )
                except ValueError:
                    row = await session.execute(
                        select(Theme.id, Theme.style_prompt).where(Theme.theme_key == key_or_id)
                    )
            found = row.first()
            return (found[0], found[1]) if found is not None else None

    async def senses_for_theming(
        self, word_id: int
    ) -> list[tuple[int, str, str | None, str | None, str]]:
        """Ordered ``(sense_id, definition, pos, guideword, tier)`` for a word.

        Sorted by ``sense_order`` then ``id`` for a deterministic prompt numbering:
        the api layer passes these ids to :meth:`persist_themed` in the SAME order
        it numbers them in the prompt, so themed index ``i`` maps to ``sense_ids[i]``.
        """
        async with session_scope(self._session_factory) as session:
            rows = await session.execute(
                select(Sense.id, Sense.definition, Sense.pos, Sense.guideword, Sense.tier)
                .where(Sense.word_id == word_id)
                .order_by(Sense.sense_order, Sense.id)
            )
            return [(sid, d, pos, gw, tier) for sid, d, pos, gw, tier in rows]

    @staticmethod
    def _resolve_cefr(sense: GeneratedSense, cefr_map: dict[str, str]) -> str | None:
        """Cambridge-first cefr (decision #13): a Cambridge reference wins.

        Both the reference id and the map keys are canonicalized (``sense#42`` /
        ``42`` collapse to the same key) so the rule cannot silently fall through
        when the model echoes the labelled form it was shown in the prompt.
        """
        for ref in sense.references:
            if ref.source == "cambridge":
                key = canonical_cambridge_ref(ref.source_ref)
                if key in cefr_map:
                    return cefr_map[key]
        return sense.cefr_level

    # --- linking + stubs --------------------------------------------------

    async def _link_related(
        self, session: AsyncSession, word: Word, related: Iterable[RelatedWord]
    ) -> None:
        for rel in related:
            stub = await self._get_or_create_stub(session, rel.norm)
            if stub.id == word.id:
                continue  # never self-link
            await self._ensure_link(session, word.id, stub.id, rel.rel_type)
        await session.flush()

    async def _ensure_link(
        self, session: AsyncSession, from_id: int, to_id: int, rel_type: str
    ) -> None:
        """Insert an entry_link if the (from, to, rel_type) triple is absent."""
        exists = await session.execute(
            select(EntryLink.id).where(
                EntryLink.from_word_id == from_id,
                EntryLink.to_word_id == to_id,
                EntryLink.rel_type == rel_type,
            )
        )
        if exists.first() is None:
            session.add(EntryLink(from_word_id=from_id, to_word_id=to_id, rel_type=rel_type))

    async def _get_or_create_stub(self, session: AsyncSession, norm: str) -> Word:
        key = match_key(norm)
        word = await self._get_word(session, key)
        if word is not None:
            return word
        return await self._insert_word(session, key, norm, status="pending")

    async def seed_phrase_unit(
        self,
        session: AsyncSession,
        phrase_title: str,
        host_display: str | None,
        entry_type: str | None,
        is_overlap: bool,
    ) -> None:
        """Seed one phrase_title (Phase 7), reusing the stub + link path.

        Orphan (no standalone row): create a pending stub so it enters the lazy
        generation queue (invisible to the Phase 3 candidate scan otherwise).
        Overlap (standalone row exists): ensure host + unit stubs and link them
        ``part_of_phrasal_family``. Idempotent: stubs dedup by match_key, links
        by the UNIQUE triple.
        """
        unit = await self._get_or_create_stub(session, phrase_title)
        if unit.entry_type is None and entry_type is not None:
            unit.entry_type = entry_type
        if is_overlap and host_display:
            host = await self._get_or_create_stub(session, host_display)
            if host.id != unit.id:
                await self._ensure_link(session, host.id, unit.id, "part_of_phrasal_family")
        await session.flush()

    async def _get_or_create_word(self, session: AsyncSession, key: str, norm: str) -> Word:
        word = await self._get_word(session, key)
        if word is not None:
            return word
        return await self._insert_word(session, key, norm, status="pending")

    async def _get_word(self, session: AsyncSession, key: str) -> Word | None:
        result = await session.execute(select(Word).where(Word.match_key == key))
        return result.scalar_one_or_none()

    async def _insert_word(self, session: AsyncSession, key: str, norm: str, status: str) -> Word:
        word = Word(norm=norm, match_key=key, status=status)
        # Add the object INSIDE the savepoint: if the flush trips the UNIQUE
        # constraint (a concurrent tx inserted the same key first), the savepoint
        # rollback discards this pending object cleanly. Adding it before the
        # savepoint would leave it attached after the failed flush and poison the
        # outer transaction, so the recovery SELECT would raise PendingRollback.
        try:
            async with session.begin_nested():
                session.add(word)
                await session.flush()
        except IntegrityError:
            # A concurrent tx inserted the same key first — adopt that row.
            existing = await self._get_word(session, key)
            if existing is None:
                raise
            return existing
        return word

    # --- error path + reload ---------------------------------------------

    async def _record_error(self, result: GeneratedResult, message: str) -> None:
        try:
            async with session_scope(self._session_factory) as session:
                for entry in result.units:
                    key = match_key(entry.norm)
                    word = await self._get_word(session, key)
                    if word is None:
                        word = Word(norm=entry.norm, match_key=key, status="error")
                        session.add(word)
                    word.status = "error"
                    word.error_msg = message[:2000]
                    await session.flush()
        except Exception:  # noqa: BLE001 - best-effort error recording
            pass

    async def _reload(self, ids: list[int]) -> list[Word]:
        """Re-fetch persisted words as detached, COLUMN-ONLY objects.

        The session closes here, so relationships are NOT eager-loaded: reading
        ``.senses``/``.aliases``/``.links_out`` on a returned object would raise
        ``MissingGreenlet``. Callers wanting the graph must go through
        ``api.Lexicon._to_entry`` (which ``selectinload``s inside a live session).
        """
        if not ids:
            return []
        async with session_scope(self._session_factory) as session:
            rows = await session.execute(select(Word).where(Word.id.in_(ids)))
            by_id = {w.id: w for w in rows.scalars()}
        return [by_id[i] for i in ids if i in by_id]
