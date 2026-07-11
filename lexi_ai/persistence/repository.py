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

import hashlib
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, NamedTuple, cast

from sqlalchemy import CursorResult, delete, func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm import aliased

from lexi_ai.constants import WSD_CANDIDATE_CAP, canonical_cambridge_ref
from lexi_ai.db import session_scope
from lexi_ai.generation.schemas import (
    ExampleGenContext,
    GeneratedAlias,
    GeneratedEntry,
    GeneratedResult,
    GeneratedSense,
    GeneratedSenseRelation,
    GeneratedTopic,
    RelatedWord,
)
from lexi_ai.models import (
    Asset,
    Collocation,
    Example,
    Question,
    Sense,
    SenseForm,
    SenseReference,
    SenseRelation,
    Tag,
    Theme,
    ThemedExample,
    ThemedSense,
    Word,
    WordAlias,
    WordRelation,
    WordTag,
    _utcnow,
)
from lexi_ai.normalize import _CTRL_RE, match_key, tag_key, theme_key
from lexi_ai.read_models import Stats

if TYPE_CHECKING:
    from lexi_ai.assets.repository import AssetRepository
    from lexi_ai.theming.schemas import ThemedResult


def sense_content_hash(definition: str) -> str:
    """Stable content fingerprint of a target sense (Phase 4/6).

    Stamped on ``sense_relation.target_hash`` at resolve time and re-checked on
    read (Phase 6): if the target sense's definition later changes (regenerate),
    the stored hash no longer matches and the edge is treated as unresolved
    rather than silently pointing at a mutated meaning. The definition is the
    load-bearing meaning carrier, so it alone keys the hash.
    """
    return hashlib.sha256(definition.encode("utf-8")).hexdigest()


class EmbeddedSenseRow(NamedTuple):
    """A done sense + its current-model vector, for semantic ranking."""

    sense_id: int
    word_id: int
    norm: str
    entry_type: str | None
    definition: str
    tier: str
    embedding: bytes


class ResolveCandidate(NamedTuple):
    """One target-sense option for a WSD task: the DB sense id + the facts the
    judge sees (``pos`` for the POS filter, ``definition`` shown in the prompt)."""

    sense_id: int
    pos: str | None
    definition: str


class ResolveTask(NamedTuple):
    """One pending edge lifted off the queue, ready for the judge. ``edge_id`` is
    the ``sense_relation`` row; ``candidates`` are its target word's senses
    (POS-filtered later). ``source_pos`` drives the POS filter; ``gloss`` +
    ``source_def`` are the (untrusted) prompt text."""

    edge_id: int
    rel_type: str
    gloss: str
    source_def: str
    source_pos: str | None
    candidates: list[ResolveCandidate]


class ResolveDecision(NamedTuple):
    """The bounds-validated ([F3]) verdict for one edge, ready to apply.
    ``to_sense_id`` None ⇒ mark unresolvable; else resolve to that sense with the
    stamped ``target_hash``."""

    edge_id: int
    to_sense_id: int | None
    target_hash: str | None


class ResolveOutcome(NamedTuple):
    """One edge's WSD result (per :meth:`Repository.apply_resolutions`).

    ``state`` is the DERIVED post-resolve state (Q1): ``resolved`` when a target
    sense was chosen, ``unresolvable`` when the judge returned none, ``noop`` when
    a racing regenerate made the conditional write a no-op ([F6]), ``error`` when
    a poison-pill isolated the edge ([F7]). Order-aligned with the input decisions.
    """

    edge_id: int
    state: str
    error: str | None = None


class Repository:
    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        assets: "AssetRepository | None" = None,
    ):
        self._session_factory = session_factory
        # Optional asset cache: when present, delete/regenerate paths enumerate the
        # affected source ids and GC their cached translations/clips on the SAME
        # session (best-effort — the read-time hash verify is the correctness net).
        self._assets = assets

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
        so a single Core ``delete`` suffices — no relationship walk. Cached assets
        have no FK to the source rows, so GC them explicitly first (best-effort),
        on the SAME session so it rolls back with the delete."""
        async with session_scope(self._session_factory) as session:
            await self._gc_word_assets(session, word_id)
            # [F4] No inbound-edge demotion needed here: ``sense_relation.to_word_id``
            # is ``ON DELETE CASCADE``, so deleting this word removes every edge that
            # points AT it outright — there is no orphaned sense->word row left to
            # strand as derived-unresolvable. (The demote helper's live job is the
            # regenerate path in ``_sync_senses``, where the word survives but its
            # senses churn and inbound edges must be re-queued.)
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
        # norm is untrusted LM text landing in a Text column: a NUL crashes the
        # Postgres INSERT (rolls the whole word back to status="error"). Clean it
        # like every other free field. The key is already NUL/control-safe
        # (match_key strips them), so cleaning norm cannot desync key from norm.
        norm = self._clean(entry.norm, self._MAX_NORM)
        word = await self._get_or_create_word(session, key, norm)
        word.norm = norm
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
            # alias_norm is untrusted LM text in a Text column: clean it (NUL crashes
            # the Postgres INSERT). alias_match_key is already NUL/control-safe.
            alias_norm = self._clean(alias.alias_norm, self._MAX_NORM)
            session.add(
                WordAlias(
                    word_id=word_id,
                    alias_norm=alias_norm,
                    alias_match_key=akey,
                    type=alias.type,
                    dialect=alias.dialect,
                )
            )
        await session.flush()

    async def _gc_word_assets(self, session: AsyncSession, word_id: int) -> None:
        """Best-effort GC of cached assets for every sense/example/collocation of a
        word, on the CALLER's session (so it rolls back with the transaction).

        Bulk Core deletes never expose child ids, so enumerate them FIRST. A missed
        row is inert (the read-time hash verify prevents a mis-serve); this is pure
        housekeeping to reclaim rows + on-disk clips. No-op when no asset cache is
        wired (``self._assets is None``)."""
        if self._assets is None:
            return
        sense_ids = (
            (await session.execute(select(Sense.id).where(Sense.word_id == word_id)))
            .scalars()
            .all()
        )
        if not sense_ids:
            return
        example_ids = (
            (await session.execute(select(Example.id).where(Example.sense_id.in_(sense_ids))))
            .scalars()
            .all()
        )
        colloc_ids = (
            (
                await session.execute(
                    select(Collocation.id).where(Collocation.sense_id.in_(sense_ids))
                )
            )
            .scalars()
            .all()
        )
        # Table-driven over source_kind so adding a kind adds a GC branch for free.
        # One bulk delete per kind (not per id) — bounds round trips inside the
        # single persist/delete transaction.
        for source_kind, ids in (
            ("sense_def", list(sense_ids)),
            ("example", list(example_ids)),
            ("collocation", list(colloc_ids)),
        ):
            await self._assets.delete_by_source_ids(session, source_kind, ids)

    async def _sync_senses(
        self,
        session: AsyncSession,
        word_id: int,
        senses: Iterable[GeneratedSense],
        cefr_map: dict[str, str],
    ) -> None:
        # GC cached assets for the OLD senses/examples/collocations before they are
        # bulk-deleted (their ids vanish after the delete). Same session → atomic.
        await self._gc_word_assets(session, word_id)
        # [Case 2 / F4] Before the old senses vanish, demote every INBOUND resolved
        # edge that points at one of them back to derived-pending: FK SET NULL will
        # clear ``to_sense_id`` on delete, but that alone leaves ``resolve_attempted_at``
        # set (→ derived unresolvable, stuck out of the queue). The helper also resets
        # it. [F13] It further re-queues inbound edges to THIS word that are currently
        # derived-unresolvable, so a better regeneration gets a fresh WSD attempt.
        old_sense_ids = list(
            (await session.execute(select(Sense.id).where(Sense.word_id == word_id)))
            .scalars()
            .all()
        )
        await self._demote_edges_for_senses(session, old_sense_ids)
        await self._requeue_unresolvable_inbound(session, word_id)
        # Deleting senses cascades to references+examples+collocations (ON DELETE CASCADE).
        await session.execute(delete(Sense).where(Sense.word_id == word_id))
        for order, gen_sense in enumerate(senses):
            cefr = self._resolve_cefr(gen_sense, cefr_map)
            sense = Sense(
                word_id=word_id,
                # definition is untrusted LM text in a non-null Text column: a NUL
                # crashes the Postgres INSERT (whole word rolls back to "error").
                # The themed path already cleans its definition; the neutral path
                # did not — route it through the same cleaner (control/NUL-strip).
                definition=self._clean(gen_sense.definition, self._MAX_DEFINITION),
                tier=gen_sense.tier,
                sense_order=order,
                pos=gen_sense.pos,
                cefr_level=cefr,
                # Enrichments: guideword is free LLM text -> sanitize; grammar is a
                # set of schema-validated tokens joined with ',' (no token contains
                # ',' — guarded by test); register/connotation are validated enum
                # values or None. Empty -> None so the read side yields [] / None.
                guideword=self._clean_opt(gen_sense.guideword, self._MAX_GUIDEWORD),
                grammar=",".join(gen_sense.grammar) or None,
                register=gen_sense.register,
                connotation=gen_sense.connotation,
                # IPA: copied from Cambridge when present, else LLM-generated for
                # out-of-Cambridge words. Untrusted free text either way -> route
                # through _clean (control-strip, incl. NUL that crashes Postgres).
                ipa_uk=self._clean_opt(gen_sense.ipa_uk, self._MAX_IPA),
                ipa_us=self._clean_opt(gen_sense.ipa_us, self._MAX_IPA),
                # domain (subject-area) + usage_note (one-line hint): free LLM text
                # -> sanitize; empty -> None so the read side yields None.
                domain=self._clean_opt(gen_sense.domain, self._MAX_DOMAIN),
                usage_note=self._clean_opt(gen_sense.usage_note, self._MAX_USAGE_NOTE),
            )
            session.add(sense)
            await session.flush()
            for ref in gen_sense.references:
                # source_ref is untrusted LM text in String(255): a NUL crashes the
                # Postgres INSERT and an over-length value raises "value too long".
                # Schema now caps it at 255 (max_length), but clean here too so a
                # NUL/control char never reaches the column on either dialect.
                session.add(
                    SenseReference(
                        sense_id=sense.id,
                        source=ref.source,
                        source_ref=self._clean(ref.source_ref, self._MAX_SOURCE_REF),
                    )
                )
            # Neutral examples: untrusted LM text in a non-null Text column -> clean
            # (NUL-strip) with the generous _MAX_EXAMPLE cap sized not to sever a
            # <t inf> tag. The themed/append paths already clean; the neutral path
            # did not. Empty/whitespace-only after cleaning is skipped (best-effort).
            for ex_order, ex in enumerate(gen_sense.examples):
                text = self._clean(ex, self._MAX_EXAMPLE)
                if not text:
                    continue
                session.add(Example(sense_id=sense.id, text=text, example_order=ex_order))
            # Collocations mirror examples: ordered child rows. text is a Text
            # column (unbounded) -> control-strip via _clean (Postgres NUL-safety)
            # with a GENEROUS sanity cap, not the tag's 255 (no DB need to truncate
            # a legit long collocation). Empty/whitespace-only is skipped (best-effort).
            for c_order, colloc in enumerate(gen_sense.collocations):
                text = self._clean(colloc, self._MAX_COLLOCATION)
                if not text:
                    continue
                session.add(Collocation(sense_id=sense.id, text=text, collocation_order=c_order))
            # Inflection forms mirror collocations: ordered child rows. ``surface``
            # is untrusted LLM text -> _clean (control-strip, incl NUL that crashes
            # Postgres); empty/whitespace-only is skipped. ``inf`` is a schema-validated
            # enum token, so it is stored verbatim.
            for f_order, form in enumerate(gen_sense.forms):
                surface = self._clean(form.surface, self._MAX_SURFACE)
                if not surface:
                    continue
                session.add(
                    SenseForm(sense_id=sense.id, inf=form.inf, surface=surface, form_order=f_order)
                )
            # Sense-level relations (synonym/antonym/hypernym/...) -> sense_relation
            # half-edges: from_sense = THIS sense, to_word = stub target, to_sense_id
            # NULL (WSD fills it later, Phase 4), gloss = LM description of the
            # target's intended meaning. resolve_attempted_at defaults NULL (Q1
            # derived: pending). [F12] gloss is load-bearing for WSD — an empty
            # gloss after _clean SKIPS the edge (no domed gloss='' rows).
            await self._link_sense_relations(session, sense, gen_sense.relations)
        await session.flush()

    async def _link_sense_relations(
        self, session: AsyncSession, sense: Sense, relations: Iterable["GeneratedSenseRelation"]
    ) -> None:
        """Persist a sense's sense-level relations as ``sense_relation`` half-edges.

        Each relation resolves its target lemma to a real (stub) ``words`` row via
        the shared ``_get_or_create_stub`` path (same normalization/dedup as
        word-level links), then inserts a ``SenseRelation`` keyed by the UNIQUE
        ``(from_sense_id, to_word_id, rel_type)`` triple. Skips:

        - [Case 8] a self-reference at sense level (target normalizes to THIS
          sense's own word — the relation would be vacuous);
        - [F12] an empty gloss after ``_clean`` (gloss is the load-bearing WSD
          signal; a domed empty gloss would only ever resolve wrong).
        """
        for rel in relations:
            gloss = self._clean(rel.gloss, self._MAX_GLOSS)
            if not gloss:
                continue  # [F12] gloss is load-bearing for WSD — no empty-gloss rows
            stub = await self._get_or_create_stub(session, rel.norm)
            if stub.id == sense.word_id:
                continue  # [Case 8] never a sense-level self-relation
            await self._ensure_sense_relation(session, sense.id, stub.id, rel.rel_type, gloss)
        await session.flush()

    async def _ensure_sense_relation(
        self, session: AsyncSession, from_sense_id: int, to_word_id: int, rel_type: str, gloss: str
    ) -> None:
        """Insert a ``sense_relation`` if the (from_sense, to_word, rel_type) triple
        is absent (dedup mirrors ``_ensure_link``). ``to_sense_id`` stays NULL
        (derived ``pending``); ``gloss`` is stored for the WSD pass."""
        exists = await session.execute(
            select(SenseRelation.id).where(
                SenseRelation.from_sense_id == from_sense_id,
                SenseRelation.to_word_id == to_word_id,
                SenseRelation.rel_type == rel_type,
            )
        )
        if exists.first() is None:
            session.add(
                SenseRelation(
                    from_sense_id=from_sense_id,
                    to_word_id=to_word_id,
                    rel_type=rel_type,
                    gloss=gloss,
                )
            )

    # --- sense-relation invalidation (Phase 5): Case 2 + F4 + F13 ---------

    async def _demote_edges_for_senses(
        self, session: AsyncSession, sense_ids: Sequence[int]
    ) -> None:
        """Demote every ``sense_relation`` resolved onto one of ``sense_ids`` back to
        derived-``pending`` ([Case 2] / [F4]).

        The ``to_sense_id`` FK is ``ON DELETE SET NULL``, so deleting a target sense
        already clears ``to_sense_id`` — but that alone leaves ``resolve_attempted_at``
        set, which reads as derived ``unresolvable`` (stuck OUT of the resolve queue,
        the F4 bug). This helper runs BEFORE the delete and explicitly resets all
        three columns (``to_sense_id`` / ``resolve_attempted_at`` / ``target_hash``)
        so the edge lands squarely in derived ``pending`` and the target's new senses
        get re-resolved. Called from ``_sync_senses`` (regenerate), ``delete_word``,
        and ``delete_entry`` — every path that removes a target sense.
        """
        if not sense_ids:
            return
        await session.execute(
            update(SenseRelation)
            .where(SenseRelation.to_sense_id.in_(sense_ids))
            .values(to_sense_id=None, resolve_attempted_at=None, target_hash=None)
        )

    async def _requeue_unresolvable_inbound(self, session: AsyncSession, word_id: int) -> None:
        """Re-queue derived-``unresolvable`` edges pointing AT ``word_id`` ([F13]).

        An ``unresolvable`` edge (``to_sense_id`` NULL, ``resolve_attempted_at`` set)
        is terminal for the resolve pass — but it often became unresolvable only
        because the target lacked the right POS/sense at the time. When the target
        word regenerates (presumably better), clearing ``resolve_attempted_at`` puts
        those edges back into derived ``pending`` so WSD retries them against the new
        content, avoiding a permanent false-negative. Only touches edges that are
        NOT resolved (``to_sense_id IS NULL``) so a live resolution is never disturbed.
        """
        await session.execute(
            update(SenseRelation)
            .where(
                SenseRelation.to_word_id == word_id,
                SenseRelation.to_sense_id.is_(None),
                SenseRelation.resolve_attempted_at.is_not(None),
            )
            .values(resolve_attempted_at=None)
        )

    # --- WSD resolve (Phase 4): work-queue read + conditional apply -------

    async def pending_relations_for_resolve(
        self, batch_size: int, word_ids: list[int] | None = None
    ) -> list["ResolveTask"]:
        """Fetch up to ``batch_size`` sense-relation edges ready for WSD.

        "Ready" = derived ``pending`` (``to_sense_id IS NULL AND
        resolve_attempted_at IS NULL``, Q1) whose target word is ``done`` and has
        ≥1 sense (Case 3). ``word_ids`` restricts to edges pointing AT those words
        — the inbound-resolve hook after a target flips ``done`` ([F11]); omit for
        the global backfill scan. Each task carries the source facts plus the
        target's candidate senses, ordered deterministically ``(sense_order, id)``
        and capped at ``_WSD_CANDIDATE_CAP`` ([F9]) so the exact ordering is
        reproducible at apply time ([F3]).
        """
        from_sense = aliased(Sense)
        async with session_scope(self._session_factory) as session:
            stmt = (
                select(
                    SenseRelation.id,
                    SenseRelation.rel_type,
                    SenseRelation.gloss,
                    SenseRelation.to_word_id,
                    from_sense.pos,
                    from_sense.definition,
                )
                .join(Word, Word.id == SenseRelation.to_word_id)
                .join(from_sense, from_sense.id == SenseRelation.from_sense_id)
                .where(
                    SenseRelation.to_sense_id.is_(None),
                    SenseRelation.resolve_attempted_at.is_(None),
                    Word.status == "done",
                    select(Sense.id).where(Sense.word_id == SenseRelation.to_word_id).exists(),
                )
                .order_by(SenseRelation.id)
                .limit(batch_size)
            )
            if word_ids is not None:
                if not word_ids:
                    return []
                stmt = stmt.where(SenseRelation.to_word_id.in_(word_ids))
            rows = (await session.execute(stmt)).all()
            tasks: list[ResolveTask] = []
            for edge_id, rel_type, gloss, to_word_id, src_pos, src_def in rows:
                cand_rows = (
                    await session.execute(
                        select(Sense.id, Sense.pos, Sense.definition)
                        .where(Sense.word_id == to_word_id)
                        .order_by(Sense.sense_order, Sense.id)
                        .limit(self._WSD_CANDIDATE_CAP)
                    )
                ).all()
                candidates = [
                    ResolveCandidate(sense_id=sid, pos=pos, definition=defn)
                    for sid, pos, defn in cand_rows
                ]
                tasks.append(
                    ResolveTask(
                        edge_id=edge_id,
                        rel_type=rel_type,
                        gloss=gloss,
                        source_def=src_def,
                        source_pos=src_pos,
                        candidates=candidates,
                    )
                )
            return tasks

    async def apply_resolutions(
        self, decisions: Iterable["ResolveDecision"]
    ) -> list["ResolveOutcome"]:
        """Apply judged decisions, each in its OWN savepoint ([F7]).

        A ``ResolveDecision`` carries the edge id and the CHOSEN target sense id
        (already bounds-validated by the caller, [F3]) or ``None`` (judge said no
        sense fits → mark unresolvable). Each edge is wrapped in ``begin_nested()``
        so one failing edge (e.g. its target sense vanished mid-batch, a racing
        regenerate) is isolated — its savepoint rolls back and it is reported as an
        error while the rest of the batch commits.

        The write is CONDITIONAL ([F6] TOCTOU): the UPDATE only fires while the
        edge is still derived-``pending`` and (for a resolve) the chosen sense
        still exists. A racing regenerate that already changed the edge or deleted
        the sense makes the UPDATE a no-op rather than writing a dead id.
        """
        outcomes: list[ResolveOutcome] = []
        async with session_scope(self._session_factory) as session:
            for dec in decisions:
                try:
                    async with session.begin_nested():
                        if dec.to_sense_id is None:
                            state = await self._mark_unresolvable(session, dec.edge_id)
                        else:
                            state = await self._apply_resolved(
                                session, dec.edge_id, dec.to_sense_id, dec.target_hash
                            )
                    outcomes.append(ResolveOutcome(edge_id=dec.edge_id, state=state))
                except Exception as exc:  # noqa: BLE001 - isolated per savepoint
                    outcomes.append(
                        ResolveOutcome(edge_id=dec.edge_id, state="error", error=str(exc))
                    )
        return outcomes

    async def _apply_resolved(
        self, session: AsyncSession, edge_id: int, to_sense_id: int, target_hash: str | None
    ) -> str:
        """Conditional UPDATE to ``resolved`` — no-op if the edge left pending or the
        target sense vanished ([F6]). Returns the derived state actually reached."""
        result = await session.execute(
            update(SenseRelation)
            .where(
                SenseRelation.id == edge_id,
                SenseRelation.to_sense_id.is_(None),
                SenseRelation.resolve_attempted_at.is_(None),
                select(Sense.id).where(Sense.id == to_sense_id).exists(),
            )
            .values(to_sense_id=to_sense_id, target_hash=target_hash)
        )
        return "resolved" if (cast("CursorResult", result).rowcount or 0) > 0 else "noop"

    async def _mark_unresolvable(self, session: AsyncSession, edge_id: int) -> str:
        """Conditional UPDATE stamping ``resolve_attempted_at`` (→ derived
        ``unresolvable``) — no-op if the edge already left pending ([F6])."""
        result = await session.execute(
            update(SenseRelation)
            .where(
                SenseRelation.id == edge_id,
                SenseRelation.to_sense_id.is_(None),
                SenseRelation.resolve_attempted_at.is_(None),
            )
            # Naive UTC (models._utcnow), NOT aware: every DateTime column is
            # TIMESTAMP WITHOUT TIME ZONE, and asyncpg raises DataError binding an
            # aware value to it. On Postgres the swallowed bind error at
            # apply_resolutions would else convert every unresolvable edge to
            # state="error" without stamping resolve_attempted_at, so it stays
            # derived-pending and is re-judged by the LLM forever. SQLite stores
            # the aware value as a string and never raises, hiding this.
            .values(resolve_attempted_at=_utcnow())
        )
        return "unresolvable" if (cast("CursorResult", result).rowcount or 0) > 0 else "noop"

    _WSD_CANDIDATE_CAP = WSD_CANDIDATE_CAP  # [F9] cap candidate senses per task (single source)

    # --- topic tags (resolve-or-create, deterministic dedup) --------------

    _MAX_TAG = 64
    _MAX_TITLE = 128
    _MAX_TAG_KEY = 255  # must match Tag.tag_key String(255) — NFKD can expand a key
    _MAX_GUIDEWORD = 64  # must match Sense.guideword String(64)
    _MAX_GLOSS = 255  # SenseRelation.gloss is Text (unbounded) — generous sanity cap only
    _MAX_IPA = 64  # must match Sense.ipa_uk/ipa_us String(64)
    _MAX_COLLOCATION = 512  # Collocation.text is Text (unbounded) — generous sanity cap only
    _MAX_SURFACE = 64  # SenseForm.surface is Text (unbounded) — generous sanity cap only
    _MAX_DOMAIN = 64  # must match Sense.domain String(64)
    _MAX_USAGE_NOTE = 255  # must match Sense.usage_note String(255)
    _MAX_THEME_NAME = 128  # Theme.name is Text (unbounded) — generous sanity cap only
    _MAX_THEME_KEY = 255  # must match Theme.theme_key String(255)
    _MAX_STYLE_PROMPT = 4000  # Theme.style_prompt is Text (unbounded) — generous sanity cap
    _MAX_THEMED_TEXT = 4000  # ThemedSense.definition / ThemedExample.text (Text) — generous cap
    # Example.text is Text (unbounded). Appended examples MUST carry <t inf> tags, so
    # the cap is generous (4000, matching themed) — a tight cap could sever a sentence
    # mid-tag and hand parse_marked_example unbalanced markup.
    _MAX_EXAMPLE = 4000
    _MAX_DEFINITION = 4000  # Sense.definition is Text (unbounded) — generous sanity cap only
    # norm / alias_norm are Text (unbounded) but feed match_key -> String(512); the
    # schema already bounds the LLM inputs to 128 (GeneratedEntry/Alias.norm), so this
    # generous cap only guards adversarial NFKD expansion, never a legit lemma.
    _MAX_NORM = 512
    _MAX_SOURCE_REF = 255  # must match SenseReference.source_ref String(255)

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

    @classmethod
    def _clean_opt(cls, s: str | None, cap: int) -> str | None:
        """``_clean`` for an OPTIONAL field: ``None``/empty in -> ``None`` out.

        Folds the six-copy ``self._clean(x, CAP) or None if x else None`` idiom
        (guideword, ipa_uk, ipa_us, domain, usage_note, ...) into one place. A
        value that cleans to empty (all control/whitespace) also collapses to
        ``None`` so the read side yields ``None``/``[]`` consistently.
        """
        if not s:
            return None
        return cls._clean(s, cap) or None

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
        """``(theme_id, style_prompt)`` for a ``theme_key`` or ``theme_id``, else ``None``.

        Resolution order (2.4 — key-first, then id-fallback for ``str``):

        - An ``int`` argument is ALWAYS an id lookup.
        - A ``str`` argument tries the ``theme_key`` lookup FIRST (normalized via
          ``theme_key`` exactly as ``create_theme`` stored it), and falls back to
          an ``int()``-by-id lookup ONLY on a key miss. This fixes the shadowing
          bug where a theme literally named "1984"/"007" was unaddressable by name
          (the old ``int()``-first path claimed it as an id), WITHOUT removing the
          stringified-id affordance JSON/HTTP callers (``?theme=42``) rely on: a
          "42" that is no theme's key still resolves by id.
        """
        async with session_scope(self._session_factory) as session:
            if isinstance(key_or_id, int):
                row = await session.execute(
                    select(Theme.id, Theme.style_prompt).where(Theme.id == key_or_id)
                )
                found = row.first()
                return (found[0], found[1]) if found is not None else None
            # str: key FIRST, so a numeric-named theme resolves by name.
            row = await session.execute(
                select(Theme.id, Theme.style_prompt).where(Theme.theme_key == theme_key(key_or_id))
            )
            found = row.first()
            if found is not None:
                return (found[0], found[1])
            # key miss: fall back to id-by-string ("42" -> id 42) if numeric.
            try:
                theme_id = int(key_or_id)
            except ValueError:
                return None
            row = await session.execute(
                select(Theme.id, Theme.style_prompt).where(Theme.id == theme_id)
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

    # --- targeted example augmentation ------------------------------------

    async def sense_context_for_examples(
        self, sense_id: int
    ) -> tuple[ExampleGenContext, list[str]] | None:
        """The ``(context, existing_examples)`` a targeted example generator needs
        for ONE sense, or ``None`` if the sense id is unknown.

        The context carries the sense's definition/pos/guideword/tier plus its
        ``(inf, surface)`` inflection paradigm; the existing example texts are
        returned alongside (fed back to the model for soft de-duplication, kept
        out of the context so it stays a pure fact carrier). Two ordered companion
        queries (forms, existing examples); no N+1."""
        async with session_scope(self._session_factory) as session:
            row = (
                await session.execute(
                    select(Sense.definition, Sense.pos, Sense.guideword, Sense.tier).where(
                        Sense.id == sense_id
                    )
                )
            ).first()
            if row is None:
                return None
            forms = (
                await session.execute(
                    select(SenseForm.inf, SenseForm.surface)
                    .where(SenseForm.sense_id == sense_id)
                    .order_by(SenseForm.form_order)
                )
            ).all()
            examples = (
                (
                    await session.execute(
                        select(Example.text)
                        .where(Example.sense_id == sense_id)
                        .order_by(Example.example_order)
                    )
                )
                .scalars()
                .all()
            )
            return ExampleGenContext(
                definition=row[0],
                pos=row[1],
                guideword=row[2],
                tier=row[3],
                forms=[(inf, surface) for inf, surface in forms],
            ), list(examples)

    async def append_examples(self, sense_id: int, texts: Sequence[str]) -> int:
        """Append cleaned, non-empty ``texts`` to a sense's examples at
        ``max(example_order) + 1``. Returns the count inserted.

        Never deletes or overwrites existing examples (contrast the whole-word
        overwrite path). Each text is ``_clean``-ed (control-strip incl. the NUL
        that crashes Postgres) with the same generous cap as neutral example text;
        empty/whitespace-only texts are skipped (mirrors collocation handling)."""
        async with session_scope(self._session_factory) as session:
            current_max = (
                await session.execute(
                    select(func.max(Example.example_order)).where(Example.sense_id == sense_id)
                )
            ).scalar_one_or_none()
            order = (current_max + 1) if current_max is not None else 0
            inserted = 0
            for text in texts:
                clean = self._clean(text, self._MAX_EXAMPLE)
                if not clean:
                    continue
                session.add(Example(sense_id=sense_id, text=clean, example_order=order))
                order += 1
                inserted += 1
            await session.flush()
            return inserted

    async def themed_overlay_for_sense(
        self, sense_id: int, theme_id: int
    ) -> tuple[int, list[str]] | None:
        """The ``(themed_sense_id, ordered_themed_example_texts)`` for one
        ``(sense, theme)`` overlay, or ``None`` when no themed row exists.

        ``None`` is the "theme the word first" signal — the api layer raises a
        ``ValueError`` telling the caller to run ``generate(theme=)`` before
        augmenting themed examples (never silently themes the whole word)."""
        async with session_scope(self._session_factory) as session:
            row = (
                await session.execute(
                    select(ThemedSense.id).where(
                        ThemedSense.sense_id == sense_id,
                        ThemedSense.theme_id == theme_id,
                    )
                )
            ).first()
            if row is None:
                return None
            themed_sense_id = row[0]
            texts = (
                (
                    await session.execute(
                        select(ThemedExample.text)
                        .where(ThemedExample.themed_sense_id == themed_sense_id)
                        .order_by(ThemedExample.example_order)
                    )
                )
                .scalars()
                .all()
            )
            return themed_sense_id, list(texts)

    async def word_id_for_sense(self, sense_id: int) -> int:
        """The owning ``word_id`` for a sense (used to overlay a themed read via
        :meth:`themed_for_word`). Assumes the sense exists (the caller validated
        it via :meth:`sense_context_for_examples`)."""
        async with session_scope(self._session_factory) as session:
            return (
                await session.execute(select(Sense.word_id).where(Sense.id == sense_id))
            ).scalar_one()

    async def stats(self) -> Stats:
        """Point-in-time dictionary counts in one session (no LLM, no N+1).

        Words grouped by status and assets grouped by kind come back as dicts;
        everything else is a scalar COUNT. ``themed_words`` counts distinct words
        with at least one themed overlay (distinct ``Sense.word_id`` over
        ``ThemedSense``). Counts are a snapshot, not cross-query txn-isolated —
        acceptable for a stats surface."""
        async with session_scope(self._session_factory) as session:
            status_rows = await session.execute(
                select(Word.status, func.count(Word.id)).group_by(Word.status)
            )
            words_by_status = {status: count for status, count in status_rows}
            asset_rows = await session.execute(
                select(Asset.kind, func.count(Asset.id)).group_by(Asset.kind)
            )
            assets_by_kind = {kind: count for kind, count in asset_rows}
            senses = (await session.execute(select(func.count(Sense.id)))).scalar_one()
            examples = (await session.execute(select(func.count(Example.id)))).scalar_one()
            tags = (await session.execute(select(func.count(Tag.id)))).scalar_one()
            themes = (await session.execute(select(func.count(Theme.id)))).scalar_one()
            themed_words = (
                await session.execute(
                    select(func.count(func.distinct(Sense.word_id)))
                    .select_from(ThemedSense)
                    .join(Sense, Sense.id == ThemedSense.sense_id)
                )
            ).scalar_one()
            questions = (await session.execute(select(func.count(Question.id)))).scalar_one()
        return Stats(
            words_by_status=words_by_status,
            senses=senses,
            examples=examples,
            tags=tags,
            themes=themes,
            themed_words=themed_words,
            assets_by_kind=assets_by_kind,
            questions=questions,
        )

    async def append_themed_examples(self, themed_sense_id: int, texts: Sequence[str]) -> int:
        """Append cleaned, non-empty ``texts`` to a themed sense's examples at
        ``max(example_order) + 1``. Returns the count inserted.

        Themed mirror of :meth:`append_examples`; never overwrites existing
        themed examples. Each text is ``_clean``-ed with the same generous cap as
        the whole-word themed path; empty/whitespace-only texts are skipped."""
        async with session_scope(self._session_factory) as session:
            current_max = (
                await session.execute(
                    select(func.max(ThemedExample.example_order)).where(
                        ThemedExample.themed_sense_id == themed_sense_id
                    )
                )
            ).scalar_one_or_none()
            order = (current_max + 1) if current_max is not None else 0
            inserted = 0
            for text in texts:
                clean = self._clean(text, self._MAX_THEMED_TEXT)
                if not clean:
                    continue
                session.add(
                    ThemedExample(themed_sense_id=themed_sense_id, text=clean, example_order=order)
                )
                order += 1
                inserted += 1
            await session.flush()
            return inserted

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
        """Insert a word_relation if the (from, to, rel_type) triple is absent."""
        exists = await session.execute(
            select(WordRelation.id).where(
                WordRelation.from_word_id == from_id,
                WordRelation.to_word_id == to_id,
                WordRelation.rel_type == rel_type,
            )
        )
        if exists.first() is None:
            session.add(WordRelation(from_word_id=from_id, to_word_id=to_id, rel_type=rel_type))

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
