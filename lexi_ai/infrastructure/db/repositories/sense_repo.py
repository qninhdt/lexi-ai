"""The ``senses`` aggregate: senses and everything hanging off one sense.

Async safety: relationship collections are never touched on a persistent object,
because that triggers a lazy load outside greenlet context. Children are cleared
with Core deletes and re-inserted with explicit foreign keys.
"""

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, cast

from sqlalchemy import CursorResult, delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import aliased

from lexi_ai.assets.repository import AssetRepository
from lexi_ai.constants import WSD_CANDIDATE_CAP, canonical_cambridge_ref
from lexi_ai.domain.models import (
    ResolveCandidate,
    ResolveDecision,
    ResolveOutcome,
    ResolveTask,
    SemanticSenseRow,
    SenseEmbeddingNeed,
    ThemingSense,
)
from lexi_ai.generation.schemas import ExampleGenContext
from lexi_ai.infrastructure.db.asset_gc import collect_word_assets
from lexi_ai.infrastructure.db.models import (
    Collocation,
    Example,
    Sense,
    SenseForm,
    SenseReference,
    SenseRelation,
    Word,
    _utcnow,
)
from lexi_ai.infrastructure.db.repositories.word_repo import SqlWordRepo
from lexi_ai.infrastructure.db.sanitize import (
    MAX_COLLOCATION,
    MAX_DEFINITION,
    MAX_DOMAIN,
    MAX_EXAMPLE,
    MAX_GLOSS,
    MAX_GUIDEWORD,
    MAX_IPA,
    MAX_SOURCE_REF,
    MAX_SURFACE,
    MAX_USAGE_NOTE,
    clean,
    clean_opt,
)

if TYPE_CHECKING:
    from lexi_ai.generation.schemas import GeneratedSense, GeneratedSenseRelation


class SqlSenseRepo:
    """Session-bound implementation of :class:`lexi_ai.domain.ports.SenseRepo`."""

    def __init__(
        self,
        session: AsyncSession,
        words: SqlWordRepo,
        assets: AssetRepository | None = None,
    ) -> None:
        self._session = session
        self._words = words
        self._assets = assets

    async def sync(
        self, word_id: int, senses: Iterable["GeneratedSense"], cefr_map: dict[str, str]
    ) -> None:
        """Replace a word's senses and every child row beneath them.

        Two invalidation steps must run BEFORE the old senses vanish:

        * Inbound resolved edges pointing at a disappearing sense are demoted back
          to pending. The foreign key clears ``to_sense_id`` on delete, but that
          alone leaves ``resolve_attempted_at`` set, which reads as unresolvable and
          strands the edge outside the queue forever.
        * Inbound edges that are currently unresolvable are re-queued, so a better
          regeneration gets a fresh attempt instead of a permanent false negative.

        Cached assets are collected first too, since the child ids vanish with the
        delete.
        """
        await collect_word_assets(self._session, self._assets, word_id)
        old_sense_ids = list(
            (await self._session.execute(select(Sense.id).where(Sense.word_id == word_id)))
            .scalars()
            .all()
        )
        await self._demote_edges_for_senses(old_sense_ids)
        await self._requeue_unresolvable_inbound(word_id)
        # Deleting senses cascades to references, examples, collocations, and forms.
        await self._session.execute(delete(Sense).where(Sense.word_id == word_id))
        for order, generated in enumerate(senses):
            sense = self._build_sense(word_id, order, generated, cefr_map)
            self._session.add(sense)
            await self._session.flush()
            self._add_children(sense.id, generated)
            await self._link_sense_relations(sense.id, word_id, generated.relations)
        await self._session.flush()

    async def word_id_for(self, sense_id: int) -> int:
        """Owning word id. Raises when the sense does not exist."""
        return (
            await self._session.execute(select(Sense.word_id).where(Sense.id == sense_id))
        ).scalar_one()

    async def needing_embedding(
        self,
        word_ids: list[int] | None = None,
        limit: int | None = None,
        after_sense_id: int | None = None,
    ) -> list[SenseEmbeddingNeed]:
        """Done senses that are candidates for embedding, oldest id first.

        This store no longer knows which senses carry a vector — the index does —
        so every done sense is a candidate and the caller subtracts what the index
        already holds. Restrict to ``word_ids`` right after generation; omit them
        for a global backfill.

        ``after_sense_id`` resumes past an id already examined, which is what lets
        a caller honour a ``limit`` without loading the whole table. Because the
        already-embedded set lives in the index rather than here, a plain
        ``LIMIT`` could return a page that is entirely embedded already and read
        as "nothing left to do"; paging with this cursor lets the caller keep
        asking until it has enough genuinely-unembedded rows.
        """
        stmt = (
            select(Sense.id, Word.norm, Sense.definition)
            .join(Word, Word.id == Sense.word_id)
            .where(Word.status == "done")
            .order_by(Sense.id)
        )
        if word_ids is not None:
            if not word_ids:
                return []
            stmt = stmt.where(Sense.word_id.in_(word_ids))
        if after_sense_id is not None:
            stmt = stmt.where(Sense.id > after_sense_id)
        if limit is not None:
            stmt = stmt.limit(limit)
        rows = await self._session.execute(stmt)
        return [SenseEmbeddingNeed(sid, norm, definition) for sid, norm, definition in rows]

    async def semantic_rows(self, sense_ids: Sequence[int]) -> list[SemanticSenseRow]:
        """Presentation rows for ranked sense ids, in the order requested.

        Unknown ids are skipped, which is what makes a stale vector harmless: an id
        left behind by a delete or a regeneration drops out of the results instead
        of raising.
        """
        if not sense_ids:
            return []
        rows = await self._session.execute(
            select(
                Sense.id,
                Sense.word_id,
                Word.norm,
                Word.entry_type,
                Sense.definition,
                Sense.tier,
            )
            .join(Word, Word.id == Sense.word_id)
            .where(Word.status == "done", Sense.id.in_(list(sense_ids)))
        )
        by_id = {row[0]: SemanticSenseRow(*row) for row in rows}
        return [by_id[sense_id] for sense_id in sense_ids if sense_id in by_id]

    async def live_sense_ids(self) -> set[int]:
        """Every existing sense id, for pruning vectors whose sense is gone."""
        rows = await self._session.execute(select(Sense.id))
        return {sense_id for (sense_id,) in rows}

    async def ids_for_word(self, word_id: int) -> list[int]:
        """Every sense id of one word, whatever the word's status.

        Read before a delete, so the caller can forget the word's vectors after the
        rows are gone. Status-agnostic on purpose: a pending word can still own
        senses from an earlier successful generation.
        """
        rows = await self._session.execute(
            select(Sense.id).where(Sense.word_id == word_id).order_by(Sense.id)
        )
        return [sense_id for (sense_id,) in rows]

    async def pending_relations(
        self, batch_size: int, word_ids: list[int] | None = None
    ) -> list[ResolveTask]:
        """Up to ``batch_size`` sense-relation edges ready for disambiguation.

        Ready means the edge is pending (no target sense, no attempt recorded) and
        its target word is done with at least one sense. ``word_ids`` restricts to
        edges pointing at those words, which is the hook that runs after a target
        becomes done; omit it for the global backfill.

        Candidate senses are ordered deterministically and capped, because the
        caller validates the judge's answer against exactly this ordering.
        """
        from_sense = aliased(Sense)
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
        rows = (await self._session.execute(stmt)).all()
        tasks: list[ResolveTask] = []
        for edge_id, rel_type, gloss, to_word_id, source_pos, source_def in rows:
            # One bounded candidate query per task, by design. Batch sizes are small
            # and clamped, so this is at worst a few dozen short selects per pass. A
            # windowed rewrite would risk the deterministic ordering and per-word cap
            # that apply-time bounds validation depends on.
            candidate_rows = (
                await self._session.execute(
                    select(Sense.id, Sense.pos, Sense.definition)
                    .where(Sense.word_id == to_word_id)
                    .order_by(Sense.sense_order, Sense.id)
                    .limit(WSD_CANDIDATE_CAP)
                )
            ).all()
            tasks.append(
                ResolveTask(
                    edge_id=edge_id,
                    rel_type=rel_type,
                    gloss=gloss,
                    source_def=source_def,
                    source_pos=source_pos,
                    candidates=[
                        ResolveCandidate(sense_id=sid, pos=pos, definition=definition)
                        for sid, pos, definition in candidate_rows
                    ],
                )
            )
        return tasks

    async def apply_resolutions(self, decisions: Iterable[ResolveDecision]) -> list[ResolveOutcome]:
        """Apply judged decisions, each inside its own savepoint.

        Isolating each edge means one poison pill (its target sense vanished
        mid-batch, say) is reported as an error while the rest of the batch still
        commits.

        Each write is conditional: it only fires while the edge is still pending and,
        for a resolve, the chosen sense still exists. A racing regeneration turns the
        update into a no-op rather than writing a dead id.
        """
        outcomes: list[ResolveOutcome] = []
        for decision in decisions:
            try:
                async with self._session.begin_nested():
                    if decision.to_sense_id is None:
                        state = await self._mark_unresolvable(decision.edge_id)
                    else:
                        state = await self._apply_resolved(
                            decision.edge_id, decision.to_sense_id, decision.target_hash
                        )
                outcomes.append(ResolveOutcome(edge_id=decision.edge_id, state=state))
            except Exception as exc:  # noqa: BLE001 - isolated per savepoint
                outcomes.append(
                    ResolveOutcome(edge_id=decision.edge_id, state="error", error=str(exc))
                )
        return outcomes

    async def example_context(self, sense_id: int) -> tuple[ExampleGenContext, list[str]] | None:
        """The context and existing examples a targeted generator needs for one sense.

        Existing example texts come back alongside the context rather than inside it,
        so the context stays a pure fact carrier while the caller can still feed the
        texts back for soft de-duplication. Returns ``None`` for an unknown sense.
        """
        row = (
            await self._session.execute(
                select(Sense.definition, Sense.pos, Sense.guideword, Sense.tier).where(
                    Sense.id == sense_id
                )
            )
        ).first()
        if row is None:
            return None
        forms = (
            await self._session.execute(
                select(SenseForm.inf, SenseForm.surface)
                .where(SenseForm.sense_id == sense_id)
                .order_by(SenseForm.form_order)
            )
        ).all()
        examples = (
            (
                await self._session.execute(
                    select(Example.text)
                    .where(Example.sense_id == sense_id)
                    .order_by(Example.example_order)
                )
            )
            .scalars()
            .all()
        )
        context = ExampleGenContext(
            definition=row[0],
            pos=row[1],
            guideword=row[2],
            tier=row[3],
            forms=[(inf, surface) for inf, surface in forms],
        )
        return context, list(examples)

    async def append_examples(self, sense_id: int, texts: Sequence[str]) -> int:
        """Append cleaned, non-empty texts after the current highest order.

        Never deletes or overwrites existing examples, unlike the whole-word path.
        """
        current_max = (
            await self._session.execute(
                select(func.max(Example.example_order)).where(Example.sense_id == sense_id)
            )
        ).scalar_one_or_none()
        order = (current_max + 1) if current_max is not None else 0
        inserted = 0
        for text in texts:
            cleaned = clean(text, MAX_EXAMPLE)
            if not cleaned:
                continue
            self._session.add(Example(sense_id=sense_id, text=cleaned, example_order=order))
            order += 1
            inserted += 1
        await self._session.flush()
        return inserted

    async def for_theming(self, word_id: int) -> list[ThemingSense]:
        """A word's senses in deterministic prompt order.

        The caller passes these ids back when persisting the themed result in the
        same order it numbered them, so themed index ``i`` maps to sense ``i``.
        """
        rows = await self._session.execute(
            select(Sense.id, Sense.definition, Sense.pos, Sense.guideword, Sense.tier)
            .where(Sense.word_id == word_id)
            .order_by(Sense.sense_order, Sense.id)
        )
        return [ThemingSense(*row) for row in rows]

    def _build_sense(
        self, word_id: int, order: int, generated: "GeneratedSense", cefr_map: dict[str, str]
    ) -> Sense:
        """Map one generated sense onto a row, cleaning every free-text field.

        ``grammar`` is a set of schema-validated tokens; the column type validates
        them again on write and handles the encoding, so no caller joins by hand.
        ``register`` and ``connotation`` are validated enum values or ``None``.
        """
        return Sense(
            word_id=word_id,
            definition=clean(generated.definition, MAX_DEFINITION),
            tier=generated.tier,
            sense_order=order,
            pos=generated.pos,
            cefr_level=self._resolve_cefr(generated, cefr_map),
            guideword=clean_opt(generated.guideword, MAX_GUIDEWORD),
            grammar=list(generated.grammar),
            register=generated.register,
            connotation=generated.connotation,
            ipa_uk=clean_opt(generated.ipa_uk, MAX_IPA),
            ipa_us=clean_opt(generated.ipa_us, MAX_IPA),
            domain=clean_opt(generated.domain, MAX_DOMAIN),
            usage_note=clean_opt(generated.usage_note, MAX_USAGE_NOTE),
        )

    def _add_children(self, sense_id: int, generated: "GeneratedSense") -> None:
        """Insert the ordered child rows of one sense.

        Every text field is untrusted model output, so each is cleaned; a value that
        cleans away entirely is skipped rather than stored blank.
        """
        for reference in generated.references:
            self._session.add(
                SenseReference(
                    sense_id=sense_id,
                    source=reference.source,
                    source_ref=clean(reference.source_ref, MAX_SOURCE_REF),
                )
            )
        for order, example in enumerate(generated.examples):
            text = clean(example, MAX_EXAMPLE)
            if text:
                self._session.add(Example(sense_id=sense_id, text=text, example_order=order))
        for order, collocation in enumerate(generated.collocations):
            text = clean(collocation, MAX_COLLOCATION)
            if text:
                self._session.add(
                    Collocation(sense_id=sense_id, text=text, collocation_order=order)
                )
        for order, form in enumerate(generated.forms):
            surface = clean(form.surface, MAX_SURFACE)
            if surface:
                self._session.add(
                    SenseForm(sense_id=sense_id, inf=form.inf, surface=surface, form_order=order)
                )

    async def _link_sense_relations(
        self, sense_id: int, word_id: int, relations: Iterable["GeneratedSenseRelation"]
    ) -> None:
        """Persist a sense's relations as half-edges awaiting disambiguation.

        Each relation resolves its target lemma to a real (stub) word through the
        shared path, then inserts an edge keyed by the unique
        ``(from_sense, to_word, rel_type)`` triple with ``to_sense_id`` still null.

        Two skips: a target that normalizes to this sense's own word would be a
        vacuous self-relation, and an empty gloss is dropped because the gloss is the
        load-bearing disambiguation signal and a blank one could only ever resolve
        wrong.
        """
        for relation in relations:
            gloss = clean(relation.gloss, MAX_GLOSS)
            if not gloss:
                continue
            target_id = await self._words.get_or_create(relation.norm)
            if target_id == word_id:
                continue
            await self._ensure_sense_relation(sense_id, target_id, relation.rel_type, gloss)
        await self._session.flush()

    async def _ensure_sense_relation(
        self, from_sense_id: int, to_word_id: int, rel_type: str, gloss: str
    ) -> None:
        exists = await self._session.execute(
            select(SenseRelation.id).where(
                SenseRelation.from_sense_id == from_sense_id,
                SenseRelation.to_word_id == to_word_id,
                SenseRelation.rel_type == rel_type,
            )
        )
        if exists.first() is None:
            self._session.add(
                SenseRelation(
                    from_sense_id=from_sense_id,
                    to_word_id=to_word_id,
                    rel_type=rel_type,
                    gloss=gloss,
                )
            )

    async def _demote_edges_for_senses(self, sense_ids: Sequence[int]) -> None:
        """Return every edge resolved onto ``sense_ids`` to pending.

        Resetting all three columns is what distinguishes pending from unresolvable;
        relying on the foreign key alone would leave the attempt timestamp set.
        """
        if not sense_ids:
            return
        await self._session.execute(
            update(SenseRelation)
            .where(SenseRelation.to_sense_id.in_(sense_ids))
            .values(to_sense_id=None, resolve_attempted_at=None, target_hash=None)
        )

    async def _requeue_unresolvable_inbound(self, word_id: int) -> None:
        """Re-queue unresolvable edges pointing at ``word_id``.

        An edge often became unresolvable only because the target lacked the right
        part of speech or sense at the time. Only untouched (unresolved) edges are
        affected, so a live resolution is never disturbed.
        """
        await self._session.execute(
            update(SenseRelation)
            .where(
                SenseRelation.to_word_id == word_id,
                SenseRelation.to_sense_id.is_(None),
                SenseRelation.resolve_attempted_at.is_not(None),
            )
            .values(resolve_attempted_at=None)
        )

    async def _apply_resolved(self, edge_id: int, to_sense_id: int, target_hash: str | None) -> str:
        result = await self._session.execute(
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

    async def _mark_unresolvable(self, edge_id: int) -> str:
        result = await self._session.execute(
            update(SenseRelation)
            .where(
                SenseRelation.id == edge_id,
                SenseRelation.to_sense_id.is_(None),
                SenseRelation.resolve_attempted_at.is_(None),
            )
            # Naive UTC, not aware: every DateTime column is TIMESTAMP WITHOUT TIME
            # ZONE, and asyncpg raises binding an aware value to it. That error would
            # be swallowed per savepoint, leaving the edge pending and re-judged by
            # the LLM forever. SQLite stores an aware value as text and never raises,
            # which hides the defect entirely.
            .values(resolve_attempted_at=_utcnow())
        )
        return "unresolvable" if (cast("CursorResult", result).rowcount or 0) > 0 else "noop"

    @staticmethod
    def _resolve_cefr(generated: "GeneratedSense", cefr_map: dict[str, str]) -> str | None:
        """Cambridge-first CEFR: a Cambridge reference wins over the model's guess.

        Both the reference id and the map keys are canonicalized, so the labelled and
        bare forms collapse to one key and the rule cannot fall through when the model
        echoes the form it was shown.
        """
        for reference in generated.references:
            if reference.source == "cambridge":
                key = canonical_cambridge_ref(reference.source_ref)
                if key in cefr_map:
                    return cefr_map[key]
        return generated.cefr_level
