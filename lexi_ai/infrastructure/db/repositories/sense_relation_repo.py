"""Word-sense disambiguation: the relation edges and how they are resolved.

Split out of `sense_repo.py`, which did four jobs because nothing said where a
repository for the senses aggregate should stop. This is the one whose vocabulary
is entirely its own — pending edges, candidate senses, judged decisions,
unresolvable outcomes — and none of it appears anywhere else in that file.

A mixin rather than a separate repository, deliberately. These methods are on the
`SenseRepo` port and callers reach them through it, so extracting them into a
class of their own would either change the port or add a delegating method per
call. Composing them back onto `SqlSenseRepo` keeps the port exactly as it was:
this is a file boundary, not an interface change.

The mixin depends on `self._session` from its host, which is the one coupling it
cannot shed without becoming that separate repository.
"""

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, cast

from sqlalchemy import CursorResult, delete, func, select, update
from sqlalchemy.orm import aliased

from lexi_ai.constants import WSD_CANDIDATE_CAP
from lexi_ai.domain.models import (
    ResolveCandidate,
    ResolveDecision,
    ResolveOutcome,
    ResolveTask,
)
from lexi_ai.infrastructure.db.models import Sense, SenseRelation, Word, _utcnow
from lexi_ai.infrastructure.db.sanitize import MAX_GLOSS, clean

if TYPE_CHECKING:
    from lexi_ai.generation.schemas import GeneratedSenseRelation


class SenseRelationResolutionMixin:
    """Relation disambiguation for `SqlSenseRepo`. See the module docstring."""

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
