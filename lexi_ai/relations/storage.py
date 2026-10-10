"""Complete candidate inventories and snapshot-bound Sense Linking revalidation."""

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import and_, select, true, update

from lexi_ai.db.collections import collection
from lexi_ai.schema import Definition, Sense, SenseRelation, Word
from lexi_ai.vocab import POS_TAGS, GenerationState


@dataclass(frozen=True)
class Candidate:
    id: int
    definition: str | None
    fingerprint: str | None
    pos: str


@dataclass(frozen=True)
class PendingLink:
    edge_id: int
    source_word: str
    source_definition: str | None
    source_pos: str
    relation_type: str
    target_word: str
    candidates: list[Candidate]


def definition_hash(definition: str) -> str:
    return hashlib.sha256(definition.encode()).hexdigest()


def _has_senses(word_id):
    """An ordered existence probe on the (word_id, pos, id) index."""
    return (
        select(Sense.id)
        .where(Sense.word_id == word_id)
        .order_by(Sense.pos, Sense.id)
        .limit(1)
        .scalar_subquery()
        .is_not(None)
    )


def _candidates(word_id, pos, *, correlate):
    definitions = Definition.__table__.alias("candidate_definition")
    definition = (
        select(definitions.c.content)
        .where(definitions.c.sense_id == Sense.id, definitions.c.theme_id.is_(None))
        .correlate(Sense.__table__)
        .scalar_subquery()
    )
    return collection(
        Sense,
        {"id": Sense.id, "pos": Sense.pos, "definition": definition},
        Sense.word_id == word_id,
        Sense.pos == pos,
        order_by=(Sense.word_id, Sense.pos, Sense.id),
        correlate=correlate,
    )


def _projection(*, with_candidates=False):
    source = Sense.__table__.alias("source_sense")
    source_word = Word.__table__.alias("source_word")
    target = Word.__table__.alias("target_word")
    definition = Definition.__table__.alias("source_definition")
    statement = (
        select(
            SenseRelation.id,
            SenseRelation.rel_type,
            SenseRelation.to_sense_id,
            SenseRelation.target_hash,
            SenseRelation.to_word_id,
            source.c.pos,
            source_word.c.lemma.label("source_word"),
            target.c.lemma.label("target_word"),
            target.c.generation_state,
            definition.c.content.label("source_definition"),
        )
        .join(source, source.c.id == SenseRelation.from_sense_id)
        .join(source_word, source_word.c.id == source.c.word_id)
        .join(target, target.c.id == SenseRelation.to_word_id)
        .outerjoin(
            definition, and_(definition.c.sense_id == source.c.id, definition.c.theme_id.is_(None))
        )
    )
    statement = statement.where(
        target.c.generation_state == GenerationState.DONE,
        _has_senses(target.c.id),
    )
    if with_candidates:
        statement = statement.add_columns(
            _candidates(target.c.id, source.c.pos, correlate=(source, target)).label("candidates")
        )
    return statement


def _pending_targets(last_id):
    """Walk distinct pending targets by index seeks, not by reading every edge.

    PostgreSQL 16 has no native loose index scan. The recursive step jumps past
    the entire previous target range; a final NULL terminates the walk. This also
    avoids scanning unrelated Words just to discover that they have no work.
    """
    remaining = select(SenseRelation.to_word_id).where(
        SenseRelation.resolve_attempted_at.is_(None),
        SenseRelation.id > last_id,
    )
    first = remaining.order_by(SenseRelation.to_word_id).limit(1).scalar_subquery()
    targets = select(first.label("word_id")).cte("pending_targets", recursive=True)
    following = (
        remaining.where(SenseRelation.to_word_id > targets.c.word_id)
        .order_by(SenseRelation.to_word_id)
        .limit(1)
        .correlate(targets)
        .scalar_subquery()
    )
    return targets.union_all(select(following).where(targets.c.word_id.is_not(None)))


def _pending_ids(last_id, size, *, postgres):
    """Probe eligible targets, rather than walking deferred edges in global ID order."""
    targets = _pending_targets(last_id)
    target_state = (
        select(Word.generation_state).where(Word.id == targets.c.word_id).scalar_subquery()
    )
    ready = (
        select(targets.c.word_id.label("id"))
        .where(
            target_state == GenerationState.DONE,
            _has_senses(targets.c.word_id),
        )
        .cte("ready_targets")
        .prefix_with("MATERIALIZED", dialect="postgresql")
    )
    pending = [SenseRelation.resolve_attempted_at.is_(None), SenseRelation.id > last_id]
    per_target = (
        select(SenseRelation.id)
        .where(*pending, SenseRelation.to_word_id == ready.c.id)
        .order_by(SenseRelation.id)
        .limit(size)
        .correlate(ready)
    )
    if postgres:
        # Each target contributes at most size edges to the global top-size merge.
        bounded = per_target.lateral("target_pending")
        return (
            select(bounded.c.id)
            .select_from(ready.join(bounded, true()))
            .order_by(bounded.c.id)
            .limit(size)
        )
    return (
        select(SenseRelation.id)
        .select_from(ready.join(SenseRelation.__table__, SenseRelation.id.in_(per_target)))
        .order_by(SenseRelation.id)
        .limit(size)
    )


def _pending_page(last_id, size, *, postgres):
    chosen = _pending_ids(last_id, size, postgres=postgres).cte("pending_ids")
    page = (
        _projection()
        .join(chosen, chosen.c.id == SenseRelation.id)
        .cte("pending_page")
        .prefix_with("MATERIALIZED", dialect="postgresql")
    )
    scopes = (
        select(page.c.to_word_id, page.c.pos)
        .distinct()
        .cte("candidate_scopes")
        .prefix_with("MATERIALIZED", dialect="postgresql")
    )
    return select(
        collection(page, dict(page.c.items()), order_by=(page.c.id,)),
        collection(
            scopes,
            {
                "word_id": scopes.c.to_word_id,
                "pos": scopes.c.pos,
                "candidates": _candidates(scopes.c.to_word_id, scopes.c.pos, correlate=(scopes,)),
            },
            order_by=(scopes.c.to_word_id, scopes.c.pos),
        ),
    )


def _candidate_values(values):
    return [
        Candidate(
            item["id"],
            item["definition"],
            definition_hash(item["definition"]) if item["definition"] is not None else None,
            item["pos"],
        )
        for item in values
    ]


async def pending_relations(db, batch_size):
    result, last_id = [], 0
    async with db.read() as connection:
        while len(result) < batch_size:
            page_size = batch_size - len(result)
            values, evidence = (
                await connection.execute(
                    _pending_page(
                        last_id,
                        page_size,
                        postgres=db.engine.dialect.name == "postgresql",
                    )
                )
            ).one()
            candidates_by_scope = {
                (item["word_id"], item["pos"]): _candidate_values(item["candidates"])
                for item in evidence
            }
            if not values:
                break
            for value in values:
                last_id = value["id"]
                definition = value["source_definition"]
                candidates = candidates_by_scope[value["to_word_id"], value["pos"]]
                if any(
                    c.id == value["to_sense_id"] and c.fingerprint == value["target_hash"]
                    for c in candidates
                ):
                    continue
                result.append(
                    PendingLink(
                        value["id"],
                        value["source_word"],
                        definition,
                        value["pos"],
                        value["rel_type"],
                        value["target_word"],
                        list(candidates),
                    )
                )
            if len(values) < page_size:
                break
    return result


async def apply_resolution(db, edge_id, candidate, *, expected=None):
    async with db.transaction(immediate=True) as session:
        # Lock first, THEN read evidence in a new READ COMMITTED statement snapshot.
        # Combining projection with the lock can read stale evidence before a lock wait.
        locked = await session.scalar(
            select(SenseRelation.id).where(SenseRelation.id == edge_id).with_for_update()
        )
        if locked is None:
            return False
        value = (
            (
                await session.execute(
                    _projection(with_candidates=True).where(SenseRelation.id == edge_id)
                )
            )
            .mappings()
            .first()
        )
        if value is None:
            return False
        if value["pos"] not in POS_TAGS or value["source_definition"] is None:
            return False
        current = _candidate_values(value["candidates"])
        if any(c.definition is None for c in current):
            return False
        if expected is not None:
            if (
                value["source_word"] != expected.source_word
                or value["rel_type"] != expected.relation_type
                or value["pos"] != expected.source_pos
                or value["target_word"] != expected.target_word
                or value["source_definition"] != expected.source_definition
                or [(c.id, c.fingerprint) for c in current]
                != [(c.id, c.fingerprint) for c in expected.candidates]
            ):
                return False
        if candidate is not None and not any(
            c.id == candidate.id and c.fingerprint == candidate.fingerprint for c in current
        ):
            return False
        await session.execute(
            update(SenseRelation)
            .where(SenseRelation.id == edge_id)
            .values(
                to_sense_id=candidate.id if candidate else None,
                target_hash=candidate.fingerprint if candidate else None,
                resolve_attempted_at=datetime.now(UTC).isoformat(),
            )
        )
    if db.content_cache is not None:
        db.content_cache.clear()
    return True
