import asyncio
from types import SimpleNamespace

import pytest
from sqlalchemy import select

from lexi_ai.db.session import Database
from lexi_ai.relations.resolve import resolve_relations
from lexi_ai.relations.storage import definition_hash, pending_relations
from lexi_ai.schema import Base, Definition, Sense, SenseRelation, Word
from lexi_ai.words.storage import get_word


class Decision:
    def __init__(self, choice):
        self.choice = choice
        self.calls = []

    async def decide(self, state, questions, **kwargs):
        self.calls.append((state, questions))
        return SimpleNamespace(choices={"matched_sense": SimpleNamespace(choice=self.choice)})


async def test_candidate_pos_filter_before_cap_and_explicit_resolution(tmp_path):
    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'generated.db'}")
    try:
        await db.create_schema(Base.metadata)
        async with db.transaction() as session:
            source = Word(
                lemma="glisten", match_key="glisten", entry_type="word", generation_state="done"
            )
            target = Word(
                lemma="shine", match_key="shine", entry_type="word", generation_state="done"
            )
            session.add_all([source, target])
            await session.flush()
            from_sense = Sense(word_id=source.id, pos="verb", tier="core")
            session.add(from_sense)
            for _ in range(12):
                session.add(Sense(word_id=target.id, pos="noun", tier="common"))
            await session.flush()
            verb = Sense(word_id=target.id, pos="verb", tier="common")
            session.add(verb)
            await session.flush()
            session.add_all(
                [
                    Definition(sense_id=from_sense.id, content="to give off small flashes"),
                    Definition(sense_id=verb.id, content="to emit light"),
                    SenseRelation(
                        from_sense_id=from_sense.id,
                        to_word_id=target.id,
                        rel_type="synonym",
                        gloss="emit light",
                    ),
                ]
            )
        queued = await pending_relations(db, 20)
        assert [candidate.id for candidate in queued[0].candidates] == [verb.id]
        judge = Decision("candidate_1")
        outcome = await resolve_relations(db, judge)
        assert outcome[0].state == "resolved"
        state, questions = judge.calls[0]
        assert state == {
            "source": {"word": "glisten", "definition": "to give off small flashes"},
            "relation": {
                "type": "synonym",
                "rule": "The target sense must express essentially the same lexicalized concept "
                "as the source sense.",
            },
            "target": {"word": "shine", "gloss": "emit light"},
        }
        assert set(questions["matched_sense"].criteria) == {"no_candidate", "candidate_1"}
        assert (await get_word(db, source.id)).senses[0].relations[0].resolution_state == "resolved"
        async with db.transaction() as session:
            definition = await session.scalar(
                select(Definition).where(Definition.sense_id == verb.id)
            )
            definition.content = "different meaning"
        assert (await get_word(db, source.id)).senses[0].relations[0].resolution_state == "pending"
        assert len(await pending_relations(db, 20)) == 1
    finally:
        await db.close()


@pytest.mark.parametrize("missing", ["source", "target"])
async def test_incomplete_evidence_errors_only_its_edge(tmp_path, missing):
    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'incomplete.db'}")
    try:
        await db.create_schema(Base.metadata)
        async with db.transaction() as session:
            session.add_all(
                [
                    Word(
                        id=i,
                        lemma=f"word{i}",
                        match_key=f"word{i}",
                        entry_type="word",
                        generation_state="done",
                    )
                    for i in range(1, 4)
                ]
            )
            await session.flush()
            session.add_all(
                [
                    Sense(id=i, word_id=1 if i <= 2 else i - 1, pos="noun", tier="core")
                    for i in range(1, 5)
                ]
            )
            await session.flush()
            excluded = 1 if missing == "source" else 3
            session.add_all(
                [
                    Definition(sense_id=i, content=f"meaning{i}")
                    for i in range(1, 5)
                    if i != excluded
                ]
            )
            await session.flush()
            session.add_all(
                [
                    SenseRelation(
                        id=1, from_sense_id=1, to_word_id=2, rel_type="synonym", gloss="bad"
                    ),
                    SenseRelation(
                        id=2, from_sense_id=2, to_word_id=3, rel_type="synonym", gloss="good"
                    ),
                ]
            )
        model = Decision("candidate_1")
        results = await resolve_relations(db, model)
        assert [result.state for result in results] == ["error", "resolved"]
        assert "neutral" in results[0].error
        assert len(model.calls) == 1
        async with db.read() as connection:
            result = (
                await connection.execute(select(SenseRelation).where(SenseRelation.id == 1))
            ).first()
            assert result.to_sense_id is None and result.resolve_attempted_at is None
    finally:
        await db.close()


@pytest.fixture
async def ready_relation(tmp_path):
    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'relation.db'}")
    try:
        await db.create_schema(Base.metadata)
        async with db.transaction() as session:
            source = Word(lemma="run", match_key="run", entry_type="word", generation_state="done")
            target = Word(
                lemma="race", match_key="race", entry_type="word", generation_state="done"
            )
            session.add_all([source, target])
            await session.flush()
            source_sense = Sense(word_id=source.id, pos="verb", tier="core")
            target_sense = Sense(word_id=target.id, pos="verb", tier="core")
            session.add_all([source_sense, target_sense])
            await session.flush()
            edge = SenseRelation(
                from_sense_id=source_sense.id,
                to_word_id=target.id,
                rel_type="synonym",
                gloss="move quickly",
            )
            session.add_all(
                [
                    edge,
                    Definition(sense_id=source_sense.id, content="move quickly"),
                    Definition(sense_id=target_sense.id, content="move quickly"),
                ]
            )
            await session.flush()
        yield db, source_sense.id, target.id, target_sense.id, edge.id
    finally:
        await db.close()


@pytest.mark.parametrize(
    "change",
    [
        "target_state",
        "zero_target_state",
        "target_pos",
        "source_pos",
        "target_owner",
        "definitions",
    ],
)
async def test_resolution_revalidates_eligibility_before_writing(ready_relation, change):
    db, source_id, target_id, candidate_id, edge_id = ready_relation

    class ChangingDecision:
        async def decide(self, state, questions, **kwargs):
            async with db.transaction() as session:
                if change in {"target_state", "zero_target_state"}:
                    (await session.get(Word, target_id)).generation_state = "pending"
                elif change == "target_pos":
                    (await session.get(Sense, candidate_id)).pos = "noun"
                elif change == "source_pos":
                    (await session.get(Sense, source_id)).pos = "noun"
                elif change == "target_owner":
                    source = await session.get(Sense, source_id)
                    candidate = await session.get(Sense, candidate_id)
                    candidate.word_id = source.word_id
                else:
                    definition = await session.scalar(
                        select(Definition).where(Definition.sense_id == candidate_id)
                    )
                    definition.content = "a changed meaning"
            return SimpleNamespace(
                choices={
                    "matched_sense": SimpleNamespace(
                        choice="no_candidate" if change == "zero_target_state" else "candidate_1"
                    )
                }
            )

    results = await resolve_relations(db, ChangingDecision())
    assert [result.state for result in results] == ["noop"]
    async with db.transaction() as session:
        edge = await session.get(SenseRelation, edge_id)
        assert edge.to_sense_id is None
        assert edge.resolve_attempted_at is None
        assert edge.target_hash is None


async def test_done_target_without_senses_is_not_ready(ready_relation):
    db, _, _, candidate_id, edge_id = ready_relation
    async with db.transaction() as session:
        await session.delete(await session.get(Sense, candidate_id))
    assert await pending_relations(db, 20) == []
    assert await resolve_relations(db, None) == []
    async with db.transaction() as session:
        edge = await session.get(SenseRelation, edge_id)
        assert edge.resolve_attempted_at is None


async def test_no_same_pos_decides_zero_without_decision_model(tmp_path):
    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'generated.db'}")
    try:
        await db.create_schema(Base.metadata)
        async with db.transaction() as session:
            source = Word(lemma="run", match_key="run", entry_type="word", generation_state="done")
            target = Word(
                lemma="marathon", match_key="marathon", entry_type="word", generation_state="done"
            )
            session.add_all([source, target])
            await session.flush()
            verb = Sense(word_id=source.id, pos="verb", tier="common")
            noun = Sense(word_id=target.id, pos="noun", tier="common")
            session.add_all([verb, noun])
            await session.flush()
            session.add_all(
                [
                    Definition(sense_id=verb.id, content="move quickly"),
                    Definition(sense_id=noun.id, content="long running race"),
                    SenseRelation(
                        from_sense_id=verb.id,
                        to_word_id=target.id,
                        rel_type="synonym",
                        gloss="a race",
                    ),
                ]
            )
        judge = Decision("999")
        assert (await resolve_relations(db, judge))[0].state == "unresolvable"
        assert judge.calls == []
        assert (await get_word(db, source.id)).senses[0].relations[
            0
        ].resolution_state == "unresolvable"
    finally:
        await db.close()


async def test_resolved_edges_before_batch_do_not_starve_pending_edge(tmp_path):
    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'generated.db'}")
    try:
        await db.create_schema(Base.metadata)
        async with db.transaction() as session:
            source = Word(lemma="run", match_key="run", entry_type="word", generation_state="done")
            target = Word(
                lemma="race", match_key="race", entry_type="word", generation_state="done"
            )
            session.add_all([source, target])
            await session.flush()
            target_sense = Sense(word_id=target.id, pos="verb", tier="core")
            source_senses = [Sense(word_id=source.id, pos="verb", tier="core") for _ in range(51)]
            session.add_all([target_sense, *source_senses])
            await session.flush()
            session.add(Definition(sense_id=target_sense.id, content="move quickly"))
            for sense in source_senses:
                session.add(Definition(sense_id=sense.id, content="move quickly"))
            for sense in source_senses[:-1]:
                session.add(
                    SenseRelation(
                        from_sense_id=sense.id,
                        to_word_id=target.id,
                        to_sense_id=target_sense.id,
                        rel_type="synonym",
                        gloss="move quickly",
                        target_hash=definition_hash("move quickly"),
                        resolve_attempted_at="2026-09-29T00:00:00Z",
                    )
                )
            pending = SenseRelation(
                from_sense_id=source_senses[-1].id,
                to_word_id=target.id,
                rel_type="synonym",
                gloss="move quickly",
            )
            session.add(pending)
            await session.flush()
        queued = await pending_relations(db, 1)
        assert [link.edge_id for link in queued] == [pending.id]
        assert (await resolve_relations(db, Decision("candidate_1"), batch_size=1))[
            0
        ].state == "resolved"
    finally:
        await db.close()


@pytest.mark.parametrize("failure", ["negative", "transport"])
async def test_parallel_resolution_isolates_invalid_verdict_and_transport_error(tmp_path, failure):
    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'generated.db'}")
    started = []
    both_started = asyncio.Event()

    class IndependentDecisions:
        async def decide(self, state, questions, **kwargs):
            started.append(state["target"]["gloss"])
            if len(started) == 2:
                both_started.set()
            await asyncio.wait_for(both_started.wait(), timeout=2)
            if state["target"]["gloss"] == "bad":
                if failure == "transport":
                    raise ConnectionError("provider unavailable")
                return SimpleNamespace(choices={"matched_sense": SimpleNamespace(choice="invalid")})
            return SimpleNamespace(choices={"matched_sense": SimpleNamespace(choice="candidate_1")})

    try:
        await db.create_schema(Base.metadata)
        async with db.transaction() as session:
            source = Word(lemma="run", match_key="run", entry_type="word", generation_state="done")
            target = Word(
                lemma="race", match_key="race", entry_type="word", generation_state="done"
            )
            session.add_all([source, target])
            await session.flush()
            senses = [Sense(word_id=source.id, pos="verb", tier="core") for _ in range(2)]
            target_sense = Sense(word_id=target.id, pos="verb", tier="core")
            session.add_all([*senses, target_sense])
            await session.flush()
            for sense in [*senses, target_sense]:
                session.add(Definition(sense_id=sense.id, content="move quickly"))
            edges = [
                SenseRelation(
                    from_sense_id=sense.id, to_word_id=target.id, rel_type="synonym", gloss=gloss
                )
                for sense, gloss in zip(senses, ["bad", "good"], strict=True)
            ]
            session.add_all(edges)
            await session.flush()
        results = await resolve_relations(db, IndependentDecisions())
        assert [item.state for item in results] == ["error", "resolved"]
        async with db.transaction() as session:
            bad, good = [await session.get(SenseRelation, edge.id) for edge in edges]
            assert bad.to_sense_id is None and bad.resolve_attempted_at is None
            assert good.to_sense_id == target_sense.id
    finally:
        await db.close()
