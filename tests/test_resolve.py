"""Tests for WSD sense-relation resolution (Phase 4).

Locks the reconciliation spine with a fake WSD judge (no network): the queue
query (only ``done`` targets with senses, derived-``pending`` edges), the POS
filter (both sides normalized, no hard-exclude of unknown POS), the conditional
UPDATE ([F6] race no-op), per-edge savepoint isolation ([F7]), index-bounds
validation ([F3]), ``batch_size`` clamp ([F9]), and the inbound-resolve hook
firing at the end of ``persist_result`` ([F11]).

State is DERIVED (Q1): assertions read ``to_sense_id`` / ``resolve_attempted_at``,
never a ``wsd_state`` column. In-memory SQLite + StaticPool, mirroring
``test_enrichment.py``.
"""

import pytest
from sqlalchemy import event, select
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import StaticPool

from lexi_ai.api import Lexicon
from lexi_ai.db import create_session_factory, init_models
from lexi_ai.generation.schemas import (
    GeneratedEntry,
    GeneratedResult,
    GeneratedSense,
    GeneratedSenseRelation,
    WsdBatch,
    WsdChoice,
)
from lexi_ai.infrastructure.db.models import Sense, SenseRelation
from lexi_ai.infrastructure.db.repositories.sense_repo import SqlSenseRepo
from tests.support.persistence_driver import PersistenceDriver

# --- harness ---------------------------------------------------------------


@pytest.fixture
async def engine():
    engine = create_async_engine(
        "sqlite+aiosqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    @event.listens_for(engine.sync_engine, "connect")
    def _fk_on(dbapi_conn, _record):
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA foreign_keys=ON")
        cur.close()

    await init_models(engine)
    yield engine
    await engine.dispose()


class _FakeJudge:
    """Fake WSD judge: returns a canned ``WsdBatch`` (or a per-call callback),
    order-aligned by the harness. Records the tasks it was handed so tests can
    assert the POS filter narrowed candidates before the judge ever saw them."""

    def __init__(self, choices=None, fn=None):
        self._choices = choices
        self._fn = fn
        self.seen_tasks = None
        self.calls = 0

    async def judge(self, tasks):
        self.calls += 1
        self.seen_tasks = list(tasks)
        if self._fn is not None:
            return self._fn(self.seen_tasks)
        batch = self._choices or WsdBatch(choices=[])
        # Mirror WsdJudge's own order-alignment (pad/truncate to task count).
        out = list(batch.choices)[: len(tasks)]
        out += [WsdChoice(chosen_index=None) for _ in range(len(tasks) - len(out))]
        return out


def _sense(defn, pos, **kw):
    return GeneratedSense(definition=defn, tier="core", pos=pos, **kw)


def _entry(norm, senses, *, entry_type="word") -> GeneratedEntry:
    return GeneratedEntry(norm=norm, entry_type=entry_type, senses=senses)


class _FakeLoader:
    """Minimal reference loader for the inbound-hook test: a custom bundle only
    (no Cambridge) so ``_run_generation`` can build+persist the target word."""

    async def bundle_custom(self, word: str):
        from lexi_ai.references.loader import ReferenceBundle

        return ReferenceBundle(word_raw=word, entry_type="word", cambridge_word_id=None)


class _FakeGenerator:
    """Returns a canned :class:`GeneratedResult` regardless of the bundle — the
    target word's senses (what the inbound edge resolves onto)."""

    def __init__(self, result: GeneratedResult):
        self._result = result

    async def generate(self, bundle, existing_tags=()) -> GeneratedResult:
        return self._result


async def _lexicon(engine, repo, judge=None) -> Lexicon:
    return Lexicon(
        create_session_factory(engine),
        None,  # type: ignore[arg-type]
        None,  # type: ignore[arg-type]
        engine=engine,
        wsd_judge=judge,  # type: ignore[arg-type]
    )


async def _seed_source_with_relation(
    repo, *, source_norm, source_pos, rel_type, target_norm, gloss
):
    """Persist a source word whose single sense emits ONE sense-level relation ->
    a pending half-edge to ``target_norm`` (a stub until its own word is done)."""
    await repo.persist_result(
        GeneratedResult(
            units=[
                _entry(
                    source_norm,
                    [
                        _sense(
                            f"def of {source_norm}",
                            source_pos,
                            relations=[
                                GeneratedSenseRelation(
                                    rel_type=rel_type, norm=target_norm, gloss=gloss
                                )
                            ],
                        )
                    ],
                )
            ]
        )
    )


async def _edge(sf) -> SenseRelation:
    async with sf() as s:
        return (await s.execute(select(SenseRelation))).scalars().one()


# --- resolve: POS-matched pick + derived resolved --------------------------


async def test_resolve_picks_pos_matched_sense(engine):
    sf = create_session_factory(engine)
    repo = PersistenceDriver(sf)
    # Source sense is an adjective; target 'dark' has an adjective + a noun sense.
    await _seed_source_with_relation(
        repo,
        source_norm="bright",
        source_pos="adjective",
        rel_type="antonym",
        target_norm="dark",
        gloss="lacking light",
    )
    # Target word becomes done with two POS-distinct senses.
    await repo.persist_result(
        GeneratedResult(
            units=[
                _entry(
                    "dark",
                    [
                        _sense("lacking light", "adjective"),
                        _sense("an evil force", "noun"),
                    ],
                )
            ]
        )
    )

    # Find the adjective target sense id (candidate index 0 after POS filter).
    async with sf() as s:
        adj_sense = (
            await s.execute(
                select(Sense.id).where(
                    Sense.definition == "lacking light", Sense.pos == "adjective"
                )
            )
        ).scalar_one()

    judge = _FakeJudge(WsdBatch(choices=[WsdChoice(chosen_index=0)]))
    lex = await _lexicon(engine, repo, judge)
    results = await lex.engine().resolve_relations()

    assert len(results) == 1
    assert results[0].value == "resolved"
    edge = await _edge(sf)
    assert edge.to_sense_id == adj_sense  # resolved to the adjective sense
    assert edge.resolve_attempted_at is None  # derived resolved, not unresolvable
    assert edge.target_hash is not None  # content hash stamped


async def test_resolve_skips_pending_target(engine):
    # Target word never generated (stub stays pending) -> edge is not eligible.
    sf = create_session_factory(engine)
    repo = PersistenceDriver(sf)
    await _seed_source_with_relation(
        repo,
        source_norm="bright",
        source_pos="adjective",
        rel_type="antonym",
        target_norm="dark",
        gloss="lacking light",
    )
    judge = _FakeJudge(WsdBatch(choices=[WsdChoice(chosen_index=0)]))
    lex = await _lexicon(engine, repo, judge)
    results = await lex.engine().resolve_relations()

    assert results == []  # nothing eligible
    assert judge.calls == 0  # judge never invoked (queue empty)
    edge = await _edge(sf)
    assert edge.to_sense_id is None and edge.resolve_attempted_at is None  # still pending


async def test_resolve_unresolvable_on_no_match(engine):
    sf = create_session_factory(engine)
    repo = PersistenceDriver(sf)
    await _seed_source_with_relation(
        repo,
        source_norm="bright",
        source_pos="adjective",
        rel_type="antonym",
        target_norm="dark",
        gloss="a meaning no sense carries",
    )
    await repo.persist_result(
        GeneratedResult(units=[_entry("dark", [_sense("lacking light", "adjective")])])
    )

    judge = _FakeJudge(WsdBatch(choices=[WsdChoice(chosen_index=None)]))  # "none"
    lex = await _lexicon(engine, repo, judge)
    results = await lex.engine().resolve_relations()
    assert results[0].value == "unresolvable"

    edge = await _edge(sf)
    assert edge.to_sense_id is None
    assert edge.resolve_attempted_at is not None  # derived unresolvable

    # A second pass must NOT re-touch it (unresolvable is a terminal stop).
    judge2 = _FakeJudge(WsdBatch(choices=[WsdChoice(chosen_index=0)]))
    lex2 = await _lexicon(engine, repo, judge2)
    assert await lex2.engine().resolve_relations() == []
    assert judge2.calls == 0


async def test_mark_unresolvable_stamps_naive_utc(engine):
    # SQLite-distinguishable teeth for the dual-DB fix: _mark_unresolvable must
    # stamp a NAIVE datetime, matching the tz-naive DateTime columns. An AWARE
    # value binds fine on SQLite (stored as a string) but raises asyncpg DataError
    # against TIMESTAMP WITHOUT TIME ZONE — silently converting every unresolvable
    # edge to state="error" and re-queueing it forever (the Postgres tier in
    # test_postgres_integration observes the downstream no-re-queue).
    #
    # A read-back has NO teeth here: SQLite's DateTime type strips tzinfo on the
    # way out, so ``edge.resolve_attempted_at.tzinfo`` is None whether the write
    # was aware or naive. Instead capture the value BOUND to the UPDATE at cursor
    # time — an aware datetime serializes with a "+00:00" offset, a naive one does
    # not. That difference is the SQLite-observable signal.
    bound_stamps: list[str] = []

    @event.listens_for(engine.sync_engine, "before_cursor_execute")
    def _capture(conn, cursor, statement, parameters, context, executemany):  # noqa: ANN001
        if "resolve_attempted_at" in statement and "UPDATE" in statement.upper():
            bound_stamps.extend(str(p) for p in parameters if "-" in str(p) and ":" in str(p))

    sf = create_session_factory(engine)
    repo = PersistenceDriver(sf)
    await _seed_source_with_relation(
        repo,
        source_norm="bright",
        source_pos="adjective",
        rel_type="antonym",
        target_norm="dark",
        gloss="a meaning no sense carries",
    )
    await repo.persist_result(
        GeneratedResult(units=[_entry("dark", [_sense("lacking light", "adjective")])])
    )
    judge = _FakeJudge(WsdBatch(choices=[WsdChoice(chosen_index=None)]))  # "none"
    lex = await _lexicon(engine, repo, judge)
    assert (await lex.engine().resolve_relations())[0].value == "unresolvable"

    edge = await _edge(sf)
    assert edge.resolve_attempted_at is not None
    # The stamp bound to the UPDATE must carry no timezone offset (naive UTC).
    assert bound_stamps, "expected a resolve_attempted_at stamp bound to the UPDATE"
    assert all("+00:00" not in s and "+0000" not in s for s in bound_stamps), bound_stamps


async def test_resolve_regenerate_race_noop(engine):
    # [F6] TOCTOU: the apply UPDATE is CONDITIONAL. Once an edge is already
    # resolved (a racing pass / regenerate moved it out of pending), re-applying a
    # decision to it is a no-op — it never overwrites or writes a dead id.
    from lexi_ai.domain.models import ResolveDecision

    sf = create_session_factory(engine)
    repo = PersistenceDriver(sf)
    await _seed_source_with_relation(
        repo,
        source_norm="bright",
        source_pos="adjective",
        rel_type="antonym",
        target_norm="dark",
        gloss="lacking light",
    )
    await repo.persist_result(
        GeneratedResult(units=[_entry("dark", [_sense("lacking light", "adjective")])])
    )
    # Resolve it once (edge leaves pending).
    judge = _FakeJudge(WsdBatch(choices=[WsdChoice(chosen_index=0)]))
    lex = await _lexicon(engine, repo, judge)
    await lex.engine().resolve_relations()
    edge = await _edge(sf)
    first_target = edge.to_sense_id
    assert first_target is not None

    # A stale decision (from a batch read BEFORE the resolve) tries to write again.
    # The condition `to_sense_id IS NULL` no longer holds -> no-op, not overwrite.
    async with sf() as s:
        other = (
            (await s.execute(select(Sense.id).where(Sense.id != first_target))).scalars().first()
        )
    outcomes = await repo.apply_resolutions(
        [ResolveDecision(edge.id, other or first_target, "deadhash")]
    )
    assert outcomes[0].state == "noop"
    edge2 = await _edge(sf)
    assert edge2.to_sense_id == first_target  # unchanged by the stale write
    assert edge2.target_hash != "deadhash"


async def test_batch_poison_pill_isolated(engine, monkeypatch):
    # [F7] One edge failing (its apply raises) must not abort the whole batch:
    # each edge is wrapped in its own savepoint, so siblings still commit and the
    # failed one is reported as an error.
    from lexi_ai.domain.models import ResolveDecision

    sf = create_session_factory(engine)
    repo = PersistenceDriver(sf)
    # Two independent source words, each with an inbound edge onto its own target.
    for src, tgt in (("bright", "dark"), ("big", "small")):
        await _seed_source_with_relation(
            repo,
            source_norm=src,
            source_pos="adjective",
            rel_type="antonym",
            target_norm=tgt,
            gloss=f"opposite of {src}",
        )
        await repo.persist_result(
            GeneratedResult(units=[_entry(tgt, [_sense(f"def of {tgt}", "adjective")])])
        )

    async with sf() as s:
        edges = (await s.execute(select(SenseRelation).order_by(SenseRelation.id))).scalars().all()
        target = (await s.execute(select(Sense.id).order_by(Sense.id))).scalars().first()
    assert len(edges) == 2 and target is not None

    # Poison the FIRST edge's apply; the second must still resolve.
    poison_id = edges[0].id
    real_apply = SqlSenseRepo._apply_resolved

    async def _maybe_boom(self, edge_id, to_sense_id, target_hash):
        if edge_id == poison_id:
            raise RuntimeError("boom")
        return await real_apply(self, edge_id, to_sense_id, target_hash)

    monkeypatch.setattr(SqlSenseRepo, "_apply_resolved", _maybe_boom)
    outcomes = await repo.apply_resolutions([ResolveDecision(e.id, target, "h") for e in edges])
    by_edge = {o.edge_id: o for o in outcomes}
    assert by_edge[poison_id].state == "error"  # isolated
    other_id = edges[1].id
    assert by_edge[other_id].state == "resolved"  # sibling committed


async def test_resolve_pos_no_match_goes_to_judge(engine):
    # [F2] Source is a verb; target has only noun senses (no clear same-POS
    # candidate). We must NOT auto-unresolvable: ALL candidates go to the judge,
    # and only the judge's "none" makes it unresolvable.
    sf = create_session_factory(engine)
    repo = PersistenceDriver(sf)
    await _seed_source_with_relation(
        repo,
        source_norm="run",
        source_pos="verb",
        rel_type="see_also",
        target_norm="marathon",
        gloss="a long race",
    )
    await repo.persist_result(
        GeneratedResult(
            units=[
                _entry(
                    "marathon",
                    [_sense("a long race", "noun"), _sense("an endurance event", "noun")],
                )
            ]
        )
    )
    judge = _FakeJudge(WsdBatch(choices=[WsdChoice(chosen_index=0)]))
    lex = await _lexicon(engine, repo, judge)
    results = await lex.engine().resolve_relations()

    assert judge.calls == 1
    # Both noun candidates were shown despite the verb source (no POS hard-drop).
    assert judge.seen_tasks is not None
    assert len(judge.seen_tasks[0].candidates) == 2
    assert results[0].value == "resolved"


async def test_wsd_index_out_of_range(engine):
    # [F3] Judge returns an index past the candidate list -> treated as "none"
    # (unresolvable), never an IndexError that aborts the batch.
    sf = create_session_factory(engine)
    repo = PersistenceDriver(sf)
    await _seed_source_with_relation(
        repo,
        source_norm="bright",
        source_pos="adjective",
        rel_type="antonym",
        target_norm="dark",
        gloss="lacking light",
    )
    await repo.persist_result(
        GeneratedResult(
            units=[
                _entry(
                    "dark",
                    [_sense("lacking light", "adjective"), _sense("gloomy", "adjective")],
                )
            ]
        )
    )
    judge = _FakeJudge(WsdBatch(choices=[WsdChoice(chosen_index=99)]))  # out of range
    lex = await _lexicon(engine, repo, judge)
    results = await lex.engine().resolve_relations()

    assert results[0].value == "unresolvable"
    edge = await _edge(sf)
    assert edge.to_sense_id is None
    assert edge.resolve_attempted_at is not None


async def test_batch_size_clamped(engine, monkeypatch):
    # [F9] A huge batch_size is clamped to the hard ceiling before the DB read.
    from lexi_ai.generation.wsd import WSD_BATCH_CEIL

    sf = create_session_factory(engine)
    repo = PersistenceDriver(sf)
    captured = {}
    orig = SqlSenseRepo.pending_relations

    async def _spy(self, batch_size, word_ids=None):
        captured["batch_size"] = batch_size
        return await orig(self, batch_size, word_ids=word_ids)

    monkeypatch.setattr(SqlSenseRepo, "pending_relations", _spy)
    judge = _FakeJudge(WsdBatch(choices=[]))
    lex = await _lexicon(engine, repo, judge)
    await lex.engine().resolve_relations(batch_size=1000)
    assert captured["batch_size"] == WSD_BATCH_CEIL


async def test_generate_target_triggers_inbound_resolve(engine):
    # [F11] Generating the TARGET word must resolve its inbound pending edges as a
    # side effect of persist_result — no manual resolve_relations() call.
    sf = create_session_factory(engine)
    repo = PersistenceDriver(sf)
    await _seed_source_with_relation(
        repo,
        source_norm="bright",
        source_pos="adjective",
        rel_type="antonym",
        target_norm="dark",
        gloss="lacking light",
    )
    # Edge is pending before the target exists.
    edge = await _edge(sf)
    assert edge.to_sense_id is None and edge.resolve_attempted_at is None

    judge = _FakeJudge(WsdBatch(choices=[WsdChoice(chosen_index=0)]))
    target_result = GeneratedResult(units=[_entry("dark", [_sense("lacking light", "adjective")])])
    lex = Lexicon(
        sf,
        _FakeLoader(),  # type: ignore[arg-type]
        _FakeGenerator(target_result),  # type: ignore[arg-type]
        engine=engine,
        wsd_judge=judge,  # type: ignore[arg-type]
    )
    # Generating 'dark' through the API path fires the inbound hook.
    await lex.generation()._run("dark", None, fence=None, method=None)  # type: ignore[attr-defined]

    edge2 = await _edge(sf)
    assert edge2.to_sense_id is not None  # auto-resolved by the hook
    assert judge.calls == 1
