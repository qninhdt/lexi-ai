"""Tests for sense-relation invalidation (Phase 5): Case 2 + Case 6 + F4/F5/F13.

Keeps the sense->sense graph from rotting as words change. Locks:

- [Case 2] target regenerate demotes inbound resolved edges back to derived
  ``pending`` (``to_sense_id`` NULL AND ``resolve_attempted_at`` NULL), keeping
  ``to_word_id`` — the edge survives at sense->word.
- [Case 6] source regenerate CASCADE-drops its emitted edges; Phase 3 re-emits
  fresh half-edges for the new senses.
- [F4] deleting the TARGET word CASCADE-removes its inbound edges outright
  (``to_word_id`` is ``ON DELETE CASCADE``, the Phase-2-locked schema): there is
  no orphaned sense->word row to strand, so no stuck derived-``unresolvable``
  edge can survive a delete. The demote helper's live job is the *regenerate*
  path (Case 2 / F13), where the word row stays but its senses churn.
- [F13] regenerating an improved target re-queues its inbound derived-
  ``unresolvable`` edges for a fresh WSD attempt.

State is DERIVED (Q1): assertions read ``to_sense_id`` / ``resolve_attempted_at``,
never a ``wsd_state`` column. In-memory SQLite + StaticPool with FK pragma ON
(SET NULL / CASCADE need it), mirroring ``test_resolve.py``.
"""

import pytest
from sqlalchemy import event, select
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import StaticPool

from lexi_ai.db import create_session_factory, init_models
from lexi_ai.generation.schemas import (
    GeneratedEntry,
    GeneratedResult,
    GeneratedSense,
    GeneratedSenseRelation,
)
from lexi_ai.infrastructure.db.models import Sense, SenseRelation, Word
from tests.support.persistence_driver import PersistenceDriver


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


def _sense(defn, pos, **kw):
    return GeneratedSense(definition=defn, tier="core", pos=pos, **kw)


def _entry(norm, senses, *, entry_type="word") -> GeneratedEntry:
    return GeneratedEntry(norm=norm, entry_type=entry_type, senses=senses)


async def _seed_a_to_b(repo, *, rel_type="antonym", gloss="lacking light"):
    """Persist source 'bright' (adjective, emits a relation to 'dark') and target
    'dark' (done, one adjective sense). Returns nothing — the edge is left pending."""
    await repo.persist_result(
        GeneratedResult(
            units=[
                _entry(
                    "bright",
                    [
                        _sense(
                            "full of light",
                            "adjective",
                            relations=[
                                GeneratedSenseRelation(rel_type=rel_type, norm="dark", gloss=gloss)
                            ],
                        )
                    ],
                )
            ]
        )
    )
    await repo.persist_result(
        GeneratedResult(units=[_entry("dark", [_sense("lacking light", "adjective")])])
    )


async def _edge(sf) -> SenseRelation:
    async with sf() as s:
        return (await s.execute(select(SenseRelation))).scalars().one()


async def _resolve_edge_manually(sf, *, target_definition="lacking light"):
    """Force the single edge to a resolved state (to_sense_id + attempted + hash)
    WITHOUT going through the judge, so an invalidation test starts from resolved."""
    from datetime import datetime, timezone

    from lexi_ai.domain.hashing import sense_content_hash

    async with sf() as s:
        tgt = (
            await s.execute(select(Sense.id).where(Sense.definition == target_definition))
        ).scalar_one()
        edge = (await s.execute(select(SenseRelation))).scalars().one()
        edge.to_sense_id = tgt
        edge.resolve_attempted_at = datetime.now(timezone.utc)
        edge.target_hash = sense_content_hash(target_definition)
        await s.commit()
    return tgt


# --- Case 2: target regenerate demotes inbound resolved edges --------------


async def test_target_regenerate_demotes_edge(engine):
    sf = create_session_factory(engine)
    repo = PersistenceDriver(sf)
    await _seed_a_to_b(repo)
    await _resolve_edge_manually(sf)

    async with sf() as s:
        b_word = (await s.execute(select(Word).where(Word.match_key == "dark"))).scalar_one()
        b_id = b_word.id

    # Regenerate 'dark' (new sense content) — the OLD target sense is deleted.
    await repo.persist_result(
        GeneratedResult(units=[_entry("dark", [_sense("without light or hope", "adjective")])])
    )

    edge = await _edge(sf)
    assert edge.to_sense_id is None  # target sense gone -> SET NULL
    assert edge.resolve_attempted_at is None  # helper reset -> derived pending, NOT unresolvable
    assert edge.target_hash is None
    assert edge.to_word_id == b_id  # sense->word survives


async def test_source_regenerate_cascades_edge(engine):
    # [Case 6] Regenerating the SOURCE deletes its old sense -> from_sense_id
    # CASCADE drops the old edge; Phase 3 re-emits a fresh half-edge for the new
    # source sense (derived pending).
    sf = create_session_factory(engine)
    repo = PersistenceDriver(sf)
    await _seed_a_to_b(repo)
    await _resolve_edge_manually(sf)
    old_from_sense_id = (await _edge(sf)).from_sense_id

    # Regenerate 'bright' — still emits the same relation to 'dark'.
    await repo.persist_result(
        GeneratedResult(
            units=[
                _entry(
                    "bright",
                    [
                        _sense(
                            "shining strongly",
                            "adjective",
                            relations=[
                                GeneratedSenseRelation(
                                    rel_type="antonym", norm="dark", gloss="lacking light"
                                )
                            ],
                        )
                    ],
                )
            ]
        )
    )

    edge = await _edge(sf)
    # Old source sense CASCADE-deleted its edge; the new source sense emits a fresh
    # half-edge (different from_sense_id), back at derived pending.
    assert edge.from_sense_id != old_from_sense_id
    assert edge.to_sense_id is None and edge.resolve_attempted_at is None  # fresh pending


# --- F4: delete_word / delete_entry drop inbound edges (to_word_id CASCADE) --


async def test_delete_word_cascades_inbound_edges(engine):
    # [F4/Phase 2] ``sense_relation.to_word_id`` is ON DELETE CASCADE: deleting the
    # TARGET word removes every edge pointing at it outright, so there is no
    # orphaned sense->word row left stranded as derived-unresolvable. (This is the
    # cleaner alternative to the "demote-on-delete" the plan first sketched before
    # to_word_id CASCADE was locked.)
    sf = create_session_factory(engine)
    repo = PersistenceDriver(sf)
    await _seed_a_to_b(repo)
    await _resolve_edge_manually(sf)

    async with sf() as s:
        b_id = (await s.execute(select(Word).where(Word.match_key == "dark"))).scalar_one().id

    assert await repo.delete_word(b_id) is True

    async with sf() as s:
        rows = (await s.execute(select(SenseRelation))).scalars().all()
    assert rows == []  # inbound edge cascade-removed with the target word


async def test_delete_entry_cascades_inbound_edges(engine):
    # delete_entry delegates to delete_word; assert the same cascade via the API path.
    from lexi_ai.api import Lexicon

    sf = create_session_factory(engine)
    repo = PersistenceDriver(sf)
    await _seed_a_to_b(repo)
    await _resolve_edge_manually(sf)

    async with sf() as s:
        b_id = (await s.execute(select(Word).where(Word.match_key == "dark"))).scalar_one().id

    lex = Lexicon(sf, None, None, engine=engine)  # type: ignore[arg-type]
    assert await lex.delete_entry(b_id) is True

    async with sf() as s:
        rows = (await s.execute(select(SenseRelation))).scalars().all()
    assert rows == []  # inbound edge cascade-removed with the target word


# --- F13: regenerate improved target re-queues unresolvable inbound --------


async def test_regenerate_target_requeues_unresolvable(engine):
    # [F13] An inbound edge stuck at derived-unresolvable (to_sense_id NULL AND
    # resolve_attempted_at NOT NULL) is re-queued to derived-pending when the
    # target word regenerates, so WSD retries against the improved content.
    from datetime import datetime, timezone

    sf = create_session_factory(engine)
    repo = PersistenceDriver(sf)
    await _seed_a_to_b(repo)

    # Mark the edge derived-unresolvable (judge previously said "none").
    async with sf() as s:
        edge = (await s.execute(select(SenseRelation))).scalars().one()
        edge.resolve_attempted_at = datetime.now(timezone.utc)
        await s.commit()

    # Regenerate 'dark' (improved) — the inbound unresolvable edge must reset.
    await repo.persist_result(
        GeneratedResult(units=[_entry("dark", [_sense("without light", "adjective")])])
    )

    edge = await _edge(sf)
    assert edge.to_sense_id is None
    assert edge.resolve_attempted_at is None  # re-queued -> derived pending
