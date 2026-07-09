"""Tests for the sense-level relation read model (Phase 6).

Surfaces sense-level relations on ``SenseView.relations`` (a new, additive field)
while ``Entry.links`` keeps its word-level shape unchanged (no consumer break).
Locks:

- a pending relation surfaces with ``to_word_id`` + ``wsd_state='pending'`` and
  ``to_sense_id=None`` (derived state, Q1 — never a DB column);
- a resolved relation additionally carries ``to_sense_id`` + ``to_sense_gloss``;
- [F5/Q2] hash-verify on read is MANDATORY: a resolved edge whose target sense
  definition changed out from under it (stale ``target_hash``) surfaces as if
  UNRESOLVED (``to_sense_id=None``, ``wsd_state='pending'``);
- ``Entry.links`` word-level relations still round-trip (no Pycil regression);
- the read is hermetic (relations eager-loaded — no MissingGreenlet).

In-memory SQLite + StaticPool with FK pragma ON, mirroring ``test_resolve.py``.
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
    RelatedWord,
)
from lexi_ai.models import Sense, SenseRelation
from lexi_ai.persistence.repository import Repository, sense_content_hash


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


def _entry(norm, senses, *, entry_type="word", related=None) -> GeneratedEntry:
    return GeneratedEntry(
        norm=norm,
        entry_type=entry_type,
        senses=senses,
        related=[RelatedWord(norm=n, rel_type=rt) for n, rt in (related or [])],
    )


async def _reading_lexicon(engine, repo) -> Lexicon:
    return Lexicon(create_session_factory(engine), None, None, repo, engine=engine)  # type: ignore[arg-type]


async def _relations_for(lex, word_id):
    """The relations on the FIRST sense of the entry (the seeded emitter)."""
    entry = await lex.get_entry(word_id)
    return entry.senses[0].relations


# --- pending relation surfaces at sense level ------------------------------


async def test_sense_view_carries_pending_relation(engine):
    sf = create_session_factory(engine)
    repo = Repository(sf)
    words = await repo.persist_result(
        GeneratedResult(
            units=[
                _entry(
                    "bright",
                    [
                        _sense(
                            "full of light",
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
    lex = await _reading_lexicon(engine, repo)
    rels = await _relations_for(lex, words[0].id)

    assert len(rels) == 1
    r = rels[0]
    assert r.rel_type == "antonym"
    assert r.to_word_display == "dark"
    assert r.to_word_id is not None
    assert r.to_word_status == "pending"  # target 'dark' is a stub, not yet generated
    assert r.to_sense_id is None
    assert r.to_sense_gloss is None
    assert r.wsd_state == "pending"  # derived, not a DB column


async def test_sense_view_carries_resolved_relation(engine):
    sf = create_session_factory(engine)
    repo = Repository(sf)
    words = await repo.persist_result(
        GeneratedResult(
            units=[
                _entry(
                    "bright",
                    [
                        _sense(
                            "full of light",
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
    # Generate the target, then resolve the edge onto its sense.
    await repo.persist_result(
        GeneratedResult(units=[_entry("dark", [_sense("without light", "adjective")])])
    )
    async with sf() as s:
        tgt = (
            await s.execute(select(Sense.id).where(Sense.definition == "without light"))
        ).scalar_one()
        edge = (await s.execute(select(SenseRelation))).scalars().one()
        edge.to_sense_id = tgt
        edge.target_hash = sense_content_hash("without light")
        await s.commit()

    lex = await _reading_lexicon(engine, repo)
    rels = await _relations_for(lex, words[0].id)

    assert len(rels) == 1
    r = rels[0]
    assert r.to_sense_id == tgt
    assert r.to_sense_gloss == "without light"
    assert r.wsd_state == "resolved"
    assert r.to_word_status == "done"  # target now generated


async def test_stale_hash_treated_unresolved(engine):
    # [F5/Q2] A resolved edge whose target sense definition changed out from under
    # it (stale target_hash) MUST surface as unresolved on read — the final safety
    # net for any target-mutation path Phase 5 invalidation might miss.
    sf = create_session_factory(engine)
    repo = Repository(sf)
    words = await repo.persist_result(
        GeneratedResult(
            units=[
                _entry(
                    "bright",
                    [
                        _sense(
                            "full of light",
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
    await repo.persist_result(
        GeneratedResult(units=[_entry("dark", [_sense("without light", "adjective")])])
    )
    # Resolve with a hash, then MUTATE the target definition in place (no demote path).
    async with sf() as s:
        tgt_sense = (
            await s.execute(select(Sense).where(Sense.definition == "without light"))
        ).scalar_one()
        edge = (await s.execute(select(SenseRelation))).scalars().one()
        edge.to_sense_id = tgt_sense.id
        edge.target_hash = sense_content_hash("without light")
        await s.commit()
    async with sf() as s:
        tgt_sense = (await s.execute(select(Sense).where(Sense.id == tgt_sense.id))).scalar_one()
        tgt_sense.definition = "an evil supernatural force"  # in-place edit, hash now stale
        await s.commit()

    lex = await _reading_lexicon(engine, repo)
    rels = await _relations_for(lex, words[0].id)

    r = rels[0]
    assert r.to_sense_id is None  # stale hash -> demoted on read
    assert r.wsd_state == "pending"
    assert r.to_word_id is not None  # sense->word still trusted


async def test_entry_links_still_word_level(engine):
    # A word-level relation (word_family) must still surface via Entry.links with
    # its original shape — no regression for consumers reading the flat list.
    sf = create_session_factory(engine)
    repo = Repository(sf)
    words = await repo.persist_result(
        GeneratedResult(
            units=[
                _entry(
                    "happy",
                    [_sense("feeling joy", "adjective")],
                    related=[("happiness", "word_family")],
                )
            ]
        )
    )
    lex = await _reading_lexicon(engine, repo)
    entry = await lex.get_entry(words[0].id)

    fam = [ln for ln in entry.links if ln.rel_type == "word_family"]
    assert len(fam) == 1
    assert fam[0].display == "happiness"
    assert fam[0].word_id is not None
    assert fam[0].status == "pending"
    # And the word-level relation is NOT duplicated onto the sense relations.
    assert entry.senses[0].relations == []
