"""Tests for phrase-overlap data-prep (Phase 7).

Uses a tiny purpose-built Cambridge fixture DB (temp file, opened read-only by
the source) so classification is deterministic, plus an in-memory generated DB.
Asserts overlap/orphan classification, stub seeding, host linking, and
idempotency.
"""

import sqlite3

import pytest
from sqlalchemy import event, func, select
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import StaticPool

from lexi_ai.db import create_session_factory, init_models, session_scope
from lexi_ai.models import Word, WordRelation
from lexi_ai.normalize import match_key
from lexi_ai.persistence.repository import Repository
from lexi_ai.prep.phrase_overlap import PhraseOverlapPrep
from lexi_ai.references.cambridge import CambridgeSource


@pytest.fixture
def cambridge_fixture(tmp_path):
    """A minimal Cambridge DB: one host word with two phrase_titles, one of
    which ('give up') ALSO exists as a standalone word (overlap); the other
    ('at the end of the day') does not (orphan)."""
    db_path = tmp_path / "cam.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE words (
            id INTEGER PRIMARY KEY, word TEXT, display_form TEXT,
            entry_type TEXT, status TEXT
        );
        CREATE TABLE entries (
            id INTEGER PRIMARY KEY, word_id INTEGER, entry_order INTEGER, pos TEXT
        );
        CREATE TABLE senses (
            id INTEGER PRIMARY KEY, entry_id INTEGER, sense_order INTEGER,
            guideword TEXT, definition TEXT, cefr_level TEXT, domain TEXT,
            phrase_title TEXT
        );
        """
    )
    # Host word "day" carries two phrase_titles.
    conn.execute("INSERT INTO words VALUES (1,'day','day','word','done')")
    # Standalone "give up" exists as its own word -> overlap.
    conn.execute("INSERT INTO words VALUES (2,'give-up','give up','phrasal_verb','done')")
    conn.execute("INSERT INTO entries VALUES (10,1,0,'noun')")
    conn.execute(
        "INSERT INTO senses VALUES "
        "(100,10,0,'','a 24-hour period','A1',NULL,NULL),"
        "(101,10,1,'','stop trying','B1',NULL,'give up'),"
        "(102,10,2,'','finally','B2',NULL,'at the end of the day')"
    )
    conn.commit()
    conn.close()
    return str(db_path)


@pytest.fixture
async def generated_repo():
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
    session_factory = create_session_factory(engine)
    yield Repository(session_factory), session_factory
    await engine.dispose()


async def _count(session_factory, model) -> int:
    async with session_scope(session_factory) as session:
        return (await session.execute(select(func.count()).select_from(model))).scalar_one()


async def test_classification_counts(cambridge_fixture, generated_repo):
    repo, _sf = generated_repo
    prep = PhraseOverlapPrep(CambridgeSource(cambridge_fixture), repo)
    report = await prep.run()
    assert report.total == 2
    assert report.overlap == 1  # "give up" has a standalone row
    assert report.orphan == 1  # "at the end of the day" does not


async def test_orphan_seeded_as_pending_stub(cambridge_fixture, generated_repo):
    repo, session_factory = generated_repo
    prep = PhraseOverlapPrep(CambridgeSource(cambridge_fixture), repo)
    await prep.run()

    async with session_scope(session_factory) as session:
        orphan = (
            await session.execute(
                select(Word).where(Word.match_key == match_key("at the end of the day"))
            )
        ).scalar_one()
        assert orphan.status == "pending"


async def test_overlap_links_host_to_unit(cambridge_fixture, generated_repo):
    repo, session_factory = generated_repo
    prep = PhraseOverlapPrep(CambridgeSource(cambridge_fixture), repo)
    await prep.run()

    async with session_scope(session_factory) as session:
        host = (
            await session.execute(select(Word).where(Word.match_key == match_key("day")))
        ).scalar_one()
        unit = (
            await session.execute(select(Word).where(Word.match_key == match_key("give up")))
        ).scalar_one()
        link = (
            await session.execute(
                select(WordRelation).where(
                    WordRelation.from_word_id == host.id,
                    WordRelation.to_word_id == unit.id,
                )
            )
        ).scalar_one()
        assert link.rel_type == "part_of_phrasal_family"


async def test_prep_is_idempotent(cambridge_fixture, generated_repo):
    repo, session_factory = generated_repo
    prep = PhraseOverlapPrep(CambridgeSource(cambridge_fixture), repo)
    await prep.run()
    words_after_first = await _count(session_factory, Word)
    links_after_first = await _count(session_factory, WordRelation)

    # Re-run: no new stubs or links.
    await prep.run()
    assert await _count(session_factory, Word) == words_after_first
    assert await _count(session_factory, WordRelation) == links_after_first


async def test_prep_against_real_cambridge_smoke(generated_repo):
    import os

    if not os.path.exists("./data"):
        pytest.skip("Cambridge ./data not present")
    repo, session_factory = generated_repo
    prep = PhraseOverlapPrep(CambridgeSource("./data"), repo)
    report = await prep.run()
    # Scout figures: ~10.5k overlap, ~3.2k orphan (folding shifts these slightly).
    assert report.total > 13000
    assert report.overlap > 9000
    assert report.orphan > 2000
