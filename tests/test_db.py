"""Tests for the schema-migration plumbing (Phase 2).

``migrate_relations`` renames the legacy ``entry_links`` table to
``word_relation`` (metadata-only, no row copy) BEFORE ``create_all`` runs, so a
real 149MB DB keeps its word-level data instead of being left orphaned next to
an empty new table. In-memory SQLite with a shared connection (StaticPool).
"""

import pytest
from sqlalchemy import event, select, text
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import StaticPool

from lexi_ai.db import (
    create_session_factory,
    init_models,
    migrate_relations,
    session_scope,
)
from lexi_ai.models import Word, WordRelation


@pytest.fixture
async def raw_engine():
    """A bare engine with FK enforcement on, but NO schema created yet."""
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

    yield engine
    await engine.dispose()


async def _seed_legacy_entry_links(engine, n_words: int = 2, n_links: int = 1) -> None:
    """Create a legacy ``entry_links`` table + ``words`` and populate rows via raw
    SQL, simulating a pre-migration production DB."""
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "CREATE TABLE words (id INTEGER PRIMARY KEY, norm TEXT NOT NULL, "
                "match_key VARCHAR(512) NOT NULL UNIQUE, entry_type VARCHAR(32), "
                "status VARCHAR(16) NOT NULL DEFAULT 'pending', pos VARCHAR(32), "
                "cambridge_word_id INTEGER, error_msg TEXT, "
                "created_at DATETIME, updated_at DATETIME)"
            )
        )
        await conn.execute(
            text(
                "CREATE TABLE entry_links (id INTEGER PRIMARY KEY, "
                "from_word_id INTEGER NOT NULL, to_word_id INTEGER NOT NULL, "
                "rel_type VARCHAR(32) NOT NULL)"
            )
        )
        for i in range(1, n_words + 1):
            await conn.execute(
                text(
                    "INSERT INTO words (id, norm, match_key, status) "
                    "VALUES (:id, :norm, :key, 'done')"
                ),
                {"id": i, "norm": f"w{i}", "key": f"w{i}"},
            )
        for j in range(1, n_links + 1):
            await conn.execute(
                text(
                    "INSERT INTO entry_links (id, from_word_id, to_word_id, rel_type) "
                    "VALUES (:id, 1, 2, 'synonym')"
                ),
                {"id": j},
            )


async def _has_table(engine, name: str) -> bool:
    async with engine.begin() as conn:
        return await conn.run_sync(
            lambda sync_conn: engine.dialect.has_table(sync_conn, name)
        )


async def test_migrate_renames_legacy_entry_links(raw_engine):
    await _seed_legacy_entry_links(raw_engine, n_words=2, n_links=1)
    await migrate_relations(raw_engine)

    # entry_links is gone; word_relation carries the same row.
    assert not await _has_table(raw_engine, "entry_links")
    assert await _has_table(raw_engine, "word_relation")
    sf = create_session_factory(raw_engine)
    async with session_scope(sf) as session:
        rows = list((await session.execute(select(WordRelation))).scalars())
    assert len(rows) == 1
    assert rows[0].rel_type == "synonym"


async def test_migrate_is_idempotent(raw_engine):
    await _seed_legacy_entry_links(raw_engine, n_words=2, n_links=1)
    await migrate_relations(raw_engine)
    # Running again must not raise nor duplicate.
    await migrate_relations(raw_engine)
    sf = create_session_factory(raw_engine)
    async with session_scope(sf) as session:
        rows = list((await session.execute(select(WordRelation))).scalars())
    assert len(rows) == 1


async def test_migrate_noop_on_fresh_db(raw_engine):
    # No legacy entry_links: migrate alone renames nothing (it does not create
    # tables — that is create_all's job). init() then builds the fresh schema.
    await migrate_relations(raw_engine)  # no-op, must not raise
    assert not await _has_table(raw_engine, "word_relation")
    await init_models(raw_engine)
    assert await _has_table(raw_engine, "word_relation")
    assert await _has_table(raw_engine, "sense_relation")


async def test_init_migrates_populated_entry_links(raw_engine):
    # [RED TEAM F1] init_models MUST migrate legacy data, not orphan it.
    await _seed_legacy_entry_links(raw_engine, n_words=3, n_links=2)
    await init_models(raw_engine)

    assert not await _has_table(raw_engine, "entry_links")  # no orphan table
    sf = create_session_factory(raw_engine)
    async with session_scope(sf) as session:
        rows = list((await session.execute(select(WordRelation))).scalars())
        words = list((await session.execute(select(Word))).scalars())
    assert len(rows) == 2  # both legacy links survived under the new table
    assert len(words) == 3
