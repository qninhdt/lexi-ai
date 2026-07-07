"""Tests for the ORM models and schema plumbing (Phase 2).

Uses in-memory SQLite. A single shared connection is kept for the lifetime of
each test (via ``poolclass=StaticPool``) so the schema persists across sessions.
"""

import pytest
from sqlalchemy import event, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.orm import selectinload
from sqlalchemy.pool import StaticPool

from lexi_ai.db import (
    create_session_factory,
    init_models,
    session_scope,
)
from lexi_ai.models import (
    EntryLink,
    Example,
    Sense,
    SenseReference,
    Word,
    WordAlias,
)


@pytest.fixture
async def session_factory():
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
    yield create_session_factory(engine)
    await engine.dispose()


async def _make_word(session, match_key="color", norm="color", **kwargs):
    """Add a Word (with any child collections passed via kwargs) and flush.

    Children must be supplied at construction time so the graph is built while
    the object is transient — appending to a relationship after flush would
    trigger an async lazy-load outside greenlet context. This mirrors how the
    Phase 5 repository assembles entries.
    """
    word = Word(norm=norm, match_key=match_key, entry_type="word", status="done", **kwargs)
    session.add(word)
    await session.flush()
    return word


async def test_init_models_creates_schema(session_factory):
    async with session_scope(session_factory) as session:
        result = await session.execute(select(Word))
        assert result.all() == []


async def test_full_graph_insert_and_traversal(session_factory):
    async with session_scope(session_factory) as session:
        sense = Sense(
            definition="the property of reflecting light",
            tier="core",
            sense_order=0,
            pos="noun",
            cefr_level="A1",
            references=[SenseReference(source="cambridge", source_ref="123")],
            examples=[Example(text="a bright color", example_order=0)],
        )
        await _make_word(
            session,
            aliases=[
                WordAlias(
                    alias_norm="colour",
                    alias_match_key="colour",
                    type="spelling_uk",
                    dialect="uk",
                )
            ],
            senses=[sense],
        )

    # Fresh session: assert the graph loads back with relationships.
    async with session_scope(session_factory) as session:
        loaded = (
            await session.execute(
                select(Word)
                .options(
                    selectinload(Word.senses).selectinload(Sense.references),
                    selectinload(Word.senses).selectinload(Sense.examples),
                    selectinload(Word.aliases),
                )
                .where(Word.match_key == "color")
            )
        ).scalar_one()

        assert loaded.norm == "color"
        assert len(loaded.aliases) == 1
        assert loaded.aliases[0].alias_match_key == "colour"
        assert len(loaded.senses) == 1
        assert loaded.senses[0].tier == "core"
        assert len(loaded.senses[0].references) == 1
        assert loaded.senses[0].references[0].source == "cambridge"
        assert len(loaded.senses[0].examples) == 1


async def test_unique_word_match_key(session_factory):
    with pytest.raises(IntegrityError):
        async with session_scope(session_factory) as session:
            await _make_word(session, match_key="dup", norm="dup")
            await _make_word(session, match_key="dup", norm="dup2")


async def test_unique_alias_word_key(session_factory):
    with pytest.raises(IntegrityError):
        async with session_scope(session_factory) as session:
            await _make_word(
                session,
                aliases=[
                    WordAlias(
                        alias_norm="colour",
                        alias_match_key="colour",
                        type="spelling_uk",
                    ),
                    WordAlias(
                        alias_norm="colour",
                        alias_match_key="colour",
                        type="spelling_other",
                    ),
                ],
            )


async def test_entry_link_unique_triple(session_factory):
    with pytest.raises(IntegrityError):
        async with session_scope(session_factory) as session:
            a = await _make_word(session, match_key="a", norm="a")
            b = await _make_word(session, match_key="b", norm="b")
            session.add_all(
                [
                    EntryLink(from_word_id=a.id, to_word_id=b.id, rel_type="synonym"),
                    EntryLink(from_word_id=a.id, to_word_id=b.id, rel_type="synonym"),
                ]
            )
            await session.flush()


async def test_cascade_delete_word_removes_children(session_factory):
    async with session_scope(session_factory) as session:
        await _make_word(
            session,
            senses=[Sense(definition="d", tier="core")],
            aliases=[WordAlias(alias_norm="c", alias_match_key="c", type="spelling_us")],
        )

    async with session_scope(session_factory) as session:
        word = (await session.execute(select(Word).where(Word.match_key == "color"))).scalar_one()
        await session.delete(word)

    async with session_scope(session_factory) as session:
        assert (await session.execute(select(Sense))).all() == []
        assert (await session.execute(select(WordAlias))).all() == []


# --- dual-dialect portability (no live Postgres needed) -------------------


def test_schema_compiles_on_both_dialects():
    """Every table's DDL must compile for SQLite AND Postgres (criterion #7).

    Proves no dialect-specific type blocks the Postgres target without needing a
    running Postgres — a mismatched type raises at compile time.
    """
    from sqlalchemy.dialects import postgresql, sqlite
    from sqlalchemy.schema import CreateTable

    from lexi_ai.models import Base

    # The tag + collocation + sense_forms + questions tables ride the same
    # portability guarantee.
    table_names = set(Base.metadata.tables)
    assert {"tags", "word_tags", "collocations", "sense_forms", "questions"} <= table_names

    for dialect in (postgresql.dialect(), sqlite.dialect()):
        for table in Base.metadata.sorted_tables:
            ddl = str(CreateTable(table).compile(dialect=dialect))
            assert "CREATE TABLE" in ddl
