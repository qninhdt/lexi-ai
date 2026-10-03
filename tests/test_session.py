import asyncio

import pytest
from sqlalchemy import select, text
from test_db_optimization import optimized_db as optimized_db
from test_db_optimization import sqlite_db as sqlite_db
from test_end_to_end import verify_consumer_flow
from test_query_work import shared_target
from test_relations import Decision

from lexi_ai import Lexicon
from lexi_ai.db.session import Database, SessionDatabase
from lexi_ai.schema import Base, SenseRelation, Word


async def test_complete_consumer_flow_uses_host_session(optimized_db, source):
    async with optimized_db.sessions() as session, session.begin():
        await verify_consumer_flow(None, source, session=session)
        assert session.in_transaction()
        await session.rollback()
    async with optimized_db.read() as connection:
        assert await connection.scalar(select(Word.id)) is None


async def test_parallel_relation_inference_serializes_borrowed_session_sql(
    optimized_db,
    monkeypatch,
):
    await shared_target(optimized_db, source_count=3)
    active = peak = 0
    async with optimized_db.sessions() as session, session.begin():
        original = session.execute

        async def tracked(*args, **kwargs):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            try:
                await asyncio.sleep(0.002)
                return await original(*args, **kwargs)
            finally:
                active -= 1

        monkeypatch.setattr(session, "execute", tracked)
        lexicon = Lexicon(session=session, llm=object(), decision_model=Decision("candidate_1"))
        try:
            results = await lexicon.resolve_relations(batch_size=3)
            assert [result.state for result in results] == ["RESOLVED"] * 3
            assert peak == 1
            await session.rollback()
        finally:
            await lexicon.close()
    async with optimized_db.read() as connection:
        assert (await connection.scalars(select(SenseRelation.to_sense_id))).all() == [None] * 3


def test_host_session_configuration_does_not_silently_ignore_database_arguments():
    from sqlalchemy.ext.asyncio import AsyncSession

    with pytest.raises(ValueError, match="exactly one"):
        Lexicon(llm=object())
    with pytest.raises(ValueError, match="exactly one"):
        Lexicon("sqlite+aiosqlite:///:memory:", session=AsyncSession(), llm=object())
    with pytest.raises(ValueError, match="host session"):
        Lexicon(session=AsyncSession(), db_schema="lexi", llm=object())


async def test_file_database_configures_every_connection_and_persists_wal(tmp_path):
    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'generated.db'}")
    try:
        await db.create_schema(Base.metadata)
        async with db.engine.connect() as first, db.engine.connect() as second:
            for connection in (first, second):
                assert await connection.scalar(text("PRAGMA journal_mode")) == "wal"
                assert await connection.scalar(text("PRAGMA busy_timeout")) == 5000
                assert await connection.scalar(text("PRAGMA foreign_keys")) == 1
        await db.engine.dispose()
        async with db.engine.connect() as connection:
            assert await connection.scalar(text("PRAGMA journal_mode")) == "wal"
    finally:
        await db.close()


async def test_host_session_keeps_transaction_and_lifecycle_ownership(tmp_path):
    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'host.db'}")
    try:
        await db.create_schema(Base.metadata)
        async with db.sessions() as session:
            adapter = SessionDatabase(session)
            lexicon = Lexicon(session=session, llm=object())
            assert lexicon.db.content_cache is None and lexicon.db.question_cache is None
            with pytest.raises(ValueError, match="borrowed transaction"):
                await lexicon.start()
            async with session.begin():
                async with adapter.transaction() as borrowed:
                    assert borrowed is session
                    borrowed.add(Word(lemma="bank", match_key="bank", generation_state="DONE"))
                    await borrowed.flush()
                await lexicon.close()
                assert session.in_transaction()
                await session.rollback()
            async with session.begin():
                assert await session.scalar(select(Word.id)) is None
                async with adapter.transaction() as borrowed:
                    borrowed.add(Word(lemma="shore", match_key="shore"))
            async with db.read() as connection:
                assert (await connection.scalars(select(Word.lemma))).all() == ["shore"]
    finally:
        await db.close()


async def test_memory_database_keeps_memory_journal():
    db = Database("sqlite+aiosqlite:///:memory:")
    try:
        async with db.engine.connect() as connection:
            assert await connection.scalar(text("PRAGMA journal_mode")) == "memory"
            assert await connection.scalar(text("PRAGMA foreign_keys")) == 1
    finally:
        await db.close()


async def test_concurrent_independent_writes_commit(tmp_path):
    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'generated.db'}")
    try:
        await db.create_schema(Base.metadata)

        async def publish(number):
            async with db.transaction() as session:
                session.add(Word(lemma=f"word{number}", match_key=f"word{number}"))
                await session.flush()
                await asyncio.sleep(0.005)

        await asyncio.gather(*(publish(number) for number in range(20)))
        async with db.transaction() as session:
            assert len((await session.scalars(select(Word))).all()) == 20
    finally:
        await db.close()


async def test_transaction_rolls_back_partial_publish(tmp_path):
    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'generated.db'}")
    try:
        await db.create_schema(Base.metadata)
        with pytest.raises(RuntimeError, match="publish failed"):
            async with db.transaction() as session:
                session.add(
                    Word(
                        lemma="glisten",
                        match_key="glisten",
                        entry_type="WORD",
                        generation_state="DONE",
                    )
                )
                await session.flush()
                raise RuntimeError("publish failed")
        async with db.transaction() as session:
            assert (await session.scalars(select(Word))).all() == []
    finally:
        await db.close()
