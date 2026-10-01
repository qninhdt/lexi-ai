import asyncio

import pytest
from sqlalchemy import select, text

from lexi_ai.db.session import Database
from lexi_ai.schema import Base, Word


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
                        entry_type="word",
                        generation_state="done",
                    )
                )
                await session.flush()
                raise RuntimeError("publish failed")
        async with db.transaction() as session:
            assert (await session.scalars(select(Word))).all() == []
    finally:
        await db.close()
