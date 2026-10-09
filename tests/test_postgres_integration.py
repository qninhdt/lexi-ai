"""Opt-in only: use LEXI_TEST_PG_URL for an explicitly disposable database."""

import asyncio
import os
import subprocess
import sys
import uuid
from pathlib import Path

import pytest
from alembic import command
from sqlalchemy import func, inspect, select, text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import create_async_engine
from test_end_to_end import verify_consumer_flow
from test_migrations import run_migration
from test_relations import Decision
from test_scale_fixes import seed_links

from lexi_ai.contract import LEXI_SCHEMA
from lexi_ai.contract import Sense as ContractSense
from lexi_ai.db.session import Database
from lexi_ai.migrations import get_migration_config, inspect_current, inspect_head, upgrade_to_head
from lexi_ai.relations.resolve import resolve_relations
from lexi_ai.relations.storage import definition_hash, pending_relations
from lexi_ai.schema import Base, Definition, Sense, SenseForm, SenseRelation, Word
from lexi_ai.words.search import Search, search

PG_URL = os.getenv("LEXI_TEST_PG_URL")
pytestmark = pytest.mark.skipif(not PG_URL, reason="no disposable LEXI_TEST_PG_URL")


async def test_migration_nondefault_schema(source):
    assert PG_URL.startswith("postgresql+asyncpg://")
    schema = f"Lexi_test_{uuid.uuid4().hex[:12]}"
    engine = create_async_engine(PG_URL)
    config = get_migration_config(PG_URL, db_schema=schema)
    try:
        async with engine.begin() as connection:
            public_before = await connection.run_sync(
                lambda sync: set(inspect(sync).get_table_names(schema="public"))
            )
            await connection.execute(text(f'CREATE SCHEMA "{schema}"'))
        # Exercise the actual online entry point, not only a shared test connection.
        script = """
import os
import sys
from lexi_ai.migrations import upgrade_to_head
upgrade_to_head(os.environ['LEXI_TEST_PG_URL'], db_schema=sys.argv[1])
"""
        env = os.environ.copy()
        env["PYTHONPATH"] = str(Path(__file__).parent)
        subprocess.run(
            [sys.executable, "-c", script, schema],
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
        assert await asyncio.to_thread(inspect_current, PG_URL, db_schema=schema) == inspect_head()
        async with engine.connect() as connection:

            def reflect(sync_connection):
                inspector = inspect(sync_connection)
                return {
                    name: (
                        {column["name"] for column in inspector.get_columns(name, schema=schema)},
                        {
                            index["name"]
                            for index in inspector.get_indexes(name, schema=schema)
                            if not index.get("duplicates_constraint")
                        },
                        {
                            (
                                tuple(fk["constrained_columns"]),
                                fk["referred_table"],
                                fk["options"].get("ondelete"),
                            )
                            for fk in inspector.get_foreign_keys(name, schema=schema)
                        },
                    )
                    for name in inspector.get_table_names(schema=schema)
                }

            reflected = await connection.run_sync(reflect)
        tables = set(reflected)
        assert tables == {"alembic_version", *Base.metadata.tables}
        for name, table in Base.metadata.tables.items():
            columns, indexes, foreign_keys = reflected[name]
            assert columns == set(table.columns.keys())
            expected_indexes = {index.name for index in table.indexes}
            assert indexes == expected_indexes
            assert foreign_keys == {
                ((column.name,), fk.column.table.name, fk.ondelete)
                for column in table.columns
                for fk in column.foreign_keys
            }
        await run_migration(config, PG_URL, command.check)
        await asyncio.to_thread(upgrade_to_head, PG_URL, db_schema=schema)
        async with engine.begin() as connection:
            path_before = await connection.scalar(text("SHOW search_path"))
            reflected_schema = connection.dialect.default_schema_name
            await connection.run_sync(
                lambda conn: upgrade_to_head(connection=conn, db_schema=schema)
            )
            assert await connection.scalar(text("SHOW search_path")) == path_before
            assert connection.dialect.default_schema_name == reflected_schema
            assert (
                await connection.run_sync(
                    lambda conn: inspect_current(connection=conn, db_schema=schema)
                )
                == inspect_head()
            )
        await verify_consumer_flow(PG_URL, source, db_schema=schema)
        await run_migration(config, PG_URL, command.downgrade, "base")
        await run_migration(config, PG_URL, command.upgrade, "head")
        await run_migration(config, PG_URL, command.check)
        db = Database(PG_URL, schema=schema)
        try:
            async with db.transaction() as session:
                assert await session.scalar(text("SELECT current_schema()")) == schema
                word = Word(
                    lemma="bank", match_key="bank", entry_type="WORD", generation_state="DONE"
                )
                session.add(word)
                await session.flush()
                sense = Sense(word_id=word.id, pos="NOUN", tier="CORE")
                session.add(sense)
                await session.flush()
                session.add(Definition(sense_id=sense.id, content="Money keeper"))
            with pytest.raises(DBAPIError):
                async with db.transaction() as session:
                    session.add(Definition(sense_id=sense.id, content="NUL\x00is invalid"))
                    await session.flush()
            async with db.engine.connect() as connection:
                connection = await connection.execution_options(
                    schema_translate_map={LEXI_SCHEMA: schema}
                )
                actual = (await connection.execute(select(ContractSense.__table__))).one()
                assert (actual.id, actual.word_id, actual.pos, actual.tier) == (
                    sense.id,
                    word.id,
                    "NOUN",
                    "CORE",
                )
                columns = await connection.run_sync(
                    lambda conn: inspect(conn).get_columns("senses", schema=schema)
                )
                assert {column["name"]: str(column["type"]) for column in columns} == {
                    column.name: str(column.type) for column in ContractSense.__table__.columns
                }
            with pytest.raises(DBAPIError):
                async with db.transaction() as session:
                    session.add(
                        Word(
                            lemma="too long",
                            match_key="x" * 513,
                            entry_type="WORD",
                            generation_state="DONE",
                        )
                    )
                    await session.flush()
            async with db.transaction() as session:
                assert await session.scalar(select(func.count()).select_from(Word)) == 1
                assert await session.scalar(select(func.count()).select_from(Definition)) == 1
                await session.delete(await session.get(Word, word.id))
            # Exercise migrated trigger DDL in this nondefault PostgreSQL schema.
            await seed_links(db)
            assert await pending_relations(db, 20) == []
            async with db.transaction() as session:
                session.add(SenseForm(sense_id=1, surface="ＴＯＯＫ　ＯＦＦ", inf="PAST"))
                await session.execute(text("UPDATE definitions SET content='themed' WHERE id=6"))
            assert await pending_relations(db, 20) == []
            db.search_index = Search(db, None)
            await db.search_index.start()
            assert (await search(db, "TOOK OFF")).items[0].match_kind == "EXACT"
            with pytest.raises(RuntimeError):
                async with db.transaction() as session:
                    await session.execute(
                        text("UPDATE definitions SET content='changed' WHERE id=2")
                    )
                    assert (await session.get(SenseRelation, 1)).resolve_attempted_at is None
                    raise RuntimeError("rollback")
            assert await pending_relations(db, 20) == []
            async with db.transaction() as session:
                await session.execute(text("UPDATE definitions SET content='changed' WHERE id=2"))
            assert [link.edge_id for link in await pending_relations(db, 20)] == [1]
            decisions = await resolve_relations(db, Decision("candidate_1"))
            assert [item.state for item in decisions] == ["RESOLVED"]
            async with db.transaction() as session:
                edge = await session.get(SenseRelation, 1)
                edge.to_sense_id = edge.target_hash = edge.resolve_attempted_at = None

            async def change_while_resolver_holds_lock():
                async with db.transaction() as session:
                    await session.execute(
                        text("UPDATE definitions SET content='racing update' WHERE id=2")
                    )

            writer = None
            try:
                async with db.transaction() as session:
                    edge = await session.scalar(
                        select(SenseRelation).where(SenseRelation.id == 1).with_for_update()
                    )
                    writer = asyncio.create_task(change_while_resolver_holds_lock())
                    # The trigger must lock even an as-yet pending edge, rather
                    # than skipping it and letting this old decision commit last.
                    with pytest.raises(asyncio.TimeoutError):
                        await asyncio.wait_for(asyncio.shield(writer), timeout=0.5)
                    edge.to_sense_id = 2
                    edge.target_hash = definition_hash("changed")
                    edge.resolve_attempted_at = "2026-09-30T00:00:00Z"
                await asyncio.wait_for(writer, timeout=10)
            finally:
                if writer is not None:
                    if not writer.done():
                        writer.cancel()
                    await asyncio.gather(writer, return_exceptions=True)
            async with db.transaction() as session:
                edge = await session.get(SenseRelation, 1)
                assert edge.resolve_attempted_at is None and edge.to_sense_id is None
            async with db.transaction() as session:
                await session.delete(await session.get(Sense, 2))
            async with db.transaction() as session:
                edge = await session.get(SenseRelation, 1)
                assert edge.resolve_attempted_at is None and edge.to_sense_id is None
                assert (await session.get(SenseRelation, 2)).to_sense_id == 4
        finally:
            await db.close()
        async with engine.connect() as connection:
            public_after = await connection.run_sync(
                lambda sync: set(inspect(sync).get_table_names(schema="public"))
            )
        assert public_after == public_before
    finally:
        async with engine.begin() as connection:
            await connection.execute(text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))
        await engine.dispose()
