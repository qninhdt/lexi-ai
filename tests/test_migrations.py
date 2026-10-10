"""Squashed baseline parity, repeatability and non-destructive bootstrap guards."""

import asyncio
import json

import pytest
from alembic import command
from alembic.script import ScriptDirectory
from sqlalchemy import inspect, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import create_async_engine

from lexi_ai.migrations import (
    get_migration_config as migration_config,
)
from lexi_ai.migrations import inspect_current, inspect_head, upgrade_to_head
from lexi_ai.schema import Base


async def snapshot(url):
    engine = create_async_engine(url)
    try:
        async with engine.connect() as connection:

            def reflect(sync_connection):
                inspector = inspect(sync_connection)
                return {
                    name: (
                        {column["name"] for column in inspector.get_columns(name)},
                        {index["name"] for index in inspector.get_indexes(name)},
                        {
                            (
                                tuple(fk["constrained_columns"]),
                                fk["referred_table"],
                                fk["options"].get("ondelete"),
                            )
                            for fk in inspector.get_foreign_keys(name)
                        },
                    )
                    for name in inspector.get_table_names()
                }

            return await connection.run_sync(reflect)
    finally:
        await engine.dispose()


async def run_migration(config, url, operation, revision=None):
    engine = create_async_engine(url)
    try:
        async with engine.begin() as connection:

            def execute(sync_connection):
                config.attributes["connection"] = sync_connection
                try:
                    if revision is None:
                        operation(config)
                    else:
                        operation(config, revision)
                finally:
                    del config.attributes["connection"]

            await connection.run_sync(execute)
    finally:
        await engine.dispose()


async def test_baseline_upgrade_drift_downgrade_and_reupgrade(tmp_path):
    url = f"sqlite+aiosqlite:///{tmp_path / 'generated.db'}"
    config = migration_config(url)
    revisions = list(ScriptDirectory.from_config(config).walk_revisions())
    assert [revision.revision for revision in revisions] == [
        "20261010_no_notes",
        "20261010_compact",
        "20261010_ref_forms",
        "20261009_base",
    ]
    assert revisions[-1].down_revision is None
    await run_migration(config, url, command.upgrade, "head")
    actual = await snapshot(url)
    assert set(actual) == {"alembic_version", *Base.metadata.tables}
    for name, table in Base.metadata.tables.items():
        columns, indexes, foreign_keys = actual[name]
        assert columns == set(table.columns.keys())
        assert indexes == {index.name for index in table.indexes}
        assert foreign_keys == {
            ((column.name,), fk.column.table.name, fk.ondelete)
            for column in table.columns
            for fk in column.foreign_keys
        }
    await run_migration(config, url, command.check)

    await run_migration(config, url, command.downgrade, "base")
    assert set(await snapshot(url)) <= {"alembic_version"}
    await run_migration(config, url, command.upgrade, "head")
    assert await snapshot(url) == actual
    await run_migration(config, url, command.check)


async def test_baseline_refuses_existing_tables_without_changing_their_data(tmp_path):
    url = f"sqlite+aiosqlite:///{tmp_path / 'existing.db'}"
    engine = create_async_engine(url)
    try:
        async with engine.begin() as connection:
            await connection.execute(text("CREATE TABLE words(id INTEGER PRIMARY KEY, lemma TEXT)"))
            await connection.execute(text("INSERT INTO words VALUES(1,'keep me')"))
        with pytest.raises(RuntimeError, match="empty generated dictionary"):
            await run_migration(migration_config(url), url, command.upgrade, "head")
        async with engine.connect() as connection:
            assert (await connection.execute(text("SELECT * FROM words"))).all() == [(1, "keep me")]
    finally:
        await engine.dispose()


async def test_baseline_does_not_create_new_live_metadata_tables(tmp_path):
    from sqlalchemy import Column, Integer, Table

    extra = Table("not_in_baseline", Base.metadata, Column("id", Integer, primary_key=True))
    try:
        url = f"sqlite+aiosqlite:///{tmp_path / 'frozen.db'}"
        await run_migration(migration_config(url), url, command.upgrade, "head")
        assert "not_in_baseline" not in await snapshot(url)
    finally:
        Base.metadata.remove(extra)


async def test_compact_content_migrates_saved_tags_and_preserves_ids(tmp_path):
    url = f"sqlite+aiosqlite:///{tmp_path / 'content.db'}"
    config = migration_config(url)
    await run_migration(config, url, command.upgrade, "20261010_ref_forms")
    engine = create_async_engine(url)
    old = 'She <t inf="PAST">ran</t> <t inf="base">home</t>.'
    payload = {"content": old, "correct": {"content": '<t inf="base">run</t>'}}
    try:
        async with engine.begin() as connection:
            await connection.execute(text("INSERT INTO words VALUES(1,'run','run','WORD','DONE')"))
            await connection.execute(
                text("INSERT INTO senses(id,word_id,pos,tier) VALUES(1,1,'VERB','CORE')")
            )
            await connection.execute(
                text("INSERT INTO examples(id,sense_id,content) VALUES(17,1,:content)"),
                {"content": old},
            )
            await connection.execute(
                text("INSERT INTO definitions(id,sense_id,content) VALUES(18,1,'move quickly')")
            )
            await connection.execute(
                text(
                    "INSERT INTO questions(id,sense_id,question_type,payload) "
                    "VALUES(19,1,'WORD_TO_USAGE',:payload)"
                ),
                {"payload": json.dumps(payload)},
            )
        await run_migration(config, url, command.upgrade, "head")
        async with engine.connect() as connection:
            assert (await connection.execute(text("SELECT id,content FROM examples"))).all() == [
                (17, "She [ran|p] [home].")
            ]
            assert (await connection.execute(text("SELECT id,content FROM definitions"))).all() == [
                (18, "move quickly")
            ]
            stored = await connection.scalar(text("SELECT payload FROM questions WHERE id=19"))
            assert json.loads(stored) == {
                "content": "She [ran|p] [home].",
                "correct": {"content": "[run]"},
            }
            columns = await connection.run_sync(
                lambda conn: {c["name"] for c in inspect(conn).get_columns("sense_relations")}
            )
            assert "gloss" not in columns
        await run_migration(config, url, command.upgrade, "head")
    finally:
        await engine.dispose()


async def test_online_migration_uses_config_not_environment(monkeypatch, tmp_path):
    url = f"sqlite+aiosqlite:///{tmp_path / 'explicit.db'}"
    monkeypatch.setenv("DB_URL", f"sqlite+aiosqlite:///{tmp_path / 'wrong.db'}")
    monkeypatch.setenv("DB_SCHEMA", "invalid; schema")
    await asyncio.to_thread(command.upgrade, migration_config(url), "head")
    assert set(await snapshot(url)) == {"alembic_version", *Base.metadata.tables}
    assert not (tmp_path / "wrong.db").exists()


async def test_remove_usage_notes_preserves_sense_content(tmp_path):
    url = f"sqlite+aiosqlite:///{tmp_path / 'notes.db'}"
    config = migration_config(url)
    await run_migration(config, url, command.upgrade, "20261010_compact")
    engine = create_async_engine(url)
    try:
        async with engine.begin() as connection:
            await connection.execute(
                text("INSERT INTO words VALUES(1,'room','room','WORD','DONE')")
            )
            await connection.execute(
                text(
                    "INSERT INTO senses(id,word_id,pos,tier,usage_note) "
                    "VALUES(7,1,'NOUN','CORE','Uncountable in this sense.')"
                )
            )
            await connection.execute(
                text("INSERT INTO definitions(id,sense_id,content) VALUES(9,7,'Available space')")
            )
        await run_migration(config, url, command.upgrade, "head")
        assert "usage_note" not in (await snapshot(url))["senses"][0]
        async with engine.connect() as connection:
            assert (
                await connection.execute(text("SELECT id,word_id,pos,tier FROM senses"))
            ).all() == [(7, 1, "NOUN", "CORE")]
            assert (
                await connection.execute(text("SELECT id,sense_id,content FROM definitions"))
            ).all() == [(9, 7, "Available space")]
        await run_migration(config, url, command.downgrade, "20261010_compact")
        async with engine.connect() as connection:
            assert (await connection.execute(text("SELECT id,usage_note FROM senses"))).all() == [
                (7, None)
            ]
        await run_migration(config, url, command.upgrade, "head")
    finally:
        await engine.dispose()


async def test_packaged_migrations_are_idempotent_and_enforce_current_tiers(tmp_path):
    url = f"sqlite+aiosqlite:///{tmp_path / 'generated.db'}"
    assert await asyncio.to_thread(inspect_current, url) is None
    await asyncio.to_thread(upgrade_to_head, url)
    assert await asyncio.to_thread(inspect_current, url) == inspect_head()
    engine = create_async_engine(url)
    try:
        async with engine.begin() as connection:
            await connection.run_sync(lambda conn: upgrade_to_head(connection=conn))
            await connection.execute(
                text("INSERT INTO words VALUES(1,'bank','bank','WORD','DONE')")
            )
            await connection.execute(
                text("INSERT INTO senses(id,word_id,pos,tier) VALUES(1,1,'NOUN','LESS_COMMON')")
            )
            with pytest.raises(IntegrityError):
                await connection.execute(text("UPDATE senses SET tier='invalid' WHERE id=1"))
            assert await connection.scalar(text("SELECT tier FROM senses")) == "LESS_COMMON"
            assert await connection.run_sync(lambda conn: inspect_current(connection=conn)) == (
                inspect_head()
            )
    finally:
        await engine.dispose()


def test_migration_requires_explicit_database():
    with pytest.raises(ValueError, match="exactly one"):
        upgrade_to_head()
    with pytest.raises(ValueError, match="explicit generated-database"):
        command.upgrade(migration_config(), "head")


async def test_url_migrations_do_not_block_an_active_event_loop(tmp_path):
    url = f"sqlite+aiosqlite:///{tmp_path / 'unused.db'}"
    with pytest.raises(RuntimeError, match="to_thread"):
        upgrade_to_head(url)
    assert not (tmp_path / "unused.db").exists()
