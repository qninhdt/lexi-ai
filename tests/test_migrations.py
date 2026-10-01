"""Squashed baseline parity, repeatability and non-destructive bootstrap guards."""

import asyncio
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy import inspect, text
from sqlalchemy.ext.asyncio import create_async_engine

from lexi_ai.schema import Base

ALEMBIC_INI = Path(__file__).resolve().parent.parent / "lexi_ai" / "alembic.ini"


def migration_config(url: str) -> Config:
    config = Config(str(ALEMBIC_INI))
    config.set_main_option("sqlalchemy.url", url.replace("%", "%%"))
    return config


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
    assert [revision.revision for revision in revisions] == ["20260930_base"]
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


async def test_online_migration_uses_config_not_environment(monkeypatch, tmp_path):
    url = f"sqlite+aiosqlite:///{tmp_path / 'explicit.db'}"
    monkeypatch.setenv("LEXI_DB_URL", f"sqlite+aiosqlite:///{tmp_path / 'wrong.db'}")
    monkeypatch.setenv("LEXI_DB_SCHEMA", "invalid; schema")
    await asyncio.to_thread(command.upgrade, migration_config(url), "head")
    assert set(await snapshot(url)) == {"alembic_version", *Base.metadata.tables}
    assert not (tmp_path / "wrong.db").exists()
