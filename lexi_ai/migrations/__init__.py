"""Packaged Alembic entry points; host async connections use ``run_sync``."""

import asyncio
from pathlib import Path

from alembic import command
from alembic.config import Config
from alembic.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import create_async_engine

from lexi_ai.config import database_schema_name

ALEMBIC_INI = Path(__file__).resolve().parent.parent / "alembic.ini"


def get_migration_config(url: str | None = None, *, db_schema: str | None = None) -> Config:
    config = Config(str(ALEMBIC_INI))
    if url is not None:
        config.set_main_option("sqlalchemy.url", url.replace("%", "%%"))
    if db_schema is not None:
        config.set_main_option("db_schema", database_schema_name(db_schema))
    return config


def inspect_head() -> str:
    return ScriptDirectory.from_config(get_migration_config()).get_current_head()


def _run(url, connection, db_schema, operation):
    if (url is None) == (connection is None):
        raise ValueError("provide exactly one of url or connection")
    db_schema = database_schema_name(db_schema)
    if connection is not None:
        return operation(connection, db_schema)
    if not url.startswith(("postgresql+asyncpg://", "sqlite+aiosqlite://")):
        raise ValueError("expected an async generated-dictionary database URL")
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass
    else:
        raise RuntimeError("use asyncio.to_thread for URL calls or AsyncConnection.run_sync")

    async def run():
        engine = create_async_engine(url)
        try:
            async with engine.begin() as conn:
                return await conn.run_sync(operation, db_schema)
        finally:
            await engine.dispose()

    return asyncio.run(run())


def inspect_current(
    url: str | None = None,
    *,
    connection: Connection | None = None,
    db_schema: str | None = None,
) -> str | None:
    """Read the revision without creating tables or a schema."""

    def inspect(conn, schema):
        if conn.dialect.name != "postgresql" and schema is not None:
            raise ValueError("named database schema requires PostgreSQL")
        opts = (
            {"version_table_schema": schema or "lexi"}
            if (conn.dialect.name == "postgresql")
            else {}
        )
        return MigrationContext.configure(conn, opts=opts).get_current_revision()

    return _run(url, connection, db_schema, inspect)


def upgrade_to_head(
    url: str | None = None,
    *,
    connection: Connection | None = None,
    db_schema: str | None = None,
) -> None:
    """Initialize the generated dictionary; supplied connections remain caller-owned."""

    def upgrade(conn, schema):
        config = get_migration_config(db_schema=schema)
        config.attributes["connection"] = conn
        command.upgrade(config, "head")

    _run(url, connection, db_schema, upgrade)
