"""Migrate only the generated DB; never point this configuration at Cambridge."""

import asyncio

from alembic import context
from sqlalchemy import pool, text
from sqlalchemy.ext.asyncio import async_engine_from_config

from lexi_ai.config import database_schema_name
from lexi_ai.schema import Base

config = context.config
target_metadata = Base.metadata


def include_object(_object, name, type_, _reflected, _compare_to):
    return not (type_ == "table" and name == "alembic_version")


def _schema_name(dialect_name):
    name = database_schema_name(config.get_main_option("db_schema") or None)
    if dialect_name == "postgresql":
        return name or "lexi"
    if name is not None:
        raise ValueError("named database schema requires PostgreSQL")
    return None


def run_migrations_offline():
    raise RuntimeError("Dictionary bootstrap requires an online migration")


def do_run_migrations(connection):
    dialect_name = connection.dialect.name
    name = _schema_name(dialect_name)
    opts = {
        "connection": connection,
        "target_metadata": target_metadata,
        "compare_type": True,
        "include_object": include_object,
    }
    if dialect_name == "postgresql":
        original_schema = connection.dialect.default_schema_name
        original_path = connection.scalar(text("SHOW search_path"))
        connection.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{name}"'))
        connection.execute(text(f'SET LOCAL search_path TO "{name}"'))
        connection.dialect.default_schema_name = name
        opts["version_table_schema"] = name
    try:
        context.configure(**opts)
        with context.begin_transaction():
            context.run_migrations()
        if dialect_name == "postgresql":
            connection.execute(
                text("SELECT set_config('search_path', :path, true)"), {"path": original_path}
            )
    finally:
        if dialect_name == "postgresql":
            connection.dialect.default_schema_name = original_schema


async def run_async_migrations():
    if not config.get_main_option("sqlalchemy.url"):
        raise ValueError("an explicit generated-database URL is required")
    engine = async_engine_from_config(
        config.get_section(config.config_ini_section),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    try:
        async with engine.begin() as connection:
            await connection.run_sync(do_run_migrations)
    finally:
        await engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    connection = config.attributes.get("connection")
    if connection is None:
        asyncio.run(run_async_migrations())
    else:
        do_run_migrations(connection)
