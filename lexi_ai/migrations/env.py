"""Alembic environment for Lexi's PostgreSQL domain schema."""

from __future__ import annotations

import asyncio
import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool, schema
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = None


def _database_url() -> str:
    return (
        os.environ.get("LEXI_DB_URL") or config.get_main_option("sqlalchemy.url")
    )


def _schema() -> str:
    return (
        context.get_x_argument(as_dictionary=True).get("schema")
        or os.environ.get("LEXI_DB_SCHEMA")
        or config.get_main_option("lexi.schema")
    )


def _configure(connection: Connection | None = None) -> str:
    schema = _schema()
    config.attributes["lexi_schema"] = schema
    options = {
        "target_metadata": target_metadata,
        "version_table_schema": schema,
        "include_schemas": True,
    }
    if connection is None:
        context.configure(
            url=_database_url(),
            literal_binds=True,
            dialect_opts={"paramstyle": "named"},
            **options,
        )
    else:
        context.configure(connection=connection, **options)
    return schema


def run_migrations_offline() -> None:
    schema_name = _configure()
    context.execute(schema.CreateSchema(schema_name, if_not_exists=True))
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    schema_name = _schema()
    connection.execute(schema.CreateSchema(schema_name, if_not_exists=True))
    _configure(connection)
    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    configuration = config.get_section(config.config_ini_section, {})
    configuration["sqlalchemy.url"] = _database_url()
    connectable = async_engine_from_config(
        configuration, prefix="sqlalchemy.", poolclass=pool.NullPool
    )
    # Engine.begin commits Alembic's version update and schema DDL on success.
    async with connectable.begin() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())
