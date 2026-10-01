"""Migrate only the generated DB; never point this configuration at Cambridge."""

import asyncio

from alembic import context
from sqlalchemy import pool, text
from sqlalchemy.ext.asyncio import async_engine_from_config

from lexi_ai.config import database_schema_name
from lexi_ai.schema import Base
from lexi_ai.words.indexes import managed_search_object

config = context.config
target_metadata = Base.metadata


def include_object(_object, name, type_, _reflected, _compare_to):
    return not managed_search_object(name, type_)


def _schema_name():
    return database_schema_name(config.get_main_option("db_schema") or None)


def run_migrations_offline():
    name = _schema_name()
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        literal_binds=True,
        compare_type=True,
        include_object=include_object,
    )
    with context.begin_transaction():
        if name:
            context.execute(f'SET search_path TO "{name}"')
        context.run_migrations()


def do_run_migrations(connection):
    name = _schema_name()
    if name and connection.dialect.name == "postgresql":
        connection.execute(text(f'SET search_path TO "{name}"'))
        connection.dialect.default_schema_name = name
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
        include_object=include_object,
    )
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations():
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
