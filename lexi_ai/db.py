"""Async engine + session factory, dual SQLite/Postgres (Phase 2).

SQLite does not enforce foreign keys (or ``ON DELETE CASCADE``) unless
``PRAGMA foreign_keys=ON`` is set on each connection, so we wire that on connect
for SQLite URLs. Postgres enforces FKs natively and needs no pragma.
"""

import os
import shutil
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from sqlalchemy import event, func, select
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from lexi_ai.config import Settings, get_settings
from lexi_ai.infrastructure.db.models import Asset, Base


def _enable_sqlite_fk(engine: AsyncEngine) -> None:
    """Turn on FK enforcement for SQLite connections (no-op elsewhere)."""
    if not engine.url.get_backend_name().startswith("sqlite"):
        return

    @event.listens_for(engine.sync_engine, "connect")
    def _set_pragma(dbapi_conn, _record):  # noqa: ANN001
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


def create_engine(settings: Settings | None = None) -> AsyncEngine:
    settings = settings or get_settings()
    engine_options = {}
    if settings.db_url.startswith("postgresql"):
        engine_options = {"execution_options": {"schema_translate_map": {None: settings.db_schema}}}
    engine = create_async_engine(settings.db_url, future=True, **engine_options)
    _enable_sqlite_fk(engine)
    return engine


def create_session_factory(
    engine: AsyncEngine,
) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


async def init_models(engine: AsyncEngine) -> None:
    """Create a fresh local SQLite database from the current ORM schema."""
    if engine.url.get_backend_name() != "sqlite":
        raise RuntimeError("library bootstrap DDL is supported only for SQLite")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def reset_assets_table(
    engine: AsyncEngine,
    *,
    cache_dir: str | None = None,
    allow_destructive: bool | None = None,
) -> None:
    """Drop + recreate the ``assets`` table and clear its on-disk cache shard.

    ``create_all`` cannot reshape an existing table, so a structural change to
    ``assets`` (the Phase 1 reference-addressing columns) needs an explicit
    drop+recreate. This is DESTRUCTIVE — it discards every cached translation/TTS
    row and its backing files — so it refuses to drop a NON-EMPTY table unless
    destructive migration is explicitly allowed (``allow_destructive`` arg, or the
    ``LEXI_ALLOW_DESTRUCTIVE_MIGRATION`` env flag). An empty table recreates freely.

    On reset the whole ``cache_dir`` is ``rmtree``d so orphaned binaries do not
    leak (3.8): this is safe ONLY because ``cache_dir`` is the assets-EXCLUSIVE
    shard (``config.asset_cache_dir`` = ``./lexi-assets``), never a shared dir —
    every file under it is a cached TTS/translation binary this table owns. Pass a
    dedicated assets dir; never point ``cache_dir`` at a path holding other data.
    """
    if engine.url.get_backend_name() != "sqlite":
        raise RuntimeError("library bootstrap DDL is supported only for SQLite")
    if allow_destructive is None:
        flag = os.environ.get("LEXI_ALLOW_DESTRUCTIVE_MIGRATION", "").strip().lower()
        allow_destructive = flag in ("1", "true", "yes")
    table = Base.metadata.tables[Asset.__tablename__]
    async with engine.begin() as conn:
        has_table = await conn.run_sync(
            lambda sync_conn: engine.dialect.has_table(sync_conn, Asset.__tablename__)
        )
        if has_table:
            count = await conn.scalar(select(func.count()).select_from(Asset))
            if count and not allow_destructive:
                raise RuntimeError(
                    f"refusing to drop non-empty {Asset.__tablename__!r} "
                    f"({count} rows): set LEXI_ALLOW_DESTRUCTIVE_MIGRATION=1 to allow"
                )
            await conn.run_sync(Base.metadata.drop_all, tables=[table], checkfirst=True)
        await conn.run_sync(Base.metadata.create_all, tables=[table], checkfirst=True)
    # Reclaim on-disk clips so orphaned binaries don't leak after the row drop.
    if cache_dir:
        shutil.rmtree(Path(cache_dir), ignore_errors=True)


@asynccontextmanager
async def session_scope(
    session_factory: async_sessionmaker[AsyncSession],
) -> AsyncIterator[AsyncSession]:
    """Transactional session scope: commit on success, rollback on error."""
    session = session_factory()
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()
