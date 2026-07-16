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
from lexi_ai.models import Asset, Base


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
    engine = create_async_engine(settings.db_url, future=True)
    _enable_sqlite_fk(engine)
    return engine


def create_session_factory(
    engine: AsyncEngine,
) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


async def migrate_relations(engine: AsyncEngine) -> None:
    """Rename the legacy ``entry_links`` table to ``word_relation`` (Phase 2).

    ``init_models`` is ``create_all`` — additive only, it never renames. On the
    real 149MB DB the old ``entry_links`` table already exists; a plain
    ``create_all`` would leave it orphaned and build an EMPTY ``word_relation``
    alongside it, so every ``Entry.links`` would read blank. This migration runs
    BEFORE ``create_all`` (wired into :func:`init_models`, [RED TEAM F1]) so the
    populated table is carried over by a metadata-only ``ALTER TABLE ... RENAME``
    (fast on both SQLite and Postgres, no row copy).

    Idempotent: it acts only when ``entry_links`` exists AND ``word_relation``
    does not, so re-running (or a fresh DB that never had ``entry_links``) is a
    no-op. ``create_all`` then adds the new ``sense_relation`` table.
    """
    async with engine.begin() as conn:
        has_old = await conn.run_sync(
            lambda sync_conn: engine.dialect.has_table(sync_conn, "entry_links")
        )
        has_new = await conn.run_sync(
            lambda sync_conn: engine.dialect.has_table(sync_conn, "word_relation")
        )
        if has_old and not has_new:
            await conn.exec_driver_sql("ALTER TABLE entry_links RENAME TO word_relation")


async def migrate_generation_epoch(engine: AsyncEngine) -> None:
    """Add the additive generation fence column to existing dictionary DBs.

    Library bootstrap remains the compatibility path for SQLite users. Service
    deployments use the equivalent Alembic revision instead and never call this
    function from API or worker startup.
    """
    async with engine.begin() as conn:
        has_words = await conn.run_sync(
            lambda sync_conn: engine.dialect.has_table(sync_conn, "words")
        )
        if not has_words:
            return

        def column_names(sync_conn):
            return {column["name"] for column in engine.dialect.get_columns(sync_conn, "words")}

        columns = await conn.run_sync(column_names)
        if "generation_epoch" not in columns:
            await conn.exec_driver_sql(
                "ALTER TABLE words ADD COLUMN generation_epoch INTEGER NOT NULL DEFAULT 0"
            )


async def init_models(engine: AsyncEngine) -> None:
    """Create all tables on a fresh generated-dictionary DB (additive only).

    This is largely ``create_all``: it never drops data. The two explicitly
    compatible migrations rename the legacy relation table and add the
    generation-fence column before metadata creation.

    [RED TEAM F1] :func:`migrate_relations` runs FIRST (before ``create_all``) so
    the legacy ``entry_links`` table is renamed to ``word_relation`` in place —
    otherwise ``create_all`` would orphan the populated table and build an empty
    one. ``init()`` is the only bootstrap path, so wiring it here guarantees no
    caller can skip the migration."""
    await migrate_relations(engine)
    await migrate_generation_epoch(engine)
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
