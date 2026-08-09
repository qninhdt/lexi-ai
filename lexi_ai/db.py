"""Async engine + session factory for SQLite and Postgres.

SQLite does not enforce foreign keys (or ``ON DELETE CASCADE``) unless
``PRAGMA foreign_keys=ON`` is set on each connection, so we wire that on connect
for SQLite URLs. Postgres enforces FKs natively and needs no pragma.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy import event
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from lexi_ai.config import Settings, get_settings
from lexi_ai.infrastructure.db.models import Base


def _enable_sqlite_fk(engine: AsyncEngine) -> None:
    """Make SQLite transactional (no-op on other backends).

    SQLite needs foreign-key enforcement enabled per connection. Its implicit
    transaction also breaks SAVEPOINT rollback, so SQLAlchemy owns BEGIN explicitly.
    """
    if not engine.url.get_backend_name().startswith("sqlite"):
        return

    @event.listens_for(engine.sync_engine, "connect")
    def _set_pragma(dbapi_conn, _record):  # noqa: ANN001
        dbapi_conn.isolation_level = None
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    @event.listens_for(engine.sync_engine, "begin")
    def _emit_begin(conn):  # noqa: ANN001
        conn.exec_driver_sql("BEGIN")


def create_engine(settings: Settings | None = None) -> AsyncEngine:
    settings = settings or get_settings()
    engine_options = {}
    if settings.db_url.startswith("postgresql"):
        engine_options = {"execution_options": {"schema_translate_map": {None: settings.db_schema}}}
    engine = create_async_engine(settings.db_url, **engine_options)
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
