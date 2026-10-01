"""How a session is created, and how a transaction is scoped.

Two halves of one concept, which is why they share a file: the engine/session
factory that decides *how* to talk to the database, and the unit of work that
decides *when* a write becomes visible.

SQLite does not enforce foreign keys (or ``ON DELETE CASCADE``) unless
``PRAGMA foreign_keys=ON`` is set on each connection, so that is wired on
connect for SQLite URLs. Postgres enforces FKs natively and needs no pragma.

Scope a unit of work to writes that must land together. Two patterns must stay
OUTSIDE it, and both are load-bearing rather than accidental:

* The generation claim commits before any provider call so competing workers can
  see the new epoch. Sharing this session with the publish would make the fence
  invisible until publish time, which defeats it entirely.
* Error recording happens after a failed write already rolled back, so it cannot
  reuse the rolled-back session. :meth:`SqlAlchemyUnitOfWork.new_session` exists
  for exactly that.
"""
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from types import TracebackType
from typing import TYPE_CHECKING

from sqlalchemy import event
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from lexi_ai.config import Settings, get_settings
from lexi_ai.db.schema import Base

if TYPE_CHECKING:
    from lexi_ai.db.queries.asset import AssetRepository


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



class SqlAlchemyUnitOfWork:
    """Bind the aggregate repositories to one session for one transaction.

    Reusing an instance is supported: each ``async with`` opens a fresh session
    and rebinds the repositories to it.
    """

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        assets: "AssetRepository | None" = None,
    ) -> None:
        self._session_factory = session_factory
        self._assets = assets
        self._session: AsyncSession | None = None

    @property
    def session(self) -> AsyncSession:
        """The active session. Only valid inside the context manager."""
        if self._session is None:
            raise RuntimeError("unit of work is not active; use 'async with uow:'")
        return self._session

    def new_session(self) -> AsyncSession:
        """An INDEPENDENT session, for writes that must survive a rollback."""
        return self._session_factory()

    async def __aenter__(self) -> "SqlAlchemyUnitOfWork":
        # Imported here, not at module scope: the query modules import
        # `session_scope` from this one, so a top-level import would close the
        # cycle. The unit of work is built once per transaction, so the cost is
        # a dict lookup after the first call.
        from lexi_ai.db.queries.entry import EntryQueries
        from lexi_ai.db.queries.sense import SenseQueries
        from lexi_ai.db.queries.stats import StatsQueries
        from lexi_ai.db.queries.tag import TagQueries
        from lexi_ai.db.queries.theme import ThemeQueries
        from lexi_ai.db.queries.word import WordQueries

        self._session = self._session_factory()
        session = self._session
        self.words = WordQueries(session, assets=self._assets)
        self.senses = SenseQueries(session, words=self.words, assets=self._assets)
        self.themes = ThemeQueries(session)
        self.tags = TagQueries(session)
        self.entries = EntryQueries(session)
        self.stats = StatsQueries(session)
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        session = self._session
        if session is None:
            return
        try:
            if exc_type is not None:
                await session.rollback()
        finally:
            await session.close()
            self._session = None

    async def commit(self) -> None:
        await self.session.commit()

    async def rollback(self) -> None:
        await self.session.rollback()

    async def flush(self) -> None:
        await self.session.flush()
