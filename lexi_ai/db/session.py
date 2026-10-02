"""Explicit, per-operation transaction scopes for the generated dictionary."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from lexi_ai.config import database_schema_name


class Database:
    def __init__(self, url: str, *, schema: str | None = None):
        if not url.startswith(("sqlite+aiosqlite://", "postgresql+asyncpg://")):
            raise ValueError("expected an async generated-dictionary database URL")
        schema = database_schema_name(schema)
        if schema is not None and not url.startswith("postgresql+asyncpg://"):
            raise ValueError("named database schema requires PostgreSQL")
        sqlite = url.startswith("sqlite+aiosqlite://")
        if not sqlite:
            schema = schema or "lexi"
        connect_args = (
            {"timeout": 5} if sqlite else {"server_settings": {"search_path": f'"{schema}"'}}
        )
        self.engine: AsyncEngine = create_async_engine(
            url, connect_args=connect_args, pool_pre_ping=not sqlite
        )
        if self.engine.dialect.name == "sqlite":
            database = self.engine.url.database
            file_database = (
                database not in (None, "", ":memory:")
                and not database.startswith("file::memory:")
                and self.engine.url.query.get("mode") != "memory"
            )

            @event.listens_for(self.engine.sync_engine, "connect")
            def _configure_sqlite(connection, _record):
                cursor = connection.cursor()
                try:
                    cursor.execute("PRAGMA foreign_keys=ON")
                    cursor.execute("PRAGMA busy_timeout=5000")
                    if file_database:
                        cursor.execute("PRAGMA journal_mode=WAL")
                        if cursor.fetchone()[0].lower() != "wal":
                            raise RuntimeError("generated SQLite database requires WAL support")
                finally:
                    cursor.close()

        self.sessions = async_sessionmaker(self.engine, expire_on_commit=False)

    @asynccontextmanager
    async def transaction(self, *, immediate: bool = False) -> AsyncIterator[AsyncSession]:
        """New session, commit once on success or roll back all changes on failure."""
        async with self.sessions() as session, session.begin():
            if immediate and self.engine.dialect.name == "sqlite":
                # Reserve the writer before Sense Linking revalidation, not after its reads.
                await session.execute(text("BEGIN IMMEDIATE"))
            yield session

    @asynccontextmanager
    async def read(self):
        """One-statement reads; no explicit DB transaction/commit around the SELECT.

        Use transaction() for atomic writes, locking or transaction-local settings.
        Execution options share the existing pool and do not create another engine.
        """
        async with self.engine.connect() as connection:
            if self.engine.dialect.name == "postgresql":
                connection = await connection.execution_options(isolation_level="AUTOCOMMIT")
            yield connection

    async def create_schema(self, metadata) -> None:
        """Test/bootstrap only: use the baseline migration for production Postgres."""
        async with self.engine.begin() as connection:
            await connection.run_sync(metadata.create_all)

    async def close(self) -> None:
        await self.engine.dispose()


class SessionDatabase:
    """Borrow a host session; serialize SQL within parallel Sense Linking work."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self.engine = session.bind
        self._lock = asyncio.Lock()

    @asynccontextmanager
    async def transaction(self, *, immediate: bool = False) -> AsyncIterator[AsyncSession]:
        async with self._lock:
            yield self.session

    @asynccontextmanager
    async def read(self):
        async with self._lock:
            yield self.session

    async def close(self) -> None:
        pass
