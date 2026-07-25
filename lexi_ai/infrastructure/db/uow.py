"""SQLAlchemy unit of work: one session, one commit boundary, six repositories.

Scope a unit of work to writes that must land together. Two patterns must stay
OUTSIDE it, and both are load-bearing rather than accidental:

* The generation claim commits before any provider call so competing workers can
  see the new epoch. Sharing this session with the publish would make the fence
  invisible until publish time, which defeats it entirely.
* Error recording happens after a failed write already rolled back, so it cannot
  reuse the rolled-back session. :func:`new_session` exists for exactly that.
"""

from types import TracebackType

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from lexi_ai.assets.repository import AssetRepository
from lexi_ai.infrastructure.db.repositories.entry_repo import SqlEntryRepo
from lexi_ai.infrastructure.db.repositories.sense_repo import SqlSenseRepo
from lexi_ai.infrastructure.db.repositories.stats_repo import SqlStatsRepo
from lexi_ai.infrastructure.db.repositories.tag_repo import SqlTagRepo
from lexi_ai.infrastructure.db.repositories.theme_repo import SqlThemeRepo
from lexi_ai.infrastructure.db.repositories.word_repo import SqlWordRepo


class SqlAlchemyUnitOfWork:
    """Bind the aggregate repositories to one session for one transaction.

    Reusing an instance is supported: each ``async with`` opens a fresh session
    and rebinds the repositories to it.
    """

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        assets: AssetRepository | None = None,
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
        self._session = self._session_factory()
        session = self._session
        self.words = SqlWordRepo(session, assets=self._assets)
        self.senses = SqlSenseRepo(session, words=self.words, assets=self._assets)
        self.themes = SqlThemeRepo(session)
        self.tags = SqlTagRepo(session)
        self.entries = SqlEntryRepo(session)
        self.stats = SqlStatsRepo(session)
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
