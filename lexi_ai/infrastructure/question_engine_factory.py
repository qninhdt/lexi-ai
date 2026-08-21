"""Assembly of the two question engines a process may hold."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

if TYPE_CHECKING:
    from lexi_ai.domain.ports import UnitOfWork, VectorIndex
    from lexi_ai.embeddings import Embedder
    from lexi_ai.infrastructure.db.repositories.question_repo import QuestionRepository
    from lexi_ai.infrastructure.providers import ProviderRegistry
    from lexi_ai.questions.engine import QuestionEngine
    from lexi_ai.read_models import Entry


class QuestionEngineFactory:
    """Builds, and holds, the question engine for each capability context."""

    def __init__(
        self,
        uow_factory: Callable[[], UnitOfWork],
        session_factory: async_sessionmaker[AsyncSession],
        embedder: Embedder,
        providers: ProviderRegistry,
        vectors: VectorIndex | None,
        load_entry: Callable[[int], Awaitable[Entry]],
    ) -> None:
        self._uow = uow_factory
        self._session_factory = session_factory
        self._embedder = embedder
        self._providers = providers
        self._vectors = vectors
        self._load_entry = load_entry
        self._repo: QuestionRepository | None = None
        # Public so a test can install a fake engine for one context.
        self.reader: QuestionEngine | None = None
        self.worker: QuestionEngine | None = None

    def engine(self, *, providers: bool) -> QuestionEngine:
        """The engine for one context, built on first request and then reused."""
        if providers:
            if self.worker is None:
                self.worker = self._build(providers=True)
            return self.worker
        if self.reader is None:
            self.reader = self._build(providers=False)
        return self.reader

    def repository(self) -> QuestionRepository:
        """The question store, shared by both contexts."""
        if self._repo is None:
            from lexi_ai.infrastructure.db.repositories.question_repo import QuestionRepository

            self._repo = QuestionRepository(self._session_factory)
        return self._repo

    def _build(self, *, providers: bool) -> QuestionEngine:
        from lexi_ai.questions.distractors import DistractorProvider
        from lexi_ai.questions.engine import QuestionEngine

        return QuestionEngine(
            self.repository(),
            DistractorProvider(self._uow, self._embedder, self._vectors),
            llm=self._providers.questions_llm() if providers else None,
            judge_llm=self._providers.judge_llm() if providers else None,
            sense_loader=SenseEntryLoader(self._uow, self._load_entry),
        )


class SenseEntryLoader:
    """Adapter that resolves a sense to its owning entry, for exposure cards.

    Lives beside the engine assembly because the question engine is its only
    consumer. The capability plugins see stays the narrow Protocol in
    ``questions.base``, which this satisfies structurally; the unknown-sense miss
    is translated to ``None`` by the repository, so no driver exception crosses.
    """

    def __init__(
        self,
        uow_factory: Callable[[], UnitOfWork],
        read_entry: Callable[[int], Awaitable[Entry]],
    ) -> None:
        self._uow = uow_factory
        self._read_entry = read_entry

    async def load_entry(self, sense_id: int) -> Entry | None:
        """The sense's owning entry, or ``None`` when the sense is gone."""
        async with self._uow() as uow:
            word_id = await uow.senses.word_id_for(sense_id)
        if word_id is None:
            return None
        return await self._read_entry(word_id)
