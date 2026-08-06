"""Assembly of the two question engines a process may hold.

A reader and a worker are distinct capability contexts, not one engine with a flag:
the reader is built without the language model, the rubric judge, or the speech port,
so a reader process cannot spend a provider call even by mistake. Both are cached per
context, because building one registers the entry-point question types.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from lexi_ai.application.question_ports import AssetTtsPort
from lexi_ai.config import get_settings

if TYPE_CHECKING:
    from lexi_ai.domain.ports import UnitOfWork, VectorIndex
    from lexi_ai.embeddings import Embedder
    from lexi_ai.infrastructure.db.repositories.question_repo import QuestionRepository
    from lexi_ai.infrastructure.providers import ProviderRegistry
    from lexi_ai.questions.base import TtsPort
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
        speak: Callable[[str, int, str, str], Awaitable[object]],
        load_entry: Callable[[int], Awaitable[Entry]],
    ) -> None:
        self._uow = uow_factory
        self._session_factory = session_factory
        self._embedder = embedder
        self._providers = providers
        self._vectors = vectors
        self._speak = speak
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
        from lexi_ai.application.question_ports import SenseEntryLoader
        from lexi_ai.questions.base import load_entry_point_types
        from lexi_ai.questions.distractors import DistractorProvider
        from lexi_ai.questions.engine import QuestionEngine

        # Third-party question types are opt-in: register only those the Settings
        # allowlist names (built-ins are already loaded by direct import).
        load_entry_point_types(set(get_settings().question_type_allowlist) or None)

        return QuestionEngine(
            self.repository(),
            DistractorProvider(self._uow, self._embedder, self._vectors),
            llm=self._providers.questions_llm() if providers else None,
            judge_llm=self._providers.judge_llm() if providers else None,
            tts=self._tts_port() if providers else None,
            sense_loader=SenseEntryLoader(self._uow, self._load_entry),
        )

    def _tts_port(self) -> TtsPort | None:
        """The audio port for the listening/spelling formats, or ``None`` when no TTS
        is configured (so those formats degrade to ``[]`` like the llm-backed ones)."""
        if not self._providers.tts_configured():
            return None
        return AssetTtsPort(self._speak, self._voice_and_format)

    @staticmethod
    def _voice_and_format() -> tuple[str, str]:
        settings = get_settings()
        return settings.tts_voice, settings.tts_format
