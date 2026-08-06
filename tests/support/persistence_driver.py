"""A TEST-ONLY driver that composes the persistence ports into one object.

Production code has no god object any more: ``Lexicon`` opens a unit of work per
operation and the generation writer owns the claim/publish/error transactions.
Tests, however, mostly care about "put this content in the database, then assert
what happened", and threading a unit of work through several hundred assertions
would obscure what each test is actually checking.

So this driver exists purely as a test fixture. It is deliberately thin: every
method delegates straight to a repository or to the generation writer, and it adds
no behavior of its own. It must never be imported by library code.
"""

from collections.abc import Iterable, Sequence

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from lexi_ai.application.generation_writer import GenerationWriter
from lexi_ai.domain.models import (
    GenerationFence,
    ResolveDecision,
    ResolveOutcome,
    ThemeRecord,
    WordRecord,
)
from lexi_ai.generation.schemas import GeneratedResult
from lexi_ai.infrastructure.db.repositories.asset_repo import AssetRepository
from lexi_ai.infrastructure.db.uow import SqlAlchemyUnitOfWork
from lexi_ai.read_models import Stats


class PersistenceDriver:
    """Drive the aggregate repositories from a test, one transaction per call."""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        assets: AssetRepository | None = None,
    ) -> None:
        self._session_factory = session_factory
        self._assets = assets
        self._writer = GenerationWriter(self.uow)

    def uow(self) -> SqlAlchemyUnitOfWork:
        return SqlAlchemyUnitOfWork(self._session_factory, assets=self._assets)

    # --- generation ---------------------------------------------------------

    async def persist_result(
        self,
        result: GeneratedResult,
        cambridge_word_id: int | None = None,
        cambridge_cefr: dict[str, str] | None = None,
        fence: GenerationFence | None = None,
    ) -> list[WordRecord]:
        return await self._writer.publish(
            result,
            cambridge_word_id=cambridge_word_id,
            cambridge_cefr=cambridge_cefr,
            fence=fence,
        )

    async def claim_generation(self, norm: str) -> GenerationFence:
        return await self._writer.claim(norm)

    # --- words --------------------------------------------------------------

    async def get_done_keys(self) -> set[str]:
        async with self.uow() as uow:
            return await uow.words.done_keys()

    async def delete_word(self, word_id: int) -> bool:
        async with self.uow() as uow:
            deleted = await uow.words.delete(word_id)
            await uow.commit()
            return deleted

    async def seed_phrase_unit(
        self,
        phrase_title: str,
        host_display: str | None = None,
        entry_type: str | None = None,
        is_overlap: bool = False,
    ) -> None:
        async with self.uow() as uow:
            await uow.words.seed_phrase_unit(
                phrase_title=phrase_title,
                host_display=host_display,
                entry_type=entry_type,
                is_overlap=is_overlap,
            )
            await uow.commit()

    # --- tags ---------------------------------------------------------------

    async def all_tags(self):
        async with self.uow() as uow:
            return await uow.tags.names()

    async def count_tags(self):
        async with self.uow() as uow:
            return await uow.tags.usage()

    # --- themes -------------------------------------------------------------

    async def create_theme(self, *args, **kwargs) -> ThemeRecord:
        async with self.uow() as uow:
            theme = await uow.themes.create(*args, **kwargs)
            await uow.commit()
            return theme

    async def list_themes(self) -> list[ThemeRecord]:
        async with self.uow() as uow:
            return await uow.themes.list_all()

    async def resolve_theme(self, key_or_id: str | int) -> tuple[int, str] | None:
        async with self.uow() as uow:
            return await uow.themes.resolve(key_or_id)

    async def persist_themed(self, theme_id: int, result, sense_ids: Sequence[int]) -> None:
        async with self.uow() as uow:
            await uow.themes.persist_themed(theme_id, result, sense_ids)
            await uow.commit()

    async def themed_for_word(self, word_id: int, theme_id: int):
        async with self.uow() as uow:
            return await uow.themes.overlay_for_word(word_id, theme_id)

    # --- senses -------------------------------------------------------------

    async def apply_resolutions(self, decisions: Iterable[ResolveDecision]) -> list[ResolveOutcome]:
        async with self.uow() as uow:
            outcomes = await uow.senses.apply_resolutions(decisions)
            await uow.commit()
            return outcomes

    async def stats(self) -> Stats:
        async with self.uow() as uow:
            return await uow.stats.snapshot()
