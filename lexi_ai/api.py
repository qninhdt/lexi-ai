"""Composition root: the object graph behind the public facades.

:class:`Lexicon` owns the things that must exist exactly once per process — the
database engine, the provider registry, the single-flight lock registries — and
hands out application services wired over them. It answers *what is connected to
what*; every use case lives in :mod:`lexi_ai.application`.

The public surface is :class:`lexi_ai.facades.LexiconReader` (free reads) and
:class:`lexi_ai.facades.LexiconEngine` (provider-backed work), obtained from
:meth:`Lexicon.reader` and :meth:`Lexicon.engine`. Services are exposed here as
plain accessors so a facade never reaches into a private.

Services are rebuilt per accessor call, which is deliberate: they are stateless
wiring over a fresh unit of work, so a caller cannot accidentally share a session
or observe another caller's transaction. The state that must be shared — locks,
built providers, cached question engines — lives on this object instead.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from lexi_ai.application.assets import AssetService
from lexi_ai.application.dictionary import DictionaryService
from lexi_ai.application.enrichment import EnrichmentService
from lexi_ai.application.generation import GenerationService
from lexi_ai.application.generation_writer import GenerationWriter
from lexi_ai.application.questions import QuestionService
from lexi_ai.application.search import SearchService
from lexi_ai.application.single_flight import SingleFlight
from lexi_ai.application.tags import TagService
from lexi_ai.application.themes import ThemeService
from lexi_ai.infrastructure.db.repositories.asset_repo import AssetRepository
from lexi_ai.config import Settings, get_settings
from lexi_ai.db import create_engine, create_session_factory, init_models
from lexi_ai.embeddings import Embedder
from lexi_ai.facades import LexiconEngine, LexiconReader
from lexi_ai.generation.generator import Generator
from lexi_ai.generation.schemas import MAX_EXAMPLES_PER_CALL
from lexi_ai.infrastructure.db.uow import SqlAlchemyUnitOfWork
from lexi_ai.infrastructure.providers import ProviderRegistry
from lexi_ai.infrastructure.question_engine_factory import QuestionEngineFactory
from lexi_ai.infrastructure.vectors import build_vector_index
from lexi_ai.references.cambridge import CambridgeSource
from lexi_ai.references.loader import ReferenceLoader
from lexi_ai.references.wordnet import WordNetSource

if TYPE_CHECKING:
    from lexi_ai.domain.ports import VectorIndex
    from lexi_ai.generation.wsd import WsdJudge
    from lexi_ai.read_models import Entry

# Distinguishes "caller said nothing" from "caller said semantic search is off",
# which ``None`` now means.
_UNSET: Any = object()


class Lexicon:
    """The wired object graph. Construct with :meth:`from_settings`."""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        loader: ReferenceLoader,
        generator: Generator,
        engine: AsyncEngine | None = None,
        embedder: Embedder | None = None,
        assets: AssetRepository | None = None,
        wsd_judge: WsdJudge | None = None,
        vectors: VectorIndex | None = _UNSET,
    ):
        self._session_factory = session_factory
        self._loader = loader
        self._writer = GenerationWriter(self._uow)
        self._engine = engine
        self._embedder = embedder or Embedder()
        self._assets = assets
        # Sense vectors live outside the primary database and are eventually
        # consistent: written post-commit, best-effort, reconciled by a backfill.
        # ``None`` is a MEANINGFUL value here — semantic search switched off — so
        # this cannot use ``vectors or build_vector_index()``: that would rebuild
        # from the ambient environment every time a caller passed the disabled
        # index, silently overriding an explicit decision (and ignoring the
        # ``settings`` that produced it).
        self._vectors = build_vector_index() if vectors is _UNSET else vectors
        # Every optional external capability (LLM, WSD judge, translator, TTS,
        # themed generators) is built on first use by the registry, which owns the
        # "is it configured?" branching. Injected collaborators are handed over so
        # there is exactly one place that answers for a provider.
        self._providers = ProviderRegistry(generator=generator, wsd_judge=wsd_judge)
        self._locks = SingleFlight()
        # A DISTINCT registry from _locks so the themed-overlay lock can never form a
        # cycle with the neutral per-key lock. Keyed on (word_id, theme_id): word_id
        # is the canonical resolution of the word's match_key, which sidesteps the
        # display-vs-norm key mismatch the raw key would carry.
        self._theme_locks = SingleFlight()
        # Reader and worker question engines are distinct capability contexts. The
        # factory holds both; the reader never receives provider capabilities.
        self._question_engines = QuestionEngineFactory(
            self._uow,
            session_factory,
            self._embedder,
            self._providers,
            self._vectors,
            self._speak,
            self._read_entry,
        )

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> Lexicon:
        """Build the graph from configuration, owning its own database engine."""
        settings = settings or get_settings()
        engine = create_engine(settings)
        session_factory = create_session_factory(engine)
        loader = ReferenceLoader(CambridgeSource(settings.cambridge_db_path), WordNetSource())
        return cls(
            session_factory,
            loader,
            Generator(settings=settings),
            engine=engine,
            embedder=Embedder(settings=settings),
            assets=AssetRepository(session_factory, settings.asset_cache_dir),
            vectors=build_vector_index(settings),
        )

    # --- lifecycle --------------------------------------------------------

    def reader(self) -> LexiconReader:
        """The provider-free public read facade over this graph."""
        return LexiconReader(self)

    def engine(self) -> LexiconEngine:
        """The provider-enabled public work facade over this graph."""
        return LexiconEngine(self)

    async def init(self) -> None:
        """Create the generated-DB schema (idempotent)."""
        await init_models(self._db_engine())

    async def close(self) -> None:
        """Release the database engine owned by this instance."""
        await self._db_engine().dispose()

    def _db_engine(self) -> AsyncEngine:
        """The engine this graph writes through, injected or taken from the factory."""
        return self._engine or self._session_factory.kw["bind"]

    def _uow(self) -> SqlAlchemyUnitOfWork:
        """A fresh unit of work over this graph's session factory.

        Built per call rather than held, so each caller gets its own session and
        transaction boundary. The asset cache is read from the attribute because it
        may be constructed lazily after this object exists.
        """
        return SqlAlchemyUnitOfWork(self._session_factory, assets=self._assets)

    def _require_assets(self) -> AssetRepository:
        """The asset cache, constructed lazily from settings if not injected."""
        if self._assets is None:
            self._assets = AssetRepository(self._session_factory, get_settings().asset_cache_dir)
        return self._assets

    # --- application services ---------------------------------------------

    def dictionary(self) -> DictionaryService:
        """Free reads: entries, senses, statuses, listings, stats."""
        return DictionaryService(self._uow, self._resolve_theme, self._vectors)

    def lookup(self) -> SearchService:
        """Free search: reference-backed lexical search and semantic ranking."""
        return SearchService(self._uow, self._loader, self._embedder, self._vectors)

    def tags(self) -> TagService:
        """Topic-tag curation: rename, delete, merge."""
        return TagService(self._uow)

    def themes(self) -> ThemeService:
        """Style themes and themed overlays."""
        return self._themes()

    def _themes(self) -> ThemeService:
        return ThemeService(
            self._uow,
            self._providers.themed,
            self._providers.theme_metadata,
            self._read_senses,
            self._read_status,
            MAX_EXAMPLES_PER_CALL,
        )

    # The theme and dictionary services each need one call from the other (a theme
    # read needs sense views; a themed read needs a resolved theme id). Crossing via
    # these late-bound hops keeps both services unaware of each other AND keeps the
    # wiring acyclic — passing bound methods of freshly built services would recurse
    # forever at construction time.
    async def _resolve_theme(self, theme: str | int) -> tuple[int, str]:
        return await self._themes().resolve_or_raise(theme)

    async def _read_senses(self, sense_ids):  # noqa: ANN001, ANN202
        return await self.dictionary().senses(sense_ids)

    async def _read_status(self, word_id: int) -> str | None:
        return await self.dictionary().status(word_id)

    def enrichment(self) -> EnrichmentService:
        """Post-generation work: examples, embeddings, sense-relation resolution."""
        return EnrichmentService(
            self._uow,
            self._embedder,
            self._providers.example_generator,
            self._providers.wsd,
            self._read_senses,
            self._themes().append_examples,
            MAX_EXAMPLES_PER_CALL,
            self._vectors,
        )

    def generation(self) -> GenerationService:
        """Entry generation, over this graph's PROCESS-scoped lock registries.

        The registries are owned here rather than by the service: rebuilt per call
        they would collapse nothing, because each caller would take its own lock.
        """
        themes = self._themes()
        return GenerationService(
            self._uow,
            self._writer,
            self._loader,
            self._providers.generator,
            self.dictionary().entry_by_theme_id,
            self._locks,
            self._theme_locks,
            themes.resolve_or_raise,
            themes.restyle_word,
            self._embed_words,
            self.enrichment().resolve_inbound,
        )

    def assets(self) -> AssetService:
        """Cached translations and speech clips."""
        settings = get_settings()
        return AssetService(
            self._require_assets(),
            self._providers.translator_provider,
            self._providers.tts_provider,
            voice=settings.tts_voice,
            fmt=settings.tts_format,
        )

    async def _embed_words(self, word_ids: list[int]) -> int:
        """Embed the senses of the given words as a post-commit generation hook.

        The ONE place a vector failure is swallowed. The entry is already persisted
        and the LLM call is already paid for; a missing embeddings extra or a device
        error must not turn that into a failed generation. The vectors stay missing
        and `backfill_embeddings()` — which does raise — reconciles them later.
        """
        try:
            return await self.enrichment().embed_missing(word_ids=word_ids)
        except Exception:  # noqa: BLE001 - see the docstring: generation is already paid for
            return 0

    # --- questions --------------------------------------------------------

    def questions(self, *, providers: bool) -> QuestionService:
        """The question service for one capability context.

        The factory caches the engine, so this wrapper is rebuilt per call and
        replacing an engine (as tests do) takes effect immediately.
        """
        return QuestionService(
            self._question_engines.engine(providers=providers),
            self._question_engines.repository(),
            self._read_entry,
        )

    async def _speak(self, source_kind: str, source_id: int, voice: str, fmt: str):  # noqa: ANN202
        """Synthesize one clip, for the question engine's audio port."""
        return await self.assets().speak(source_kind, source_id, voice, fmt)

    async def _read_entry(self, word_id: int) -> Entry:
        """One neutral entry, for the question engine's sense loader."""
        return await self.dictionary().entry(word_id)
