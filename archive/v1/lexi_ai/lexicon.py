"""The public surface: one class, every use case.

:class:`Lexicon` owns the things that must exist exactly once per process — the
database engine, the provider registry, the single-flight lock registries — and
builds each service once over them. Every method below is a use case: reads that
never touch a provider, and work that spends a model call or changes a row.

Two invariants hold across the whole surface:

* Services are built once, in ``__init__``. They hold a unit-of-work *factory*,
  not a session, so each operation still opens its own session and transaction
  boundary — sharing the service shares nothing mutable.
* The unit of work is what composes; the dictionary and theme services read the
  same tables through it rather than calling each other, so neither can close a
  cycle at construction time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from lexi_ai.assets import AssetService
from lexi_ai.config import Settings, get_settings
from lexi_ai.db.queries.asset import AssetRepository
from lexi_ai.db.session import (
    SqlAlchemyUnitOfWork,
    create_engine,
    create_session_factory,
    init_models,
)
from lexi_ai.dictionary.enrich import EnrichmentService
from lexi_ai.dictionary.generate import GenerationService
from lexi_ai.dictionary.read import DictionaryService
from lexi_ai.dictionary.search import SearchService
from lexi_ai.dictionary.single_flight import SingleFlight
from lexi_ai.dictionary.tags import TagService
from lexi_ai.dictionary.writer import GenerationWriter
from lexi_ai.providers.generate import Generator
from lexi_ai.providers.references.bundle import ReferenceLoader
from lexi_ai.providers.references.cambridge import CambridgeSource
from lexi_ai.providers.references.wordnet import WordNetSource
from lexi_ai.providers.registry import ProviderRegistry
from lexi_ai.questions.factory import QuestionEngineFactory
from lexi_ai.questions.service import QuestionService
from lexi_ai.schemas import MAX_EXAMPLES_PER_CALL
from lexi_ai.themes import ThemeService

if TYPE_CHECKING:
    from collections.abc import Sequence

    from lexi_ai.models import (
        Asset,
        BatchResult,
        Entry,
        SearchResult,
        SenseView,
        Stats,
        TagCount,
        Theme,
    )
    from lexi_ai.providers.wsd import WsdJudge
    from lexi_ai.questions.types import (
        AnswerSubmission,
        Evaluation,
        PrepareDemand,
        PrepareReport,
        PresentedQuestion,
        QuestionTypeInfo,
    )


class Lexicon:
    """The wired object graph, and the public API over it."""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        loader: ReferenceLoader,
        generator: Generator,
        engine: AsyncEngine | None = None,
        assets: AssetRepository | None = None,
        wsd_judge: WsdJudge | None = None,
        settings: Settings | None = None,
    ):
        self._settings = settings or get_settings()
        self._session_factory = session_factory
        self._loader = loader
        self._engine = engine
        self._assets_repo = assets
        # Every optional external capability (LLM, WSD judge, translator, TTS,
        # themed generators) is built on first use by the registry, which owns the
        # "is it configured?" branching.
        self._providers = ProviderRegistry(
            settings=self._settings,
            generator=generator,
            wsd_judge=wsd_judge,
        )
        # Two DISTINCT lock registries, so the themed-overlay lock can never form a
        # cycle with the neutral per-key lock. The themed one is keyed on
        # (word_id, theme_id): word_id is the canonical resolution of the word's
        # match_key, which sidesteps the display-vs-norm key mismatch the raw key
        # would carry.
        self._locks = SingleFlight()
        self._theme_locks = SingleFlight()
        self._question_engines = QuestionEngineFactory(
            self._uow,
            session_factory,
            self._providers,
            self._read_entry,
        )

        # Services, built once over the unit-of-work FACTORY, so each operation
        # still opens its own session.
        self._dictionary = DictionaryService(self._uow)
        self._lookup = SearchService(self._uow, self._loader)
        self._tags = TagService(self._uow)
        self._themes = ThemeService(
            self._uow,
            self._providers.themed,
            self._providers.theme_metadata,
            MAX_EXAMPLES_PER_CALL,
        )
        self._assets_service = AssetService(
            self._require_assets(),
            self._providers.translator_provider,
            self._providers.tts_provider,
            voice=self._settings.tts_voice,
            fmt=self._settings.tts_format,
        )
        self._writer = GenerationWriter(self._uow)
        self._enrichment = EnrichmentService(
            self._uow,
            self._providers.example_generator,
            self._providers.wsd,
            self._themes.append_examples,
            MAX_EXAMPLES_PER_CALL,
        )
        self._generation = GenerationService(
            self._uow,
            self._writer,
            self._loader,
            self._providers.generator,
            self._dictionary.entry_by_theme_id,
            self._locks,
            self._theme_locks,
            self._themes.resolve_or_raise,
            self._themes.restyle_word,
            self._enrichment.resolve_inbound,
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
            assets=AssetRepository(session_factory, settings.asset_cache_dir),
            settings=settings,
        )

    # --- lifecycle --------------------------------------------------------

    async def init(self) -> None:
        """Create the generated-DB schema (idempotent)."""
        await init_models(self._db_engine())

    async def close(self) -> None:
        """Release the database engine owned by this instance."""
        await self._db_engine().dispose()

    def _db_engine(self) -> AsyncEngine:
        return self._engine or self._session_factory.kw["bind"]

    def _uow(self) -> SqlAlchemyUnitOfWork:
        return SqlAlchemyUnitOfWork(self._session_factory, assets=self._assets_repo)

    def _require_assets(self) -> AssetRepository:
        if self._assets_repo is None:
            self._assets_repo = AssetRepository(
                self._session_factory, self._settings.asset_cache_dir
            )
        return self._assets_repo

    def _questions(self, *, providers: bool) -> QuestionService:
        """The question service for one capability context.

        The factory caches the engine, so this wrapper is rebuilt per call and
        replacing an engine (as tests do) takes effect immediately.
        """
        return QuestionService(
            self._question_engines.engine(providers=providers),
            self._question_engines.repository(),
            self._read_entry,
        )

    async def _read_entry(self, word_id: int) -> Entry:
        return await self._dictionary.entry(word_id)

    # --- search -----------------------------------------------------------

    async def search(self, query: str) -> list[SearchResult]:
        """Search the dictionary for a raw string. Never generates.

        Returns one ranked list (best first) mixing two kinds of hit:

        * **generated** — a word already in the dictionary (``lexi_word_id`` set);
          pass the id to :meth:`get_entry`.
        * **suggestion** — a reference word that *can* be generated
          (``cambridge_id`` set); pass the result to :meth:`generate`.

        A reference word whose ``match_key`` is already generated is folded into the
        generated hit, so nothing is offered for regeneration by mistake.
        """
        return await self._lookup.search(query)

    # --- entries ----------------------------------------------------------

    async def get_entry(self, word_id: int, theme: str | int | None = None) -> Entry | None:
        """Load a generated entry by its dictionary id, or ``None`` if there is
        none. Never generates.

        ``theme`` (key or id) overlays the themed definition and examples where a
        themed row exists, falling back to neutral per sense. An unknown ``theme``
        raises ``ValueError`` — returning neutral silently would hide a caller bug.
        That is deliberately not the same as an unknown ``word_id``: a lookup miss
        is ordinary and answers ``None``, like every sibling read here.
        """
        return await self._dictionary.entry(word_id, theme)

    async def get_many(
        self, word_ids: list[int], theme: str | int | None = None
    ) -> list[BatchResult]:
        """Batch :meth:`get_entry`, one result per input id in order. A missing or
        invalid id is reported as a failed item rather than aborting the batch."""
        return await self._dictionary.entries(word_ids, theme)

    async def get_senses(self, sense_ids: list[int]) -> list[SenseView]:
        """Batch-resolve senses by id, preserving input order. Ids with no row are
        skipped, so a caller tolerating missing senses needs no error handling."""
        return await self._dictionary.senses(sense_ids)

    async def word_id_for(self, sense_id: int) -> int | None:
        """A sense's owning word id, without loading its lexical content."""
        return await self._dictionary.word_id_for(sense_id)

    async def get_status(self, word_id: int) -> str | None:
        """Status of a word (``done`` | ``pending`` | ``error``), or ``None`` when
        no such id exists."""
        return await self._dictionary.status(word_id)

    async def get_status_many(self, word_ids: list[int]) -> list[BatchResult]:
        """Batch :meth:`get_status`, in order. ``value`` is ``None`` for an unknown
        id — that is a valid answer, not a failure."""
        return await self._dictionary.statuses(word_ids)

    async def list_entries(
        self, *, status: str = "done", limit: int | None = None, offset: int = 0
    ) -> list[SearchResult]:
        """Paginated browse of the whole dictionary, norm-sorted. Lightweight rows —
        pass a hit's ``lexi_word_id`` to :meth:`get_entry` for the full entry."""
        return await self._dictionary.list_entries(
            status=status, limit=limit, offset=offset
        )

    async def list_entries_by_tag(
        self, tag: str, *, limit: int | None = None
    ) -> list[SearchResult]:
        """Generated words carrying ``tag``, as generated-hit results.

        The query is resolved through the same key function the write path uses, so
        ``"Business"``, ``"business"``, and ``"cars"`` all hit the right tag.
        """
        return await self._dictionary.list_entries_by_tag(tag, limit=limit)

    async def list_tags(self) -> list[TagCount]:
        """Every topic tag with its live member count over ``done`` words, sorted
        count-desc then name."""
        return await self._dictionary.list_tags()

    async def stats(self) -> Stats:
        """Read-only dictionary counts in one grouped snapshot."""
        return await self._dictionary.stats()

    # --- themes -----------------------------------------------------------

    async def list_themes(self) -> list[Theme]:
        """Every style theme, name-sorted."""
        return await self._themes.list_all()

    async def get_theme(self, key: str) -> Theme | None:
        """A style theme by key (a raw display name is normalized as on the write
        path), or ``None`` if unknown."""
        return await self._themes.get(key)

    # --- cached assets ----------------------------------------------------

    async def get_asset(self, asset_id: int) -> Asset | None:
        return await self._assets_service.get(asset_id)

    async def list_assets(
        self, *, kind: str | None = None, limit: int | None = None, offset: int = 0
    ) -> list[Asset]:
        """Cached assets, oldest first, optionally filtered by kind."""
        return await self._assets_service.list(kind=kind, limit=limit, offset=offset)

    async def source_hash(self, source_kind: str, source_id: int) -> str | None:
        """The current content fingerprint of a translatable source, else ``None``.

        A worker fences a delayed translation job on this before calling a provider.
        """
        return await self._assets_service.source_hash(source_kind, source_id)

    # --- generation -------------------------------------------------------

    async def generate(
        self,
        source: SearchResult | str,
        *,
        force: bool = False,
        theme: str | int | None = None,
        structured_method: str | None = None,
    ) -> Entry:
        """Generate (or return) the entry for a search result or a custom string.

        * ``SearchResult`` — anchored to its Cambridge reference. If already
          generated, returns the existing entry with no model call unless ``force``.
        * ``str`` — a custom word Cambridge lacks; anchored to WordNet only.

        A suggestion whose word already exists converges on that entry instead of
        duplicating it. With ``force=True`` the entry is regenerated in place.

        A ``theme`` (name or key) additionally restyles the resolved entry in that
        voice if it is not already styled.
        """
        return await self._generation.generate(
            source, force=force, theme=theme, structured_method=structured_method
        )

    async def generate_many(
        self,
        sources: list[SearchResult | str],
        *,
        force: bool = False,
        theme: str | int | None = None,
        concurrency: int = 5,
    ) -> list[BatchResult]:
        """Batch :meth:`generate`, in order, up to ``concurrency`` in flight.

        Two inputs resolving to one word still generate exactly once — the
        per-``match_key`` lock and DB double-check are reused here, not
        reimplemented.
        """
        return await self._generation.generate_many(
            sources, force=force, theme=theme, concurrency=concurrency
        )

    async def generate_fenced(
        self, source: SearchResult | str, *, structured_method: str | None = None
    ) -> Entry:
        """Generate once under a database fence, for independently deployed workers.

        Deliberately has no ``force``: a remote caller must not be able to use a
        delayed job to replace an entry that a newer claim owns.
        """
        return await self._generation.generate_fenced(
            source, structured_method=structured_method
        )

    # --- enrichment -------------------------------------------------------

    async def add_examples(
        self, sense_id: int, n: int = 3, theme: str | int | None = None
    ) -> SenseView:
        """Append up to ``n`` fresh examples to ONE sense, returning it updated.

        This is the one clean generation gap: an example is an open-ended
        illustration of a sense, so producing more never fabricates a linguistic
        fact. Existing examples are never deleted or overwritten — ``example_order``
        continues from the current max, and they are fed back to the model for soft
        de-duplication. ``n`` is a best-effort maximum; ``n <= 0`` is a no-op.

        ``theme`` augments the sense's themed overlay instead of its neutral
        examples; that overlay must already exist. An unknown sense, missing
        overlay, or absent LLM raises ``ValueError``.
        """
        return await self._enrichment.add_examples(sense_id, n, theme)

    async def resolve_relations(self, batch_size: int = 20) -> list[BatchResult]:
        """Reconcile one batch of pending sense-relation edges (manual backfill)."""
        return await self._enrichment.resolve_relations(batch_size)

    # --- curation ---------------------------------------------------------

    async def delete_entry(self, word_id: int) -> bool:
        """Delete a word and all its content; returns whether a row was removed.
        Senses, aliases, links, tags, and questions cascade at the FK level."""
        return await self._dictionary.delete_entry(word_id)

    async def rename_tag(
        self, tag: str, *, name: str | None = None, title: str | None = None
    ) -> bool:
        """Update a topic tag's display fields. The dedup key is immutable, so this
        never merges or re-keys (see :meth:`merge_tags`). Returns whether it existed."""
        return await self._tags.rename(tag, name=name, title=title)

    async def delete_tag(self, tag: str) -> bool:
        """Delete a topic tag; returns whether one was found. Tagged words are
        untouched — they simply lose this one topic."""
        return await self._tags.delete(tag)

    async def merge_tags(self, sources: list[str], into: str) -> int:
        """Fold ``sources`` into ``into``, then delete the sources. ``into`` must
        already exist. Returns the number of word-tag associations re-pointed."""
        return await self._tags.merge(sources, into)

    # --- themes -----------------------------------------------------------

    async def create_theme(
        self,
        name: str,
        style_prompt: str,
        description: str | None = None,
        tone: str | None = None,
    ) -> Theme:
        """Create (or resolve and update) a style theme by its normalized key.

        With ``description`` and ``tone`` supplied the theme is registered as given
        and no model is called; with either missing, the LLM expands the name and
        style prompt into a full profile first.
        """
        return await self._themes.create(name, style_prompt, description, tone)

    async def update_theme(
        self,
        key: str,
        *,
        name: str | None = None,
        style_prompt: str | None = None,
        description: str | None = None,
        tone: str | None = None,
    ) -> Theme:
        """Partially update an EXISTING theme; unset arguments are left unchanged.

        The key is immutable, so renaming ``name`` never re-keys the theme. Raises
        ``ValueError`` for an unknown key — unlike :meth:`create_theme`, this never
        creates.
        """
        return await self._themes.update(
            key, name=name, style_prompt=style_prompt, description=description, tone=tone
        )

    async def delete_theme(self, key: str) -> bool:
        """Delete a style theme by key; returns whether one was removed. Its themed
        senses and examples cascade; neutral entries are untouched."""
        return await self._themes.delete(key)

    # --- translation and speech -------------------------------------------

    async def translate_field(self, source_kind: str, source_id: int, lang: str) -> str:
        """Translate a source into ``lang``, cache-first over the reference store."""
        return await self._assets_service.translate(source_kind, source_id, lang)

    async def translate_sense(self, sense_id: int, lang: str) -> str:
        """Translate a sense's definition."""
        return await self.translate_field("sense_def", sense_id, lang)

    async def translate_many(
        self, refs: list[tuple[str, int]], lang: str, *, concurrency: int = 5
    ) -> list[BatchResult]:
        """Batch :meth:`translate_field`, order-aligned, cache-first per item."""
        return await self._assets_service.translate_many(refs, lang, concurrency=concurrency)

    async def tts_field(
        self, source_kind: str, source_id: int, voice: str | None = None, fmt: str | None = None
    ) -> Asset:
        """Synthesize speech for a source, cache-first over the reference store."""
        return await self._assets_service.speak(source_kind, source_id, voice, fmt)

    async def tts_sense(
        self, sense_id: int, voice: str | None = None, fmt: str | None = None
    ) -> Asset:
        """Synthesize a sense's definition."""
        return await self.tts_field("sense_def", sense_id, voice, fmt)

    async def tts_many(
        self,
        refs: list[tuple[str, int]],
        voice: str | None = None,
        fmt: str | None = None,
        *,
        concurrency: int = 5,
    ) -> list[BatchResult]:
        """Batch :meth:`tts_field`, order-aligned; one failure never aborts the rest."""
        return await self._assets_service.speak_many(refs, voice, fmt, concurrency=concurrency)

    async def delete_asset(self, asset_id: int) -> bool:
        """Delete one cached asset and its backing file."""
        return await self._assets_service.delete(asset_id)

    async def purge_assets(self, *, kind: str | None = None) -> int:
        """Delete every cached asset, unlinking backing files."""
        return await self._assets_service.purge(kind=kind)

    async def sweep_asset_orphans(self, *, min_age_seconds: float = 3600.0) -> int:
        """Remove old binary cache files that have no committed asset row."""
        return await self._assets_service.sweep_orphans(min_age_seconds=min_age_seconds)

    # --- questions --------------------------------------------------------

    def question_types(self) -> list[QuestionTypeInfo]:
        """The registered question formats, including provider-backed ones."""
        return self._questions(providers=True).question_types()

    async def prepare_questions(
        self, word_id: int, demands: Sequence[PrepareDemand]
    ) -> PrepareReport:
        """Materialize the demanded questions for a word, using every provider."""
        return await self._questions(providers=True).prepare(word_id, list(demands))

    async def get_question(self, question_id: int) -> PresentedQuestion | None:
        """One persisted question in presentation form, or ``None``."""
        return await self._questions(providers=True).get(question_id)

    async def list_questions_for_sense(
        self, sense_id: int, type_id: str | None = None
    ) -> list[PresentedQuestion]:
        """Every persisted question for a sense, optionally one format only."""
        return await self._questions(providers=True).list_for_sense(sense_id, type_id)

    async def retrieve_question(
        self,
        sense_id: int,
        difficulty_level: int,
        excluded_ids: frozenset[int],
        type_id: str,
    ) -> PresentedQuestion | None:
        """Pick one unseen question for a sense at a difficulty, or ``None``."""
        return await self._questions(providers=True).retrieve(
            sense_id, difficulty_level, excluded_ids, type_id
        )

    async def retrieve_exposure(self, sense_id: int) -> PresentedQuestion:
        """The exposure card for a sense."""
        return await self._questions(providers=True).retrieve_exposure(sense_id)

    async def evaluate_answer(
        self, question_id: int, submission: AnswerSubmission
    ) -> Evaluation | None:
        """Grade a submission authoritatively, with the rubric judge available."""
        return await self._questions(providers=True).evaluate(question_id, submission)
