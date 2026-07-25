"""Lazy lookup API (Phase 6): the public ``Lexicon.get_entry`` surface.

Flow: normalize input -> resolve by match_key against words AND aliases ->
branch on 0 / 1-done / 1-pending / N. Misses run the full lazy pipeline
(reference -> generate -> persist) exactly once per key, guarded by a per-key
asyncio lock plus a DB double-check (library, single-process — decision #18).

``display`` is always ``render(norm)``; no display column is ever read.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy.exc import NoResultFound
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from lexi_ai.application.assets import AssetService
from lexi_ai.application.batching import gather_batch
from lexi_ai.application.dictionary import DictionaryService
from lexi_ai.application.enrichment import EnrichmentService
from lexi_ai.application.generation import GenerationService
from lexi_ai.application.generation_writer import GenerationWriter
from lexi_ai.application.questions import QuestionService
from lexi_ai.application.search import SearchService
from lexi_ai.application.single_flight import SingleFlight
from lexi_ai.application.tags import TagService
from lexi_ai.application.themes import ThemeService
from lexi_ai.assets.repository import AssetRepository
from lexi_ai.config import Settings, get_settings
from lexi_ai.db import create_engine, create_session_factory, init_models
from lexi_ai.embeddings import Embedder
from lexi_ai.generation.generator import Generator
from lexi_ai.generation.schemas import ExampleBatch
from lexi_ai.infrastructure.db.uow import SqlAlchemyUnitOfWork
from lexi_ai.read_models import (
    Asset,
    BatchResult,
    Entry,
    SearchResult,
    SemanticHit,
    SenseView,
    Stats,
    TagCount,
    Theme,
)
from lexi_ai.references.cambridge import CambridgeSource
from lexi_ai.references.loader import ReferenceLoader
from lexi_ai.references.wordnet import WordNetSource

if TYPE_CHECKING:
    from lexi_ai.contracts.questions import (
        AnswerSubmission,
        Evaluation,
        PrepareDemand,
        PresentedQuestion,
        QuestionTypeInfo,
    )
    from lexi_ai.generation.wsd import WsdJudge
    from lexi_ai.llm import StructuredLLM
    from lexi_ai.questions.base import PrepareReport, TtsPort
    from lexi_ai.questions.engine import QuestionEngine

# Upper bound for a single add_examples call, taken from ExampleBatch's own
# max_length so the two never drift: prompting for more than the schema accepts
# would guarantee a validation failure and burn the structured-output retries.
_MAX_EXAMPLES_PER_CALL = ExampleBatch.model_fields["examples"].metadata[0].max_length


def _to_internal_demands(demands: list) -> list:
    """Map public ``PrepareDemand`` inputs (string sense id) to the internal
    ``QuestionDemand`` (resolved int sense id) the engine consumes."""
    from lexi_ai.questions.base import QuestionDemand

    return [
        QuestionDemand(
            sense_id=int(demand.sense_id),
            difficulty_level=demand.difficulty_level,
            expected_count=demand.expected_count,
        )
        for demand in demands
    ]


class Lexicon:
    """Lazy-generation dictionary. Construct with :meth:`from_settings`."""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        loader: ReferenceLoader,
        generator: Generator,
        engine: AsyncEngine | None = None,
        embedder: Embedder | None = None,
        assets: AssetRepository | None = None,
        wsd_judge: WsdJudge | None = None,
    ):
        self._session_factory = session_factory
        self._loader = loader
        self._generator = generator
        self._writer = GenerationWriter(self._uow)
        self._engine = engine
        self._embedder = embedder or Embedder()
        self._assets = assets
        # WSD judge (sense-relation reconciliation). Injectable for hermetic tests;
        # lazily built from settings otherwise. ``None`` sentinel = not-yet-built,
        # resolved via the ``_wsd`` property.
        self._wsd_judge = wsd_judge
        self._wsd_built = wsd_judge is not None
        # Providers built on first use. Explicit fields rather than getattr
        # sentinels so the state a caller can inject is visible on the class.
        self._translator_impl = None
        self._tts_impl = None
        self._themed_gen = None
        self._theme_meta_gen = None
        self._locks = SingleFlight()
        # Single-flight lock for the THEMED overlay step (2.6), keyed on
        # (word_id, theme_id) — word_id is the canonical resolution of the word's
        # match_key (using it, not the raw display key, sidesteps the 2.1
        # display-vs-norm key mismatch). A DISTINCT map from _locks so the overlay
        # lock can never form a cycle with the neutral per-key lock (the neutral
        # lock is fully released before the overlay block runs, never nested).
        self._theme_locks = SingleFlight()
        # Reader and worker question engines are distinct capability contexts.
        # The reader never receives provider capabilities, even when configured.
        self._question_repo = None
        self._reader_questions: QuestionEngine | None = None
        self._worker_questions: QuestionEngine | None = None

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> Lexicon:
        settings = settings or get_settings()
        engine = create_engine(settings)
        session_factory = create_session_factory(engine)
        loader = ReferenceLoader(CambridgeSource(settings.cambridge_db_path), WordNetSource())
        generator = Generator(settings=settings)
        embedder = Embedder(settings=settings)
        assets = AssetRepository(session_factory, settings.asset_cache_dir)
        return cls(
            session_factory,
            loader,
            generator,
            engine=engine,
            embedder=embedder,
            assets=assets,
        )

    def _uow(self) -> SqlAlchemyUnitOfWork:
        """A fresh unit of work over this dictionary's session factory.

        Built per call rather than held, so each caller gets its own session and
        transaction boundary. The asset cache is read from the attribute because it
        may be constructed lazily after this Lexicon exists.
        """
        return SqlAlchemyUnitOfWork(self._session_factory, assets=self._assets)

    def reader(self):
        """Return the provider-free public read facade for this dictionary."""
        from lexi_ai.facades import LexiconReader

        return LexiconReader(self)

    def engine(self):
        """Return the provider-enabled public work facade for this dictionary."""
        from lexi_ai.facades import LexiconEngine

        return LexiconEngine(self)

    async def init(self) -> None:
        """Create the generated-DB schema (idempotent)."""
        engine = self._engine or self._session_factory.kw["bind"]
        await init_models(engine)

    async def close(self) -> None:
        """Release the database engine owned by this Lexicon instance."""
        engine = self._engine or self._session_factory.kw["bind"]
        await engine.dispose()

    @property
    def reader_questions(self) -> QuestionEngine:
        """Provider-free question context used by API reader processes."""
        if self._reader_questions is None:
            self._reader_questions = self._build_question_engine(providers=False)
        return self._reader_questions

    @property
    def worker_questions(self) -> QuestionEngine:
        """Provider-enabled question context used by worker processes."""
        if self._worker_questions is None:
            self._worker_questions = self._build_question_engine(providers=True)
        return self._worker_questions

    def _question_repository(self):
        if self._question_repo is None:
            from lexi_ai.questions.repository import QuestionRepository

            self._question_repo = QuestionRepository(self._session_factory)
        return self._question_repo

    def _build_question_engine(self, *, providers: bool) -> QuestionEngine:
        from lexi_ai.questions.base import load_entry_point_types
        from lexi_ai.questions.distractors import DistractorProvider
        from lexi_ai.questions.engine import QuestionEngine

        # Third-party question types are opt-in: register only those the Settings
        # allowlist names (built-ins are already loaded by direct import).
        load_entry_point_types(set(get_settings().question_type_allowlist) or None)

        return QuestionEngine(
            self._question_repository(),
            DistractorProvider(self._uow, self._embedder),
            llm=self._build_questions_llm() if providers else None,
            judge_llm=self._build_judge_llm() if providers else None,
            tts=self._build_tts_port() if providers else None,
            sense_loader=_LexiconSenseLoader(self),
        )

    # Public question API -------------------------------------------------------

    def question_types(self) -> list[QuestionTypeInfo]:
        return self._questions(providers=True).question_types()

    async def prepare_questions(self, word_id: int, demands: list[PrepareDemand]) -> PrepareReport:
        return await self._questions(providers=True).prepare(word_id, demands)

    async def get_question(self, question_id: int) -> PresentedQuestion | None:
        return await self._questions(providers=True).get(question_id)

    async def list_questions_for_sense(
        self, sense_id: int, type_id: str | None = None
    ) -> list[PresentedQuestion]:
        return await self._questions(providers=True).list_for_sense(sense_id, type_id)

    async def retrieve_question(
        self,
        sense_id: int,
        difficulty_level: int,
        excluded_ids: frozenset[int],
        type_id: str,
    ) -> PresentedQuestion | None:
        return await self._questions(providers=True).retrieve(
            sense_id, difficulty_level, excluded_ids, type_id
        )

    async def retrieve_exposure(self, sense_id: int) -> PresentedQuestion:
        return await self._questions(providers=True).retrieve_exposure(sense_id)

    async def evaluate_answer(
        self, question_id: int, submission: AnswerSubmission
    ) -> Evaluation | None:
        return await self._questions(providers=True).evaluate(question_id, submission)

    def questions(self, *, providers: bool) -> QuestionService:
        """The question service for one capability context.

        Public because the facades need it and must not reach into privates.
        """
        return self._questions(providers=providers)

    def _questions(self, *, providers: bool) -> QuestionService:
        """Build the question service over the engine for this context.

        The engine itself is cached by the properties, so this wrapper is rebuilt per
        call and replacing the engine (as tests do) takes effect immediately.
        """
        engine = self.worker_questions if providers else self.reader_questions
        return QuestionService(engine, self._question_repository(), self.get_entry)

    def _build_questions_llm(self) -> StructuredLLM | None:
        """Structured LLM for the contextual-MCQ plugin (bound to ``GeneratedMCQ``
        at the call site via :func:`ainvoke_structured`)."""
        return self._build_structured()

    def _build_judge_llm(self) -> StructuredLLM | None:
        """Structured LLM for the rubric scorer (bound to ``Judgment`` at the call
        site via :func:`ainvoke_structured`)."""
        return self._build_structured()

    def _build_structured(self) -> StructuredLLM | None:
        """Build the openai-backed structured LLM from settings.

        Returns ``None`` when no LLM is configured (empty api key), so the
        llm-dependent formats degrade gracefully instead of failing at import.
        The schema is supplied per-call by ``parse``, so one client serves both
        the MCQ generator and the judge.
        """
        settings = get_settings()
        if not settings.llm_api_key:
            return None
        from lexi_ai.llm import build_structured_llm

        return build_structured_llm(settings)

    def _build_wsd_judge(self):
        """WSD judge for sense-relation reconciliation, or ``None`` when no LLM is
        configured (resolve degrades to a no-op, like the other llm formats)."""
        llm = self._build_structured()
        if llm is None:
            return None
        from lexi_ai.generation.wsd import WsdJudge

        return WsdJudge(llm)

    @property
    def _wsd(self) -> WsdJudge | None:
        """The WSD judge, built once from settings on first use (or the injected
        fake). ``None`` when no LLM is configured — resolve is then a no-op."""
        if not self._wsd_built:
            self._wsd_judge = self._build_wsd_judge()
            self._wsd_built = True
        return self._wsd_judge

    def _build_tts_port(self) -> TtsPort | None:
        """Audio-synthesis port for the listening/spelling formats, or ``None`` when
        no TTS is configured (so those formats degrade to ``[]`` like the llm ones).

        The port ensures a clip cache-first via :meth:`tts_field` and returns its
        ``(source_kind, source_id, voice, fmt)`` reference tuple — never a row id, so
        the payload stays durable across a purge/regenerate.
        """
        settings = get_settings()
        if not (settings.tts_api_key or settings.tts_base_url):
            return None
        return _LexiconTtsPort(self)

    # --- public API -------------------------------------------------------

    async def search(self, query: str) -> list[SearchResult]:
        """Search the dictionary for a raw string. Never generates (FREE).

        Returns one ranked list (best first) mixing two kinds of hit:

        * **generated** — a word already in the dictionary (``lexi_word_id`` set);
          pass the id to :meth:`get`.
        * **suggestion** — a reference word that *can* be generated
          (``cambridge_id`` set); pass the result to :meth:`generate`.

        A reference word whose ``match_key`` is already generated is folded into the
        generated hit (shown once, as generated), so nothing is offered for
        regeneration by mistake.
        """
        return await self._search().search(query)

    async def get_entry(self, word_id: int, theme: str | int | None = None) -> Entry:
        """Load a generated entry by its dictionary id. Never generates (FREE).

        ``theme`` (theme key or ID) overlays themed definition +
        examples where a themed row exists, falling back to neutral per-sense.
        An unknown ``theme`` raises ``ValueError`` (silently returning neutral
        would hide a caller bug). ``None`` (default) is the neutral entry unchanged.
        """
        return await self._dictionary().entry(word_id, theme)

    async def _resolve_theme_or_raise(self, theme: str | int) -> tuple[int, str]:
        """Resolve a theme key/id to ``(theme_id, style_prompt)`` or raise (3.6).

        Single home for the resolve-or-raise the three themed call sites shared.
        ``resolve_theme`` is key-first-then-id for a ``str`` (2.4), so a raw pass
        suffices; the ``_norm_theme_key`` retry is kept as a defensive fallback for
        a caller that passes an already-un-normalized display name."""
        return await self._themes().resolve_or_raise(theme)

    async def _uow_resolve_theme(self, theme: str | int) -> tuple[int, str] | None:
        """Resolve a theme key/id, retrying once through the key normalizer.

        ``resolve`` is already key-first-then-id for a string, so the retry only
        covers a caller that passed an un-normalized display name.
        """
        return await self._themes().resolve(theme)

    async def get_senses(self, sense_ids: list[int]) -> list[SenseView]:
        """Batch-resolve senses by their DB ids. Never generates (FREE).

        Returns a SenseView per found id, preserving the input order. Ids with no
        row are silently skipped (caller tolerates missing senses). Relationships
        are eager-loaded inside the session so views survive after it closes.
        """
        return await self._dictionary().senses(sense_ids)

    async def add_examples(
        self, sense_id: int, n: int = 3, theme: str | int | None = None
    ) -> SenseView:
        """Append up to ``n`` fresh example sentences to ONE sense, returning the
        updated :class:`SenseView`.

        Targeted augmentation — the ONE clean generation gap: an example is an
        open-ended illustration of a sense, so generating more never fabricates a
        linguistic fact. Never deletes or overwrites existing examples;
        ``example_order`` continues from the current max. ``n`` is a best-effort
        MAX (the model may return fewer and never fabricates); ``n <= 0`` is a
        no-op. Existing examples are fed to the generator for soft de-duplication.
        Embeddings are untouched (they are computed on the definition only).

        ``theme`` (key or id) augments the sense's themed overlay instead of its
        neutral examples; that overlay must already exist (see :meth:`generate`
        with ``theme=``) — a missing theme or overlay raises ``ValueError``.
        Unknown ``sense_id`` raises ``ValueError``; no LLM configured raises
        ``ValueError``.
        """
        return await self._enrichment().add_examples(sense_id, n, theme)

    def _example_generator(self) -> Generator:
        """The neutral generator, used for targeted example augmentation. Reuses
        the injected :class:`Generator`; raises ``ValueError`` when none is wired
        (mirrors ``translate_field``'s no-LLM posture)."""
        if self._generator is None:
            raise ValueError("no LLM configured for example generation")
        return self._generator

    async def _add_themed_examples(self, sense_id: int, n: int, theme: str | int) -> SenseView:
        """Append up to ``n`` in-voice examples to a sense's themed overlay.

        The overlay must already exist (word themed via :meth:`generate` with
        ``theme=``): a missing theme or a sense without a themed row for that
        theme raises ``ValueError`` — never silently themes the whole word.
        """
        theme_id, style_prompt = await self._resolve_theme_or_raise(theme)
        async with self._uow() as uow:
            ctx = await uow.senses.example_context(sense_id)
            if ctx is None:
                raise ValueError(f"unknown sense_id: {sense_id}")
            overlay = await uow.themes.overlay_for_sense(sense_id, theme_id)
        if overlay is None:
            raise ValueError(
                f"sense {sense_id} has no themed overlay for theme {theme!r}; "
                "theme the word first via generate(theme=)"
            )
        themed_sense_id, existing_themed = overlay
        context, _neutral_examples = ctx
        n = min(n, _MAX_EXAMPLES_PER_CALL)
        if n > 0:
            batch = await self._themed_generator().generate_examples(
                style_prompt, context, existing_themed, n
            )
            async with self._uow() as uow:
                await uow.themes.append_themed_examples(themed_sense_id, batch.examples)
                await uow.commit()
        return await self._themed_sense_view(sense_id, theme_id)

    async def _themed_sense_view(self, sense_id: int, theme_id: int) -> SenseView:
        """A :class:`SenseView` overlaying the themed definition + examples on the
        neutral sense (all other fields neutral, matching the entry read model)."""
        return await self._themes().sense_view(sense_id, theme_id)

    async def get_many(
        self, word_ids: list[int], theme: str | int | None = None
    ) -> list[BatchResult]:
        """Batch :meth:`get_entry` — one :class:`BatchResult` per input id, in
        order. Never generates (FREE). A missing/invalid id is reported as a
        failed item (``error`` set) rather than aborting the whole batch."""

        return await self._dictionary().entries(word_ids, theme)

    async def get_status(self, word_id: int) -> str | None:
        """Status of a dictionary word (``done`` | ``pending`` | ``error``), or
        ``None`` if no such id exists. Never generates (FREE)."""
        return await self._dictionary().status(word_id)

    async def get_status_many(self, word_ids: list[int]) -> list[BatchResult]:
        """Batch :meth:`get_status` — one :class:`BatchResult` per input id, in
        order (``value`` is ``None`` for an unknown id — that is a valid answer,
        not a failure). Never generates (FREE)."""

        return await self._dictionary().statuses(word_ids)

    async def semantic_search(self, query: str, k: int = 10) -> list[SemanticHit]:
        """Rank already-generated senses by meaning similarity to ``query``.

        Embeds the query locally and ranks every done sense that carries a
        current-model vector by cosine similarity, best first. FREE — never
        generates a dictionary entry (only the short query is embedded). Returns
        an empty list when nothing is embedded yet (e.g. the ``[embeddings]``
        extra isn't installed) or ``k <= 0``.
        """
        return await self._search().semantic_search(query, k)

    async def backfill_embeddings(self, *, limit: int | None = None) -> int:
        """Embed done senses that lack a current-model vector. Returns count embedded.

        Fills gaps left by best-effort generation (extra not installed at gen
        time) or by an embedding-model change (rows tagged with a different
        model). Idempotent: a second call with everything embedded returns 0. No
        LLM. Best-effort: returns 0 if the embeddings extra is unavailable.
        """
        return await self._enrichment().backfill_embeddings(limit=limit)

    async def list_tags(self) -> list[TagCount]:
        """Every topic tag with its live member count (over ``done`` words),
        sorted count-desc then name. Never generates (FREE)."""
        return await self._dictionary().list_tags()

    async def list_entries_by_tag(
        self, tag: str, *, limit: int | None = None
    ) -> list[SearchResult]:
        """Generated words carrying ``tag``, as generated-hit ``SearchResult``s.

        The query is resolved via ``tag_key`` (the write-path function) so
        ``"Business"``/``"business"``/``"cars"`` all hit the right tag. Never
        generates (FREE); pass a hit's ``lexi_word_id`` to :meth:`get_entry`.
        """
        return await self._dictionary().list_entries_by_tag(tag, limit=limit)

    async def list_entries(
        self, *, status: str = "done", limit: int | None = None, offset: int = 0
    ) -> list[SearchResult]:
        """Paginated browse of the whole dictionary, norm-sorted. Never
        generates (FREE). Lightweight rows (like :meth:`list_entries_by_tag`) —
        pass a hit's ``lexi_word_id`` to :meth:`get_entry` for the full entry."""
        return await self._dictionary().list_entries(status=status, limit=limit, offset=offset)

    async def delete_entry(self, word_id: int) -> bool:
        """Delete a dictionary word and all its content; return whether a row
        was removed. Cascades senses, aliases, links, tags, and questions (the
        DB-level ``ON DELETE CASCADE`` FKs, already wired)."""
        return await self._dictionary().delete_entry(word_id)

    async def rename_tag(
        self, tag: str, *, name: str | None = None, title: str | None = None
    ) -> bool:
        """Update an existing topic tag's display ``name``/``title``. The
        underlying dedup key is immutable — this never merges/re-keys a tag
        (see :meth:`merge_tags` for that). Returns whether ``tag`` was found."""
        return await self._tags().rename(tag, name=name, title=title)

    async def delete_tag(self, tag: str) -> bool:
        """Delete a topic tag; return whether one was found. Tagged words are
        untouched — they simply lose this one topic."""
        return await self._tags().delete(tag)

    async def merge_tags(self, sources: list[str], into: str) -> int:
        """Fold ``sources`` tags into ``into``, then delete the sources.
        ``into`` must already exist (``ValueError`` otherwise). Returns the
        number of word-tag associations re-pointed."""
        return await self._tags().merge(sources, into)

    async def create_theme(
        self,
        name: str,
        style_prompt: str,
        description: str | None = None,
        tone: str | None = None,
    ) -> Theme:
        """Create (or resolve/update) a style theme by its normalized ``theme_key``.

        If ``description`` and ``tone`` are provided, the theme is registered
        directly (no LLM). Otherwise, the LLM expands the theme name/key and style_prompt
        into a detailed theme profile (name, description, style_prompt, tone) before saving.
        """
        return await self._themes().create(name, style_prompt, description, tone)

    async def list_themes(self) -> list[Theme]:
        """Every style theme, name-sorted. Never generates (FREE)."""
        return await self._themes().list_all()

    async def get_theme(self, key: str) -> Theme | None:
        """A style theme by key (raw display name resolved via the same
        normalizer as :meth:`create_theme`), or ``None`` if unknown. FREE."""
        return await self._themes().get(key)

    async def update_theme(
        self,
        key: str,
        *,
        name: str | None = None,
        style_prompt: str | None = None,
        description: str | None = None,
        tone: str | None = None,
    ) -> Theme:
        """Partially update an EXISTING theme's fields (unset args are left
        unchanged). The theme's key is immutable — renaming ``name`` never
        re-keys it. Raises ``ValueError`` if ``key`` is unknown (unlike
        :meth:`create_theme`, this never creates)."""
        return await self._themes().update(
            key,
            name=name,
            style_prompt=style_prompt,
            description=description,
            tone=tone,
        )

    async def delete_theme(self, key: str) -> bool:
        """Delete a style theme by key; return whether one was removed.
        Cascades its themed senses/examples; neutral entries are untouched."""
        return await self._themes().delete(key)

    async def _run_themed_generation(
        self, lexi_word_id: int, theme_id: int, style_prompt: str
    ) -> None:
        await self._themes().restyle_word(lexi_word_id, theme_id, style_prompt)

    def _themed_generator(self):
        """Lazy themed generator; uses settings/OpenAI proxy by default."""
        if self._themed_gen is None:
            from lexi_ai.theming.generator import ThemedGenerator

            self._themed_gen = ThemedGenerator(settings=get_settings())
        return self._themed_gen

    def _theme_metadata_generator(self):
        """Lazy theme metadata generator."""
        if self._theme_meta_gen is None:
            from lexi_ai.theming.generator import ThemeMetadataGenerator

            self._theme_meta_gen = ThemeMetadataGenerator(settings=get_settings())
        return self._theme_meta_gen

    # --- cached assets ----------------------------------------------------

    def _generation(self) -> GenerationService:
        """The generation service over this Lexicon's PROCESS-scoped locks.

        The lock registries are owned by the Lexicon, not the service: rebuilt per
        call they would collapse nothing, because each caller would take its own.
        """
        return GenerationService(
            self._uow,
            self._writer,
            self._loader,
            self._generator,
            self._to_entry,
            self._locks,
            self._theme_locks,
            self._resolve_theme_or_raise,
            self._run_themed_generation,
            self._embed_words,
            self._enrichment().resolve_inbound,
        )

    def _search(self) -> SearchService:
        """The lookup service, rebuilt per call over this Lexicon's collaborators."""
        return SearchService(self._uow, self._loader, self._embedder)

    def _enrichment(self) -> EnrichmentService:
        """The enrichment service, rebuilt per call over this Lexicon's collaborators."""
        return EnrichmentService(
            self._uow,
            self._embedder,
            self._example_generator,
            lambda: self._wsd,
            self.get_senses,
            self._add_themed_examples,
            _MAX_EXAMPLES_PER_CALL,
        )

    def _tags(self) -> TagService:
        """The tag curation service, rebuilt per call."""
        return TagService(self._uow)

    def _themes(self) -> ThemeService:
        """The theme service, rebuilt per call over this Lexicon's collaborators."""
        return ThemeService(
            self._uow,
            self._themed_generator,
            self._theme_metadata_generator,
            self.get_senses,
            self.get_status,
        )

    def _dictionary(self) -> DictionaryService:
        """The read service, rebuilt per call over this Lexicon's unit of work."""
        return DictionaryService(self._uow, self._resolve_theme_or_raise)

    def _asset_service(self) -> AssetService:
        """The asset service, rebuilt per call over the lazily-built cache."""
        settings = get_settings()
        return AssetService(
            self._require_assets(),
            self._translator,
            self._tts_provider,
            voice=settings.tts_voice,
            fmt=settings.tts_format,
            gather=self._gather_batch,
        )

    async def source_hash(self, source_kind: str, source_id: int) -> str | None:
        """The current content fingerprint of a translatable source, else ``None``."""
        return await self._asset_service().source_hash(source_kind, source_id)

    async def translate_field(self, source_kind: str, source_id: int, lang: str) -> str:
        """Translate a source into ``lang``, cache-first over the reference store."""
        return await self._asset_service().translate(source_kind, source_id, lang)

    async def translate_sense(self, sense_id: int, lang: str) -> str:
        """Translate a sense's definition — the everyday translation surface."""
        return await self.translate_field("sense_def", sense_id, lang)

    async def translate_many(
        self, refs: list[tuple[str, int]], lang: str, *, concurrency: int = 5
    ) -> list[BatchResult]:
        """Batch :meth:`translate_field`, order-aligned, cache-first per item."""
        return await self._asset_service().translate_many(refs, lang, concurrency=concurrency)

    async def tts_many(
        self,
        refs: list[tuple[str, int]],
        voice: str | None = None,
        fmt: str | None = None,
        *,
        concurrency: int = 5,
    ) -> list[BatchResult]:
        """Batch :meth:`tts_field`, order-aligned; one failure never aborts the rest."""
        return await self._asset_service().speak_many(
            refs, voice, fmt, concurrency=concurrency
        )

    async def stats(self) -> Stats:
        """Read-only dictionary counts in one grouped snapshot (no LLM)."""
        return await self._dictionary().stats()

    def _require_assets(self) -> AssetRepository:
        """The asset cache, constructed lazily from settings if not injected."""
        if self._assets is None:
            self._assets = AssetRepository(self._session_factory, get_settings().asset_cache_dir)
        return self._assets

    def _translator(self):
        """The translator, or ``None`` when no LLM is configured."""
        if self._translator_impl is not None:
            return self._translator_impl
        settings = get_settings()
        if not settings.llm_api_key:
            return None
        from lexi_ai.assets.translate import Translator

        self._translator_impl = Translator(settings=settings)
        return self._translator_impl

    async def tts_field(
        self, source_kind: str, source_id: int, voice: str | None = None, fmt: str | None = None
    ) -> Asset:
        """Synthesize speech for a source, cache-first over the reference store."""
        return await self._asset_service().speak(source_kind, source_id, voice, fmt)

    async def tts_sense(
        self, sense_id: int, voice: str | None = None, fmt: str | None = None
    ) -> Asset:
        """Synthesize a sense's definition — the everyday speech surface."""
        return await self.tts_field("sense_def", sense_id, voice, fmt)

    def _tts_provider(self):
        """The speech provider: the real one when configured, else the stub.

        The stub raises instead of returning audio, so an unconfigured install fails
        loudly rather than caching something fake.
        """
        if self._tts_impl is not None:
            return self._tts_impl
        settings = get_settings()
        if settings.tts_api_key or settings.tts_base_url:
            from lexi_ai.assets.tts import OpenAICompatibleTTSProvider

            self._tts_impl = OpenAICompatibleTTSProvider(
                base_url=settings.tts_base_url,
                api_key=settings.tts_api_key,
                model=settings.tts_model,
            )
        else:
            from lexi_ai.assets.tts import StubTTSProvider

            self._tts_impl = StubTTSProvider()
        return self._tts_impl

    async def get_asset(self, asset_id: int) -> Asset | None:
        """A cached asset by id, or ``None``."""
        return await self._asset_service().get(asset_id)

    async def list_assets(
        self, *, kind: str | None = None, limit: int | None = None, offset: int = 0
    ) -> list[Asset]:
        """Cached assets, oldest first, optionally filtered by kind."""
        return await self._asset_service().list(kind=kind, limit=limit, offset=offset)

    async def delete_asset(self, asset_id: int) -> bool:
        """Delete one cached asset and its backing file."""
        return await self._asset_service().delete(asset_id)

    async def purge_assets(self, *, kind: str | None = None) -> int:
        """Delete every cached asset, unlinking backing files."""
        return await self._asset_service().purge(kind=kind)

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
          generated, returns the existing entry (no LLM) unless ``force``.
        * ``str`` — a custom word Cambridge lacks; anchored to WordNet only.

        A suggestion whose word already exists converges on that entry instead of
        duplicating it. With ``force=True`` the entry is regenerated and overwritten
        in place.

        If a ``theme`` (name or key) is provided, the generated/resolved entry is
        automatically restyled in that theme's voice (if not already done).
        """
        return await self._generation().generate(
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
        """Batch :meth:`generate` — one :class:`BatchResult` per input source, in
        order, up to ``concurrency`` in flight. Each item goes through the SAME
        :meth:`generate`, so two items resolving to the same word still generate
        exactly once (the existing per-``match_key`` lock + DB double-check is
        reused unchanged) — no new locking logic here."""

        return await self._generation().generate_many(
            sources, force=force, theme=theme, concurrency=concurrency
        )

    async def generate_fenced(
        self, source: SearchResult | str, *, structured_method: str | None = None
    ) -> Entry:
        """Generate once under a database fence, for independently deployed workers.

        Deliberately has no ``force``: a remote caller must not be able to use a
        delayed job to replace an entry a newer claim owns.
        """
        return await self._generation().generate_fenced(
            source, structured_method=structured_method
        )

    async def resolve_relations(self, batch_size: int = 20) -> list[BatchResult]:
        """Reconcile one batch of pending sense-relation edges (manual/backfill)."""
        return await self._enrichment().resolve_relations(batch_size)

    async def _embed_words(self, word_ids: list[int]) -> int:
        """Embed the senses of the given words, best-effort."""
        return await self._enrichment().embed_missing(word_ids=word_ids)

    async def _run_generation(
        self,
        word: str,
        cambridge_id: int | None,
        *,
        fence=None,
        structured_method: str | None = None,
    ):
        """Build the bundle, generate, publish, then enrich after the commit."""
        return await self._generation()._run(
            word, cambridge_id, fence=fence, method=structured_method
        )

    @staticmethod
    async def _gather_batch(items: list, fn, concurrency: int | None = None) -> list[BatchResult]:
        """Order-aligned batch execution; one failure never cancels its siblings."""
        return await gather_batch(items, fn, concurrency)

    # --- read model assembly ---------------------------------------------

    async def _to_entry(self, word_id: int, theme_id: int | None = None) -> Entry:
        async with self._uow() as uow:
            overlay = (
                await uow.themes.overlay_for_word(word_id, theme_id)
                if theme_id is not None
                else None
            )
            return await uow.entries.entry(word_id, overlay)


class _LexiconSenseLoader:
    """Resolve a sense to its owning entry for provider-free exposure cards."""

    def __init__(self, lexicon: Lexicon):
        self._lexicon = lexicon

    async def load_entry(self, sense_id: int) -> Entry | None:
        """Resolve the sense's owning entry, or ``None`` when the sense is gone."""
        async with self._lexicon._uow() as uow:
            try:
                word_id = await uow.senses.word_id_for(sense_id)
            except NoResultFound:
                return None
        return await self._lexicon.get_entry(word_id)


class _LexiconTtsPort:
    """Adapts the :class:`Lexicon` TTS surface to the questions ``TtsPort`` seam.

    ``ensure_clip`` synthesizes (cache-first) via :meth:`Lexicon.tts_field` and
    returns the clip's ``(source_kind, source_id, voice, fmt)`` reference tuple —
    never a row id, so a frozen question payload survives a purge/regenerate.
    Voice/fmt are resolved from settings here so plugins stay decoupled from config.
    Returns ``None`` when no clip can be made (source text gone, or empty-text
    short-circuit) so the audio formats degrade rather than fabricating an asset.
    """

    def __init__(self, lexicon: Lexicon):
        self._lexicon = lexicon

    async def ensure_clip(
        self, source_kind: str, source_id: int
    ) -> tuple[str, int, str, str] | None:
        settings = get_settings()
        voice, fmt = settings.tts_voice, settings.tts_format
        try:
            asset = await self._lexicon.tts_field(source_kind, source_id, voice, fmt)
        except ValueError:
            return None  # no source text for the ref — nothing to synthesize
        if not asset.ready:
            return None  # empty/whitespace source short-circuited — no real clip
        return (source_kind, source_id, voice, fmt)
