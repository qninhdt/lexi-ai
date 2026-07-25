"""Lazy lookup API (Phase 6): the public ``Lexicon.get_entry`` surface.

Flow: normalize input -> resolve by match_key against words AND aliases ->
branch on 0 / 1-done / 1-pending / N. Misses run the full lazy pipeline
(reference -> generate -> persist) exactly once per key, guarded by a per-key
asyncio lock plus a DB double-check (library, single-process — decision #18).

``display`` is always ``render(norm)``; no display column is ever read.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from sqlalchemy.exc import NoResultFound
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from lexi_ai.application.generation_writer import GenerationWriter
from lexi_ai.application.questions import QuestionService
from lexi_ai.assets.repository import AssetRepository, content_hash, normalize_asset_params
from lexi_ai.config import Settings, get_settings
from lexi_ai.constants import canonical_cambridge_ref
from lexi_ai.db import create_engine, create_session_factory, init_models
from lexi_ai.domain.hashing import sense_content_hash
from lexi_ai.domain.models import WordListing
from lexi_ai.embeddings import Embedder
from lexi_ai.generation.generator import Generator
from lexi_ai.generation.schemas import ExampleBatch
from lexi_ai.infrastructure.db.mappers import theme_view
from lexi_ai.infrastructure.db.uow import SqlAlchemyUnitOfWork
from lexi_ai.normalize import match_key, render, tag_key
from lexi_ai.normalize import theme_key as _norm_theme_key
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
from lexi_ai.references.loader import ReferenceBundle, ReferenceLoader
from lexi_ai.references.wordnet import WordNetSource
from lexi_ai.vectors import cosine, pack_vector, unpack_vector

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
        self._locks: dict[str, asyncio.Lock] = {}
        # Single-flight lock for the THEMED overlay step (2.6), keyed on
        # (word_id, theme_id) — word_id is the canonical resolution of the word's
        # match_key (using it, not the raw display key, sidesteps the 2.1
        # display-vs-norm key mismatch). A DISTINCT map from _locks so the overlay
        # lock can never form a cycle with the neutral per-key lock (the neutral
        # lock is fully released before the overlay block runs, never nested).
        self._theme_locks: dict[tuple[int, int], asyncio.Lock] = {}
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
        exact = await self._loader.cambridge.resolve_exact(query)
        exact_ids = {ref.word_id for ref in exact}
        ranked = await self._loader.cambridge.rank_similar(query)
        # (cambridge_id, display, entry_type, score) — exact first at 1.0, then
        # fuzzy; dedup by cambridge_id preserving the best (first) score.
        refs: list[tuple[int, str, str | None, float]] = [
            (r.word_id, r.display_form, r.entry_type, 1.0) for r in exact
        ]
        refs += [
            (r.word_id, r.display_form, r.entry_type, score)
            for r, score in ranked
            if r.word_id not in exact_ids
        ]
        seen_ids: set[int] = set()
        deduped: list[tuple[int, str, str | None, float]] = []
        for cid, display, entry_type, score in refs:
            if cid not in seen_ids:
                seen_ids.add(cid)
                deduped.append((cid, display, entry_type, score))

        # A reference is "already generated" if a word carries its cambridge_id
        # provenance (robust to display/norm key differences). Fold those into a
        # single generated hit per lexi word so nothing is re-offered.
        generated = await self._generated_by_cambridge([cid for cid, _, _, _ in deduped])
        glosses = await self._loader.cambridge.first_definitions([cid for cid, _, _, _ in deduped])
        results: list[SearchResult] = []
        seen_lexi: set[int] = set()
        for cid, display, entry_type, score in deduped:
            hit = generated.get(cid)
            if hit is not None:
                lexi_id, gen_display, gen_type = hit
                if lexi_id in seen_lexi:
                    continue  # two Cambridge ids fold to one generated word
                seen_lexi.add(lexi_id)
                results.append(
                    SearchResult(
                        display=gen_display,
                        entry_type=gen_type,
                        score=score,
                        lexi_word_id=lexi_id,
                    )
                )
            else:
                results.append(
                    SearchResult(
                        display=display,
                        entry_type=entry_type,
                        score=score,
                        cambridge_id=cid,
                        gloss=glosses.get(cid),
                    )
                )
        results.sort(key=lambda r: (-r.score, r.display))
        return results

    async def get_entry(self, word_id: int, theme: str | int | None = None) -> Entry:
        """Load a generated entry by its dictionary id. Never generates (FREE).

        ``theme`` (theme key or ID) overlays themed definition +
        examples where a themed row exists, falling back to neutral per-sense.
        An unknown ``theme`` raises ``ValueError`` (silently returning neutral
        would hide a caller bug). ``None`` (default) is the neutral entry unchanged.
        """
        if theme is None:
            return await self._to_entry(word_id)
        theme_id, _style = await self._resolve_theme_or_raise(theme)
        return await self._to_entry(word_id, theme_id=theme_id)

    async def _resolve_theme_or_raise(self, theme: str | int) -> tuple[int, str]:
        """Resolve a theme key/id to ``(theme_id, style_prompt)`` or raise (3.6).

        Single home for the resolve-or-raise the three themed call sites shared.
        ``resolve_theme`` is key-first-then-id for a ``str`` (2.4), so a raw pass
        suffices; the ``_norm_theme_key`` retry is kept as a defensive fallback for
        a caller that passes an already-un-normalized display name."""
        resolved = await self._uow_resolve_theme(theme)
        if resolved is None:
            raise ValueError(f"unknown theme: {theme!r}")
        return resolved

    async def _uow_resolve_theme(self, theme: str | int) -> tuple[int, str] | None:
        """Resolve a theme key/id, retrying once through the key normalizer.

        ``resolve`` is already key-first-then-id for a string, so the retry only
        covers a caller that passed an un-normalized display name.
        """
        async with self._uow() as uow:
            resolved = await uow.themes.resolve(theme)
            if resolved is None and isinstance(theme, str):
                resolved = await uow.themes.resolve(_norm_theme_key(theme))
            return resolved

    async def get_senses(self, sense_ids: list[int]) -> list[SenseView]:
        """Batch-resolve senses by their DB ids. Never generates (FREE).

        Returns a SenseView per found id, preserving the input order. Ids with no
        row are silently skipped (caller tolerates missing senses). Relationships
        are eager-loaded inside the session so views survive after it closes.
        """
        if not sense_ids:
            return []
        async with self._uow() as uow:
            return await uow.entries.sense_views(sense_ids)

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
        if theme is not None:
            return await self._add_themed_examples(sense_id, n, theme)
        async with self._uow() as uow:
            ctx = await uow.senses.example_context(sense_id)
        if ctx is None:
            raise ValueError(f"unknown sense_id: {sense_id}")
        context, existing = ctx
        # Clamp to ExampleBatch's schema ceiling: a larger n would prompt for more
        # than the model can validly return, wasting the structured-output retries.
        n = min(n, _MAX_EXAMPLES_PER_CALL)
        if n > 0:
            batch = await self._example_generator().generate_examples(context, existing, n)
            async with self._uow() as uow:
                await uow.senses.append_examples(sense_id, batch.examples)
                await uow.commit()
        return (await self.get_senses([sense_id]))[0]

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
        base = (await self.get_senses([sense_id]))[0]
        async with self._uow() as uow:
            word_id = await uow.senses.word_id_for(sense_id)
            overlay = await uow.themes.overlay_for_word(word_id, theme_id)
        if sense_id in overlay:
            themed_def, themed_examples = overlay[sense_id]
            base.definition = themed_def
            base.examples = themed_examples
        return base

    async def get_many(
        self, word_ids: list[int], theme: str | int | None = None
    ) -> list[BatchResult]:
        """Batch :meth:`get_entry` — one :class:`BatchResult` per input id, in
        order. Never generates (FREE). A missing/invalid id is reported as a
        failed item (``error`` set) rather than aborting the whole batch."""

        async def _one(word_id: int) -> Entry:
            return await self.get_entry(word_id, theme=theme)

        return await self._gather_batch(word_ids, _one)

    async def get_status(self, word_id: int) -> str | None:
        """Status of a dictionary word (``done`` | ``pending`` | ``error``), or
        ``None`` if no such id exists. Never generates (FREE)."""
        async with self._uow() as uow:
            return await uow.words.status(word_id)

    async def get_status_many(self, word_ids: list[int]) -> list[BatchResult]:
        """Batch :meth:`get_status` — one :class:`BatchResult` per input id, in
        order (``value`` is ``None`` for an unknown id — that is a valid answer,
        not a failure). Never generates (FREE)."""

        async def _one(word_id: int) -> str | None:
            return await self.get_status(word_id)

        return await self._gather_batch(word_ids, _one)

    async def semantic_search(self, query: str, k: int = 10) -> list[SemanticHit]:
        """Rank already-generated senses by meaning similarity to ``query``.

        Embeds the query locally and ranks every done sense that carries a
        current-model vector by cosine similarity, best first. FREE — never
        generates a dictionary entry (only the short query is embedded). Returns
        an empty list when nothing is embedded yet (e.g. the ``[embeddings]``
        extra isn't installed) or ``k <= 0``.
        """
        if k <= 0:
            return []
        async with self._uow() as uow:
            rows = await uow.senses.embedded(self._embedder.model_name)
        if not rows:
            return []
        try:
            qvec = await self._embedder.embed_one(query)
        except Exception:  # noqa: BLE001 - best-effort: search degrades to [] on embed failure
            return []
        scored = sorted(
            ((cosine(qvec, unpack_vector(row.embedding)), row) for row in rows),
            key=lambda s: -s[0],
        )
        return [
            SemanticHit(
                lexi_word_id=row.word_id,
                display=render(row.norm),
                entry_type=row.entry_type,
                score=score,
                sense=SenseView(
                    definition=row.definition, tier=row.tier, pos=None, cefr_level=None
                ),
            )
            for score, row in scored[:k]
        ]

    async def backfill_embeddings(self, *, limit: int | None = None) -> int:
        """Embed done senses that lack a current-model vector. Returns count embedded.

        Fills gaps left by best-effort generation (extra not installed at gen
        time) or by an embedding-model change (rows tagged with a different
        model). Idempotent: a second call with everything embedded returns 0. No
        LLM. Best-effort: returns 0 if the embeddings extra is unavailable.
        """
        return await self._embed_missing(limit=limit)

    async def list_tags(self) -> list[TagCount]:
        """Every topic tag with its live member count (over ``done`` words),
        sorted count-desc then name. Never generates (FREE)."""
        async with self._uow() as uow:
            rows = await uow.tags.usage()
        return [TagCount(name=name, title=title, count=count) for name, title, count in rows]

    async def list_entries_by_tag(
        self, tag: str, *, limit: int | None = None
    ) -> list[SearchResult]:
        """Generated words carrying ``tag``, as generated-hit ``SearchResult``s.

        The query is resolved via ``tag_key`` (the write-path function) so
        ``"Business"``/``"business"``/``"cars"`` all hit the right tag. Never
        generates (FREE); pass a hit's ``lexi_word_id`` to :meth:`get_entry`.
        """
        async with self._uow() as uow:
            rows = await uow.tags.words_for_key(tag_key(tag), limit=limit)
        return [
            SearchResult(display=render(norm), entry_type=etype, lexi_word_id=wid)
            for wid, norm, etype in rows
        ]

    async def list_entries(
        self, *, status: str = "done", limit: int | None = None, offset: int = 0
    ) -> list[SearchResult]:
        """Paginated browse of the whole dictionary, norm-sorted. Never
        generates (FREE). Lightweight rows (like :meth:`list_entries_by_tag`) —
        pass a hit's ``lexi_word_id`` to :meth:`get_entry` for the full entry."""
        async with self._uow() as uow:
            rows = await uow.words.listing(status=status, limit=limit, offset=offset)
        return [
            SearchResult(display=render(norm), entry_type=etype, lexi_word_id=wid)
            for wid, norm, etype in rows
        ]

    async def delete_entry(self, word_id: int) -> bool:
        """Delete a dictionary word and all its content; return whether a row
        was removed. Cascades senses, aliases, links, tags, and questions (the
        DB-level ``ON DELETE CASCADE`` FKs, already wired)."""
        async with self._uow() as uow:
            deleted = await uow.words.delete(word_id)
            await uow.commit()
            return deleted

    async def rename_tag(
        self, tag: str, *, name: str | None = None, title: str | None = None
    ) -> bool:
        """Update an existing topic tag's display ``name``/``title``. The
        underlying dedup key is immutable — this never merges/re-keys a tag
        (see :meth:`merge_tags` for that). Returns whether ``tag`` was found."""
        async with self._uow() as uow:
            renamed = await uow.tags.rename(tag, name=name, title=title)
            await uow.commit()
            return renamed

    async def delete_tag(self, tag: str) -> bool:
        """Delete a topic tag; return whether one was found. Tagged words are
        untouched — they simply lose this one topic."""
        async with self._uow() as uow:
            deleted = await uow.tags.delete(tag)
            await uow.commit()
            return deleted

    async def merge_tags(self, sources: list[str], into: str) -> int:
        """Fold ``sources`` tags into ``into``, then delete the sources.
        ``into`` must already exist (``ValueError`` otherwise). Returns the
        number of word-tag associations re-pointed."""
        async with self._uow() as uow:
            moved = await uow.tags.merge(sources, into)
            await uow.commit()
            return moved

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
        norm_key = _norm_theme_key(name)
        if not norm_key:
            raise ValueError(f"theme name yields no valid key: {name!r}")

        if description is None or tone is None:
            generator = self._theme_metadata_generator()
            generated = await generator.generate(norm_key, style_prompt)
            fields = {
                "name": generated.name,
                "style_prompt": generated.style_prompt,
                "description": generated.description,
                "tone": ",".join(generated.tone) if generated.tone else None,
            }
        else:
            fields = {
                "name": name,
                "style_prompt": style_prompt,
                "description": description,
                "tone": tone,
            }

        # The metadata call above stays outside the transaction: it is a provider
        # round trip, and holding a write transaction across it would idle a
        # connection for the duration.
        async with self._uow() as uow:
            theme = await uow.themes.create(key=norm_key, overwrite=True, **fields)
            await uow.commit()
        return theme_view(theme)

    async def list_themes(self) -> list[Theme]:
        """Every style theme, name-sorted. Never generates (FREE)."""
        async with self._uow() as uow:
            themes = await uow.themes.list_all()
        return [theme_view(theme) for theme in themes]

    async def get_theme(self, key: str) -> Theme | None:
        """A style theme by key (raw display name resolved via the same
        normalizer as :meth:`create_theme`), or ``None`` if unknown. FREE."""
        async with self._uow() as uow:
            theme = await uow.themes.get(_norm_theme_key(key))
        return theme_view(theme) if theme is not None else None

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
        async with self._uow() as uow:
            theme = await uow.themes.update(
                _norm_theme_key(key),
                name=name,
                style_prompt=style_prompt,
                description=description,
                tone=tone,
            )
            await uow.commit()
        if theme is None:
            raise ValueError(f"unknown theme: {key!r}")
        return theme_view(theme)

    async def delete_theme(self, key: str) -> bool:
        """Delete a style theme by key; return whether one was removed.
        Cascades its themed senses/examples; neutral entries are untouched."""
        async with self._uow() as uow:
            deleted = await uow.themes.delete(_norm_theme_key(key))
            await uow.commit()
            return deleted

    async def _run_themed_generation(
        self, lexi_word_id: int, theme_id: int, style_prompt: str
    ) -> None:
        status = await self.get_status(lexi_word_id)
        if status != "done":
            raise ValueError(f"word {lexi_word_id} is not done (status={status!r})")

        async with self._uow() as uow:
            neutral = await uow.senses.for_theming(lexi_word_id)
        if not neutral:
            raise ValueError(f"word {lexi_word_id} has no senses to theme")
        sense_ids = [row.sense_id for row in neutral]
        facts = [(row.definition, row.pos, row.guideword, row.tier) for row in neutral]

        generator = self._themed_generator()
        result = await generator.generate(style_prompt, facts)
        async with self._uow() as uow:
            await uow.themes.persist_themed(theme_id, result, sense_ids)
            await uow.commit()

    def _themed_generator(self):
        """Lazy themed generator; uses settings/OpenAI proxy by default."""
        if getattr(self, "_themed_gen", None) is None:
            from lexi_ai.theming.generator import ThemedGenerator

            self._themed_gen = ThemedGenerator(settings=get_settings())
        return self._themed_gen

    def _theme_metadata_generator(self):
        """Lazy theme metadata generator."""
        if getattr(self, "_theme_meta_gen", None) is None:
            from lexi_ai.theming.generator import ThemeMetadataGenerator

            self._theme_meta_gen = ThemeMetadataGenerator(settings=get_settings())
        return self._theme_meta_gen

    # --- cached assets ----------------------------------------------------

    async def source_hash(self, source_kind: str, source_id: int) -> str | None:
        """Return the current content fingerprint for a translatable source.

        Service workers use this narrow read to fence delayed translation jobs
        before they call a provider. ``None`` means the source no longer exists.
        """
        text = await self._require_assets().resolve_source_text(source_kind, source_id)
        return None if text is None else content_hash(text)

    async def translate_field(self, source_kind: str, source_id: int, lang: str) -> str:
        """Translate the source text at ``(source_kind, source_id)`` into ``lang``,
        cache-first over the reference store (hash-verified).

        Resolves the CURRENT source text, then reads the cache verified against it:
        a repeat call with unchanged source spends ZERO LLM; a regenerated/reused
        source id re-translates (miss), never returns stale text. Empty/whitespace
        source returns as-is (no LLM, no row). Raises ``ValueError`` on a bad ref
        (source row absent) or when no LLM is configured.
        """
        assets = self._require_assets()
        text = await assets.resolve_source_text(source_kind, source_id)
        if text is None:
            raise ValueError(f"no source text for ({source_kind!r}, {source_id})")
        if not text.strip():
            return text
        params = normalize_asset_params("translate", lang=lang)
        cached = await assets.get(source_kind, source_id, "translate", params, text)
        if cached is not None and cached.text_value is not None:
            return cached.text_value
        translator = self._translator()
        if translator is None:
            raise ValueError("no LLM configured for translation")
        result = await translator.translate(text, lang)
        stored = await assets.put_text(source_kind, source_id, "translate", params, text, result)
        return stored.text_value or result

    async def translate_sense(self, sense_id: int, lang: str) -> str:
        """Translate a sense's definition into ``lang`` (the everyday surface).

        Convenience for ``translate_field("sense_def", sense_id, lang)``."""
        return await self.translate_field("sense_def", sense_id, lang)

    async def translate_many(
        self, refs: list[tuple[str, int]], lang: str, *, concurrency: int = 5
    ) -> list[BatchResult]:
        """Batch :meth:`translate_field` — one :class:`BatchResult` per
        ``(source_kind, source_id)`` ref, in order, up to ``concurrency`` in
        flight. Cache-first per item, so a repeated source spends zero LLM."""

        async def _one(ref: tuple[str, int]) -> str:
            return await self.translate_field(ref[0], ref[1], lang)

        return await self._gather_batch(refs, _one, concurrency=concurrency)

    async def tts_many(
        self,
        refs: list[tuple[str, int]],
        voice: str | None = None,
        fmt: str | None = None,
        *,
        concurrency: int = 5,
    ) -> list[BatchResult]:
        """Batch :meth:`tts_field` — one :class:`BatchResult` per
        ``(source_kind, source_id)`` ref, in order, up to ``concurrency`` in
        flight. Cache-first per item, so a source already synthesized by an
        EARLIER call spends zero provider call (two identical refs in the SAME
        batch may both miss and synthesize — the content-addressed put path
        dedups the row, worst case one wasted call). Mirror of
        :meth:`translate_many`; one item's failure (e.g. the unconfigured-TTS
        stub raising) is reported without aborting the rest."""

        async def _one(ref: tuple[str, int]) -> Asset:
            return await self.tts_field(ref[0], ref[1], voice, fmt)

        return await self._gather_batch(refs, _one, concurrency=concurrency)

    async def stats(self) -> Stats:
        """Read-only dictionary counts (never generates, no LLM). One round of
        grouped COUNT queries — words by status, senses, examples, tags, themes,
        words with any themed overlay, assets by kind, and questions."""
        async with self._uow() as uow:
            snapshot = await uow.stats.snapshot()
        return snapshot

    def _require_assets(self) -> AssetRepository:
        """The asset cache, constructed lazily from settings if not injected."""
        if self._assets is None:
            self._assets = AssetRepository(self._session_factory, get_settings().asset_cache_dir)
        return self._assets

    def _translator(self):
        """Lazy translator; ``None`` when no LLM is configured (empty api key).

        An injected translator (``self._translator_impl`` pre-set) is used as-is.
        """
        if getattr(self, "_translator_impl", None) is not None:
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
        """Synthesize speech for the source at ``(source_kind, source_id)``,
        cache-first over the reference store (hash-verified).

        A verified cache hit returns the ``Asset`` WITHOUT calling the provider.
        On a miss the provider is invoked; when it is the STUB (no ``LEXI_TTS_*``
        configured) it raises ``NotImplementedError`` and no fake audio is cached.
        Empty/whitespace source short-circuits. Raises ``ValueError`` on a bad ref.
        The audio bytes are stored via ``put_file`` keyed by the reference tuple.
        """
        settings = get_settings()
        voice = voice if voice is not None else settings.tts_voice
        fmt = fmt if fmt is not None else settings.tts_format
        params = normalize_asset_params("tts", voice=voice, fmt=fmt)
        assets = self._require_assets()
        text = await assets.resolve_source_text(source_kind, source_id)
        if text is None:
            raise ValueError(f"no source text for ({source_kind!r}, {source_id})")
        if not text.strip():
            return Asset(source_kind=source_kind, source_id=source_id, kind="tts", params=params)
        cached = await assets.get(source_kind, source_id, "tts", params, text)
        if cached is not None:
            return cached
        provider = self._tts_provider()
        data = await provider.synthesize(text, voice, fmt)  # stub raises here
        return await assets.put_file(source_kind, source_id, "tts", params, text, data, ext=fmt)

    async def tts_sense(
        self, sense_id: int, voice: str | None = None, fmt: str | None = None
    ) -> Asset:
        """Synthesize speech for a sense's definition (the everyday surface).

        Convenience for ``tts_field("sense_def", sense_id, voice, fmt)``."""
        return await self.tts_field("sense_def", sense_id, voice, fmt)

    def _tts_provider(self):
        """Lazy TTS provider: the real OpenAI-compatible one when ``LEXI_TTS_*`` is
        configured, else the stub (so an unconfigured install fails loudly rather
        than caching fake audio). An injected provider is used as-is.
        """
        if getattr(self, "_tts_impl", None) is not None:
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
        """A cached asset (translation or TTS) by its id, or ``None``. FREE."""
        return await self._require_assets().get_by_id(asset_id)

    async def list_assets(
        self, *, kind: str | None = None, limit: int | None = None, offset: int = 0
    ) -> list[Asset]:
        """Cached assets, oldest first, optionally filtered by ``kind``
        (``"translate"`` | ``"tts"``). FREE."""
        return await self._require_assets().list(kind=kind, limit=limit, offset=offset)

    async def delete_asset(self, asset_id: int) -> bool:
        """Delete a cached asset by id (and its backing file, if any); return
        whether one was removed."""
        return await self._require_assets().delete(asset_id)

    async def purge_assets(self, *, kind: str | None = None) -> int:
        """Delete every cached asset (optionally one ``kind``), unlinking their
        backing files. Returns the number removed."""
        return await self._require_assets().purge(kind=kind)

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
        # Custom string: anchor to WordNet only, dedup by the string's match_key.
        if isinstance(source, str):
            entry = await self._generate_locked(
                match_key(source), source, None, force, structured_method=structured_method
            )
        # Already-generated hit: return it, or re-anchor to its Cambridge id on force.
        elif source.lexi_word_id is not None:
            if not force:
                entry = await self._to_entry(source.lexi_word_id)
            else:
                norm, cam_id = await self._word_norm_and_cambridge(source.lexi_word_id)
                entry = await self._generate_locked(
                    match_key(norm), norm, cam_id, True, structured_method=structured_method
                )
        # Suggestion: cache-check by Cambridge provenance, then generate.
        else:
            if source.cambridge_id is None:
                raise ValueError("SearchResult has neither lexi_word_id nor cambridge_id")
            if not force:
                hit = await self._generated_by_cambridge([source.cambridge_id])
                if source.cambridge_id in hit:
                    entry = await self._to_entry(hit[source.cambridge_id][0])
                else:
                    entry = await self._generate_locked(
                        match_key(source.display),
                        source.display,
                        source.cambridge_id,
                        force,
                        structured_method=structured_method,
                    )
            else:
                entry = await self._generate_locked(
                    match_key(source.display),
                    source.display,
                    source.cambridge_id,
                    force,
                    structured_method=structured_method,
                )

        if theme is not None:
            theme_id, style_prompt = await self._resolve_theme_or_raise(theme)

            # Single-flight the overlay step (2.6): the LLM call in
            # _run_themed_generation runs BEFORE persist_themed, so an unguarded
            # check-then-act let two concurrent generate(word, theme=T) both see no
            # overlay and both call the LLM. Serialize on (word_id, theme_id) and
            # RE-CHECK the overlay inside the lock so the second waiter adopts the
            # first's result instead of regenerating.
            theme_lock_key = (entry.word_id, theme_id)
            theme_lock = self._theme_locks.setdefault(theme_lock_key, asyncio.Lock())
            try:
                async with theme_lock:
                    async with self._uow() as uow:
                        overlay = await uow.themes.overlay_for_word(entry.word_id, theme_id)
                    if not overlay or force:
                        await self._run_themed_generation(entry.word_id, theme_id, style_prompt)
            finally:
                self._evict_theme_lock(theme_lock_key, theme_lock)

            # Reload entry with the theme overlay
            entry = await self._to_entry(entry.word_id, theme_id)

        return entry

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

        async def _one(source: SearchResult | str) -> Entry:
            return await self.generate(source, force=force, theme=theme)

        return await self._gather_batch(sources, _one, concurrency=concurrency)

    async def _generate_locked(
        self,
        key: str,
        word: str,
        cambridge_id: int | None,
        force: bool,
        *,
        structured_method: str | None = None,
    ) -> Entry:
        """Locked generate-and-persist for one word key (double-checked)."""
        lock = self._locks.setdefault(key, asyncio.Lock())
        try:
            async with lock:
                if not force:
                    done = await self._done_ids(key)
                    if done:
                        return await self._to_entry(done[0])
                result = await self._run_generation(
                    word, cambridge_id, structured_method=structured_method
                )
        finally:
            self._evict_lock(key, lock)
        return await self._entry_for_key(key, result)

    async def generate_fenced(
        self, source: SearchResult | str, *, structured_method: str | None = None
    ) -> Entry:
        """Generate once under a database fence for independently deployed workers.

        This service-facing seam deliberately has no ``force`` flag: remote
        callers cannot use a delayed job to replace an entry that a newer claim
        owns. Library callers retain :meth:`generate` and its local single-flight
        semantics.
        """
        if isinstance(source, str):
            key, word, cambridge_id = match_key(source), source, None
        elif source.lexi_word_id is not None:
            norm, cambridge_id = await self._word_norm_and_cambridge(source.lexi_word_id)
            key, word = match_key(norm), norm
        elif source.cambridge_id is not None:
            key, word, cambridge_id = match_key(source.display), source.display, source.cambridge_id
        else:
            raise ValueError("SearchResult has neither lexi_word_id nor cambridge_id")

        lock = self._locks.setdefault(key, asyncio.Lock())
        try:
            async with lock:
                done = await self._done_ids(key)
                if done:
                    return await self._to_entry(done[0])
                fence = await self._writer.claim(word)
                result = await self._run_generation(
                    word, cambridge_id, fence=fence, structured_method=structured_method
                )
        finally:
            self._evict_lock(key, lock)
        return await self._entry_for_key(key, result)

    async def _word_norm_and_cambridge(self, lexi_word_id: int) -> tuple[str, int | None]:
        async with self._uow() as uow:
            return await uow.words.norm_and_cambridge(lexi_word_id)

    async def _resolve(self, key: str) -> list[tuple[int, str]]:
        """Return (word_id, status) for every word matching key via headword or alias."""
        async with self._uow() as uow:
            return list(await uow.words.resolve_key(key))

    async def _done_ids(self, key: str) -> list[int]:
        """word_ids with a ``done`` entry for this match_key (headword or alias)."""
        return [wid for wid, status in await self._resolve(key) if status == "done"]

    async def _generated_by_cambridge(self, cambridge_ids: list[int]) -> dict[int, WordListing]:
        """Which Cambridge ids are already generated, keyed by that provenance."""
        async with self._uow() as uow:
            return await uow.words.generated_by_cambridge(cambridge_ids)

    # --- generation path --------------------------------------------------

    async def _run_generation(
        self,
        word: str,
        cambridge_id: int | None,
        *,
        fence=None,
        structured_method: str | None = None,
    ):
        """Build the bundle (Cambridge-anchored or custom), generate, persist.

        After persistence, embed the new senses best-effort: an embedding failure
        (extra missing, model load error) never fails the generation.
        """
        if cambridge_id is None:
            bundle = await self._loader.bundle_custom(word)
            cefr_map: dict[str, str] = {}
        else:
            got = await self._loader.bundle_by_id(cambridge_id)
            if got is None:
                raise ValueError(f"Cambridge word_id {cambridge_id} not found")
            bundle = got
            cefr_map = self._cefr_map(bundle)
        try:
            async with self._uow() as uow:
                existing_tags = await uow.tags.names()
        except Exception:  # noqa: BLE001 - vocab is best-effort; empty on failure
            existing_tags = []
        if structured_method is None:
            result = await self._generator.generate(bundle, existing_tags=existing_tags)
        else:
            result = await self._generator.generate(
                bundle, existing_tags=existing_tags, structured_method=structured_method
            )
        words = await self._writer.publish(
            result, cambridge_word_id=cambridge_id, cambridge_cefr=cefr_map, fence=fence
        )
        await self._embed_words([w.id for w in words])
        # [F11] Inbound-resolve hook: these words just flipped to ``done``, so any
        # pending sense-relation edge pointing AT them can now be reconciled.
        # Coverage grows with traffic — no scheduler needed. Best-effort: a WSD
        # failure must never fail an already-persisted generation.
        await self._resolve_inbound([w.id for w in words])
        return result

    # --- WSD relation resolution (Phase 4) --------------------------------

    async def resolve_relations(self, batch_size: int = 20) -> list[BatchResult]:
        """Reconcile one batch of pending sense-relation half-edges (manual/backfill).

        Lifts up to ``batch_size`` derived-``pending`` edges whose target word is
        ``done`` with senses, POS-filters each edge's candidate target senses,
        LM-judges them in ONE batched prompt, and applies the verdicts under
        per-edge savepoints with conditional writes ([F6]/[F7]). ``batch_size`` is
        clamped to a hard ceiling ([F9]). Returns one :class:`BatchResult` per
        edge (``value`` = derived state: ``resolved``/``unresolvable``/``noop``).

        Complements the automatic inbound hook in :meth:`_run_generation` — this
        is for words done BEFORE the feature existed, or a hook that was skipped.
        """
        return await self._resolve_core(batch_size, word_ids=None)

    async def _resolve_inbound(self, word_ids: list[int]) -> list[BatchResult]:
        """Best-effort resolve of edges pointing at the just-generated ``word_ids``.

        Wraps :meth:`_resolve_core` and swallows every error: the generation that
        triggered this hook is already committed, so a WSD hiccup (LLM down, judge
        error) must degrade to "leave the edges pending", never propagate.
        """
        if not word_ids:
            return []
        try:
            return await self._resolve_core(len(word_ids) * self._WSD_INBOUND_FACTOR, word_ids)
        except Exception:  # noqa: BLE001 - inbound resolve is strictly best-effort
            return []

    # A generated word may be the target of many pending edges; give the inbound
    # hook headroom over the raw word count, still clamped by the hard ceiling.
    _WSD_INBOUND_FACTOR = 20

    async def _resolve_core(self, batch_size: int, word_ids: list[int] | None) -> list[BatchResult]:
        """Shared resolve engine for both the hook and the public batch API.

        Steps: clamp ``batch_size`` ([F9]) → read the pending queue → per edge,
        POS-filter candidates ([F2]) and build a :class:`WsdTask` → ONE judge call
        → validate each ``chosen_index`` against the (deterministically-ordered)
        candidate list ([F3]) → apply as conditional, per-savepoint writes ([F6]/
        [F7]). Degrades to ``[]`` when no judge is configured.
        """
        judge = self._wsd
        if judge is None:
            return []
        from lexi_ai.domain.models import ResolveDecision
        from lexi_ai.generation.schemas import WsdCandidate, WsdTask
        from lexi_ai.generation.wsd import WSD_BATCH_CEIL, pos_filtered_candidates

        capped = max(1, min(batch_size, WSD_BATCH_CEIL))
        async with self._uow() as uow:
            tasks = await uow.senses.pending_relations(capped, word_ids=word_ids)
        if not tasks:
            return []

        # Build judge tasks, remembering the POS-filtered candidate order PER edge
        # so the judge's ``chosen_index`` maps back to the exact sense it saw ([F3]).
        filtered_by_edge: dict[int, list] = {}
        wsd_tasks: list[WsdTask] = []
        for task in tasks:
            cands = pos_filtered_candidates(task.source_pos, task.candidates)
            filtered_by_edge[task.edge_id] = cands
            wsd_tasks.append(
                WsdTask(
                    rel_type=task.rel_type,
                    gloss=task.gloss,
                    source_def=task.source_def,
                    candidates=[
                        WsdCandidate(index=i, definition=c.definition) for i, c in enumerate(cands)
                    ],
                )
            )

        choices = await judge.judge(wsd_tasks)

        decisions: list[ResolveDecision] = []
        for task, choice in zip(tasks, choices, strict=True):
            cands = filtered_by_edge[task.edge_id]
            idx = choice.chosen_index
            # [F3] Never trust the model index: out-of-range / None ⇒ unresolvable.
            if idx is None or not (0 <= idx < len(cands)):
                decisions.append(ResolveDecision(task.edge_id, None, None))
            else:
                chosen = cands[idx]
                decisions.append(
                    ResolveDecision(
                        task.edge_id,
                        chosen.sense_id,
                        sense_content_hash(chosen.definition),
                    )
                )

        async with self._uow() as uow:
            outcomes = await uow.senses.apply_resolutions(decisions)
            await uow.commit()
        return [
            BatchResult(key=o.edge_id, value=o.state)
            if o.error is None
            else BatchResult(key=o.edge_id, error=o.error)
            for o in outcomes
        ]

    # --- embeddings -------------------------------------------------------

    async def _embed_words(self, word_ids: list[int]) -> int:
        """Embed the senses of the given words, best-effort. Returns count embedded."""
        return await self._embed_missing(word_ids=word_ids)

    async def _embed_missing(
        self, word_ids: list[int] | None = None, limit: int | None = None
    ) -> int:
        """Embed senses lacking a current-model vector; persist packed bytes.

        Shared by the post-generation hook (``word_ids`` set) and
        :meth:`backfill_embeddings` (``word_ids`` None). Best-effort: ANY embedding
        failure (extra missing, model load, OOM, device, a misbehaving encoder)
        embeds nothing and returns 0 — an embedding error must never fail an
        already-persisted generation.
        """
        async with self._uow() as uow:
            pending = await uow.senses.needing_embedding(
                self._embedder.model_name, word_ids=word_ids, limit=limit
            )
        if not pending:
            return 0
        texts = [self._embed_text(norm, definition) for _sid, norm, definition in pending]
        try:
            vectors = await self._embedder.embed(texts)
            if not vectors:
                return 0
            dim = len(vectors[0])
            packed = [
                (sid, pack_vector(vec))
                for (sid, _norm, _definition), vec in zip(pending, vectors, strict=True)
            ]
        except Exception:  # noqa: BLE001 - best-effort: never fail generation on embed
            return 0
        async with self._uow() as uow:
            written = await uow.senses.store_embeddings(packed, self._embedder.model_name, dim)
            await uow.commit()
        return written

    @staticmethod
    def _embed_text(norm: str, definition: str) -> str:
        """The text embedded for one sense: display headword + its definition."""
        return f"{render(norm)}: {definition}"

    async def _entry_for_key(self, key: str, result) -> Entry:
        """Return the entry for the just-generated key, or the first unit as fallback."""
        done = await self._done_ids(key)
        if done:
            return await self._to_entry(done[0])
        # The queried key didn't match any generated unit exactly (e.g. the model
        # normalized the norm differently); fall back to the first persisted unit.
        fallback = await self._resolve(match_key(result.units[0].norm))
        if not fallback:
            raise ValueError(
                f"no persisted entry found for key {key!r} or "
                f"first-unit norm {result.units[0].norm!r} — generation may have errored"
            )
        return await self._to_entry(fallback[0][0])

    def _evict_lock(self, lock_key: str, lock: asyncio.Lock) -> None:
        """Drop a per-key lock once idle, so _locks does not grow unbounded."""
        if not lock.locked() and self._locks.get(lock_key) is lock:
            del self._locks[lock_key]

    def _evict_theme_lock(self, lock_key: tuple[int, int], lock: asyncio.Lock) -> None:
        """Drop a per-(word, theme) overlay lock once idle (2.6). A DISTINCT map
        from ``_locks`` so the overlay lock never nests against the neutral
        per-key lock — no cross-lock cycle."""
        if not lock.locked() and self._theme_locks.get(lock_key) is lock:
            del self._theme_locks[lock_key]

    @staticmethod
    async def _gather_batch(items: list, fn, concurrency: int | None = None) -> list[BatchResult]:
        """Run ``fn(item)`` for every item, wrapping outcomes as order-aligned
        ``BatchResult``s. One item's exception is captured, never cancels or
        aborts the others (``asyncio.gather(..., return_exceptions=True)``).
        ``concurrency`` bounds in-flight calls via a semaphore (for LLM-backed
        batches); ``None`` runs everything at once (cheap DB-only batches)."""
        if not items:
            return []
        if concurrency is None:
            raw = await asyncio.gather(*(fn(item) for item in items), return_exceptions=True)
        else:
            sem = asyncio.Semaphore(concurrency)

            async def _guarded(item):
                async with sem:
                    return await fn(item)

            raw = await asyncio.gather(*(_guarded(item) for item in items), return_exceptions=True)
        return [
            BatchResult(key=item, error=str(r))
            if isinstance(r, Exception)
            else BatchResult(key=item, value=r)
            for item, r in zip(items, raw, strict=True)
        ]

    @staticmethod
    def _cefr_map(bundle: ReferenceBundle) -> dict[str, str]:
        """Cambridge sense_id -> cefr, for the repository's Cambridge-first rule.

        Keyed by the canonical ref form so it matches whatever the model echoes
        back as ``source_ref`` (bare ``42`` or the prompt-shown ``sense#42``).
        """
        return {
            canonical_cambridge_ref(str(s.cambridge_sense_id)): s.cefr_level
            for s in bundle.cambridge_senses
            if s.cefr_level
        }

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
