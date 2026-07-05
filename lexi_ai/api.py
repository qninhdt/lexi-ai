"""Lazy lookup API (Phase 6): the public ``Lexicon.get`` surface.

Flow: normalize input -> resolve by match_key against words AND aliases ->
branch on 0 / 1-done / 1-pending / N. Misses run the full lazy pipeline
(reference -> generate -> persist) exactly once per key, guarded by a per-key
asyncio lock plus a DB double-check (library, single-process — decision #18).

``display`` is always ``render(norm)``; no display column is ever read.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import selectinload

from lexi_ai.assets.repository import AssetRepository, content_hash, normalize_asset_params
from lexi_ai.config import Settings, get_settings
from lexi_ai.constants import TIER_ORDER, canonical_cambridge_ref
from lexi_ai.db import create_engine, create_session_factory, init_models, session_scope
from lexi_ai.embeddings import Embedder
from lexi_ai.generation.generator import Generator
from lexi_ai.models import EntryLink, Sense, Word, WordAlias, WordTag
from lexi_ai.normalize import match_key, render, tag_key
from lexi_ai.normalize import theme_key as _norm_theme_key
from lexi_ai.persistence.repository import Repository
from lexi_ai.read_models import (
    AliasView,
    Asset,
    Entry,
    LinkView,
    ReferenceView,
    SearchResult,
    SemanticHit,
    SenseView,
    TagCount,
    Theme,
    TopicView,
)
from lexi_ai.references.cambridge import CambridgeSource
from lexi_ai.references.loader import ReferenceBundle, ReferenceLoader
from lexi_ai.references.wordnet import WordNetSource
from lexi_ai.vectors import cosine, pack_vector, unpack_vector

if TYPE_CHECKING:
    from lexi_ai.questions.engine import QuestionEngine


class Lexicon:
    """Lazy-generation dictionary. Construct with :meth:`from_settings`."""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        loader: ReferenceLoader,
        generator: Generator,
        repository: Repository,
        engine: AsyncEngine | None = None,
        embedder: Embedder | None = None,
        assets: AssetRepository | None = None,
    ):
        self._session_factory = session_factory
        self._loader = loader
        self._generator = generator
        self._repo = repository
        self._engine = engine
        self._embedder = embedder or Embedder()
        self._assets = assets
        self._locks: dict[str, asyncio.Lock] = {}
        # Lazy questions engine (built on first access; see the `questions` property).
        self._questions: QuestionEngine | None = None

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> Lexicon:
        settings = settings or get_settings()
        engine = create_engine(settings)
        session_factory = create_session_factory(engine)
        loader = ReferenceLoader(CambridgeSource(settings.cambridge_db_path), WordNetSource())
        generator = Generator(settings=settings)
        repository = Repository(session_factory)
        embedder = Embedder(settings=settings)
        assets = AssetRepository(session_factory, settings.asset_cache_dir)
        return cls(
            session_factory,
            loader,
            generator,
            repository,
            engine=engine,
            embedder=embedder,
            assets=assets,
        )

    async def init(self) -> None:
        """Create the generated-DB schema (idempotent)."""
        engine = self._engine or self._session_factory.kw["bind"]
        await init_models(engine)

    @property
    def questions(self) -> QuestionEngine:
        """The question engine (generate/list/get/delete/grade). Built lazily.

        Constructing a :class:`Lexicon` costs nothing extra until this is first
        accessed. The two llm runnables (contextual-MCQ generator, rubric judge)
        are built from settings the same way generation builds its model; a caller
        wanting fakes constructs the engine directly instead of via this property.
        """
        if self._questions is None:
            from lexi_ai.questions.distractors import DistractorProvider
            from lexi_ai.questions.engine import QuestionEngine
            from lexi_ai.questions.repository import QuestionRepository

            self._questions = QuestionEngine(
                QuestionRepository(self._session_factory),
                DistractorProvider(self._repo, self._embedder),
                llm=self._build_questions_llm(),
                judge_llm=self._build_judge_llm(),
            )
        return self._questions

    def _build_questions_llm(self) -> object | None:
        """ChatOpenAI bound to ``GeneratedMCQ`` for the contextual-MCQ plugin."""
        from lexi_ai.questions.schemas import GeneratedMCQ

        return self._build_structured(GeneratedMCQ)

    def _build_judge_llm(self) -> object | None:
        """ChatOpenAI bound to ``Judgment`` for the rubric scorer."""
        from lexi_ai.questions.schemas import Judgment

        return self._build_structured(Judgment)

    def _build_structured(self, schema: type) -> object | None:
        """Build a structured-output runnable bound to ``schema`` from settings.

        Returns ``None`` when no LLM is configured (empty api key), so the
        llm-dependent formats degrade gracefully instead of failing at import.
        Imported lazily so constructing a Lexicon needs no langchain/creds.
        """
        settings = get_settings()
        if not settings.llm_api_key:
            return None
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            model=settings.llm_model,
            temperature=settings.llm_temperature,
        )
        return llm.with_structured_output(schema)

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

    async def get(self, lexi_word_id: int, theme_key: str | None = None) -> Entry:
        """Load a generated entry by its dictionary id. Never generates (FREE).

        ``theme_key`` (the normalized key string from ``create_theme``/
        ``list_themes`` — NOT a name or object) overlays themed definition +
        examples where a themed row exists, falling back to neutral per-sense.
        An unknown ``theme_key`` raises ``ValueError`` (silently returning neutral
        would hide a caller bug). ``None`` (default) is the neutral entry unchanged.
        """
        if theme_key is None:
            return await self._to_entry(lexi_word_id)
        resolved = await self._repo.resolve_theme(_norm_theme_key(theme_key))
        if resolved is None:
            raise ValueError(f"unknown theme: {theme_key!r}")
        theme_id, _style = resolved
        return await self._to_entry(lexi_word_id, theme_id=theme_id)

    async def get_senses(self, sense_ids: list[int]) -> list[SenseView]:
        """Batch-resolve senses by their DB ids. Never generates (FREE).

        Returns a SenseView per found id, preserving the input order. Ids with no
        row are silently skipped (caller tolerates missing senses). Relationships
        are eager-loaded inside the session so views survive after it closes.
        """
        if not sense_ids:
            return []
        async with session_scope(self._session_factory) as session:
            rows = (
                await session.execute(
                    select(Sense)
                    .options(
                        selectinload(Sense.references),
                        selectinload(Sense.examples),
                        selectinload(Sense.collocations),
                    )
                    .where(Sense.id.in_(sense_ids))
                )
            ).scalars()
            by_id = {s.id: self._build_sense_view(s) for s in rows}
        return [by_id[sid] for sid in sense_ids if sid in by_id]

    @staticmethod
    def _build_sense_view(s: Sense) -> SenseView:
        return SenseView(
            definition=s.definition,
            tier=s.tier,
            pos=s.pos,
            cefr_level=s.cefr_level,
            examples=[e.text for e in sorted(s.examples, key=lambda e: e.example_order)],
            references=[
                ReferenceView(source=r.source, source_ref=r.source_ref) for r in s.references
            ],
            guideword=s.guideword,
            grammar=s.grammar.split(",") if s.grammar else [],
            register=s.register,
            connotation=s.connotation,
            collocations=[
                c.text for c in sorted(s.collocations, key=lambda c: c.collocation_order)
            ],
            sense_id=s.id,
        )

    async def status(self, lexi_word_id: int) -> str | None:
        """Status of a dictionary word (``done`` | ``pending`` | ``error``), or
        ``None`` if no such id exists. Never generates (FREE)."""
        async with session_scope(self._session_factory) as session:
            row = await session.execute(select(Word.status).where(Word.id == lexi_word_id))
            return row.scalar_one_or_none()

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
        rows = await self._repo.embedded_senses(self._embedder.model_name)
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
        rows = await self._repo.count_tags()
        return [TagCount(name=name, title=title, count=count) for name, title, count in rows]

    async def words_by_tag(self, tag: str, *, limit: int | None = None) -> list[SearchResult]:
        """Generated words carrying ``tag``, as generated-hit ``SearchResult``s.

        The query is resolved via ``tag_key`` (the write-path function) so
        ``"Business"``/``"business"``/``"cars"`` all hit the right tag. Never
        generates (FREE); pass a hit's ``lexi_word_id`` to :meth:`get`.
        """
        rows = await self._repo.words_for_tag_key(tag_key(tag), limit=limit)
        return [
            SearchResult(display=render(norm), entry_type=etype, lexi_word_id=wid)
            for wid, norm, etype in rows
        ]

    async def create_theme(self, name: str, style_prompt: str) -> Theme:
        """Create (or resolve) a style theme by its normalized ``theme_key``.

        Re-creating a name that normalizes to an existing key returns that theme
        (first-seen ``name``/``style_prompt`` win). An empty-key name raises
        ``ValueError``. Never calls an LLM.
        """
        theme = await self._repo.create_theme(name, style_prompt)
        return Theme(key=theme.theme_key, name=theme.name, style_prompt=theme.style_prompt)

    async def list_themes(self) -> list[Theme]:
        """Every style theme, name-sorted. Never generates (FREE)."""
        return [
            Theme(key=t.theme_key, name=t.name, style_prompt=t.style_prompt)
            for t in await self._repo.list_themes()
        ]

    async def generate_theme(self, lexi_word_id: int, theme_key: str) -> Entry:
        """Restyle a done word's senses into a theme's voice (one LLM call).

        ``theme_key`` is the normalized key string (from ``create_theme``/
        ``list_themes``) — NOT a name or object. An unknown key raises
        ``ValueError``. The word must be ``done`` (mirrors the questions-engine
        guard). Writes one themed_senses row per neutral sense + fresh in-voice
        examples; re-running overwrites in place. Returns the themed entry (overlay).
        """
        resolved = await self._repo.resolve_theme(_norm_theme_key(theme_key))
        if resolved is None:
            raise ValueError(f"unknown theme: {theme_key!r}")
        theme_id, style_prompt = resolved

        status = await self.status(lexi_word_id)
        if status != "done":
            raise ValueError(f"word {lexi_word_id} is not done (status={status!r})")

        neutral = await self._repo.senses_for_theming(lexi_word_id)
        if not neutral:
            raise ValueError(f"word {lexi_word_id} has no senses to theme")
        sense_ids = [row[0] for row in neutral]
        facts = [(d, pos, gw, tier) for _sid, d, pos, gw, tier in neutral]

        generator = self._themed_generator()
        if generator is None:
            raise ValueError("no LLM configured for themed generation")
        result = await generator.generate(style_prompt, facts)
        await self._repo.persist_themed(theme_id, result, sense_ids)
        return await self._to_entry(lexi_word_id, theme_id=theme_id)

    def _themed_generator(self):
        """Lazy themed generator; ``None`` when no LLM is configured (empty key).

        An injected generator (``self._themed_gen`` pre-set, e.g. a test fake) is
        used as-is, bypassing the api-key gate.
        """
        if getattr(self, "_themed_gen", None) is not None:
            return self._themed_gen
        settings = get_settings()
        if not settings.llm_api_key:
            return None
        from lexi_ai.theming.generator import ThemedGenerator

        self._themed_gen = ThemedGenerator(settings=settings)
        return self._themed_gen

    # --- cached assets ----------------------------------------------------

    async def translate(self, text: str, lang: str) -> str:
        """Translate ``text`` into ``lang``, cache-first (content-addressed).

        A second call for the same ``(text, lang)`` spends ZERO LLM (cache hit).
        Themed and neutral text hash differently, so each gets its own asset;
        identical text dedups. Empty/whitespace text returns as-is (no LLM, no
        row). Raises ``ValueError`` when no LLM is configured.
        """
        if not text.strip():
            return text
        assets = self._require_assets()
        params = normalize_asset_params("translate", lang=lang)
        h = content_hash(text)
        cached = await assets.get(h, "translate", params)
        if cached is not None and cached.text_value is not None:
            return cached.text_value
        translator = self._translator()
        if translator is None:
            raise ValueError("no LLM configured for translation")
        result = await translator.translate(text, lang)
        stored = await assets.put_text(h, "translate", params, result)
        return stored.text_value or result

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

    async def tts(self, text: str, voice: str | None = None, fmt: str | None = None) -> Asset:
        """Synthesize speech for ``text``, cache-first (content-addressed).

        A cache hit returns the ``Asset`` WITHOUT calling the provider. On a miss
        the provider is invoked — this round it is a STUB that raises
        ``NotImplementedError`` (no fake audio is cached, so the miss leaves no
        row/file). Empty/whitespace text short-circuits. When a real provider is
        configured, its bytes are stored via ``put_file`` and the ``Asset`` returned.
        """
        settings = get_settings()
        voice = voice if voice is not None else settings.tts_voice
        fmt = fmt if fmt is not None else settings.tts_format
        if not text.strip():
            return Asset(kind="tts", params=normalize_asset_params("tts", voice=voice, fmt=fmt))
        assets = self._require_assets()
        params = normalize_asset_params("tts", voice=voice, fmt=fmt)
        h = content_hash(text)
        cached = await assets.get(h, "tts", params)
        if cached is not None:
            return cached
        provider = self._tts_provider()
        data = await provider.synthesize(text, voice, fmt)  # stub raises here
        return await assets.put_file(h, "tts", params, data, ext=fmt)

    def _tts_provider(self):
        """Lazy TTS provider; defaults to the stub until a real one is injected."""
        if getattr(self, "_tts_impl", None) is None:
            from lexi_ai.assets.tts import StubTTSProvider

            self._tts_impl = StubTTSProvider()
        return self._tts_impl

    async def generate(self, source: SearchResult | str, *, force: bool = False) -> Entry:
        """Generate (or return) the entry for a search result or a custom string.

        * ``SearchResult`` — anchored to its Cambridge reference. If already
          generated, returns the existing entry (no LLM) unless ``force``.
        * ``str`` — a custom word Cambridge lacks; anchored to WordNet only.

        A suggestion whose word already exists converges on that entry instead of
        duplicating it. With ``force=True`` the entry is regenerated and overwritten
        in place.
        """
        # Custom string: anchor to WordNet only, dedup by the string's match_key.
        if isinstance(source, str):
            return await self._generate_locked(match_key(source), source, None, force)

        # Already-generated hit: return it, or re-anchor to its Cambridge id on force.
        if source.lexi_word_id is not None:
            if not force:
                return await self._to_entry(source.lexi_word_id)
            norm, cam_id = await self._word_norm_and_cambridge(source.lexi_word_id)
            return await self._generate_locked(match_key(norm), norm, cam_id, True)

        # Suggestion: cache-check by Cambridge provenance (robust to display/norm
        # key differences), then generate anchored to that Cambridge id.
        if source.cambridge_id is None:
            raise ValueError("SearchResult has neither lexi_word_id nor cambridge_id")
        if not force:
            hit = await self._generated_by_cambridge([source.cambridge_id])
            if source.cambridge_id in hit:
                return await self._to_entry(hit[source.cambridge_id][0])
        return await self._generate_locked(
            match_key(source.display), source.display, source.cambridge_id, force
        )

    async def _generate_locked(
        self, key: str, word: str, cambridge_id: int | None, force: bool
    ) -> Entry:
        """Locked generate-and-persist for one word key (double-checked)."""
        lock = self._locks.setdefault(key, asyncio.Lock())
        try:
            async with lock:
                if not force:
                    done = await self._done_ids(key)
                    if done:
                        return await self._to_entry(done[0])
                result = await self._run_generation(word, cambridge_id)
        finally:
            self._evict_lock(key, lock)
        return await self._entry_for_key(key, result)

    async def _word_norm_and_cambridge(self, lexi_word_id: int) -> tuple[str, int | None]:
        async with session_scope(self._session_factory) as session:
            row = (
                await session.execute(
                    select(Word.norm, Word.cambridge_word_id).where(Word.id == lexi_word_id)
                )
            ).one()
            return row[0], row[1]

    async def _resolve(self, key: str) -> list[tuple[int, str]]:
        """Return (word_id, status) for every word matching key via headword or alias."""
        async with session_scope(self._session_factory) as session:
            direct = await session.execute(
                select(Word.id, Word.status).where(Word.match_key == key)
            )
            found: dict[int, str] = {wid: status for wid, status in direct}
            via_alias = await session.execute(
                select(Word.id, Word.status)
                .join(WordAlias, WordAlias.word_id == Word.id)
                .where(WordAlias.alias_match_key == key)
            )
            for wid, status in via_alias:
                found.setdefault(wid, status)
            return list(found.items())

    async def _done_ids(self, key: str) -> list[int]:
        """word_ids with a ``done`` entry for this match_key (headword or alias)."""
        return [wid for wid, status in await self._resolve(key) if status == "done"]

    async def _generated_by_cambridge(
        self, cambridge_ids: list[int]
    ) -> dict[int, tuple[int, str, str | None]]:
        """For each Cambridge id already generated (``done``), its (lexi id, norm,
        type). Keyed by ``cambridge_word_id`` provenance — robust to display/norm
        key differences. First ``done`` row per id wins."""
        if not cambridge_ids:
            return {}
        async with session_scope(self._session_factory) as session:
            rows = await session.execute(
                select(Word.cambridge_word_id, Word.id, Word.norm, Word.entry_type).where(
                    Word.cambridge_word_id.in_(cambridge_ids), Word.status == "done"
                )
            )
            out: dict[int, tuple[int, str, str | None]] = {}
            for cam_id, wid, norm, entry_type in rows:
                if cam_id is not None:
                    out.setdefault(cam_id, (wid, render(norm), entry_type))
            return out

    # --- generation path --------------------------------------------------

    async def _run_generation(self, word: str, cambridge_id: int | None):
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
            existing_tags = await self._repo.all_tags()
        except Exception:  # noqa: BLE001 - vocab is best-effort; empty on failure
            existing_tags = []
        result = await self._generator.generate(bundle, existing_tags=existing_tags)
        words = await self._repo.persist_result(
            result, cambridge_word_id=cambridge_id, cambridge_cefr=cefr_map
        )
        await self._embed_words([w.id for w in words])
        return result

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
        pending = await self._repo.senses_needing_embedding(
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
        return await self._repo.store_embeddings(packed, self._embedder.model_name, dim)

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
        return await self._to_entry(fallback[0][0])

    def _evict_lock(self, lock_key: str, lock: asyncio.Lock) -> None:
        """Drop a per-key lock once idle, so _locks does not grow unbounded."""
        if not lock.locked() and self._locks.get(lock_key) is lock:
            del self._locks[lock_key]

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
        themed_map = (
            await self._repo.themed_for_word(word_id, theme_id) if theme_id is not None else None
        )
        async with session_scope(self._session_factory) as session:
            word = (
                await session.execute(
                    select(Word)
                    .options(
                        selectinload(Word.senses).selectinload(Sense.references),
                        selectinload(Word.senses).selectinload(Sense.examples),
                        selectinload(Word.senses).selectinload(Sense.collocations),
                        selectinload(Word.aliases),
                        selectinload(Word.links_out).selectinload(EntryLink.to_word),
                        selectinload(Word.tags).selectinload(WordTag.tag),
                    )
                    .where(Word.id == word_id)
                )
            ).scalar_one()
            return self._build_entry(word, themed_map=themed_map)

    @staticmethod
    def _build_entry(
        word: Word, themed_map: dict[int, tuple[str, list[str]]] | None = None
    ) -> Entry:
        senses = sorted(word.senses, key=lambda s: (TIER_ORDER.get(s.tier, 99), s.sense_order))
        overlay = themed_map or {}
        return Entry(
            display=render(word.norm),
            norm=word.norm,
            entry_type=word.entry_type,
            pos=word.pos,
            status=word.status,
            word_id=word.id,
            senses=[
                SenseView(
                    # Overlay themed def+examples per-sense where a themed row
                    # exists; neutral fallback otherwise. All OTHER fields stay
                    # neutral — themes cover definition + examples only this round.
                    definition=overlay[s.id][0] if s.id in overlay else s.definition,
                    tier=s.tier,
                    pos=s.pos,
                    cefr_level=s.cefr_level,
                    examples=(
                        overlay[s.id][1]
                        if s.id in overlay
                        else [e.text for e in sorted(s.examples, key=lambda e: e.example_order)]
                    ),
                    references=[
                        ReferenceView(source=r.source, source_ref=r.source_ref)
                        for r in s.references
                    ],
                    guideword=s.guideword,
                    # Stored comma-joined (a join of validated tokens or None);
                    # split back to a list, None -> [].
                    grammar=s.grammar.split(",") if s.grammar else [],
                    register=s.register,
                    connotation=s.connotation,
                    collocations=[
                        c.text for c in sorted(s.collocations, key=lambda c: c.collocation_order)
                    ],
                    sense_id=s.id,
                )
                for s in senses
            ],
            aliases=[
                AliasView(
                    display=render(a.alias_norm),
                    alias_norm=a.alias_norm,
                    type=a.type,
                    dialect=a.dialect,
                )
                for a in word.aliases
            ],
            # word_family / confused_with word-references surface HERE, via their
            # rel_type — no dedicated field. They ride the normalized links_out
            # path like synonyms; grouping by rel_type is a consumer concern.
            links=[
                LinkView(
                    display=render(link.to_word.norm),
                    norm=link.to_word.norm,
                    rel_type=link.rel_type,
                )
                for link in word.links_out
            ],
            topics=[
                TopicView(name=wt.tag.name, title=wt.tag.title)
                for wt in sorted(word.tags, key=lambda wt: wt.tag.name)
            ],
        )
