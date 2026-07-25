"""The work surface: everything that mutates a row or calls a provider.

The split from :class:`~lexi_ai.facades.reader.LexiconReader` is a capability
boundary, not a convenience one. Holding this facade means holding the ability to
spend a model call and to change the dictionary, so a deployment that only serves
reads should never construct it.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from lexi_ai.config import Settings

if TYPE_CHECKING:
    from lexi_ai.api import Lexicon
    from lexi_ai.contracts.questions import (
        AnswerSubmission,
        Evaluation,
        PrepareDemand,
        PresentedQuestion,
        QuestionTypeInfo,
    )
    from lexi_ai.questions.base import PrepareReport
    from lexi_ai.read_models import (
        Asset,
        BatchResult,
        Entry,
        SearchResult,
        SenseView,
        Theme,
    )


class LexiconEngine:
    """Provider-enabled generation, enrichment, curation, translation, and speech."""

    def __init__(self, lexicon: Lexicon):
        self._lexicon = lexicon

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> LexiconEngine:
        """Build an engine over its own object graph, from configuration."""
        from lexi_ai.api import Lexicon

        return cls(Lexicon.from_settings(settings))

    async def init(self) -> None:
        """Create the generated-DB schema (idempotent)."""
        await self._lexicon.init()

    async def close(self) -> None:
        """Release the database engine behind this engine."""
        await self._lexicon.close()

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
        return await self._lexicon.generation().generate(
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

        Every item goes through the same path, so two inputs resolving to one word
        still generate exactly once — the per-``match_key`` lock and DB double-check
        are reused rather than reimplemented here.
        """
        return await self._lexicon.generation().generate_many(
            sources, force=force, theme=theme, concurrency=concurrency
        )

    async def generate_fenced(
        self, source: SearchResult | str, *, structured_method: str | None = None
    ) -> Entry:
        """Generate once under a database fence, for independently deployed workers.

        Deliberately has no ``force``: a remote caller must not be able to use a
        delayed job to replace an entry that a newer claim owns.
        """
        return await self._lexicon.generation().generate_fenced(
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
        Embeddings are untouched (they cover the definition only).

        ``theme`` augments the sense's themed overlay instead of its neutral
        examples; that overlay must already exist (see :meth:`generate` with
        ``theme=``). An unknown sense, missing overlay, or absent LLM raises
        ``ValueError``.
        """
        return await self._lexicon.enrichment().add_examples(sense_id, n, theme)

    async def backfill_embeddings(self, *, limit: int | None = None) -> int:
        """Embed done senses lacking a current-model vector; returns the count.

        Fills gaps left by best-effort generation (extra missing at the time) or by
        an embedding-model change. Idempotent, no LLM, and returns 0 rather than
        failing when the embeddings extra is unavailable.
        """
        return await self._lexicon.enrichment().backfill_embeddings(limit=limit)

    async def resolve_relations(self, batch_size: int = 20) -> list[BatchResult]:
        """Reconcile one batch of pending sense-relation edges (manual backfill)."""
        return await self._lexicon.enrichment().resolve_relations(batch_size)

    # --- curation ---------------------------------------------------------

    async def delete_entry(self, word_id: int) -> bool:
        """Delete a word and all its content; returns whether a row was removed.
        Senses, aliases, links, tags, and questions cascade at the FK level."""
        return await self._lexicon.dictionary().delete_entry(word_id)

    async def rename_tag(
        self, tag: str, *, name: str | None = None, title: str | None = None
    ) -> bool:
        """Update a topic tag's display fields. The dedup key is immutable, so this
        never merges or re-keys (see :meth:`merge_tags`). Returns whether it existed."""
        return await self._lexicon.tags().rename(tag, name=name, title=title)

    async def delete_tag(self, tag: str) -> bool:
        """Delete a topic tag; returns whether one was found. Tagged words are
        untouched — they simply lose this one topic."""
        return await self._lexicon.tags().delete(tag)

    async def merge_tags(self, sources: list[str], into: str) -> int:
        """Fold ``sources`` into ``into``, then delete the sources. ``into`` must
        already exist. Returns the number of word-tag associations re-pointed."""
        return await self._lexicon.tags().merge(sources, into)

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
        return await self._lexicon.themes().create(name, style_prompt, description, tone)

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
        return await self._lexicon.themes().update(
            key, name=name, style_prompt=style_prompt, description=description, tone=tone
        )

    async def delete_theme(self, key: str) -> bool:
        """Delete a style theme by key; returns whether one was removed. Its themed
        senses and examples cascade; neutral entries are untouched."""
        return await self._lexicon.themes().delete(key)

    # --- translation and speech -------------------------------------------

    async def translate_field(self, source_kind: str, source_id: int, lang: str) -> str:
        """Translate a source into ``lang``, cache-first over the reference store."""
        return await self._lexicon.assets().translate(source_kind, source_id, lang)

    async def translate_sense(self, sense_id: int, lang: str) -> str:
        """Translate a sense's definition — the everyday translation surface."""
        return await self.translate_field("sense_def", sense_id, lang)

    async def translate_many(
        self, refs: list[tuple[str, int]], lang: str, *, concurrency: int = 5
    ) -> list[BatchResult]:
        """Batch :meth:`translate_field`, order-aligned, cache-first per item."""
        return await self._lexicon.assets().translate_many(refs, lang, concurrency=concurrency)

    async def tts_field(
        self, source_kind: str, source_id: int, voice: str | None = None, fmt: str | None = None
    ) -> Asset:
        """Synthesize speech for a source, cache-first over the reference store."""
        return await self._lexicon.assets().speak(source_kind, source_id, voice, fmt)

    async def tts_sense(
        self, sense_id: int, voice: str | None = None, fmt: str | None = None
    ) -> Asset:
        """Synthesize a sense's definition — the everyday speech surface."""
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
        return await self._lexicon.assets().speak_many(refs, voice, fmt, concurrency=concurrency)

    async def delete_asset(self, asset_id: int) -> bool:
        """Delete one cached asset and its backing file."""
        return await self._lexicon.assets().delete(asset_id)

    async def purge_assets(self, *, kind: str | None = None) -> int:
        """Delete every cached asset, unlinking backing files."""
        return await self._lexicon.assets().purge(kind=kind)

    # --- questions --------------------------------------------------------

    def question_types(self) -> list[QuestionTypeInfo]:
        """The registered question formats, including provider-backed ones."""
        return self._lexicon.questions(providers=True).question_types()

    async def prepare_questions(
        self, word_id: int, demands: Sequence[PrepareDemand]
    ) -> PrepareReport:
        """Materialize the demanded questions for a word, using every provider."""
        return await self._lexicon.questions(providers=True).prepare(word_id, list(demands))

    async def get_question(self, question_id: int) -> PresentedQuestion | None:
        """One persisted question in presentation form, or ``None``."""
        return await self._lexicon.questions(providers=True).get(question_id)

    async def list_questions_for_sense(
        self, sense_id: int, type_id: str | None = None
    ) -> list[PresentedQuestion]:
        """Every persisted question for a sense, optionally one format only."""
        return await self._lexicon.questions(providers=True).list_for_sense(sense_id, type_id)

    async def retrieve_question(
        self,
        sense_id: int,
        difficulty_level: int,
        excluded_ids: frozenset[int],
        type_id: str,
    ) -> PresentedQuestion | None:
        """Pick one unseen question for a sense at a difficulty, or ``None``."""
        return await self._lexicon.questions(providers=True).retrieve(
            sense_id, difficulty_level, excluded_ids, type_id
        )

    async def retrieve_exposure(self, sense_id: int) -> PresentedQuestion:
        """The exposure card for a sense."""
        return await self._lexicon.questions(providers=True).retrieve_exposure(sense_id)

    async def evaluate_answer(
        self, question_id: int, submission: AnswerSubmission
    ) -> Evaluation | None:
        """Grade a submission authoritatively, with the rubric judge available."""
        return await self._lexicon.questions(providers=True).evaluate(question_id, submission)
