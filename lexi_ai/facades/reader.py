"""The free surface: reads that never mutate and never call a provider.

Everything here is safe to expose to an API process handling user traffic. No method
on this facade can spend money, block on a model round trip, or change a row, so a
reader deployment needs no LLM or TTS credentials at all.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lexi_ai.api import Lexicon
    from lexi_ai.application.questions import QuestionService
    from lexi_ai.contracts.questions import (
        AnswerSubmission,
        Evaluation,
        PresentedQuestion,
        QuestionTypeInfo,
    )
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


class LexiconReader:
    """Provider-free dictionary, asset, and persisted-question reads."""

    def __init__(self, lexicon: Lexicon):
        self._lexicon = lexicon

    async def close(self) -> None:
        """Release the database engine behind this reader."""
        await self._lexicon.close()

    # --- search -----------------------------------------------------------

    async def search(self, query: str) -> list[SearchResult]:
        """Search the dictionary for a raw string. Never generates.

        Returns one ranked list (best first) mixing two kinds of hit:

        * **generated** — a word already in the dictionary (``lexi_word_id`` set);
          pass the id to :meth:`get_entry`.
        * **suggestion** — a reference word that *can* be generated
          (``cambridge_id`` set); pass the result to ``LexiconEngine.generate``.

        A reference word whose ``match_key`` is already generated is folded into the
        generated hit, so nothing is offered for regeneration by mistake.
        """
        return await self._lexicon.lookup().search(query)

    async def semantic_search(self, query: str, k: int = 10) -> list[SemanticHit]:
        """Rank already-generated senses by meaning similarity to ``query``.

        Embeds the query locally and ranks every done sense carrying a current-model
        vector, best first. Free — only the short query is embedded, never a whole
        entry. Empty when nothing is embedded yet (e.g. the ``[embeddings]`` extra
        is not installed) or ``k <= 0``.
        """
        return await self._lexicon.lookup().semantic_search(query, k)

    # --- entries ----------------------------------------------------------

    async def get_entry(self, word_id: int, theme: str | int | None = None) -> Entry:
        """Load a generated entry by its dictionary id. Never generates.

        ``theme`` (key or id) overlays the themed definition and examples where a
        themed row exists, falling back to neutral per sense. An unknown ``theme``
        raises ``ValueError`` — returning neutral silently would hide a caller bug.
        """
        return await self._lexicon.dictionary().entry(word_id, theme)

    async def get_many(
        self, word_ids: list[int], theme: str | int | None = None
    ) -> list[BatchResult]:
        """Batch :meth:`get_entry`, one result per input id in order. A missing or
        invalid id is reported as a failed item rather than aborting the batch."""
        return await self._lexicon.dictionary().entries(word_ids, theme)

    async def get_senses(self, sense_ids: list[int]) -> list[SenseView]:
        """Batch-resolve senses by id, preserving input order. Ids with no row are
        skipped, so a caller tolerating missing senses needs no error handling."""
        return await self._lexicon.dictionary().senses(sense_ids)

    async def get_status(self, word_id: int) -> str | None:
        """Status of a word (``done`` | ``pending`` | ``error``), or ``None`` when
        no such id exists."""
        return await self._lexicon.dictionary().status(word_id)

    async def get_status_many(self, word_ids: list[int]) -> list[BatchResult]:
        """Batch :meth:`get_status`, in order. ``value`` is ``None`` for an unknown
        id — that is a valid answer, not a failure."""
        return await self._lexicon.dictionary().statuses(word_ids)

    async def list_entries(
        self, *, status: str = "done", limit: int | None = None, offset: int = 0
    ) -> list[SearchResult]:
        """Paginated browse of the whole dictionary, norm-sorted. Lightweight rows —
        pass a hit's ``lexi_word_id`` to :meth:`get_entry` for the full entry."""
        return await self._lexicon.dictionary().list_entries(
            status=status, limit=limit, offset=offset
        )

    async def list_entries_by_tag(
        self, tag: str, *, limit: int | None = None
    ) -> list[SearchResult]:
        """Generated words carrying ``tag``, as generated-hit results.

        The query is resolved through the same key function the write path uses, so
        ``"Business"``, ``"business"``, and ``"cars"`` all hit the right tag.
        """
        return await self._lexicon.dictionary().list_entries_by_tag(tag, limit=limit)

    async def list_tags(self) -> list[TagCount]:
        """Every topic tag with its live member count over ``done`` words, sorted
        count-desc then name."""
        return await self._lexicon.dictionary().list_tags()

    async def stats(self) -> Stats:
        """Read-only dictionary counts in one grouped snapshot."""
        return await self._lexicon.dictionary().stats()

    # --- themes -----------------------------------------------------------

    async def list_themes(self) -> list[Theme]:
        """Every style theme, name-sorted."""
        return await self._lexicon.themes().list_all()

    async def get_theme(self, key: str) -> Theme | None:
        """A style theme by key (a raw display name is normalized the same way the
        write path normalizes it), or ``None`` if unknown."""
        return await self._lexicon.themes().get(key)

    # --- cached assets ----------------------------------------------------

    async def get_asset(self, asset_id: int) -> Asset | None:
        """A cached asset by id, or ``None``."""
        return await self._lexicon.assets().get(asset_id)

    async def list_assets(
        self, *, kind: str | None = None, limit: int | None = None, offset: int = 0
    ) -> list[Asset]:
        """Cached assets, oldest first, optionally filtered by kind."""
        return await self._lexicon.assets().list(kind=kind, limit=limit, offset=offset)

    async def source_hash(self, source_kind: str, source_id: int) -> str | None:
        """The current content fingerprint of a translatable source, else ``None``.

        A worker fences a delayed translation job on this before calling a provider.
        """
        return await self._lexicon.assets().source_hash(source_kind, source_id)

    # --- questions --------------------------------------------------------

    def question_types(self) -> list[QuestionTypeInfo]:
        """The registered question formats visible to a reader process."""
        return self._questions.question_types()

    async def get_question(self, question_id: int) -> PresentedQuestion | None:
        """One persisted question in presentation form, or ``None``."""
        return await self._questions.get(question_id)

    async def list_questions_for_sense(
        self, sense_id: int, type_id: str | None = None
    ) -> list[PresentedQuestion]:
        """Every persisted question for a sense, optionally one format only."""
        return await self._questions.list_for_sense(sense_id, type_id)

    async def retrieve_question(
        self,
        sense_id: int,
        difficulty_level: int,
        excluded_ids: frozenset[int],
        type_id: str,
    ) -> PresentedQuestion | None:
        """Pick one unseen question for a sense at a difficulty, or ``None``."""
        return await self._questions.retrieve(sense_id, difficulty_level, excluded_ids, type_id)

    async def retrieve_exposure(self, sense_id: int) -> PresentedQuestion:
        """The provider-free exposure card for a sense (no generation involved)."""
        return await self._questions.retrieve_exposure(sense_id)

    async def evaluate_answer(
        self, question_id: int, submission: AnswerSubmission
    ) -> Evaluation | None:
        """Grade a submission with the PROVIDER-FREE engine.

        A rubric-graded type needs the judge, which this context deliberately lacks,
        so such a type degrades here rather than being scored. Use
        ``LexiconEngine.evaluate_answer`` when grading must be authoritative.
        """
        return await self._questions.evaluate(question_id, submission)

    @property
    def _questions(self) -> QuestionService:
        return self._lexicon.questions(providers=False)
