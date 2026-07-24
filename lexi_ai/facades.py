"""Narrow public capability facades over :class:`lexi_ai.api.Lexicon`."""

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
        Entry,
        SearchResult,
        SenseView,
    )


class LexiconReader:
    """Provider-free dictionary and persisted-question reads."""

    def __init__(self, lexicon: Lexicon):
        self._lexicon = lexicon

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> LexiconReader:
        from lexi_ai.api import Lexicon

        return cls(Lexicon.from_settings(settings))

    async def search(self, query: str) -> list[SearchResult]:
        return await self._lexicon.search(query)

    async def get_entry(self, word_id: int) -> Entry:
        return await self._lexicon.get_entry(word_id)

    async def get_senses(self, sense_ids: list[int]) -> list[SenseView]:
        return await self._lexicon.get_senses(sense_ids)

    def question_types(self) -> list[QuestionTypeInfo]:
        return self._lexicon.reader_questions.question_types()

    async def get_question(self, question_id: int) -> PresentedQuestion | None:
        return await self._lexicon.get_question(question_id)

    async def list_questions_for_sense(
        self, sense_id: int, type_id: str | None = None
    ) -> list[PresentedQuestion]:
        return await self._lexicon.list_questions_for_sense(sense_id, type_id)

    async def retrieve_question(
        self,
        sense_id: int,
        difficulty_level: int,
        excluded_ids: frozenset[int],
        type_id: str,
    ) -> PresentedQuestion | None:
        return await self._lexicon.reader_questions.retrieve(
            sense_id, difficulty_level, excluded_ids, type_id
        )

    async def retrieve_exposure(self, sense_id: int) -> PresentedQuestion:
        return await self._lexicon.reader_questions.retrieve_exposure(sense_id)

    async def evaluate_answer(
        self, question_id: int, submission: AnswerSubmission
    ) -> Evaluation | None:
        return await self._lexicon._evaluate_answer(
            self._lexicon.reader_questions, question_id, submission
        )


class LexiconEngine:
    """Provider-enabled generation, question, translation, and TTS operations."""

    def __init__(self, lexicon: Lexicon):
        self._lexicon = lexicon

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> LexiconEngine:
        from lexi_ai.api import Lexicon

        return cls(Lexicon.from_settings(settings))

    def question_types(self) -> list[QuestionTypeInfo]:
        return self._lexicon.worker_questions.question_types()

    async def prepare_questions(
        self, word_id: int, demands: Sequence[PrepareDemand]
    ) -> PrepareReport:
        return await self._lexicon.prepare_questions(word_id, list(demands))

    async def evaluate_answer(
        self, question_id: int, submission: AnswerSubmission
    ) -> Evaluation | None:
        return await self._lexicon.evaluate_answer(question_id, submission)

    async def generate(
        self, source: SearchResult | str, *, structured_method: str | None = None
    ) -> Entry:
        return await self._lexicon.generate(source, structured_method=structured_method)

    async def generate_fenced(
        self, source: SearchResult | str, *, structured_method: str | None = None
    ) -> Entry:
        return await self._lexicon.generate_fenced(source, structured_method=structured_method)

    async def translate_field(self, source_kind: str, source_id: int, lang: str) -> str:
        return await self._lexicon.translate_field(source_kind, source_id, lang)

    async def tts_field(
        self, source_kind: str, source_id: int, voice: str | None = None, fmt: str | None = None
    ) -> Asset:
        return await self._lexicon.tts_field(source_kind, source_id, voice, fmt)
