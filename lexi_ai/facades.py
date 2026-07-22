"""Narrow public capability facades over :class:`lexi_ai.api.Lexicon`."""

from __future__ import annotations

from typing import TYPE_CHECKING

from lexi_ai.config import Settings

if TYPE_CHECKING:
    from lexi_ai.api import Lexicon
    from lexi_ai.read_models import Asset, Entry, Question, Score, SearchResult, SenseView


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

    async def get_question(self, question_id: int) -> Question | None:
        return await self._lexicon.get_question(question_id)

    async def get_questions_for_sense(
        self, sense_id: int, fmt: str | None = None
    ) -> list[Question]:
        return await self._lexicon.list_questions_for_sense(sense_id, fmt)


class LexiconEngine:
    """Provider-enabled generation, question, translation, and TTS operations."""

    def __init__(self, lexicon: Lexicon):
        self._lexicon = lexicon

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> LexiconEngine:
        from lexi_ai.api import Lexicon

        return cls(Lexicon.from_settings(settings))

    async def generate(
        self, source: SearchResult | str, *, structured_method: str | None = None
    ) -> Entry:
        return await self._lexicon.generate(source, structured_method=structured_method)

    async def generate_fenced(
        self, source: SearchResult | str, *, structured_method: str | None = None
    ) -> Entry:
        return await self._lexicon.generate_fenced(source, structured_method=structured_method)

    async def generate_questions_for_sense(
        self, word_id: int, sense_id: int, formats: list[str], count: int
    ) -> list[Question]:
        return await self._lexicon.generate_questions_for_sense(word_id, sense_id, formats, count)

    async def grade_question(self, question_id: int, answer: object) -> Score | None:
        return await self._lexicon.grade_question(question_id, answer)

    async def translate_field(self, source_kind: str, source_id: int, lang: str) -> str:
        return await self._lexicon.translate_field(source_kind, source_id, lang)

    async def tts_field(
        self, source_kind: str, source_id: int, voice: str | None = None, fmt: str | None = None
    ) -> Asset:
        return await self._lexicon.tts_field(source_kind, source_id, voice, fmt)
