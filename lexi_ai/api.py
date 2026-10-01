"""The Python library entry point; this is not an HTTP API or scheduler."""

from .db.session import Database
from .inference.config import DecisionConfig, LLMConfig
from .inference.decision import DecisionModel
from .inference.llm import OpenAIStructuredLLM
from .questions import storage as question_rows
from .questions.generate import generate_questions
from .questions.grade import grade_answer
from .references.cambridge import Cambridge
from .relations.resolve import resolve_relations
from .themes.service import (
    create_theme,
    delete_theme,
    ensure_word_theme,
    get_theme,
    list_themes,
    update_theme,
)
from .translation import storage as translation_rows
from .translation.generate import translate_text
from .words.generate import generate_word
from .words.search import search
from .words.storage import get_senses, get_word


class Lexicon:
    """A library, not a scheduler. Callers coordinate overlapping calls, retries and batches.

    Full Question artifacts include answers and are for trusted consumer servers only.
    Never forward their correct option/explanations to learners before grading.
    """

    def __init__(
        self,
        db_url: str,
        cambridge_path: str,
        *,
        decision_config: DecisionConfig,
        db_schema: str | None = None,
        llm=None,
        decision_model=None,
        llm_config: LLMConfig | None = None,
        decision_fallback_model: str | None = None,
    ):
        self.db = Database(db_url, schema=db_schema)
        self.cambridge = Cambridge(cambridge_path)
        self.llm = llm
        self.decision_model = decision_model
        self.llm_config = llm_config or LLMConfig()
        self.decision_fallback_model = decision_fallback_model
        self.decision_config = decision_config
        self._owned_llm = False
        self._owned_decision = False
        self._closed = False

    def _open(self):
        if self._closed:
            raise RuntimeError("Lexicon is closed")

    def _llm(self):
        if self.llm is None:
            self.llm = OpenAIStructuredLLM(self.llm_config)
            self._owned_llm = True
        return self.llm

    def _decision_model(self):
        if self.decision_model is None:
            self.decision_model = DecisionModel(
                self.decision_config,
                llm_config=self.llm_config,
                fallback_model=self.decision_fallback_model,
            )
            self._owned_decision = True
        return self.decision_model

    async def search(self, query: str, include_available: bool = False):
        self._open()
        return await search(self.db, self.cambridge, query, include_available)

    async def generate(
        self, available_id: str, theme: str | None = None, *, example_count: int = 5
    ):
        """Generate/reuse a selected entry, neutral first.

        example_count applies to new examples per Sense in each created namespace.
        Saved content is reused regardless of the requested count. The caller
        serializes overlapping requests on the same Word.
        """
        self._open()
        if type(example_count) is not int or example_count < 1:
            raise ValueError("example count must be a positive integer")
        word_id = await generate_word(
            self.db, self.cambridge, self._llm(), available_id, example_count, theme_key=theme
        )
        if theme is not None:
            return await ensure_word_theme(self.db, self._llm(), word_id, theme, example_count)
        return await get_word(self.db, word_id)

    async def get_word(self, word_id: int, theme: str | None = None):
        self._open()
        return await get_word(self.db, word_id, theme_key=theme)

    async def get_senses(self, ids: list[int]):
        self._open()
        return await get_senses(self.db, ids)

    async def create_theme(self, key: str, name: str, concept: str):
        self._open()
        return await create_theme(self.db, self._llm(), key, name, concept)

    async def get_theme(self, key: str):
        self._open()
        return await get_theme(self.db, key)

    async def list_themes(self, *, after_key: str | None = None, limit: int | None = None):
        self._open()
        return await list_themes(self.db, after_key=after_key, limit=limit)

    async def update_theme(self, key: str, *, name=None, voice=None, diction=None):
        self._open()
        return await update_theme(self.db, key, name=name, voice=voice, diction=diction)

    async def delete_theme(self, key: str):
        self._open()
        return await delete_theme(self.db, key)

    async def generate_questions(
        self,
        sense_id: int,
        question_type: str,
        count: int,
        *,
        distractor_count: int,
        theme: str | None = None,
        target_placement: str | None = None,
    ):
        self._open()
        return await generate_questions(
            self.db,
            self._llm(),
            sense_id,
            question_type,
            count,
            distractor_count=distractor_count,
            theme_key=theme,
            target_placement=target_placement,
        )

    async def get_question(self, question_id: int):
        self._open()
        return await question_rows.get(self.db, question_id)

    async def list_questions(
        self,
        sense_id: int,
        question_type: str | None = None,
        *,
        theme: str | None = None,
        after_id: int | None = None,
        limit: int | None = None,
    ):
        self._open()
        return await question_rows.list_for_sense(
            self.db, sense_id, question_type, theme_key=theme, after_id=after_id, limit=limit
        )

    async def retrieve_question(
        self,
        sense_id: int,
        question_type: str | None = None,
        *,
        theme: str | None = None,
    ):
        self._open()
        return await question_rows.retrieve(self.db, sense_id, question_type, theme_key=theme)

    async def delete_question(self, question_id: int):
        self._open()
        return await question_rows.remove(self.db, question_id)

    async def grade_answer(self, question_id: int, fmt: str, answer: str):
        self._open()
        if fmt == "single_choice":
            return await grade_answer(
                self.db, None, question_id, fmt, answer, config=self.decision_config
            )
        return await grade_answer(
            self.db,
            self._decision_model(),
            question_id,
            fmt,
            answer,
            config=self.decision_config,
        )

    async def resolve_relations(self, batch_size: int = 20):
        self._open()
        return await resolve_relations(self.db, self._decision_model(), batch_size)

    async def translate_text(self, content: str, target_language: str):
        self._open()
        return await translate_text(self.db, self._llm(), content, target_language)

    async def get_translation(self, identifier: int):
        self._open()
        return await translation_rows.get(self.db, identifier)

    async def list_translations(self, *, after_id: int | None = None, limit: int | None = None):
        self._open()
        return await translation_rows.list_rows(self.db, after_id=after_id, limit=limit)

    async def delete_translation(self, identifier: int):
        self._open()
        return await translation_rows.remove(self.db, identifier)

    async def purge_translations(self):
        self._open()
        return await translation_rows.purge(self.db)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self.db.close()
        if self._owned_llm:
            await self.llm.close()
        if self._owned_decision:
            await self.decision_model.close()
