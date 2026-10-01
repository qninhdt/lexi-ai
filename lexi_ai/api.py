"""The Python library entry point; this is not an HTTP API or scheduler."""

from .db.session import Database
from .errors import MissingProviderError
from .inference.config import DecisionConfig, DecisionMode, LLMConfig
from .inference.decision import DecisionModel
from .inference.llm import OpenAIStructuredLLM, StructuredLLM
from .inference.usage import UsageRecorder
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
        decision_config: DecisionConfig | None = None,
        db_schema: str | None = None,
        llm: StructuredLLM | None = None,
        decision_model: DecisionModel | None = None,
        llm_config: LLMConfig | None = None,
        decision_fallback_model: str | None = None,
    ):
        if llm is None and (
            llm_config is None or not llm_config.api_key or not llm_config.api_key.strip()
        ):
            raise MissingProviderError(
                "Lexicon requires an LLM configuration with credentials or llm"
            )
        self.db = Database(db_url, schema=db_schema)
        self.cambridge = Cambridge(cambridge_path)
        self.llm = llm
        self.decision_model = decision_model
        self.llm_config = llm_config or LLMConfig()
        self.decision_fallback_model = decision_fallback_model
        self.decision_config = decision_config or DecisionConfig(0.8)
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

    def _decision_model(self, mode=DecisionMode.LLM_FALLBACK):
        mode = DecisionMode(mode)
        if (
            mode == DecisionMode.DECISION_ONLY
            and (self.decision_model is None or self._owned_decision)
            and not (self.decision_config.api_key and self.decision_config.api_key.strip())
        ):
            raise MissingProviderError("decision_only requires a configured decision provider")
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
        self,
        available_id: str,
        theme: str | None = None,
        *,
        example_count: int = 5,
        with_usage: bool = False,
    ):
        """Generate/reuse a selected entry, neutral first.

        example_count applies to new examples per Sense in each created namespace.
        Saved content is reused regardless of the requested count. The caller
        serializes overlapping requests on the same Word.
        """
        self._open()
        with UsageRecorder(with_usage) as usage:
            if type(example_count) is not int or example_count < 1:
                raise ValueError("example count must be a positive integer")
            llm = usage.wrap(self._llm())
            word_id = await generate_word(
                self.db, self.cambridge, llm, available_id, example_count, theme_key=theme
            )
            word = (
                await ensure_word_theme(self.db, llm, word_id, theme, example_count)
                if theme is not None
                else await get_word(self.db, word_id)
            )
            return usage.finish(word)

    async def get_word(self, word_id: int, theme: str | None = None):
        self._open()
        return await get_word(self.db, word_id, theme_key=theme)

    async def get_senses(self, ids: list[int]):
        self._open()
        return await get_senses(self.db, ids)

    async def create_theme(self, key: str, name: str, concept: str, *, with_usage: bool = False):
        self._open()
        with UsageRecorder(with_usage) as usage:
            theme = await create_theme(self.db, usage.wrap(self._llm()), key, name, concept)
            return usage.finish(theme)

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
        with_usage: bool = False,
    ):
        self._open()
        with UsageRecorder(with_usage) as usage:
            questions = await generate_questions(
                self.db,
                usage.wrap(self._llm()),
                sense_id,
                question_type,
                count,
                distractor_count=distractor_count,
                theme_key=theme,
                target_placement=target_placement,
            )
            return usage.finish(questions)

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

    async def grade_answer(
        self,
        question_id: int,
        fmt: str,
        answer: str,
        *,
        mode: DecisionMode = DecisionMode.LLM_FALLBACK,
        with_usage: bool = False,
    ):
        self._open()
        with UsageRecorder(with_usage) as usage:
            decision = self._decision_model(mode)
            model = None if fmt == "single_choice" else usage.wrap(decision)
            grade = await grade_answer(
                self.db, model, question_id, fmt, answer, config=self.decision_config, mode=mode
            )
            return usage.finish(grade)

    async def resolve_relations(
        self,
        batch_size: int = 20,
        *,
        mode: DecisionMode = DecisionMode.LLM_FALLBACK,
        with_usage: bool = False,
    ):
        self._open()
        with UsageRecorder(with_usage) as usage:
            results = await resolve_relations(
                self.db, usage.wrap(self._decision_model(mode)), batch_size, mode=mode
            )
            return usage.finish(results)

    async def translate_text(self, content: str, target_language: str, *, with_usage: bool = False):
        self._open()
        with UsageRecorder(with_usage) as usage:
            text = await translate_text(self.db, usage.wrap(self._llm()), content, target_language)
            return usage.finish(text)

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
