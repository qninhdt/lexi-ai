"""The Python library entry point; this is not an HTTP API or scheduler."""

import asyncio
import logging

from sqlalchemy.ext.asyncio import AsyncSession

from .cache import Cache
from .db.session import Database, SessionDatabase
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
from .vocab import QuestionType, ResponseFormat, TargetPlacement
from .words.generate import generate_word
from .words.search import Search
from .words.storage import get_sense_previews, get_senses, get_word


class Lexicon:
    """A library, not a scheduler. Callers coordinate overlapping calls, retries and batches.

    Full Question artifacts include answers. Consumers decide whether to deliver
    saved keys for self-learning or conceal them for an examination.
    """

    def __init__(
        self,
        db_url: str | None = None,
        cambridge_path: str = "",
        *,
        session: AsyncSession | None = None,
        decision_config: DecisionConfig | None = None,
        db_schema: str | None = None,
        llm: StructuredLLM | None = None,
        decision_model: DecisionModel | None = None,
        llm_config: LLMConfig | None = None,
        decision_fallback_model: str | None = None,
        content_cache_bytes: int = 32 * 1024 * 1024,
        question_cache_bytes: int = 32 * 1024 * 1024,
        search_cache_bytes: int = 4 * 1024 * 1024,
        refresh_seconds: float = 30,
    ):
        if (db_url is None) == (session is None):
            raise ValueError("Lexicon requires exactly one of db_url or session")
        if session is not None and db_schema is not None:
            raise ValueError("configure the database schema on the host session")
        if llm is None and (
            llm_config is None or not llm_config.api_key or not llm_config.api_key.strip()
        ):
            raise MissingProviderError(
                "Lexicon requires an LLM configuration with credentials or llm"
            )
        if session is not None:
            self.db = SessionDatabase(session)
        else:
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
        if min(content_cache_bytes, question_cache_bytes, search_cache_bytes, refresh_seconds) <= 0:
            raise ValueError("cache budgets and refresh interval must be positive")
        if session is None:
            self.db.content_cache = Cache(content_cache_bytes, ttl=refresh_seconds)
            self.db.question_cache = Cache(question_cache_bytes, ttl=refresh_seconds)
        self._search = Search(
            self.db,
            self.cambridge if cambridge_path else None,
            search_cache_bytes,
            refresh_seconds=refresh_seconds,
        )
        self.db.search_index = self._search
        self._refresh_seconds = refresh_seconds
        self._refresh_task = None

    async def start(self):
        """Preload search and own bounded-delay reconciliation across processes."""
        self._open()
        if isinstance(self.db, SessionDatabase):
            raise ValueError("a borrowed transaction cannot own a background refresh")
        await self._search.reload(include_available=self._search.cambridge is not None)
        if self._refresh_task is None:
            self._refresh_task = asyncio.create_task(self._reconcile())

    def _expire_content(self):
        for name in ("content_cache", "question_cache"):
            cache = getattr(self.db, name)
            if cache is not None:
                cache.clear()

    async def _reconcile(self):
        while True:
            await asyncio.sleep(self._refresh_seconds)
            try:
                await self._search.reload(
                    include_available=self._search.reference is not None, force=False
                )
            except Exception:
                logging.getLogger(__name__).exception("Lexical search reconciliation failed")

    async def _refresh_search(self):
        self._search.snapshot = None
        self._search.results.clear()
        # Publication committed already. A refresh failure is not a rollback;
        # lazy reads/background reconciliation retry and never reuse old results.
        try:
            await self._search.reload(include_available=self._search.reference is not None)
        except Exception:
            logging.getLogger(__name__).exception("Published search refresh failed")

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
        return await self._search.search(query, include_available)

    async def generate(
        self,
        available_id: str,
        theme: str | None = None,
        *,
        target: str,
        example_count: int = 5,
        with_usage: bool = False,
    ):
        """Generate/reuse a selected entry, neutral first.

        target is the selected lexical item, supplied explicitly by the caller.
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
                self.db,
                self.cambridge,
                llm,
                available_id,
                example_count,
                target=target,
                theme_key=theme,
            )
            await self._refresh_search()
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

    async def get_sense_previews(self, ids: list[int]):
        self._open()
        return await get_sense_previews(self.db, ids)

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
        question_type: QuestionType,
        count: int,
        *,
        distractor_count: int,
        theme: str | None = None,
        target_placement: TargetPlacement | None = None,
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

    async def get_questions(self, question_ids: list[int]):
        """Retrieve selected saved Question artifacts in identity order."""
        self._open()
        return await question_rows.get_many(self.db, question_ids)

    async def retrieve_questions(
        self, requests: list[tuple[int, QuestionType, int]], *, theme: str | None = None
    ):
        """Random artifacts for explicit (Sense, type, quantity) bank allocations.

        Raises QuestionBankChangedError if any bank cannot fulfill its quantity.
        Type selection and allocation belong to the caller.
        """
        self._open()
        return await question_rows.retrieve_many(self.db, requests, theme_key=theme)

    async def list_questions(
        self,
        sense_id: int,
        question_type: QuestionType | None = None,
        *,
        theme: str | None = None,
        after_id: int | None = None,
        limit: int | None = None,
    ):
        self._open()
        return await question_rows.list_for_sense(
            self.db, sense_id, question_type, theme_key=theme, after_id=after_id, limit=limit
        )

    async def list_questions_for_senses(
        self,
        sense_ids: list[int],
        question_types: list[QuestionType] | None = None,
        *,
        theme: str | None = None,
        limit_per_type: int = 8,
    ):
        self._open()
        return await question_rows.list_for_senses(
            self.db, sense_ids, question_types, theme_key=theme, limit_per_type=limit_per_type
        )

    async def count_questions_for_senses(
        self,
        sense_ids: list[int],
        question_types: list[QuestionType] | None = None,
        *,
        theme: str | None = None,
    ) -> dict[tuple[int, QuestionType], int]:
        """Count saved banks by Sense/type without retrieving Question payloads."""
        self._open()
        return await question_rows.count_for_senses(
            self.db, sense_ids, question_types, theme_key=theme
        )

    async def retrieve_question(
        self,
        sense_id: int,
        question_type: QuestionType | None = None,
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
        fmt: ResponseFormat,
        answer: str,
        *,
        mode: DecisionMode = DecisionMode.LLM_FALLBACK,
        with_usage: bool = False,
        allowed_pairs: set[tuple[QuestionType, ResponseFormat]] | None = None,
    ):
        self._open()
        fmt = ResponseFormat(fmt)
        with UsageRecorder(with_usage) as usage:
            decision = self._decision_model()
            model = None if fmt is ResponseFormat.SINGLE_CHOICE else usage.wrap(decision)
            grade = await grade_answer(
                self.db,
                model,
                question_id,
                fmt,
                answer,
                config=self.decision_config,
                mode=mode,
                allowed_pairs=allowed_pairs,
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
                self.db, usage.wrap(self._decision_model()), batch_size, mode=mode
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
        if self._refresh_task is not None:
            self._refresh_task.cancel()
            try:
                await self._refresh_task
            except asyncio.CancelledError:
                pass
        self._expire_content()
        self._search.snapshot = None
        self._search.reference = None
        self._search.results.clear()
        self.db.search_index = None
        await self.db.close()
        if self._owned_llm:
            await self.llm.close()
        if self._owned_decision:
            await self.decision_model.close()
