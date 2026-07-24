"""Pure dispatcher over registered question types."""

from __future__ import annotations

import logging
from collections.abc import Sequence

from lexi_ai.llm import StructuredLLM
from lexi_ai.questions.base import (
    REGISTRY,
    NotAssessable,
    PrepareReport,
    QuestionContext,
    QuestionDemand,
    QuestionQuery,
    QuestionStore,
    QuestionType,
    QuestionTypeDescriptor,
    SenseEntryLoader,
    TtsPort,
    UnknownQuestionType,
)
from lexi_ai.questions.distractors import DistractorProvider
from lexi_ai.read_models import Entry, Evaluation, Question

logger = logging.getLogger(__name__)


class QuestionEngine:
    """Prepare, retrieve, and evaluate questions through registered plugins."""

    def __init__(
        self,
        repo: QuestionStore,
        distractors: DistractorProvider,
        llm: StructuredLLM | None = None,
        judge_llm: StructuredLLM | None = None,
        tts: TtsPort | None = None,
        sense_loader: SenseEntryLoader | None = None,
    ):
        self._repo = repo
        self._distractors = distractors
        self._llm = llm
        self._judge = judge_llm
        self._tts = tts
        self._sense_loader = sense_loader

    def question_types(self) -> list[QuestionTypeDescriptor]:
        """Return descriptors for every registered type in registry order."""
        return [question_type.descriptor for question_type in REGISTRY.values()]

    async def prepare(
        self, entry: Entry, demands: Sequence[QuestionDemand]
    ) -> PrepareReport:
        """Best-effort preparation across assessment types.

        A plugin failure is logged and contributes zero rather than blocking
        preparation by other types. Counts aggregate when multiple types supply
        the same ``(sense_id, difficulty_level)`` demand.
        """
        produced: dict[tuple[int, int], int] = {}
        for question_type in REGISTRY.values():
            prepare = getattr(question_type, "prepare", None)
            if prepare is None:
                continue
            relevant = [
                demand
                for demand in demands
                if demand.expected_count > 0
                and demand.difficulty_level in question_type.descriptor.supported_levels
            ]
            if not relevant:
                continue
            for demand in relevant:
                produced.setdefault((demand.sense_id, demand.difficulty_level), 0)
            try:
                report = await prepare(self._ctx(entry), relevant)
            except Exception:
                logger.warning(
                    "question preparation failed for type %s",
                    question_type.descriptor.type_id,
                    exc_info=True,
                )
                continue
            for key, count in report.produced.items():
                produced[key] = produced.get(key, 0) + count
        return PrepareReport(produced)

    async def retrieve(
        self,
        sense_id: int,
        difficulty_level: int,
        excluded_ids: frozenset[int],
        type_id: str,
    ) -> Question | None:
        """Retrieve one exact stored assessment; never generate or fall back."""
        question_type = self._question_type(type_id)
        if question_type.descriptor.interaction_mode != "assessment":
            raise NotAssessable(None)
        return await question_type.retrieve(
            self._ctx(None),
            QuestionQuery(sense_id, difficulty_level, excluded_ids),
        )

    async def retrieve_exposure(self, sense_id: int) -> Question:
        """Build a non-null level-0 exposure card through the loader seam."""
        question_type = self._question_type("flashcard")
        return await question_type.retrieve(
            self._ctx(None), QuestionQuery(sense_id, 0, frozenset())
        )

    async def evaluate(self, question: Question, answer: object) -> Evaluation:
        """Evaluate an assessment, rejecting exposure before plugin dispatch."""
        if question.interaction_mode == "exposure":
            raise NotAssessable(question.question_id)
        question_type = self._question_type(question.type_id)
        evaluate = getattr(question_type, "evaluate", None)
        if evaluate is None:
            raise NotAssessable(question.question_id)
        return await evaluate(self._ctx(None), question, answer)

    def _ctx(self, entry: Entry | None) -> QuestionContext:
        return QuestionContext(
            entry=entry,
            distractors=self._distractors,
            llm=self._llm,
            judge=self._judge,
            store=self._repo,
            tts=self._tts,
            sense_loader=self._sense_loader,
        )

    @staticmethod
    def _question_type(type_id: str) -> QuestionType:
        try:
            return REGISTRY[type_id]
        except KeyError as exc:
            raise UnknownQuestionType(type_id) from exc


__all__ = ["NotAssessable", "QuestionEngine", "UnknownQuestionType"]
