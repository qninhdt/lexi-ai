"""Pure dispatcher over registered question types."""

from __future__ import annotations

import logging
from collections.abc import Sequence

from lexi_ai.contracts.questions import (
    AnswerSubmission,
    Evaluation,
    PresentedQuestion,
    QuestionTypeInfo,
)
from lexi_ai.domain.questions import PersistedQuestion
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
    SenseEntryLoader,
    TtsPort,
    UnknownQuestionType,
)
from lexi_ai.questions.distractors import DistractorProvider
from lexi_ai.questions.render import to_presented
from lexi_ai.read_models import Entry

logger = logging.getLogger(__name__)


class QuestionEngine:
    """Prepare, retrieve, and evaluate questions through registered plugins.

    The engine speaks the answer-safe contract on its public seams: retrieval
    returns :class:`PresentedQuestion` (no answer) via the projection layer, and
    grading returns a typed :class:`Evaluation` whose ``reveal`` is the sanctioned
    disclosure. It still carries the judge LLM; a provider-free reader engine
    (``judge_llm=None``) cannot grade rubric/free-text types — that outcome stays
    ``pending`` rather than silently succeeding.
    """

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

    def question_types(self) -> list[QuestionTypeInfo]:
        """Return the capability info for every registered type in registry order."""
        return [question_type.info for question_type in REGISTRY.values()]

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
                and demand.difficulty_level in question_type.info.difficulty_levels
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
                    question_type.info.type_id,
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
    ) -> PresentedQuestion | None:
        """Retrieve one exact stored assessment as an answer-free presentation."""
        question_type = self._question_type(type_id)
        if question_type.info.interaction != "assessment":
            raise NotAssessable(None)
        persisted = await question_type.retrieve(
            self._ctx(None),
            QuestionQuery(sense_id, difficulty_level, excluded_ids),
        )
        return to_presented(persisted) if persisted is not None else None

    async def retrieve_exposure(self, sense_id: int) -> PresentedQuestion:
        """Build a non-null level-0 exposure card through the loader seam."""
        question_type = self._question_type("flashcard")
        persisted = await question_type.retrieve(
            self._ctx(None), QuestionQuery(sense_id, 0, frozenset())
        )
        return to_presented(persisted)

    async def evaluate(
        self, persisted: PersistedQuestion, submission: AnswerSubmission
    ) -> Evaluation:
        """Grade an assessment, rejecting exposure before plugin dispatch."""
        if persisted.interaction == "exposure":
            raise NotAssessable(persisted.question_id)
        question_type = self._question_type(persisted.type_id)
        grade = getattr(question_type, "grade", None)
        if grade is None:
            raise NotAssessable(persisted.question_id)
        return await grade(self._ctx(None), persisted, submission)

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
