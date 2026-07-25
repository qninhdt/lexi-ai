"""Preparing, retrieving, and grading vocabulary questions.

The service is constructed with a question engine, and the engine it gets decides
what the service can do. That is deliberate rather than incidental: a reader
process builds a provider-free engine, and a worker builds one with the language
model and the rubric judge.

Collapsing the two into a single provider-free path would silently break rubric
grading — `use_in_sentence` scores by asking the judge, so without one the grade
would degrade rather than fail. The composition root decides which engine each
facade receives; this service does not choose.
"""

from typing import TYPE_CHECKING

from lexi_ai.contracts.questions import (
    AnswerSubmission,
    Evaluation,
    PrepareDemand,
    PresentedQuestion,
    QuestionTypeInfo,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from lexi_ai.questions.base import PrepareReport
    from lexi_ai.questions.engine import QuestionEngine
    from lexi_ai.questions.repository import QuestionRepository
    from lexi_ai.read_models import Entry


def _to_internal_demands(demands: list[PrepareDemand]) -> list:
    """Map public demands (string sense id) onto the engine's internal form."""
    from lexi_ai.questions.base import QuestionDemand

    return [
        QuestionDemand(
            sense_id=int(demand.sense_id),
            difficulty_level=demand.difficulty_level,
            expected_count=demand.expected_count,
        )
        for demand in demands
    ]


class QuestionService:
    """Question use cases over one engine and the question store."""

    def __init__(
        self,
        engine: "QuestionEngine",
        repository: "QuestionRepository",
        load_entry: "Callable[[int], object]",
    ) -> None:
        self._engine = engine
        self._repository = repository
        # Preparing questions needs the whole entry; taking a loader keeps this
        # service off the dictionary service and out of an import cycle.
        self._load_entry = load_entry

    def question_types(self) -> list[QuestionTypeInfo]:
        return self._engine.question_types()

    async def prepare(self, word_id: int, demands: list[PrepareDemand]) -> "PrepareReport":
        entry: Entry = await self._load_entry(word_id)
        return await self._engine.prepare(entry, _to_internal_demands(demands))

    async def get(self, question_id: int) -> PresentedQuestion | None:
        from lexi_ai.questions.render import to_presented

        persisted = await self._repository.get(question_id)
        return to_presented(persisted) if persisted is not None else None

    async def list_for_sense(
        self, sense_id: int, type_id: str | None = None
    ) -> list[PresentedQuestion]:
        from lexi_ai.questions.render import to_presented

        rows = await self._repository.list_for_sense(sense_id, type_id)
        return [to_presented(row) for row in rows]

    async def retrieve(
        self,
        sense_id: int,
        difficulty_level: int,
        excluded_ids: frozenset[int],
        type_id: str,
    ) -> PresentedQuestion | None:
        return await self._engine.retrieve(sense_id, difficulty_level, excluded_ids, type_id)

    async def retrieve_exposure(self, sense_id: int) -> PresentedQuestion:
        return await self._engine.retrieve_exposure(sense_id)

    async def evaluate(self, question_id: int, submission: AnswerSubmission) -> Evaluation | None:
        """Grade a submission, or report a miss for an unknown question."""
        persisted = await self._repository.get(question_id)
        if persisted is None:
            return None
        return await self._engine.evaluate(persisted, submission)
