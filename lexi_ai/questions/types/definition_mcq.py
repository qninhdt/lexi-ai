"""Level-1 direct definition recognition."""

from collections.abc import Sequence

from lexi_ai.contracts.questions import (
    AnswerSubmission,
    Evaluation,
    QuestionTypeInfo,
    RenderKind,
)
from lexi_ai.domain.questions import PersistedQuestion
from lexi_ai.normalize import match_key
from lexi_ai.questions.base import (
    PrepareReport,
    QuestionContext,
    QuestionDemand,
    QuestionQuery,
    register,
)
from lexi_ai.questions.scoring import grade_single_choice
from lexi_ai.questions.types._shared import _MCQ_OPTIONS, _mcq_question


class DefinitionMCQ:
    info = QuestionTypeInfo(
        type_id="definition_mcq",
        render_kind=RenderKind.SINGLE_CHOICE,
        interaction="assessment",
        difficulty_levels=frozenset({1}),
    )

    async def prepare(
        self, ctx: QuestionContext, demands: Sequence[QuestionDemand]
    ) -> PrepareReport:
        produced: dict[tuple[int, int], int] = {}
        for demand in demands:
            if demand.difficulty_level != 1 or demand.expected_count <= 0:
                continue
            key = (demand.sense_id, 1)
            question = await self._build(ctx, demand.sense_id)
            if question is None or ctx.store is None:
                produced[key] = 0
                continue
            await ctx.store.insert(question)
            produced[key] = 1
        return PrepareReport(produced)

    async def _build(self, ctx: QuestionContext, sense_id: int) -> PersistedQuestion | None:
        entry = ctx.entry
        if entry is None:
            return None
        sense = next((item for item in entry.senses if item.sense_id == sense_id), None)
        if sense is None:
            return None
        distractors = await ctx.distractors.for_word(
            entry, k=_MCQ_OPTIONS - 1, pos=sense.pos
        )
        return _mcq_question(
            entry,
            sense,
            f"Which word means: {sense.definition}",
            f"definition_mcq:{match_key(entry.norm)}:1",
            distractors,
            type_id=self.info.type_id,
            difficulty_level=1,
        )

    async def retrieve(
        self, ctx: QuestionContext, query: QuestionQuery
    ) -> PersistedQuestion | None:
        if query.difficulty_level != 1 or ctx.store is None:
            return None
        return await ctx.store.retrieve_one(
            query.sense_id,
            query.difficulty_level,
            self.info.type_id,
            query.excluded_question_ids,
        )

    async def grade(
        self,
        ctx: QuestionContext,
        persisted: PersistedQuestion,
        submission: AnswerSubmission,
    ) -> Evaluation:
        return await grade_single_choice(persisted, submission)


register(DefinitionMCQ)
