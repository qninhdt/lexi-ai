"""Level-1 direct definition recognition."""

from collections.abc import Sequence

from lexi_ai.normalize import match_key
from lexi_ai.questions.base import (
    PrepareReport,
    QuestionContext,
    QuestionDemand,
    QuestionQuery,
    QuestionTypeDescriptor,
    register,
)
from lexi_ai.questions.formats._shared import _MCQ_OPTIONS, _mcq_question
from lexi_ai.questions.scoring import grade_single_choice
from lexi_ai.read_models import Evaluation, Question


class DefinitionMCQ:
    descriptor = QuestionTypeDescriptor(
        type_id="definition_mcq",
        render_format="single_choice",
        supported_levels=frozenset({1}),
        interaction_mode="assessment",
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

    async def _build(self, ctx: QuestionContext, sense_id: int) -> Question | None:
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
            type_id=self.descriptor.type_id,
            difficulty_level=1,
        )

    async def retrieve(
        self, ctx: QuestionContext, query: QuestionQuery
    ) -> Question | None:
        if query.difficulty_level != 1 or ctx.store is None:
            return None
        return await ctx.store.retrieve_one(
            query.sense_id,
            query.difficulty_level,
            self.descriptor.type_id,
            query.excluded_question_ids,
        )

    async def evaluate(
        self, ctx: QuestionContext, question: Question, answer: object
    ) -> Evaluation:
        return await grade_single_choice(question, answer)


register(DefinitionMCQ)
