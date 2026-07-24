"""Level-3/4 free-production assessment."""

from collections.abc import Sequence

from lexi_ai.questions.base import (
    PrepareReport,
    QuestionContext,
    QuestionDemand,
    QuestionQuery,
    QuestionTypeDescriptor,
    register,
)
from lexi_ai.questions.schemas import UseInSentencePayload
from lexi_ai.questions.scoring import grade_rubric
from lexi_ai.read_models import Evaluation, Question


class UseInSentence:
    descriptor = QuestionTypeDescriptor(
        type_id="use_in_sentence",
        render_format="free_text",
        supported_levels=frozenset({3, 4}),
        interaction_mode="assessment",
    )

    async def prepare(
        self, ctx: QuestionContext, demands: Sequence[QuestionDemand]
    ) -> PrepareReport:
        produced: dict[tuple[int, int], int] = {}
        for demand in demands:
            level = demand.difficulty_level
            if level not in self.descriptor.supported_levels or demand.expected_count <= 0:
                continue
            key = (demand.sense_id, level)
            question = self._build(ctx, demand.sense_id, level)
            if question is None or ctx.store is None:
                produced[key] = 0
                continue
            await ctx.store.insert(question)
            produced[key] = 1
        return PrepareReport(produced)

    def _build(
        self, ctx: QuestionContext, sense_id: int, level: int
    ) -> Question | None:
        entry = ctx.entry
        if entry is None:
            return None
        sense = next((item for item in entry.senses if item.sense_id == sense_id), None)
        if sense is None:
            return None
        if level == 3:
            prompt = (
                f"Write a sentence using '{entry.display}' to mean: "
                f"{sense.definition}. Include at least six words."
            )
            rubric = (
                f"Sentence must use '{entry.display}' with the sense "
                f"'{sense.definition}'; grammatical; at least 6 words."
            )
        else:
            prompt = f"Use '{entry.display}' naturally in an original sentence."
            rubric = (
                f"Sentence must use '{entry.display}' accurately and naturally "
                f"with the sense '{sense.definition}'."
            )
        payload = UseInSentencePayload(
            prompt=prompt,
            target_norm=entry.norm,
            rubric=rubric,
        )
        return Question(
            question_id=None,
            word_id=entry.word_id,
            sense_id=sense.sense_id,
            type_id=self.descriptor.type_id,
            render_format=self.descriptor.render_format,
            difficulty_level=level,
            interaction_mode=self.descriptor.interaction_mode,
            payload=payload.model_dump(),
        )

    async def retrieve(
        self, ctx: QuestionContext, query: QuestionQuery
    ) -> Question | None:
        if query.difficulty_level not in self.descriptor.supported_levels or ctx.store is None:
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
        return await grade_rubric(question, answer, judge=ctx.judge)


register(UseInSentence)
