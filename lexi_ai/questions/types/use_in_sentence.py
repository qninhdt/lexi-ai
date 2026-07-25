"""Level-3/4 free-production assessment."""

from collections.abc import Sequence

from lexi_ai.contracts.questions import (
    AnswerSubmission,
    Evaluation,
    QuestionTypeInfo,
    RenderKind,
)
from lexi_ai.domain.questions import PersistedQuestion
from lexi_ai.questions.base import (
    PrepareReport,
    QuestionContext,
    QuestionDemand,
    QuestionQuery,
    register,
)
from lexi_ai.questions.schemas import UseInSentencePayload
from lexi_ai.questions.scoring import grade_rubric


class UseInSentence:
    info = QuestionTypeInfo(
        type_id="use_in_sentence",
        render_kind=RenderKind.FREE_TEXT,
        interaction="assessment",
        difficulty_levels=frozenset({3, 4}),
    )

    async def prepare(
        self, ctx: QuestionContext, demands: Sequence[QuestionDemand]
    ) -> PrepareReport:
        produced: dict[tuple[int, int], int] = {}
        for demand in demands:
            level = demand.difficulty_level
            if level not in self.info.difficulty_levels or demand.expected_count <= 0:
                continue
            key = (demand.sense_id, level)
            question = self._build(ctx, demand.sense_id, level)
            if question is None or ctx.store is None:
                produced[key] = 0
                continue
            await ctx.store.insert(question)
            produced[key] = 1
        return PrepareReport(produced)

    def _build(self, ctx: QuestionContext, sense_id: int, level: int) -> PersistedQuestion | None:
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
        return PersistedQuestion(
            question_id=None,
            word_id=entry.word_id,
            sense_id=sense.sense_id,
            type_id=self.info.type_id,
            render_kind=self.info.render_kind,
            difficulty_level=level,
            interaction=self.info.interaction,
            payload=payload.model_dump(),
        )

    async def retrieve(
        self, ctx: QuestionContext, query: QuestionQuery
    ) -> PersistedQuestion | None:
        if query.difficulty_level not in self.info.difficulty_levels or ctx.store is None:
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
        return await grade_rubric(persisted, submission, judge=ctx.judge)


register(UseInSentence)
