"""Level-2/3 fill-in-the-blank assessment."""

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
from lexi_ai.questions.schemas import ClozePayload
from lexi_ai.questions.scoring import grade_text_span
from lexi_ai.questions.types._shared import (
    _MCQ_OPTIONS,
    _accepted_forms,
    _blank_target,
    _shuffled_options,
)


class Cloze:
    info = QuestionTypeInfo(
        type_id="cloze",
        render_kind=RenderKind.TEXT_SPAN,
        interaction="assessment",
        difficulty_levels=frozenset({2, 3}),
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
            question = await self._build(ctx, demand.sense_id, level)
            if question is None or ctx.store is None:
                produced[key] = 0
                continue
            await ctx.store.insert(question)
            produced[key] = 1
        return PrepareReport(produced)

    async def _build(
        self, ctx: QuestionContext, sense_id: int, level: int
    ) -> PersistedQuestion | None:
        entry = ctx.entry
        if entry is None:
            return None
        sense = next((item for item in entry.senses if item.sense_id == sense_id), None)
        if sense is None:
            return None
        blanked = next(
            (value for example in sense.examples if (value := _blank_target(example, entry))),
            None,
        )
        if blanked is None:
            return None
        word_bank: list[str] = []
        if level == 2:
            distractors = await ctx.distractors.for_word(entry, k=_MCQ_OPTIONS - 1, pos=sense.pos)
            word_bank, _ = _shuffled_options(
                entry.display,
                distractors,
                f"cloze:{match_key(entry.norm)}:{level}",
            )
        payload = ClozePayload(
            stem_with_blank=blanked,
            answer_norm=entry.norm,
            accepted_forms=_accepted_forms(sense),
            word_bank=word_bank,
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
        return await grade_text_span(persisted, submission)


register(Cloze)
