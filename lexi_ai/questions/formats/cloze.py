"""Level-2/3 fill-in-the-blank assessment."""

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
from lexi_ai.questions.formats._shared import (
    _MCQ_OPTIONS,
    _accepted_forms,
    _blank_target,
    _shuffled_options,
)
from lexi_ai.questions.schemas import ClozePayload
from lexi_ai.questions.scoring import grade_text_span
from lexi_ai.read_models import Evaluation, Question


class Cloze:
    descriptor = QuestionTypeDescriptor(
        type_id="cloze",
        render_format="text_span",
        supported_levels=frozenset({2, 3}),
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
            question = await self._build(ctx, demand.sense_id, level)
            if question is None or ctx.store is None:
                produced[key] = 0
                continue
            await ctx.store.insert(question)
            produced[key] = 1
        return PrepareReport(produced)

    async def _build(
        self, ctx: QuestionContext, sense_id: int, level: int
    ) -> Question | None:
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
            distractors = await ctx.distractors.for_word(
                entry, k=_MCQ_OPTIONS - 1, pos=sense.pos
            )
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
        return await grade_text_span(question, answer)


register(Cloze)
