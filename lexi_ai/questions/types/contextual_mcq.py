"""Level-1 direct and level-2 contextual multiple choice."""

from collections.abc import Sequence

from lexi_ai.contracts.questions import (
    AnswerSubmission,
    Evaluation,
    QuestionTypeInfo,
    RenderKind,
)
from lexi_ai.domain.questions import PersistedQuestion
from lexi_ai.llm import ainvoke_structured, guarded_messages
from lexi_ai.normalize import match_key
from lexi_ai.prompts import PromptLoader
from lexi_ai.questions.base import (
    PrepareReport,
    QuestionContext,
    QuestionDemand,
    QuestionQuery,
    register,
)
from lexi_ai.questions.dedup import DistractorDedup
from lexi_ai.questions.schemas import GeneratedMCQ
from lexi_ai.questions.scoring import grade_single_choice
from lexi_ai.questions.types._shared import (
    _CONTEXTUAL_SYSTEM,
    _MCQ_OPTIONS,
    _mcq_question,
)
from lexi_ai.read_models import Entry, SenseView


class ContextualMCQ:
    info = QuestionTypeInfo(
        type_id="contextual_mcq",
        render_kind=RenderKind.SINGLE_CHOICE,
        interaction="assessment",
        difficulty_levels=frozenset({1, 2}),
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
        if level == 1:
            stem = f"Which word means: {sense.definition}"
            distractors = await ctx.distractors.for_word(
                entry, k=_MCQ_OPTIONS - 1, pos=sense.pos
            )
        else:
            if ctx.llm is None:
                return None
            generated = await self._generate_context(ctx, entry, sense)
            stem = generated.stem
            distractors = await self._merge_distractors(ctx, entry, sense, generated)
        return _mcq_question(
            entry,
            sense,
            stem,
            f"contextual_mcq:{match_key(entry.norm)}:{level}",
            distractors,
            type_id=self.info.type_id,
            difficulty_level=level,
        )

    @staticmethod
    async def _generate_context(
        ctx: QuestionContext, entry: Entry, sense: SenseView
    ) -> GeneratedMCQ:
        human = PromptLoader.render(
            "contextual_mcq_user",
            word=entry.display,
            definition=sense.definition,
        )
        return await ainvoke_structured(
            ctx.llm,
            guarded_messages(_CONTEXTUAL_SYSTEM, human),
            GeneratedMCQ,
        )

    @staticmethod
    async def _merge_distractors(
        ctx: QuestionContext, entry: Entry, sense: SenseView, generated: GeneratedMCQ
    ) -> list[str]:
        dedup = DistractorDedup(entry)
        wanted = _MCQ_OPTIONS - 1
        for candidate in generated.distractors:
            if len(dedup.items) >= wanted:
                break
            dedup.take(candidate)
        if len(dedup.items) < wanted:
            for candidate in await ctx.distractors.for_word(
                entry, k=wanted, pos=sense.pos
            ):
                if dedup.take(candidate) and len(dedup.items) >= wanted:
                    break
        return dedup.items[:wanted]

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
        return await grade_single_choice(persisted, submission)


register(ContextualMCQ)
