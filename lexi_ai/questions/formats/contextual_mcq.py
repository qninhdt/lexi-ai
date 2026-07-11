"""``contextual_mcq`` — LLM MCQ from a novel context; PERSISTS its output."""

from lexi_ai.llm import ainvoke_structured, guarded_messages
from lexi_ai.normalize import match_key
from lexi_ai.prompts import PromptLoader
from lexi_ai.questions.base import FormatSpec, QuestionContext, register
from lexi_ai.questions.dedup import DistractorDedup
from lexi_ai.questions.formats._shared import (
    _CONTEXTUAL_SYSTEM,
    _MCQ_OPTIONS,
    _core_sense,
    _mcq_question,
)
from lexi_ai.questions.schemas import GeneratedMCQ
from lexi_ai.questions.scoring import grade_single_choice
from lexi_ai.read_models import Entry, Question, Score, SenseView


class ContextualMCQ:
    """LLM MCQ from a novel context — PERSISTS its output via ``ctx.store``."""

    format = "contextual_mcq"
    answer_kind = "single_choice"

    async def generate(self, ctx: QuestionContext, n: int = 1) -> list[Question]:
        entry = ctx.entry
        if entry is None or n <= 0 or ctx.llm is None:
            return []  # no llm configured -> this format is unavailable, best-effort
        sense = _core_sense(entry)
        if sense is None:
            return []
        human = PromptLoader.render(
            "contextual_mcq_user",
            word=entry.display,
            definition=sense.definition,
        )
        mcq = await ainvoke_structured(
            ctx.llm,
            guarded_messages(_CONTEXTUAL_SYSTEM, human),
            GeneratedMCQ,
        )
        # We use only the llm's stem + distractors; the correct answer is always the
        # target word (entry.display), NOT the model's claimed `mcq.correct` — trusting
        # a generated answer would risk a hallucinated key. `correct` steers the model
        # to build coherent distractors, then is intentionally discarded.
        distractors = await self._merge_distractors(ctx, entry, sense, mcq)
        question_id = f"contextual_mcq:{match_key(entry.norm)}"
        q = _mcq_question(entry, sense, mcq.stem, question_id, distractors, self.format)
        if q is None:
            return []
        if ctx.store is not None:  # the plugin decides to persist; the engine does not
            q = await ctx.store.insert(q)
        return [q]

    @staticmethod
    async def _merge_distractors(
        ctx: QuestionContext, entry: Entry, sense: SenseView, mcq: GeneratedMCQ
    ) -> list[str]:
        """LLM-proposed distractors first (answer-filtered), topped up from the ladder.

        Uses the shared :class:`DistractorDedup` (3.3) so the exclude+dedup rule
        (never the answer or an alias variant, never a repeat) is enforced by the
        SAME code the ladder provider uses — not a hand-rolled second copy."""
        dedup = DistractorDedup(entry)
        want = _MCQ_OPTIONS - 1
        for cand in mcq.distractors:
            if len(dedup.items) >= want:
                break
            dedup.take(cand)
        if len(dedup.items) < want:
            for cand in await ctx.distractors.for_word(entry, k=want, pos=sense.pos):
                if dedup.take(cand) and len(dedup.items) >= want:
                    break
        return dedup.items[:want]

    async def grade(self, ctx: QuestionContext, question: Question, answer: object) -> Score:
        # Same helper the rule DefinitionMCQ uses — the cross-axis proof.
        return await grade_single_choice(question, answer)


register(FormatSpec("contextual_mcq", "single_choice", ContextualMCQ))
