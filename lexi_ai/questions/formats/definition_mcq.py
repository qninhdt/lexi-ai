"""``definition_mcq`` — rule MCQ: 'which word means <definition>?' (ephemeral)."""

from lexi_ai.normalize import match_key
from lexi_ai.questions.base import FormatSpec, QuestionContext, register
from lexi_ai.questions.formats._shared import _MCQ_OPTIONS, _core_sense, _mcq_question
from lexi_ai.questions.scoring import grade_single_choice
from lexi_ai.read_models import Question, Score


class DefinitionMCQ:
    """Rule MCQ: 'which word means <definition>?' — ephemeral (not persisted)."""

    format = "definition_mcq"
    answer_kind = "single_choice"

    async def generate(self, ctx: QuestionContext, n: int = 1) -> list[Question]:
        entry = ctx.entry
        if entry is None or n <= 0:
            return []
        sense = _core_sense(entry)
        if sense is None:
            return []
        distractors = await ctx.distractors.for_word(entry, k=_MCQ_OPTIONS - 1, pos=sense.pos)
        stem = f"Which word means: {sense.definition}"
        seed = f"definition_mcq:{match_key(entry.norm)}"
        q = _mcq_question(entry, sense, stem, seed, distractors)
        if q is None:
            return []
        q.format = self.format
        return [q]

    async def grade(self, ctx: QuestionContext, question: Question, answer: object) -> Score:
        return await grade_single_choice(question, answer)


register(FormatSpec("definition_mcq", "single_choice", DefinitionMCQ))
