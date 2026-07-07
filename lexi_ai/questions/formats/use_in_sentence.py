"""``use_in_sentence`` — rule prompt, graded by the llm rubric judge."""

from lexi_ai.questions.base import FormatSpec, QuestionContext, register
from lexi_ai.questions.formats._shared import _core_sense
from lexi_ai.questions.schemas import UseInSentencePayload
from lexi_ai.questions.scoring import grade_rubric
from lexi_ai.read_models import Question, Score


class UseInSentence:
    """Rule prompt to use the word in a sentence — graded by the llm rubric judge."""

    format = "use_in_sentence"
    answer_kind = "free_text"

    async def generate(self, ctx: QuestionContext, n: int = 1) -> list[Question]:
        entry = ctx.entry
        if entry is None or n <= 0:
            return []
        sense = _core_sense(entry)
        if sense is None:
            return []
        payload = UseInSentencePayload(
            prompt=f"Write a sentence using '{entry.display}' to mean: {sense.definition}",
            target_norm=entry.norm,
            rubric=(
                f"Sentence must use '{entry.display}' with the sense: "
                f"'{sense.definition}'; grammatical; at least 6 words."
            ),
        )
        return [
            Question(
                id=None,
                word_id=entry.word_id,
                sense_id=sense.sense_id,
                format=self.format,
                answer_kind=self.answer_kind,
                payload=payload.model_dump(),
            )
        ]

    async def grade(self, ctx: QuestionContext, question: Question, answer: object) -> Score:
        return await grade_rubric(question, answer, judge=ctx.judge)


register(FormatSpec("use_in_sentence", "free_text", UseInSentence))
