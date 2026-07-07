"""``cloze`` — rule fill-in-the-blank from a sense example (ephemeral)."""

from lexi_ai.questions.base import FormatSpec, QuestionContext, register
from lexi_ai.questions.formats._shared import _accepted_forms, _blank_target, _core_sense
from lexi_ai.questions.schemas import ClozePayload
from lexi_ai.questions.scoring import grade_text_span
from lexi_ai.read_models import Question, Score


class Cloze:
    """Rule fill-in-the-blank from a sense example — ephemeral."""

    format = "cloze"
    answer_kind = "text_span"

    async def generate(self, ctx: QuestionContext, n: int = 1) -> list[Question]:
        entry = ctx.entry
        if entry is None or n <= 0:
            return []
        sense = _core_sense(entry)
        if sense is None:
            return []
        for example in sense.examples:
            blanked = _blank_target(example, entry)
            if blanked is None:
                continue  # target not locatable in this example — try the next
            payload = ClozePayload(
                stem_with_blank=blanked,
                answer_norm=entry.norm,
                accepted_forms=_accepted_forms(sense),
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
        return []  # no example contained the target

    async def grade(self, ctx: QuestionContext, question: Question, answer: object) -> Score:
        return await grade_text_span(question, answer)


register(FormatSpec("cloze", "text_span", Cloze))
