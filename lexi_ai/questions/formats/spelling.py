"""``spelling`` — audio dictation: hear the sense definition, TYPE the word; ephemeral."""

from lexi_ai.questions.base import FormatSpec, QuestionContext, register
from lexi_ai.questions.formats._shared import _accepted_forms, _core_audio_ref, _core_sense
from lexi_ai.questions.schemas import SpellingPayload
from lexi_ai.questions.scoring import grade_text_span
from lexi_ai.read_models import Question, Score


class Spelling:
    """Audio dictation: hear the sense definition spoken, TYPE the word — ephemeral.

    Reuses ``text_span`` + ``grade_text_span``. ``accepted_forms`` (the sense's
    inflected surfaces) let a learner typing ``ran`` for ``run`` score right; the
    lemma norm alone otherwise. Payload stores the clip REFERENCE tuple (durability
    rationale as ``Listening``). Grading is text-only and never touches the clip, so
    a dangling audio ref still grades.
    """

    format = "spelling"
    answer_kind = "text_span"

    async def generate(self, ctx: QuestionContext, n: int = 1) -> list[Question]:
        entry = ctx.entry
        if entry is None or n <= 0:
            return []
        sense = _core_sense(entry)
        if sense is None:
            return []
        audio_ref = await _core_audio_ref(ctx, sense)
        if audio_ref is None:
            return []
        payload = SpellingPayload(
            prompt="Listen to the audio, then type the word you hear.",
            audio_ref=audio_ref,
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

    async def grade(self, ctx: QuestionContext, question: Question, answer: object) -> Score:
        return await grade_text_span(question, answer)


register(FormatSpec("spelling", "text_span", Spelling))
