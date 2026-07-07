"""``listening`` — audio MCQ: hear the sense definition, choose the word; PERSISTS."""

from lexi_ai.normalize import match_key
from lexi_ai.questions.base import FormatSpec, QuestionContext, register
from lexi_ai.questions.formats._shared import (
    _MCQ_MIN_DISTRACTORS,
    _MCQ_OPTIONS,
    _core_audio_ref,
    _core_sense,
    _shuffled_options,
)
from lexi_ai.questions.schemas import ListeningPayload
from lexi_ai.questions.scoring import grade_single_choice
from lexi_ai.read_models import Question, Score


class Listening:
    """Audio MCQ: hear the sense definition spoken, choose the word — PERSISTS.

    Reuses the ``single_choice`` answer kind and the shared ``grade_single_choice``
    helper (audio is a presentation layer, not a new grading axis). The payload
    stores the clip's ``(source_kind, source_id, voice, fmt)`` REFERENCE tuple, not
    a row id, so a purge/regenerate re-resolves the current clip cache-first at play
    time instead of dangling. Unavailable (no ``ctx.tts``) or unmakeable clip → ``[]``.
    """

    format = "listening"
    answer_kind = "single_choice"

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
        distractors = await ctx.distractors.for_word(entry, k=_MCQ_OPTIONS - 1, pos=sense.pos)
        if len(distractors) < _MCQ_MIN_DISTRACTORS:
            return []
        options, correct_index = _shuffled_options(
            entry.display, distractors, f"listening:{match_key(entry.norm)}"
        )
        payload = ListeningPayload(
            prompt="Listen to the audio, then choose the matching word.",
            audio_ref=audio_ref,
            options=options,
            correct_index=correct_index,
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
        return await grade_single_choice(question, answer)


register(FormatSpec("listening", "single_choice", Listening))
