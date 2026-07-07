"""``pronunciation_mcq`` — rule MCQ: 'which word is pronounced /ipa/?' (ephemeral)."""

from lexi_ai.normalize import match_key
from lexi_ai.questions.base import FormatSpec, QuestionContext, register
from lexi_ai.questions.formats._shared import _MCQ_OPTIONS, _core_sense, _mcq_question
from lexi_ai.questions.scoring import grade_single_choice
from lexi_ai.read_models import Question, Score


class PronunciationMCQ:
    """Rule MCQ: 'which word is pronounced /ipa/?' — ephemeral.

    Reuses the shared distractor ladder + ``MCQPayload`` + ``grade_single_choice``,
    exactly like ``DefinitionMCQ`` — only the stem source differs (the sense's IPA
    instead of its definition). Prefers ``ipa_uk``, falls back to ``ipa_us``; a
    sense with neither is unquestionable, so it degrades to ``[]`` (best-effort,
    like ``Listening`` without a ``ctx.tts`` port)."""

    format = "pronunciation_mcq"
    answer_kind = "single_choice"

    async def generate(self, ctx: QuestionContext, n: int = 1) -> list[Question]:
        entry = ctx.entry
        if entry is None or n <= 0:
            return []
        sense = _core_sense(entry)
        if sense is None:
            return []
        ipa = sense.ipa_uk or sense.ipa_us
        if not ipa:
            return []  # no pronunciation to ask about — best-effort
        distractors = await ctx.distractors.for_word(entry, k=_MCQ_OPTIONS - 1, pos=sense.pos)
        stem = f"Which word is pronounced /{ipa.strip('/')}/?"
        seed = f"pronunciation_mcq:{match_key(entry.norm)}"
        q = _mcq_question(entry, sense, stem, seed, distractors)
        if q is None:
            return []
        q.format = self.format
        return [q]

    async def grade(self, ctx: QuestionContext, question: Question, answer: object) -> Score:
        return await grade_single_choice(question, answer)


register(FormatSpec("pronunciation_mcq", "single_choice", PronunciationMCQ))
