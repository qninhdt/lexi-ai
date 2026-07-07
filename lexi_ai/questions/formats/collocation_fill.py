"""``collocation_fill`` — rule fill-in-the-blank from a stored collocation."""

from lexi_ai.questions.base import FormatSpec, QuestionContext, register
from lexi_ai.questions.formats._shared import _accepted_forms, _blank_in_phrase, _core_sense
from lexi_ai.questions.schemas import ClozePayload
from lexi_ai.questions.scoring import grade_text_span
from lexi_ai.read_models import Question, Score


class CollocationFill:
    """Rule fill-in-the-blank from a stored collocation — ephemeral.

    Reuses the ``ClozePayload`` + ``grade_text_span`` machinery (text_span), but
    the stem source is a partner phrase (``make a decision``) rather than a full
    sentence: blank the target within one collocation. Collocations carry no
    ``<t inf>`` markup, so blanking folds each token through the accepted-surface
    set (lemma + inflected forms), which also lets ``heavy rains`` blank against
    ``rain``. First collocation containing the target wins; none → ``[]``."""

    format = "collocation_fill"
    answer_kind = "text_span"

    async def generate(self, ctx: QuestionContext, n: int = 1) -> list[Question]:
        entry = ctx.entry
        if entry is None or n <= 0:
            return []
        sense = _core_sense(entry)
        if sense is None:
            return []
        accepted = _accepted_forms(sense)
        for colloc in sense.collocations:
            blanked = _blank_in_phrase(colloc, entry, accepted)
            if blanked is None:
                continue  # target not locatable in this collocation — try the next
            payload = ClozePayload(
                stem_with_blank=blanked,
                answer_norm=entry.norm,
                accepted_forms=accepted,
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
        return []  # no collocation contained the target

    async def grade(self, ctx: QuestionContext, question: Question, answer: object) -> Score:
        return await grade_text_span(question, answer)


register(FormatSpec("collocation_fill", "text_span", CollocationFill))
