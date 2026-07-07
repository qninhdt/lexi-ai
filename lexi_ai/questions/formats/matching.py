"""``matching`` — pair each sense's guideword (cue) with its definition."""

from lexi_ai.normalize import match_key
from lexi_ai.questions.base import FormatSpec, QuestionContext, register
from lexi_ai.questions.formats._shared import _shuffled_pairs
from lexi_ai.questions.schemas import MatchingPayload
from lexi_ai.questions.scoring import grade_matching
from lexi_ai.read_models import Question, Score


class Matching:
    """Rule format: pair each sense's guideword (cue) with its definition.

    The first format to declare a new ``answer_kind`` (``matching``), so it proves
    the plugin abstraction extends along that axis with no engine change. Pairs are
    built from the entry's OWN senses — a self-contained source needing no
    cross-word lookup. Requires ≥2 guideworded senses; degrades to ``[]`` otherwise
    (a single sense has nothing to match, and a sense without a guideword has no
    cue), mirroring the MCQ min-distractor floor rather than fabricating a pair.

    The definition column is shuffled with a LOCAL ``random.Random(seed)`` (no
    global RNG) so option order is stable and testable; ``correct_map[i]`` is the
    index in that shuffled column of the definition belonging to ``lefts[i]``.
    """

    format = "matching"
    answer_kind = "matching"

    _MIN_PAIRS = 2

    async def generate(self, ctx: QuestionContext, n: int = 1) -> list[Question]:
        entry = ctx.entry
        if entry is None or n <= 0:
            return []
        pairs = [(s, s.guideword, s.definition) for s in entry.senses if s.guideword]
        if len(pairs) < self._MIN_PAIRS:
            return []  # too few cues to match — best-effort, no fabrication
        lefts = [g for _s, g, _d in pairs]
        defs = [d for _s, _g, d in pairs]
        rights, correct_map = _shuffled_pairs(defs, f"matching:{match_key(entry.norm)}")
        payload = MatchingPayload(
            prompt="Match each word sense to its definition.",
            lefts=lefts,
            rights=rights,
            correct_map=correct_map,
        )
        return [
            Question(
                id=None,
                word_id=entry.word_id,
                sense_id=None,  # spans multiple senses — no single-sense provenance
                format=self.format,
                answer_kind=self.answer_kind,
                payload=payload.model_dump(),
            )
        ]

    async def grade(self, ctx: QuestionContext, question: Question, answer: object) -> Score:
        return await grade_matching(question, answer)


register(FormatSpec("matching", "matching", Matching))
