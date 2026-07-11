"""Shared grade helpers — the deterministic grading logic plugins delegate to.

Grading is dispatched by the engine to a plugin's ``grade``; a plugin grades by
calling the helper matching its ``answer_kind``. Keeping the logic here (not in a
class hierarchy) is pure DRY: multiple plugins grade ``single_choice`` the same
way, and that sameness is exactly the cross-axis proof — the llm-authored
``contextual_mcq`` and the rule ``definition_mcq`` both grade through
:func:`grade_single_choice`, so "who generated it" is irrelevant to grading.
"""

import re

from lexi_ai.llm import StructuredLLM, ainvoke_structured, guarded_messages
from lexi_ai.normalize import match_key
from lexi_ai.prompts import PromptLoader
from lexi_ai.questions.schemas import Judgment
from lexi_ai.read_models import Question, Score

_RUBRIC_SYSTEM = PromptLoader.render("rubric_scoring_system")

# An option index is an ASCII integer only. Guards the choice grader against
# unicode digits and repeated signs that str.isdigit()/lstrip("-") let through.
_ASCII_INT_RE = re.compile(r"^-?[0-9]+$")


async def grade_single_choice(question: Question, answer: object) -> Score:
    """Grade a single-choice answer given as an option index OR the option text.

    An int (or int-like string) is the option index. Any other string is matched
    by ``match_key`` against the options (lenient — a client can answer by value).
    An unmatched value scores wrong rather than raising.
    """
    options = question.payload["options"]
    correct_index = question.payload["correct_index"]
    idx = _resolve_choice(answer, options)
    correct = idx is not None and idx == correct_index
    return Score(correct=correct, score=1.0 if correct else 0.0, kind="rule")


async def grade_text_span(question: Question, answer: object) -> Score:
    """Grade a text-span answer by ``match_key`` equality with the stored answer.

    Rides the ONE normalizer, so cloze grading can never drift from how the
    dictionary keys words — ``"colour"`` folds equal to a stored ``"color"`` only
    if ``match_key`` says so.

    ``accepted_forms`` (optional) widens the accepted set with the sense's
    inflected surfaces, so a learner typing ``ran`` for ``run`` scores right. This
    does NOT alter ``match_key`` — each accepted surface is normalized the same
    way and added to the target set, closing the documented inflection gap without
    touching the invariant.
    """
    payload = question.payload
    want = {match_key(payload["answer_norm"])}
    want.update(match_key(s) for s in payload.get("accepted_forms", []))
    correct = match_key(str(answer)) in want
    return Score(correct=correct, score=1.0 if correct else 0.0, kind="rule")


async def grade_rubric(question: Question, answer: object, *, judge: StructuredLLM | None) -> Score:
    """Grade a free-text answer against the payload rubric via an llm judge.

    Best-effort posture is deliberately NOT used here: grading is the caller's
    explicit request (unlike best-effort embeddings), so a persistent judge
    failure raises rather than silently scoring wrong. Requires an injected
    ``judge`` :class:`StructuredLLM` producing a :class:`Judgment`.
    """
    if judge is None:
        raise ValueError("rubric grading requires a judge (ctx.judge is None)")
    payload = question.payload
    human = PromptLoader.render(
        "rubric_scoring_user",
        target_norm=payload["target_norm"],
        rubric=payload["rubric"],
        prompt=payload["prompt"],
        answer=answer,
    )
    judgment = await ainvoke_structured(
        judge,
        guarded_messages(_RUBRIC_SYSTEM, human),
        Judgment,
    )
    return Score(
        correct=judgment.correct,
        score=judgment.score,
        kind="llm",
        feedback=judgment.feedback,
    )


async def grade_matching(question: Question, answer: object) -> Score:
    """Grade a matching answer (a list of right-indices) against ``correct_map``.

    ``score`` is the fraction of lefts paired with their correct right; ``correct``
    is True only when every pair is right. A malformed answer (not a list, or wrong
    length) scores wrong rather than raising — same lenient posture as the choice
    grader. Order-independent: the shuffle lives in the payload, so submitting the
    stored ``correct_map`` back always scores full regardless of display order.
    """
    correct_map = question.payload["correct_map"]
    if not isinstance(answer, (list, tuple)) or len(answer) != len(correct_map):
        return Score(correct=False, score=0.0, kind="rule")
    hits = sum(1 for got, want in zip(answer, correct_map, strict=True) if got == want)
    score = hits / len(correct_map)
    return Score(correct=hits == len(correct_map), score=score, kind="rule")


def _resolve_choice(answer: object, options: list[str]) -> int | None:
    """Interpret an answer as an option index: an int index, or an option value."""
    if isinstance(answer, bool):  # bool is an int subclass — never an index
        return None
    if isinstance(answer, int):
        return answer
    text = str(answer).strip()
    # Parse an option index TOTALLY: only an ASCII ``-?\d+`` is an index, and even
    # then ``int()`` can still reject it (e.g. a 5000-digit string trips CPython's
    # int_max_str_digits). ``str.isdigit()`` was wrong here — it accepts unicode
    # digits ("²") and ``lstrip("-")`` allowed "--5", both of which ``int()`` then
    # raised on, crashing the public grade() contract. A non-index (or unparseable)
    # string must fall through to the match_key option lookup and score wrong.
    if _ASCII_INT_RE.match(text):
        try:
            return int(text)
        except ValueError:
            return None
    want = match_key(text)
    for i, opt in enumerate(options):
        if match_key(opt) == want:
            return i
    return None
