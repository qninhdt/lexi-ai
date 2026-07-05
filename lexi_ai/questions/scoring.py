"""Shared grade helpers — the deterministic grading logic plugins delegate to.

Grading is dispatched by the engine to a plugin's ``grade``; a plugin grades by
calling the helper matching its ``answer_kind``. Keeping the logic here (not in a
class hierarchy) is pure DRY: multiple plugins grade ``single_choice`` the same
way, and that sameness is exactly the cross-axis proof — the llm-authored
``contextual_mcq`` and the rule ``definition_mcq`` both grade through
:func:`grade_single_choice`, so "who generated it" is irrelevant to grading.
"""

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable

from lexi_ai.normalize import match_key
from lexi_ai.llm import ainvoke_structured
from lexi_ai.questions.schemas import Judgment
from lexi_ai.read_models import Question, Score

from lexi_ai.prompts import PromptLoader

_RUBRIC_SYSTEM = PromptLoader.render("rubric_scoring_system")


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
    """
    correct = match_key(str(answer)) == match_key(question.payload["answer_norm"])
    return Score(correct=correct, score=1.0 if correct else 0.0, kind="rule")


async def grade_rubric(question: Question, answer: object, *, judge: Runnable) -> Score:
    """Grade a free-text answer against the payload rubric via an llm judge.

    Best-effort posture is deliberately NOT used here: grading is the caller's
    explicit request (unlike best-effort embeddings), so a persistent judge
    failure raises rather than silently scoring wrong. Requires an injected
    ``judge`` runnable bound to :class:`Judgment`.
    """
    if judge is None:
        raise ValueError("rubric grading requires a judge runnable (ctx.judge is None)")
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
        [SystemMessage(content=_RUBRIC_SYSTEM), HumanMessage(content=human)],
        Judgment,
    )
    return Score(
        correct=judgment.correct,
        score=judgment.score,
        kind="llm",
        feedback=judgment.feedback,
    )


def _resolve_choice(answer: object, options: list[str]) -> int | None:
    """Interpret an answer as an option index: an int index, or an option value."""
    if isinstance(answer, bool):  # bool is an int subclass — never an index
        return None
    if isinstance(answer, int):
        return answer
    text = str(answer).strip()
    if text.lstrip("-").isdigit():
        return int(text)
    want = match_key(text)
    for i, opt in enumerate(options):
        if match_key(opt) == want:
            return i
    return None
