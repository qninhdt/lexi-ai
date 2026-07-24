"""Shared evaluation helpers for assessment question types."""

import re

from lexi_ai.llm import StructuredLLM, ainvoke_structured, guarded_messages
from lexi_ai.normalize import match_key
from lexi_ai.prompts import PromptLoader
from lexi_ai.questions.schemas import Judgment
from lexi_ai.read_models import Evaluation, Question

_RUBRIC_SYSTEM = PromptLoader.render("rubric_scoring_system")
_ASCII_INT_RE = re.compile(r"^-?[0-9]+$")


def _graded(verdict: bool, score: float | None = None, feedback: str | None = None) -> Evaluation:
    return Evaluation(
        status="graded",
        verdict=verdict,
        score=(1.0 if verdict else 0.0) if score is None else score,
        feedback=feedback,
    )


async def grade_single_choice(question: Question, answer: object) -> Evaluation:
    """Grade an option index or normalized option value without raising."""
    options = question.payload["options"]
    correct_index = question.payload["correct_index"]
    selected = _resolve_choice(answer, options)
    return _graded(selected is not None and selected == correct_index)


async def grade_text_span(question: Question, answer: object) -> Evaluation:
    """Grade text against the lemma and any accepted inflected surfaces."""
    payload = question.payload
    accepted = {match_key(payload["answer_norm"])}
    accepted.update(match_key(value) for value in payload.get("accepted_forms", []))
    return _graded(match_key(str(answer)) in accepted)


async def grade_rubric(
    question: Question, answer: object, *, judge: StructuredLLM | None
) -> Evaluation:
    """Return pending without a judge; otherwise evaluate through the rubric."""
    if judge is None:
        return Evaluation(status="pending", verdict=None, score=None)
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
    return _graded(judgment.correct, judgment.score, judgment.feedback)


async def grade_matching(question: Question, answer: object) -> Evaluation:
    """Retain the unregistered matching helper for its future plugin migration."""
    correct_map = question.payload["correct_map"]
    if not isinstance(answer, (list, tuple)) or len(answer) != len(correct_map):
        return _graded(False)
    hits = sum(1 for got, want in zip(answer, correct_map, strict=True) if got == want)
    return _graded(hits == len(correct_map), hits / len(correct_map))


def _resolve_choice(answer: object, options: list[str]) -> int | None:
    if isinstance(answer, bool):
        return None
    if isinstance(answer, int):
        return answer
    text = str(answer).strip()
    if _ASCII_INT_RE.fullmatch(text):
        try:
            return int(text)
        except ValueError:
            return None
    wanted = match_key(text)
    return next(
        (
            index
            for index, option in enumerate(options)
            if match_key(option) == wanted
        ),
        None,
    )
