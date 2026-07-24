"""Shared grading helpers for assessment question types.

Each grader takes the internal :class:`PersistedQuestion` carrier plus the public
:class:`AnswerSubmission` and returns a typed :class:`Evaluation` — the answer is
disclosed only through ``Evaluation.reveal`` (built by :mod:`lexi_ai.questions.render`),
never on the presentation. Rubric grading needs the judge LLM; without one it
returns ``pending`` (the provider-free reader path cannot grade free text).
"""

import re

from lexi_ai.contracts.questions import (
    AnswerSubmission,
    ChoiceResponse,
    Evaluation,
    Response,
    Reveal,
    RubricReveal,
    TextResponse,
)
from lexi_ai.domain.questions import PersistedQuestion
from lexi_ai.llm import StructuredLLM, ainvoke_structured, guarded_messages
from lexi_ai.normalize import match_key
from lexi_ai.prompts import PromptLoader
from lexi_ai.questions.render import to_grading, to_reveal
from lexi_ai.questions.schemas import Judgment

_RUBRIC_SYSTEM = PromptLoader.render("rubric_scoring_system")
_ASCII_INT_RE = re.compile(r"^-?[0-9]+$")


def _graded(
    question_id: str,
    correct: bool,
    score: float | None = None,
    feedback: str | None = None,
    reveal: Reveal | None = None,
) -> Evaluation:
    return Evaluation(
        question_id=question_id,
        status="graded",
        correct=correct,
        score=(1.0 if correct else 0.0) if score is None else score,
        feedback=feedback,
        reveal=reveal,
    )


async def grade_single_choice(
    persisted: PersistedQuestion, submission: AnswerSubmission
) -> Evaluation:
    """Grade a selected option index (or a text option value) without raising.

    The answer key comes from the typed :class:`ChoiceGrading`; the option values
    (needed to resolve a text response) are presentation data read from payload.
    """
    grading = to_grading(persisted.render_kind, persisted.payload)
    selected = _resolve_choice(submission.response, persisted.payload["options"])
    reveal = to_reveal(persisted.render_kind, persisted.payload)
    correct = selected is not None and selected == grading.correct_index
    return _graded(submission.question_id, correct, reveal=reveal)


async def grade_text_span(
    persisted: PersistedQuestion, submission: AnswerSubmission
) -> Evaluation:
    """Grade text against the lemma and any accepted inflected surfaces."""
    grading = to_grading(persisted.render_kind, persisted.payload)
    accepted = {match_key(grading.answer_norm)}
    accepted.update(match_key(form) for form in grading.accepted_forms)
    text = _response_text(submission.response, persisted.payload.get("word_bank"))
    reveal = to_reveal(persisted.render_kind, persisted.payload)
    return _graded(submission.question_id, match_key(text) in accepted, reveal=reveal)


async def grade_rubric(
    persisted: PersistedQuestion,
    submission: AnswerSubmission,
    *,
    judge: StructuredLLM | None,
) -> Evaluation:
    """Return ``pending`` without a judge; otherwise evaluate through the rubric.

    The provider-free reader path passes ``judge=None`` and therefore cannot grade
    free text — the outcome stays ``pending`` (no reveal until graded).
    """
    if judge is None:
        return Evaluation(question_id=submission.question_id, status="pending")
    grading = to_grading(persisted.render_kind, persisted.payload)
    human = PromptLoader.render(
        "rubric_scoring_user",
        target_norm=grading.target_norm,
        rubric=grading.rubric,
        prompt=persisted.payload["prompt"],
        answer=_response_text(submission.response),
    )
    judgment = await ainvoke_structured(
        judge,
        guarded_messages(_RUBRIC_SYSTEM, human),
        Judgment,
    )
    reveal = RubricReveal(feedback=judgment.feedback)
    return _graded(
        submission.question_id, judgment.correct, judgment.score, judgment.feedback, reveal
    )


def _response_text(response: Response, options: list[str] | None = None) -> str:
    """The learner's text: a ``TextResponse`` verbatim, or a chosen option value
    (word bank) when a ``ChoiceResponse`` is submitted for a text-span question."""
    if isinstance(response, TextResponse):
        return response.text
    if isinstance(response, ChoiceResponse):
        idx = response.selected_index
        if options and 0 <= idx < len(options):
            return options[idx]
        return str(idx)
    return ""


def _resolve_choice(response: Response, options: list[str]) -> int | None:
    """Resolve a response to an option index: the selected index for a
    ``ChoiceResponse``, or a text match (numeric or ``match_key`` on the option
    values) for a ``TextResponse``."""
    if isinstance(response, ChoiceResponse):
        return response.selected_index
    if isinstance(response, TextResponse):
        text = response.text.strip()
        if _ASCII_INT_RE.fullmatch(text):
            try:
                return int(text)
            except ValueError:
                return None
        wanted = match_key(text)
        return next(
            (index for index, option in enumerate(options) if match_key(option) == wanted),
            None,
        )
    return None
