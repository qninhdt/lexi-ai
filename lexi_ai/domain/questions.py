"""Internal question domain types. NEVER re-exported from ``lexi_ai.contracts``.

``GradingSpec`` holds the correct answer / rubric — the data that must never cross
the consumer boundary. ``StoredQuestion`` binds an answer-free
``PresentedQuestion`` to its ``GradingSpec`` for persistence and grading.
"""

from __future__ import annotations

from dataclasses import dataclass

from lexi_ai.contracts.questions import PresentedQuestion


@dataclass(frozen=True, slots=True)
class ChoiceGrading:
    """Correct option for a single-choice question."""

    correct_index: int


@dataclass(frozen=True, slots=True)
class SpanGrading:
    """Correct fill + accepted inflected surfaces for a cloze/text-span question."""

    answer_norm: str
    accepted_forms: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RubricGrading:
    """Target word + rubric for judge-graded free-text questions."""

    target_norm: str
    rubric: str


GradingSpec = ChoiceGrading | SpanGrading | RubricGrading


@dataclass(frozen=True, slots=True)
class StoredQuestion:
    """Full persisted record: the answer-free presentation plus its grading spec.

    ``grading`` is ``None`` for exposure (flashcard) questions, which are never
    graded.
    """

    presentation: PresentedQuestion
    grading: GradingSpec | None
