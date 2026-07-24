"""Internal question domain types. NEVER re-exported from ``lexi_ai.contracts``.

``GradingSpec`` holds the correct answer / rubric — the data that must never cross
the consumer boundary. ``StoredQuestion`` binds an answer-free
``PresentedQuestion`` to its ``GradingSpec`` for persistence and grading.
"""

from __future__ import annotations

from dataclasses import dataclass

from lexi_ai.contracts.questions import PresentedQuestion, RenderKind


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


@dataclass(frozen=True, slots=True)
class PersistedQuestion:
    """Internal carrier bridging the flat stored ``payload`` and the typed public
    boundary.

    Plugins build one as a DRAFT (``question_id=None``) and hand it to the store;
    the store returns one stamped with the real row id. ``payload`` is the SAME
    flat dict persisted in the single ``payload`` column (so ``content_hash`` and
    dedup identity never change); it stays internal and NEVER crosses the consumer
    boundary — the projection layer (:mod:`lexi_ai.questions.render`) turns it into
    the answer-free :class:`~lexi_ai.contracts.questions.PresentedQuestion`, a
    :class:`GradingSpec`, or a post-grading ``Reveal``.
    """

    question_id: int | None
    word_id: int
    sense_id: int | None
    type_id: str
    render_kind: RenderKind
    difficulty_level: int
    interaction: str
    payload: dict
