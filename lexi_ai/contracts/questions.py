"""Answer-safe question contract.

The correct answer NEVER appears on a presentation type. ``PresentedQuestion`` and
every ``RenderContract`` variant carry only what a learner may see. The answer is
disclosed solely through ``Evaluation.reveal``, produced by grading; the delivery
layer decides whether/when to surface it (gate by attempt/terminal state). Grading
keys live in ``lexi_ai.domain.questions.GradingSpec``, which is NOT importable from
this package (see ``tests/test_contract_answer_safety.py``).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal

Interaction = Literal["exposure", "assessment"]


class RenderKind(str, Enum):
    """The presentation shape a question type declares."""

    SINGLE_CHOICE = "single_choice"
    TEXT_SPAN = "text_span"
    FREE_TEXT = "free_text"
    FLASHCARD = "flashcard"


# --- Presentation (answer-free) -------------------------------------------


@dataclass(frozen=True, slots=True)
class SingleChoice:
    """MCQ prompt. ``options`` carry NO correct marker."""

    stem: str
    options: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TextSpan:
    """Cloze prompt: the blanked sentence plus an optional word bank. No answer."""

    stem_with_blank: str
    word_bank: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class FreeText:
    """Open prompt. The target word / rubric are never exposed here."""

    prompt: str


@dataclass(frozen=True, slots=True)
class Flashcard:
    """Exposure card. ``definition``/``example`` are the recall 'back' and are
    present by design; a flashcard MUST NOT be used as an assessment grading path
    (exposure only, never graded)."""

    word: str
    definition: str
    pos: str | None = None
    example: str | None = None
    ipa_uk: str | None = None
    ipa_us: str | None = None


RenderContract = SingleChoice | TextSpan | FreeText | Flashcard


@dataclass(frozen=True, slots=True)
class PresentedQuestion:
    """A question exactly as a learner sees it. Contains no correct answer."""

    question_id: str
    type_id: str
    interaction: Interaction
    difficulty_level: int
    render: RenderContract
    sense_id: str | None = None
    word_id: str | None = None


# --- Submission -----------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ChoiceResponse:
    selected_index: int


@dataclass(frozen=True, slots=True)
class TextResponse:
    text: str


Response = ChoiceResponse | TextResponse


@dataclass(frozen=True, slots=True)
class AnswerSubmission:
    question_id: str
    response: Response


# --- Reveal (post-grading disclosure — MAY carry the answer) --------------


@dataclass(frozen=True, slots=True)
class ChoiceReveal:
    correct_index: int
    correct_option: str


@dataclass(frozen=True, slots=True)
class SpanReveal:
    correct_answer: str
    accepted_forms: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RubricReveal:
    feedback: str | None = None


Reveal = ChoiceReveal | SpanReveal | RubricReveal


@dataclass(frozen=True, slots=True)
class Evaluation:
    """Grading outcome. ``reveal`` is the sanctioned answer disclosure; its
    *release* is gated by attempt/terminal state at the delivery layer."""

    question_id: str
    status: Literal["graded", "pending"]
    correct: bool | None = None
    score: float | None = None
    feedback: str | None = None
    reveal: Reveal | None = None


# --- Capability & input ---------------------------------------------------


@dataclass(frozen=True, slots=True)
class QuestionTypeInfo:
    """What a consumer needs to discover a question type's capabilities."""

    type_id: str
    render_kind: RenderKind
    interaction: Interaction
    difficulty_levels: frozenset[int]


@dataclass(frozen=True, slots=True)
class PrepareDemand:
    """Input DTO: request questions for a sense at a difficulty level.

    Replaces the internal ``questions.base.QuestionDemand`` on the public surface.
    """

    sense_id: str
    difficulty_level: int
    expected_count: int = 1
    type_id: str | None = None
