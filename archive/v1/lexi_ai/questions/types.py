"""The question contract: what a learner sees, and what grading holds back.

The correct answer NEVER appears on a presentation type. ``PresentedQuestion`` and every
``RenderContract`` variant carry only what a learner may see; the answer is disclosed solely through
``Evaluation.reveal``, produced by grading.

The grading types live here too, being the same contract from the other side — but a consumer
projecting a question to a learner must never read past ``PresentedQuestion``.
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


# --- presentation (answer-free) -------------------------------------------


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
    """Exposure card. ``definition``/``example`` are the recall 'back' and are present by design; a
    flashcard MUST NOT be used as an assessment grading path (exposure only, never graded).
    """

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

    question_id: int
    type_id: str
    interaction: Interaction
    render_kind: RenderKind
    difficulty_level: int
    render: RenderContract
    sense_id: int | None = None
    word_id: int | None = None


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
    question_id: int
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
    """Grading outcome. ``reveal`` is the sanctioned answer disclosure; its *release* is gated by
    attempt/terminal state at the delivery layer.
    """

    question_id: int
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
    """Input DTO: request questions for a sense at a difficulty level."""

    sense_id: int
    difficulty_level: int


@dataclass(frozen=True, slots=True)
class PrepareReport:
    """Outcome DTO: how many questions each prepare demand produced, keyed by ``(sense_id,
    difficulty_level)``. Counts aggregate when several types supply the same demand.
    """

    produced: dict[tuple[int, int], int]


# --- grading (the withheld half) ------------------------------------------


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
class PersistedQuestion:
    """Internal carrier bridging the flat stored ``payload`` and the typed public boundary.

    Plugins build one as a DRAFT (``question_id=None``) and hand it to the store, which returns one
    stamped with the real row id. ``payload`` is the SAME flat dict persisted in the single
    ``payload`` column (so ``content_hash`` and dedup identity never change) and NEVER crosses the
    consumer boundary — the projection layer (:mod:`lexi_ai.questions.render`) turns it into the
    answer-free :class:`PresentedQuestion`, a :class:`GradingSpec`, or a post-grading ``Reveal``.
    """

    question_id: int | None
    word_id: int
    sense_id: int | None
    type_id: str
    render_kind: RenderKind
    difficulty_level: int
    interaction: str
    payload: dict
