"""Public contract surface for lexi-ai. Consumers import ONLY from here.

Frozen, typed, dependency-free: no SQLAlchemy/ORM, no domain/infrastructure/
application imports (enforced by import-linter, see ``pyproject.toml``).

Answer-safe question contract: the correct answer never appears on
``PresentedQuestion`` / ``RenderContract``; it is disclosed only through
``Evaluation.reveal`` after grading.

Dictionary DTOs (``Entry``, ``SenseView``, ``Theme``, ``Asset``, ...) migrate into
``contracts/dictionary.py`` in Phase 3, when the ``read_models`` importers are
repointed; today they still live in ``lexi_ai.read_models``.
"""

from lexi_ai.contracts.questions import (
    AnswerSubmission,
    ChoiceResponse,
    ChoiceReveal,
    Evaluation,
    Flashcard,
    FreeText,
    Interaction,
    PrepareDemand,
    PresentedQuestion,
    QuestionTypeInfo,
    RenderContract,
    RenderKind,
    Response,
    Reveal,
    RubricReveal,
    SingleChoice,
    SpanReveal,
    TextResponse,
    TextSpan,
)

__all__ = [
    "AnswerSubmission",
    "ChoiceResponse",
    "ChoiceReveal",
    "Evaluation",
    "Flashcard",
    "FreeText",
    "Interaction",
    "PrepareDemand",
    "PresentedQuestion",
    "QuestionTypeInfo",
    "RenderContract",
    "RenderKind",
    "Response",
    "Reveal",
    "RubricReveal",
    "SingleChoice",
    "SpanReveal",
    "TextResponse",
    "TextSpan",
]
