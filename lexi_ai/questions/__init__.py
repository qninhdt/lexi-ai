"""Questions subsystem: CRUD + grade vocabulary questions from a done Entry.

One plugin per format owns generate + grade + its own persistence decision; the
engine is a dispatcher. Importing this package populates the format registry.
"""

# Populate REGISTRY on package import (runs the register() calls).
from lexi_ai.questions import formats as _formats  # noqa: E402,F401
from lexi_ai.questions.base import (
    REGISTRY,
    FormatSpec,
    QuestionContext,
    QuestionFormat,
    QuestionStore,
    register,
)

__all__ = [
    "REGISTRY",
    "FormatSpec",
    "QuestionContext",
    "QuestionFormat",
    "QuestionStore",
    "register",
]
