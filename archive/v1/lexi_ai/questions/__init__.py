"""Questions subsystem: CRUD + grade vocabulary questions from a done Entry.

ONE plugin per question type declares a typed ``QuestionTypeInfo`` and owns prepare/retrieve/grade;
the engine is a pure dispatcher. Importing this package populates the registry by direct import of
the built-in plugins.
"""

# Populate REGISTRY on package import (runs the register() calls for built-ins).
from lexi_ai.questions import plugins as _plugins  # noqa: E402,F401
from lexi_ai.questions.base import (
    REGISTRY,
    QuestionContext,
    QuestionStore,
    register,
)
from lexi_ai.questions.render import to_grading, to_presented, to_render, to_reveal
from lexi_ai.questions.types import Interaction, RenderKind

__all__ = [
    "Interaction",
    "QuestionContext",
    "QuestionStore",
    "REGISTRY",
    "RenderKind",
    "register",
    "to_grading",
    "to_presented",
    "to_render",
    "to_reveal",
]
