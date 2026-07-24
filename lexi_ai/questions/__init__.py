"""Questions subsystem: CRUD + grade vocabulary questions from a done Entry.

ONE plugin per question type declares a typed ``QuestionTypeInfo`` and owns
prepare/retrieve/grade; the engine is a pure dispatcher. Importing this package
populates the registry by direct import of the built-in types. The answer-free
projection from the stored flat payload lives in ``render``.
"""

# Populate REGISTRY on package import (runs the register() calls for built-ins).
from lexi_ai.questions import types as _types  # noqa: E402,F401
from lexi_ai.questions.base import (
    REGISTRY,
    QuestionContext,
    QuestionStore,
    load_entry_point_types,
    register,
)
from lexi_ai.questions.render import to_grading, to_presented, to_render, to_reveal

__all__ = [
    "QuestionContext",
    "QuestionStore",
    "REGISTRY",
    "load_entry_point_types",
    "register",
    "to_grading",
    "to_presented",
    "to_render",
    "to_reveal",
]
