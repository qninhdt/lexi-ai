"""The plugin contract for the questions subsystem.

One format = one plugin. A :class:`QuestionFormat` owns BOTH halves for its
questions — ``async generate`` (Entry -> Questions) and ``async grade``
((Question, answer) -> Score) — so the two can never drift. The engine is a pure
dispatcher: it hands each plugin a :class:`QuestionContext` of capabilities and
``await``s it, never inspecting which backend the plugin uses or whether it
persists. A plugin that wants persistence calls ``ctx.store.insert(...)`` itself,
inside its own ``generate``; the engine cannot tell the difference.

The :class:`FormatSpec` registry binds a format id to its plugin and validates
the vocabulary coupling at import time (fail fast, like the constants->schema
wiring in the generation subsystem).
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from lexi_ai.constants import ANSWER_KINDS, QUESTION_FORMATS
from lexi_ai.read_models import Entry, Question, Score

if TYPE_CHECKING:
    from langchain_core.runnables import Runnable

    from lexi_ai.questions.distractors import DistractorProvider


class QuestionStore(Protocol):
    """The narrow CRUD surface a plugin receives via ``ctx.store``.

    Plugins get a DB *door*, not a session, so the repository stays the single
    write path. Satisfied by :class:`lexi_ai.questions.repository.QuestionRepository`.
    """

    async def insert(self, q: Question) -> Question: ...
    async def list_for_word(self, word_id: int, fmt: str | None = None) -> list[Question]: ...
    async def get(self, question_id: int) -> Question | None: ...
    async def delete(self, question_id: int) -> bool: ...


@dataclass
class QuestionContext:
    """Capabilities the engine hands a plugin. A plugin uses only what it needs.

    ``entry`` is the done word being questioned (``None`` when grading, which needs
    no entry). ``llm``/``judge`` are bound structured runnables for plugins that
    generate or grade via an LLM; ``store`` is the persistence door for plugins
    that choose to save their output.
    """

    entry: Entry | None
    distractors: "DistractorProvider"
    llm: "Runnable | None" = None
    judge: "Runnable | None" = None
    store: "QuestionStore | None" = None


class QuestionFormat(Protocol):
    """A self-contained question format: generates and grades its own questions."""

    format: str  # ∈ QUESTION_FORMATS
    answer_kind: str  # ∈ ANSWER_KINDS

    async def generate(self, ctx: QuestionContext, n: int = 1) -> list[Question]: ...
    async def grade(self, ctx: QuestionContext, question: Question, answer: object) -> Score: ...


@dataclass(frozen=True)
class FormatSpec:
    """Registry row: a format id, the answer_kind it declares, and its plugin factory."""

    format: str
    answer_kind: str
    make_plugin: Callable[..., QuestionFormat]


REGISTRY: dict[str, FormatSpec] = {}


def register(spec: FormatSpec) -> None:
    """Add a format to the registry, validating the coupling at import time.

    Fails fast on drift — an out-of-vocab id/answer_kind, or a plugin whose
    declared ``answer_kind`` disagrees with the spec — so a mis-wire is an import
    error, not a runtime surprise. Same philosophy as ``Literal[tuple(constants)]``
    rejecting bad enums in the generation schema.
    """
    if spec.format not in QUESTION_FORMATS:
        raise ValueError(f"unknown question format: {spec.format!r}")
    if spec.answer_kind not in ANSWER_KINDS:
        raise ValueError(f"unknown answer_kind: {spec.answer_kind!r}")
    probe = spec.make_plugin()
    if probe.answer_kind != spec.answer_kind:
        raise ValueError(
            f"format {spec.format!r} declares answer_kind {spec.answer_kind!r} "
            f"but its plugin reports {probe.answer_kind!r}"
        )
    REGISTRY[spec.format] = spec


__all__ = [
    "QuestionContext",
    "QuestionFormat",
    "QuestionStore",
    "FormatSpec",
    "REGISTRY",
    "register",
    "Entry",
    "Question",
    "Score",
]
