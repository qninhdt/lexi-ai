"""Plugin contracts and registry for the unified questions subsystem.

A question type is ONE plugin declaring a typed
:class:`~lexi_ai.contracts.questions.QuestionTypeInfo` (``info``). It owns
prepare/retrieve and — for assessments — ``grade``. The collapsed model replaces
the old split of ``QuestionTypeDescriptor`` (type) and a separate render-format
registry: the render shape now rides ``info.render_kind`` and the answer-safe
projection lives in :mod:`lexi_ai.questions.render`.

Built-in types register by DIRECT import (see ``lexi_ai.questions.types``).
Third-party types are discovered via the ``lexi_ai.question_types`` entry-point
group, but only when their ``type_id`` appears in an explicit allowlist
(:func:`load_entry_point_types`) — untrusted discovery is opt-in for safety.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from importlib.metadata import entry_points
from typing import TYPE_CHECKING, Protocol, TypeAlias

from lexi_ai.constants import (
    DIFFICULTY_LEVELS,
    INTERACTION_MODES,
    QUESTION_TYPES,
    RENDER_FORMATS,
)
from lexi_ai.contracts.questions import (
    AnswerSubmission,
    Evaluation,
    QuestionTypeInfo,
    RenderKind,
)
from lexi_ai.domain.questions import PersistedQuestion
from lexi_ai.read_models import Entry

if TYPE_CHECKING:
    from lexi_ai.llm import StructuredLLM
    from lexi_ai.questions.distractors import DistractorProvider

# Entry-point group third-party question types advertise themselves under.
ENTRY_POINT_GROUP = "lexi_ai.question_types"


class QuestionStore(Protocol):
    """Question-only persistence capability supplied to assessment plugins."""

    async def insert(self, draft: PersistedQuestion) -> PersistedQuestion: ...

    async def retrieve_one(
        self,
        sense_id: int,
        difficulty_level: int,
        type_id: str,
        excluded_ids: frozenset[int],
    ) -> PersistedQuestion | None: ...

    async def list_for_word(
        self, word_id: int, type_id: str | None = None
    ) -> list[PersistedQuestion]: ...

    async def get(self, question_id: int) -> PersistedQuestion | None: ...

    async def delete(self, question_id: int) -> bool: ...


class SenseEntryLoader(Protocol):
    """Narrow read capability used to build exposure cards from a sense id."""

    async def load_entry(self, sense_id: int) -> Entry | None: ...


class TtsPort(Protocol):
    """Narrow audio-synthesis capability retained for future question types."""

    async def ensure_clip(
        self, source_kind: str, source_id: int
    ) -> tuple[str, int, str, str] | None: ...


@dataclass
class QuestionContext:
    """Per-call capabilities; plugins use only the seams they require."""

    entry: Entry | None
    distractors: "DistractorProvider"
    llm: "StructuredLLM | None" = None
    judge: "StructuredLLM | None" = None
    store: "QuestionStore | None" = None
    tts: "TtsPort | None" = None
    sense_loader: "SenseEntryLoader | None" = None


@dataclass(frozen=True)
class QuestionDemand:
    """Internal prepare input (sense id is a resolved int, unlike the public
    ``contracts.PrepareDemand`` whose sense id is a string)."""

    sense_id: int
    difficulty_level: int
    expected_count: int


@dataclass(frozen=True)
class QuestionQuery:
    sense_id: int
    difficulty_level: int
    excluded_question_ids: frozenset[int] = frozenset()


@dataclass(frozen=True)
class PrepareReport:
    produced: dict[tuple[int, int], int]


class UnknownQuestionType(ValueError):
    """Raised when dispatch is requested for an unregistered type id."""

    def __init__(self, type_id: str):
        self.type_id = type_id
        super().__init__(f"unknown question type: {type_id!r}")


class NotAssessable(ValueError):
    """Raised when a caller attempts to evaluate an exposure question."""

    def __init__(self, question_id: int | None):
        self.question_id = question_id
        super().__init__(f"question {question_id!r} is not assessable")


class AssessmentType(Protocol):
    info: QuestionTypeInfo

    async def prepare(
        self, ctx: QuestionContext, demands: Sequence[QuestionDemand]
    ) -> PrepareReport: ...

    async def retrieve(
        self, ctx: QuestionContext, query: QuestionQuery
    ) -> PersistedQuestion | None: ...

    async def grade(
        self,
        ctx: QuestionContext,
        persisted: PersistedQuestion,
        submission: AnswerSubmission,
    ) -> Evaluation: ...


class ExposureType(Protocol):
    info: QuestionTypeInfo

    async def retrieve(self, ctx: QuestionContext, query: QuestionQuery) -> PersistedQuestion: ...


QuestionType: TypeAlias = AssessmentType | ExposureType
QuestionTypeFactory: TypeAlias = Callable[[], QuestionType]
REGISTRY: dict[str, QuestionType] = {}


def register(make_plugin: QuestionTypeFactory) -> None:
    """Instantiate and register one type after fail-fast contract validation.

    Validates the declared :class:`QuestionTypeInfo` against the controlled
    vocabularies and the exposure/assessment invariant: level 0 is flashcard-only
    exposure with NO ``grade``; every other level is a graded assessment.
    """
    plugin = make_plugin()
    info = plugin.info

    if info.type_id not in QUESTION_TYPES:
        raise ValueError(f"unknown question type: {info.type_id!r}")
    if info.render_kind.value not in RENDER_FORMATS:
        raise ValueError(f"unknown render format: {info.render_kind.value!r}")
    if info.interaction not in INTERACTION_MODES:
        raise ValueError(f"unknown interaction mode: {info.interaction!r}")
    if not info.difficulty_levels or not info.difficulty_levels <= DIFFICULTY_LEVELS:
        raise ValueError("difficulty_levels must be a non-empty difficulty-level subset")

    if 0 in info.difficulty_levels:
        if (
            info.difficulty_levels != frozenset({0})
            or info.render_kind is not RenderKind.FLASHCARD
            or info.interaction != "exposure"
        ):
            raise ValueError("level 0 types must be flashcard-only exposure types")
        if hasattr(plugin, "grade"):
            raise ValueError("an exposure type must not define grade")
    else:
        if info.interaction != "assessment":
            raise ValueError("non-zero difficulty types must be assessments")
        if not hasattr(plugin, "grade"):
            raise ValueError("an assessment type must define grade")

    REGISTRY[info.type_id] = plugin


def load_entry_point_types(allowlist: set[str] | None = None) -> None:
    """Discover and register third-party question types via entry points.

    SECURITY: an installed distribution can advertise a plugin under the
    ``lexi_ai.question_types`` group, but it is registered ONLY when its
    ``type_id`` (the entry-point name) is present in ``allowlist``. ``None`` / an
    empty allowlist is a no-op — built-in-only, the default posture. A type_id
    already in the registry (e.g. a built-in loaded by direct import) is skipped,
    never double-registered.
    """
    if not allowlist:
        return
    for ep in entry_points(group=ENTRY_POINT_GROUP):
        if ep.name not in allowlist or ep.name in REGISTRY:
            continue
        register(ep.load())


__all__ = [
    "AssessmentType",
    "ENTRY_POINT_GROUP",
    "ExposureType",
    "NotAssessable",
    "PrepareReport",
    "QuestionContext",
    "QuestionDemand",
    "QuestionQuery",
    "QuestionStore",
    "QuestionType",
    "REGISTRY",
    "SenseEntryLoader",
    "TtsPort",
    "UnknownQuestionType",
    "load_entry_point_types",
    "register",
]
