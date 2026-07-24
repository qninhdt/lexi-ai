"""Plugin contracts and registry for the questions subsystem."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, TypeAlias

from pydantic import BaseModel

from lexi_ai.constants import (
    DIFFICULTY_LEVELS,
    INTERACTION_MODES,
    QUESTION_TYPES,
    RENDER_FORMAT_PAYLOAD,
    RENDER_FORMATS,
)
from lexi_ai.questions import schemas as question_schemas
from lexi_ai.read_models import Entry, Evaluation, Question

if TYPE_CHECKING:
    from lexi_ai.llm import StructuredLLM
    from lexi_ai.questions.distractors import DistractorProvider


class QuestionStore(Protocol):
    """Question-only persistence capability supplied to assessment plugins."""

    async def insert(self, question: Question) -> Question: ...

    async def retrieve_one(
        self,
        sense_id: int,
        difficulty_level: int,
        type_id: str,
        excluded_ids: frozenset[int],
    ) -> Question | None: ...

    async def list_for_word(
        self, word_id: int, type_id: str | None = None
    ) -> list[Question]: ...

    async def get(self, question_id: int) -> Question | None: ...

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
class QuestionTypeDescriptor:
    type_id: str
    render_format: str
    supported_levels: frozenset[int]
    interaction_mode: str


@dataclass(frozen=True)
class QuestionDemand:
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
    descriptor: QuestionTypeDescriptor

    async def prepare(
        self, ctx: QuestionContext, demands: Sequence[QuestionDemand]
    ) -> PrepareReport: ...

    async def retrieve(
        self, ctx: QuestionContext, query: QuestionQuery
    ) -> Question | None: ...

    async def evaluate(
        self, ctx: QuestionContext, question: Question, answer: object
    ) -> Evaluation: ...


class ExposureType(Protocol):
    descriptor: QuestionTypeDescriptor

    async def retrieve(self, ctx: QuestionContext, query: QuestionQuery) -> Question: ...


QuestionType: TypeAlias = AssessmentType | ExposureType
QuestionTypeFactory: TypeAlias = Callable[[], QuestionType]
# Keep the package facade importable until its public exports migrate with the engine.
QuestionFormat = QuestionType
FormatSpec = QuestionTypeDescriptor
REGISTRY: dict[str, QuestionType] = {}


def payload_model_for(render_format: str) -> type[BaseModel]:
    """Resolve a render contract to its validator without a constants/schema cycle."""
    model_name = RENDER_FORMAT_PAYLOAD.get(render_format)
    model = getattr(question_schemas, model_name, None) if model_name else None
    if not isinstance(model, type) or not issubclass(model, BaseModel):
        raise ValueError(f"render format {render_format!r} has no payload validator")
    return model


def register(make_plugin: QuestionTypeFactory) -> None:
    """Instantiate and register one type after fail-fast contract validation."""
    plugin = make_plugin()
    descriptor = plugin.descriptor

    if descriptor.type_id not in QUESTION_TYPES:
        raise ValueError(f"unknown question type: {descriptor.type_id!r}")
    if descriptor.render_format not in RENDER_FORMATS:
        raise ValueError(f"unknown render format: {descriptor.render_format!r}")
    if descriptor.interaction_mode not in INTERACTION_MODES:
        raise ValueError(f"unknown interaction mode: {descriptor.interaction_mode!r}")
    if not descriptor.supported_levels or not descriptor.supported_levels <= DIFFICULTY_LEVELS:
        raise ValueError("supported_levels must be a non-empty difficulty-level subset")

    payload_model_for(descriptor.render_format)
    if 0 in descriptor.supported_levels:
        if (
            descriptor.supported_levels != frozenset({0})
            or descriptor.render_format != "flashcard"
            or descriptor.interaction_mode != "exposure"
        ):
            raise ValueError("level 0 types must be flashcard-only exposure types")
        if hasattr(plugin, "evaluate"):
            raise ValueError("an exposure type must not define evaluate")
    else:
        if descriptor.interaction_mode != "assessment":
            raise ValueError("non-zero difficulty types must be assessments")
        if not hasattr(plugin, "evaluate"):
            raise ValueError("an assessment type must define evaluate")

    REGISTRY[descriptor.type_id] = plugin


__all__ = [
    "AssessmentType",
    "ExposureType",
    "NotAssessable",
    "PrepareReport",
    "QuestionContext",
    "QuestionDemand",
    "QuestionQuery",
    "QuestionStore",
    "QuestionType",
    "QuestionTypeDescriptor",
    "REGISTRY",
    "SenseEntryLoader",
    "TtsPort",
    "UnknownQuestionType",
    "payload_model_for",
    "register",
]
