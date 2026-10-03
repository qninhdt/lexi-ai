"""Detached public values. No database entities or provider objects cross this boundary."""

from dataclasses import field

from pydantic.dataclasses import dataclass

from .vocab import (
    ALLOWED_PAIRS,
    CEFRLevel,
    DefinitionAccuracy,
    DefinitionCoverage,
    EntryType,
    GenerationState,
    Inflection,
    MatchKind,
    PartOfSpeech,
    QuestionType,
    Register,
    ResolutionState,
    ResponseFormat,
    SenseRelationType,
    TargetPlacement,
    Tier,
    UsageAppropriacy,
    UsageCollocation,
    UsageForm,
    UsageMeaning,
    WordRelationType,
)


@dataclass(frozen=True)
class TokenUsage:
    """Provider-reported counts; input includes cached tokens, None means unreported."""

    model_id: str | None
    input_tokens: int | None
    cache_read_tokens: int | None
    cache_write_tokens: int | None
    output_tokens: int | None


@dataclass(frozen=True)
class Definition:
    id: int
    content: str
    theme_id: int | None = None


@dataclass(frozen=True)
class Example:
    id: int
    content: str
    theme_id: int | None = None


@dataclass(frozen=True)
class Form:
    surface: str
    inf: Inflection


@dataclass(frozen=True)
class SenseRelation:
    rel_type: SenseRelationType
    to_word_id: int
    to_word_lemma: str
    resolution_state: ResolutionState
    to_sense_id: int | None = None


@dataclass(frozen=True)
class WordRelation:
    rel_type: WordRelationType
    to_word_id: int
    to_word_lemma: str


@dataclass(frozen=True)
class Sense:
    id: int
    word_id: int
    pos: PartOfSpeech
    tier: Tier
    definition: Definition | None = None
    examples: list[Example] = field(default_factory=list)
    forms: list[Form] = field(default_factory=list)
    patterns: list[str] = field(default_factory=list)
    collocations: list[str] = field(default_factory=list)
    relations: list[SenseRelation] = field(default_factory=list)
    cefr_level: CEFRLevel | None = None
    register: Register | None = None
    usage_note: str | None = None
    ipa_uk: str | None = None
    ipa_us: str | None = None


@dataclass(frozen=True)
class Word:
    id: int
    lemma: str
    type: EntryType
    generation_state: GenerationState
    senses: list[Sense] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)
    related: list[WordRelation] = field(default_factory=list)


@dataclass(frozen=True)
class Theme:
    id: int
    key: str
    name: str
    voice: str
    diction: str


@dataclass(frozen=True)
class Option:
    id: str
    content: str
    explanation: str


@dataclass(frozen=True)
class Question:
    """Full artifact; consumers choose self-learning or exam answer-key visibility."""

    id: int
    sense_id: int
    theme_id: int | None
    question_type: QuestionType
    content: str | list[dict[str, str | None]]
    correct: Option
    distractors: list[Option]
    correct_alternatives: list[str] = field(default_factory=list)
    target_placement: TargetPlacement | None = None

    def supports(self, fmt: ResponseFormat) -> bool:
        return (self.question_type, fmt) in ALLOWED_PAIRS


@dataclass(frozen=True)
class SingleWordGrade:
    task_fit: bool
    spelling_error: bool
    sense_id: int | None


@dataclass(frozen=True)
class DefinitionGrade:
    sense_id: int | None
    accuracy: DefinitionAccuracy | None
    coverage: DefinitionCoverage | None


@dataclass(frozen=True)
class UsageGrade:
    used: bool
    meaning: UsageMeaning | None
    form: UsageForm | None
    construction: bool | None
    collocation: UsageCollocation | None
    appropriacy: UsageAppropriacy | None


@dataclass(frozen=True)
class WordHit:
    word_id: int
    lemma: str
    entry_type: EntryType
    match_kind: MatchKind
    matched_surface: str


@dataclass(frozen=True)
class AvailableHit:
    available_id: str
    display: str
    entry_type: EntryType


@dataclass(frozen=True)
class SearchResult:
    words: list[WordHit] = field(default_factory=list)
    available: list[AvailableHit] = field(default_factory=list)


@dataclass(frozen=True)
class Translation:
    id: int
    input_hash: str
    target_language: str
    content: str
