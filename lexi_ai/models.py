"""Detached public values. No database entities or provider objects cross this boundary."""

from dataclasses import dataclass, field

from .vocab import ALLOWED_PAIRS


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
    inf: str


@dataclass(frozen=True)
class SenseRelation:
    rel_type: str
    to_word_id: int
    to_word_lemma: str
    resolution_state: str
    to_sense_id: int | None = None


@dataclass(frozen=True)
class WordRelation:
    rel_type: str
    to_word_id: int
    to_word_lemma: str


@dataclass(frozen=True)
class Sense:
    id: int
    word_id: int
    pos: str
    tier: str
    definition: Definition | None = None
    examples: list[Example] = field(default_factory=list)
    forms: list[Form] = field(default_factory=list)
    patterns: list[str] = field(default_factory=list)
    collocations: list[str] = field(default_factory=list)
    relations: list[SenseRelation] = field(default_factory=list)
    cefr_level: str | None = None
    register: str | None = None
    usage_note: str | None = None
    ipa_uk: str | None = None
    ipa_us: str | None = None


@dataclass(frozen=True)
class Word:
    id: int
    lemma: str
    entry_type: str
    generation_state: str
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
    """Trusted-consumer artifact: answer and explanations must not be sent to learners."""

    id: int
    sense_id: int
    theme_id: int | None
    question_type: str
    content: str | list[dict[str, str | None]]
    correct: Option
    distractors: list[Option]
    correct_alternatives: list[str] = field(default_factory=list)
    target_placement: str | None = None

    def supports(self, fmt: str) -> bool:
        return (self.question_type, fmt) in ALLOWED_PAIRS


@dataclass(frozen=True)
class SingleWordGrade:
    task_fit: bool
    spelling_error: bool
    sense_id: int | None


@dataclass(frozen=True)
class DefinitionGrade:
    sense_id: int | None
    accuracy: str | None
    coverage: str | None


@dataclass(frozen=True)
class UsageGrade:
    used: bool
    meaning: str | None
    form: str | None
    construction: bool | None
    collocation: str | None
    appropriacy: str | None


@dataclass(frozen=True)
class WordHit:
    word_id: int
    lemma: str
    entry_type: str
    match_kind: str
    matched_surface: str


@dataclass(frozen=True)
class AvailableHit:
    available_id: str
    display: str
    entry_type: str


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
