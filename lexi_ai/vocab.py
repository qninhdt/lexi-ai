"""Canonical controlled vocabularies shared by the library and its consumers."""

from enum import StrEnum


class QuestionType(StrEnum):
    DEFINITION_TO_WORD = "DEFINITION_TO_WORD"
    CONTEXT_TO_WORD = "CONTEXT_TO_WORD"
    CLOZE_TO_WORD = "CLOZE_TO_WORD"
    WORD_TO_DEFINITION = "WORD_TO_DEFINITION"
    WORD_TO_USAGE = "WORD_TO_USAGE"
    DIALOGUE_COMPLETION = "DIALOGUE_COMPLETION"
    MEANING_IN_CONTEXT = "MEANING_IN_CONTEXT"


FIXED_STEM_QUESTION_TYPES = frozenset(
    {
        QuestionType.DEFINITION_TO_WORD,
        QuestionType.WORD_TO_DEFINITION,
        QuestionType.WORD_TO_USAGE,
    }
)


class ResponseFormat(StrEnum):
    SINGLE_CHOICE = "SINGLE_CHOICE"
    SINGLE_WORD = "SINGLE_WORD"
    SHORT_ANSWER = "SHORT_ANSWER"


class TargetPlacement(StrEnum):
    DIALOGUE = "DIALOGUE"
    OPTIONS = "OPTIONS"


class EntryType(StrEnum):
    WORD = "WORD"
    PHRASAL_VERB = "PHRASAL_VERB"
    IDIOM = "IDIOM"
    PHRASE = "PHRASE"
    EXPRESSION = "EXPRESSION"


class GenerationState(StrEnum):
    PENDING = "PENDING"
    DONE = "DONE"
    ERROR = "ERROR"


class PartOfSpeech(StrEnum):
    NOUN = "NOUN"
    VERB = "VERB"
    ADJECTIVE = "ADJECTIVE"
    ADVERB = "ADVERB"
    PRONOUN = "PRONOUN"
    PREPOSITION = "PREPOSITION"
    CONJUNCTION = "CONJUNCTION"
    DETERMINER = "DETERMINER"
    INTERJECTION = "INTERJECTION"
    NUMERAL = "NUMERAL"
    ARTICLE = "ARTICLE"
    AUXILIARY = "AUXILIARY"


class Tier(StrEnum):
    CORE = "CORE"
    COMMON = "COMMON"
    LESS_COMMON = "LESS_COMMON"
    RARE = "RARE"


class Inflection(StrEnum):
    BASE = "BASE"
    PAST = "PAST"
    PAST_PARTICIPLE = "PAST_PARTICIPLE"
    PRESENT_3SG = "PRESENT_3SG"
    ING = "ING"
    PLURAL = "PLURAL"
    COMPARATIVE = "COMPARATIVE"
    SUPERLATIVE = "SUPERLATIVE"


class SenseRelationType(StrEnum):
    SYNONYM = "SYNONYM"
    ANTONYM = "ANTONYM"
    HYPERNYM = "HYPERNYM"
    HYPONYM = "HYPONYM"
    MERONYM = "MERONYM"
    HOLONYM = "HOLONYM"


class WordRelationType(StrEnum):
    WORD_FAMILY = "WORD_FAMILY"
    CONFUSED_WITH = "CONFUSED_WITH"
    PART_OF_PHRASAL_FAMILY = "PART_OF_PHRASAL_FAMILY"


class ResolutionState(StrEnum):
    PENDING = "PENDING"
    RESOLVED = "RESOLVED"
    UNRESOLVABLE = "UNRESOLVABLE"
    NOOP = "NOOP"
    ERROR = "ERROR"


class CEFRLevel(StrEnum):
    A1 = "A1"
    A2 = "A2"
    B1 = "B1"
    B2 = "B2"
    C1 = "C1"
    C2 = "C2"


class Register(StrEnum):
    FORMAL = "FORMAL"
    INFORMAL = "INFORMAL"
    SLANG = "SLANG"
    LITERARY = "LITERARY"
    SPECIALIST = "SPECIALIST"


class MatchKind(StrEnum):
    EXACT = "EXACT"
    PREFIX = "PREFIX"
    SUBSTRING = "SUBSTRING"
    FUZZY = "FUZZY"


class DefinitionAccuracy(StrEnum):
    ACCURATE = "ACCURATE"
    MIXED = "MIXED"
    INACCURATE = "INACCURATE"


class DefinitionCoverage(StrEnum):
    SUFFICIENT = "SUFFICIENT"
    PARTIAL = "PARTIAL"
    MINIMAL = "MINIMAL"


class UsageMeaning(StrEnum):
    CORRECT = "CORRECT"
    APPROXIMATE = "APPROXIMATE"
    WRONG = "WRONG"


class UsageForm(StrEnum):
    CORRECT = "CORRECT"
    SPELLING_ERROR = "SPELLING_ERROR"
    FORM_ERROR = "FORM_ERROR"


class UsageCollocation(StrEnum):
    NATURAL = "NATURAL"
    ACCEPTABLE = "ACCEPTABLE"
    UNNATURAL = "UNNATURAL"


class UsageAppropriacy(StrEnum):
    APPROPRIATE = "APPROPRIATE"
    MARKED = "MARKED"
    INAPPROPRIATE = "INAPPROPRIATE"


# Derived capabilities, never independently maintained token lists.
ENTRY_TYPES = frozenset(EntryType)
GENERATION_STATES = frozenset(GenerationState)
POS_TAGS = frozenset(PartOfSpeech)
TIERS = tuple(Tier)
INFLECTIONS = frozenset(Inflection)
SENSE_REL_TYPES = frozenset(SenseRelationType)
GENERATED_WORD_REL_TYPES = frozenset({WordRelationType.WORD_FAMILY, WordRelationType.CONFUSED_WITH})
WORD_REL_TYPES = frozenset(WordRelationType)
REL_LEVEL = {**dict.fromkeys(WORD_REL_TYPES, "word"), **dict.fromkeys(SENSE_REL_TYPES, "sense")}
SLOTS = frozenset(
    {
        "{sb}",
        "{sth}",
        "{one's}",
        "{oneself}",
        "{place}",
        "{doing}",
        "{do}",
        "{done}",
        "{adj}",
        "{adv}",
        "{num}",
        "{clause}",
    }
)
QUESTION_FORMATS = {
    QuestionType.DEFINITION_TO_WORD: frozenset(
        {ResponseFormat.SINGLE_CHOICE, ResponseFormat.SINGLE_WORD}
    ),
    QuestionType.CONTEXT_TO_WORD: frozenset(
        {ResponseFormat.SINGLE_CHOICE, ResponseFormat.SINGLE_WORD}
    ),
    QuestionType.CLOZE_TO_WORD: frozenset(
        {ResponseFormat.SINGLE_CHOICE, ResponseFormat.SINGLE_WORD}
    ),
    QuestionType.WORD_TO_DEFINITION: frozenset(
        {ResponseFormat.SINGLE_CHOICE, ResponseFormat.SHORT_ANSWER}
    ),
    QuestionType.WORD_TO_USAGE: frozenset(
        {ResponseFormat.SINGLE_CHOICE, ResponseFormat.SHORT_ANSWER}
    ),
    QuestionType.DIALOGUE_COMPLETION: frozenset({ResponseFormat.SINGLE_CHOICE}),
    QuestionType.MEANING_IN_CONTEXT: frozenset({ResponseFormat.SINGLE_CHOICE}),
}
QUESTION_TYPES = frozenset(QuestionType)
FORMATS = frozenset(ResponseFormat)
ALLOWED_PAIRS = frozenset(
    (question_type, fmt) for question_type, formats in QUESTION_FORMATS.items() for fmt in formats
)

_POS_ALIASES = {
    "n": PartOfSpeech.NOUN,
    "n.": PartOfSpeech.NOUN,
    "v": PartOfSpeech.VERB,
    "v.": PartOfSpeech.VERB,
    "adj": PartOfSpeech.ADJECTIVE,
    "adj.": PartOfSpeech.ADJECTIVE,
    "adv": PartOfSpeech.ADVERB,
    "adv.": PartOfSpeech.ADVERB,
    "pron": PartOfSpeech.PRONOUN,
    "prep": PartOfSpeech.PREPOSITION,
    "conj": PartOfSpeech.CONJUNCTION,
    "det": PartOfSpeech.DETERMINER,
    "interj": PartOfSpeech.INTERJECTION,
    "num": PartOfSpeech.NUMERAL,
    "art": PartOfSpeech.ARTICLE,
    "aux": PartOfSpeech.AUXILIARY,
    "modal": PartOfSpeech.AUXILIARY,
    # WordNet's externally specified one-letter POS codes.
    "a": PartOfSpeech.ADJECTIVE,
    "s": PartOfSpeech.ADJECTIVE,
    "r": PartOfSpeech.ADVERB,
}


def normalize_pos(value: str | None) -> PartOfSpeech | None:
    """Normalize external corpus POS labels; unknown labels are not guessed."""
    if not value:
        return None
    value = value.strip()
    try:
        return PartOfSpeech(value.upper())
    except ValueError:
        return _POS_ALIASES.get(value.lower())
