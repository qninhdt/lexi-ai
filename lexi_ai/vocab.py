"""Closed lexical and exercise vocabularies."""

ENTRY_TYPES = frozenset({"word", "phrasal_verb", "idiom", "phrase", "expression"})
GENERATION_STATES = frozenset({"pending", "done", "error"})
POS_TAGS = frozenset(
    {
        "noun",
        "verb",
        "adjective",
        "adverb",
        "pronoun",
        "preposition",
        "conjunction",
        "determiner",
        "interjection",
        "numeral",
        "article",
        "auxiliary",
    }
)
TIERS = ("core", "common", "less_common", "rare")
SLOTS = frozenset(
    {"{sb}", "{sth}", "{one's}", "{oneself}", "{place}", "{doing}", "{do}", "{num}", "{clause}"}
)
INFLECTIONS = frozenset(
    {
        "base",
        "past",
        "past_participle",
        "present_3sg",
        "ing",
        "plural",
        "comparative",
        "superlative",
    }
)
SENSE_REL_TYPES = frozenset({"synonym", "antonym", "hypernym", "hyponym", "meronym", "holonym"})
GENERATED_WORD_REL_TYPES = frozenset({"word_family", "confused_with"})
WORD_REL_TYPES = GENERATED_WORD_REL_TYPES | {"part_of_phrasal_family"}
REL_LEVEL = {**dict.fromkeys(WORD_REL_TYPES, "word"), **dict.fromkeys(SENSE_REL_TYPES, "sense")}
QUESTION_FORMATS = {
    "definition_to_word": frozenset({"single_choice", "single_word"}),
    "context_to_word": frozenset({"single_choice", "single_word"}),
    "cloze_to_word": frozenset({"single_choice", "single_word"}),
    "word_to_definition": frozenset({"single_choice", "short_answer"}),
    "word_to_usage": frozenset({"single_choice", "short_answer"}),
    "dialogue_completion": frozenset({"single_choice"}),
    "meaning_in_context": frozenset({"single_choice"}),
}
QUESTION_TYPES = frozenset(QUESTION_FORMATS)
FORMATS = frozenset({"single_choice", "single_word", "short_answer"})
ALLOWED_PAIRS = frozenset(
    (question_type, fmt) for question_type, formats in QUESTION_FORMATS.items() for fmt in formats
)

_POS_ALIASES = {
    "n": "noun",
    "n.": "noun",
    "v": "verb",
    "v.": "verb",
    "adj": "adjective",
    "adj.": "adjective",
    "adv": "adverb",
    "adv.": "adverb",
    "pron": "pronoun",
    "prep": "preposition",
    "conj": "conjunction",
    "det": "determiner",
    "interj": "interjection",
    "num": "numeral",
    "art": "article",
    "aux": "auxiliary",
    "modal": "auxiliary",
}


def normalize_pos(value: str | None) -> str | None:
    """Normalize known source abbreviations; never guess an unknown POS."""
    if not value:
        return None
    value = value.strip().lower()
    if value in POS_TAGS:
        return value
    return _POS_ALIASES.get(value)
