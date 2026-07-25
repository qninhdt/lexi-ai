"""Tests for controlled-vocab constants (Phase 1: POS controlled vocab).

``POS_TAGS`` closes the previously-free ``pos`` field; ``normalize_pos`` folds
loose/legacy surface forms to a canonical label (and returns ``None`` for junk,
never guessing) so the WSD POS-filter (Phase 4) can compare both sides safely.
"""

from lexi_ai.constants import (
    DIFFICULTY_LEVELS,
    EXPOSURE_LEVEL,
    INTERACTION_MODES,
    POS_TAGS,
    QUESTION_TYPES,
    REL_LEVEL,
    REL_TYPES,
    RENDER_FORMATS,
    SENSE_REL_TYPES,
    WORD_REL_TYPES,
    normalize_pos,
)


def test_pos_tags_has_twelve_labels():
    assert len(POS_TAGS) == 12
    assert POS_TAGS == {
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


def test_normalize_pos_maps_aliases():
    assert normalize_pos("adj") == "adjective"
    assert normalize_pos("N") == "noun"
    assert normalize_pos("v.") == "verb"
    assert normalize_pos("ADV") == "adverb"
    assert normalize_pos("prep") == "preposition"


def test_normalize_pos_accepts_exact_vocab():
    for tag in POS_TAGS:
        assert normalize_pos(tag) == tag
        assert normalize_pos(tag.upper()) == tag


def test_normalize_pos_returns_none_for_junk():
    assert normalize_pos("garbage") is None
    assert normalize_pos("") is None
    assert normalize_pos(None) is None
    assert normalize_pos("   ") is None


def test_normalize_pos_ambiguous_aliases_return_none():
    # 3.8a: "a" (article in most contexts) and "int" (integer abbreviation) are
    # intentionally absent from _POS_ALIASES so the function never guesses a wrong
    # POS. Use "adj" / "interj" instead of the dropped shorthands.
    assert normalize_pos("a") is None, "'a' must not guess 'adjective'"
    assert normalize_pos("int") is None, "'int' must not guess 'interjection'"


# --- REL_LEVEL routing table (Phase 3) ------------------------------------


def test_rel_level_covers_every_rel_type():
    # Invariant: no rel_type is orphaned from a level — the router must classify
    # every relation the LM can emit as word- or sense-level.
    assert set(REL_LEVEL) == set(REL_TYPES)
    assert all(level in ("word", "sense") for level in REL_LEVEL.values())


def test_word_and_sense_rel_types_partition_rel_types():
    assert WORD_REL_TYPES.isdisjoint(SENSE_REL_TYPES)
    assert WORD_REL_TYPES | SENSE_REL_TYPES == REL_TYPES


def test_sense_rel_types_are_the_semantic_relations():
    assert SENSE_REL_TYPES == {
        "synonym",
        "antonym",
        "hypernym",
        "hyponym",
        "meronym",
        "holonym",
        "see_also",
    }


# --- question engine contracts -------------------------------------------


def test_question_engine_vocabularies_are_canonical():
    assert QUESTION_TYPES == {
        "flashcard",
        "definition_mcq",
        "contextual_mcq",
        "cloze",
        "use_in_sentence",
    }
    assert RENDER_FORMATS == {"flashcard", "single_choice", "text_span", "free_text"}
    assert DIFFICULTY_LEVELS == {0, 1, 2, 3, 4}
    assert EXPOSURE_LEVEL == 0
    assert INTERACTION_MODES == {"exposure", "assessment"}


def test_legacy_question_vocabularies_are_removed():
    import lexi_ai.constants as constants

    assert not hasattr(constants, "QUESTION_FORMATS")
    assert not hasattr(constants, "ANSWER_KINDS")
