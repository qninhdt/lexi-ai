from dataclasses import fields

from lexi_ai.models import (
    Definition,
    DefinitionGrade,
    Question,
    SearchResult,
    Sense,
    SenseRelation,
    SingleWordGrade,
    UsageGrade,
    Word,
)
from lexi_ai.vocab import (
    ALLOWED_PAIRS,
    GENERATION_STATES,
    POS_TAGS,
    QUESTION_TYPES,
    REL_LEVEL,
    SLOTS,
    TIERS,
    normalize_pos,
)


def test_closed_contract():
    assert len(QUESTION_TYPES) == 7
    assert len(ALLOWED_PAIRS) == 12
    assert len(SLOTS) == 9
    assert len(POS_TAGS) == 12
    assert TIERS == ("core", "common", "extended", "rare")
    assert GENERATION_STATES == {"pending", "done", "error"}
    assert "see_also" not in REL_LEVEL
    assert "part_of_phrasal_family" in REL_LEVEL
    assert "flashcard" not in QUESTION_TYPES


def test_pos_normalization_keeps_known_aliases_and_does_not_guess():
    for pos in POS_TAGS:
        assert normalize_pos(f" {pos.upper()} ") == pos
    assert normalize_pos(" N. ") == "noun"
    assert normalize_pos("adj.") == "adjective"
    assert normalize_pos("modal") == "auxiliary"
    assert normalize_pos("unknown") is None
    assert normalize_pos("") is None
    assert normalize_pos(None) is None


def test_public_shapes_do_not_carry_legacy_fields():
    assert not {"pos", "norm", "status", "cambridge_id", "units"} & {
        field.name for field in fields(Word)
    }
    assert not {"domain", "grammar", "guideword", "connotation", "definitions", "position"} & {
        field.name for field in fields(Sense)
    }
    assert {field.name for field in fields(SingleWordGrade)} == {
        "task_fit",
        "spelling_error",
        "sense_id",
    }
    assert {field.name for field in fields(DefinitionGrade)} == {"sense_id", "accuracy", "coverage"}
    assert {field.name for field in fields(UsageGrade)} == {
        "used",
        "meaning",
        "form",
        "construction",
        "collocation",
        "appropriacy",
    }
    assert {field.name for field in fields(Definition)} == {"id", "content", "theme_id"}
    assert {field.name for field in fields(SearchResult)} == {"words", "available"}
    assert {field.name for field in fields(SenseRelation)} == {
        "rel_type",
        "to_word_id",
        "to_word_lemma",
        "resolution_state",
        "to_sense_id",
    }
    question = Question(1, 2, None, "dialogue_completion", "x", None, [])
    assert question.supports("single_word") is False
