"""Retained benchmark files remain historical; only detached inputs are adapted."""

from copy import deepcopy

import pytest

from benchmarks.run import prepare_case
from lexi_ai import QuestionType, ResponseFormat
from lexi_ai.models import Question, Word


def test_historical_question_case_is_adapted_without_mutating_evidence():
    case = {
        "id": "historical-case",
        "task": "grade_single_word_1",
        "input": {"question_type": "cloze_to_word", "question": "Please _.", "answer": "bank"},
        "expected": {"task_fit": True, "spelling_error": False},
    }
    original = deepcopy(case)
    state, questions = prepare_case(case)
    assert case == original
    assert state == {"question": "Please _.", "answer": "bank"}
    assert "inserting the lexical expression" in questions["task_fit"].instructions


def test_historical_pos_is_adapted_without_mutating_inventory():
    case = {
        "id": "historical-inventory",
        "task": "grade_single_word_2",
        "input": {
            "question": "A money keeper",
            "answer": "bank",
            "matched_word": {
                "lemma": "bank",
                "senses": [{"id": 1, "pos": "noun", "definition": "Money keeper"}],
            },
        },
        "expected": {"matched_sense": "sense_1"},
    }
    original = deepcopy(case)
    _, questions = prepare_case(case)
    assert questions["matched_sense"].criteria["sense_1"] == "bank - NOUN - Money keeper"
    assert case == original


def test_evidence_adapter_does_not_add_public_legacy_aliases():
    for enum, value in ((QuestionType, "cloze_to_word"), (ResponseFormat, "single_word")):
        with pytest.raises(ValueError):
            enum(value)
    with pytest.raises(ValueError):
        Word(1, "bank", "word", "done")
    with pytest.raises(ValueError):
        Question(1, 1, None, "cloze_to_word", "Please _.", None, [])
