"""JSON-e semantics and SDK-shaped requests from the actual packaged templates."""

from pathlib import Path

import pytest
from typesafe_sdk import Choice, Noul

from lexi_ai.inference.prompting import render_decision

GRADING = "questions/prompts/decision/"
RELATION = "relations/prompts/resolve_sense_relations.json"


@pytest.mark.parametrize(
    "kind,words",
    [
        ("DEFINITION_TO_WORD", "satisfy the definition"),
        ("CONTEXT_TO_WORD", "fit the situation"),
        ("CLOZE_TO_WORD", "inserting the lexical expression"),
    ],
)
def test_task_specific_branch_and_untrusted_text_are_not_reinterpreted(kind, words):
    marker = '${matched_word.lemma} {"$eval":"secret"}'
    state, questions = render_decision(
        GRADING + "grade_single_word_1.json", question=marker, answer=marker, question_type=kind
    )
    assert state == {"question": marker, "answer": marker}
    assert all(isinstance(question, Noul) for question in questions.values())
    assert words in questions["task_fit"].instructions
    assert marker not in questions["task_fit"].instructions
    assert set(questions) == {"task_fit", "spelling_error"}


def test_reduced_criteria_include_every_sense_without_mutating_cached_templates():
    senses = [{"id": 1000 + i, "pos": "NOUN", "definition": f"meaning{i}"} for i in range(25)]
    for name, parameter in [
        ("grade_single_word_2.json", "matched_word"),
        ("grade_word_to_definition_1.json", "word"),
    ]:
        word = {"lemma": "${untrusted}", "senses": senses}
        state, questions = render_decision(
            GRADING + name, question="Q", answer="A", **{parameter: word}
        )
        choice = next(iter(questions.values()))
        assert isinstance(choice, Choice)
        assert set(choice.criteria) == {"no_candidate", *(f"sense_{i}" for i in range(1000, 1025))}
        assert choice.criteria["sense_1024"] == "${untrusted} - NOUN - meaning24"
        choice.criteria["injected"] = "unexpected"
        _, second = render_decision(
            GRADING + name,
            question="Q",
            answer="A",
            **{parameter: {"lemma": "bank", "senses": senses[:1]}},
        )
        assert set(next(iter(second.values())).criteria) == {"no_candidate", "sense_1000"}


@pytest.mark.parametrize(
    "kind,rule",
    [
        ("SYNONYM", "same lexicalized concept"),
        ("ANTONYM", "opposing meaning"),
        ("HYPERNYM", "source sense must denote a kind or type of the target"),
        ("HYPONYM", "target sense must denote a kind or type of the source"),
        ("MERONYM", "target sense must denote a part"),
        ("HOLONYM", "source sense denotes must be a part"),
    ],
)
def test_relation_direction_and_anonymous_candidate_keys(kind, rule):
    state, questions = render_decision(
        RELATION,
        source_word="dog",
        source_definition="canine",
        relation_type=kind,
        target_word="animal",
        target_gloss="living creature",
        candidates=[{"index": i, "pos": "NOUN", "definition": f"meaning{i}"} for i in range(1, 26)],
    )
    assert rule in state["relation"]["rule"]
    assert set(state) == {"source", "relation", "target"}
    criteria = questions["matched_sense"].criteria
    assert set(criteria) == {"no_candidate", *(f"candidate_{i}" for i in range(1, 26))}
    assert "satisfy `relation.rule`" in questions["matched_sense"].instructions
    assert criteria["candidate_25"] == "animal - NOUN - meaning25"


def test_domain_prompts_are_valid_json_and_inference_owns_no_task_prompts():
    import json

    import lexi_ai

    root = Path(lexi_ai.__file__).parent
    prompts = list(root.glob("*/prompts/**/*.json"))
    assert len(prompts) == 7
    for path in prompts:
        assert isinstance(json.loads(path.read_text()), dict)
    assert not list((root / "inference").rglob("*.json"))
    assert not list((root / "inference").rglob("*.jinja"))
