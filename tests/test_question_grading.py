"""Executable supplied grading design: independent diagnostics and staged gates."""

from dataclasses import asdict
from types import SimpleNamespace

import pytest
from sqlalchemy import event, insert, update
from test_db_optimization import optimized_db as optimized_db
from test_db_optimization import sqlite_db as sqlite_db
from typesafe_sdk import Choice, Noul

from lexi_ai import schema as row
from lexi_ai.cache import Cache
from lexi_ai.errors import (
    InvalidOutputError,
    InvalidResourceError,
    MissingProviderError,
    QuestionNotFoundError,
    UnsupportedQuestionFormatError,
)
from lexi_ai.inference.config import DecisionConfig, DecisionMode
from lexi_ai.models import DefinitionGrade, Option, Question, SingleWordGrade, UsageGrade
from lexi_ai.questions.grade import grade_answer
from lexi_ai.questions.storage import append

CONFIG = DecisionConfig(0.7)


async def test_warm_native_grading_uses_artifact_cache_and_keeps_validation(bank):
    db, questions = bank
    db.question_cache = Cache(64 * 1024)
    single = questions[0]
    await grade_answer(db, None, single.id, "SINGLE_CHOICE", "yes", config=CONFIG)
    statements = []

    def capture(_conn, _cursor, sql, _parameters, _context, _many):
        statements.append(sql)

    event.listen(db.engine.sync_engine, "before_cursor_execute", capture)
    try:
        assert (
            await grade_answer(db, None, single.id, "SINGLE_CHOICE", "yes", config=CONFIG)
        ).task_fit
        with pytest.raises(UnsupportedQuestionFormatError):
            await grade_answer(
                db, None, single.id, "SINGLE_CHOICE", "yes", config=CONFIG, allowed_pairs=set()
            )
        assert statements == []
    finally:
        event.remove(db.engine.sync_engine, "before_cursor_execute", capture)


class Decision:
    def __init__(self, *, choices=None, nouls=None):
        self.calls = []
        self.choices = {
            "matched_sense": "no_candidate",
            "defined_meaning": "no_candidate",
            "accuracy": "accurate",
            "coverage": "sufficient",
            "meaning": "correct",
            "form": "correct",
            "collocation": "natural",
            "appropriacy": "appropriate",
            **(choices or {}),
        }
        self.nouls = {
            "task_fit": 0.9,
            "spelling_error": 0.1,
            "used": 0.9,
            "construction": 0.9,
            **(nouls or {}),
        }

    async def decide(self, state, questions, *, mode=DecisionMode.LLM_FALLBACK):
        self.calls.append((state, questions))
        return SimpleNamespace(
            choices={
                name: SimpleNamespace(choice=self.choices[name])
                for name, q in questions.items()
                if isinstance(q, Choice)
            },
            nouls={
                name: SimpleNamespace(noul=self.nouls[name])
                for name, q in questions.items()
                if isinstance(q, Noul)
            },
        )


@pytest.fixture
async def bank(optimized_db):
    db = optimized_db
    async with db.transaction() as session:
        await session.execute(
            insert(row.Word),
            [
                dict(
                    id=1, lemma="bank", match_key="bank", entry_type="WORD", generation_state="DONE"
                ),
                dict(
                    id=2,
                    lemma="lender",
                    match_key="lender",
                    entry_type="WORD",
                    generation_state="DONE",
                ),
            ],
        )
        await session.execute(
            insert(row.Sense),
            [
                dict(id=1, word_id=1, pos="NOUN", tier="CORE"),
                dict(id=2, word_id=1, pos="NOUN", tier="COMMON"),
            ]
            + [dict(id=200 + i, word_id=2, pos="NOUN", tier="CORE") for i in range(20)],
        )
        await session.execute(
            insert(row.Definition),
            [
                dict(sense_id=1, content="Financial institution"),
                dict(sense_id=2, content="Land along a river"),
            ]
            + [dict(sense_id=200 + i, content=f"lender meaning{i}") for i in range(20)],
        )
    questions = await append(
        db,
        [
            Question(
                0,
                1,
                None,
                "DEFINITION_TO_WORD",
                "Financial institution",
                Option("yes", "bank", "Fits"),
                [Option("no", "ship", "Wrong")],
            ),
            Question(
                0,
                1,
                None,
                "WORD_TO_DEFINITION",
                '<t inf="base">bank</t>',
                Option("yes", "Financial institution", "Fits"),
                [],
            ),
            Question(
                0,
                1,
                None,
                "WORD_TO_USAGE",
                '<t inf="base">bank</t> — Financial institution',
                Option("yes", "The bank opens.", "Fits"),
                [],
            ),
        ],
    )
    from lexi_ai.words.search import Search

    db.search_index = Search(db, None)
    await db.search_index.start()
    return db, questions


async def test_choice_and_saved_surface_are_provider_free(bank):
    db, (single, _, _) = bank
    assert await grade_answer(
        db, None, single.id, "SINGLE_CHOICE", "yes", config=CONFIG
    ) == SingleWordGrade(True, False, 1)
    assert await grade_answer(
        db, None, single.id, "SINGLE_CHOICE", "no", config=CONFIG
    ) == SingleWordGrade(False, False, None)
    assert await grade_answer(
        db, None, single.id, "SINGLE_WORD", " BANK ", config=CONFIG
    ) == SingleWordGrade(True, False, 1)
    with pytest.raises(InvalidResourceError, match="option ID"):
        await grade_answer(db, None, single.id, "SINGLE_CHOICE", "forged", config=CONFIG)
    with pytest.raises(InvalidResourceError, match="unsupported"):
        await grade_answer(db, None, single.id, "SHORT_ANSWER", "bank", config=CONFIG)
    with pytest.raises(MissingProviderError):
        await grade_answer(db, None, single.id, "SINGLE_WORD", "other", config=CONFIG)
    with pytest.raises(QuestionNotFoundError):
        await grade_answer(db, None, 999999, "SINGLE_WORD", "bank", config=CONFIG)
    with pytest.raises(UnsupportedQuestionFormatError):
        await grade_answer(
            db, None, single.id, "SINGLE_WORD", "bank", config=CONFIG, allowed_pairs=set()
        )


@pytest.mark.parametrize("fit,typo", [(False, False), (False, True), (True, True)])
async def test_single_word_diagnostics_are_independent_and_gate_search(
    bank, monkeypatch, fit, typo
):
    db, (single, _, _) = bank

    async def unexpected(*args, **kwargs):
        pytest.fail("gated answer reached dictionary search")

    monkeypatch.setattr("lexi_ai.questions.grade.search", unexpected)
    model = Decision(
        nouls={"task_fit": 0.9 if fit else 0.1, "spelling_error": 0.9 if typo else 0.1}
    )
    result = await grade_answer(db, model, single.id, "SINGLE_WORD", "dgo", config=CONFIG)
    assert asdict(result) == {"task_fit": fit, "spelling_error": typo, "sense_id": None}
    assert len(model.calls) == 1
    assert set(model.calls[0][1]) == {"task_fit", "spelling_error"}
    assert model.calls[0][0] == {"question": "Financial institution", "answer": "dgo"}


async def test_single_word_searches_top_one_and_passes_all_its_senses(bank):
    db, (single, _, _) = bank
    model = Decision(choices={"matched_sense": "sense_219"})
    result = await grade_answer(db, model, single.id, "SINGLE_WORD", "lender", config=CONFIG)
    assert result == SingleWordGrade(True, False, 219)
    assert len(model.calls) == 2
    criteria = model.calls[1][1]["matched_sense"].criteria
    assert set(criteria) == {"no_candidate", *(f"sense_{i}" for i in range(200, 220))}
    assert all("bank -" not in text for text in criteria.values())


async def test_missing_dictionary_sense_does_not_change_correctness(bank):
    db, (single, _, _) = bank
    model = Decision()
    result = await grade_answer(db, model, single.id, "SINGLE_WORD", "lender", config=CONFIG)
    assert result == SingleWordGrade(True, False, None)
    assert len(model.calls) == 2
    model = Decision()
    result = await grade_answer(db, model, single.id, "SINGLE_WORD", "zzzzzzzzz", config=CONFIG)
    assert result == SingleWordGrade(True, False, None)
    assert len(model.calls) == 1


async def test_definition_resolves_intent_then_grades_only_selected_meaning(bank):
    db, (_, definition, _) = bank
    model = Decision(
        choices={"defined_meaning": "sense_2", "accuracy": "mixed", "coverage": "partial"}
    )
    result = await grade_answer(
        db, model, definition.id, "SHORT_ANSWER", "a shore, sometimes water", config=CONFIG
    )
    assert result == DefinitionGrade(2, "MIXED", "PARTIAL")
    assert len(model.calls) == 2
    assert set(model.calls[0][1]) == {"defined_meaning"}
    assert set(model.calls[0][1]["defined_meaning"].criteria) == {
        "no_candidate",
        "sense_1",
        "sense_2",
    }
    assert model.calls[1][0] == {
        "word": "bank",
        "meaning": "Land along a river",
        "answer": "a shore, sometimes water",
    }
    assert set(model.calls[1][1]) == {"accuracy", "coverage"}
    assert set(asdict(result)) == {"sense_id", "accuracy", "coverage"}


@pytest.mark.parametrize("mode", list(DecisionMode))
async def test_explicit_mode_is_forwarded_to_each_grading_stage(bank, mode):
    db, (single, definition, usage) = bank

    class RoutedDecision(Decision):
        async def decide(self, state, questions, *, mode):
            assert mode == selected_mode
            return await super().decide(state, questions)

    selected_mode = mode
    model = RoutedDecision(choices={"defined_meaning": "sense_1"})
    for question, fmt, answer in (
        (single, "SINGLE_WORD", "lender"),
        (definition, "SHORT_ANSWER", "A financial institution"),
        (usage, "SHORT_ANSWER", "The bank opens early."),
    ):
        await grade_answer(db, model, question.id, fmt, answer, config=CONFIG, mode=mode)
    assert len(model.calls) == 6


async def test_unidentified_definition_stops_before_diagnostic_query(bank):
    db, (_, definition, _) = bank
    model = Decision()
    assert await grade_answer(
        db, model, definition.id, "SHORT_ANSWER", "unrelated", config=CONFIG
    ) == DefinitionGrade(None, None, None)
    assert len(model.calls) == 1


async def test_usage_gate_stops_without_dictionary_or_second_model_call(bank):
    db, (_, _, usage) = bank
    model = Decision(nouls={"used": 0.2})
    assert await grade_answer(
        db, model, usage.id, "SHORT_ANSWER", "a lender opens", config=CONFIG
    ) == UsageGrade(False, None, None, None, None, None)
    assert len(model.calls) == 1
    assert set(model.calls[0][1]) == {"used"}


async def test_usage_preserves_all_diagnostics_and_saved_anchor(bank):
    db, (_, _, usage) = bank
    async with db.transaction() as session:
        await session.execute(
            update(row.Definition)
            .where(row.Definition.sense_id == 1)
            .values(content="Edited later")
        )
    model = Decision(
        choices={
            "meaning": "wrong",
            "form": "spelling_error",
            "collocation": "acceptable",
            "appropriacy": "marked",
        },
        nouls={"construction": 0.1},
    )
    statements = []

    def count(_c, _u, sql, _p, _x, _many):
        statements.append(sql)

    event.listen(db.engine.sync_engine, "before_cursor_execute", count)
    try:
        result = await grade_answer(
            db, model, usage.id, "SHORT_ANSWER", "The banck grew tall.", config=CONFIG
        )
    finally:
        event.remove(db.engine.sync_engine, "before_cursor_execute", count)
    assert result == UsageGrade(True, "WRONG", "SPELLING_ERROR", False, "ACCEPTABLE", "MARKED")
    assert len(statements) == 1
    assert model.calls[1][0]["meaning"] == "Financial institution"
    assert set(model.calls[1][1]) == {
        "meaning",
        "form",
        "construction",
        "collocation",
        "appropriacy",
    }
    assert set(asdict(result)) == {
        "used",
        "meaning",
        "form",
        "construction",
        "collocation",
        "appropriacy",
    }


async def test_invalid_sense_choice_is_not_silently_no_candidate(bank):
    db, (_, definition, _) = bank
    with pytest.raises(InvalidOutputError):
        await grade_answer(
            db,
            Decision(choices={"defined_meaning": "sense_999"}),
            definition.id,
            "SHORT_ANSWER",
            "shore",
            config=CONFIG,
        )


async def test_threshold_boundary_is_inclusive_and_typo_is_not_correctness(bank):
    db, (single, _, usage) = bank
    model = Decision(nouls={"task_fit": 0.7, "spelling_error": 0.7})
    assert await grade_answer(
        db, model, single.id, "SINGLE_WORD", "banck", config=CONFIG
    ) == SingleWordGrade(True, True, None)
    model = Decision(nouls={"used": 0.7, "construction": 0.7})
    result = await grade_answer(
        db, model, usage.id, "SHORT_ANSWER", "The bank opens.", config=CONFIG
    )
    assert result.used and result.construction
