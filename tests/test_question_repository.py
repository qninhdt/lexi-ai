"""Focused persistence acceptance tests for the question type engine."""

from dataclasses import replace

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import StaticPool

from lexi_ai.contracts.questions import RenderKind
from lexi_ai.db import create_session_factory, init_models, session_scope
from lexi_ai.domain.questions import PersistedQuestion
from lexi_ai.infrastructure.db.models import Question as QuestionRow
from lexi_ai.infrastructure.db.models import Sense, Word
from lexi_ai.questions.repository import QuestionRepository


@pytest.fixture
async def question_repo():
    engine = create_async_engine(
        "sqlite+aiosqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    await init_models(engine)
    session_factory = create_session_factory(engine)
    async with session_scope(session_factory) as session:
        word = Word(norm="eloquent", match_key="eloquent", status="done")
        session.add(word)
        await session.flush()
        core = Sense(word_id=word.id, definition="persuasive", tier="core")
        other = Sense(word_id=word.id, definition="expressive", tier="common")
        session.add_all([core, other])
        await session.flush()
        ids = (word.id, core.id, other.id)
    try:
        yield QuestionRepository(session_factory), session_factory, ids
    finally:
        await engine.dispose()


def _question(word_id: int, sense_id: int | None, **overrides) -> PersistedQuestion:
    values = {
        "question_id": None,
        "word_id": word_id,
        "sense_id": sense_id,
        "type_id": "definition_mcq",
        "render_kind": RenderKind.SINGLE_CHOICE,
        "difficulty_level": 1,
        "interaction": "assessment",
        "payload": {"stem": "Meaning?", "options": ["b", "a"], "correct_index": 1},
    }
    values.update(overrides)
    return PersistedQuestion(**values)


async def test_insert_round_trips_carrier_maps_id_and_is_idempotent(question_repo):
    repo, session_factory, (word_id, sense_id, _) = question_repo
    question = _question(word_id, sense_id)

    first = await repo.insert(question)
    second = await repo.insert(question)

    assert first == second
    assert first.question_id is not None
    assert first.type_id == "definition_mcq"
    assert first.render_kind is RenderKind.SINGLE_CHOICE
    assert first.difficulty_level == 1
    assert first.interaction == "assessment"
    async with session_scope(session_factory) as session:
        count = await session.scalar(select(func.count()).select_from(QuestionRow))
        row = (await session.execute(select(QuestionRow))).scalar_one()
    assert count == 1
    # Storage stays FLAT with the SAME canonical json + content_hash (unchanged).
    assert row.render_format == "single_choice"
    assert row.payload == ('{"correct_index":1,"options":["b","a"],"stem":"Meaning?"}')
    assert len(row.content_hash) == 64


@pytest.mark.parametrize(
    "payload",
    [
        {"nested": {"value": "bad\x00value"}},
        {"nested": ["ok", {"key\x00": "value"}]},
    ],
)
async def test_insert_rejects_recursive_nul(question_repo, payload):
    repo, _, (word_id, sense_id, _) = question_repo
    with pytest.raises(ValueError, match="NUL"):
        await repo.insert(_question(word_id, sense_id, payload=payload))


async def test_insert_rejects_payload_over_utf8_byte_limit(question_repo):
    repo, _, (word_id, sense_id, _) = question_repo
    with pytest.raises(ValueError, match="65,536"):
        await repo.insert(_question(word_id, sense_id, payload={"text": "é" * 40_000}))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("type_id", "unknown"),
        ("difficulty_level", 9),
        ("interaction", "unknown"),
    ],
)
async def test_insert_rejects_out_of_vocab_contract(question_repo, field, value):
    repo, _, (word_id, sense_id, _) = question_repo
    with pytest.raises(ValueError, match=field):
        await repo.insert(replace(_question(word_id, sense_id), **{field: value}))


async def test_insert_rejects_assessment_without_sense(question_repo):
    repo, _, (word_id, _, _) = question_repo
    with pytest.raises(ValueError, match="sense_id"):
        await repo.insert(_question(word_id, None))


async def test_retrieve_one_is_exact_ordered_and_honors_exclusions(question_repo):
    repo, _, (word_id, sense_id, other_sense_id) = question_repo
    wanted = await repo.insert(_question(word_id, sense_id))
    await repo.insert(
        _question(
            word_id,
            sense_id,
            type_id="contextual_mcq",
            payload={"stem": "Context?", "options": ["a", "b"], "correct_index": 0},
        )
    )
    await repo.insert(
        _question(
            word_id,
            other_sense_id,
            payload={"stem": "Other?", "options": ["a", "b"], "correct_index": 0},
        )
    )

    assert await repo.retrieve_one(sense_id, 1, "definition_mcq", frozenset()) == wanted
    assert (
        await repo.retrieve_one(sense_id, 1, "definition_mcq", frozenset({wanted.question_id}))
        is None
    )
    assert await repo.retrieve_one(sense_id, 2, "definition_mcq", frozenset()) is None


async def test_list_filters_use_type_id_and_list_for_sense_excludes_whole_word(question_repo):
    repo, session_factory, (word_id, sense_id, _) = question_repo
    assessment = await repo.insert(_question(word_id, sense_id))
    async with session_scope(session_factory) as session:
        session.add(
            QuestionRow(
                word_id=word_id,
                sense_id=None,
                type_id="flashcard",
                render_format="flashcard",
                difficulty_level=0,
                interaction_mode="exposure",
                payload='{"definition":"x","example":null,"ipa_uk":null,"ipa_us":null,"pos":null,"word":"x"}',
                content_hash="0" * 64,
            )
        )

    assert await repo.list_for_word(word_id, "definition_mcq") == [assessment]
    assert await repo.list_for_word(word_id, "flashcard")
    assert await repo.list_for_sense(sense_id, "definition_mcq") == [assessment]
    assert await repo.list_for_sense(sense_id, "flashcard") == []
