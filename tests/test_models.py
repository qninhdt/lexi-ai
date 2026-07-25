"""Tests for the ORM models and schema plumbing (Phase 2).

Uses in-memory SQLite. A single shared connection is kept for the lifetime of
each test (via ``poolclass=StaticPool``) so the schema persists across sessions.
"""

import pytest
from sqlalchemy import event, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.orm import selectinload
from sqlalchemy.pool import StaticPool

from lexi_ai.db import (
    create_session_factory,
    init_models,
    session_scope,
)
from lexi_ai.infrastructure.db.models import (
    Example,
    Sense,
    SenseReference,
    SenseRelation,
    Word,
    WordAlias,
    WordRelation,
)
from lexi_ai.infrastructure.db.models import (
    Question as QuestionRow,
)


@pytest.fixture
async def session_factory():
    engine = create_async_engine(
        "sqlite+aiosqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    @event.listens_for(engine.sync_engine, "connect")
    def _fk_on(dbapi_conn, _record):
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA foreign_keys=ON")
        cur.close()

    await init_models(engine)
    yield create_session_factory(engine)
    await engine.dispose()


async def _make_word(session, match_key="color", norm="color", **kwargs):
    """Add a Word (with any child collections passed via kwargs) and flush.

    Children must be supplied at construction time so the graph is built while
    the object is transient — appending to a relationship after flush would
    trigger an async lazy-load outside greenlet context. This mirrors how the
    Phase 5 repository assembles entries.
    """
    word = Word(norm=norm, match_key=match_key, entry_type="word", status="done", **kwargs)
    session.add(word)
    await session.flush()
    return word


async def test_init_models_creates_schema(session_factory):
    async with session_scope(session_factory) as session:
        result = await session.execute(select(Word))
        assert result.all() == []


async def test_full_graph_insert_and_traversal(session_factory):
    async with session_scope(session_factory) as session:
        sense = Sense(
            definition="the property of reflecting light",
            tier="core",
            sense_order=0,
            pos="noun",
            cefr_level="A1",
            references=[SenseReference(source="cambridge", source_ref="123")],
            examples=[Example(text="a bright color", example_order=0)],
        )
        await _make_word(
            session,
            aliases=[
                WordAlias(
                    alias_norm="colour",
                    alias_match_key="colour",
                    type="spelling_uk",
                    dialect="uk",
                )
            ],
            senses=[sense],
        )

    # Fresh session: assert the graph loads back with relationships.
    async with session_scope(session_factory) as session:
        loaded = (
            await session.execute(
                select(Word)
                .options(
                    selectinload(Word.senses).selectinload(Sense.references),
                    selectinload(Word.senses).selectinload(Sense.examples),
                    selectinload(Word.aliases),
                )
                .where(Word.match_key == "color")
            )
        ).scalar_one()

        assert loaded.norm == "color"
        assert len(loaded.aliases) == 1
        assert loaded.aliases[0].alias_match_key == "colour"
        assert len(loaded.senses) == 1
        assert loaded.senses[0].tier == "core"
        assert len(loaded.senses[0].references) == 1
        assert loaded.senses[0].references[0].source == "cambridge"
        assert len(loaded.senses[0].examples) == 1


async def test_unique_word_match_key(session_factory):
    with pytest.raises(IntegrityError):
        async with session_scope(session_factory) as session:
            await _make_word(session, match_key="dup", norm="dup")
            await _make_word(session, match_key="dup", norm="dup2")


async def test_unique_alias_word_key(session_factory):
    with pytest.raises(IntegrityError):
        async with session_scope(session_factory) as session:
            await _make_word(
                session,
                aliases=[
                    WordAlias(
                        alias_norm="colour",
                        alias_match_key="colour",
                        type="spelling_uk",
                    ),
                    WordAlias(
                        alias_norm="colour",
                        alias_match_key="colour",
                        type="spelling_other",
                    ),
                ],
            )


async def test_entry_link_unique_triple(session_factory):
    with pytest.raises(IntegrityError):
        async with session_scope(session_factory) as session:
            a = await _make_word(session, match_key="a", norm="a")
            b = await _make_word(session, match_key="b", norm="b")
            session.add_all(
                [
                    WordRelation(from_word_id=a.id, to_word_id=b.id, rel_type="synonym"),
                    WordRelation(from_word_id=a.id, to_word_id=b.id, rel_type="synonym"),
                ]
            )
            await session.flush()


async def test_cascade_delete_word_removes_children(session_factory):
    async with session_scope(session_factory) as session:
        await _make_word(
            session,
            senses=[Sense(definition="d", tier="core")],
            aliases=[WordAlias(alias_norm="c", alias_match_key="c", type="spelling_us")],
        )

    async with session_scope(session_factory) as session:
        word = (await session.execute(select(Word).where(Word.match_key == "color"))).scalar_one()
        await session.delete(word)

    async with session_scope(session_factory) as session:
        assert (await session.execute(select(Sense))).all() == []
        assert (await session.execute(select(WordAlias))).all() == []


# --- sense_relation (Phase 2): edge + derived-state work queue ------------


async def _make_sense(session, word, definition="d", tier="core", pos="noun"):
    sense = Sense(word_id=word.id, definition=definition, tier=tier, pos=pos)
    session.add(sense)
    await session.flush()
    return sense


async def test_sense_relation_insert_pending(session_factory):
    # A half-edge: to_sense_id NULL + resolve_attempted_at NULL == derived pending.
    async with session_scope(session_factory) as session:
        a = await _make_word(session, match_key="a", norm="a")
        b = await _make_word(session, match_key="b", norm="b")
        src = await _make_sense(session, a, definition="lacking light", pos="adjective")
        rel = SenseRelation(
            from_sense_id=src.id,
            to_word_id=b.id,
            to_sense_id=None,
            rel_type="antonym",
            gloss="full of light",
        )
        session.add(rel)
        await session.flush()

    async with session_scope(session_factory) as session:
        row = (await session.execute(select(SenseRelation))).scalar_one()
        assert row.to_sense_id is None
        assert row.resolve_attempted_at is None  # derived: pending


async def test_sense_relation_unique_triple(session_factory):
    with pytest.raises(IntegrityError):
        async with session_scope(session_factory) as session:
            a = await _make_word(session, match_key="a", norm="a")
            b = await _make_word(session, match_key="b", norm="b")
            src = await _make_sense(session, a)
            session.add_all(
                [
                    SenseRelation(
                        from_sense_id=src.id, to_word_id=b.id, rel_type="synonym", gloss="g1"
                    ),
                    SenseRelation(
                        from_sense_id=src.id, to_word_id=b.id, rel_type="synonym", gloss="g2"
                    ),
                ]
            )
            await session.flush()


async def test_target_sense_delete_sets_null_keeps_edge(session_factory):
    # Deleting the TARGET sense must SET NULL to_sense_id (not delete the edge):
    # the sense->word relation survives.
    async with session_scope(session_factory) as session:
        a = await _make_word(session, match_key="a", norm="a")
        b = await _make_word(session, match_key="b", norm="b")
        src = await _make_sense(session, a, pos="adjective")
        tgt = await _make_sense(session, b, pos="adjective")
        session.add(
            SenseRelation(
                from_sense_id=src.id,
                to_word_id=b.id,
                to_sense_id=tgt.id,
                rel_type="antonym",
                gloss="g",
                target_hash="deadbeef",
            )
        )
        await session.flush()
        tgt_id = tgt.id

    async with session_scope(session_factory) as session:
        await session.execute(select(Sense).where(Sense.id == tgt_id))  # ensure loaded path
        tgt = (await session.execute(select(Sense).where(Sense.id == tgt_id))).scalar_one()
        await session.delete(tgt)

    async with session_scope(session_factory) as session:
        row = (await session.execute(select(SenseRelation))).scalar_one()
        assert row.to_sense_id is None  # SET NULL — edge survives at sense->word


async def test_target_sense_delete_auto_demotes_via_fk(session_factory):
    # [VALIDATE Q1] resolved edge, then target sense deleted: FK SET NULL alone
    # drops to_sense_id -> derived state is NO LONGER 'resolved'. (Full re-queue to
    # 'pending' needs the Phase 5 helper to also clear resolve_attempted_at.)
    async with session_scope(session_factory) as session:
        a = await _make_word(session, match_key="a", norm="a")
        b = await _make_word(session, match_key="b", norm="b")
        src = await _make_sense(session, a)
        tgt = await _make_sense(session, b)
        session.add(
            SenseRelation(
                from_sense_id=src.id,
                to_word_id=b.id,
                to_sense_id=tgt.id,
                rel_type="synonym",
                gloss="g",
            )
        )
        await session.flush()
        tgt_id = tgt.id

    async with session_scope(session_factory) as session:
        tgt = (await session.execute(select(Sense).where(Sense.id == tgt_id))).scalar_one()
        await session.delete(tgt)

    async with session_scope(session_factory) as session:
        row = (await session.execute(select(SenseRelation))).scalar_one()
        assert row.to_sense_id is None  # not resolved anymore, no code demote needed


async def test_source_sense_delete_cascades_edge(session_factory):
    # Deleting the SOURCE sense must CASCADE-delete the edge (Case 6).
    async with session_scope(session_factory) as session:
        a = await _make_word(session, match_key="a", norm="a")
        b = await _make_word(session, match_key="b", norm="b")
        src = await _make_sense(session, a)
        session.add(
            SenseRelation(from_sense_id=src.id, to_word_id=b.id, rel_type="synonym", gloss="g")
        )
        await session.flush()
        src_id = src.id

    async with session_scope(session_factory) as session:
        src = (await session.execute(select(Sense).where(Sense.id == src_id))).scalar_one()
        await session.delete(src)

    async with session_scope(session_factory) as session:
        assert (await session.execute(select(SenseRelation))).all() == []


# --- dual-dialect portability (no live Postgres needed) -------------------


def test_schema_compiles_on_both_dialects():
    """Every table's DDL must compile for SQLite AND Postgres (criterion #7).

    Proves no dialect-specific type blocks the Postgres target without needing a
    running Postgres — a mismatched type raises at compile time.
    """
    from sqlalchemy.dialects import postgresql, sqlite
    from sqlalchemy.schema import CreateTable

    from lexi_ai.infrastructure.db.models import Base

    # The tag + collocation + sense_forms + questions tables ride the same
    # portability guarantee.
    table_names = set(Base.metadata.tables)
    assert {"tags", "word_tags", "collocations", "sense_forms", "questions"} <= table_names

    for dialect in (postgresql.dialect(), sqlite.dialect()):
        for table in Base.metadata.sorted_tables:
            ddl = str(CreateTable(table).compile(dialect=dialect))
            assert "CREATE TABLE" in ddl



# --- question type contracts ---------------------------------------------


def test_public_presented_question_uses_type_and_render_contracts():
    from lexi_ai.contracts.questions import PresentedQuestion, TextSpan

    question = PresentedQuestion(
        question_id="7",
        type_id="cloze",
        interaction="assessment",
        difficulty_level=2,
        render=TextSpan(stem_with_blank="A ____.", word_bank=("word", "other")),
        sense_id="2",
        word_id="1",
    )

    assert question.question_id == "7"
    assert question.type_id == "cloze"
    assert isinstance(question.render, TextSpan)
    # No answer key and no storage-shaped leak on the public presentation.
    assert not hasattr(question, "payload")
    assert not hasattr(question, "render_format")
    assert not hasattr(question, "answer_kind")
    assert not hasattr(question.render, "answer_norm")


def test_evaluation_distinguishes_graded_from_pending():
    from lexi_ai.contracts.questions import ChoiceReveal, Evaluation

    graded = Evaluation(
        question_id="7",
        status="graded",
        correct=True,
        score=1.0,
        reveal=ChoiceReveal(correct_index=0, correct_option="eloquent"),
    )
    pending = Evaluation(question_id="7", status="pending")

    assert graded.status == "graded" and graded.correct is True
    assert isinstance(graded.reveal, ChoiceReveal)
    # A pending outcome carries no verdict and no reveal (nothing graded yet).
    assert pending.status == "pending"
    assert pending.correct is None and pending.reveal is None


def test_question_orm_has_new_columns_and_idempotency_constraint():
    columns = set(QuestionRow.__table__.columns.keys())
    constraints = {
        constraint.name: tuple(column.name for column in constraint.columns)
        for constraint in QuestionRow.__table__.constraints
        if constraint.name is not None
    }

    assert {
        "id",
        "word_id",
        "sense_id",
        "type_id",
        "render_format",
        "difficulty_level",
        "interaction_mode",
        "payload",
        "content_hash",
        "created_at",
    } == columns
    assert "format" not in columns
    assert "answer_kind" not in columns
    assert constraints["uq_question_content"] == (
        "sense_id",
        "type_id",
        "difficulty_level",
        "content_hash",
    )
