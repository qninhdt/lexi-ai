"""Query budgets and dense banks, exercised on SQLite and disposable PostgreSQL."""

import asyncio
import json
from collections import Counter, defaultdict

import pytest
from sqlalchemy import delete, event, insert, select, text, update
from sqlalchemy.exc import IntegrityError

from lexi_ai import schema as row
from lexi_ai.cache import Cache
from lexi_ai.db.session import Database
from lexi_ai.errors import InvalidResourceError
from lexi_ai.models import Option, Question
from lexi_ai.questions.storage import append, generation_context, get_many, retrieve
from lexi_ai.references.cambridge import SourceEntry, SourceSense, encode_reference_id
from lexi_ai.relations.storage import apply_resolution, pending_relations
from lexi_ai.words.generate import generate_word
from lexi_ai.words.storage import get_senses, get_word


@pytest.fixture(params=["sqlite", "postgresql"])
def optimized_db(request):
    return request.getfixturevalue("pg_db" if request.param == "postgresql" else "sqlite_db")


@pytest.fixture
async def sqlite_db(tmp_path):
    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'optimized.db'}")
    try:
        await db.create_schema(row.Base.metadata)
        yield db
    finally:
        await db.close()


async def seed(db):
    async with db.transaction() as session:
        await session.execute(
            insert(row.Word),
            [
                dict(
                    id=i,
                    lemma=f"word{i}",
                    match_key=f"word{i}",
                    entry_type="WORD",
                    generation_state="DONE",
                )
                for i in (1, 2)
            ],
        )
        await session.execute(
            insert(row.Theme),
            [
                dict(id=i, key=f"style{i}", name=f"Style {i}", voice="Voice", diction="Diction")
                for i in (1, 2)
            ],
        )
        await session.execute(
            insert(row.Sense), [dict(id=i, word_id=i, pos="NOUN", tier="CORE") for i in (1, 2)]
        )
        await session.execute(
            insert(row.Definition), [dict(sense_id=i, content=f"meaning{i}") for i in (1, 2)]
        )


async def assert_dense(db):
    async with db.read() as connection:
        records = (await connection.execute(select(row.Question).order_by(row.Question.id))).all()
    scope, types = defaultdict(list), defaultdict(list)
    for record in records:
        scope[record.sense_id, record.theme_id].append(record.position)
        types[record.sense_id, record.theme_id, record.question_type].append(record.type_position)
    for bank in (*scope.values(), *types.values()):
        assert sorted(bank) == list(range(1, len(bank) + 1))


async def test_dense_slots_raw_sql_moves_deletes_and_cascades(optimized_db):
    db = optimized_db
    await seed(db)
    async with db.transaction() as session:
        await session.execute(
            insert(row.Question),
            [
                dict(
                    id=i,
                    sense_id=1,
                    theme_id=None if i < 7 else 1,
                    question_type="DEFINITION_TO_WORD" if i % 2 else "WORD_TO_DEFINITION",
                    payload="{}",
                    position=999,
                    type_position=999,
                )
                for i in range(1, 13)
            ],
        )
    await assert_dense(db)
    async with db.transaction() as session:
        await session.execute(delete(row.Question).where(row.Question.id == 3))
        await session.execute(
            update(row.Question)
            .where(row.Question.id == 2)
            .values(question_type="DEFINITION_TO_WORD")
        )
        await session.execute(update(row.Question).where(row.Question.id == 5).values(theme_id=1))
        await session.execute(
            update(row.Question).where(row.Question.id == 8).values(theme_id=None)
        )
        await session.execute(update(row.Question).where(row.Question.id == 10).values(sense_id=2))
    await assert_dense(db)
    async with db.transaction() as session:
        await session.execute(delete(row.Question).where(row.Question.id.in_([1, 4, 7, 12])))
    await assert_dense(db)
    async with db.transaction() as session:
        await session.execute(delete(row.Theme).where(row.Theme.id == 1))
        await session.execute(delete(row.Sense).where(row.Sense.id == 2))
    await assert_dense(db)


async def test_single_definition_uniqueness_and_exact_style(optimized_db):
    db = optimized_db
    await seed(db)
    async with db.transaction() as session:
        session.add_all(
            [row.Definition(sense_id=1, theme_id=i, content=f"style{i}") for i in (1, 2)]
        )
        session.add_all(
            [row.Example(sense_id=1, theme_id=i, content=f"[word1] {i}") for i in (1, 2)]
        )
    for style in (None, 1, 2):
        with pytest.raises(IntegrityError):
            async with db.transaction() as session:
                session.add(row.Definition(sense_id=1, theme_id=style, content="duplicate"))
    assert (await get_word(db, 1)).senses[0].definition.content == "meaning1"
    for style in (1, 2):
        word = await get_word(db, 1, theme_key=f"style{style}")
        assert word.senses[0].definition.content == f"style{style}"
        assert word.senses[0].definition.theme_id == style


async def test_warm_random_retrieval_one_indexed_statement_and_namespace(optimized_db):
    db = optimized_db
    await seed(db)
    artifacts = await append(
        db,
        [
            Question(0, 1, style, kind, str(i), Option("yes", "word1", "Fits"), [])
            for style in (None, 1)
            for kind in ("DEFINITION_TO_WORD", "WORD_TO_DEFINITION")
            for i in range(4)
        ],
    )
    db.question_cache = Cache(64 * 1024)
    await get_many(db, [q.id for q in artifacts])
    statements = []

    def capture(_conn, _cursor, statement, parameters, _context, _many):
        statements.append((statement, parameters))

    event.listen(db.engine.sync_engine, "before_cursor_execute", capture)
    try:
        seen = Counter()
        for _ in range(160):
            question = await retrieve(db, 1, "DEFINITION_TO_WORD", 1)
            assert question.theme_id == 1 and question.question_type == "DEFINITION_TO_WORD"
            seen[question.id] += 1
        assert len(seen) == 4
        assert len(statements) == 160
        assert all(
            "count(" not in sql.lower() and "order by random" not in sql.lower()
            for sql, _ in statements
        )
        statement, parameters = statements[0]
        event.remove(db.engine.sync_engine, "before_cursor_execute", capture)
        async with db.transaction() as session:
            if db.engine.dialect.name == "postgresql":
                await session.execute(text("SET LOCAL enable_seqscan=off"))
                connection = await session.connection()
                plan = await connection.exec_driver_sql("EXPLAIN " + statement, parameters)
            else:
                connection = await session.connection()
                plan = await connection.exec_driver_sql(
                    "EXPLAIN QUERY PLAN " + statement, parameters
                )
            assert "ix_questions_type_scope" in str(plan.all())
        question = await retrieve(db, 1)
        assert question.id in {artifact.id for artifact in artifacts if artifact.theme_id is None}
        assert await retrieve(db, 2) is None
    finally:
        if event.contains(db.engine.sync_engine, "before_cursor_execute", capture):
            event.remove(db.engine.sync_engine, "before_cursor_execute", capture)


async def test_read_and_sense_linking_query_budgets(optimized_db):
    db = optimized_db
    await seed(db)
    async with db.transaction() as session:
        session.add(row.SenseRelation(from_sense_id=1, to_word_id=2, rel_type="SYNONYM"))
    statements = []

    def capture(_conn, _cursor, statement, _parameters, _context, _many):
        if statement.lstrip().startswith(("SELECT", "WITH")):
            statements.append(statement)

    event.listen(db.engine.sync_engine, "before_cursor_execute", capture)
    try:
        for operation in (
            lambda: get_word(db, 1),
            lambda: get_senses(db, [1, 2]),
            lambda: generation_context(db, 1),
        ):
            statements.clear()
            assert await operation()
            assert len(statements) == 1
        statements.clear()
        link = (await pending_relations(db, 20))[0]
        assert len(statements) == 1
        statements.clear()
        assert await apply_resolution(db, link.edge_id, link.candidates[0], expected=link)
        assert len(statements) == 2  # Lock edge, then re-read all current evidence.
    finally:
        event.remove(db.engine.sync_engine, "before_cursor_execute", capture)


async def test_large_bank_deserializes_only_one_question_and_validates_theme(
    optimized_db, monkeypatch
):
    from lexi_ai.questions import storage as questions

    db = optimized_db
    await seed(db)
    payload = json.dumps(
        dict(
            content="Prompt",
            correct=dict(id="yes", content="word1", explanation="Fits"),
            distractors=[],
            correct_alternatives=[],
            target_placement=None,
        )
    )
    async with db.transaction() as session:
        await session.execute(
            insert(row.Question),
            [
                dict(sense_id=1, theme_id=1, question_type="DEFINITION_TO_WORD", payload=payload)
                for _ in range(1000)
            ],
        )
    original = questions._dto
    deserialized = []

    def counted(record):
        deserialized.append(record.id)
        return original(record)

    monkeypatch.setattr(questions, "_dto", counted)
    selected = await retrieve(db, 1, theme_key="style1")
    assert selected.theme_id == 1
    assert deserialized == [selected.id]
    assert await retrieve(db, 1, theme_key="style2") is None
    with pytest.raises(InvalidResourceError, match="unknown Theme"):
        await retrieve(db, 1, theme_key="missing")
    with pytest.raises(InvalidResourceError, match="unknown Theme"):
        await questions.list_for_sense(db, 1, theme_key="missing")


async def test_postgres_concurrent_question_inserts(pg_db):
    await seed(pg_db)

    async def write(number):
        async with pg_db.transaction() as session:
            await session.execute(
                insert(row.Question),
                [
                    dict(sense_id=1, question_type="DEFINITION_TO_WORD", payload=str(number))
                    for _ in range(10)
                ],
            )

    await asyncio.wait_for(asyncio.gather(*(write(i) for i in range(6))), timeout=20)
    await assert_dense(pg_db)
    async with pg_db.read() as connection:
        assert len((await connection.execute(select(row.Question.id))).all()) == 60


@pytest.mark.parametrize("size", [1, 20])
async def test_publication_is_batched_not_per_sense(optimized_db, monkeypatch, size):
    from test_word_generation import payload, stage_payload

    db = optimized_db

    entry = SourceEntry(
        1,
        "bank",
        "bank",
        "word",
        [SourceSense(101 + i, "NOUN", f"meaning{i}") for i in range(size)],
    )
    output = payload()
    output["senses"] = []
    for i in range(size):
        sense = payload(source_ref=f"a{i + 1}")["senses"][0]
        sense["definition"] = f"meaning{i}"
        sense["forms"] = ["banks" + "|pl"]
        sense["patterns"] = ["bank {sth}"]
        sense["collocations"] = ["central bank"]
        sense["relations"] = {"SYNONYM": ["vault"]}
        output["senses"].append(sense)

    class Source:
        async def fetch_by_id(self, _identifier):
            return entry

    class LLM:
        async def complete(self, _instruction, _data, schema):
            return stage_payload(output, _data, schema)

    async def no_wordnet(_lemma):
        return []

    monkeypatch.setattr("lexi_ai.words.generate.lookup", no_wordnet)
    statements = []

    def capture(_conn, _cursor, statement, _parameters, _context, _many):
        statements.append(statement)

    event.listen(db.engine.sync_engine, "before_cursor_execute", capture)
    try:
        word_id = await generate_word(db, Source(), LLM(), encode_reference_id(1), 1, target="bank")
        assert len(statements) <= 18
        for table in (
            "senses",
            "definitions",
            "examples",
            "sense_forms",
            "sense_patterns",
            "collocations",
            "sense_references",
            "sense_relations",
        ):
            assert sum(sql.startswith(f"INSERT INTO {table} ") for sql in statements) == 1
        assert sum(sql.startswith("SELECT words.match_key, words.id") for sql in statements) == 1
        statements.clear()
        word = await get_word(db, word_id)
        assert len(statements) == 1
        assert [sense.definition.content for sense in word.senses] == [
            f"meaning{i}" for i in range(size)
        ]
        assert len({sense.relations[0].to_word_id for sense in word.senses}) == 1
    finally:
        event.remove(db.engine.sync_engine, "before_cursor_execute", capture)
