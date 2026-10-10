"""Work-size regressions: deduplicated evidence, bounded inputs and safe SQLite IDs."""

import asyncio
import importlib

import pytest
from sqlalchemy import event, insert, select, update
from test_db_optimization import optimized_db as optimized_db
from test_db_optimization import sqlite_db as sqlite_db
from test_prompting import bound_content, prompt_context
from test_question_grading import CONFIG

from lexi_ai import schema as row
from lexi_ai.db.bulk import insert_identified_rows, insert_rows
from lexi_ai.errors import InvalidResourceError
from lexi_ai.models import Option, Question
from lexi_ai.questions.generate import generate_questions
from lexi_ai.questions.grade import grade_answer
from lexi_ai.questions.storage import append, list_for_sense
from lexi_ai.references.cambridge import encode_reference_id
from lexi_ai.relations.storage import definition_hash, pending_relations
from lexi_ai.themes.service import list_themes
from lexi_ai.translation import storage as translations
from lexi_ai.words.generate import generate_word
from lexi_ai.words.storage import get_senses, get_word


async def shared_target(db, *, source_count=50, definition="target"):
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
            insert(row.Sense),
            [dict(id=i, word_id=1, pos="NOUN", tier="CORE") for i in range(1, source_count + 1)]
            + [dict(id=1000 + i, word_id=2, pos="NOUN", tier="CORE") for i in range(12)],
        )
        await session.execute(
            insert(row.Definition),
            [dict(sense_id=i, content="source") for i in range(1, source_count + 1)]
            + [dict(sense_id=1000 + i, content=definition) for i in range(12)],
        )
        await session.execute(
            insert(row.SenseRelation),
            [
                dict(id=i, from_sense_id=i, to_word_id=2, rel_type="SYNONYM")
                for i in range(1, source_count + 1)
            ],
        )


async def test_pending_hashes_candidates_once_per_distinct_scope(optimized_db, monkeypatch):
    await shared_target(optimized_db)
    relations = importlib.import_module("lexi_ai.relations.storage")
    hashed = []
    original = relations.definition_hash

    def counted(content):
        hashed.append(content)
        return original(content)

    monkeypatch.setattr(relations, "definition_hash", counted)
    links = await pending_relations(optimized_db, 50)
    assert len(links) == 50
    assert len(hashed) == 12
    assert all(len(link.candidates) == 12 for link in links)


async def test_reader_transfers_shared_target_text_once(optimized_db, monkeypatch):
    await shared_target(optimized_db, definition="x" * 16000)
    async with optimized_db.transaction() as session:
        await session.execute(
            update(row.SenseRelation).values(
                to_sense_id=1000,
                target_hash=definition_hash("x" * 16000),
                resolve_attempted_at="resolved",
            )
        )
    words = importlib.import_module("lexi_ai.words.storage")
    original = words.evidence_fingerprints
    received = []

    def counted(evidence):
        received.append(evidence)
        return original(evidence)

    monkeypatch.setattr(words, "evidence_fingerprints", counted)
    word = await get_word(optimized_db, 1)
    senses = await get_senses(optimized_db, list(range(1, 51)))
    assert all(s.relations[0].resolution_state == "RESOLVED" for s in [*word.senses, *senses])
    assert [len(evidence) for evidence in received] == [1, 1]
    assert all(evidence[0]["content"] == "x" * 16000 for evidence in received)
    # A mismatched imported fingerprint must not be trusted, even without an evidence edit.
    async with optimized_db.transaction() as session:
        await session.execute(
            update(row.SenseRelation).where(row.SenseRelation.id == 1).values(target_hash="stale")
        )
    assert (await get_word(optimized_db, 1)).senses[0].relations[0].resolution_state == "PENDING"


@pytest.mark.parametrize("kind", ["DEFINITION_TO_WORD", "WORD_TO_DEFINITION", "CONTEXT_TO_WORD"])
async def test_long_question_definition_is_sent_once(sqlite_db, kind):
    await shared_target(sqlite_db, source_count=1)
    anchor = "x" * 6000
    async with sqlite_db.transaction() as session:
        await session.execute(
            update(row.Definition).where(row.Definition.sense_id == 1).values(content=anchor)
        )

    class LLM:
        async def complete(self, instruction, data, schema):
            assert len(data) < 16000
            assert data.count(anchor) == 1
            context = prompt_context(data)
            assert not {"meaning_anchor", "fixed_content", "fixed_answer"} & context.keys()
            return schema.model_validate(
                {
                    "questions": [
                        {
                            "content": bound_content(context) or "Where should I keep my money?",
                            "correct_explanation": "Fits.",
                            "distractors": [
                                {"content": f"wrong{i}", "explanation": "Does not fit."}
                                for i in range(3)
                            ],
                        }
                    ]
                }
            )

    question = (await generate_questions(sqlite_db, LLM(), 1, kind, 1, distractor_count=3))[0]
    assert question.correct.content == (anchor if kind == "WORD_TO_DEFINITION" else "word1")


async def test_oversize_short_answer_never_reaches_provider(sqlite_db):
    await shared_target(sqlite_db, source_count=1)
    question = (
        await append(
            sqlite_db,
            [
                Question(
                    0,
                    1,
                    None,
                    "WORD_TO_DEFINITION",
                    "[word1]",
                    Option("yes", "source", "Fits"),
                    [],
                )
            ],
        )
    )[0]

    class Decision:
        async def decide(self, *args, **kwargs):
            pytest.fail("oversize input reached provider")

    with pytest.raises(InvalidResourceError, match="invalid answer"):
        await grade_answer(
            sqlite_db, Decision(), question.id, "SHORT_ANSWER", "x" * 16001, config=CONFIG
        )


async def test_sqlite_bulk_sense_ids_are_safe_across_writers_and_rollback(sqlite_db):
    async with sqlite_db.transaction() as session:
        session.add(row.Word(id=1, lemma="word", match_key="word", generation_state="PENDING"))

    async def write(number):
        async with sqlite_db.transaction(immediate=True) as session:
            ids = await insert_identified_rows(
                session,
                row.Sense,
                [dict(word_id=1, pos="NOUN", tier="CORE", ipa_uk=str(number)) for _ in range(20)],
            )
            return ids

    groups = await asyncio.gather(*(write(i) for i in range(4)))
    assert len({identifier for group in groups for identifier in group}) == 80
    with pytest.raises(RuntimeError):
        async with sqlite_db.transaction(immediate=True) as session:
            await insert_identified_rows(
                session, row.Sense, [dict(word_id=1, pos="NOUN", tier="CORE")]
            )
            raise RuntimeError("rollback")
    async with sqlite_db.read() as connection:
        records = (await connection.execute(select(row.Sense.id, row.Sense.ipa_uk))).all()
    assert len(records) == 80
    for number, group in enumerate(groups):
        assert all(ipa == str(number) for identifier, ipa in records if identifier in group)


async def test_keyset_lists_keep_existing_order_and_namespace(optimized_db):
    await shared_target(optimized_db, source_count=1)
    async with optimized_db.transaction() as session:
        await session.execute(
            insert(row.Theme),
            [
                dict(id=i, key=key, name=key, voice="voice", diction="diction")
                for i, key in [(1, "zebra"), (2, "alpha"), (3, "middle")]
            ],
        )
    await append(
        optimized_db,
        [
            Question(
                0,
                1,
                style,
                "DEFINITION_TO_WORD",
                str(i),
                Option("yes", "word1", "Fits"),
                [],
            )
            for style in (None, 1)
            for i in range(5)
        ],
    )
    for i in range(5):
        await translations.insert(optimized_db, str(i), "vi", str(i))
    for list_page in (
        lambda **kw: list_for_sense(optimized_db, 1, **kw),
        lambda **kw: translations.list_rows(optimized_db, **kw),
    ):
        complete = await list_page()
        first = await list_page(limit=2)
        second = await list_page(limit=2, after_id=first[-1].id)
        rest = await list_page(after_id=second[-1].id, limit=2)
        assert first + second + rest == complete
        assert await list_page(after_id=complete[-1].id, limit=2) == []
        with pytest.raises(InvalidResourceError):
            await list_page(limit=0)
    assert [theme.key for theme in await list_themes(optimized_db, limit=2)] == ["alpha", "middle"]
    assert [
        theme.key for theme in await list_themes(optimized_db, after_key="middle", limit=2)
    ] == ["zebra"]
    assert all(
        q.theme_id == 1 for q in await list_for_sense(optimized_db, 1, theme_key="zebra", limit=2)
    )
    with pytest.raises(InvalidResourceError, match="unknown Theme"):
        await list_for_sense(optimized_db, 1, theme_key="missing", after_id=100000, limit=2)


async def test_pattern_projection_and_warm_matching(sqlite_db, monkeypatch):
    await shared_target(sqlite_db, source_count=1)
    async with sqlite_db.transaction() as session:
        await session.execute(
            insert(row.SensePattern),
            [dict(sense_id=1, content=f"word1 {{sth}} fixed{i}") for i in range(5)],
        )
        await session.execute(
            insert(row.SenseForm),
            [dict(sense_id=1, surface=f"word1x{i}", inf="BASE") for i in range(20)],
        )
    search = importlib.import_module("lexi_ai.words.search")
    original = search.matches_pattern
    patterns = []

    def counted(pattern, *args, **kwargs):
        patterns.append(pattern)
        return original(pattern, *args, **kwargs)

    monkeypatch.setattr(search, "matches_pattern", counted)
    engine = search.Search(sqlite_db, None)
    await engine.start()
    hits = (await engine.search("word1 thing fixed3")).items
    assert hits[0].matched_surface == "word1 {sth} fixed3"
    assert len(patterns) <= 5
    assert "word1 {sth} fixed3" in patterns
    patterns.clear()
    assert (await engine.search("word1 thing fixed3")).items == hits
    assert len(patterns) <= 5  # Warm searches reuse the index, without a result cache.


async def test_append_returns_detached_artifact_without_json_roundtrip(sqlite_db, monkeypatch):
    await shared_target(sqlite_db, source_count=1)
    questions = importlib.import_module("lexi_ai.questions.storage")
    artifact = Question(
        0,
        1,
        None,
        "DIALOGUE_COMPLETION",
        [{"speaker": "Maya", "text": "Hello"}],
        Option("yes", "Reply", "Fits"),
        [],
    )

    def unexpected_decode(_record):
        pytest.fail("append decoded freshly serialized JSON")

    monkeypatch.setattr(questions, "_dto", unexpected_decode)
    saved = (await append(sqlite_db, [artifact]))[0]
    artifact.content[0]["text"] = "changed"
    artifact.distractors.append(Option("no", "wrong", "No"))
    assert saved.content == [{"speaker": "Maya", "text": "Hello"}]
    assert saved.distractors == []


async def test_generation_checks_source_reuse_and_theme_in_one_read(optimized_db):
    await shared_target(optimized_db, source_count=1)
    async with optimized_db.transaction() as session:
        session.add(row.WordSource(word_id=1, source_id=1))
        session.add(row.Theme(key="style", name="Style", voice="V", diction="D"))
    statements = []

    def capture(_connection, _cursor, sql, _parameters, _context, _many):
        statements.append(sql)

    class Source:
        async def fetch_by_id(self, _id):
            pytest.fail("source fetched during reuse or invalid Theme precheck")

    event.listen(optimized_db.engine.sync_engine, "before_cursor_execute", capture)
    try:
        assert (
            await generate_word(
                optimized_db,
                Source(),
                None,
                encode_reference_id(1),
                1,
                target="word0",
                theme_key="style",
            )
            == 1
        )
        assert len(statements) == 1
        with pytest.raises(InvalidResourceError, match="unknown Theme"):
            await generate_word(
                optimized_db,
                Source(),
                None,
                encode_reference_id(2),
                1,
                target="word0",
                theme_key="missing",
            )
        assert len(statements) == 2
    finally:
        event.remove(optimized_db.engine.sync_engine, "before_cursor_execute", capture)


async def test_child_batches_are_true_multirow_and_keep_key_defaults(optimized_db):
    await shared_target(optimized_db, source_count=1)
    statements = []

    def capture(_connection, _cursor, sql, parameters, _context, many):
        statements.append((sql, parameters, many))

    event.listen(optimized_db.engine.sync_engine, "before_cursor_execute", capture)
    try:
        async with optimized_db.transaction() as session:
            await insert_rows(
                session,
                row.SenseForm,
                [dict(sense_id=1, surface=f"ＦＯＲＭ{i}", inf="BASE") for i in range(1001)],
            )
    finally:
        event.remove(optimized_db.engine.sync_engine, "before_cursor_execute", capture)
    assert len(statements) == 3
    assert all(not many for _, _, many in statements)
    assert all(sql.count("), (") == 499 for sql, _, _ in statements[:2])
    async with optimized_db.read() as connection:
        records = (
            await connection.execute(select(row.SenseForm.match_key, row.SenseForm.head_key))
        ).all()
    assert len(records) == 1001
    assert ("form1000", "form1000") in records
