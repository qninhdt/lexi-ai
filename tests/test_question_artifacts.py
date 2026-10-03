import pytest
from sqlalchemy import event
from test_db_optimization import optimized_db as optimized_db
from test_db_optimization import seed
from test_db_optimization import sqlite_db as sqlite_db

from lexi_ai.cache import Cache
from lexi_ai.db.session import Database
from lexi_ai.errors import InvalidResourceError, QuestionBankChangedError
from lexi_ai.models import Option, Question
from lexi_ai.questions.storage import (
    append,
    get,
    get_many,
    list_for_sense,
    remove,
    retrieve,
    retrieve_many,
)
from lexi_ai.schema import Base, Sense, Theme, Word
from lexi_ai.vocab import QuestionType


async def test_batch_retrieval_dense_slots_cache_and_insufficient_banks(optimized_db):
    db = optimized_db
    db.question_cache = Cache(64 * 1024)
    await seed(db)
    artifacts = await append(
        db,
        [
            Question(0, sense_id, theme_id, kind, f"prompt{i}", Option("yes", "word", "Fits"), [])
            for sense_id in (1, 2)
            for theme_id in (None, 1)
            for kind in (QuestionType.DEFINITION_TO_WORD, QuestionType.WORD_TO_DEFINITION)
            for i in range(5)
        ],
    )
    requests = [(2, QuestionType.WORD_TO_DEFINITION, 3), (1, QuestionType.DEFINITION_TO_WORD, 5)]
    statements = []

    def capture(_conn, _cursor, sql, _parameters, _context, _many):
        statements.append(sql)

    event.listen(db.engine.sync_engine, "before_cursor_execute", capture)
    try:
        selected = await retrieve_many(db, requests)
        assert len(selected) == len({q.id for q in selected}) == 8
        assert sum(q.sense_id == 2 for q in selected) == 3
        assert all(q.theme_id is None for q in selected)
        assert len(statements) == 3  # Bank sizes, selected IDs, bulk artifact misses.
        assert "LIMIT" in statements[0] and "count(" not in statements[0].lower()
        assert "payload" not in statements[0] and "payload" not in statements[1]
        await get_many(db, [q.id for q in artifacts])  # Warm all banks without caching selection.
        statements.clear()
        assert len(await retrieve_many(db, requests)) == 8
        assert len(statements) == 2
        selected[0].distractors.append(Option("no", "bad", "No"))
        statements.clear()
        reread = await get(db, selected[0].id)
        assert reread.distractors == []
        assert reread.question_type is selected[0].question_type
        assert statements == []
        assert len(await retrieve_many(db, requests, theme_key="style1")) == 8
        with pytest.raises(QuestionBankChangedError):
            await retrieve_many(db, [(1, QuestionType.DEFINITION_TO_WORD, 6)])
        for count in (0, -1, True, 1.5):
            with pytest.raises(ValueError, match="positive integer"):
                await retrieve_many(db, [(1, QuestionType.DEFINITION_TO_WORD, count)])
        with pytest.raises(ValueError, match="once"):
            await retrieve_many(db, [requests[0], requests[0]])
        with pytest.raises(InvalidResourceError, match="unknown Theme"):
            await retrieve_many(db, [], theme_key="missing")
        assert await retrieve_many(db, []) == []
        assert await remove(db, selected[0].id)
        assert await get(db, selected[0].id) is None
    finally:
        event.remove(db.engine.sync_engine, "before_cursor_execute", capture)


async def test_append_get_list_retrieve_delete_and_exact_namespace(tmp_path):
    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'generated.db'}")
    try:
        await db.create_schema(Base.metadata)
        async with db.transaction() as session:
            word = Word(lemma="bank", match_key="bank", entry_type="WORD", generation_state="DONE")
            theme = Theme(key="pirate", name="Pirate", voice="Captain", diction="nautical")
            session.add_all([word, theme])
            await session.flush()
            sense = Sense(word_id=word.id, pos="NOUN", tier="CORE")
            session.add(sense)
            await session.flush()
        artifact = Question(
            0,
            sense.id,
            None,
            "DEFINITION_TO_WORD",
            "A place for money",
            Option("correct", "bank", "Fits."),
            [Option("wrong", "boat", "Does not fit.")],
        )
        first, second = await append(db, [artifact, artifact])
        assert first.id != second.id
        assert (await get(db, first.id)).correct.id == "correct"
        assert len(await list_for_sense(db, sense.id)) == 2
        assert await list_for_sense(db, sense.id, theme_id=theme.id) == []
        assert (await retrieve(db, sense.id)) is not None
        assert await remove(db, first.id)
        assert not await remove(db, first.id)
    finally:
        await db.close()


async def test_list_for_senses_batch_reading(tmp_path):
    import pytest

    from lexi_ai.errors import InvalidResourceError
    from lexi_ai.questions.storage import list_for_senses

    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'batch_test.db'}")
    try:
        await db.create_schema(Base.metadata)
        async with db.transaction() as session:
            word = Word(lemma="test", match_key="test", entry_type="WORD", generation_state="DONE")
            theme = Theme(key="cyberpunk", name="Cyberpunk", voice="Runner", diction="tech")
            session.add_all([word, theme])
            await session.flush()
            sense1 = Sense(word_id=word.id, pos="NOUN", tier="CORE")
            sense2 = Sense(word_id=word.id, pos="VERB", tier="CORE")
            session.add_all([sense1, sense2])
            await session.flush()

        # Add 3 definition_to_word for sense1 (neutral)
        q1 = Question(0, sense1.id, None, "DEFINITION_TO_WORD", "c1", Option("1", "a", "exp"), [])
        q2 = Question(0, sense1.id, None, "DEFINITION_TO_WORD", "c2", Option("2", "b", "exp"), [])
        q3 = Question(0, sense1.id, None, "DEFINITION_TO_WORD", "c3", Option("3", "c", "exp"), [])
        # Add 1 word_to_definition for sense1 (neutral)
        q4 = Question(0, sense1.id, None, "WORD_TO_DEFINITION", "c4", Option("4", "d", "exp"), [])
        # Add 1 definition_to_word for sense2 (neutral)
        q5 = Question(0, sense2.id, None, "DEFINITION_TO_WORD", "c5", Option("5", "e", "exp"), [])
        # Add 1 definition_to_word for sense1 under theme
        q6 = Question(
            0, sense1.id, theme.id, "DEFINITION_TO_WORD", "c6", Option("6", "f", "exp"), []
        )

        await append(db, [q1, q2, q3, q4, q5, q6])

        # 1. Empty input
        assert await list_for_senses(db, []) == []

        # 2. Multiple senses with limit_per_type=2 (should limit q1,q2,q3 to 2 items)
        results = await list_for_senses(db, [sense1.id, sense2.id], limit_per_type=2)
        # sense1: 2 def_to_word, 1 word_to_def. sense2: 1 def_to_word. Total: 4
        assert len(results) == 4
        s1_dtw = [
            q
            for q in results
            if q.sense_id == sense1.id and q.question_type == "DEFINITION_TO_WORD"
        ]
        assert len(s1_dtw) == 2
        s1_wtd = [
            q
            for q in results
            if q.sense_id == sense1.id and q.question_type == "WORD_TO_DEFINITION"
        ]
        assert len(s1_wtd) == 1
        s2_dtw = [
            q
            for q in results
            if q.sense_id == sense2.id and q.question_type == "DEFINITION_TO_WORD"
        ]
        assert len(s2_dtw) == 1

        # 3. Filter by question_types
        filtered = await list_for_senses(
            db, [sense1.id, sense2.id], question_types=["WORD_TO_DEFINITION"]
        )
        assert len(filtered) == 1
        assert filtered[0].question_type == "WORD_TO_DEFINITION"

        # 4. Unknown sense id
        assert await list_for_senses(db, [99999]) == []

        # 5. Invalid limit_per_type
        with pytest.raises(ValueError, match="limit_per_type must be a positive integer"):
            await list_for_senses(db, [sense1.id], limit_per_type=0)

        # 6. Neutral namespace isolation (theme questions not included)
        assert all(q.theme_id is None for q in results)

        # 7. Valid theme_key
        theme_results = await list_for_senses(db, [sense1.id], theme_key="cyberpunk")
        assert len(theme_results) == 1
        assert theme_results[0].theme_id == theme.id

        # 8. Invalid theme_key
        with pytest.raises(InvalidResourceError, match="unknown Theme"):
            await list_for_senses(db, [sense1.id], theme_key="non_existent_theme")
        with pytest.raises(InvalidResourceError, match="unknown Theme"):
            await list_for_senses(db, [], theme_key="non_existent_theme")
    finally:
        await db.close()
