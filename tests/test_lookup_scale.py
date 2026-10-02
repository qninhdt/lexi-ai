"""Complete search ranking, bounded pattern matching and scoped projections."""

import importlib
import random
import subprocess
import sys

import pytest
from sqlalchemy import event, insert, text
from test_question_grading import CONFIG, Decision

from lexi_ai import schema as row
from lexi_ai.db.session import Database
from lexi_ai.models import Option, Question, SearchResult, WordHit
from lexi_ai.patterns import matches_pattern
from lexi_ai.questions.grade import grade_answer
from lexi_ai.questions.storage import append
from lexi_ai.relations.storage import definition_hash
from lexi_ai.text import answer_key, match_key
from lexi_ai.words.search import search
from lexi_ai.words.storage import get_senses, get_word, meaning_inventory


def test_ten_clause_slots_finish_without_regex_backtracking():
    script = """
from lexi_ai.patterns import matches_pattern
pattern = 'say ' + '{clause} ' * 10 + 'end'
assert matches_pattern(pattern, 'say ' + 'a b ' * 10 + 'end')
assert not matches_pattern(pattern, 'say ' + 'a b ' * 50 + 'wrong')
"""
    subprocess.run([sys.executable, "-c", script], check=True, timeout=2, capture_output=True)


def brute_search(words, aliases, forms, patterns, query):
    key = answer_key(query)
    ranked = []
    for word in words:
        if word.generation_state != "done":
            continue
        candidates = []
        if word.match_key == key:
            candidates.append((0, 1.0, "lemma", word.lemma))
        candidates += [
            (1, 1.0, "alias", item.content)
            for item in aliases
            if item.word_id == word.id and item.match_key == key
        ]
        candidates += [
            (2, 1.0, "form", item.surface)
            for owner, item in forms
            if owner == word.id and answer_key(item.surface) == key
        ]
        head, _, tail = answer_key(word.lemma).partition(" ")
        for owner, pattern in patterns:
            licensed = [
                answer_key(item.surface).partition(" ")[0]
                for form_owner, item in forms
                if form_owner == word.id
                and item.sense_id == pattern.sense_id
                and answer_key(item.surface).partition(" ")[2] == tail
            ]
            if owner == word.id and matches_pattern(pattern.content, query, forms={head: licensed}):
                candidates.append((3, 1.0, "pattern", pattern.content))
        if word.match_key.startswith(key) and word.match_key != key:
            candidates.append((4, 0.0, "prefix", word.lemma))
        candidates += [
            (4, 0.0, "prefix", item.content)
            for item in aliases
            if item.word_id == word.id and item.match_key.startswith(key) and item.match_key != key
        ]
        candidates += [
            (4, 0.0, "prefix", item.surface)
            for owner, item in forms
            if owner == word.id
            and answer_key(item.surface).startswith(key)
            and answer_key(item.surface) != key
        ]
        if candidates:
            rank, score, kind, surface = min(
                candidates,
                key=lambda item: (item[0], -item[1], item[3], item[2]),
            )
            ranked.append(
                (
                    rank,
                    -score,
                    word.lemma,
                    word.id,
                    WordHit(word.id, word.lemma, word.entry_type, kind, surface),
                )
            )
    return [item[-1] for item in sorted(ranked)[:30]]


async def test_indexed_search_preserves_lexical_ranking_on_reference_corpus(tmp_path):
    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'parity.db'}")
    try:
        await db.create_schema(row.Base.metadata)
        names = ["take off", "lift off", "ı", "alpha", "ßeta", "Élan", "a_b", "a%b"]
        names += [f"stone {number:02d}" for number in range(55)]
        names += ["ston", "stane", "stone", "shone", "alone", "store", "tone", "pending"]
        random.Random(9).shuffle(names)
        async with db.transaction() as session:
            words = [
                row.Word(
                    lemma=name,
                    match_key=match_key(name),
                    entry_type="word",
                    generation_state="pending" if name == "pending" else "done",
                )
                for name in names
            ]
            session.add_all(words)
            await session.flush()
            by_name = {word.lemma: word for word in words}
            senses = {
                word.id: row.Sense(word_id=word.id, pos="verb", tier="core") for word in words
            }
            session.add_all(senses.values())
            await session.flush()
            aliases = [
                row.WordAlias(word_id=by_name[name].id, content="stone", match_key="stone")
                for name in names
                if name.startswith("stone ")
            ]
            forms = [
                (
                    by_name[name].id,
                    row.SenseForm(
                        sense_id=senses[by_name[name].id].id, surface=surface, inf="past"
                    ),
                )
                for name, surface in [
                    ("take off", "took off"),
                    ("lift off", "lifted off"),
                    ("alpha", "i"),
                    ("stone 01", "stone"),
                ]
            ]
            specifications = [
                ("take off", "take {sth} off"),
                ("lift off", "lift {sth} off"),
                ("alpha", "alpha {sth}"),
                ("ı", "ı {sth}"),
                ("Élan", "{sb} says hello"),
                ("ßeta", "pre{sth}fix"),
            ]
            specifications += [(name, "stone {sth}") for name in names if name.startswith("stone ")]
            patterns = [
                (
                    by_name[name].id,
                    row.SensePattern(
                        sense_id=senses[by_name[name].id].id,
                        content=pattern,
                    ),
                )
                for name, pattern in specifications
            ]
            session.add_all(aliases + [item for _, item in forms] + [item for _, item in patterns])
        for query in [
            "stone",
            "ston",
            "stane",
            "stone X",
            "TOOK IT OFF",
            "lifted it off",
            "i x",
            "ı x",
            "John says hello",
            "premidfix",
            "a_",
            "a%",
            "él",
            "pending",
        ]:
            assert (await search(db, None, query)).words == brute_search(
                words,
                aliases,
                forms,
                patterns,
                query,
            ), query
    finally:
        await db.close()


async def test_100000_word_search_has_no_id_window_and_narrows_patterns(tmp_path, monkeypatch):
    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'large.db'}")
    statements, matched_patterns = [], []

    def count(_conn, _cursor, statement, parameters, _context, _executemany):
        statements.append((statement, parameters))

    module = importlib.import_module("lexi_ai.words.search")
    original = module.matches_pattern

    def match(pattern, *args, **kwargs):
        matched_patterns.append(pattern)
        return original(pattern, *args, **kwargs)

    monkeypatch.setattr(module, "matches_pattern", match)
    try:
        await db.create_schema(row.Base.metadata)
        async with db.transaction() as session:
            for first in range(1, 100001, 5000):
                ids = range(first, first + 5000)
                await session.execute(
                    insert(row.Word),
                    [
                        {
                            "id": i,
                            "lemma": f"unrelated{i:06d}",
                            "match_key": f"unrelated{i:06d}",
                            "entry_type": "word",
                            "generation_state": "done",
                        }
                        for i in ids
                    ],
                )
                await session.execute(
                    insert(row.Sense),
                    [{"id": i, "word_id": i, "pos": "noun", "tier": "core"} for i in ids],
                )
                await session.execute(
                    insert(row.SensePattern),
                    [{"sense_id": i, "content": f"item{i:06d} {{sth}}"} for i in ids],
                )
            await session.execute(
                insert(row.Word),
                [
                    {
                        "id": i,
                        "lemma": name,
                        "match_key": name,
                        "entry_type": "word",
                        "generation_state": "done",
                    }
                    for i, name in [
                        (100001, "needle"),
                        (100002, "needlework"),
                        (100003, "take off"),
                    ]
                ],
            )
            await session.execute(
                insert(row.Sense),
                [{"id": 100003, "word_id": 100003, "pos": "verb", "tier": "core"}],
            )
            session.add(row.SenseForm(sense_id=100003, surface="took off", inf="past"))
            session.add(row.SensePattern(sense_id=100003, content="take {sth} off"))
        event.listen(db.engine.sync_engine, "before_cursor_execute", count)
        hits = (await search(db, None, "needle")).words
        assert [(hit.word_id, hit.match_kind) for hit in hits[:2]] == [
            (100001, "lemma"),
            (100002, "prefix"),
        ]
        assert len(statements) == 2
        prefix_statement, parameters = statements[0]
        async with db.engine.connect() as connection:
            plan = (
                await connection.exec_driver_sql(
                    "EXPLAIN QUERY PLAN " + prefix_statement,
                    parameters,
                )
            ).all()
            assert "ix_words_match_key_prefix" in str(plan)
            pattern_plan = (
                await connection.execute(
                    text(
                        "EXPLAIN QUERY PLAN SELECT id FROM sense_patterns WHERE head_key='take' "
                        "OR head_key IS NULL OR sense_id IN "
                        "(SELECT sense_id FROM sense_forms WHERE head_key='took')"
                    )
                )
            ).all()
            assert "ix_sense_patterns_head_key" in str(pattern_plan)
            assert "ix_sense_forms_head_key" in str(pattern_plan)
        # SQLite is explicitly lexical-only: no fuzzy index or application scan.
        assert (await search(db, None, "neadle")).words == []
        statements.clear()
        matched_patterns.clear()
        hit = (await search(db, None, "took it off")).words[0]
        assert (hit.word_id, hit.match_kind) == (100003, "pattern")
        assert matched_patterns == ["take {sth} off"]
        assert len(statements) == 2  # One lexical query and one Sense-scoped pattern/form page.
        assert not any("definitions" in statement for statement, _ in statements)
    finally:
        if event.contains(db.engine.sync_engine, "before_cursor_execute", count):
            event.remove(db.engine.sync_engine, "before_cursor_execute", count)
        await db.close()


async def test_grading_projects_neutral_meanings_in_two_queries(tmp_path, monkeypatch):
    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'grading.db'}")
    statements = []

    def count(_conn, _cursor, statement, _parameters, _context, _executemany):
        statements.append(statement)

    try:
        await db.create_schema(row.Base.metadata)
        async with db.transaction() as session:
            await session.execute(
                insert(row.Word),
                [
                    {
                        "id": i,
                        "lemma": f"word{i}",
                        "match_key": f"word{i}",
                        "entry_type": "word",
                        "generation_state": "done",
                    }
                    for i in range(1, 11)
                ],
            )
            session.add(row.Theme(id=1, key="test", name="Test", voice="Test", diction="Test"))
            await session.flush()
            await session.execute(
                insert(row.Sense),
                [
                    {"id": i * 10 + j, "word_id": i, "pos": "noun", "tier": "core"}
                    for i in range(1, 11)
                    for j in range(3)
                ],
            )
            await session.execute(
                insert(row.Definition),
                [
                    {"sense_id": i * 10 + j, "content": f"meaning{i}-{j}"}
                    for i in range(1, 11)
                    for j in range(3)
                ],
            )
            session.add(row.Definition(sense_id=100, theme_id=1, content="Never include themed"))
        question = (
            await append(
                db,
                [
                    Question(
                        0,
                        10,
                        None,
                        "cloze_to_word",
                        "Please _.",
                        Option("yes", "saved", "Fits"),
                        [],
                    )
                ],
            )
        )[0]

        async def ranked_search(_db, _source, _answer, *, limit):
            assert limit == 1
            return SearchResult(
                [WordHit(i, f"word{i}", "word", "fuzzy", f"word{i}") for i in range(10, 0, -1)]
            )

        monkeypatch.setattr("lexi_ai.questions.grade.search", ranked_search)
        event.listen(db.engine.sync_engine, "before_cursor_execute", count)
        decision = Decision(choices={"matched_sense": "sense_101"})
        grade = await grade_answer(
            db,
            decision,
            question.id,
            "single_word",
            "submitted",
            config=CONFIG,
        )
        assert grade.task_fit and grade.sense_id == 101
        assert len(statements) == 2
        criteria = decision.calls[1][1]["matched_sense"].criteria
        assert len(criteria) == 4
        assert list(criteria)[1:] == ["sense_100", "sense_101", "sense_102"]
        assert criteria["sense_100"] == "word10 - noun - meaning10-0"
        assert not any(
            "examples" in statement or "sense_forms" in statement for statement in statements
        )
        statements.clear()
        exact = await grade_answer(db, None, question.id, "single_word", "SAVED", config=CONFIG)
        assert exact.task_fit
        assert len(statements) == 1
        monkeypatch.setattr("lexi_ai.questions.grade.search", search)
        statements.clear()
        actual = await grade_answer(
            db,
            Decision(choices={"matched_sense": "sense_10"}),
            question.id,
            "single_word",
            "word",
            config=CONFIG,
        )
        assert actual.task_fit and actual.sense_id == 10
        assert len(statements) == 4  # Saved artifact + lexical query + pattern page + projection.
        with pytest.raises(ValueError):
            await meaning_inventory(db)
    finally:
        if event.contains(db.engine.sync_engine, "before_cursor_execute", count):
            event.remove(db.engine.sync_engine, "before_cursor_execute", count)
        await db.close()


async def test_reader_scope_order_namespace_and_shared_fingerprint(tmp_path, monkeypatch):
    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'reader.db'}")
    statements, hashes = [], []

    def count(_conn, _cursor, statement, parameters, _context, _executemany):
        statements.append((statement, parameters))

    def fingerprint(values):
        hashes.append(values)
        return definition_hash(values)

    monkeypatch.setattr("lexi_ai.words.storage.definition_hash", fingerprint)
    try:
        await db.create_schema(row.Base.metadata)
        async with db.transaction() as session:
            await session.execute(
                insert(row.Word),
                [
                    {
                        "id": i,
                        "lemma": f"word{i}",
                        "match_key": f"word{i}",
                        "entry_type": "word",
                        "generation_state": "done",
                    }
                    for i in [1, 2]
                ],
            )
            session.add(row.Theme(id=1, key="test", name="Test", voice="Test", diction="Test"))
            await session.flush()
            await session.execute(
                insert(row.Sense),
                [
                    {
                        "id": i,
                        "word_id": 1 if i <= 501 else 2,
                        "pos": "noun",
                        "tier": "core",
                    }
                    for i in range(1, 503)
                ],
            )
            await session.execute(
                insert(row.Definition),
                [{"sense_id": i, "content": f"meaning{i}"} for i in range(1, 503)],
            )
            await session.execute(
                insert(row.Example),
                [{"sense_id": i, "content": f"example{i}"} for i in range(1, 503)],
            )
            session.add(row.Definition(sense_id=1, theme_id=1, content="themed"))
            await session.flush()
            session.add_all(
                [
                    row.SenseRelation(
                        from_sense_id=i,
                        to_word_id=2,
                        to_sense_id=502,
                        rel_type="synonym",
                        gloss="target",
                        target_hash=definition_hash("meaning502"),
                        resolve_attempted_at="2026-09-30",
                    )
                    for i in [1, 2]
                ]
            )
        event.listen(db.engine.sync_engine, "before_cursor_execute", count)
        senses = await get_senses(db, [2, 999, 1, 2])
        assert [sense.id for sense in senses] == [2, 1, 2]
        assert [sense.definition.content for sense in senses] == [
            "meaning2",
            "meaning1",
            "meaning2",
        ]
        assert all(sense.relations[0].resolution_state == "resolved" for sense in senses)
        assert hashes == ["meaning502"]
        assert len(statements) == 1
        assert "WHERE senses.id IN" in statements[0][0]
        assert statements[0][1][-3:] == (2, 999, 1)
        assert (await get_senses(db, [1], theme_id=1))[0].definition.content == "themed"
        assert await get_senses(db, []) == []
        statements.clear()
        assert len(await get_senses(db, list(range(1, 502)))) == 501
        assert len(statements) == 2  # One fresh aggregate per bounded ID batch.
        hashes.clear()
        statements.clear()
        word = await get_word(db, 1)
        assert len(word.senses) == 501
        assert len(statements) == 1
        assert hashes == ["meaning502"]
    finally:
        if event.contains(db.engine.sync_engine, "before_cursor_execute", count):
            event.remove(db.engine.sync_engine, "before_cursor_execute", count)
        await db.close()
