"""Default-planner and nested-trigger work, not just application query counts."""

import json
from contextlib import asynccontextmanager

import pytest
from sqlalchemy import event, insert, select, text
from test_db_optimization import seed
from test_query_work import shared_target

from lexi_ai import schema as row
from lexi_ai.questions.storage import list_for_sense, retrieve
from lexi_ai.relations.storage import _pending_page, definition_hash, pending_relations
from lexi_ai.words.search import Search


def plan_nodes(node):
    yield node
    for child in node.get("Plans", []):
        yield from plan_nodes(child)


async def explain(db, statement, parameters=None):
    async with db.read() as connection:
        if parameters is None:
            statement = str(
                statement.compile(
                    dialect=db.engine.dialect,
                    compile_kwargs={"literal_binds": True},
                )
            )
            parameters = ()
        result = await connection.exec_driver_sql(
            "EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) " + statement,
            parameters,
        )
        return result.scalar_one()[0]


@asynccontextmanager
async def nested_plans(session):
    connection = await session.connection()
    raw = await connection.get_raw_connection()
    driver = raw.driver_connection
    plans = []

    def capture(_connection, message):
        if "plan:\n" in message.message:
            plans.append(json.loads(message.message.split("plan:\n", 1)[1]))

    await session.execute(text("LOAD 'auto_explain'"))
    await session.execute(text("SET LOCAL auto_explain.log_nested_statements=on"))
    await session.execute(text("SET LOCAL auto_explain.log_analyze=on"))
    await session.execute(text("SET LOCAL auto_explain.log_buffers=on"))
    await session.execute(text("SET LOCAL auto_explain.log_format='json'"))
    await session.execute(text("SET LOCAL auto_explain.log_level='notice'"))
    await session.execute(text("SET LOCAL auto_explain.log_min_duration=0"))
    driver.add_log_listener(capture)
    try:
        yield plans
    finally:
        driver.remove_log_listener(capture)


async def test_question_read_and_trigger_seek_scope_without_forced_planner(pg_db):
    await seed(pg_db)
    async with pg_db.transaction() as session:
        await session.execute(text("ALTER TABLE questions DISABLE TRIGGER USER"))
        await session.execute(
            text("""
            INSERT INTO questions(id,sense_id,theme_id,question_type,payload,position,type_position)
            SELECT n,1,CASE WHEN n<=20000 THEN NULL ELSE 1 END,
              CASE WHEN n%2=0 THEN 'DEFINITION_TO_WORD' ELSE 'WORD_TO_DEFINITION' END,
              '{}',CASE WHEN n<=20000 THEN n ELSE n-20000 END,
              CASE WHEN n<=20000 THEN (n+1)/2 ELSE (n-20000+1)/2 END
            FROM generate_series(1,40000) n
        """)
        )
        await session.execute(text("ALTER TABLE questions ENABLE TRIGGER USER"))
        await session.execute(text("ANALYZE questions"))
    captured = []

    def capture(_connection, _cursor, sql, parameters, _context, _many):
        captured.append((sql, parameters))

    for theme in (None, 1):
        for kind in (None, "DEFINITION_TO_WORD"):
            event.listen(pg_db.engine.sync_engine, "before_cursor_execute", capture)
            try:
                with pytest.raises(KeyError):  # Fixture payloads omit artifact fields.
                    await retrieve(pg_db, 1, kind, theme)
            finally:
                event.remove(pg_db.engine.sync_engine, "before_cursor_execute", capture)
            plan = await explain(pg_db, *captured[-1])
            scans = [n for n in plan_nodes(plan["Plan"]) if n.get("Relation Name") == "questions"]
            assert scans and all(n["Actual Rows"] <= 1 for n in scans), scans
            assert all("Index" in n["Node Type"] for n in scans), scans
            event.listen(pg_db.engine.sync_engine, "before_cursor_execute", capture)
            try:
                with pytest.raises(KeyError):
                    await list_for_sense(
                        pg_db,
                        1,
                        kind,
                        theme,
                        after_id=1000 if theme is None else 21000,
                        limit=25,
                    )
            finally:
                event.remove(pg_db.engine.sync_engine, "before_cursor_execute", capture)
            plan = await explain(pg_db, *captured[-1])
            scans = [n for n in plan_nodes(plan["Plan"]) if n.get("Relation Name") == "questions"]
            assert scans and all(n["Actual Rows"] <= 25 for n in scans), scans
            assert all("Index" in n["Node Type"] for n in scans), scans
        async with pg_db.transaction() as session, nested_plans(session) as plans:
            await session.execute(
                text("""
                INSERT INTO questions(id,sense_id,theme_id,question_type,payload)
                VALUES(:id,1,:theme,'DEFINITION_TO_WORD','{}')
            """),
                {"id": 50000 if theme is None else 50001, "theme": theme},
            )
            await session.execute(
                text("DELETE FROM questions WHERE id=:id"), {"id": 2 if theme is None else 20002}
            )
        scans = [
            n
            for plan in plans
            for n in plan_nodes(plan["Plan"])
            if n.get("Relation Name") == "questions" and "Scan" in n["Node Type"]
        ]
        assert scans and all("Index" in n["Node Type"] for n in scans), scans
        assert all(
            n["Actual Rows"] <= 1 and n.get("Rows Removed by Filter", 0) <= 1 for n in scans
        ), scans


async def test_definition_batch_updates_each_relation_once(pg_db):
    await shared_target(pg_db)
    async with pg_db.transaction() as session:
        await session.execute(
            insert(row.Sense).values(
                [dict(id=2000 + i, word_id=2, pos="NOUN", tier="CORE") for i in range(20)]
            )
        )
        async with nested_plans(session) as plans:
            await session.execute(
                insert(row.Definition).values(
                    [dict(sense_id=2000 + i, content="additional meaning") for i in range(20)]
                )
            )
    updates = [p for p in plans if p["Query Text"].lstrip().startswith("UPDATE sense_relations")]
    assert len(updates) == 1
    updated = [n for n in updates[0]["Plan"]["Plans"] if n.get("Parent Relationship") == "Outer"]
    assert sum(n["Actual Rows"] * n["Actual Loops"] for n in updated) == 50


async def test_pending_page_skips_deferred_edges_and_reuses_target_candidates(pg_db):
    await shared_target(pg_db)
    async with pg_db.transaction() as session:
        await session.execute(
            insert(row.Sense).values(
                [dict(id=1000 + i, word_id=2, pos="NOUN", tier="CORE") for i in range(12, 25)]
            )
        )
        await session.execute(
            insert(row.Definition).values(
                [dict(sense_id=1000 + i, content=f"additional target{i}") for i in range(12, 25)]
            )
        )
        await session.execute(text("UPDATE sense_relations SET id=id+100000"))
        await session.execute(
            text(
                "INSERT INTO words(id,lemma,match_key,generation_state) "
                "VALUES(3,'later','later','PENDING')"
            )
        )
        await session.execute(
            text("""
            INSERT INTO words(id,lemma,match_key,generation_state)
            SELECT 200000+n,'unrelated'||n,'unrelated'||n,'DONE' FROM generate_series(1,10000)n
        """)
        )
        await session.execute(
            text("""
            INSERT INTO senses(id,word_id,pos,tier)
            SELECT 10000+n,1,'NOUN','CORE' FROM generate_series(1,100000)n
        """)
        )
        await session.execute(
            text("""
            INSERT INTO definitions(sense_id,content)
            SELECT 10000+n,'source' FROM generate_series(1,100000)n
        """)
        )
        await session.execute(
            text("""
            INSERT INTO sense_relations(id,from_sense_id,to_word_id,rel_type)
            SELECT n,10000+n,3,'SYNONYM' FROM generate_series(1,100000)n
        """)
        )
        await session.execute(text("ANALYZE"))
    plan = await explain(pg_db, _pending_page(0, 50, postgres=True))
    nodes = list(plan_nodes(plan["Plan"]))
    edges = [n for n in nodes if n.get("Relation Name") == "sense_relations"]
    assert edges and all("Index" in n["Node Type"] for n in edges), edges
    assert all(n["Actual Rows"] * n["Actual Loops"] <= 150 for n in edges), edges
    candidate_scans = [n for n in nodes if n.get("Alias") == "candidate_definition"]
    assert sum(n["Actual Rows"] * n["Actual Loops"] for n in candidate_scans) == 25
    assert len((await pending_relations(pg_db, 50))[0].candidates) == 25
    assert not any(
        n.get("Relation Name") == "senses" and n["Node Type"] == "Seq Scan" for n in nodes
    )
    assert all(
        n["Actual Rows"] * n["Actual Loops"] <= 150
        for n in nodes
        if n.get("Relation Name") == "words"
    )
    assert [link.edge_id for link in await pending_relations(pg_db, 50)] == list(
        range(100001, 100051)
    )
    # Deferred work becomes eligible after publication; it must not be lost to a cursor/cap.
    async with pg_db.transaction() as session:
        await session.execute(
            text("INSERT INTO senses(id,word_id,pos,tier) VALUES(200000,3,'VERB','CORE')")
        )
        await session.execute(text("UPDATE words SET generation_state='DONE' WHERE id=3"))
    assert [link.edge_id for link in await pending_relations(pg_db, 1)] == [1]


@pytest.mark.parametrize(
    "change",
    [
        "definition_noop",
        "sense_noop",
        "themed",
        "content",
        "neutral_to_theme",
        "theme_to_neutral",
        "sense_pos",
        "sense_owner",
        "sense_delete",
        "definition_delete",
    ],
)
async def test_transition_invalidation_keeps_namespace_and_noop_rules(pg_db, change):
    await shared_target(pg_db, source_count=1)
    async with pg_db.transaction() as session:
        await session.execute(
            text("INSERT INTO themes(id,key,name,voice,diction) VALUES(1,'style','Style','V','D')")
        )
        await session.execute(
            text("INSERT INTO definitions(sense_id,theme_id,content) VALUES(1001,1,'style')")
        )
        await session.execute(
            text("""
            UPDATE sense_relations SET to_sense_id=1000,target_hash=:hash,
              resolve_attempted_at='done'
        """),
            {"hash": definition_hash("target")},
        )
        commands = {
            "definition_noop": "UPDATE definitions SET content=content WHERE sense_id=1000",
            "sense_noop": "UPDATE senses SET pos=pos WHERE id=1000",
            "themed": "UPDATE definitions SET content='different' WHERE theme_id=1",
            "content": "UPDATE definitions SET content='different' WHERE sense_id=1000",
            "neutral_to_theme": "UPDATE definitions SET theme_id=1 WHERE sense_id=1000",
            "theme_to_neutral": "DELETE FROM definitions WHERE sense_id=1001 AND theme_id IS NULL",
            "sense_pos": "UPDATE senses SET pos='VERB' WHERE id=1000",
            "sense_owner": "UPDATE senses SET word_id=1 WHERE id=1000",
            "sense_delete": "DELETE FROM senses WHERE id=1000",
            "definition_delete": "DELETE FROM definitions WHERE sense_id=1000",
        }
        await session.execute(text(commands[change]))
        if change == "theme_to_neutral":
            await session.execute(
                text("UPDATE definitions SET theme_id=NULL WHERE sense_id=1001 AND theme_id=1")
            )
    async with pg_db.read() as connection:
        attempted = await connection.scalar(select(row.SenseRelation.resolve_attempted_at))
    assert (attempted is None) == (change not in {"definition_noop", "sense_noop", "themed"})


async def test_pattern_projection_loads_once_then_matches_in_ram(pg_db):
    await seed(pg_db)
    async with pg_db.transaction() as session:
        await session.execute(
            insert(row.SensePattern.__table__).values(
                [dict(sense_id=1, content=f"word1 {{sth}} fixed{i}") for i in range(50)]
            )
        )
        await session.execute(
            insert(row.SenseForm.__table__).values(
                [dict(sense_id=1, surface=f"word1x{i}", inf="BASE") for i in range(20)]
            )
        )
        await session.execute(text("ANALYZE"))
    captured = []

    def capture(_connection, _cursor, sql, parameters, _context, _many):
        captured.append((sql, parameters))

    event.listen(pg_db.engine.sync_engine, "before_cursor_execute", capture)
    try:
        engine = Search(pg_db, None)
        await engine.start()
        assert len(captured) == 5
        assert all("definitions" not in sql and "questions" not in sql for sql, _ in captured)
        captured.clear()
        assert (await engine.search("word1 thing fixed1")).items[0].match_kind == "EXACT"
        assert captured == []
    finally:
        event.remove(pg_db.engine.sync_engine, "before_cursor_execute", capture)
