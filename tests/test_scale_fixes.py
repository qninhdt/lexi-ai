"""Indexed form lookup and transactional, targeted Sense Linking requeueing."""

from types import SimpleNamespace

import pytest
from sqlalchemy import event, insert, select, text, update

from lexi_ai import schema as row
from lexi_ai.db.session import Database
from lexi_ai.relations.resolve import resolve_relations
from lexi_ai.relations.storage import definition_hash, pending_relations
from lexi_ai.words.search import search


async def seed_links(db):
    async with db.transaction() as session:
        session.add_all(
            [
                row.Word(
                    id=i,
                    lemma=f"word{i}",
                    match_key=f"word{i}",
                    entry_type="word",
                    generation_state="done",
                )
                for i in range(1, 6)
            ]
        )
        session.add(row.Theme(id=1, key="test", name="Test", voice="Test", diction="Test"))
        await session.flush()
        session.add_all([row.Sense(id=i, word_id=i, pos="verb", tier="core") for i in range(1, 6)])
        await session.flush()
        session.add_all(
            [row.Definition(id=i, sense_id=i, content=f"meaning{i}") for i in range(1, 6)]
        )
        session.add(row.Definition(id=6, sense_id=2, theme_id=1, content="themed meaning"))
        await session.flush()
        session.add_all(
            [
                row.SenseRelation(
                    id=i,
                    from_sense_id=source,
                    to_word_id=target,
                    to_sense_id=target,
                    rel_type="synonym",
                    gloss=f"meaning{target}",
                    target_hash=definition_hash(f"meaning{target}"),
                    resolve_attempted_at="2026-09-30T00:00:00Z",
                )
                for i, source, target in [(1, 1, 2), (2, 3, 4)]
            ]
        )


async def test_form_keys_cover_unicode_spacing_updates_and_core_inserts(tmp_path):
    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'forms.db'}")
    try:
        await db.create_schema(row.Base.metadata)
        await seed_links(db)
        async with db.transaction() as session:
            form = row.SenseForm(sense_id=1, surface="ＴＯＯＫ　ＯＦＦ", inf="past")
            session.add(form)
            await session.flush()
            form_id = form.id
            assert form.match_key == "took off"
            await session.execute(
                insert(row.SenseForm),
                [
                    {"sense_id": 1, "surface": "Straße", "inf": "base"},
                ],
            )
        hit = (await search(db, None, "TOOK OFF")).words[0]
        assert (hit.word_id, hit.match_kind) == (1, "form")
        async with db.transaction() as session:
            assert (
                await session.scalar(
                    select(row.SenseForm.match_key).where(row.SenseForm.surface == "Straße")
                )
                == "strasse"
            )
            (await session.get(row.SenseForm, form_id)).surface = "  Taken   Off "
        assert (await search(db, None, "TAKEN OFF")).words[0].match_kind == "form"
        async with db.engine.connect() as connection:
            plan = (
                await connection.execute(
                    text(
                        "EXPLAIN QUERY PLAN SELECT id FROM sense_forms WHERE match_key='taken off'"
                    )
                )
            ).all()
        assert "ix_sense_forms_match_key" in str(plan)
    finally:
        await db.close()


@pytest.mark.parametrize(
    "change",
    [
        "target_definition",
        "source_definition",
        "add_definition",
        "delete_definition",
        "neutral_to_theme",
        "theme_to_neutral",
        "sense_pos",
        "sense_owner",
        "add_sense",
        "delete_sense",
        "themed_update",
        "themed_delete",
        "themed_insert",
        "noop",
    ],
)
async def test_relation_invalidation_is_targeted_and_ignores_themed_or_noop_changes(
    tmp_path, change
):
    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'relations.db'}")
    try:
        await db.create_schema(row.Base.metadata)
        await seed_links(db)
        async with db.transaction() as session:
            if change in {"target_definition", "source_definition", "noop"}:
                identifier = 1 if change == "source_definition" else 2
                await session.execute(
                    text("UPDATE definitions SET content=:value WHERE id=:id"),
                    {
                        "id": identifier,
                        "value": "meaning2" if change == "noop" else "changed meaning",
                    },
                )
            elif change in {"add_definition", "themed_insert"}:
                await session.execute(
                    text("DELETE FROM definitions WHERE id=:id"),
                    {"id": 6 if change == "themed_insert" else 2},
                )
                session.add(
                    row.Definition(
                        sense_id=2,
                        content="extra",
                        theme_id=(1 if change == "themed_insert" else None),
                    )
                )
            elif change in {"delete_definition", "themed_delete"}:
                await session.execute(
                    text("DELETE FROM definitions WHERE id=:id"),
                    {"id": 6 if change == "themed_delete" else 2},
                )
            elif change in {"neutral_to_theme", "theme_to_neutral", "themed_update"}:
                if change == "themed_update":
                    (await session.get(row.Definition, 6)).content = "new themed text"
                else:
                    await session.execute(
                        text("DELETE FROM definitions WHERE id=:id"),
                        {"id": 6 if change == "neutral_to_theme" else 2},
                    )
                    definition = await session.get(
                        row.Definition,
                        2 if change == "neutral_to_theme" else 6,
                    )
                    definition.theme_id = 1 if change == "neutral_to_theme" else None
            elif change == "add_sense":
                session.add(row.Sense(word_id=2, pos="noun", tier="core"))
            elif change == "delete_sense":
                await session.delete(await session.get(row.Sense, 2))
            else:
                sense = await session.get(row.Sense, 2)
                if change == "sense_pos":
                    sense.pos = "noun"
                else:
                    sense.word_id = 5
        invalidated = change not in {"themed_update", "themed_delete", "themed_insert", "noop"}
        async with db.transaction() as session:
            edge = await session.get(row.SenseRelation, 1)
            assert (edge.resolve_attempted_at is None) == invalidated
            assert (edge.to_sense_id is None) == invalidated
            assert (edge.target_hash is None) == invalidated
            unaffected = await session.get(row.SenseRelation, 2)
            assert unaffected.to_sense_id == 4
            assert unaffected.resolve_attempted_at is not None
    finally:
        await db.close()


async def test_definition_invalidation_rolls_back_and_requeues_no_match(tmp_path):
    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'relations.db'}")
    try:
        await db.create_schema(row.Base.metadata)
        await seed_links(db)
        with pytest.raises(RuntimeError):
            async with db.transaction() as session:
                await session.execute(text("UPDATE definitions SET content='changed' WHERE id=2"))
                assert (
                    await session.scalar(
                        select(row.SenseRelation.to_sense_id).where(row.SenseRelation.id == 1)
                    )
                    is None
                )
                raise RuntimeError("rollback")
        async with db.transaction() as session:
            assert (await session.get(row.SenseRelation, 1)).to_sense_id == 2
            assert (await session.get(row.Definition, 2)).content == "meaning2"
            await session.execute(
                update(row.SenseRelation)
                .where(row.SenseRelation.id == 1)
                .values(to_sense_id=None, target_hash=None)
            )
        assert await pending_relations(db, 20) == []
        async with db.transaction() as session:
            await session.execute(text("UPDATE definitions SET content='changed' WHERE id=2"))
        assert [link.edge_id for link in await pending_relations(db, 20)] == [1]
    finally:
        await db.close()


@pytest.mark.parametrize(
    "choice,change",
    [("no_candidate", "target"), ("candidate_1", "source"), ("no_candidate", "inventory")],
)
async def test_changed_provider_evidence_is_not_written_as_a_completed_decision(
    tmp_path,
    choice,
    change,
):
    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'relations.db'}")

    class ChangingDecision:
        async def decide(self, _state, _questions):
            async with db.transaction() as session:
                if change == "inventory":
                    sense = row.Sense(word_id=2, pos="verb", tier="core")
                    session.add(sense)
                    await session.flush()
                    session.add(row.Definition(sense_id=sense.id, content="new candidate"))
                else:
                    identifier = 1 if change == "source" else 2
                    (await session.get(row.Definition, identifier)).content = "changed"
            return SimpleNamespace(choices={"matched_sense": SimpleNamespace(choice=choice)})

    try:
        await db.create_schema(row.Base.metadata)
        await seed_links(db)
        async with db.transaction() as session:
            await session.execute(
                update(row.SenseRelation)
                .where(row.SenseRelation.id == 1)
                .values(to_sense_id=None, target_hash=None, resolve_attempted_at=None)
            )
        results = await resolve_relations(db, ChangingDecision())
        assert [result.state for result in results] == ["noop"]
        assert [link.edge_id for link in await pending_relations(db, 20)] == [1]
    finally:
        await db.close()


async def test_pending_lookup_skips_100000_fresh_resolutions(tmp_path):
    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'scale.db'}")
    statements = []

    def count(_conn, _cursor, statement, parameters, _context, _executemany):
        statements.append((statement, parameters))

    try:
        await db.create_schema(row.Base.metadata)
        fingerprint = definition_hash("target")
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
            await session.execute(
                insert(row.Sense),
                [{"id": 200001, "word_id": 2, "pos": "verb", "tier": "core"}],
            )
            await session.execute(
                insert(row.Definition), [{"sense_id": 200001, "content": "target"}]
            )
            for first in range(1, 100002, 5000):
                ids = range(first, min(first + 5000, 100002))
                await session.execute(
                    insert(row.Sense),
                    [{"id": i, "word_id": 1, "pos": "verb", "tier": "core"} for i in ids],
                )
                await session.execute(
                    insert(row.Definition), [{"sense_id": i, "content": "source"} for i in ids]
                )
                await session.execute(
                    insert(row.SenseRelation),
                    [
                        {
                            "id": i,
                            "from_sense_id": i,
                            "to_word_id": 2,
                            "rel_type": "synonym",
                            "gloss": "target",
                            "to_sense_id": 200001 if i <= 100000 else None,
                            "target_hash": fingerprint if i <= 100000 else None,
                            "resolve_attempted_at": "2026-09-30T00:00:00Z" if i <= 100000 else None,
                        }
                        for i in ids
                    ],
                )
        event.listen(db.engine.sync_engine, "before_cursor_execute", count)
        assert [link.edge_id for link in await pending_relations(db, 1)] == [100001]
        assert len(statements) == 1
        statement, parameters = statements[0]
        async with db.engine.connect() as connection:
            plan = (
                await connection.exec_driver_sql(
                    "EXPLAIN QUERY PLAN " + statement,
                    parameters,
                )
            ).all()
        assert "ix_sense_relations_target_pending" in str(plan)
        async with db.transaction() as session:
            await session.execute(
                update(row.SenseRelation)
                .where(row.SenseRelation.id == 100001)
                .values(
                    to_sense_id=200001,
                    target_hash=fingerprint,
                    resolve_attempted_at="2026-09-30T00:00:00Z",
                )
            )
        statements.clear()
        assert await pending_relations(db, 1) == []
        assert len(statements) == 1
    finally:
        if event.contains(db.engine.sync_engine, "before_cursor_execute", count):
            event.remove(db.engine.sync_engine, "before_cursor_execute", count)
        await db.close()
