"""Sense Linking reads stay batched with complete same-POS neutral inventories."""

import pytest
from sqlalchemy import event, insert

from lexi_ai import schema as row
from lexi_ai.db.session import Database
from lexi_ai.relations.storage import definition_hash, pending_relations


@pytest.mark.parametrize("batch_size", [20, 50])
async def test_pending_relations_uses_one_query_per_page(tmp_path, batch_size):
    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'generated.db'}")
    statements = []

    def count(_connection, _cursor, statement, _parameters, _context, _executemany):
        statements.append(statement)

    try:
        await db.create_schema(row.Base.metadata)
        async with db.transaction() as session:
            await session.execute(
                insert(row.Theme),
                [{"id": 1, "key": "test", "name": "Test", "voice": "Test", "diction": "Test"}],
            )
            await session.execute(
                insert(row.Word),
                [
                    {
                        "id": i + 1,
                        "lemma": f"word{i}",
                        "match_key": f"word{i}",
                        "entry_type": "WORD",
                        "generation_state": "DONE",
                    }
                    for i in range(51)
                ],
            )
            await session.execute(
                insert(row.Sense),
                [{"id": i + 1, "word_id": 1, "pos": "VERB", "tier": "CORE"} for i in range(50)]
                + [
                    {
                        "id": 1000 + i * 20 + j,
                        "word_id": i + 2,
                        "pos": "NOUN" if j == 0 else "VERB",
                        "tier": "CORE",
                    }
                    for i in range(50)
                    for j in range(14)
                ],
            )
            await session.execute(
                insert(row.Definition),
                [{"sense_id": i + 1, "content": "source"} for i in range(50)]
                + [
                    {"sense_id": 1000 + i * 20 + j, "content": f"neutral {j}"}
                    for i in range(50)
                    for j in range(14)
                ],
            )
            await session.execute(
                insert(row.Definition),
                [
                    {"sense_id": 1000 + i * 20 + j, "theme_id": 1, "content": "themed"}
                    for i in range(50)
                    for j in range(14)
                ],
            )
            await session.execute(
                insert(row.SenseRelation),
                [
                    {
                        "id": i + 1,
                        "from_sense_id": i + 1,
                        "to_word_id": i + 2,
                        "rel_type": "SYNONYM",
                    }
                    for i in range(50)
                ],
            )
        event.listen(db.engine.sync_engine, "before_cursor_execute", count)
        links = await pending_relations(db, batch_size)
        assert len(statements) == 1
        assert [link.edge_id for link in links] == list(range(1, batch_size + 1))
        for i, link in enumerate(links):
            assert link.source_definition == "source"
            assert [candidate.id for candidate in link.candidates] == [
                1000 + i * 20 + j for j in range(1, 14)
            ]
            for j, candidate in enumerate(link.candidates, start=1):
                assert candidate.definition == f"neutral {j}"
                assert candidate.fingerprint == definition_hash(f"neutral {j}")
    finally:
        if event.contains(db.engine.sync_engine, "before_cursor_execute", count):
            event.remove(db.engine.sync_engine, "before_cursor_execute", count)
        await db.close()
