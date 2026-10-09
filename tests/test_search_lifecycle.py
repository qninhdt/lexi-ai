"""Search owns one immutable reader snapshot and publishes incremental changes."""

import asyncio
import threading

import pytest
from sqlalchemy import event

from lexi_ai.db.session import Database
from lexi_ai.references.cambridge import SourceHit
from lexi_ai.schema import Base, Word, WordAlias, WordSource
from lexi_ai.words.search import Search, search


async def test_incremental_update_preserves_old_snapshot_and_search_performs_no_sql(tmp_path):
    class Reference:
        async def projection(self):
            return [(SourceHit(1, "run", "word"), ["run", "running"], [])]

    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'content.sqlite'}")
    engine = Search(db, Reference())
    statements = []

    def capture(*args):
        statements.append(args[2])

    try:
        await db.create_schema(Base.metadata)
        await engine.start()
        db.search_index = engine
        old = engine.snapshot
        index = engine.index
        assert (await search(db, "running", True)).items[0].kind == "REFERENCE"
        async with db.transaction() as session:
            word = Word(lemma="run", match_key="run", entry_type="WORD", generation_state="DONE")
            session.add(word)
            await session.flush()
            session.add_all(
                [
                    WordSource(word_id=word.id, source_id=1),
                    WordAlias(word_id=word.id, content="running", match_key="running"),
                ]
            )
        event.listen(db.engine.sync_engine, "before_cursor_execute", capture)
        await engine.update(word.id)
        assert len(statements) == 5
        statements.clear()
        assert engine.index is index
        assert engine.snapshot is not old
        assert Search._find(old, "running", None, 30)[0][1].kind == "REFERENCE"
        for _ in range(3):
            hits = (await search(db, "running", True)).items
            assert len(hits) == 1 and hits[0].word_id == word.id
        await engine.start()
        assert statements == []
    finally:
        event.remove(db.engine.sync_engine, "before_cursor_execute", capture)
        await engine.close()
        await db.close()


async def test_helper_requires_explicit_startup(tmp_path):
    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'content.sqlite'}")
    try:
        with pytest.raises(RuntimeError, match="not started"):
            await search(db, "run")
        assert db.search_index is None
    finally:
        await db.close()


async def test_cancelled_update_finishes_native_write_before_close(tmp_path, monkeypatch):
    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'content.sqlite'}")
    engine = Search(db, None)
    entered, release = threading.Event(), threading.Event()
    try:
        await db.create_schema(Base.metadata)
        await engine.start()
        old = engine.snapshot
        async with db.transaction() as session:
            word = Word(lemma="run", match_key="run", entry_type="WORD", generation_state="DONE")
            session.add(word)
            await session.flush()
        original = engine._documents

        def blocked(*args):
            entered.set()
            assert release.wait(3)
            yield from original(*args)

        monkeypatch.setattr(engine, "_documents", blocked)
        update = asyncio.create_task(engine.update(word.id))
        assert await asyncio.to_thread(entered.wait, 3)
        update.cancel()
        closing = asyncio.create_task(engine.close())
        await asyncio.sleep(0)
        assert not closing.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await update
        await closing
        assert Search._find(old, "run", None, 30) == []
    finally:
        release.set()
        await engine.close()
        await db.close()
