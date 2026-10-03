import asyncio

from sqlalchemy import event, select

from lexi_ai import Lexicon
from lexi_ai.cache import Cache
from lexi_ai.db.session import Database
from lexi_ai.schema import Base, Definition, Example, Sense, Theme, Word
from lexi_ai.words.storage import get_senses, get_word


async def test_byte_cache_eviction_oversize_expiry_and_mutation_isolation():
    cache = Cache(24)
    value = {"items": [1]}
    cache.put("first", value)
    value["items"].append(2)
    cached = cache.get("first")
    assert cached == {"items": [1]}
    cached["items"].append(3)
    assert cache.get("first") == {"items": [1]}
    cache.put("second", {"items": [2]})
    assert cache.get("first") is None
    cache.put("second", "x" * 30)
    assert cache.get("second") is None
    cache.put("missing", None)
    assert cache.get_many(["missing", "absent"]) == {}
    expiring = Cache(64, ttl=0.01)
    expiring.put("key", value)
    await asyncio.sleep(0.02)
    assert expiring.get("key") is None


async def test_detached_and_namespace_exact(tmp_path):
    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'dictionary.db'}")
    try:
        await db.create_schema(Base.metadata)
        async with db.transaction() as session:
            word = Word(lemma="bank", match_key="bank", entry_type="WORD", generation_state="DONE")
            theme = Theme(key="pirate", name="Pirate", voice="captain", diction="nautical")
            session.add_all([word, theme])
            await session.flush()
            sense = Sense(word_id=word.id, pos="NOUN", tier="CORE")
            session.add(sense)
            await session.flush()
            session.add_all(
                [
                    Definition(sense_id=sense.id, content="A financial institution"),
                    Example(sense_id=sense.id, content='The <t inf="base">bank</t> closed.'),
                ]
            )
        neutral = await get_word(db, word.id)
        assert neutral.senses[0].definition.content == "A financial institution"
        assert await get_word(db, word.id, theme.id) is None
        async with db.transaction() as session:
            session.add_all(
                [
                    Definition(sense_id=sense.id, theme_id=theme.id, content="A pirate's lender"),
                    Example(
                        sense_id=sense.id,
                        theme_id=theme.id,
                        content='The <t inf="base">bank</t> took my coin.',
                    ),
                ]
            )
        themed = await get_word(db, word.id, theme.id)
        assert themed.senses[0].definition.content == "A pirate's lender"
        assert (await get_senses(db, [sense.id, 999]))[0].id == sense.id
        async with db.transaction() as session:
            assert len((await session.scalars(select(Word))).all()) == 1
    finally:
        await db.close()


async def test_word_sense_and_preview_caches_bulk_misses_and_detached_models(tmp_path):
    lexicon = Lexicon(f"sqlite+aiosqlite:///{tmp_path / 'cache.db'}", llm=object())
    db = lexicon.db
    statements = []

    def capture(_conn, _cursor, sql, _parameters, _context, _many):
        statements.append(sql)

    try:
        await db.create_schema(Base.metadata)
        async with db.transaction() as session:
            word = Word(lemma="bank", match_key="bank", entry_type="WORD", generation_state="DONE")
            session.add(word)
            await session.flush()
            senses = [Sense(word_id=word.id, pos="NOUN", tier="CORE") for _ in range(3)]
            session.add_all(senses)
            await session.flush()
            session.add_all([Definition(sense_id=s.id, content=f"meaning{s.id}") for s in senses])
            session.add(Example(sense_id=senses[0].id, content="An example"))
        event.listen(db.engine.sync_engine, "before_cursor_execute", capture)
        value = await lexicon.get_word(word.id)
        assert len(statements) == 1
        value.aliases.append("changed")
        value.senses[0].examples.clear()
        statements.clear()
        cached = await lexicon.get_word(word.id)
        assert cached.aliases == [] and len(cached.senses[0].examples) == 1
        assert cached.type is value.type
        assert (await lexicon.get_senses([senses[2].id, senses[0].id])) == [
            cached.senses[2],
            cached.senses[0],
        ]
        assert statements == []  # Word reads warm the neutral Sense entries.
        db.content_cache.clear()
        await lexicon.get_senses([senses[0].id])
        statements.clear()
        ids = [senses[2].id, senses[0].id, senses[1].id, 999]
        assert [s.id for s in await lexicon.get_senses(ids)] == ids[:3]
        assert len(statements) == 1  # All misses including the absent ID in one read.
        statements.clear()
        assert [s.id for s in await lexicon.get_senses(ids[:3])] == ids[:3]
        assert statements == []
        previews = await lexicon.get_sense_previews(ids[:3])
        statements.clear()
        assert await lexicon.get_sense_previews(ids[:3]) == previews
        assert statements == []
    finally:
        event.remove(db.engine.sync_engine, "before_cursor_execute", capture)
        await lexicon.close()
