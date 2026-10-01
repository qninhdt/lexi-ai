from sqlalchemy import select

from lexi_ai.db.session import Database
from lexi_ai.schema import Base, Definition, Example, Sense, Theme, Word
from lexi_ai.words.storage import get_senses, get_word


async def test_detached_and_namespace_exact(tmp_path):
    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'dictionary.db'}")
    try:
        await db.create_schema(Base.metadata)
        async with db.transaction() as session:
            word = Word(lemma="bank", match_key="bank", entry_type="word", generation_state="done")
            theme = Theme(key="pirate", name="Pirate", voice="captain", diction="nautical")
            session.add_all([word, theme])
            await session.flush()
            sense = Sense(word_id=word.id, pos="noun", tier="core")
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
