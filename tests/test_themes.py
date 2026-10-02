from sqlalchemy import select
from test_prompting import prompt_context

from lexi_ai.db.session import Database
from lexi_ai.schema import Base, Definition, Example, Sense, Word
from lexi_ai.themes.service import create_theme, delete_theme, ensure_word_theme, update_theme
from lexi_ai.words.storage import get_word


class LLM:
    def __init__(self):
        self.calls = 0

    async def complete(self, instruction, data, schema):
        self.calls += 1
        if schema.__name__ == "ThemeParts":
            return schema(voice="Captain", diction="nautical")
        return schema(
            senses=[
                {
                    "definition": "A safe place for coin",
                    "examples": [
                        'The <t inf="base">bank</t> was open.',
                        'I visited the <t inf="base">bank</t>.',
                    ],
                }
            ]
        )


async def test_crud_delete_and_generate_reuse(tmp_path):
    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'dictionary.db'}")
    llm = LLM()
    try:
        await db.create_schema(Base.metadata)
        async with db.transaction() as session:
            word = Word(lemma="bank", match_key="bank", entry_type="word", generation_state="done")
            session.add(word)
            await session.flush()
            sense = Sense(word_id=word.id, pos="noun", tier="core")
            session.add(sense)
            await session.flush()
            session.add_all(
                [
                    Definition(sense_id=sense.id, content="A financial institution"),
                    Example(sense_id=sense.id, content='The <t inf="base">bank</t> opened.'),
                ]
            )
        theme = await create_theme(db, llm, "pirate", "Pirate", "Write like a pirate")
        assert await get_word(db, word.id, theme.id) is None
        themed = await ensure_word_theme(db, llm, word.id, "pirate", 2)
        assert themed.senses[0].definition.content == "A safe place for coin"
        reused = await ensure_word_theme(db, llm, word.id, "pirate", 3)
        assert reused.id == word.id
        assert llm.calls == 2
        assert (await update_theme(db, "pirate", name="Pirates", voice="Admiral")).key == "pirate"
        assert (await get_word(db, word.id, theme.id)).senses[0].definition.content == (
            "A safe place for coin"
        )
        assert await delete_theme(db, "pirate")
        assert (await get_word(db, word.id)).senses[0].definition.content == (
            "A financial institution"
        )
        async with db.transaction() as session:
            assert len((await session.scalars(select(Definition))).all()) == 1
    finally:
        await db.close()


async def test_theme_uses_single_neutral_meaning(tmp_path):
    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'dictionary.db'}")
    try:
        await db.create_schema(Base.metadata)
        async with db.transaction() as session:
            word = Word(lemma="bank", match_key="bank", entry_type="word", generation_state="done")
            session.add(word)
            await session.flush()
            sense = Sense(word_id=word.id, pos="noun", tier="core")
            session.add(sense)
            await session.flush()
            session.add_all(
                [
                    Definition(sense_id=sense.id, content="A financial institution"),
                    Example(sense_id=sense.id, content='The <t inf="base">bank</t> opened.'),
                ]
            )
        llm = LLM()
        await create_theme(db, llm, "pirate", "Pirate", "Speak like a captain")

        class InspectingLLM(LLM):
            async def complete(self, instruction, data, schema):
                if schema.__name__ == "ThemedWord":
                    assert (
                        prompt_context(data, "neutral_word")["senses"][0]["meaning_anchor"]
                        == "A financial institution"
                    )
                return await super().complete(instruction, data, schema)

        themed = await ensure_word_theme(db, InspectingLLM(), word.id, "pirate", 2)
        assert themed.senses[0].definition.content == "A safe place for coin"
    finally:
        await db.close()
