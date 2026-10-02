from lexi_ai.db.session import Database
from lexi_ai.models import Option, Question
from lexi_ai.questions.storage import append, get, list_for_sense, remove, retrieve
from lexi_ai.schema import Base, Sense, Theme, Word


async def test_append_get_list_retrieve_delete_and_exact_namespace(tmp_path):
    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'generated.db'}")
    try:
        await db.create_schema(Base.metadata)
        async with db.transaction() as session:
            word = Word(lemma="bank", match_key="bank", entry_type="word", generation_state="done")
            theme = Theme(key="pirate", name="Pirate", voice="Captain", diction="nautical")
            session.add_all([word, theme])
            await session.flush()
            sense = Sense(word_id=word.id, pos="noun", tier="core")
            session.add(sense)
            await session.flush()
        artifact = Question(
            0,
            sense.id,
            None,
            "definition_to_word",
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
