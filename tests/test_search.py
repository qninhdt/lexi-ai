from lexi_ai.db.session import Database
from lexi_ai.references.cambridge import SourceHit
from lexi_ai.schema import Base, Sense, SenseForm, SensePattern, Word, WordAlias, WordSource
from lexi_ai.words.search import search


class Source:
    async def search(self, query):
        assert query == "take off"
        return [SourceHit(1, "take off", "phrasal_verb"), SourceHit(2, "take off", "phrasal_verb")]


async def test_generated_ranking_and_consumed_handles(tmp_path):
    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'generated.db'}")
    try:
        await db.create_schema(Base.metadata)
        async with db.transaction() as session:
            word = Word(
                lemma="take off",
                match_key="take off",
                entry_type="phrasal_verb",
                generation_state="done",
            )
            session.add(word)
            await session.flush()
            session.add(WordSource(word_id=word.id, source_id=1))
            sense = Sense(word_id=word.id, pos="verb", tier="common")
            session.add(sense)
            await session.flush()
            session.add_all(
                [
                    SenseForm(sense_id=sense.id, inf="past", surface="took off"),
                    SensePattern(sense_id=sense.id, content="take {sth} off"),
                ]
            )
        assert (await search(db, None, "take off")).words[0].match_kind == "lemma"
        separated = await search(db, None, "took it off")
        assert separated.words[0].word_id == word.id
        assert separated.words[0].match_kind == "pattern"
        result = await search(db, Source(), "take off", include_available=True)
        assert len(result.available) == 1
        assert result.words[0].word_id == word.id
        assert (await search(db, None, "took it")).words == []
    finally:
        await db.close()


async def test_exact_alias_and_form_are_not_hidden_behind_fuzzy_window(tmp_path):
    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'generated.db'}")
    try:
        await db.create_schema(Base.metadata)
        async with db.transaction() as session:
            session.add_all(
                [
                    Word(
                        lemma=f"word{i}",
                        match_key=f"word{i}",
                        entry_type="word",
                        generation_state="done",
                    )
                    for i in range(5000)
                ]
            )
            await session.flush()
            target = Word(
                lemma="take off",
                match_key="take off",
                entry_type="phrasal_verb",
                generation_state="done",
            )
            session.add(target)
            await session.flush()
            session.add(WordAlias(word_id=target.id, content="lift off", match_key="lift off"))
            sense = Sense(word_id=target.id, pos="verb", tier="core")
            session.add(sense)
            await session.flush()
            session.add(SenseForm(sense_id=sense.id, inf="past", surface="took off"))
        assert (await search(db, None, "lift off")).words[0].word_id == target.id
        assert (await search(db, None, "TOOK OFF")).words[0].word_id == target.id
    finally:
        await db.close()
