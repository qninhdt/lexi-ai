import pytest

from lexi_ai.db.session import Database
from lexi_ai.references.cambridge import SourceHit
from lexi_ai.schema import Base, Sense, SenseForm, SensePattern, Word, WordAlias, WordSource
from lexi_ai.words.search import Search, search


class Source:
    async def projection(self):
        return [(SourceHit(i, "take off", "phrasal_verb"), ["take off"], []) for i in (1, 2)]


async def test_lexi_ranking_and_consumed_reference_handles(tmp_path):
    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'generated.db'}")
    try:
        await db.create_schema(Base.metadata)
        async with db.transaction() as session:
            word = Word(
                lemma="take off",
                match_key="take off",
                entry_type="PHRASAL_VERB",
                generation_state="DONE",
            )
            session.add(word)
            await session.flush()
            session.add(WordSource(word_id=word.id, source_id=1))
            sense = Sense(word_id=word.id, pos="VERB", tier="COMMON")
            session.add(sense)
            await session.flush()
            session.add_all(
                [
                    SenseForm(sense_id=sense.id, inf="PAST", surface="took off"),
                    SensePattern(sense_id=sense.id, content="take {sth} off"),
                ]
            )
        db.search_index = Search(db, Source())
        await db.search_index.start()
        assert (await search(db, "take off")).items[0].match_kind == "EXACT"
        separated = await search(db, "took it off")
        assert separated.items[0].word_id == word.id
        assert separated.items[0].match_kind == "EXACT"
        result = await search(db, "take off", include_reference=True)
        assert [hit.kind for hit in result.items] == ["WORD", "REFERENCE"]
        assert result.items[0].word_id == word.id
        assert result.items[1].match_kind == "EXACT"
        assert (await search(db, "took it")).items == []
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
                        entry_type="WORD",
                        generation_state="DONE",
                    )
                    for i in range(5000)
                ]
            )
            await session.flush()
            target = Word(
                lemma="take off",
                match_key="take off",
                entry_type="PHRASAL_VERB",
                generation_state="DONE",
            )
            session.add(target)
            await session.flush()
            session.add(WordAlias(word_id=target.id, content="lift off", match_key="lift off"))
            sense = Sense(word_id=target.id, pos="VERB", tier="CORE")
            session.add(sense)
            await session.flush()
            session.add(SenseForm(sense_id=sense.id, inf="PAST", surface="took off"))
        db.search_index = Search(db, None)
        await db.search_index.start()
        assert (await search(db, "lift off")).items[0].word_id == target.id
        assert (await search(db, "TOOK OFF")).items[0].word_id == target.id
    finally:
        await db.close()


@pytest.mark.parametrize(
    "lemma,pattern,surface,inflected,inf",
    [
        ("have", "have {done}", "have finished the report", "had", "PAST"),
        ("go", "go {adj}", "go very quiet", "went", "PAST"),
        ("do", "do {adv}", "do very well", "did", "PAST"),
    ],
)
async def test_grammar_phrase_slots_match_through_search(
    tmp_path, lemma, pattern, surface, inflected, inf
):
    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'content.sqlite'}")
    try:
        await db.create_schema(Base.metadata)
        async with db.transaction() as session:
            word = Word(lemma=lemma, match_key=lemma, entry_type="WORD", generation_state="DONE")
            session.add(word)
            await session.flush()
            sense = Sense(word_id=word.id, pos="VERB", tier="CORE")
            session.add(sense)
            await session.flush()
            session.add_all(
                [
                    SensePattern(sense_id=sense.id, content=pattern),
                    SenseForm(sense_id=sense.id, surface=inflected, inf=inf),
                ]
            )
        db.search_index = Search(db, None)
        await db.search_index.start()
        for query in [surface, inflected + surface[len(lemma) :]]:
            result = (await search(db, query)).items
            assert result[0].word_id == word.id
            assert result[0].match_kind == "EXACT"
            assert result[0].matched_surface == pattern
    finally:
        if db.search_index is not None:
            await db.search_index.close()
        await db.close()
