import asyncio

import pytest
from test_public_api import LLM

from lexi_ai import Lexicon
from lexi_ai.config import MAX_QUERY_LENGTH
from lexi_ai.db.session import Database
from lexi_ai.errors import InvalidResourceError
from lexi_ai.models import Option, Question
from lexi_ai.questions.storage import append
from lexi_ai.references.cambridge import SourceHit
from lexi_ai.schema import Base, Sense, SenseForm, SensePattern, Word, WordAlias, WordSource
from lexi_ai.words.search import search


class Source:
    async def projection(self):
        return [(SourceHit(i, "take off", "phrasal_verb"), ["take off"]) for i in (1, 2)]


async def test_lexi_ranking_and_consumed_cambridge_handles(tmp_path):
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
        assert (await search(db, None, "take off")).words[0].match_kind == "LEMMA"
        separated = await search(db, None, "took it off")
        assert separated.words[0].word_id == word.id
        assert separated.words[0].match_kind == "PATTERN"
        result = await search(db, Source(), "take off", include_available=True)
        assert len(result.available) == 1
        assert result.words[0].word_id == word.id
        assert (await search(db, None, "took it")).words == []
    finally:
        await db.close()


async def test_process_reconciliation_and_cache_lifecycle(tmp_path, source):
    url = f"sqlite+aiosqlite:///{tmp_path / 'shared.db'}"
    reader = Lexicon(url, str(source), llm=object(), refresh_seconds=0.02)
    worker = Lexicon(url, str(source), llm=LLM())
    try:
        await reader.db.create_schema(Base.metadata)
        await reader.start()
        initial = await reader.search("bank", include_available=True)
        assert len(initial.available) == 1 and initial.words == []
        initial.available.clear()
        assert len((await reader.search("bank", include_available=True)).available) == 1
        word = await worker.generate(
            (await worker.search("bank", include_available=True)).available[0].available_id,
            target="bank",
            example_count=1,
        )
        # Polling includes index rebuild time; a started reader reconciles without
        # any message from the independent publishing Lexicon.
        async with asyncio.timeout(2):
            while not (result := await reader.search("bank", include_available=True)).words:
                await asyncio.sleep(0.01)
        assert result.words[0].word_id == word.id and result.available == []
        result.words.clear()
        assert (await reader.search("bank")).words[0].word_id == word.id
        question = (
            await append(
                worker.db,
                [
                    Question(
                        0,
                        word.senses[0].id,
                        None,
                        "DEFINITION_TO_WORD",
                        "Prompt",
                        Option("yes", "bank", "Fits"),
                        [],
                    )
                ],
            )
        )[0]
        assert await reader.get_question(question.id) == question
        await worker.delete_question(question.id)
        await asyncio.sleep(0.03)
        assert await reader.get_question(question.id) is None
        for query in ("", " " * 3, "x" * (MAX_QUERY_LENGTH + 1)):
            with pytest.raises(InvalidResourceError):
                await reader.search(query)
    finally:
        await worker.close()
        await reader.close()
    assert reader._refresh_task.done()
    assert reader.db.search_index is None
    assert reader._search.snapshot is None and reader._search.reference is None
    assert reader.db.question_cache.get(question.id) is None


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
        assert (await search(db, None, "lift off")).words[0].word_id == target.id
        assert (await search(db, None, "TOOK OFF")).words[0].word_id == target.id
    finally:
        await db.close()
