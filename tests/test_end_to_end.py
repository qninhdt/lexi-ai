"""Consumer flow across migrated storage and a read-only Reference fixture."""

import hashlib
from types import SimpleNamespace

import pytest
from alembic import command
from sqlalchemy import select
from test_migrations import migration_config, run_migration
from test_prompting import bound_content, prompt_context
from test_word_generation import stage_payload

from lexi_ai import DecisionConfig, Lexicon
from lexi_ai.db.session import Database
from lexi_ai.errors import InvalidOutputError
from lexi_ai.inference.config import LLMConfig
from lexi_ai.inference.llm import OpenAIStructuredLLM
from lexi_ai.references.cambridge import SourceEntry, SourceSense, encode_reference_id
from lexi_ai.schema import Base, Definition, Example, Sense, Theme, Word
from lexi_ai.themes.service import ThemedWord, ensure_word_theme
from lexi_ai.words.generate import generate_word


class LLM:
    def __init__(self):
        self.calls = []

    async def complete(self, instruction, data, schema):
        self.calls.append(schema.__name__)
        if schema.__name__ in {"InventoryOutput", "EnrichmentBatch"}:
            return stage_payload(
                {
                    "lemma": "bank",
                    "type": "WORD",
                    "aliases": [],
                    "related": {},
                    "senses": [
                        {
                            "definition": "A place to keep money",
                            "pos": "NOUN",
                            "tier": "CORE",
                            "cefr_level": "A1",
                            "register": None,
                            "examples": ["The [bank] opened."],
                            "forms": [],
                            "patterns": [],
                            "collocations": [],
                            "relations": {"SYNONYM": ["vault"]},
                            "references": ["a1"],
                        }
                    ],
                },
                data,
                schema,
            )
        if schema.__name__ == "ThemeParts":
            return schema(voice="Captain", diction="nautical")
        if schema.__name__ == "ThemedWord":
            return schema(
                senses=[
                    {
                        "definition": "A safe house for treasure",
                        "examples": ["The [bank] holds coin."],
                    }
                ]
            )
        if schema.__name__ in {"QuestionBatch", "AnchoredQuestionBatch"}:
            context = prompt_context(data)
            kind = context["question_type"]
            content = bound_content(context)
            if kind == "WORD_TO_USAGE":
                correct = "The [bank] opened."
            else:
                correct = "bank"
            return schema.model_validate(
                {
                    "questions": [
                        {
                            "content": content,
                            **(
                                {"correct_explanation": "Fits the intended meaning."}
                                if schema.__name__ == "AnchoredQuestionBatch"
                                else {
                                    "correct": {
                                        "content": correct,
                                        "explanation": "Fits the intended meaning.",
                                    }
                                }
                            ),
                            "distractors": [
                                {
                                    "content": f"wrong{len(self.calls)}q{item}x{i}",
                                    "explanation": "Does not fit.",
                                }
                                for i in range(context["distractors_per_question"])
                            ],
                        }
                        for item in range(context["count"])
                    ]
                }
            )
        if schema.__name__ == "TranslationOutput":
            return schema(content="ngân hàng")
        raise AssertionError("unexpected LLM schema")


class Decision:
    def __init__(self):
        self.calls = 0

    async def decide(self, state, questions, **kwargs):
        self.calls += 1
        return SimpleNamespace(
            choices={
                name: SimpleNamespace(
                    choice={
                        "matched_sense": "candidate_1",
                        "meaning": "correct",
                        "form": "correct",
                        "collocation": "natural",
                        "appropriacy": "appropriate",
                    }[name]
                )
                for name in questions
                if name not in {"used", "construction"}
            },
            nouls={
                name: SimpleNamespace(noul=0.85)
                for name in questions
                if name in {"used", "construction"}
            },
        )


async def test_selected_to_theme_question_grade_sense_linking_and_translation(tmp_path, source):
    url = f"sqlite+aiosqlite:///{tmp_path / 'generated.db'}"
    await run_migration(migration_config(url), url, command.upgrade, "head")
    await verify_consumer_flow(url, source)


async def verify_consumer_flow(url, source, *, db_schema=None, session=None):
    before = hashlib.sha256(source.read_bytes()).digest()
    llm, decision = LLM(), Decision()
    postgres = (
        (session.get_bind().dialect.name == "postgresql")
        if session
        else url.startswith("postgresql")
    )
    ai = Lexicon(
        url,
        "" if postgres else str(source),
        decision_config=DecisionConfig(0.8),
        db_schema=db_schema,
        llm=llm,
        decision_model=decision,
        session=session,
    )
    try:
        if postgres and session is None:
            await ai.import_reference(source)
        async with ai.db.transaction() as session:
            target = Word(
                lemma="vault", match_key="vault", entry_type="WORD", generation_state="DONE"
            )
            session.add(target)
            await session.flush()
            target_sense = Sense(word_id=target.id, pos="NOUN", tier="CORE")
            session.add(target_sense)
            await session.flush()
            session.add(Definition(sense_id=target_sense.id, content="secure place for valuables"))
        await ai.start()
        references = (await ai.search("bank", include_reference=True)).items
        assert len(references) == 1
        assert llm.calls == []
        word = await ai.generate_word(
            "bank", reference_id=references[0].reference_id, example_count=1
        )
        sense_id = word.senses[0].id
        theme = await ai.create_theme("pirate", "Pirate", "nautical voice")
        assert await ai.get_theme(theme.key) == theme
        assert await ai.list_themes() == [theme]
        assert (await ai.get_word(word.id, theme.key)) is None
        themed = await ai.generate_word(
            "bank", theme=theme.key, reference_id=references[0].reference_id, example_count=1
        )
        assert themed.senses[0].definition.content == "A safe house for treasure"
        assert (await ai.get_word(word.id)).senses[0].definition.content == (
            "A place to keep money"
        )
        generated = await ai.generate_questions(
            sense_id, "DEFINITION_TO_WORD", 2, distractor_count=3, theme=theme.key
        )
        await ai.generate_questions(
            sense_id, "DEFINITION_TO_WORD", 2, distractor_count=3, theme=theme.key
        )
        assert len(await ai.list_questions(sense_id, theme=theme.key)) == 4
        assert await ai.list_questions(sense_id) == []
        assert (await ai.retrieve_question(sense_id, theme=theme.key)).id in {
            item.id for item in await ai.list_questions(sense_id, theme=theme.key)
        }
        assert (
            await ai.grade_answer(generated[0].id, "SINGLE_CHOICE", generated[0].correct.id)
        ).task_fit
        assert (await ai.grade_answer(generated[0].id, "SINGLE_WORD", "BANK")).task_fit
        usage = (await ai.generate_questions(sense_id, "WORD_TO_USAGE", 1, distractor_count=3))[0]
        assert (
            await ai.grade_answer(usage.id, "SHORT_ANSWER", "The bank safeguards our money.")
        ).used
        assert decision.calls == 2
        assert (await ai.resolve_relations())[0].state == "RESOLVED"
        assert (await ai.get_word(word.id)).senses[0].relations[0].to_sense_id == target_sense.id
        assert decision.calls == 3
        assert await ai.translate_text("bank", "vi") == "ngân hàng"
        translations = await ai.list_translations()
        assert len(translations) == 1
        assert await ai.get_translation(translations[0].id) == translations[0]
        calls_before_reads = len(llm.calls)
        assert await ai.translate_text("bank", "vi") == "ngân hàng"
        assert await ai.get_question(usage.id) == usage
        assert (await ai.get_word(word.id)).id == word.id
        assert len(llm.calls) == calls_before_reads
        assert llm.calls.count("AnchoredQuestionBatch") == 2
        assert llm.calls.count("QuestionBatch") == 1
        assert await ai.delete_theme(theme.key)
        assert await ai.list_questions(sense_id) == [usage]
        async with ai.db.transaction() as session:
            assert len((await session.scalars(select(Definition))).all()) == 2
    finally:
        await ai.close()
    assert hashlib.sha256(source.read_bytes()).digest() == before


async def test_oversized_selected_source_fails_without_partial_word(tmp_path, monkeypatch):
    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'generated.db'}")

    class HugeSource:
        async def fetch_by_id(self, entry_id):
            assert entry_id == 1
            return SourceEntry(
                1,
                "bank",
                "bank",
                "word",
                [
                    SourceSense(101, "noun", "meaning " * 10000),
                ],
            )

    async def no_wordnet(_citation):
        return []

    monkeypatch.setattr("lexi_ai.words.generate.lookup", no_wordnet)
    try:
        await db.create_schema(Base.metadata)
        llm = OpenAIStructuredLLM(LLMConfig())
        with pytest.raises(ValueError, match="invalid structured request"):
            await generate_word(db, HugeSource(), llm, encode_reference_id(1), 1, target="bank")
        async with db.transaction() as session:
            assert (await session.scalars(select(Word))).all() == []
    finally:
        await db.close()


async def test_last_invalid_themed_sense_rolls_back_entire_namespace(tmp_path):
    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'generated.db'}")

    class IncompleteTheme:
        async def complete(self, _instruction, _data, schema):
            assert schema is ThemedWord
            return schema(
                senses=[
                    {
                        "definition": "safe place for coins",
                        "examples": ["The [bank] opens."],
                    },
                    {"definition": "river's edge", "examples": []},
                ]
            )

    try:
        await db.create_schema(Base.metadata)
        async with db.transaction() as session:
            word = Word(lemma="bank", match_key="bank", entry_type="WORD", generation_state="DONE")
            theme = Theme(key="pirate", name="Pirate", voice="Captain", diction="nautical")
            session.add_all([word, theme])
            await session.flush()
            for index in range(2):
                sense = Sense(word_id=word.id, pos="NOUN", tier="CORE")
                session.add(sense)
                await session.flush()
                session.add_all(
                    [
                        Definition(sense_id=sense.id, content=f"Meaning {index}"),
                        Example(sense_id=sense.id, content="The [bank] opened."),
                    ]
                )
        with pytest.raises(InvalidOutputError, match="incomplete themed"):
            await ensure_word_theme(db, IncompleteTheme(), word.id, "pirate", 1)
        async with db.transaction() as session:
            assert not (
                await session.scalars(select(Definition).where(Definition.theme_id == theme.id))
            ).all()
            assert not (
                await session.scalars(select(Example).where(Example.theme_id == theme.id))
            ).all()
            assert len((await session.scalars(select(Definition))).all()) == 2
    finally:
        await db.close()
