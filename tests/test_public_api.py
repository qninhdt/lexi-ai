from types import SimpleNamespace

import pytest
from test_decision import DecisionTransport
from test_prompting import prompt_context
from test_word_generation import stage_payload

from lexi_ai import DecisionConfig, DecisionMode, Lexicon, LLMConfig
from lexi_ai.errors import InvalidResourceError, MissingProviderError
from lexi_ai.references.cambridge import encode_reference_id


class LLM:
    def __init__(self):
        self.calls = 0

    async def complete(self, instruction, data, schema):
        self.calls += 1
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
                            "examples": ["The [bank] opens early."],
                            "forms": [],
                            "patterns": [],
                            "collocations": [],
                            "relations": {},
                            "references": ["a1"],
                        }
                    ],
                },
                data,
                schema,
            )
        if schema.__name__ == "TranslationOutput":
            return schema(content="ngân hàng")
        raise AssertionError("unexpected provider call")


async def test_selected_search_generate_read_and_close(tmp_path, source):
    llm = LLM()
    ai = Lexicon(
        f"sqlite+aiosqlite:///{tmp_path / 'generated.db'}",
        str(source),
        llm=llm,
    )
    try:
        from lexi_ai.schema import Base

        await ai.db.create_schema(Base.metadata)
        await ai.start()
        references = (await ai.search("bank", include_reference=True)).items
        assert len(references) == 1
        word = await ai.generate_word(
            "bank", reference_id=references[0].reference_id, example_count=1
        )
        assert word.lemma == "bank"
        assert word.type == "WORD"
        assert (await ai.get_word(word.id)).senses[0].definition.content == (
            "A place to keep money"
        )
        assert await ai.generate_word("bank", reference_id=references[0].reference_id) == word
        assert llm.calls == 2
        original_path = ai._cambridge.path
        ai._cambridge.path = tmp_path / "absent-source.db"
        assert await ai.generate_word("bank", reference_id=references[0].reference_id) == word
        ai._cambridge.path = original_path
        assert (await ai.search("bank", include_reference=True)).items[0].kind == "WORD"
        assert await ai.translate_text("bank", "vi") == "ngân hàng"
        assert await ai.translate_text("bank", "vi") == "ngân hàng"
        assert llm.calls == 3
        with pytest.raises(InvalidResourceError):
            await ai.get_word(word.id, theme="unknown")
    finally:
        await ai.close()
    await ai.close()
    with pytest.raises(RuntimeError, match="closed"):
        await ai.get_word(word.id)


async def test_explicit_config_ignores_environment(monkeypatch, tmp_path, source):
    monkeypatch.setenv("LLM_API_KEY", "ignored-key")
    monkeypatch.setenv("DB_URL", "ignored-url")
    ai = Lexicon(
        f"sqlite+aiosqlite:///{tmp_path / 'explicit.db'}",
        str(source),
        decision_config=DecisionConfig(
            0.7, api_key="decision-key", model="selected-decision", base_url="https://decision.test"
        ),
        llm_config=LLMConfig(api_key="llm-key", model="selected-llm", base_url="https://llm.test"),
    )
    try:
        assert ai.decision_config.accepts(0.7)
        assert not ai.decision_config.accepts(0.69)
        assert ai._decision_model().config is ai.decision_config
        assert ai._decision_model().llm_config is ai.llm_config
        assert ai.db.engine.url.database == str(tmp_path / "explicit.db")
        assert ai._cambridge.path == source
    finally:
        await ai.close()


async def test_close_only_owned_providers_once(monkeypatch, tmp_path, source):
    class Provider:
        def __init__(self):
            self.closes = 0

        async def close(self):
            self.closes += 1

    llm, decision = Provider(), Provider()
    url = f"sqlite+aiosqlite:///{tmp_path / 'generated.db'}"
    options = dict(
        decision_config=DecisionConfig(0.8),
    )
    injected = Lexicon(url, str(source), llm=llm, decision_model=decision, **options)
    await injected.close()
    await injected.close()
    assert (llm.closes, decision.closes) == (0, 0)

    monkeypatch.setattr("lexi_ai.api.OpenAIStructuredLLM", lambda config: llm)
    monkeypatch.setattr("lexi_ai.api.DecisionModel", lambda *args, **kwargs: decision)
    owned = Lexicon(url, str(source), llm_config=LLMConfig(api_key="fake-key"), **options)
    assert owned._llm() is llm and owned._decision_model() is decision
    await owned.close()
    await owned.close()
    assert (llm.closes, decision.closes) == (1, 1)


@pytest.mark.parametrize("neutral_first", [False, True])
async def test_example_counts_are_per_generate_call(tmp_path, source, neutral_first):
    from lexi_ai.schema import Base

    class CountingLLM(LLM):
        async def complete(self, instruction, data, schema):
            if schema.__name__ == "ThemeParts":
                self.calls += 1
                return schema(voice="Captain", diction="nautical")
            if schema.__name__ == "InventoryOutput":
                return await super().complete(instruction, data, schema)
            tag = "generation_parameters" if schema.__name__ == "ThemedWord" else "sense_request"
            count = prompt_context(data, tag)["examples_per_sense"]
            if schema.__name__ == "ThemedWord":
                self.calls += 1
                return schema(
                    senses=[
                        {
                            "definition": "A safe place for coin",
                            "examples": ["The [bank] opens."] * count,
                        }
                    ]
                )
            output = await super().complete(instruction, data, schema)
            for sense in output.senses:
                sense.examples = [f"The [bank] opens at {hour}." for hour in range(count)]
            return output

    llm = CountingLLM()
    lexicon = Lexicon(
        f"sqlite+aiosqlite:///{tmp_path / 'counts.db'}",
        str(source),
        decision_config=DecisionConfig(0.8),
        llm=llm,
    )
    try:
        await lexicon.db.create_schema(Base.metadata)
        await lexicon.start()
        handle = (await lexicon.search("bank", include_reference=True)).items[0].reference_id
        if neutral_first:
            await lexicon.generate_word("bank", reference_id=handle, example_count=2)
        await lexicon.create_theme("pirate", "Pirate", "nautical voice")
        themed = await lexicon.generate_word(
            "bank", reference_id=handle, theme="pirate", example_count=4
        )
        assert len(themed.senses[0].examples) == 4
        neutral = await lexicon.get_word(themed.id)
        assert len(neutral.senses[0].examples) == (2 if neutral_first else 4)
        assert neutral.senses[0].definition is not None
        assert themed.senses[0].definition is not None
        assert llm.calls == 4
        assert await lexicon.generate_word("bank", reference_id=handle, example_count=9) == neutral
        assert (
            await lexicon.generate_word(
                "bank", reference_id=handle, theme="pirate", example_count=9
            )
        ) == themed
        assert llm.calls == 4
    finally:
        await lexicon.close()


@pytest.mark.parametrize("count", [0, -1, True, 1.5, "3", None])
async def test_generate_rejects_invalid_count_before_io(tmp_path, source, count):
    lexicon = Lexicon(
        f"sqlite+aiosqlite:///{tmp_path / 'unused.db'}",
        str(source),
        decision_config=DecisionConfig(0.8),
        llm_config=LLMConfig(api_key="fake-key"),
    )
    try:
        with pytest.raises(ValueError, match="positive integer"):
            await lexicon.generate_word("bank", reference_id="invalid-handle", example_count=count)
        assert lexicon.llm is None
        assert not (tmp_path / "unused.db").exists()
    finally:
        await lexicon.close()


@pytest.mark.parametrize("llm_config", [None, LLMConfig(), LLMConfig(api_key=" ")])
async def test_reads_and_imports_do_not_require_llm_credentials(tmp_path, source, llm_config):
    lexicon = Lexicon(
        f"sqlite+aiosqlite:///{tmp_path / 'unused.db'}", str(source), llm_config=llm_config
    )
    try:
        with pytest.raises(MissingProviderError, match="requires an LLM"):
            lexicon._llm()
        assert not (tmp_path / "unused.db").exists()
        await lexicon.validate_reference(encode_reference_id(1))
    finally:
        await lexicon.close()


async def test_decision_only_requires_credentials_only_for_inference(tmp_path, source, monkeypatch):
    from lexi_ai.schema import Base

    lexicon = Lexicon(
        f"sqlite+aiosqlite:///{tmp_path / 'unused.db'}",
        str(source),
        llm_config=LLMConfig(api_key="fake-key"),
    )

    async def question(*args):
        return SimpleNamespace(
            sense_id=1,
            content="Financial institution",
            question_type="DEFINITION_TO_WORD",
            correct=SimpleNamespace(id="yes", content="bank"),
            distractors=[],
            supports=lambda fmt: fmt in {"SINGLE_CHOICE", "SINGLE_WORD"},
        )

    monkeypatch.setattr("lexi_ai.questions.grade.get_question", question)
    try:
        for fmt, answer in (("SINGLE_WORD", " BANK "), ("SINGLE_CHOICE", "yes")):
            grade, usage = await lexicon.grade_answer(
                1, fmt, answer, mode=DecisionMode.DECISION_ONLY, with_usage=True
            )
            assert grade.task_fit is True and usage == []
        with pytest.raises(MissingProviderError, match="decision model credentials"):
            await lexicon.grade_answer(1, "SINGLE_WORD", "answer", mode=DecisionMode.DECISION_ONLY)
        await lexicon.db.create_schema(Base.metadata)
        await lexicon.start()
        assert await lexicon.resolve_relations(mode=DecisionMode.DECISION_ONLY) == []
        assert lexicon.decision_model.primary is None and lexicon.llm is None
    finally:
        await lexicon.close()


async def test_public_grading_defaults_to_llm_without_jev(tmp_path, source, monkeypatch):
    lexicon = Lexicon(
        f"sqlite+aiosqlite:///{tmp_path / 'unused.db'}",
        str(source),
        llm_config=LLMConfig(api_key="fake-key"),
    )
    fallback = DecisionTransport(probability=0.1)
    lexicon._decision_model().fallback = fallback

    async def question(*args):
        return SimpleNamespace(
            content="A financial institution",
            question_type="DEFINITION_TO_WORD",
            correct=SimpleNamespace(content="bank"),
            supports=lambda fmt: fmt == "SINGLE_WORD",
        )

    monkeypatch.setattr("lexi_ai.questions.grade.get_question", question)
    try:
        grade, usage = await lexicon.grade_answer(1, "SINGLE_WORD", "tree", with_usage=True)
        assert grade.task_fit is False and grade.spelling_error is False
        assert grade.sense_id is None
        assert len(fallback.calls) == 1
        assert lexicon.decision_model.primary is None
        assert len(usage) == 1 and usage[0].input_tokens is None
    finally:
        await lexicon.close()


async def test_public_api_list_questions_for_senses(tmp_path, source):
    llm = LLM()
    lexicon = Lexicon(
        f"sqlite+aiosqlite:///{tmp_path / 'batch_api.db'}",
        str(source),
        llm=llm,
    )
    try:
        from lexi_ai.models import Option, Question
        from lexi_ai.questions.storage import append
        from lexi_ai.schema import Base, Sense, Word

        await lexicon.db.create_schema(Base.metadata)
        await lexicon.start()
        async with lexicon.db.transaction() as session:
            word = Word(lemma="coin", match_key="coin", entry_type="WORD", generation_state="DONE")
            session.add(word)
            await session.flush()
            sense = Sense(word_id=word.id, pos="NOUN", tier="CORE")
            session.add(sense)
            await session.flush()

        q = Question(
            0,
            sense.id,
            None,
            "DEFINITION_TO_WORD",
            "A metal currency",
            Option("1", "coin", "exp"),
            [],
        )
        await append(lexicon.db, [q])

        res = await lexicon.list_questions_for_senses([sense.id])
        assert len(res) == 1
        assert res[0].sense_id == sense.id
        assert res[0].question_type == "DEFINITION_TO_WORD"
        from lexi_ai import QuestionType

        selected = await lexicon.retrieve_questions(
            [(sense.id, QuestionType.DEFINITION_TO_WORD, 1)]
        )
        assert selected == res
    finally:
        await lexicon.close()


def test_postgresql_reference_path_requires_import_before_runtime():
    with pytest.raises(ValueError, match="import_reference"):
        Lexicon("postgresql+asyncpg://unused/lexicon", reference_path="reference.sqlite")
