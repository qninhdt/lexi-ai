import pytest
from test_prompting import prompt_context

from lexi_ai import DecisionConfig, Lexicon, LLMConfig
from lexi_ai.errors import InvalidResourceError


class LLM:
    def __init__(self):
        self.calls = 0

    async def complete(self, instruction, data, schema):
        self.calls += 1
        if schema.__name__ == "WordOutput":
            return schema.model_validate(
                {
                    "lemma": "bank",
                    "entry_type": "word",
                    "aliases": [],
                    "related": [],
                    "senses": [
                        {
                            "definition": "A place to keep money",
                            "pos": "noun",
                            "tier": "core",
                            "examples": ['The <t inf="base">bank</t> opens early.'],
                            "forms": [],
                            "patterns": [],
                            "collocations": [],
                            "relations": [],
                            "references": [{"source": "cambridge", "source_ref": "101"}],
                        }
                    ],
                }
            )
        if schema.__name__ == "TranslationOutput":
            return schema(content="ngân hàng")
        raise AssertionError("unexpected provider call")


async def test_selected_search_generate_read_and_close(tmp_path, source):
    llm = LLM()
    ai = Lexicon(
        f"sqlite+aiosqlite:///{tmp_path / 'generated.db'}",
        str(source),
        decision_config=DecisionConfig(0.8),
        llm=llm,
    )
    try:
        from lexi_ai.schema import Base

        await ai.db.create_schema(Base.metadata)
        available = (await ai.search("bank", include_available=True)).available
        assert len(available) == 1
        assert not hasattr(available[0], "cambridge_id")
        word = await ai.generate(available[0].available_id, example_count=1)
        assert word.lemma == "bank"
        assert (await ai.get_word(word.id)).senses[0].definition.content == (
            "A place to keep money"
        )
        assert await ai.generate(available[0].available_id) == word
        assert llm.calls == 1
        original_path = ai.cambridge.path
        ai.cambridge.path = tmp_path / "absent-source.db"
        assert await ai.generate(available[0].available_id) == word
        ai.cambridge.path = original_path
        assert (await ai.search("bank", include_available=True)).available == []
        assert await ai.translate_text("bank", "vi") == "ngân hàng"
        assert await ai.translate_text("bank", "vi") == "ngân hàng"
        assert llm.calls == 2
        with pytest.raises(InvalidResourceError):
            await ai.get_word(word.id, theme="unknown")
    finally:
        await ai.close()
    await ai.close()
    with pytest.raises(RuntimeError, match="closed"):
        await ai.get_word(word.id)


async def test_explicit_config_ignores_environment(monkeypatch, tmp_path, source):
    settings = {
        "LEXI_DB_URL": f"sqlite+aiosqlite:///{tmp_path / 'generated.db'}",
        "LEXI_CAMBRIDGE_DB_PATH": str(source),
        "LEXI_NEUTRAL_EXAMPLES": "3",
        "LEXI_THEMED_EXAMPLES": "5",
        "LEXI_DECISION_THRESHOLD": "0.85",
    }
    for name, value in settings.items():
        monkeypatch.setenv(name, value)
    ai = Lexicon(
        f"sqlite+aiosqlite:///{tmp_path / 'explicit.db'}",
        str(source),
        decision_config=DecisionConfig(
            0.7, api_key="decision-key", model="selected-decision", base_url="https://decision.test"
        ),
        llm_config=LLMConfig(api_key="llm-key", model="selected-llm", base_url="https://llm.test"),
    )
    try:
        assert not hasattr(Lexicon, "from_settings")
        assert not hasattr(ai, "neutral_counts") and not hasattr(ai, "themed_counts")
        assert ai.decision_config.accepts(0.7)
        assert not ai.decision_config.accepts(0.69)
        assert ai._decision_model().config is ai.decision_config
        assert ai._decision_model().llm_config is ai.llm_config
        assert ai.db.engine.url.database == str(tmp_path / "explicit.db")
        assert ai.cambridge.path == source
    finally:
        await ai.close()


def test_topic_surface_is_removed():
    assert not any("topic" in name for name in dir(Lexicon))


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
    owned = Lexicon(url, str(source), **options)
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
            count = prompt_context(data, "generation_parameters")["examples_per_sense"]
            if schema.__name__ == "ThemedWord":
                self.calls += 1
                return schema(
                    senses=[
                        {
                            "definition": "A safe place for coin",
                            "examples": ['The <t inf="base">bank</t> opens.'] * count,
                        }
                    ]
                )
            output = await super().complete(instruction, data, schema)
            output.senses[0].examples *= count
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
        handle = (await lexicon.search("bank", include_available=True)).available[0].available_id
        if neutral_first:
            await lexicon.generate(handle, example_count=2)
        await lexicon.create_theme("pirate", "Pirate", "nautical voice")
        themed = await lexicon.generate(handle, theme="pirate", example_count=4)
        assert len(themed.senses[0].examples) == 4
        neutral = await lexicon.get_word(themed.id)
        assert len(neutral.senses[0].examples) == (2 if neutral_first else 4)
        assert neutral.senses[0].definition is not None
        assert themed.senses[0].definition is not None
        assert llm.calls == 3
        assert await lexicon.generate(handle, example_count=9) == neutral
        assert await lexicon.generate(handle, theme="pirate", example_count=9) == themed
        assert llm.calls == 3
    finally:
        await lexicon.close()


@pytest.mark.parametrize("count", [0, -1, True, 1.5, "3", None])
async def test_generate_rejects_invalid_count_before_io(tmp_path, source, count):
    lexicon = Lexicon(
        f"sqlite+aiosqlite:///{tmp_path / 'unused.db'}",
        str(source),
        decision_config=DecisionConfig(0.8),
    )
    try:
        with pytest.raises(ValueError, match="positive integer"):
            await lexicon.generate("invalid-handle", example_count=count)
        assert lexicon.llm is None
        assert not (tmp_path / "unused.db").exists()
    finally:
        await lexicon.close()
