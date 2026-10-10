import asyncio
import sys
from collections import Counter
from dataclasses import replace
from types import SimpleNamespace

import pytest
from pydantic import ValidationError
from test_prompting import prompt_context
from test_questions_from_word import LLM as QuestionLLM
from test_word_generation import payload, stage_payload

import benchmarks.generate as generator
from benchmarks.generate import DEFAULT_CONFIG, load_config, read_records, run_generation
from benchmarks.run import load_dataset
from benchmarks.synthetic import (
    DEFINITION,
    SINGLE_WORD,
    USAGE,
    AnswerBatch,
    AnswerList,
    GenerationConfig,
    grade_answers,
    question_plan,
    stage_contexts,
    synthesis_context,
    synthesize,
)
from lexi_ai import DecisionConfig, Lexicon, LLMConfig, TokenUsage
from lexi_ai.errors import InvalidOutputError
from lexi_ai.inference.decision import DecisionModel
from lexi_ai.inference.llm import OpenAIStructuredLLM
from lexi_ai.inference.prompting import render_prompt
from lexi_ai.models import Definition, Option, Question, Sense, Word
from lexi_ai.schema import Base


def config():
    return GenerationConfig(
        entries=[{"target": "bank", "source_id": 1, "group": "word"}],
        themes=[
            {"key": f"theme{i}", "name": f"Theme {i}", "concept": "Creative world"}
            for i in range(3)
        ],
        questions_per_type=2,
    )


@pytest.mark.parametrize("override,expected", [(None, "xhigh"), ("low", "low")])
def test_generator_reasoning_uses_shared_env_unless_config_overrides(
    tmp_path,
    source,
    monkeypatch,
    override,
    expected,
):
    settings = config().model_copy(update={"reasoning_effort": override})
    captured = {}

    class Lexicon:
        def __init__(self, *_args, llm_config, **_kwargs):
            captured["config"] = llm_config

        async def start(self):
            pass

        async def close(self):
            pass

    async def generation(*_args, **_kwargs):
        return {}

    monkeypatch.setattr(generator, "Lexicon", Lexicon)
    monkeypatch.setattr(generator, "run_generation", generation)
    monkeypatch.setattr(generator, "load_config", lambda _path: settings)
    monkeypatch.setattr(
        generator,
        "load_provider_values",
        lambda _path: {
            "LLM_API_KEY": "fake",
            "LLM_BASE_URL": "https://llm.test/v1",
            "LLM_MODEL": "fixture",
            "LLM_STRUCTURED_OUTPUTS": "false",
            "LLM_REASONING_EFFORT": "xhigh",
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate",
            "--db-url",
            "sqlite+aiosqlite:///unused.db",
            "--output",
            str(tmp_path / "unused"),
            "--reference-path",
            str(source),
        ],
    )
    generator.main()
    assert captured["config"].reasoning_effort == expected
    assert captured["config"].structured_outputs is False
    assert not (tmp_path / "unused").exists()


def test_default_coverage_is_60_questions_180_answers_not_cross_product():
    settings = load_config(DEFAULT_CONFIG)
    plan = question_plan(settings)
    assert len(settings.entries) == 24
    assert len(plan) == 60
    assert Counter(job["question_type"] for job in plan) == {
        "DEFINITION_TO_WORD": 12,
        "CONTEXT_TO_WORD": 12,
        "CLOZE_TO_WORD": 12,
        "WORD_TO_DEFINITION": 12,
        "WORD_TO_USAGE": 12,
    }
    assert settings.themes == []
    assert Counter(job["theme"] for job in plan) == {None: 60}
    assert Counter(settings.entries[job["entry"]].group for job in plan) == {
        "word": 30,
        "phrasal_verb": 12,
        "idiom": 9,
        "phrase": 6,
        "expression": 3,
    }
    assert len({job["entry"] for job in plan}) == 24
    assert len({job["id"] for job in plan}) == 60
    assert question_plan(settings) == plan


@pytest.mark.parametrize("count", [0, 1, 2, 3])
def test_optional_themes_use_only_configured_namespaces(count):
    settings = config().model_dump()
    settings["themes"] = settings["themes"][:count]
    settings = GenerationConfig.model_validate(settings)
    plan = question_plan(settings)
    assert {job["theme"] for job in plan} == {None, *[theme.key for theme in settings.themes]}


async def test_neutral_source_stage_never_generates_themes_questions_or_answers(tmp_path, lexicon):
    library, llm = lexicon
    settings = config().model_dump(exclude={"themes"})
    settings = GenerationConfig.model_validate(settings)
    output = tmp_path / "neutral"
    await run_generation(library, settings, output=output, stage="sources", progress=False)
    assert llm.calls == {"InventoryOutput": 1, "EnrichmentBatch": 1}
    assert not llm.grading_calls
    saved = read_records(output / "sources.jsonl")
    assert set(saved) == {"word:0:neutral"}
    assert all(
        sense["definition"]["theme_id"] is None
        for sense in saved["word:0:neutral"]["data"]["senses"]
    )
    assert not any(
        (output / name).exists()
        for name in (
            "questions.jsonl",
            "answers.jsonl",
            "labels.jsonl",
            "cases",
        )
    )
    await run_generation(
        library,
        settings,
        output=output,
        stage="sources",
        resume=True,
        progress=False,
    )
    assert llm.calls == {"InventoryOutput": 1, "EnrichmentBatch": 1}


def test_blueprint_prose_is_rendered_from_jinja_not_python():
    kinds = [*SINGLE_WORD, *DEFINITION, *USAGE, "inflection", "particle"]
    instruction, data = render_prompt(
        "questions/prompts/generate_synthetic_answers.jinja",
        context={"target": "bank", "definition": "A place for money", "answer_types": kinds},
        question_type="DEFINITION_TO_WORD",
    )
    assert prompt_context(data, "synthetic_context")["answer_types"] == kinds
    assert all(f"- {kind}: " in instruction for kind in kinds)
    assert all(isinstance(kind, str) and " " not in kind for kind in kinds)


class GeneratorLLM:
    def __init__(self):
        self.calls = Counter()
        self.questions = QuestionLLM()
        self.active = 0
        self.peak = 0
        self.question_barrier = None
        self.themed_active = 0
        self.fail_synthesis = False
        self.fail_grading = False
        self.grading_calls = []

    async def complete(self, instruction, data, schema, *, with_usage=False):
        self.calls[schema.__name__] += 1
        self.active += 1
        self.peak = max(self.peak, self.active)
        if schema.__name__ == "ThemedWord":
            self.themed_active += 1
            assert self.themed_active == 1  # Same Word's Theme generation is serialized.
        try:
            if self.question_barrier is not None and schema.__name__ in {
                "AnchoredQuestionBatch",
                "QuestionBatch",
            }:
                if self.calls["AnchoredQuestionBatch"] + self.calls["QuestionBatch"] >= 3:
                    self.question_barrier.set()
                await asyncio.wait_for(self.question_barrier.wait(), timeout=5)
            await asyncio.sleep(0.002)
            if schema.__name__ in {"InventoryOutput", "EnrichmentBatch"}:
                tag = "sense_request" if schema.__name__ == "EnrichmentBatch" else "word_request"
                request = prompt_context(data, tag)
                output = payload()
                output["senses"][0]["examples"] = [
                    f"The [bank] opened at {hour}."
                    for hour in range(request.get("examples_per_sense", 1))
                ]
                value = stage_payload(output, data, schema)
            elif schema.__name__ == "ThemeParts":
                value = schema(voice="Adventure", diction="worldbuilding")
            elif schema.__name__ == "ThemedWord":
                count = prompt_context(data, "generation_parameters")["examples_per_sense"]
                value = schema(
                    senses=[
                        {
                            "definition": "A place for money in this world",
                            "examples": ["The [bank] holds coin."] * count,
                        }
                    ]
                )
            elif schema.__name__ in {"QuestionBatch", "AnchoredQuestionBatch"}:
                value = await self.questions.complete(instruction, data, schema)
            elif schema.__name__ in {"AnswerBatch", "AnswerList"}:
                if self.fail_synthesis:
                    raise ValueError("synthetic failure")
                context = prompt_context(data, "synthetic_context")
                assert set(context) in (
                    {"target", "definition", "answer_types"},
                    {"target", "context", "answer_types"},
                    {"target", "sentence", "answer_types"},
                )
                # Fixed strings deliberately ignore requested types: grading must
                # label the actual answer, not assume the allocation was followed.
                if "to a WORD_TO_DEFINITION question." in instruction:
                    entries = ["A place for money", "An unrelated remark", "Money keeper"]
                elif "to a WORD_TO_USAGE question." in instruction:
                    entries = ["The bank opens.", "I kept my money safe.", "The bank closed."]
                else:
                    entries = ["bank", "bnak", "mountain"]
                value = schema.model_validate(
                    {"answers": entries} if schema is AnswerBatch else entries
                )
            else:
                raise AssertionError(schema.__name__)
            usage = [TokenUsage("fixture", 100, 0, 0, 10)]
            return (value, usage) if with_usage else value
        finally:
            self.active -= 1
            if schema.__name__ == "ThemedWord":
                self.themed_active -= 1


class GradingClient:
    def __init__(self, llm):
        self.llm = llm

    async def system_one(self, *, state, questions):
        self.llm.grading_calls.append((state, questions))
        if self.llm.fail_grading:
            raise ValueError("grading failure")
        await asyncio.sleep(0.002)
        assert set(state) <= {"question", "word", "meaning", "answer"}
        answer = state["answer"]
        response = SimpleNamespace(
            nouls={},
            choices={},
            model="fixture",
            usage=SimpleNamespace(input_tokens=50, output_tokens=5),
        )
        for name, rule in questions.items():
            if name in {"task_fit", "spelling_error", "used", "construction"}:
                value = {
                    "task_fit": answer in {"bank", "bnak"},
                    "spelling_error": answer == "bnak",
                    "used": "bank" in answer,
                    "construction": True,
                }[name]
                response.nouls[name] = SimpleNamespace(noul=float(value))
            else:
                if name in {"matched_sense", "defined_meaning"}:
                    value = (
                        "no_candidate"
                        if answer == "An unrelated remark"
                        else next(key for key in rule.criteria if key != "no_candidate")
                    )
                else:
                    value = {
                        "accuracy": "mixed",
                        "coverage": "partial",
                        "meaning": "correct",
                        "form": "correct",
                        "collocation": "natural",
                        "appropriacy": "appropriate",
                    }[name]
                response.choices[name] = SimpleNamespace(choice=value, confidence=1.0)
        return response


@pytest.fixture
async def lexicon(tmp_path, source, monkeypatch):
    async def empty_wordnet(_citation):
        return []

    monkeypatch.setattr("lexi_ai.words.generate.lookup", empty_wordnet)
    llm = GeneratorLLM()
    llm_config = LLMConfig(model="fixture", api_key="fake", base_url="https://fixture.test/v1")

    def grading_client(model):
        assert model.llm_config == llm_config
        assert model.fallback_model is None  # Never inherit a different fallback model.
        return GradingClient(llm)

    monkeypatch.setattr(DecisionModel, "_fallback", grading_client)
    lexicon = Lexicon(
        f"sqlite+aiosqlite:///{tmp_path / 'generated.db'}",
        str(source),
        llm=llm,
        llm_config=llm_config,
        decision_fallback_model="must-not-use",
    )
    await lexicon.db.create_schema(Base.metadata)
    await lexicon.start()
    yield lexicon, llm
    await lexicon.close()


async def test_two_stages_resume_three_answers_and_benchmark_export(tmp_path, lexicon):
    library, llm = lexicon
    llm.question_barrier = asyncio.Event()
    output = tmp_path / "dataset"
    settings = config()
    sources = await run_generation(
        library,
        settings,
        output=output,
        stage="sources",
        workers=3,
        progress=False,
    )
    assert sources == {"planned_questions": 10, "planned_answers": 30}
    assert not llm.calls["AnswerBatch"] and not llm.calls["QuestionBatch"]
    assert llm.calls["InventoryOutput"] == llm.calls["EnrichmentBatch"] == 1
    assert llm.calls["ThemeParts"] == 3
    assert llm.calls["ThemedWord"] == 3
    assert llm.peak >= 2
    summary = await run_generation(
        library,
        settings,
        output=output,
        stage="questions",
        workers=3,
        progress=False,
        resume=True,
    )
    assert summary["questions"] == llm.calls["AnswerBatch"] == 10
    assert summary["answers"] == 30
    cases = load_dataset(output / "cases")
    assert len(cases) == sum(summary["cases_by_task"].values()) == 44
    assert all(case["label_status"] == "proposed" for case in cases)
    assert all(case["label_source"] == "llm_grading" for case in cases)
    assert all("blueprint" not in case and "rationale" not in case for case in cases)
    assert sum(case["task"] == "grade_single_word_2" for case in cases) == 6
    assert sum(case["task"] == "grade_word_to_usage_2" for case in cases) == 4
    saved = read_records(output / "answers.jsonl")
    assert all(len(record["answers"]) == 3 for record in saved.values())
    assert all(isinstance(text, str) for record in saved.values() for text in record["answers"])
    assert all(record["usage"][0]["model_id"] == "fixture" for record in saved.values())
    # Labels can disagree with the requested type; they come solely from step 2.
    assert all(
        case["expected"] == {"accuracy": "mixed", "coverage": "partial"}
        for case in cases
        if case["task"] == "grade_word_to_definition_2"
    )
    assert len(llm.grading_calls) == len(cases)
    calls = llm.calls.copy()
    grading_count = len(llm.grading_calls)
    await run_generation(library, settings, output=output, resume=True, progress=False)
    assert llm.calls == calls  # No duplicated source/Question/synthesis requests on resume.
    assert len(llm.grading_calls) == grading_count
    with pytest.raises(FileExistsError):
        await run_generation(library, settings, output=output, progress=False)
    with pytest.raises(ValueError, match="original generation config"):
        await run_generation(
            library,
            settings.model_copy(update={"questions_per_type": 3}),
            output=output,
            resume=True,
            progress=False,
        )


async def test_failed_synthesis_keeps_questions_for_resume(tmp_path, lexicon):
    library, llm = lexicon
    output = tmp_path / "failed"
    settings = config()
    await run_generation(library, settings, output=output, stage="sources", progress=False)
    llm.fail_synthesis = True
    with pytest.raises(ValueError, match="synthetic failure"):
        await run_generation(
            library,
            settings,
            output=output,
            stage="questions",
            workers=1,
            resume=True,
            progress=False,
        )
    saved_question = read_records(output / "questions.jsonl")["q001"]["data"]["id"]
    llm.fail_synthesis = False
    await run_generation(
        library,
        settings,
        output=output,
        stage="questions",
        resume=True,
        progress=False,
    )
    assert read_records(output / "questions.jsonl")["q001"]["data"]["id"] == saved_question
    assert llm.calls["QuestionBatch"] + llm.calls["AnchoredQuestionBatch"] == 10


def test_usage_anchor_is_saved_question_not_current_definition():
    word = Word(1, "bank", "WORD", "DONE")
    sense = Sense(1, 1, "NOUN", "CORE", definition=Definition(1, "Changed definition"))
    word = replace(word, senses=[sense])
    question = Question(
        1,
        1,
        None,
        "WORD_TO_USAGE",
        "[bank] — Saved meaning",
        Option("correct", "The bank opens.", "Fits"),
        [],
    )
    contexts = stage_contexts(question, word, sense)
    assert contexts["grade_word_to_usage_2"]["meaning"] == "Saved meaning"


@pytest.mark.parametrize("structured", [True, False])
@pytest.mark.parametrize("texts", [["bank", "bnak"], ["bank", " ", "river"], [1, 2, 3]])
async def test_synthesis_rejects_wrong_count_blank_and_nonstring_outputs(structured, texts):
    class BadLLM:
        async def complete(self, _instruction, _data, schema, **_kwargs):
            return {"answers": texts} if structured else texts, []

    with pytest.raises(ValueError):
        await synthesize(
            BadLLM(),
            {
                "target": "bank",
                "definition": "Money keeper",
                "answer_types": ["canonical", "typo", "wrong_meaning"],
            },
            question_type="DEFINITION_TO_WORD",
            structured_outputs=structured,
        )


@pytest.mark.parametrize("structured", [True, False])
async def test_synthesis_returns_only_k_strings_and_renders_only_selected_types(structured):
    context = {
        "target": "bank",
        "definition": "Money keeper",
        "answer_types": ["canonical", "typo"],
    }

    class LLM:
        async def complete(self, instruction, data, schema, **_kwargs):
            assert prompt_context(data, "synthetic_context") == context
            assert "- canonical:" in instruction and "- typo:" in instruction
            assert "- injection:" not in instruction
            assert schema is (AnswerBatch if structured else AnswerList)
            assert "exactly 2 nonblank strings" in instruction
            return schema.model_validate(
                {"answers": ["bank", "bnak"]} if structured else ["bank", "bnak"]
            ), []

    texts, usage = await synthesize(
        LLM(),
        context,
        question_type="DEFINITION_TO_WORD",
        structured_outputs=structured,
    )
    assert texts == ["bank", "bnak"] and usage == []


def test_native_schema_has_only_string_array_no_generated_labels_or_metadata():
    schema = AnswerBatch.model_json_schema()
    assert list(schema["properties"]) == ["answers"]
    assert schema["properties"]["answers"]["items"]["type"] == "string"
    with pytest.raises(ValidationError):
        AnswerBatch.model_validate({"answers": ["bank"], "labels": {"task_fit": True}})


async def test_text_transport_parses_json_array_without_object_wrapper():
    async def create(**kwargs):
        assert "response_format" not in kwargs
        assert '"type":"array"' in kwargs["messages"][0]["content"]
        return SimpleNamespace(
            model="fixture",
            usage=None,
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(
                        content='["bank", "bnak"]',
                        refusal=None,
                    ),
                )
            ],
        )

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    llm = OpenAIStructuredLLM(LLMConfig(structured_outputs=False), client)
    texts, usage = await synthesize(
        llm,
        {
            "target": "bank",
            "definition": "A financial institution",
            "answer_types": ["canonical", "typo"],
        },
        question_type="DEFINITION_TO_WORD",
        structured_outputs=False,
    )
    assert texts == ["bank", "bnak"]
    assert usage[0].input_tokens is None


@pytest.mark.parametrize(
    "kind,content,correct,expected",
    [
        ("DEFINITION_TO_WORD", "A money keeper", "bank", {"definition": "A money keeper"}),
        ("CONTEXT_TO_WORD", "Deposit savings here", "bank", {"context": "Deposit savings here"}),
        (
            "CLOZE_TO_WORD",
            "Several _ offer savings",
            "banks",
            {"target": "banks", "sentence": "Several _ offer savings"},
        ),
        (
            "WORD_TO_DEFINITION",
            "[bank]",
            "A money keeper",
            {"definition": "A money keeper"},
        ),
        (
            "WORD_TO_USAGE",
            "[bank] — Saved meaning",
            "The bank opens.",
            {"definition": "Saved meaning"},
        ),
    ],
)
def test_minimal_synthesis_input_per_question_type(kind, content, correct, expected):
    question = Question(1, 1, None, kind, content, Option("correct", correct, "Not sent"), [])
    word = Word(1, "bank", "WORD", "DONE")
    assert synthesis_context(question, word, ["canonical"]) == {
        "target": "bank",
        "answer_types": ["canonical"],
        **expected,
    }


async def test_grading_failure_preserves_answers_and_resume_does_not_regenerate(tmp_path, lexicon):
    library, llm = lexicon
    output = tmp_path / "grading-failed"
    settings = config()
    await run_generation(library, settings, output=output, stage="sources", progress=False)
    llm.fail_grading = True
    with pytest.raises(ValueError, match="grading failure"):
        await run_generation(
            library,
            settings,
            output=output,
            stage="questions",
            workers=1,
            resume=True,
            progress=False,
        )
    assert read_records(output / "answers.jsonl")["q001"]["answers"] == ["bank", "bnak", "mountain"]
    assert not read_records(output / "labels.jsonl")
    llm.fail_grading = False
    await run_generation(library, settings, output=output, resume=True, progress=False)
    assert llm.calls["AnswerBatch"] == 10
    assert len(read_records(output / "labels.jsonl")) == 10


async def test_single_word_grading_uses_search_candidate_full_inventory(monkeypatch):
    owner_sense = Sense(1, 1, "NOUN", "CORE", definition=Definition(1, "A financial institution"))
    word = Word(1, "bank", "WORD", "DONE", senses=[owner_sense])
    question = Question(
        1,
        1,
        None,
        "DEFINITION_TO_WORD",
        "A financial institution",
        Option("correct", "bank", "Not sent"),
        [],
    )
    matched = {
        "lemma": "credit union",
        "senses": [
            {"id": 41, "pos": "NOUN", "definition": "A member-owned financial institution"},
            {"id": 42, "pos": "NOUN", "definition": "An extended meaning"},
        ],
    }
    database = object()

    async def search(db, text, *, limit):
        assert db is database and text == "credit union" and limit == 1
        return SimpleNamespace(items=[SimpleNamespace(word_id=4)])

    async def meanings(db, *, word_id):
        assert db is database and word_id == 4
        return matched

    class Client(GradingClient):
        async def system_one(self, *, state, questions):
            response = await super().system_one(state=state, questions=questions)
            if "task_fit" in questions:
                response.nouls["task_fit"].noul = 1.0
            return response

    monkeypatch.setattr("benchmarks.synthetic.search", search)
    monkeypatch.setattr("benchmarks.synthetic.meaning_inventory", meanings)
    llm = GeneratorLLM()
    model = DecisionModel(DecisionConfig(0.8), fallback=Client(llm))
    cases, _ = await grade_answers(
        SimpleNamespace(db=database),
        model,
        {"id": "q001", "theme": None},
        question,
        word,
        owner_sense,
        ["credit union"],
    )
    assert len(cases) == 2
    assert cases[1]["input"]["matched_word"] == matched
    assert cases[1]["expected"] == {"matched_sense": "sense_41"}
    assert set(llm.grading_calls[1][1]["matched_sense"].criteria) == {
        "no_candidate",
        "sense_41",
        "sense_42",
    }


async def test_definition_grading_selects_another_sense_from_full_inventory():
    senses = [
        Sense(1, 1, "NOUN", "CORE", definition=Definition(1, "A financial institution")),
        Sense(2, 1, "NOUN", "RARE", definition=Definition(2, "Land beside a river")),
    ]
    word = Word(1, "bank", "WORD", "DONE", senses=senses)
    question = Question(
        1,
        1,
        None,
        "WORD_TO_DEFINITION",
        "[bank]",
        Option("correct", "A financial institution", "Not sent"),
        [],
    )

    class Client(GradingClient):
        async def system_one(self, *, state, questions):
            response = await super().system_one(state=state, questions=questions)
            if "defined_meaning" in questions:
                assert set(questions["defined_meaning"].criteria) == {
                    "no_candidate",
                    "sense_1",
                    "sense_2",
                }
                response.choices["defined_meaning"].choice = "sense_2"
            else:
                assert state["meaning"] == "Land beside a river"
            return response

    model = DecisionModel(DecisionConfig(0.8), fallback=Client(GeneratorLLM()))
    cases, _ = await grade_answers(
        None,
        model,
        {"id": "q004", "theme": None},
        question,
        word,
        senses[0],
        ["River's edge"],
    )
    assert cases[0]["expected"] == {"defined_meaning": "sense_2"}
    assert cases[1]["input"]["selectedSense"]["id"] == 2


async def test_unknown_grading_sense_rejected_before_export():
    sense = Sense(1, 1, "NOUN", "CORE", definition=Definition(1, "A financial institution"))
    word = Word(1, "bank", "WORD", "DONE", senses=[sense])
    question = Question(
        1,
        1,
        None,
        "WORD_TO_DEFINITION",
        "[bank]",
        Option("correct", "A financial institution", "Not sent"),
        [],
    )

    class Client(GradingClient):
        async def system_one(self, *, state, questions):
            response = await super().system_one(state=state, questions=questions)
            response.choices["defined_meaning"].choice = "sense_999"
            return response

    model = DecisionModel(DecisionConfig(0.8), fallback=Client(GeneratorLLM()))
    with pytest.raises(InvalidOutputError, match="invalid decision option"):
        await grade_answers(
            None,
            model,
            {"id": "q004", "theme": None},
            question,
            word,
            sense,
            ["Money keeper"],
        )
