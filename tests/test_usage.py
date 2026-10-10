"""Opt-in token accounting across transports, staged tasks and concurrent requests."""

import asyncio
from types import SimpleNamespace

import pytest
from system_one_adapter import AsyncSystemOneAdapterClient
from test_decision import DecisionTransport, Provider
from test_end_to_end import LLM, Decision
from test_llm import FakeClient, Reply
from typesafe_sdk import Choice, Noul

from lexi_ai import DecisionConfig, Lexicon, TokenUsage
from lexi_ai.errors import InvalidOutputError
from lexi_ai.inference.config import LLMConfig
from lexi_ai.inference.decision import DecisionModel
from lexi_ai.inference.llm import OpenAIStructuredLLM
from lexi_ai.inference.usage import decision_usage, merge_usage, openai_usage
from lexi_ai.schema import Base


def sdk_reply(parsed=None, *, model="actual-llm", usage=None, refusal=None):
    return SimpleNamespace(
        model=model,
        usage=usage,
        choices=[SimpleNamespace(message=SimpleNamespace(parsed=parsed, refusal=refusal))],
    )


def chat_usage(input_tokens=100, cached=25, output_tokens=10):
    return SimpleNamespace(
        prompt_tokens=input_tokens,
        prompt_tokens_details=SimpleNamespace(cached_tokens=cached),
        completion_tokens=output_tokens,
    )


@pytest.mark.parametrize(
    "input_tokens,output,reasoning,total,expected",
    [
        (100, 10, 4, 110, 10),  # Standard OpenAI: reasoning already included.
        (100, 10, 4, 114, 14),  # Compatible provider: reasoning is separate.
        (100, 10, 4, 113, 10),  # Inconsistent metadata cannot justify adding counts.
        (100, 10, 4, None, 10),
        (100, 10, "4", 114, 10),
        (100, None, 4, 114, None),
        (None, 10, 4, 114, 10),
        (100, 10, 0, 110, 10),
    ],
)
def test_reasoning_output_counts_only_add_separate_tokens_when_total_confirms(
    input_tokens,
    output,
    reasoning,
    total,
    expected,
):
    raw = {
        "model": "actual",
        "usage": {
            "prompt_tokens": input_tokens,
            "completion_tokens": output,
            "completion_tokens_details": {"reasoning_tokens": reasoning},
            "total_tokens": total,
        },
    }
    assert openai_usage(raw).output_tokens == expected
    # The official grading adapter's raw attempt trace uses the same collector.
    assert decision_usage({"debug": {"llm_attempts": [{"llm_response": raw}]}})[0] == (
        openai_usage(raw)
    )


@pytest.mark.parametrize("counts", [None, chat_usage()])
async def test_llm_tuple_preserves_result_instance_and_actual_model(counts):
    parsed = Reply(answer="yes")
    client = FakeClient()

    async def parse(**kwargs):
        assert kwargs["model"] == "requested-alias"
        return sdk_reply(parsed, usage=counts)

    client.completions.parse = parse
    llm = OpenAIStructuredLLM(LLMConfig(model="requested-alias"), client)
    try:
        value, usage = await llm.complete("system", "data", Reply, with_usage=True)
        assert value is parsed
        assert usage == [
            TokenUsage(
                "actual-llm",
                100 if counts else None,
                25 if counts else None,
                None,
                10 if counts else None,
            )
        ]
        assert await llm.complete("system", "data", Reply) is parsed
    finally:
        await llm.close()


async def test_llm_refusal_keeps_usage_on_original_error():
    client = FakeClient()

    async def parse(**kwargs):
        return sdk_reply(usage=chat_usage(), refusal="refused")

    client.completions.parse = parse
    llm = OpenAIStructuredLLM(LLMConfig(), client)
    try:
        with pytest.raises(InvalidOutputError) as caught:
            await llm.complete("system", "data", Reply, with_usage=True)
        assert caught.value.usage == [TokenUsage("actual-llm", 100, 25, None, 10)]
    finally:
        await llm.close()


async def test_llm_sdk_parse_error_keeps_completion_usage():
    error = ValueError("malformed provider output")
    error.completion = sdk_reply(usage=chat_usage())
    client = FakeClient()

    async def parse(**kwargs):
        raise error

    client.completions.parse = parse
    llm = OpenAIStructuredLLM(LLMConfig(), client)
    try:
        with pytest.raises(ValueError) as caught:
            await llm.complete("system", "data", Reply, with_usage=True)
        assert caught.value is error
        assert error.usage == [TokenUsage("actual-llm", 100, 25, None, 10)]
    finally:
        await llm.close()


async def test_llm_transport_error_does_not_invent_zero_counts():
    client = FakeClient()

    async def parse(**kwargs):
        raise ConnectionError("unavailable")

    client.completions.parse = parse
    llm = OpenAIStructuredLLM(LLMConfig(), client)
    try:
        with pytest.raises(ConnectionError) as caught:
            await llm.complete("system", "data", Reply, with_usage=True)
        assert caught.value.usage == [TokenUsage(None, None, None, None, None)]
    finally:
        await llm.close()


class ReportedTransport(DecisionTransport):
    async def system_one(self, **kwargs):
        response = await super().system_one(**kwargs)
        response.model = "actual-decision"
        response.usage = SimpleNamespace(input_tokens=20, output_tokens=2)
        return response


async def test_decision_counts_primary_and_official_fallback_once():
    primary = ReportedTransport(confidence=0.1)
    provider = Provider({"match": "1"})
    fallback = AsyncSystemOneAdapterClient(
        structured_outputs=True, llm_answer_mode="discrete", model=provider
    )
    model = DecisionModel(DecisionConfig(0.7), primary, fallback)
    try:
        response, usage = await model.decide(
            {},
            {"match": Choice(instructions="Which?", criteria={"0": "none", "1": "one"})},
            with_usage=True,
        )
        assert response.choices["match"].choice == "1"
        assert usage == [
            TokenUsage("actual-decision", 20, None, None, 2),
            TokenUsage(None, 1, None, None, 1),
        ]
        assert len(primary.calls) == len(provider.calls) == 1
    finally:
        await model.close()
        await fallback.aclose()


async def test_low_confidence_does_not_discard_primary_usage_on_fallback_error():
    class FailingFallback:
        async def system_one(self, **kwargs):
            raise ConnectionError("failed fallback")

    model = DecisionModel(DecisionConfig(0.7), ReportedTransport(confidence=0.1), FailingFallback())
    with pytest.raises(ConnectionError) as caught:
        await model.decide(
            {},
            {"match": Choice(instructions="Which?", criteria={"0": "none", "1": "one"})},
            with_usage=True,
        )
    assert caught.value.usage == [
        TokenUsage("actual-decision", 20, None, None, 2),
        TokenUsage(None, None, None, None, None),
    ]
    await model.close()


async def test_invalid_decision_response_keeps_reported_usage():
    model = DecisionModel(DecisionConfig(0.7), ReportedTransport(probability=float("nan")))
    with pytest.raises(InvalidOutputError) as caught:
        await model.decide({}, {"used": Noul(instructions="Used?")}, with_usage=True)
    assert caught.value.usage == [TokenUsage("actual-decision", 20, None, None, 2)]
    await model.close()


def test_adapter_raw_attempts_preserve_actual_models_cache_and_failed_attempts():
    response = SimpleNamespace(
        model="alias",
        usage=SimpleNamespace(input_tokens_total=999, output_tokens_total=999),
        debug={
            "llm_attempts": [
                {
                    "llm_response": {
                        "model": "actual",
                        "usage": {
                            "prompt_tokens": 10,
                            "completion_tokens": 2,
                            "prompt_tokens_details": {"cached_tokens": 4},
                        },
                    }
                },
                {
                    "llm_response": {
                        "model": "actual",
                        "usage": {
                            "input_tokens": 20,
                            "output_tokens": 3,
                            "input_tokens_details": {"cached_tokens": 0},
                        },
                    }
                },
                {"llm_response": None},
            ]
        },
    )
    assert merge_usage(decision_usage(response)) == [
        TokenUsage("actual", 30, 4, None, 5),
        TokenUsage(None, None, None, None, None),
    ]


def test_adapter_does_not_replace_an_unreported_actual_model_with_its_alias():
    response = SimpleNamespace(
        model="requested-alias",
        debug={
            "llm_attempts": [
                {"llm_response": {"model": "actual", "usage": None}},
                {"llm_response": {"text": "reply", "input_tokens": 5, "output_tokens": 1}},
            ]
        },
    )
    assert decision_usage(response) == [
        TokenUsage("actual", None, None, None, None),
        TokenUsage(None, 5, None, None, 1),
    ]


@pytest.mark.parametrize("total", [None, 50])
def test_adapter_cumulative_usage_does_not_fall_back_from_unknown_total(total):
    response = SimpleNamespace(
        model="actual",
        usage=SimpleNamespace(
            input_tokens=10,
            output_tokens=2,
            input_tokens_total=total,
            output_tokens_total=8,
        ),
    )
    assert decision_usage(response) == [TokenUsage("actual", total, None, None, 8)]


def test_unknown_counts_propagate_independently_and_results_sort_by_model():
    records = [
        TokenUsage("z", 5, 0, 0, 1),
        TokenUsage("a", 10, 4, None, 2),
        TokenUsage("a", None, 0, 0, 3),
    ]
    assert merge_usage(records) == [
        TokenUsage("a", None, 4, None, 5),
        TokenUsage("z", 5, 0, 0, 1),
    ]


@pytest.mark.parametrize("value", [-1, True, "4", 1.5])
def test_malformed_counts_are_unknown_not_billable_integers(value):
    usage = openai_usage(sdk_reply(usage=chat_usage(value, value, value)))
    assert usage == TokenUsage("actual-llm", None, None, None, None)


@pytest.mark.parametrize("model", [None, "", "  ", 4, ["model"]])
def test_unreported_or_invalid_model_id_is_not_replaced_by_an_alias(model):
    assert openai_usage(sdk_reply(model=model, usage=chat_usage())).model_id is None


def test_explicit_zero_cache_reads_and_reported_cache_writes_are_preserved():
    assert openai_usage(
        {
            "model": "actual",
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 10,
                "prompt_tokens_details": {"cached_tokens": 0},
                "cache_write_tokens": 8,
            },
        }
    ) == TokenUsage("actual", 100, 0, 8, 10)


LLM_USAGE = TokenUsage("actual-llm", 10, 3, None, 2)
DECISION_USAGE = TokenUsage("actual-decision", 20, None, None, 4)


class MeteredLLM(LLM):
    async def complete(self, *args, with_usage=False, **kwargs):
        value = await super().complete(*args, **kwargs)
        return (value, [LLM_USAGE]) if with_usage else value


class MeteredDecision(Decision):
    async def decide(self, *args, with_usage=False, **kwargs):
        value = await super().decide(*args, **kwargs)
        return (value, [DECISION_USAGE]) if with_usage else value


@pytest.fixture
async def lexicon(tmp_path, source):
    instance = Lexicon(
        f"sqlite+aiosqlite:///{tmp_path / 'usage.db'}",
        str(source),
        decision_config=DecisionConfig(0.8),
        llm=MeteredLLM(),
        decision_model=MeteredDecision(),
    )
    try:
        await instance.db.create_schema(Base.metadata)
        await instance.start()
        yield instance
    finally:
        await instance.close()


async def generate_bank(lexicon):
    handle = (await lexicon.search("bank", include_reference=True)).items[0].reference_id
    word = await lexicon.generate_word("bank", reference_id=handle, example_count=1)
    return handle, word


async def test_public_generation_theme_and_question_usage_and_cached_reads(lexicon):
    handle = (await lexicon.search("bank", include_reference=True)).items[0].reference_id
    _, usage = await lexicon.create_theme("pirate", "Pirate", "nautical", with_usage=True)
    assert usage == [LLM_USAGE]
    word, usage = await lexicon.generate_word(
        "bank", reference_id=handle, theme="pirate", example_count=1, with_usage=True
    )
    assert usage == [TokenUsage("actual-llm", 30, 9, None, 6)]
    stored, usage = await lexicon.generate_word(
        "bank", reference_id=handle, theme="pirate", with_usage=True
    )
    assert stored == word and usage == []
    questions, usage = await lexicon.generate_questions(
        word.senses[0].id,
        "DEFINITION_TO_WORD",
        2,
        distractor_count=3,
        with_usage=True,
    )
    assert len(questions) == 2 and usage == [LLM_USAGE]
    question = questions[0]
    for fmt, answer in [("SINGLE_CHOICE", question.correct.id), ("SINGLE_WORD", "BANK")]:
        grade, usage = await lexicon.grade_answer(question.id, fmt, answer, with_usage=True)
        assert grade.task_fit and usage == []


async def test_multistage_grading_totals_and_provider_free_empty_relations(lexicon):
    assert await lexicon.resolve_relations(with_usage=True) == ([], [])
    _, word = await generate_bank(lexicon)
    question = (
        await lexicon.generate_questions(
            word.senses[0].id,
            "WORD_TO_USAGE",
            1,
            distractor_count=3,
        )
    )[0]
    grade, usage = await lexicon.grade_answer(
        question.id,
        "SHORT_ANSWER",
        "The bank opens early.",
        with_usage=True,
    )
    assert grade.used
    assert usage == [TokenUsage("actual-decision", 40, None, None, 8)]
    assert lexicon.decision_model.calls == 2


@pytest.mark.parametrize(
    "kind,fmt,answer,stages",
    [
        ("DEFINITION_TO_WORD", "SINGLE_WORD", "lender", 2),
        ("WORD_TO_DEFINITION", "SHORT_ANSWER", "A financial institution", 2),
        ("WORD_TO_USAGE", "SHORT_ANSWER", "The bank opens early.", 2),
        ("WORD_TO_USAGE", "SHORT_ANSWER", "We went home.", 1),
    ],
)
async def test_all_grading_stages_and_gates_collect_only_their_own_calls(
    lexicon, kind, fmt, answer, stages
):
    from lexi_ai import schema as row

    _, word = await generate_bank(lexicon)
    async with lexicon.db.transaction() as session:
        session.add(row.WordAlias(word_id=word.id, content="lender", match_key="lender"))
    # Direct writes explicitly notify the index after committing.
    await lexicon._search.update(word.id)
    question = (await lexicon.generate_questions(word.senses[0].id, kind, 1, distractor_count=3))[0]

    class GradingDecision:
        calls = 0

        async def decide(self, state, questions, *, with_usage=False, **kwargs):
            self.calls += 1
            choices, nouls = {}, {}
            for name, q in questions.items():
                if isinstance(q, Noul):
                    value = 0.1 if name == "spelling_error" or stages == 1 else 0.9
                    nouls[name] = SimpleNamespace(noul=value)
                else:
                    key = next(key for key in q.criteria if key != "no_candidate")
                    choices[name] = SimpleNamespace(choice=key)
            value = SimpleNamespace(choices=choices, nouls=nouls)
            return (value, [DECISION_USAGE]) if with_usage else value

    lexicon.decision_model = GradingDecision()
    grade, usage = await lexicon.grade_answer(question.id, fmt, answer, with_usage=True)
    assert usage == [TokenUsage("actual-decision", 20 * stages, None, None, 4 * stages)]
    assert lexicon.decision_model.calls == stages
    if stages == 1:
        assert not grade.used and grade.meaning is None
    elif fmt == "SINGLE_WORD" or kind == "WORD_TO_DEFINITION":
        assert grade.sense_id == word.senses[0].id


async def test_translation_cache_and_original_default_return(lexicon):
    text, usage = await lexicon.translate_text("bank", "vi", with_usage=True)
    assert text == "ngân hàng" and usage == [LLM_USAGE]
    assert await lexicon.translate_text("[bank]", "vi", with_usage=True) == (
        text,
        [],
    )
    assert await lexicon.translate_text("bank", "vi") == text


async def test_domain_validation_error_keeps_usage_after_provider_succeeded(lexicon):
    class InvalidLLM:
        async def complete(self, *args, with_usage=False, **kwargs):
            return ({"invalid": "publication output"}, [LLM_USAGE])

    lexicon.llm = InvalidLLM()
    with pytest.raises(InvalidOutputError) as caught:
        await lexicon.translate_text("bank", "vi", with_usage=True)
    assert caught.value.usage == [LLM_USAGE]


async def test_concurrent_api_calls_do_not_mix_usage(lexicon):
    entered = 0
    barrier = asyncio.Event()

    class ConcurrentLLM:
        async def complete(self, instruction, data, schema, *, with_usage=False):
            nonlocal entered
            index = entered
            entered += 1
            if entered == 2:
                barrier.set()
            await asyncio.wait_for(barrier.wait(), timeout=5)
            usage = TokenUsage(f"actual-{index}", 10 + index, 0, None, 2)
            value = schema(content=f"translation{index}")
            return (value, [usage]) if with_usage else value

    lexicon.llm = ConcurrentLLM()
    results = await asyncio.gather(
        lexicon.translate_text("bank", "vi", with_usage=True),
        lexicon.translate_text("river", "vi", with_usage=True),
    )
    assert sorted((text, usage[0].model_id, usage[0].input_tokens) for text, usage in results) == [
        ("translation0", "actual-0", 10),
        ("translation1", "actual-1", 11),
    ]


async def test_parallel_relation_error_preserves_usage_without_canceling_success(lexicon):
    from sqlalchemy import insert, select

    from lexi_ai import schema as row

    async with lexicon.db.transaction() as session:
        await session.execute(
            insert(row.Word),
            [
                dict(
                    id=i,
                    lemma=f"word{i}",
                    match_key=f"word{i}",
                    entry_type="WORD",
                    generation_state="DONE",
                )
                for i in range(1, 4)
            ],
        )
        await session.execute(
            insert(row.Sense), [dict(id=i, word_id=i, pos="NOUN", tier="CORE") for i in range(1, 4)]
        )
        await session.execute(
            insert(row.Definition), [dict(sense_id=i, content=f"meaning{i}") for i in range(1, 4)]
        )
        await session.execute(
            insert(row.SenseRelation),
            [dict(id=i, from_sense_id=1, to_word_id=i + 1, rel_type="SYNONYM") for i in (1, 2)],
        )

    class PartialDecision:
        async def decide(self, state, questions, *, with_usage=False, **kwargs):
            await asyncio.sleep(0)
            if state["target"]["word"] == "word2":
                error = InvalidOutputError("invalid verdict")
                error.usage = [DECISION_USAGE]
                raise error
            return (
                SimpleNamespace(choices={"matched_sense": SimpleNamespace(choice="candidate_1")}),
                [DECISION_USAGE],
            )

    lexicon.decision_model = PartialDecision()
    results, usage = await lexicon.resolve_relations(with_usage=True)
    assert [result.state for result in results] == ["ERROR", "RESOLVED"]
    assert usage == [TokenUsage("actual-decision", 40, None, None, 8)]
    async with lexicon.db.read() as connection:
        attempted = await connection.scalar(
            select(row.SenseRelation.resolve_attempted_at).where(row.SenseRelation.id == 1)
        )
        assert attempted is None


async def test_nonboolean_opt_in_is_rejected_before_io(lexicon):
    with pytest.raises(TypeError, match="boolean"):
        await lexicon.translate_text("bank", "vi", with_usage="yes")
    assert lexicon.llm.calls == []
