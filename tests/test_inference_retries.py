"""Single ceilings across transport, malformed output and usage in both LLM paths."""

import asyncio
import json

import httpx
import pytest
from openai import AsyncOpenAI, AuthenticationError, InternalServerError
from pydantic import BaseModel
from typesafe_sdk import Choice, TypeSafeAPIResponseValidationError

from lexi_ai import DecisionConfig, DecisionMode, LLMConfig
from lexi_ai.errors import InvalidOutputError
from lexi_ai.inference.decision import DecisionModel
from lexi_ai.inference.llm import OpenAIStructuredLLM
from lexi_ai.inference.retry import retry


class Reply(BaseModel):
    answer: str


@pytest.fixture(autouse=True)
def no_retry_wait(monkeypatch):
    async def wait(_delay):
        pass

    monkeypatch.setattr("lexi_ai.inference.retry.asyncio.sleep", wait)


def completion(content, model="actual"):
    return {
        "id": "test",
        "object": "chat.completion",
        "created": 1,
        "model": model,
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": content,
                },
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 2,
            "total_tokens": 15,
            "completion_tokens_details": {"reasoning_tokens": 3},
        },
    }


@pytest.mark.parametrize("structured", [True, False])
async def test_generation_one_budget_for_http_and_parse_with_all_attempt_usage(structured):
    calls = []

    def respond(request):
        calls.append(json.loads(request.content))
        if len(calls) == 1:
            return httpx.Response(503)
        content = "not JSON" if len(calls) == 2 else '{"answer":"yes"}'
        return httpx.Response(200, json=completion(content))

    async with AsyncOpenAI(
        api_key="fake",
        max_retries=10,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
    ) as client:
        model = OpenAIStructuredLLM(
            LLMConfig(structured_outputs=structured, max_retries=2),
            client,
        )
        result, usage = await model.complete("task", "data", Reply, with_usage=True)
        assert result.answer == "yes"
        assert client.max_retries == 10  # Borrowed client configuration is not mutated.
    assert len(calls) == 3 and calls[0] == calls[1] == calls[2]
    actual = next(item for item in usage if item.model_id == "actual")
    assert actual.input_tokens == 20 and actual.output_tokens == 10
    assert next(item for item in usage if item.model_id is None).input_tokens is None


@pytest.mark.parametrize("retries", [0, 2])
async def test_generation_bad_json_stops_at_ceiling_and_keeps_failure_usage(retries):
    calls = []

    def respond(request):
        calls.append(request)
        return httpx.Response(200, json=completion("not JSON"))

    async with AsyncOpenAI(
        api_key="fake",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
    ) as client:
        model = OpenAIStructuredLLM(
            LLMConfig(structured_outputs=False, max_retries=retries),
            client,
        )
        with pytest.raises(InvalidOutputError) as caught:
            await model.complete("task", "data", Reply, with_usage=True)
    assert len(calls) == retries + 1
    assert caught.value.usage[0].input_tokens == 10 * len(calls)
    assert caught.value.usage[0].output_tokens == 5 * len(calls)


@pytest.mark.parametrize("terminal", ["authentication", "refusal"])
async def test_generation_terminal_errors_are_not_retried(terminal):
    calls = []

    def respond(request):
        calls.append(request)
        if terminal == "authentication":
            return httpx.Response(401, json={"error": {"message": "bad key"}})
        payload = completion("refused")
        payload["choices"][0]["message"]["refusal"] = "refused"
        return httpx.Response(200, json=payload)

    async with AsyncOpenAI(
        api_key="fake",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
    ) as client:
        model = OpenAIStructuredLLM(LLMConfig(structured_outputs=False, max_retries=2), client)
        error_type = AuthenticationError if terminal == "authentication" else InvalidOutputError
        with pytest.raises(error_type):
            await model.complete("task", "data", Reply)
    assert len(calls) == 1


@pytest.mark.parametrize("success", [True, False])
async def test_grading_retry_budget_and_attempt_usage(monkeypatch, success):
    calls = []

    def respond(request):
        calls.append(json.loads(request.content))
        if len(calls) == 1:
            return httpx.Response(503)
        text = '{"answers":{"match":"one"}}' if success and len(calls) == 3 else "not JSON"
        return httpx.Response(200, json=completion(text))

    client = AsyncOpenAI(
        api_key="fake",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
    )
    monkeypatch.setattr("openai.AsyncOpenAI", lambda **_kwargs: client)
    model = DecisionModel(
        DecisionConfig(0.8),
        llm_config=LLMConfig(
            api_key="fake",
            structured_outputs=False,
            max_retries=2,
        ),
    )
    try:
        questions = {"match": Choice(instructions="Pick", criteria={"one": "One", "two": "Two"})}
        if success:
            result, usage = await model.decide(
                {},
                questions,
                mode=DecisionMode.LLM_ONLY,
                with_usage=True,
            )
            assert result.choices["match"].choice == "one"
        else:
            with pytest.raises(TypeSafeAPIResponseValidationError) as caught:
                await model.decide({}, questions, mode=DecisionMode.LLM_ONLY, with_usage=True)
            usage = caught.value.usage
        assert len(calls) == 3 and calls[0] == calls[1] == calls[2]
        actual = next(item for item in usage if item.model_id == "actual")
        assert actual.input_tokens == 20 and actual.output_tokens == 10
    finally:
        await model.close()


async def test_retry_cancellation_is_terminal():
    calls = 0

    async def operation():
        nonlocal calls
        calls += 1
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await retry(operation, 2)
    assert calls == 1


async def test_http_failure_budget_is_not_multiplied():
    calls = []

    def respond(request):
        calls.append(request)
        return httpx.Response(503)

    async with AsyncOpenAI(
        api_key="fake",
        max_retries=5,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
    ) as client:
        with pytest.raises(InternalServerError):
            model = OpenAIStructuredLLM(LLMConfig(max_retries=2), client)
            await model.complete("task", "data", Reply)
    assert len(calls) == 3


@pytest.mark.parametrize("value", [None, True, 1.0, "2", -1])
def test_retry_config_rejects_invalid_counts(value):
    with pytest.raises(ValueError, match="max_retries"):
        LLMConfig(max_retries=value)
