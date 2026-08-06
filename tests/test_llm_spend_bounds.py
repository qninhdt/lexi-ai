"""Cost and latency ceilings on the structured-LLM seam.

Three separate ways an unbounded call costs real money or hangs a request:

* No ``max_tokens``: structured output is bounded by its schema, so a response
  that runs long is a model that stopped emitting the schema — billed per token
  regardless.
* No timeout: a provider that accepts the connection and then stalls holds the
  caller open indefinitely. Generation is awaited inside a request, so the stall
  propagates to the user.
* Retrying ``ValidationError``: a schema violation is deterministic given the same
  prompt and temperature. Retrying buys two more full-price calls to be told the
  same thing.

The first two are asserted against the arguments actually handed to the client,
because "the setting exists" and "the setting is sent" are different claims.
"""

import asyncio

import pytest
from pydantic import BaseModel, ValidationError

from lexi_ai.llm import ainvoke_structured, build_structured_llm, user_msg

pytestmark = pytest.mark.asyncio


class _Shape(BaseModel):
    value: int


class _RecordingCompletions:
    """Captures the kwargs of whichever completion method gets called."""

    def __init__(self) -> None:
        self.parse_kwargs: list[dict] = []
        self.create_kwargs: list[dict] = []

    async def parse(self, **kwargs):
        self.parse_kwargs.append(kwargs)
        raise AssertionError("not reached: these tests only inspect the arguments")

    async def create(self, **kwargs):
        self.create_kwargs.append(kwargs)
        raise AssertionError("not reached: these tests only inspect the arguments")


class _Settings:
    """The duck-typed settings object ``build_structured_llm`` reads."""

    llm_base_url = "http://localhost:9/v1"
    llm_api_key = "test-key"
    llm_model = "test-model"
    llm_temperature = 0.1
    llm_structured_method = ""
    llm_reasoning_effort = ""
    llm_max_tokens = 1234
    llm_timeout_seconds = 7.5


def _with_recording_client(llm) -> _RecordingCompletions:
    recording = _RecordingCompletions()

    class _Chat:
        completions = recording

    llm._client.chat = _Chat()  # noqa: SLF001 - inspecting the seam is the point
    return recording


async def test_max_tokens_is_sent_on_the_json_schema_path():
    llm = build_structured_llm(_Settings())
    recording = _with_recording_client(llm)

    with pytest.raises(AssertionError, match="not reached"):
        await llm.parse([user_msg("hi")], _Shape)

    assert recording.parse_kwargs[0]["max_tokens"] == 1234


async def test_max_tokens_is_sent_on_the_function_calling_path():
    class _ToolSettings(_Settings):
        llm_structured_method = "function_calling"

    llm = build_structured_llm(_ToolSettings())
    recording = _with_recording_client(llm)

    with pytest.raises(AssertionError, match="not reached"):
        await llm.parse([user_msg("hi")], _Shape)

    assert recording.create_kwargs[0]["max_tokens"] == 1234


async def test_the_client_carries_a_request_timeout():
    llm = build_structured_llm(_Settings())

    # Read off the constructed client rather than a private copy: this is the
    # value that actually bounds a stalled provider.
    assert llm._client.timeout == 7.5  # noqa: SLF001


async def test_a_zero_setting_omits_the_field_rather_than_sending_zero():
    """max_tokens=0 would be an invalid request, not 'unlimited'."""

    class _Unset(_Settings):
        llm_max_tokens = 0

    llm = build_structured_llm(_Unset())
    recording = _with_recording_client(llm)

    with pytest.raises(AssertionError, match="not reached"):
        await llm.parse([user_msg("hi")], _Shape)

    assert "max_tokens" not in recording.parse_kwargs[0]


async def test_a_schema_violation_is_not_retried():
    """One call, not three, when the model returns output the schema rejects."""
    calls: list[int] = []

    class _BadOutput:
        async def parse(self, _messages, _schema):
            calls.append(1)
            # What a loose proxy actually does: right shape, wrong type.
            return _Shape.model_validate({"value": "not-an-integer"})

    with pytest.raises(ValidationError):
        await ainvoke_structured(_BadOutput(), [user_msg("hi")], _Shape, base_delay=0)

    assert len(calls) == 1, f"paid for {len(calls)} calls on a deterministic failure"


async def test_a_validation_error_raised_while_coercing_a_dict_is_not_retried():
    """The other ValidationError site: a provider returning a raw dict."""
    calls: list[int] = []

    class _DictOutput:
        async def parse(self, _messages, _schema):
            calls.append(1)
            return {"value": "not-an-integer"}

    with pytest.raises(ValidationError):
        await ainvoke_structured(_DictOutput(), [user_msg("hi")], _Shape, base_delay=0)

    assert len(calls) == 1, f"paid for {len(calls)} calls on a deterministic failure"


async def test_transient_failures_are_still_retried():
    """The retry must survive for the failures a second attempt actually fixes."""
    calls: list[int] = []

    class _FlakyThenFine:
        async def parse(self, _messages, _schema):
            calls.append(1)
            if len(calls) < 3:
                raise asyncio.TimeoutError("provider stalled")
            return _Shape(value=42)

    result = await ainvoke_structured(
        _FlakyThenFine(), [user_msg("hi")], _Shape, base_delay=0
    )

    assert result.value == 42
    assert len(calls) == 3, "a transient failure stopped being retried"


async def test_a_persistent_transient_failure_still_exhausts_and_raises():
    calls: list[int] = []

    class _AlwaysTimingOut:
        async def parse(self, _messages, _schema):
            calls.append(1)
            raise asyncio.TimeoutError("provider stalled")

    with pytest.raises(asyncio.TimeoutError):
        await ainvoke_structured(
            _AlwaysTimingOut(), [user_msg("hi")], _Shape, max_retries=3, base_delay=0
        )

    assert len(calls) == 3
