"""What the tool-calling seam actually puts on the wire.

Some OpenAI-compatible endpoints answer a forced tool call whose schema contains
``anyOf: [X, null]`` with ``{}`` — a well-formed, ~11-token call carrying no
arguments, reported as ``finish_reason="tool_calls"`` rather than as an error. The
same request succeeds once the union is gone, and a single nullable field anywhere
in the schema is enough to trigger it.

So these assert the transmitted schema, not the model type: "the field is optional"
and "the field is sent as optional" are different claims, and only the second one
is what the endpoint sees. The narrowness is asserted too — a genuine ``int | str``
union carries meaning, and collapsing it would silently drop a branch.
"""

import json

import pytest
from pydantic import BaseModel, Field

from lexi_ai.llm import drop_nullable_unions


def tool_schema(model: type[BaseModel]) -> dict:
    """The parameters block as ``_parse_via_tool`` transmits it."""
    from openai import pydantic_function_tool

    tool = pydantic_function_tool(model, name="emit")
    return drop_nullable_unions(tool["function"]["parameters"])


class Nullable(BaseModel):
    word: str
    note: str | None = Field(default=None, description="an optional remark")


class GenuineUnion(BaseModel):
    word: str
    count: int | str


class Inner(BaseModel):
    label: str
    hint: str | None = None


class Outer(BaseModel):
    items: list[Inner] = Field(min_length=1)


def test_a_nullable_field_is_sent_without_anyof():
    """The whole bug in one assertion: one ``anyOf`` is enough to get ``{}`` back."""
    schema = tool_schema(Nullable)
    assert "anyOf" not in json.dumps(schema)
    assert schema["properties"]["note"]["type"] == "string"


def test_a_collapsed_field_is_not_required():
    """Absence has to carry what ``null`` carried, or the model owes a value it
    cannot express."""
    schema = tool_schema(Nullable)
    assert "note" not in schema["required"]
    assert "word" in schema["required"]


def test_the_description_survives_the_collapse():
    """Pydantic hangs the prose on the union, not on the branch. Dropping it would
    leave the model guessing what the field means."""
    assert tool_schema(Nullable)["properties"]["note"]["description"] == "an optional remark"


def test_a_genuine_union_is_left_alone():
    """The bound on the transform. ``int | str`` has no null branch, and collapsing
    it would silently pick one type and discard the other."""
    schema = tool_schema(GenuineUnion)
    variants = schema["properties"]["count"]["anyOf"]
    assert {v["type"] for v in variants} == {"integer", "string"}
    assert "count" in schema["required"]


def test_nested_definitions_are_collapsed_too():
    """Nullables live inside ``$defs`` far more often than at the top level, and a
    top-level-only rewrite would look correct while fixing nothing."""
    schema = tool_schema(Outer)
    inner = schema["$defs"]["Inner"]
    assert "anyOf" not in json.dumps(inner)
    assert "hint" not in inner["required"]


def test_omitting_a_collapsed_field_still_validates_as_none():
    """The round trip. Validation must be exactly as strict as before — the wire
    schema changed, the model did not."""
    assert Nullable.model_validate(json.loads('{"word": "truculent"}')).note is None


def test_a_required_field_stays_required_after_a_sibling_collapses():
    """The rewrite edits ``required`` in place, so a bug here would strip more than
    it should and let a genuinely mandatory field go missing."""
    assert tool_schema(Nullable)["required"] == ["word"]


@pytest.mark.asyncio
async def test_an_empty_tool_call_reports_what_came_back():
    """The diagnosis this cost. A bare pydantic error says ``units`` is missing but
    not that the payload was empty, which is a different failure with a different
    fix — and reading it once needed a reproduction inside the container."""
    from lexi_ai.llm import OpenAIStructuredLLM, user_msg

    class _Fn:
        arguments = "{}"

    class _Call:
        function = _Fn()

    class _Message:
        tool_calls = [_Call()]

    class _Choice:
        message = _Message()
        finish_reason = "tool_calls"

    class _Completion:
        choices = [_Choice()]

    class _Completions:
        async def create(self, **_kwargs):
            return _Completion()

    llm = OpenAIStructuredLLM(
        base_url="http://localhost:9/v1",
        api_key="test-key",
        model="test-model",
        temperature=0.1,
        method="function_calling",
    )
    llm._client.chat.completions = _Completions()  # type: ignore[assignment]

    with pytest.raises(ValueError) as caught:
        await llm.parse([user_msg("hi")], Nullable)

    message = str(caught.value)
    assert "finish_reason='tool_calls'" in message
    assert "'{}'" in message
    assert "Nullable" in message
