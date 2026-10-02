"""Per-request generation options at the adapter's existing OpenAI boundary."""

import json_repair
from system_one_adapter.providers import ProviderResult
from system_one_adapter.providers.base import (
    record_request,
    record_response,
    render_messages,
    translating,
)
from system_one_adapter.providers.openai import AsyncOpenAIProvider
from typesafe_sdk import TypeSafeError


class ConfiguredOpenAIProvider(AsyncOpenAIProvider):
    """Repair text JSON only; retain adapter schema validation and raw response traces."""

    def __init__(
        self,
        model_name,
        *,
        temperature=None,
        reasoning_effort=None,
        timeout=30.0,
        max_completion_tokens=4096,
        **kwargs,
    ):
        super().__init__(model_name, **kwargs)
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort
        self.timeout = timeout
        self.max_completion_tokens = max_completion_tokens

    async def request(self, messages, *, schema, structured):
        kwargs = {"model": self.model_name, "timeout": self.timeout}
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        with translating(self.translate_error):
            if self.api == "responses":
                kwargs["max_output_tokens"] = self.max_completion_tokens
                if self.reasoning_effort is not None:
                    kwargs["reasoning"] = {"effort": self.reasoning_effort}
                output_format = (
                    {"type": "json_schema", "name": "evaluation", "schema": schema, "strict": True}
                    if structured
                    else {"type": "json_object"}
                )
                kwargs.update(
                    input=render_messages(messages),
                    text={"format": output_format},
                    store=False,
                )
                if structured:
                    kwargs["instructions"] = "\n\n".join(
                        message.content for message in messages if message.role == "system"
                    )
                    kwargs["input"] = render_messages(
                        [message for message in messages if message.role != "system"]
                    )
                record_request(kwargs, api=self.api)
                response = await self._client.responses.create(**kwargs)
                record_response(response, finish_reason=response.status)
                if response.status != "completed":
                    raise TypeSafeError("LLM decision response did not complete")
                if any(
                    part.type == "refusal"
                    for item in response.output
                    if item.type == "message"
                    for part in item.content
                ):
                    raise TypeSafeError("LLM decision response was refused")
                text = response.output_text
                input_tokens = getattr(response.usage, "input_tokens", None)
                output_tokens = getattr(response.usage, "output_tokens", None)
            else:
                kwargs["max_completion_tokens"] = self.max_completion_tokens
                if self.reasoning_effort is not None:
                    kwargs["reasoning_effort"] = self.reasoning_effort
                kwargs["messages"] = render_messages(messages)
                if structured:
                    kwargs["response_format"] = {
                        "type": "json_schema",
                        "json_schema": {"name": "evaluation", "schema": schema, "strict": True},
                    }
                record_request(kwargs, api=self.api)
                response = await self._client.chat.completions.create(**kwargs)
                finish_reason = response.choices[0].finish_reason if response.choices else None
                record_response(response, finish_reason=finish_reason)
                if not response.choices or finish_reason not in ("stop", None):
                    raise TypeSafeError("LLM decision response did not complete")
                if response.choices[0].message.refusal:
                    raise TypeSafeError("LLM decision response was refused")
                text = response.choices[0].message.content or ""
                input_tokens = getattr(response.usage, "prompt_tokens", None)
                output_tokens = getattr(response.usage, "completion_tokens", None)
        if not structured:
            text = json_repair.repair_json(text, ensure_ascii=False)
        return ProviderResult(text=text, input_tokens=input_tokens, output_tokens=output_tokens)
