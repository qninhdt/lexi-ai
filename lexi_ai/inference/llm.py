"""Native structured output or prompted JSON; no tools or application retries."""

import json
from typing import Protocol, TypeVar

from pydantic import BaseModel, ValidationError

from ..config import MAX_TEXT_LENGTH
from ..errors import InvalidOutputError, MissingProviderError
from ..models import TokenUsage
from .config import LLMConfig
from .usage import UsageRecorder, openai_usage

Output = TypeVar("Output", bound=BaseModel)


def _parse_text[Output: BaseModel](content, schema: type[Output]) -> Output:
    if not isinstance(content, str) or not content.strip():
        raise InvalidOutputError("LLM text response was empty")
    text = content.strip()
    # Accept one fenced JSON document, not prose or a guessed JSON substring.
    lines = text.splitlines()
    if len(lines) >= 3 and lines[0].lower() in ("```", "```json") and lines[-1] == "```":
        text = "\n".join(lines[1:-1])
    try:
        return schema.model_validate_json(text)
    except ValidationError as error:
        raise InvalidOutputError("LLM text response does not match the required schema") from error


class StructuredLLM(Protocol):
    async def complete(
        self,
        instruction: str,
        data: str,
        schema: type[Output],
        *,
        model: str | None = None,
        with_usage: bool = False,
    ) -> Output | tuple[Output, list[TokenUsage]]: ...


class OpenAIStructuredLLM:
    def __init__(self, config: LLMConfig, client=None, *, own_client: bool = False):
        self.config = config
        self._client = client
        self._owned = own_client
        self._closed = False

    def _get_client(self):
        if self._closed:
            raise RuntimeError("LLM client is closed")
        if self._client is None:
            from openai import AsyncOpenAI

            if not self.config.api_key or not self.config.api_key.strip():
                raise MissingProviderError("structured LLM credentials not configured")
            self._client = AsyncOpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
                timeout=self.config.timeout,
                max_retries=2,
            )
            self._owned = True
        return self._client

    async def complete(
        self,
        instruction: str,
        data: str,
        schema: type[Output],
        *,
        model: str | None = None,
        with_usage: bool = False,
    ) -> Output | tuple[Output, list[TokenUsage]]:
        """Task rules have system role; untrusted user/source text stays in user role."""
        with UsageRecorder(with_usage) as usage:
            if not instruction or not isinstance(data, str) or len(data) > MAX_TEXT_LENGTH:
                raise ValueError("invalid structured request")
            kwargs = {}
            if self.config.reasoning_effort is not None:
                kwargs["reasoning_effort"] = self.config.reasoning_effort
            if self.config.temperature is not None:
                kwargs["temperature"] = self.config.temperature
            messages = [
                {"role": "system", "content": instruction},
                {"role": "user", "content": data},
            ]
            if not self.config.structured_outputs:
                messages[0]["content"] += (
                    "\n\nReturn only one JSON document matching this schema, "
                    "without prose or Markdown fences:\n"
                    + json.dumps(schema.model_json_schema(), ensure_ascii=False)
                )
            client = self._get_client()
            try:
                if self.config.structured_outputs:
                    request = client.chat.completions.parse
                    kwargs["response_format"] = schema
                else:
                    request = client.chat.completions.create
                response = await request(
                    model=model or self.config.model,
                    messages=messages,
                    max_completion_tokens=self.config.max_completion_tokens,
                    timeout=self.config.timeout,
                    **kwargs,
                )
            except Exception as error:
                if with_usage:
                    usage.records.append(openai_usage(getattr(error, "completion", None)))
                raise
            if with_usage:
                usage.records.append(openai_usage(response))
            if not response.choices or response.choices[0].message.refusal:
                raise InvalidOutputError("structured request was refused or empty")
            if not self.config.structured_outputs:
                if response.choices[0].finish_reason not in ("stop", None):
                    raise InvalidOutputError("LLM text response did not complete")
                return usage.finish(_parse_text(response.choices[0].message.content, schema))
            parsed = response.choices[0].message.parsed
            if parsed is None:
                raise InvalidOutputError("structured response was empty")
            # The SDK already parsed and validated response_format into this instance.
            return usage.finish(parsed)

    async def close(self) -> None:
        if not self._closed and self._owned and self._client is not None:
            await self._client.close()
        self._closed = True
