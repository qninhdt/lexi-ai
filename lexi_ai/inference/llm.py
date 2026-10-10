"""Native structured output or prompted JSON with bounded same-model retries."""

import json
from typing import Protocol, TypeVar

import json_repair
from pydantic import BaseModel

from ..config import MAX_PROMPT_LENGTH
from ..errors import InvalidOutputError, MissingProviderError
from ..models import TokenUsage
from .config import LLMConfig
from .retry import RefusedOutputError, retry
from .usage import UsageRecorder, openai_usage

Output = TypeVar("Output", bound=BaseModel)


def _prompt_schema(value):
    """Titles repeat field names; preserve descriptions and validation constraints."""
    if isinstance(value, dict):
        return {key: _prompt_schema(item) for key, item in value.items() if key != "title"}
    if isinstance(value, list):
        return [_prompt_schema(item) for item in value]
    return value


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
                max_retries=0,
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
            if not instruction or not isinstance(data, str) or len(data) > MAX_PROMPT_LENGTH:
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
                    "\n\nReturn only one compact JSON document matching this schema, "
                    "without indentation or unnecessary whitespace outside string values, "
                    "and without prose or Markdown fences:\n"
                    + json.dumps(
                        _prompt_schema(schema.model_json_schema()),
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                )
            client = self._get_client()
            # SDK transport retries must not multiply the shared attempt ceiling.
            # with_options shares the injected client's connection pool without
            # changing its owner's configuration or lifecycle.
            if getattr(client, "max_retries", 0):
                client = client.with_options(max_retries=0)
            raw = (
                getattr(client.chat.completions, "with_raw_response", None)
                if with_usage and self.config.structured_outputs
                else None
            )
            if self.config.structured_outputs:
                request = raw.parse if raw is not None else client.chat.completions.parse
                kwargs["response_format"] = schema
            else:
                request = client.chat.completions.create

            async def attempt():
                recorded = False
                try:
                    response = await request(
                        model=model or self.config.model,
                        messages=messages,
                        max_completion_tokens=self.config.max_completion_tokens,
                        timeout=self.config.timeout,
                        **kwargs,
                    )
                    if raw is not None:
                        # Native SDK parsing can raise without attaching the
                        # completion. Capture its usage before that parser runs.
                        usage.records.append(openai_usage(response.http_response.json()))
                        recorded = True
                        response = response.parse()
                except Exception as error:
                    if with_usage and not recorded:
                        usage.records.append(openai_usage(getattr(error, "completion", None)))
                    raise
                if with_usage and not recorded:
                    usage.records.append(openai_usage(response))
                if not response.choices:
                    raise InvalidOutputError("structured response was empty")
                if response.choices[0].message.refusal:
                    raise RefusedOutputError("structured request was refused")
                if not self.config.structured_outputs:
                    if response.choices[0].finish_reason == "content_filter":
                        raise RefusedOutputError("LLM text response was filtered")
                    if response.choices[0].finish_reason not in ("stop", None):
                        raise InvalidOutputError("LLM text response did not complete")
                    content = response.choices[0].message.content
                    if not isinstance(content, str) or not content.strip():
                        raise InvalidOutputError("LLM text response was empty")
                    try:
                        return schema.model_validate(json_repair.loads(content))
                    except ValueError as error:
                        raise InvalidOutputError(
                            "LLM text response does not match the required schema"
                        ) from error
                parsed = response.choices[0].message.parsed
                if parsed is None:
                    raise InvalidOutputError("structured response was empty")
                # The SDK already parsed/validated this instance; do not redo it.
                return parsed

            return usage.finish(await retry(attempt, self.config.max_retries))

    async def close(self) -> None:
        if not self._closed and self._owned and self._client is not None:
            await self._client.close()
        self._closed = True
