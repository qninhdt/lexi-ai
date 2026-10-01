"""One bounded native structured-output transport; no tools or application retries."""

from typing import Protocol, TypeVar

from pydantic import BaseModel

from ..config import MAX_TEXT_LENGTH
from ..errors import InvalidOutputError, MissingProviderError
from .config import LLMConfig

Output = TypeVar("Output", bound=BaseModel)


class StructuredLLM(Protocol):
    async def complete(
        self, instruction: str, data: str, schema: type[Output], *, model: str | None = None
    ) -> Output: ...


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
        self, instruction: str, data: str, schema: type[Output], *, model: str | None = None
    ) -> Output:
        """Task rules have system role; untrusted user/source text stays in user role."""
        if not instruction or not isinstance(data, str) or len(data) > MAX_TEXT_LENGTH:
            raise ValueError("invalid structured request")
        kwargs = {}
        if self.config.reasoning_effort is not None:
            kwargs["reasoning_effort"] = self.config.reasoning_effort
        response = await self._get_client().chat.completions.parse(
            model=model or self.config.model,
            messages=[
                {"role": "system", "content": instruction},
                {"role": "user", "content": data},
            ],
            response_format=schema,
            max_completion_tokens=self.config.max_completion_tokens,
            timeout=self.config.timeout,
            **kwargs,
        )
        if not response.choices or response.choices[0].message.refusal:
            raise InvalidOutputError("structured request was refused or empty")
        parsed = response.choices[0].message.parsed
        if parsed is None:
            raise InvalidOutputError("structured response was empty")
        # The SDK already parsed and validated response_format into this instance.
        return parsed

    async def close(self) -> None:
        if not self._closed and self._owned and self._client is not None:
            await self._client.close()
        self._closed = True
