"""One bounded retry budget for LLM transport and output validation failures."""

import asyncio
from json import JSONDecodeError

from pydantic import ValidationError

from ..errors import InvalidOutputError


class RefusedOutputError(InvalidOutputError):
    """A refusal is terminal, not a malformed answer to regenerate."""


def retryable(error):
    # Keep the public StructuredLLM protocol import free of provider SDKs.
    from openai import (
        APIConnectionError,
        APIResponseValidationError,
        APIStatusError,
        LengthFinishReasonError,
    )
    from typesafe_sdk import (
        TypeSafeAPIConnectionError,
        TypeSafeAPIError,
        TypeSafeAPIResponseValidationError,
    )

    if isinstance(error, RefusedOutputError):
        return False
    if isinstance(
        error,
        (
            InvalidOutputError,
            ValidationError,
            JSONDecodeError,
            APIResponseValidationError,
            LengthFinishReasonError,
            TypeSafeAPIResponseValidationError,
            APIConnectionError,
            TypeSafeAPIConnectionError,
        ),
    ):
        return True
    if isinstance(error, (APIStatusError, TypeSafeAPIError)):
        status = getattr(error, "status_code", getattr(error, "status", None))
        return status in (408, 409, 429) or isinstance(status, int) and status >= 500
    return False


async def retry(operation, max_retries):
    for attempt in range(max_retries + 1):
        try:
            return await operation()
        except Exception as error:
            if attempt == max_retries or not retryable(error):
                raise
            await asyncio.sleep(0.5 * 2 ** min(attempt, 4))
