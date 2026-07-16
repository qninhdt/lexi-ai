"""Safe public errors; adapters must never serialize arbitrary exceptions."""

from dataclasses import dataclass
from enum import Enum
from uuid import uuid4


class ErrorCode(str, Enum):
    UNAUTHENTICATED = "unauthenticated"
    FORBIDDEN = "forbidden"
    VALIDATION = "validation_failed"
    NOT_FOUND = "not_found"
    CONFLICT = "conflict"
    DEADLINE_EXCEEDED = "deadline_exceeded"
    PRECONDITION_FAILED = "precondition_failed"
    INTERNAL = "internal"


@dataclass(frozen=True)
class PublicError:
    code: ErrorCode
    message: str
    retryable: bool
    incident_id: str


class ApplicationError(Exception):
    def __init__(self, error: PublicError):
        super().__init__(error.code.value)
        self.error = error


def public_error(code: ErrorCode, message: str, *, retryable: bool = False) -> ApplicationError:
    return ApplicationError(PublicError(code, message, retryable, uuid4().hex))


def to_public_error(exc: Exception) -> PublicError:
    """Map errors without exposing provider, DB, or filesystem diagnostics."""
    if isinstance(exc, ApplicationError):
        return exc.error
    return PublicError(
        ErrorCode.INTERNAL,
        "The service could not complete the request.",
        retryable=True,
        incident_id=uuid4().hex,
    )
