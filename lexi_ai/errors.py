"""Errors at the consumer boundary."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .models import TokenUsage


class LexiconError(Exception):
    """Base error for an unsuccessful dictionary operation."""

    usage: list[TokenUsage] | None = None


class InvalidHandleError(LexiconError):
    """A reference entry handle is malformed or no longer valid."""


class InvalidResourceError(LexiconError):
    """A requested resource or parameter is invalid."""


class QuestionNotFoundError(InvalidResourceError):
    """The requested saved Question does not exist."""


class QuestionBankChangedError(InvalidResourceError):
    """A requested bank cannot supply the allocated number of Questions."""


class UnsupportedQuestionFormatError(InvalidResourceError):
    """A format is unsupported by the Question or the caller's allowed pairs."""


class MissingProviderError(LexiconError):
    """A requested AI operation needs a provider that was not configured."""


class InvalidOutputError(LexiconError):
    """A provider returned content that cannot be published safely."""


class WordCollisionError(LexiconError):
    """A selected source conflicts with an already completed lexical identity."""
