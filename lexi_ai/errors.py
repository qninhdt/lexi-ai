"""Errors at the consumer boundary."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .models import TokenUsage


class LexiconError(Exception):
    """Base error for an unsuccessful dictionary operation."""

    usage: list[TokenUsage] | None = None


class InvalidHandleError(LexiconError):
    """An available entry handle is malformed or no longer valid."""


class InvalidResourceError(LexiconError):
    """A requested resource or parameter is invalid."""


class MissingProviderError(LexiconError):
    """A requested AI operation needs a provider that was not configured."""


class InvalidOutputError(LexiconError):
    """A provider returned content that cannot be published safely."""


class WordCollisionError(LexiconError):
    """A selected source conflicts with an already completed lexical identity."""
