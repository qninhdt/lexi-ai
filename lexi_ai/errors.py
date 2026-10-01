"""Errors at the consumer boundary."""


class LexiconError(Exception):
    """Base error for an unsuccessful dictionary operation."""


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
