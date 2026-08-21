"""Domain-level failures that callers are expected to branch on."""


class StaleGenerationError(RuntimeError):
    """A newer generation claim superseded this worker before it could publish."""


class SemanticSearchUnavailable(RuntimeError):
    """Semantic search cannot run at all — the feature is off or a dep is missing.

    The one exception a caller needs to catch to degrade gracefully. Semantic
    search is an opt-in feature with two optional dependency sets (an encoder and
    a vector backend), so "cannot run" has several causes and a caller that had to
    enumerate them would miss one. It never means "no match": an empty result is a
    successful search.
    """


class SemanticSearchDisabled(SemanticSearchUnavailable):
    """The feature is switched off (``LEXI_VECTOR_BACKEND=none``, the default)."""


class VectorBackendUnavailable(SemanticSearchUnavailable):
    """The selected vector backend's optional dependency is not installed."""


class EmbeddingUnavailable(SemanticSearchUnavailable):
    """Raised when the ``[embeddings]`` extra (torch/transformers) is not installed.

    A ``SemanticSearchUnavailable`` so a caller can catch the base class once
    instead of listing every way the feature can be missing. Defined here with
    its siblings rather than beside the encoder adapter, so the whole "semantic
    search cannot run" family lives in one place.
    """


class ThemeError(ValueError):
    """A theme operation cannot proceed. Base for the specific causes below.

    Subclasses :class:`ValueError` because every one of these was raised as a
    plain ``ValueError`` before: existing ``except ValueError`` handlers keep
    working, while a caller that wants to distinguish causes branches on the
    subclasses instead of parsing messages.
    """


class UnknownTheme(ThemeError):
    """No theme matches the requested key or id."""


class InvalidThemeName(ThemeError):
    """A proposed theme name normalizes to no usable key, so none can be created."""


class ThemeTargetInvalid(ThemeError):
    """The word or sense a theme operation targets is missing or not in a
    themeable state (only a done word with neutral senses can be restyled)."""


class ThemedOverlayMissing(ThemeError):
    """The sense has no themed overlay for the requested theme; the word must be
    themed first via ``generate(theme=...)``."""
