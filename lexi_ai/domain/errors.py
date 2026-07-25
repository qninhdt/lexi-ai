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
