"""Domain-level failures that callers are expected to branch on."""


class StaleGenerationError(RuntimeError):
    """A newer generation claim superseded this worker before it could publish."""
