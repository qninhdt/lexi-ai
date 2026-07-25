"""Vector-index adapters and the settings-driven selection between them.

Adding a backend means adding a module here and a branch in :func:`build_vector_index`
— no domain or application code changes, because everything upstream depends on the
:class:`~lexi_ai.domain.ports.VectorIndex` port.
"""

from lexi_ai.config import Settings, get_settings
from lexi_ai.infrastructure.vectors.memory_index import InMemoryVectorIndex

__all__ = ["InMemoryVectorIndex", "build_vector_index"]


def build_vector_index(settings: Settings | None = None):  # noqa: ANN201 - a VectorIndex
    """The vector index named by settings.

    The LanceDB adapter is constructed WITHOUT importing lancedb: the import
    happens on first table access, so a base install that never touches a vector
    pays nothing and an install missing the extra degrades at the call site
    (semantic search returns nothing) rather than failing at construction.
    """
    settings = settings or get_settings()
    backend = settings.vector_backend
    if backend == "memory":
        return InMemoryVectorIndex()
    if backend == "lancedb":
        from lexi_ai.infrastructure.vectors.lancedb_index import LanceDbVectorIndex

        return LanceDbVectorIndex(settings.vector_path, metric=settings.vector_metric)
    raise ValueError(f"unknown vector backend: {backend!r} (expected 'lancedb' or 'memory')")
