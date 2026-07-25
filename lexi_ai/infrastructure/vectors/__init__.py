"""Vector-index adapters and the settings-driven selection between them.

Adding a backend means adding a module here and a branch in :func:`build_vector_index`
— no domain or application code changes, because everything upstream depends on the
:class:`~lexi_ai.domain.ports.VectorIndex` port.

Semantic search is an OPT-IN feature: the default backend is ``none``, which means
no index exists at all and the services that would use one say so instead of
pretending. See :func:`build_vector_index`.
"""

import importlib.util

from lexi_ai.config import Settings, get_settings
from lexi_ai.domain.errors import VectorBackendUnavailable
from lexi_ai.infrastructure.vectors.memory_index import InMemoryVectorIndex

__all__ = ["InMemoryVectorIndex", "build_vector_index"]

# Backend -> the extra that provides it. A backend absent from this map needs no
# optional dependency.
_REQUIRED_EXTRA = {"lancedb": ("lancedb", "lancedb")}


def build_vector_index(settings: Settings | None = None):  # noqa: ANN201 - VectorIndex | None
    """The vector index named by settings, or ``None`` when the feature is off.

    ``none`` (the default) returns ``None``. That is not a null object: a null
    object would have to guess whether a given caller wants a hard error or a
    silent skip, and the two callers want opposite things — an explicit
    ``semantic_search`` must raise, while the post-commit embed hook must not fail
    a generation. ``None`` lets each call site decide, and it makes the optional
    dependency visible in the type instead of hiding behind a fake index.

    A backend whose optional dependency is missing fails HERE, at construction,
    rather than on first use. Selecting a backend is an explicit opt-in, so the
    error belongs at the moment of that choice, with the install command in it.
    """
    settings = settings or get_settings()
    backend = settings.vector_backend
    if backend == "none":
        return None
    if backend == "memory":
        return InMemoryVectorIndex()
    if backend == "lancedb":
        _require_extra(backend)
        from lexi_ai.infrastructure.vectors.lancedb_index import LanceDbVectorIndex

        return LanceDbVectorIndex(settings.vector_path, metric=settings.vector_metric)
    raise ValueError(
        f"unknown vector backend: {backend!r} (expected 'none', 'memory' or 'lancedb')"
    )


def _require_extra(backend: str) -> None:
    """Fail with an actionable message when the backend's extra is not installed."""
    module, extra = _REQUIRED_EXTRA[backend]
    if importlib.util.find_spec(module) is None:
        raise VectorBackendUnavailable(
            f"LEXI_VECTOR_BACKEND={backend!r} needs the '[{extra}]' extra. "
            f"Install with: uv sync --extra {extra}  (or: pip install 'lexi-ai[{extra}]')"
        )
