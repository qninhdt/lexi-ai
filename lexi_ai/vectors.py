"""Cosine similarity in plain Python, zero dependencies.

Used by the in-memory vector index, which ranks by exact scan. The durable
backend (LanceDB) computes distance itself, so this is deliberately not a hot
path: it exists so the hermetic test tier and small local runs need no native
extension and no vector service.
"""

import math


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity of two equal-length vectors, in ``[-1.0, 1.0]``.

    Returns ``0.0`` if either vector is all-zeros (undefined direction) or the
    lengths differ (never rank a mismatched-dim vector). Vectors produced by the
    Embedder are already L2-normalized, so this is effectively a dot product, but
    it normalizes anyway so the function is correct for any input.
    """
    if len(a) != len(b) or not a:
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b, strict=True):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))
