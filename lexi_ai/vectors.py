"""Portable float32 vector codec + cosine — pure-Python, zero deps.

Sense embeddings are stored as a ``LargeBinary`` BLOB (SQLite) / BYTEA (Postgres)
so the schema stays dialect-agnostic (no pgvector, no ARRAY). A vector is packed
as little-endian float32 via the stdlib :mod:`array` module — compact, exact
round-trip, and readable back by numpy/torch (``frombuffer(..., '<f4')``) if a
caller ever wants to.

Cosine similarity is computed in plain Python: the generated dictionary grows
lazily, so the candidate set is small and a C extension (numpy) is not warranted.
If that ever changes, swap :func:`cosine` for a numpy/pgvector path without
touching the storage format.
"""

import array
import math
import sys

# Struct code for float32. array is native-endian, so on a big-endian host we
# byteswap to keep every stored BLOB little-endian — portable across machines
# and directly readable by numpy/torch as '<f4'.
_F32 = "f"
_BIG_ENDIAN = sys.byteorder == "big"


def pack_vector(values: list[float]) -> bytes:
    """Pack a float sequence into little-endian float32 bytes."""
    arr = array.array(_F32, values)
    if _BIG_ENDIAN:  # pragma: no cover - dev/CI are little-endian
        arr.byteswap()
    return arr.tobytes()


def unpack_vector(blob: bytes) -> list[float]:
    """Unpack little-endian float32 bytes back into a list of floats."""
    arr = array.array(_F32)
    arr.frombytes(blob)
    if _BIG_ENDIAN:  # pragma: no cover - dev/CI are little-endian
        arr.byteswap()
    return arr.tolist()


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity of two equal-length vectors, in ``[-1.0, 1.0]``.

    Returns ``0.0`` if either vector is all-zeros (undefined direction) or the
    lengths differ (never rank a mismatched-dim vector). Vectors produced by the
    Embedder are already L2-normalized, so this is effectively a dot product, but
    we normalize here too so the function is correct for any input.
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
