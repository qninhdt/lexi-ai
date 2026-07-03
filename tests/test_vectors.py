"""Unit tests for the pure-Python vector codec + cosine (lexi_ai.vectors)."""

import math

from lexi_ai.vectors import cosine, pack_vector, unpack_vector


def test_pack_unpack_roundtrip_float32():
    v = [0.1, -0.2, 0.3, 1.0, -1.0, 0.0, 123.456, -987.654]
    blob = pack_vector(v)
    assert len(blob) == 4 * len(v)  # float32 = 4 bytes each
    back = unpack_vector(blob)
    assert len(back) == len(v)
    # float32 precision, not float64 — compare with tolerance.
    assert all(abs(a - b) < 1e-3 for a, b in zip(v, back, strict=True))


def test_pack_empty():
    assert pack_vector([]) == b""
    assert unpack_vector(b"") == []


def test_cosine_identity_orthogonal_opposite():
    v = [1.0, 2.0, 3.0, 4.0]
    assert abs(cosine(v, v) - 1.0) < 1e-9
    assert abs(cosine([1.0, 0.0], [0.0, 1.0])) < 1e-9
    assert abs(cosine([1.0, 0.0], [-1.0, 0.0]) + 1.0) < 1e-9


def test_cosine_is_magnitude_invariant():
    a = [1.0, 2.0, 3.0]
    b = [2.0, 4.0, 6.0]  # same direction, 2x magnitude
    assert abs(cosine(a, b) - 1.0) < 1e-9


def test_cosine_zero_vector_is_zero():
    assert cosine([0.0, 0.0, 0.0], [1.0, 2.0, 3.0]) == 0.0
    assert cosine([1.0, 2.0], [0.0, 0.0]) == 0.0


def test_cosine_dim_mismatch_is_zero():
    # Never rank a mismatched-dimension vector.
    assert cosine([1.0, 2.0, 3.0], [1.0, 2.0]) == 0.0
    assert cosine([], []) == 0.0


def test_cosine_matches_manual():
    a = [1.0, 2.0, 2.0]
    b = [2.0, 0.0, 1.0]
    dot = 1 * 2 + 2 * 0 + 2 * 1
    expected = dot / (math.sqrt(1 + 4 + 4) * math.sqrt(4 + 0 + 1))
    assert abs(cosine(a, b) - expected) < 1e-9


def test_roundtrip_preserves_cosine():
    # A vector packed→unpacked still cosines ~1.0 with its original.
    v = [0.11, 0.22, -0.33, 0.44, -0.55]
    back = unpack_vector(pack_vector(v))
    assert abs(cosine(v, back) - 1.0) < 1e-6
