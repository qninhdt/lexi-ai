"""Unit tests for the pure-Python cosine used by the in-memory vector index."""

import math

from lexi_ai.vectors import cosine


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
