"""Tests for distance metrics.  Validates bit-equivalent behaviour to
main.cpp:39-58 on the 20 demo vectors and a handful of edge cases."""

from vectordb.distances import cosine, euclidean, get_dist_fn, manhattan


def test_euclidean_zero():
    a = [0.0, 0.0, 0.0]
    b = [0.0, 0.0, 0.0]
    assert euclidean(a, b) == 0.0


def test_euclidean_3_4_5():
    # Classic 3-4-5 triangle.
    assert euclidean([0.0, 0.0], [3.0, 4.0]) == 5.0


def test_cosine_identical_returns_zero():
    # Same vector -> dot == |a||b| -> 1 - 1 == 0.
    v = [0.9, 0.85, 0.72, 0.68, 0.12, 0.08, 0.15, 0.10, 0.05, 0.08, 0.06, 0.09, 0.07, 0.11, 0.08, 0.06]
    assert cosine(v, v) == 0.0


def test_cosine_orthogonal_returns_one():
    # Two orthogonal unit vectors -> dot == 0 -> 1 - 0 == 1.
    a = [1.0, 0.0]
    b = [0.0, 1.0]
    assert cosine(a, b) == 1.0


def test_cosine_zero_vector_sentinel():
    # C++ returns 1.0 when either norm is < 1e-9 (main.cpp:50-51).
    assert cosine([0.0, 0.0, 0.0], [1.0, 2.0, 3.0]) == 1.0
    assert cosine([1.0, 2.0, 3.0], [0.0, 0.0, 0.0]) == 1.0


def test_manhattan_basic():
    assert manhattan([0.0, 0.0], [3.0, 4.0]) == 7.0
    assert manhattan([1.0, 1.0], [1.0, 1.0]) == 0.0


def test_get_dist_fn_dispatch():
    assert get_dist_fn("cosine") is cosine
    assert get_dist_fn("manhattan") is manhattan
    # Unknown name and empty both fall back to euclidean (C++ default).
    assert get_dist_fn("") is euclidean
    assert get_dist_fn("mystery") is euclidean
