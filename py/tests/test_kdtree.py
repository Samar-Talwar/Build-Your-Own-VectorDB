"""Tests for KDTree.  Validates parity with BruteForce on the demo data
and on random points (so the tree doesn't get pathologically unbalanced
on a single test case)."""

import random
import math

from vectordb.bruteforce import BruteForce, VectorItem
from vectordb.distances import euclidean
from vectordb.kdtree import KDTree


def _populate(kdt: KDTree, bf: BruteForce, n: int, dims: int, seed: int = 0) -> list[VectorItem]:
    rng = random.Random(seed)
    items: list[VectorItem] = []
    for i in range(1, n + 1):
        emb = [rng.random() for _ in range(dims)]
        v = VectorItem(id=i, metadata=f"m{i}", category="c", emb=emb)
        items.append(v)
        bf.insert(v)
        kdt.insert(v)
    return items


def test_matches_bruteforce_on_demo():
    """On the 20 demo vectors, the KD-Tree's top-3 nearest must match
    brute force for any random query vector (within float rounding)."""
    from vectordb.demo_data import demo_items

    items = demo_items()
    bf = BruteForce()
    kdt = KDTree(16)
    for v in items:
        bf.insert(v)
        kdt.insert(v)

    q = [0.9, 0.85, 0.72, 0.68, 0.12, 0.08, 0.15, 0.10, 0.05, 0.08, 0.06, 0.09, 0.07, 0.11, 0.08, 0.06]
    bf_out = bf.knn(q, 5, euclidean)
    kd_out = kdt.knn(q, 5, euclidean)
    bf_ids = [i for _, i in bf_out]
    kd_ids = [i for _, i in kd_out]
    assert bf_ids == kd_ids


def test_matches_bruteforce_random():
    """Random 5D points: KD-Tree must produce the same nearest IDs as BF."""
    n = 50
    k = 3
    dims = 5
    bf = BruteForce()
    kdt = KDTree(dims)
    rng = random.Random(7)
    for i in range(n):
        emb = [rng.random() for _ in range(dims)]
        v = VectorItem(id=i, metadata="x", category="y", emb=emb)
        bf.insert(v)
        kdt.insert(v)
    q = [rng.random() for _ in range(dims)]
    bf_out = bf.knn(q, k, euclidean)
    kd_out = kdt.knn(q, k, euclidean)
    assert [i for _, i in bf_out] == [i for _, i in kd_out]


def test_empty_tree_returns_empty():
    kdt = KDTree(3)
    assert kdt.knn([0.0, 0.0, 0.0], 5, euclidean) == []


def test_k_larger_than_n_returns_all():
    bf = BruteForce()
    kdt = KDTree(3)
    _populate(kdt, bf, n=3, dims=3)
    assert len(kdt.knn([0.0, 0.0, 0.0], 10, euclidean)) == 3
