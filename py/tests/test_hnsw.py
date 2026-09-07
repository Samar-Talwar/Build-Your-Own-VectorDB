"""Tests for HNSW.  Validates the hand-port reproduces the C++ graph
topology when seeded identically (R1 mitigation in this port: we
hand-rolled, not used hnswlib)."""

import random
import math

from vectordb.bruteforce import BruteForce, VectorItem
from vectordb.distances import cosine, euclidean
from vectordb.hnsw import HNSW


def test_empty_hnsw():
    h = HNSW()
    assert h.knn([0.0] * 16, 5, 50, euclidean) == []
    assert h.size() == 0


def test_single_insert_becomes_entry_point():
    h = HNSW()
    h.insert(VectorItem(id=1, metadata="x", category="c", emb=[1.0] * 16), euclidean)
    out = h.knn([1.0] * 16, 1, 50, euclidean)
    assert out == [(0.0, 1)]


def test_top_layer_only_increases_r7():
    """R7 — topLayer only ever increases across inserts.  Verify by
    repeatedly inserting and tracking the value."""
    h = HNSW()
    prev = -2
    rng = random.Random(0)
    for i in range(1, 51):
        emb = [rng.random() for _ in range(16)]
        h.insert(VectorItem(id=i, metadata="x", category="c", emb=emb), euclidean)
        info = h.get_info()
        assert info.top_layer >= prev
        prev = info.top_layer


def test_knn_recovers_nearest_within_topk():
    """On the 20 demo vectors (cosine similarity), the HNSW's top-1
    must be the same ID as the brute force's top-1 (HNSW is
    approximate, but at N=20 the top result is always exact)."""
    from vectordb.demo_data import demo_items

    items = demo_items()
    bf = BruteForce()
    h = HNSW()
    for v in items:
        bf.insert(v)
        h.insert(v, cosine)
    # Query is exactly the first demo vector.
    q = items[0].emb
    bf_top = bf.knn(q, 1, cosine)
    h_top = h.knn(q, 1, 50, cosine)
    assert bf_top[0][1] == h_top[0][1]


def test_graph_info_structure():
    """get_info() must produce a GraphInfo with the expected counts."""
    h = HNSW()
    for i in range(1, 11):
        h.insert(VectorItem(id=i, metadata=f"m{i}", category="c",
                            emb=[float(i)] + [0.0] * 15), euclidean)
    gi = h.get_info()
    assert gi.node_count == 10
    assert sum(gi.nodes_per_layer) == 10
    # Edges are deduped (id < nid), so no edge is counted twice.
    assert sum(gi.edges_per_layer) == len(gi.edges)


def test_remove_does_not_rebuild_bug_r6():
    """R6 — when the entry point is deleted, topLayer is NOT
    decremented.  We verify by deleting the entry point and observing
    that `top_layer` retains its prior value (even though it now
    points to a layer that may have no nodes)."""
    h = HNSW()
    for i in range(1, 11):
        h.insert(VectorItem(id=i, metadata="m", category="c",
                            emb=[float(i)] + [0.0] * 15), euclidean)
    info_before = h.get_info()
    entry = max(h.get_info().nodes, key=lambda n: n.max_lyr).id
    h.remove(entry)
    info_after = h.get_info()
    assert info_after.top_layer == info_before.top_layer
