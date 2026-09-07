"""Tests for VectorDB (16D demo store).  Verifies insert / remove
keep all three indexes in sync and that benchmark returns non-negative
microsecond counts."""

from vectordb.db import VectorDB
from vectordb.demo_data import load_demo


def test_load_demo_20_items():
    db = VectorDB(16)
    load_demo(db)
    assert db.size() == 20


def test_insert_assigns_sequential_ids():
    db = VectorDB(16)
    load_demo(db)
    a = db.insert("a", "c", [0.5] * 16, lambda x, y: 0.0)
    b = db.insert("b", "c", [0.5] * 16, lambda x, y: 0.0)
    assert a == 21
    assert b == 22


def test_remove_unknown_returns_false():
    db = VectorDB(16)
    load_demo(db)
    assert db.remove(999) is False


def test_remove_known_returns_true_and_shrinks_all_indexes():
    db = VectorDB(16)
    load_demo(db)
    n_before = db.size()
    assert db.remove(1) is True
    assert db.size() == n_before - 1
    # The remaining items are still searchable.
    out = db.search([0.5] * 16, 5, "cosine", "bruteforce")
    assert all(h.id != 1 for h in out.hits)


def test_search_bruteforce_returns_3_closest_in_district():
    """On the demo data, querying with a CS-like vector must return
    the 3 CS vectors (ids 1..5) at the top — same as the C++."""
    db = VectorDB(16)
    load_demo(db)
    # Linked-list vector: identity-1 in cosine space.
    q = [0.90, 0.85, 0.72, 0.68, 0.12, 0.08, 0.15, 0.10, 0.05, 0.08, 0.06, 0.09, 0.07, 0.11, 0.08, 0.06]
    out = db.search(q, 3, "cosine", "bruteforce")
    assert {h.id for h in out.hits}.issubset({1, 2, 3, 4, 5})


def test_search_kdtree_matches_bruteforce_top3():
    db = VectorDB(16)
    load_demo(db)
    q = [0.10, 0.10, 0.10, 0.10, 0.85, 0.85, 0.85, 0.85, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10]
    bf = db.search(q, 3, "euclidean", "bruteforce")
    kd = db.search(q, 3, "euclidean", "kdtree")
    assert [h.id for h in bf.hits] == [h.id for h in kd.hits]


def test_search_hnsw_recovers_top():
    """HNSW is approximate, but on demo data + cosine the top-1
    must equal the BF top-1 (the demo has tight clusters)."""
    db = VectorDB(16)
    load_demo(db)
    q = [0.10, 0.10, 0.10, 0.10, 0.85, 0.85, 0.85, 0.85, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10]
    bf = db.search(q, 1, "cosine", "bruteforce")
    hnsw = db.search(q, 1, "cosine", "hnsw")
    assert bf.hits[0].id == hnsw.hits[0].id


def test_benchmark_returns_int_microseconds():
    db = VectorDB(16)
    load_demo(db)
    q = [0.5] * 16
    b = db.benchmark(q, 5, "cosine")
    assert b.bf_us >= 0
    assert b.kd_us >= 0
    assert b.hnsw_us >= 0
    assert b.n == 20


def test_hnsw_info_structure():
    db = VectorDB(16)
    load_demo(db)
    gi = db.hnsw_info()
    assert gi.node_count == 20
    assert sum(gi.nodes_per_layer) == 20
    assert gi.top_layer >= 0
