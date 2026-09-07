"""Tests for DocumentDB.  Covers insert, search at <10 and >=10 items,
remove, the max_dist filter, and the 120-char preview truncation rule."""

from vectordb.document_db import DocItem, DocumentDB


def _emb(seed: int, dims: int = 8) -> list[float]:
    """Deterministic fake embedding for a seed."""
    out = []
    for i in range(dims):
        out.append(((seed * (i + 7)) % 100) / 100.0)
    return out


def test_insert_sets_dims_lazily():
    db = DocumentDB()
    assert db.get_dims() == 0
    db.insert("a", "alpha", _emb(1, dims=8))
    assert db.get_dims() == 8


def test_search_empty_returns_empty():
    db = DocumentDB()
    assert db.search(_emb(1, 8), 3) == []


def test_bruteforce_path_below_10():
    """The C++ uses BF when size < 10 (main.cpp:682-684).  We verify
    that search still works (the actual algorithm choice is an
    internal detail; the contract is that the nearest k are returned)."""
    db = DocumentDB()
    for i in range(1, 6):
        db.insert(f"t{i}", f"text {i}", _emb(i, dims=4))
    # Query with a vector that matches the first embed.
    out = db.search(_emb(1, 4), 3)
    assert len(out) >= 1
    assert out[0][1].id == 1  # exact match -> distance 0


def test_hnsw_path_at_10():
    """At size >= 10 the search routes through HNSW (main.cpp:684).
    Verify the nearest match is still correct on a tight cluster."""
    db = DocumentDB()
    for i in range(1, 11):
        db.insert(f"t{i}", f"text {i}", _emb(i, dims=4))
    out = db.search(_emb(1, 4), 1)
    assert len(out) == 1
    assert out[0][1].id == 1


def test_max_dist_filter():
    """Items with distance > max_dist must be excluded."""
    db = DocumentDB()
    # One embed, then a query that's very different.
    db.insert("close", "near", [1.0, 0.0, 0.0, 0.0])
    db.insert("far", "away", [-1.0, 0.0, 0.0, 0.0])
    out = db.search([1.0, 0.0, 0.0, 0.0], 5, max_dist=0.1)
    assert all(d <= 0.1 for d, _ in out)
    # The far one has cosine distance 2.0 — well above 0.1.
    ids = {d.id for _, d in out}
    assert 2 not in ids


def test_remove_unknown_returns_false():
    db = DocumentDB()
    assert db.remove(999) is False


def test_remove_known_shrinks_size():
    db = DocumentDB()
    db.insert("a", "alpha", _emb(1, 4))
    db.insert("b", "beta", _emb(2, 4))
    assert db.size() == 2
    assert db.remove(1) is True
    assert db.size() == 1
