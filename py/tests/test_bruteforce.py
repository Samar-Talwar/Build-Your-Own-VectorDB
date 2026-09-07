"""Tests for BruteForce k-NN.  Covers insert, knn ordering, and remove."""

from vectordb.bruteforce import BruteForce, VectorItem
from vectordb.distances import euclidean


def _items(n: int = 5) -> list[VectorItem]:
    return [VectorItem(id=i, metadata=f"m{i}", category="c", emb=[float(i), 0.0, 0.0])
            for i in range(1, n + 1)]


def test_insert_and_size():
    bf = BruteForce()
    for v in _items():
        bf.insert(v)
    assert len(bf.items) == 5


def test_knn_returns_k_closest_ordered():
    bf = BruteForce()
    for v in _items():
        bf.insert(v)
    # Query at (2.5, 0, 0): id 2 and 3 tie at 0.5; std::sort on pair
    # breaks ties by ascending id, so id 2 comes first (matches the C++).
    out = bf.knn([2.5, 0.0, 0.0], 3, euclidean)
    assert len(out) == 3
    ids = [i for _, i in out]
    assert ids[0] == 2
    # Distances must be in ascending order.
    ds = [d for d, _ in out]
    assert ds == sorted(ds)


def test_knn_k_larger_than_n_returns_all():
    bf = BruteForce()
    for v in _items(3):
        bf.insert(v)
    out = bf.knn([0.0, 0.0, 0.0], 10, euclidean)
    assert len(out) == 3


def test_remove():
    bf = BruteForce()
    for v in _items(3):
        bf.insert(v)
    bf.remove(2)
    assert {v.id for v in bf.items} == {1, 3}


def test_remove_missing_is_noop():
    bf = BruteForce()
    bf.insert(_items(1)[0])
    bf.remove(999)  # no error
    assert len(bf.items) == 1
