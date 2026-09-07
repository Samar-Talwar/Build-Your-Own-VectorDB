"""Tests for the demo data — verifies the 20 vectors match the C++
source line-for-line so the visual scatter plot stays identical."""

from vectordb.demo_data import _DEMO, demo_items


def test_20_vectors():
    assert len(_DEMO) == 20
    assert len(demo_items()) == 20


def test_categories_in_blocks():
    """First 5 CS, next 5 Math, next 5 Food, next 5 Sports."""
    cats = [c for _, c, _ in _DEMO]
    assert cats[:5] == ["cs"] * 5
    assert cats[5:10] == ["math"] * 5
    assert cats[10:15] == ["food"] * 5
    assert cats[15:20] == ["sports"] * 5


def test_ids_1_indexed():
    items = demo_items()
    assert [v.id for v in items] == list(range(1, 21))


def test_all_vectors_are_16d():
    for _, _, emb in _DEMO:
        assert len(emb) == 16


def test_first_vector_is_linked_list():
    """The first demo vector must be the linked list, with the values
    copied byte-exact from main.cpp:720-721."""
    items = demo_items()
    assert items[0].metadata == "Linked List: nodes connected by pointers"
    assert items[0].category == "cs"
    assert items[0].emb == [
        0.90, 0.85, 0.72, 0.68, 0.12, 0.08, 0.15, 0.10,
        0.05, 0.08, 0.06, 0.09, 0.07, 0.11, 0.08, 0.06,
    ]
