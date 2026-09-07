"""The 20 hand-tuned 16D demo vectors — direct port of main.cpp:717-760.

These are the product's visible artefact: the scatter plot the user sees
on page load.  Any drift (regenerating them, rounding, transposing rows)
breaks the visual contract.  The values are copied byte-for-byte from
the C++ source.

Categorical block layout (4 dimensions each):
  * dims 0-3   — CS
  * dims 4-7   — Math
  * dims 8-11  — Food
  * dims 12-15 — Sports
"""

from __future__ import annotations

from .bruteforce import VectorItem
from .distances import get_dist_fn
from .db import VectorDB


_DEMO: list[tuple[str, str, list[float]]] = [
    ("Linked List: nodes connected by pointers", "cs",
        [0.90,0.85,0.72,0.68,0.12,0.08,0.15,0.10,0.05,0.08,0.06,0.09,0.07,0.11,0.08,0.06]),
    ("Binary Search Tree: O(log n) search and insert", "cs",
        [0.88,0.82,0.78,0.74,0.15,0.10,0.08,0.12,0.06,0.07,0.08,0.05,0.09,0.06,0.07,0.10]),
    ("Dynamic Programming: memoization overlapping subproblems", "cs",
        [0.82,0.76,0.88,0.80,0.20,0.18,0.12,0.09,0.07,0.06,0.08,0.07,0.08,0.09,0.06,0.07]),
    ("Graph BFS and DFS: breadth and depth first traversal", "cs",
        [0.85,0.80,0.75,0.82,0.18,0.14,0.10,0.08,0.06,0.09,0.07,0.06,0.10,0.08,0.09,0.07]),
    ("Hash Table: O(1) lookup with collision chaining", "cs",
        [0.87,0.78,0.70,0.76,0.13,0.11,0.09,0.14,0.08,0.07,0.06,0.08,0.07,0.10,0.08,0.09]),
    ("Calculus: derivatives integrals and limits", "math",
        [0.12,0.15,0.18,0.10,0.91,0.86,0.78,0.72,0.08,0.06,0.07,0.09,0.07,0.08,0.06,0.10]),
    ("Linear Algebra: matrices eigenvalues eigenvectors", "math",
        [0.20,0.18,0.15,0.12,0.88,0.90,0.82,0.76,0.09,0.07,0.08,0.06,0.10,0.07,0.08,0.09]),
    ("Probability: distributions random variables Bayes theorem", "math",
        [0.15,0.12,0.20,0.18,0.84,0.80,0.88,0.82,0.07,0.08,0.06,0.10,0.09,0.06,0.09,0.08]),
    ("Number Theory: primes modular arithmetic RSA cryptography", "math",
        [0.22,0.16,0.14,0.20,0.80,0.85,0.76,0.90,0.08,0.09,0.07,0.06,0.08,0.10,0.07,0.06]),
    ("Combinatorics: permutations combinations generating functions", "math",
        [0.18,0.20,0.16,0.14,0.86,0.78,0.84,0.80,0.06,0.07,0.09,0.08,0.06,0.09,0.10,0.07]),
    ("Neapolitan Pizza: wood-fired dough San Marzano tomatoes", "food",
        [0.08,0.06,0.09,0.07,0.07,0.08,0.06,0.09,0.90,0.86,0.78,0.72,0.08,0.06,0.09,0.07]),
    ("Sushi: vinegared rice raw fish and nori rolls", "food",
        [0.06,0.08,0.07,0.09,0.09,0.06,0.08,0.07,0.86,0.90,0.82,0.76,0.07,0.09,0.06,0.08]),
    ("Ramen: noodle soup with chashu pork and soft-boiled eggs", "food",
        [0.09,0.07,0.06,0.08,0.08,0.09,0.07,0.06,0.82,0.78,0.90,0.84,0.09,0.07,0.08,0.06]),
    ("Tacos: corn tortillas with carnitas salsa and cilantro", "food",
        [0.07,0.09,0.08,0.06,0.06,0.07,0.09,0.08,0.78,0.82,0.86,0.90,0.06,0.08,0.07,0.09]),
    ("Croissant: laminated pastry with buttery flaky layers", "food",
        [0.06,0.07,0.10,0.09,0.10,0.06,0.07,0.10,0.85,0.80,0.76,0.82,0.09,0.07,0.10,0.06]),
    ("Basketball: fast-paced shooting dribbling slam dunks", "sports",
        [0.09,0.07,0.08,0.10,0.08,0.09,0.07,0.06,0.08,0.07,0.09,0.06,0.91,0.85,0.78,0.72]),
    ("Football: tackles touchdowns field goals and strategy", "sports",
        [0.07,0.09,0.06,0.08,0.09,0.07,0.10,0.08,0.07,0.09,0.08,0.07,0.87,0.89,0.82,0.76]),
    ("Tennis: racket volleys groundstrokes and Wimbledon serves", "sports",
        [0.08,0.06,0.09,0.07,0.07,0.08,0.06,0.09,0.09,0.06,0.07,0.08,0.83,0.80,0.88,0.82]),
    ("Chess: openings endgames tactics strategic board game", "sports",
        [0.25,0.20,0.22,0.18,0.22,0.18,0.20,0.15,0.06,0.08,0.07,0.09,0.80,0.84,0.78,0.90]),
    ("Swimming: butterfly freestyle backstroke Olympic competition", "sports",
        [0.06,0.08,0.07,0.09,0.08,0.06,0.09,0.07,0.10,0.08,0.06,0.07,0.85,0.82,0.86,0.80]),
]


def demo_items() -> list[VectorItem]:
    """Return the 20 demo vectors as a list of `VectorItem` (no DB needed).

    IDs are 1..20, matching the order they appear in the C++ `loadDemo`.
    Used by the parity tests so the test doesn't need to spin up the DB.
    """
    return [VectorItem(id=i + 1, metadata=m, category=c, emb=list(emb))
            for i, (m, c, emb) in enumerate(_DEMO)]


def load_demo(db: VectorDB) -> None:
    """Insert all 20 demo vectors into the given VectorDB.  Port of main.cpp:717."""
    dist = get_dist_fn("cosine")
    for meta, cat, emb in _DEMO:
        db.insert(meta, cat, list(emb), dist)
