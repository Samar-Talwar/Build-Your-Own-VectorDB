"""Brute-force k-NN search — direct port of main.cpp:70-91.

Stores all items in a Python list and computes distance to every item on
each query.  This is the ground-truth path: any other algorithm must return
results whose IDs and distances match this one (within float precision).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

from .distances import DistFn


@dataclass
class VectorItem:
    """A single vector with metadata.  Port of main.cpp:26-31.

    Used by the demo (16D) index; the same shape feeds HNSW with the
    `category="doc"` sentinel for document chunks.
    """
    id: int
    metadata: str
    category: str
    emb: List[float] = field(default_factory=list)


class BruteForce:
    """Linear-scan k-NN.  Direct port of main.cpp:70-91.

    Returned pairs are (distance, id) ordered by ascending distance, capped
    to `k` entries.  The list is materialized fully before truncation; for
    small N (the demo has 20 items) this is the fastest correct option.
    """

    def __init__(self) -> None:
        self.items: List[VectorItem] = []

    def insert(self, v: VectorItem) -> None:
        self.items.append(v)

    def knn(self, q: List[float], k: int, dist: DistFn) -> List[Tuple[float, int]]:
        scored = [(dist(q, v.emb), v.id) for v in self.items]
        scored.sort()
        if len(scored) > k:
            scored = scored[:k]
        return scored

    def remove(self, id_: int) -> None:
        self.items = [v for v in self.items if v.id != id_]
