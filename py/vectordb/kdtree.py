"""KD-Tree for k-NN search — direct port of main.cpp:97-159.

A naive unbalanced axis-rotating KD-Tree.  Inherits the C++'s degenerate
behaviour on adversarial input (O(N) per query) — preserved deliberately
per risk R5 in docs/migration/08-risks-and-decisions.md.  For the demo's
20 well-distributed 16D vectors this never triggers.
"""

from __future__ import annotations

import heapq
from typing import List, Optional, Tuple

from .bruteforce import VectorItem
from .distances import DistFn


class _KDNode:
    __slots__ = ("item", "left", "right")

    def __init__(self, item: VectorItem) -> None:
        self.item = item
        self.left: Optional["_KDNode"] = None
        self.right: Optional["_KDNode"] = None


class KDTree:
    """Unbalanced axis-rotating KD-Tree.  Port of main.cpp:104-159.

    `knn` returns the k nearest items by the supplied distance function.
    The C++ uses a max-heap of size k during traversal; the port uses
    `heapq` (a min-heap) over negated distances — same effect, but it
    lets us cap memory strictly and avoid an explicit pop/push for
    "should I keep this candidate?" decisions.
    """

    def __init__(self, dims: int) -> None:
        self.dims = dims
        self._root: Optional[_KDNode] = None

    @staticmethod
    def _destroy(n: Optional[_KDNode]) -> None:
        if n is None:
            return
        KDTree._destroy(n.left)
        KDTree._destroy(n.right)

    def __del__(self) -> None:
        KDTree._destroy(self._root)

    def _ins(self, n: Optional[_KDNode], v: VectorItem, d: int) -> _KDNode:
        if n is None:
            return _KDNode(v)
        ax = d % self.dims
        if v.emb[ax] < n.item.emb[ax]:
            n.left = self._ins(n.left, v, d + 1)
        else:
            n.right = self._ins(n.right, v, d + 1)
        return n

    def insert(self, v: VectorItem) -> None:
        self._root = self._ins(self._root, v, 0)

    def _knn(
        self,
        n: Optional[_KDNode],
        q: List[float],
        k: int,
        d: int,
        dist: DistFn,
        heap: List[Tuple[float, int]],
    ) -> None:
        if n is None:
            return
        dn = dist(q, n.item.emb)
        if len(heap) < k or dn < -heap[0][0]:
            heapq.heappush(heap, (-dn, n.item.id))
            if len(heap) > k:
                heapq.heappop(heap)
        ax = d % self.dims
        diff = q[ax] - n.item.emb[ax]
        closer = n.left if diff < 0 else n.right
        farther = n.right if diff < 0 else n.left
        self._knn(closer, q, k, d + 1, dist, heap)
        if len(heap) < k or abs(diff) < -heap[0][0]:
            self._knn(farther, q, k, d + 1, dist, heap)

    def knn(self, q: List[float], k: int, dist: DistFn) -> List[Tuple[float, int]]:
        heap: List[Tuple[float, int]] = []
        self._knn(self._root, q, k, 0, dist, heap)
        # Drain the heap into a list sorted ascending by distance (matches
        # the C++ which builds the same shape at main.cpp:149-152).
        out = sorted(((-neg_d, i) for neg_d, i in heap), key=lambda t: t[0])
        return out

    def rebuild(self, items: List[VectorItem]) -> None:
        KDTree._destroy(self._root)
        self._root = None
        for v in items:
            self.insert(v)
