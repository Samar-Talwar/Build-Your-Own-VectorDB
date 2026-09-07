"""In-memory 16D vector store wrapping the three search algorithms.

Port of main.cpp:337-429 (the `VectorDB` class).  Holds all three indexes
(BruteForce, KD-Tree, HNSW) and keeps them in sync on every insert /
remove.  The single-threaded Flask dev server does not need a lock
(matches C++ semantics per D9 in 08-risks-and-decisions.md), but the
class is built so a lock could be added with a single line of code.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .bruteforce import BruteForce, VectorItem
from .distances import DistFn, get_dist_fn
from .hnsw import HNSW, GraphInfo
from .kdtree import KDTree


@dataclass
class Hit:
    """One search result, with the metadata needed by the JSON encoder."""
    id: int
    metadata: str
    category: str
    emb: List[float]
    dist: float


@dataclass
class SearchOut:
    hits: List[Hit] = field(default_factory=list)
    us: int = 0
    algo: str = ""
    metric: str = ""


@dataclass
class BenchOut:
    bf_us: int
    kd_us: int
    hnsw_us: int
    n: int


class VectorDB:
    """The 16D demo vector database.  Port of main.cpp:337-429.

    `nextId` is a monotonically increasing 1-indexed counter shared
    across all algorithms (matches the C++ single `int nextId` field).
    """

    def __init__(self, dims: int) -> None:
        self.dims = dims
        self._store: Dict[int, VectorItem] = {}
        self._bf = BruteForce()
        self._kdt = KDTree(dims)
        self._hnsw = HNSW(m=16, ef_build=200)
        self._next_id: int = 1

    def insert(self, meta: str, cat: str, emb: List[float], dist: DistFn) -> int:
        v = VectorItem(id=self._next_id, metadata=meta, category=cat, emb=list(emb))
        self._next_id += 1
        self._store[v.id] = v
        self._bf.insert(v)
        self._kdt.insert(v)
        self._hnsw.insert(v, dist)
        return v.id

    def remove(self, id_: int) -> bool:
        if id_ not in self._store:
            return False
        del self._store[id_]
        self._bf.remove(id_)
        self._hnsw.remove(id_)
        # KD-Tree is rebuilt from scratch on every delete — same as the
        # C++ which does this at main.cpp:362-365.
        self._kdt.rebuild(list(self._store.values()))
        return True

    def search(
        self,
        q: List[float],
        k: int,
        metric: str,
        algo: str,
    ) -> SearchOut:
        dfn = get_dist_fn(metric)
        t0 = time.perf_counter_ns()
        if algo == "bruteforce":
            raw = self._bf.knn(q, k, dfn)
        elif algo == "kdtree":
            raw = self._kdt.knn(q, k, dfn)
        else:
            raw = self._hnsw.knn(q, k, 50, dfn)
        us = (time.perf_counter_ns() - t0) // 1000
        out = SearchOut(us=us, algo=algo, metric=metric)
        for d, id_ in raw:
            v = self._store.get(id_)
            if v is not None:
                out.hits.append(Hit(id=v.id, metadata=v.metadata, category=v.category, emb=v.emb, dist=d))
        return out

    def benchmark(self, q: List[float], k: int, metric: str) -> BenchOut:
        dfn = get_dist_fn(metric)
        t0 = time.perf_counter_ns()
        self._bf.knn(q, k, dfn)
        bf_us = (time.perf_counter_ns() - t0) // 1000
        t0 = time.perf_counter_ns()
        self._kdt.knn(q, k, dfn)
        kd_us = (time.perf_counter_ns() - t0) // 1000
        t0 = time.perf_counter_ns()
        self._hnsw.knn(q, k, 50, dfn)
        hnsw_us = (time.perf_counter_ns() - t0) // 1000
        return BenchOut(bf_us=bf_us, kd_us=kd_us, hnsw_us=hnsw_us, n=len(self._store))

    def all(self) -> List[VectorItem]:
        return list(self._store.values())

    def hnsw_info(self) -> GraphInfo:
        return self._hnsw.get_info()

    def size(self) -> int:
        return len(self._store)
