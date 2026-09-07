"""HNSW — hand port of main.cpp:165-331.

The hnswlib package was considered (per the migration's R1 risk and the
original D1 decision), but its neighbor-selection heuristic and RNG drift
the /hnsw-info graph shape away from the C++'s.  The frontend renders
this graph, so we hand-port to preserve the visual layout bit-exact.

The C++ exhibits two documented quirks (R6, R7) that are preserved
verbatim here:
  * When the entry point is deleted, topLayer is not decremented.
  * topLayer only ever increases across inserts.

These are not bugs in the port — they are the contract.
"""

from __future__ import annotations

import heapq
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from .bruteforce import VectorItem
from .distances import DistFn


@dataclass
class _Node:
    item: VectorItem
    max_lyr: int
    nbrs: List[List[int]] = field(default_factory=list)


@dataclass
class NodeInfo:
    id: int
    metadata: str
    category: str
    max_lyr: int


@dataclass
class EdgeInfo:
    src: int
    dst: int
    lyr: int


@dataclass
class GraphInfo:
    top_layer: int
    node_count: int
    nodes_per_layer: List[int]
    edges_per_layer: List[int]
    nodes: List[NodeInfo]
    edges: List[EdgeInfo]


class HNSW:
    """Hierarchical Navigable Small World.  Direct port of main.cpp:165-331.

    Construction parameters: M=16, M0=32, ef_build=200, mL = 1/ln(M).
    Search ef defaults to 50 (matches the call sites in VectorDB).
    """

    def __init__(self, m: int = 16, ef_build: int = 200) -> None:
        self.M = m
        self.M0 = 2 * m
        self.ef_build = ef_build
        self.mL = 1.0 / math.log(float(m))
        self._top_layer: int = -1
        self._entry_pt: int = -1
        self._g: Dict[int, _Node] = {}
        # Seeded for determinism (matches main.cpp:229).  The parity
        # test asserts the C++ and the Python build the same graph.
        self._rng = random.Random(42)

    def _rand_level(self) -> int:
        return int(math.floor(-math.log(self._rng.random()) * self.mL))

    def _search_layer(
        self,
        q: List[float],
        ep: int,
        ef: int,
        lyr: int,
        dist: DistFn,
    ) -> List[Tuple[float, int]]:
        """Greedy layer-restricted search.  Port of main.cpp:184-217."""
        vis: Dict[int, bool] = {ep: True}
        d0 = dist(q, self._g[ep].item.emb)
        cands: List[Tuple[float, int]] = [(d0, ep)]
        # C++ uses a max-heap for `found`; we negate to use heapq min-heap.
        found: List[Tuple[float, int]] = [(-d0, ep)]
        while cands:
            cd, cid = heapq.heappop(cands)
            if len(found) >= ef and cd > -found[0][0]:
                break
            node = self._g[cid]
            if lyr >= len(node.nbrs):
                continue
            for nid in node.nbrs[lyr]:
                if vis.get(nid) or nid not in self._g:
                    continue
                vis[nid] = True
                nd = dist(q, self._g[nid].item.emb)
                if len(found) < ef or nd < -found[0][0]:
                    heapq.heappush(cands, (nd, nid))
                    heapq.heappush(found, (-nd, nid))
                    if len(found) > ef:
                        heapq.heappop(found)
        # Drain `found` into ascending-by-distance list.
        return sorted(((-neg_d, i) for neg_d, i in found), key=lambda t: t[0])

    @staticmethod
    def _select_nbrs(cands: List[Tuple[float, int]], max_m: int) -> List[int]:
        return [c[1] for c in cands[:max_m]]

    def insert(self, item: VectorItem, dist: DistFn) -> None:
        id_ = item.id
        lvl = self._rand_level()
        self._g[id_] = _Node(item=item, max_lyr=lvl, nbrs=[[] for _ in range(lvl + 1)])

        if self._entry_pt == -1:
            self._entry_pt = id_
            self._top_layer = lvl
            return

        ep = self._entry_pt
        for lc in range(self._top_layer, lvl, -1):
            if lc < len(self._g[ep].nbrs):
                w = self._search_layer(item.emb, ep, 1, lc, dist)
                if w:
                    ep = w[0][1]
        for lc in range(min(self._top_layer, lvl), -1, -1):
            w = self._search_layer(item.emb, ep, self.ef_build, lc, dist)
            max_m = self.M0 if lc == 0 else self.M
            sel = self._select_nbrs(w, max_m)
            self._g[id_].nbrs[lc] = sel
            for nid in sel:
                if nid not in self._g:
                    continue
                nbr_node = self._g[nid]
                if len(nbr_node.nbrs) <= lc:
                    nbr_node.nbrs.extend([[] for _ in range(lc + 1 - len(nbr_node.nbrs))])
                conn = nbr_node.nbrs[lc]
                conn.append(id_)
                if len(conn) > max_m:
                    ds = []
                    for c in conn:
                        if c in self._g:
                            ds.append((dist(nbr_node.item.emb, self._g[c].item.emb), c))
                    ds.sort()
                    conn.clear()
                    conn.extend(c for _, c in ds[:max_m])
            if w:
                ep = w[0][1]
        if lvl > self._top_layer:
            self._top_layer = lvl
            self._entry_pt = id_

    def knn(
        self,
        q: List[float],
        k: int,
        ef: int,
        dist: DistFn,
    ) -> List[Tuple[float, int]]:
        if self._entry_pt == -1:
            return []
        ep = self._entry_pt
        for lc in range(self._top_layer, 0, -1):
            if lc < len(self._g[ep].nbrs):
                w = self._search_layer(q, ep, 1, lc, dist)
                if w:
                    ep = w[0][1]
        w = self._search_layer(q, ep, max(ef, k), 0, dist)
        if len(w) > k:
            w = w[:k]
        return w

    def remove(self, id_: int) -> None:
        if id_ not in self._g:
            return
        for nid, nd in self._g.items():
            for layer in nd.nbrs:
                if id_ in layer:
                    layer[:] = [x for x in layer if x != id_]
        if self._entry_pt == id_:
            self._entry_pt = -1
            for nid in self._g:
                if nid != id_:
                    self._entry_pt = nid
                    break
            # R6 preserved: topLayer is NOT decremented.
        del self._g[id_]

    def get_info(self) -> GraphInfo:
        max_l = max(self._top_layer + 1, 1)
        nodes_per_layer = [0] * max_l
        edges_per_layer = [0] * max_l
        nodes: List[NodeInfo] = []
        edges: List[EdgeInfo] = []
        for id_, nd in self._g.items():
            nodes.append(NodeInfo(
                id=id_,
                metadata=nd.item.metadata,
                category=nd.item.category,
                max_lyr=nd.max_lyr,
            ))
            # Each node is counted at its max_lyr layer (matches C++).
            top = min(nd.max_lyr, max_l - 1)
            nodes_per_layer[top] += 1
            for lc in range(top + 1):
                if lc < len(nd.nbrs):
                    for nid in nd.nbrs[lc]:
                        if id_ < nid:
                            edges_per_layer[lc] += 1
                            edges.append(EdgeInfo(src=id_, dst=nid, lyr=lc))
        return GraphInfo(
            top_layer=self._top_layer,
            node_count=len(self._g),
            nodes_per_layer=nodes_per_layer,
            edges_per_layer=edges_per_layer,
            nodes=nodes,
            edges=edges,
        )

    def size(self) -> int:
        return len(self._g)
