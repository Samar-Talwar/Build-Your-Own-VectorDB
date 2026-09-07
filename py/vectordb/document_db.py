"""Document database for chunked + embedded text.

Port of main.cpp:644-711 (`DocItem` + `DocumentDB`).  Each chunk is
embedded to a fixed dimension (typically 768 for `nomic-embed-text`)
and stored with its source text.  Search uses the HNSW index unless
the store has fewer than 10 items, in which case it falls back to
brute force (matches the C++ at main.cpp:682-684).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from .bruteforce import BruteForce, VectorItem
from .distances import cosine
from .hnsw import HNSW


@dataclass
class DocItem:
    id: int
    title: str
    text: str
    emb: List[float] = field(default_factory=list)


class DocumentDB:
    """Chunk store with semantic (cosine) search.  Port of main.cpp:651-711.

    `dims` is set lazily on the first insert.  `getDims` has the same
    non-atomic read of `dims` that the C++ has — preserved per R8.
    """

    def __init__(self) -> None:
        self._store: Dict[int, DocItem] = {}
        self._hnsw = HNSW(m=16, ef_build=200)
        self._bf = BruteForce()
        self._next_id: int = 1
        self.dims: int = 0

    def insert(self, title: str, text: str, emb: List[float]) -> int:
        if self.dims == 0:
            self.dims = len(emb)
        item = DocItem(id=self._next_id, title=title, text=text, emb=list(emb))
        self._next_id += 1
        self._store[item.id] = item
        # Insert into both indexes; HNSW + a BF mirror for the small-N path.
        vi = VectorItem(id=item.id, metadata=title, category="doc", emb=list(emb))
        self._hnsw.insert(vi, cosine)
        self._bf.insert(vi)
        return item.id

    def search(self, q: List[float], k: int, max_dist: float = 0.7) -> List[Tuple[float, DocItem]]:
        if not self._store:
            return []
        raw = self._bf.knn(q, k, cosine) if len(self._store) < 10 \
            else self._hnsw.knn(q, k, 50, cosine)
        out: List[Tuple[float, DocItem]] = []
        for d, id_ in raw:
            doc = self._store.get(id_)
            if doc is not None and d <= max_dist:
                out.append((d, doc))
        return out

    def remove(self, id_: int) -> bool:
        if id_ not in self._store:
            return False
        del self._store[id_]
        self._hnsw.remove(id_)
        self._bf.remove(id_)
        return True

    def all(self) -> List[DocItem]:
        return list(self._store.values())

    def size(self) -> int:
        return len(self._store)

    def get_dims(self) -> int:
        # R8: read of `dims` is not synchronized.  Matches C++.
        return self.dims
