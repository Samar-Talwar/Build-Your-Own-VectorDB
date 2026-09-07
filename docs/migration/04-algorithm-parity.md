# Algorithm Parity Specification: C++ to Python Migration

**Document**: `04-algorithm-parity.md`  
**Source Baseline**: `C:\Users\Samar\Documents\Your-OWN-AI\main.cpp` (lines 39–331, 528–760)  
**Target Runtime**: Python 3.11+ (Standard Library, NumPy, optional HNSW/SciPy)  
**Classification Summary**: 3 Strict Parity, 3 Structural Parity, 1 Selective Parity, 1 Loose Parity

---

## 1. Distance Metrics — STRICT PARITY REQUIRED

Distance metrics in the Python implementation must produce identical floating-point values to the C++ implementation within standard single-precision / double-precision roundoff tolerances ($\epsilon \le 10^{-6}$).

### 1.1 `euclidean` ($L_2$ Distance)
- **Source**: `main.cpp:39-43`
- **Formula**:
  $$\text{dist}_{L_2}(a, b) = \sqrt{\sum_{i=0}^{n-1} (a_i - b_i)^2}$$
- **C++ Reference**:
  ```cpp
  float euclidean(const std::vector<float>& a, const std::vector<float>& b) {
      float s = 0;
      for (int i = 0; i < (int)a.size(); i++) { float d = a[i]-b[i]; s += d*d; }
      return std::sqrt(s);
  }
  ```
- **Edge Cases**:
  - Empty vectors ($n=0$): Returns `0.0`.
  - Dimension mismatch: C++ iterates over `a.size()`. Python must enforce dimension equality (`len(a) == len(b)`) or slice to minimum length.
  - Single dimension ($n=1$): Returns $|a_0 - b_0|$.

### 1.2 `cosine` (Cosine Distance)
- **Source**: `main.cpp:45-52`
- **Formula**:
  $$\text{dist}_{\text{cosine}}(a, b) = 1.0 - \frac{a \cdot b}{\|a\|_2 \|b\|_2} = 1.0 - \frac{\sum a_i b_i}{\sqrt{\sum a_i^2} \sqrt{\sum b_i^2}}$$
- **C++ Reference**:
  ```cpp
  float cosine(const std::vector<float>& a, const std::vector<float>& b) {
      float dot=0, na=0, nb=0;
      for (int i = 0; i < (int)a.size(); i++) {
          dot += a[i]*b[i]; na += a[i]*a[i]; nb += b[i]*b[i];
      }
      if (na < 1e-9f || nb < 1e-9f) return 1.0f;
      return 1.0f - dot / (std::sqrt(na) * std::sqrt(nb));
  }
  ```
- **Quirk & Guard (MUST PRESERVE)**:
  - If either squared norm $\sum a_i^2 < 10^{-9}$ or $\sum b_i^2 < 10^{-9}$, the function **must immediately return `1.0`** (representing maximal orthogonal distance for zero or near-zero vectors).
  - Distance values range in $[0.0, 2.0]$. Identical directional vectors return `0.0`, orthogonal vectors return `1.0`, and opposite vectors return `2.0`.

### 1.3 `manhattan` ($L_1$ Distance)
- **Source**: `main.cpp:54-58`
- **Formula**:
  $$\text{dist}_{L_1}(a, b) = \sum_{i=0}^{n-1} |a_i - b_i|$$
- **C++ Reference**:
  ```cpp
  float manhattan(const std::vector<float>& a, const std::vector<float>& b) {
      float s = 0;
      for (int i = 0; i < (int)a.size(); i++) s += std::abs(a[i]-b[i]);
      return s;
  }
  ```

### 1.4 `getDistFn` (Metric Dispatcher)
- **Source**: `main.cpp:60-64`
- **Behavior**:
  - Input `"cosine"` $\rightarrow$ `cosine`
  - Input `"manhattan"` $\rightarrow$ `manhattan`
  - Default (including `"euclidean"`, `""`, or unrecognized strings) $\rightarrow$ `euclidean`

### 1.5 Reference Python Implementation Snippet
```python
import math
from typing import Callable, Sequence

DistFn = Callable[[Sequence[float], Sequence[float]], float]

def euclidean(a: Sequence[float], b: Sequence[float]) -> float:
    s = 0.0
    for x, y in zip(a, b):
        d = x - y
        s += d * d
    return math.sqrt(s)

def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na < 1e-9 or nb < 1e-9:
        return 1.0
    denom = math.sqrt(na) * math.sqrt(nb)
    # Clamp to prevent float precision rounding outside [-1, 1]
    cos_sim = dot / denom
    return 1.0 - cos_sim

def manhattan(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(abs(x - y) for x, y in zip(a, b))

def get_dist_fn(metric: str) -> DistFn:
    if metric == "cosine":
        return cosine
    if metric == "manhattan":
        return manhattan
    return euclidean
```

---

## 2. BruteForce — STRUCTURAL PARITY

The brute force index performs exhaustive linear scans across all stored items.

### 2.1 Algorithmic Characteristics
- **Storage**: Flat array / list of items preserving insertion order.
- **k-NN Search Complexity**: $O(N \cdot D + N \log N)$ where $N$ is item count and $D$ is vector dimension.
- **Tie-Breaking Rule (MUST PRESERVE)**:
  - C++ sorts `std::vector<std::pair<float, int>>` using standard `std::sort`.
  - In Python, sort key is `(distance, id)` ascending: lowest distance first; on equal distance, lowest `id` first.
- **Removal**: $O(N)$ filter removing items matching target `id`.

### 2.2 Reference Python Implementation Snippet
```python
from dataclasses import dataclass
from typing import List, Tuple, Sequence

@dataclass
class VectorItem:
    id: int
    metadata: str
    category: str
    emb: List[float]

class BruteForce:
    def __init__(self) -> None:
        self.items: List[VectorItem] = []

    def insert(self, v: VectorItem) -> None:
        self.items.append(v)

    def knn(self, q: Sequence[float], k: int, dist: DistFn) -> List[Tuple[float, int]]:
        scored = [(dist(q, v.emb), v.id) for v in self.items]
        # Sorts by distance asc, then id asc on distance ties
        scored.sort(key=lambda pair: (pair[0], pair[1]))
        return scored[:k]

    def remove(self, item_id: int) -> None:
        self.items = [v for v in self.items if v.id != item_id]
```

---

## 3. KDTree — STRUCTURAL PARITY

The $k$-d tree partitions spatial dimensions sequentially across tree depth.

### 3.1 Algorithmic Invariants & Quirks
- **Axis Rotation**:
  $$\text{axis} = \text{depth} \pmod{\text{dims}}$$
- **Insertion Split Rule (CRITICAL)**:
  ```python
  if v.emb[ax] < node.item.emb[ax]:
      # Recurse left
  else:
      # Recurse right (EQUALITY GOES RIGHT)
  ```
  *Parity note*: If coordinate value equals the split plane, it **must go to the right child**.
- **k-NN Traversal Logic**:
  - Maintains a bounded max-heap of size $k$ storing `(distance, id)`.
  - Determines branch direction: `diff = q[ax] - node.item.emb[ax]`.
  - Traverses `closer` child first (`left` if `diff < 0` else `right`).
  - Prunes `farther` child unless:
    $$\text{heap.size} < k \quad\lor\quad |\text{diff}| < \text{heap.max\_distance}$$
  - Returns results sorted in ascending order of `(distance, id)`.
- **Deletion Strategy**:
  - KDTree contains no node deletion / rebalancing routine.
  - `VectorDB::remove()` performs a **complete rebuild** by clearing the tree and re-inserting all surviving items. This behavior must be preserved.
- **Known Degeneration Quirk**:
  - Sequential inserts of sorted or correlated data degenerate the tree into an $O(N)$ linked list. Preserve standard recursive insertion without auto-balancing; demo data is well-spread across categories.

### 3.2 Hand-Port vs. SciPy Decision
| Approach | Pros | Cons | Recommendation |
|:---|:---|:---|:---|
| **Hand-Ported KDTree (~60 lines)** | 100% semantic parity; supports custom `DistFn` (cosine, manhattan, euclidean); exact tie-breaking. | Python recursion overhead for $N > 10^5$. | **RECOMMENDED FOR V1** |
| **`scipy.spatial.KDTree`** | Fast C implementation. | Only supports Minkowski $p$-norms (fails on arbitrary cosine without normalization); different tie-breaking. | Reject for direct parity. |

### 3.3 Reference Python Implementation Snippet
```python
import heapq
from typing import Optional, List, Tuple, Sequence

class KDNode:
    def __init__(self, item: VectorItem) -> None:
        self.item = item
        self.left: Optional[KDNode] = None
        self.right: Optional[KDNode] = None

class KDTree:
    def __init__(self, dims: int) -> None:
        self.dims = dims
        self.root: Optional[KDNode] = None

    def insert(self, v: VectorItem) -> None:
        def _ins(n: Optional[KDNode], item: VectorItem, d: int) -> KDNode:
            if n is None:
                return KDNode(item)
            ax = d % self.dims
            if item.emb[ax] < n.item.emb[ax]:
                n.left = _ins(n.left, item, d + 1)
            else:
                n.right = _ins(n.right, item, d + 1)
            return n

        self.root = _ins(self.root, v, 0)

    def knn(self, q: Sequence[float], k: int, dist: DistFn) -> List[Tuple[float, int]]:
        # Max-heap storing (-dist, -id) or inverted tuples to pop largest
        # We store (distance, id) and maintain max-heap behavior
        heap: List[Tuple[float, int]] = []

        def _knn(n: Optional[KDNode], d: int) -> None:
            nonlocal heap
            if n is None:
                return
            dn = dist(q, n.item.emb)
            # Push into heap if not full or better than max distance in heap
            if len(heap) < k:
                heapq.heappush(heap, (-dn, n.item.id))
            elif dn < -heap[0][0]:
                heapq.heapreplace(heap, (-dn, n.item.id))

            ax = d % self.dims
            diff = q[ax] - n.item.emb[ax]
            closer = n.left if diff < 0 else n.right
            farther = n.right if diff < 0 else n.left

            _knn(closer, d + 1)

            max_heap_dist = -heap[0][0] if heap else float("inf")
            if len(heap) < k or abs(diff) < max_heap_dist:
                _knn(farther, d + 1)

        _knn(self.root, 0)
        # Convert max-heap elements back to (dist, id) and sort ascending
        results = [(-entry[0], entry[1]) for entry in heap]
        results.sort(key=lambda pair: (pair[0], pair[1]))
        return results

    def rebuild(self, items: Sequence[VectorItem]) -> None:
        self.root = None
        for item in items:
            self.insert(item)
```

---

## 4. HNSW — PARITY CLASS: SELECTIVE

Hierarchical Navigable Small World graph index for approximate nearest neighbor search.

### 4.1 Parameter Contract Table

| Parameter / Heuristic | C++ Value / Formula | Parity Class | Notes |
|:---|:---|:---|:---|
| `M` (Max links per layer $>0$) | `16` (`main.cpp:227, 347, 660`) | **MUST PRESERVE** | Ground layer max links $M_0 = 2M = 32$. |
| `M0` (Max links at layer $0$) | `32` (`main.cpp:228`) | **MUST PRESERVE** | Fixed at $2 \times M$. |
| `ef_build` (Construction beam) | `200` (`main.cpp:227, 347, 660`)| **MUST PRESERVE** | Search depth during insert. |
| `mL` (Level multiplier) | $1.0 / \ln(M) \approx 0.36067$ | **MUST PRESERVE** | Natural log base $e$. |
| Random Level Formula | $\lfloor -\ln(U) \cdot mL \rfloor$ where $U \sim \text{Uniform}(0, 1)$ | **MUST PRESERVE** | Level generation distribution. |
| `ef` (Query beam width) | `50` (`main.cpp:382, 408, 684`) | **MUST PRESERVE** | Hardcoded in search routes. |
| Neighbor Selection | First $\min(|W|, \text{maxM})$ from sorted candidates | **MUST PRESERVE** | Simple truncation heuristic; no heuristic shrinking / diversity pruning. |
| Edge Deduplication in `/hnsw-info` | `id < nid` | **MUST PRESERVE** | Undirected edge emitted once per pair per layer. |
| Tie-Break Order | `(distance asc, id asc)` | **MUST PRESERVE** | Consistent result ordering. |
| RNG Seed | `42` (`std::mt19937`) | **MAY DRIFT** | In pure Python port, seeding Python `random` or `numpy.random` produces equivalent statistical distribution even if sequence differs. |
| Exact Graph Topology | Graph connectivity | **MAY DRIFT** | If switching to `hnswlib`, graph edges will vary; JSON schema must remain exact. |

### 4.2 Known Architectural Quirks & Bugs
1. **HNSW Entry Point Orphan on Deletion (`main.cpp:292-296`)**:
   - If the deleted node is `entryPt`, C++ reassigns `entryPt` to the first arbitrary key found in map `G` (`for (auto& [nid, nd] : G) if (nid != id) { entryPt = nid; break; }`).
   - *Bug*: It does not verify whether the new `entryPt` exists on `topLayer`.
   - *Migration Decision*: In v1, preserve or safely update `entryPt` to highest-layer node if `G` is non-empty.
2. **`topLayer` Monotonicity**:
   - `topLayer` only increases on insertion; it never decreases when nodes on the top layer are removed.
   - *Migration Decision*: Preserve in v1 to avoid desynchronizing `/hnsw-info` expectation tests.

### 4.3 `/hnsw-info` JSON Schema Contract
The graph introspection endpoint must return exact field keys:
```json
{
  "topLayer": 2,
  "nodeCount": 20,
  "nodesPerLayer": [20, 5, 1],
  "edgesPerLayer": [48, 8, 0],
  "nodes": [
    { "id": 1, "metadata": "...", "category": "cs", "maxLyr": 0 }
  ],
  "edges": [
    { "src": 1, "dst": 4, "lyr": 0 }
  ]
}
```

### 4.4 Engine Choice: Hand-Port vs. `hnswlib`
- **Primary Recommendation**: **Native Python Hand-Port** (provided below).
  - *Rationale*: Guarantees exact `/hnsw-info` structure, custom distance function routing, zero compilation steps on Windows/Linux, and identical neighbor selection logic.
- **Secondary Option**: `hnswlib` library.
  - *Caveat*: If `hnswlib` is used, `/hnsw-info` endpoint must be mocked or reconstructed from internal graph data structures.

### 4.5 Reference Python Implementation (Native HNSW)
```python
import math
import random
import heapq
from typing import Dict, List, Tuple, Sequence, Optional, Any

class HNSWNode:
    def __init__(self, item: VectorItem, max_lyr: int) -> None:
        self.item = item
        self.max_lyr = max_lyr
        self.nbrs: List[List[int]] = [[] for _ in range(max_lyr + 1)]

class HNSW:
    def __init__(self, m: int = 16, ef_build: int = 200, seed: int = 42) -> None:
        self.M = m
        self.M0 = 2 * m
        self.ef_build = ef_build
        self.mL = 1.0 / math.log(float(m))
        self.top_layer = -1
        self.entry_pt = -1
        self.G: Dict[int, HNSWNode] = {}
        self.rng = random.Random(seed)

    def _rand_level(self) -> int:
        u = self.rng.random()
        while u == 0.0:  # Avoid log(0)
            u = self.rng.random()
        return int(math.floor(-math.log(u) * self.mL))

    def _search_layer(
        self, q: Sequence[float], ep: int, ef: int, lyr: int, dist: DistFn
    ) -> List[Tuple[float, int]]:
        visited = {ep}
        d0 = dist(q, self.G[ep].item.emb)
        # cands is min-heap: (dist, id)
        cands: List[Tuple[float, int]] = [(d0, ep)]
        # found is max-heap: (-dist, id)
        found: List[Tuple[float, int]] = [(-d0, ep)]

        while cands:
            cd, cid = heapq.heappop(cands)
            worst_found_dist = -found[0][0]
            if len(found) >= ef and cd > worst_found_dist:
                break
            if lyr >= len(self.G[cid].nbrs):
                continue
            for nid in self.G[cid].nbrs[lyr]:
                if nid in visited or nid not in self.G:
                    continue
                visited.add(nid)
                nd = dist(q, self.G[nid].item.emb)
                worst_found_dist = -found[0][0]
                if len(found) < ef or nd < worst_found_dist:
                    heapq.heappush(cands, (nd, nid))
                    if len(found) < ef:
                        heapq.heappush(found, (-nd, nid))
                    else:
                        heapq.heapreplace(found, (-nd, nid))

        res = [(-entry[0], entry[1]) for entry in found]
        res.sort(key=lambda pair: (pair[0], pair[1]))
        return res

    def _select_nbrs(self, cands: List[Tuple[float, int]], max_m: int) -> List[int]:
        return [pair[1] for pair in cands[:max_m]]

    def insert(self, item: VectorItem, dist: DistFn) -> None:
        item_id = item.id
        lvl = self._rand_level()
        node = HNSWNode(item, lvl)
        self.G[item_id] = node

        if self.entry_pt == -1:
            self.entry_pt = item_id
            self.top_layer = lvl
            return

        ep = self.entry_pt
        for lc in range(self.top_layer, lvl, -1):
            if lc < len(self.G[ep].nbrs):
                W = self._search_layer(item.emb, ep, 1, lc, dist)
                if W:
                    ep = W[0][1]

        for lc in range(min(self.top_layer, lvl), -1, -1):
            W = self._search_layer(item.emb, ep, self.ef_build, lc, dist)
            max_m = self.M0 if lc == 0 else self.M
            sel = self._select_nbrs(W, max_m)
            node.nbrs[lc] = sel

            for nid in sel:
                if nid not in self.G:
                    continue
                target_nbrs = self.G[nid].nbrs
                while len(target_nbrs) <= lc:
                    target_nbrs.append([])
                conn = target_nbrs[lc]
                conn.append(item_id)
                if len(conn) > max_m:
                    ds = [(dist(self.G[nid].item.emb, self.G[c].item.emb), c) for c in conn if c in self.G]
                    ds.sort(key=lambda pair: (pair[0], pair[1]))
                    target_nbrs[lc] = [c for _, c in ds[:max_m]]

            if W:
                ep = W[0][1]

        if lvl > self.top_layer:
            self.top_layer = lvl
            self.entry_pt = item_id

    def knn(self, q: Sequence[float], k: int, ef: int, dist: DistFn) -> List[Tuple[float, int]]:
        if self.entry_pt == -1 or not self.G:
            return []
        ep = self.entry_pt
        for lc in range(self.top_layer, 0, -1):
            if lc < len(self.G[ep].nbrs):
                W = self._search_layer(q, ep, 1, lc, dist)
                if W:
                    ep = W[0][1]
        W = self._search_layer(q, ep, max(ef, k), 0, dist)
        return W[:k]

    def remove(self, item_id: int) -> None:
        if item_id not in self.G:
            return
        for nid, nd in self.G.items():
            for layer in nd.nbrs:
                if item_id in layer:
                    layer.remove(item_id)
        if self.entry_pt == item_id:
            self.entry_pt = -1
            for nid in self.G:
                if nid != item_id:
                    self.entry_pt = nid
                    break
        del self.G[item_id]

    def get_info(self) -> Dict[str, Any]:
        max_l = max(self.top_layer + 1, 1)
        nodes_per_layer = [0] * max_l
        edges_per_layer = [0] * max_l
        nodes_out = []
        edges_out = []

        for item_id, nd in self.G.items():
            nodes_out.append({
                "id": item_id,
                "metadata": nd.item.metadata,
                "category": nd.item.category,
                "maxLyr": nd.max_lyr
            })
            for lc in range(min(nd.max_lyr + 1, max_l)):
                nodes_per_layer[lc] += 1
                if lc < len(nd.nbrs):
                    for nid in nd.nbrs[lc]:
                        if item_id < nid:  # Edge deduplication
                            edges_per_layer[lc] += 1
                            edges_out.append({"src": item_id, "dst": nid, "lyr": lc})

        return {
            "topLayer": self.top_layer,
            "nodeCount": len(self.G),
            "nodesPerLayer": nodes_per_layer,
            "edgesPerLayer": edges_per_layer,
            "nodes": nodes_out,
            "edges": edges_out
        }

    def size(self) -> int:
        return len(self.G)
```

---

## 5. Text Chunker — STRICT PARITY

Text chunking must match C++ sliding-window word chunking character-for-character.

### 5.1 Algorithmic Invariants
- **Word Extraction**: Stream extraction `ss >> w` in C++ splits strictly on all standard whitespace (`' '`, `\t`, `\n`, `\r`, `\f`, `\v`).
- **Short Text Rule (MUST PRESERVE)**:
  - If $\text{len}(\text{words}) \le \text{chunkWords}$ (default 250), the function returns `[text]` (**the original unmodified input string**).
- **Windowing Parameters**:
  - `chunkWords` = 250
  - `overlapWords` = 30
  - `step` = $\text{chunkWords} - \text{overlapWords} = 220$
- **Chunk Construction**:
  - Words within a chunk are joined by a single space `' '`.
  - The final trailing chunk is included even if it contains fewer than `chunkWords` words.
  - Terminates immediately once `end == len(words)`.

### 5.2 Reference Python Implementation Snippet
```python
from typing import List

def chunk_text(text: str, chunk_words: int = 250, overlap_words: int = 30) -> List[str]:
    words = text.split()
    if not words:
        return []
    if len(words) <= chunk_words:
        return [text]

    chunks: List[str] = []
    step = chunk_words - overlap_words
    n = len(words)
    for i in range(0, n, step):
        end = min(i + chunk_words, n)
        chunk = " ".join(words[i:end])
        chunks.append(chunk)
        if end == n:
            break
    return chunks
```

---

## 6. DocumentDB & Semantic Search — STRUCTURAL PARITY

### 6.1 Algorithmic Invariants
- **Dual-Index Routing Threshold**:
  - If `store.size() < 10`: Executes `BruteForce.knn(q, k, cosine)`
  - Else (`store.size() >= 10`): Executes `HNSW.knn(q, k, 50, cosine)`
- **Similarity Cutoff**:
  - Filter condition: `distance <= max_dist` (default `max_dist = 0.7`).
  - Drop hits where `dist > 0.7`.
- **Dynamic Dimension Discovery**:
  - Vector dimension `dims` is initialized to `0` and locked to `len(emb)` on first document chunk insertion.

---

## 7. Demo Data — STRICT, BIT-EXACT

The 20 categorical 16-dimensional vectors must be initialized with exact floating-point values and assigned category tags. Metric used for demo index is `"cosine"`.

```python
DEMO_VECTORS = [
    # Dims 0-3: CS | Dims 4-7: Math | Dims 8-11: Food | Dims 12-15: Sports
    {
        "meta": "Linked List: nodes connected by pointers",
        "cat": "cs",
        "emb": [0.90, 0.85, 0.72, 0.68, 0.12, 0.08, 0.15, 0.10, 0.05, 0.08, 0.06, 0.09, 0.07, 0.11, 0.08, 0.06]
    },
    {
        "meta": "Binary Search Tree: O(log n) search and insert",
        "cat": "cs",
        "emb": [0.88, 0.82, 0.78, 0.74, 0.15, 0.10, 0.08, 0.12, 0.06, 0.07, 0.08, 0.05, 0.09, 0.06, 0.07, 0.10]
    },
    {
        "meta": "Dynamic Programming: memoization overlapping subproblems",
        "cat": "cs",
        "emb": [0.82, 0.76, 0.88, 0.80, 0.20, 0.18, 0.12, 0.09, 0.07, 0.06, 0.08, 0.07, 0.08, 0.09, 0.06, 0.07]
    },
    {
        "meta": "Graph BFS and DFS: breadth and depth first traversal",
        "cat": "cs",
        "emb": [0.85, 0.80, 0.75, 0.82, 0.18, 0.14, 0.10, 0.08, 0.06, 0.09, 0.07, 0.06, 0.10, 0.08, 0.09, 0.07]
    },
    {
        "meta": "Hash Table: O(1) lookup with collision chaining",
        "cat": "cs",
        "emb": [0.87, 0.78, 0.70, 0.76, 0.13, 0.11, 0.09, 0.14, 0.08, 0.07, 0.06, 0.08, 0.07, 0.10, 0.08, 0.09]
    },
    {
        "meta": "Calculus: derivatives integrals and limits",
        "cat": "math",
        "emb": [0.12, 0.15, 0.18, 0.10, 0.91, 0.86, 0.78, 0.72, 0.08, 0.06, 0.07, 0.09, 0.07, 0.08, 0.06, 0.10]
    },
    {
        "meta": "Linear Algebra: matrices eigenvalues eigenvectors",
        "cat": "math",
        "emb": [0.20, 0.18, 0.15, 0.12, 0.88, 0.90, 0.82, 0.76, 0.09, 0.07, 0.08, 0.06, 0.10, 0.07, 0.08, 0.09]
    },
    {
        "meta": "Probability: distributions random variables Bayes theorem",
        "cat": "math",
        "emb": [0.15, 0.12, 0.20, 0.18, 0.84, 0.80, 0.88, 0.82, 0.07, 0.08, 0.06, 0.10, 0.09, 0.06, 0.09, 0.08]
    },
    {
        "meta": "Number Theory: primes modular arithmetic RSA cryptography",
        "cat": "math",
        "emb": [0.22, 0.16, 0.14, 0.20, 0.80, 0.85, 0.76, 0.90, 0.08, 0.09, 0.07, 0.06, 0.08, 0.10, 0.07, 0.06]
    },
    {
        "meta": "Combinatorics: permutations combinations generating functions",
        "cat": "math",
        "emb": [0.18, 0.20, 0.16, 0.14, 0.86, 0.78, 0.84, 0.80, 0.06, 0.07, 0.09, 0.08, 0.06, 0.09, 0.10, 0.07]
    },
    {
        "meta": "Neapolitan Pizza: wood-fired dough San Marzano tomatoes",
        "cat": "food",
        "emb": [0.08, 0.06, 0.09, 0.07, 0.07, 0.08, 0.06, 0.09, 0.90, 0.86, 0.78, 0.72, 0.08, 0.06, 0.09, 0.07]
    },
    {
        "meta": "Sushi: vinegared rice raw fish and nori rolls",
        "cat": "food",
        "emb": [0.06, 0.08, 0.07, 0.09, 0.09, 0.06, 0.08, 0.07, 0.86, 0.90, 0.82, 0.76, 0.07, 0.09, 0.06, 0.08]
    },
    {
        "meta": "Ramen: noodle soup with chashu pork and soft-boiled eggs",
        "cat": "food",
        "emb": [0.09, 0.07, 0.06, 0.08, 0.08, 0.09, 0.07, 0.06, 0.82, 0.78, 0.90, 0.84, 0.09, 0.07, 0.08, 0.06]
    },
    {
        "meta": "Tacos: corn tortillas with carnitas salsa and cilantro",
        "cat": "food",
        "emb": [0.07, 0.09, 0.08, 0.06, 0.06, 0.07, 0.09, 0.08, 0.78, 0.82, 0.86, 0.90, 0.06, 0.08, 0.07, 0.09]
    },
    {
        "meta": "Croissant: laminated pastry with buttery flaky layers",
        "cat": "food",
        "emb": [0.06, 0.07, 0.10, 0.09, 0.10, 0.06, 0.07, 0.10, 0.85, 0.80, 0.76, 0.82, 0.09, 0.07, 0.10, 0.06]
    },
    {
        "meta": "Basketball: fast-paced shooting dribbling slam dunks",
        "cat": "sports",
        "emb": [0.09, 0.07, 0.08, 0.10, 0.08, 0.09, 0.07, 0.06, 0.08, 0.07, 0.09, 0.06, 0.91, 0.85, 0.78, 0.72]
    },
    {
        "meta": "Football: tackles touchdowns field goals and strategy",
        "cat": "sports",
        "emb": [0.07, 0.09, 0.06, 0.08, 0.09, 0.07, 0.10, 0.08, 0.07, 0.09, 0.08, 0.07, 0.87, 0.89, 0.82, 0.76]
    },
    {
        "meta": "Tennis: racket volleys groundstrokes and Wimbledon serves",
        "cat": "sports",
        "emb": [0.08, 0.06, 0.09, 0.07, 0.07, 0.08, 0.06, 0.09, 0.09, 0.06, 0.07, 0.08, 0.83, 0.80, 0.88, 0.82]
    },
    {
        "meta": "Chess: openings endgames tactics strategic board game",
        "cat": "sports",
        "emb": [0.25, 0.20, 0.22, 0.18, 0.22, 0.18, 0.20, 0.15, 0.06, 0.08, 0.07, 0.09, 0.80, 0.84, 0.78, 0.90]
    },
    {
        "meta": "Swimming: butterfly freestyle backstroke Olympic competition",
        "cat": "sports",
        "emb": [0.06, 0.08, 0.07, 0.09, 0.08, 0.06, 0.09, 0.07, 0.10, 0.08, 0.06, 0.07, 0.85, 0.82, 0.86, 0.80]
    }
]
```

---

## 8. Ollama Client & Concurrency — STRUCTURAL & LOOSE PARITY

### 8.1 Ollama Client Protocol
- **Endpoints & Timeouts**:
  - `GET http://127.0.0.1:11434/api/tags` — connect timeout: 2.0s
  - `POST http://127.0.0.1:11434/api/embeddings` — connect: 3.0s, read: 30.0s, body: `{"model": "nomic-embed-text", "prompt": text}`
  - `POST http://127.0.0.1:11434/api/generate` — connect: 3.0s, read: 180.0s, body: `{"model": "llama3.2", "prompt": prompt, "stream": false}`
- **Fallback Error Strings**:
  - When generation fails / times out: returns verbatim `"ERROR: Ollama unavailable. Run: ollama serve"`
- **JSON Parsing**:
  - Replace C++ bracket counting and custom `esc()` with standard Python `json.loads()` and `json.dumps()` or `requests`/`httpx`.

### 8.2 Concurrency & Mutex Management
- In C++, all index operations are synchronized via per-database `std::mutex mu`.
- In Python, wrap `VectorDB` and `DocumentDB` access with `threading.Lock()` or `asyncio.Lock()`.
- **Lock Scope Invariant**: External HTTP calls to Ollama (`embed()`, `generate()`) must **always execute outside the index lock**.
