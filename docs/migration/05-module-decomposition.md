# Module Decomposition: Python Package Layout

**Document**: `05-module-decomposition.md`  
**Source**: `C:\Users\Samar\Documents\Your-OWN-AI\main.cpp` (1089 lines)  
**Target**: Python 3.11+ package `vectordb/` + project-root shim  
**Status**: Design baseline for phases P1–P8 of the execution plan

---

## 1. Proposed Layout

The migration moves the single `main.cpp` translation unit into a thin Python
package. The project root keeps the frontend (`index.html`) and the README;
the Python sources live under `py/vectordb/` and the tests under `py/tests/`.

```
Your-OWN-AI/
├── index.html                    # unchanged — web frontend
├── README.md                     # updated with new run instructions
├── main.py                       # thin shim: from vectordb.__main__ import main; main()
└── py/
    ├── vectordb/
    │   ├── __init__.py           # re-exports public API
    │   ├── __main__.py           # python -m vectordb entry point
    │   ├── app.py                # Flask app factory + 14 HTTP routes
    │   ├── distances.py          # euclidean, cosine, manhattan, get_dist_fn, DistFn
    │   ├── bruteforce.py         # VectorItem, BruteForce
    │   ├── kdtree.py             # KDNode, KDTree
    │   ├── hnsw.py               # HNSWNode, HNSW (+ HNSWInfo)
    │   ├── db.py                 # VectorDB (multi-index container, mutex)
    │   ├── document_db.py        # DocItem, DocumentDB
    │   ├── ollama.py             # OllamaClient
    │   ├── chunker.py            # chunk_text
    │   ├── demo_data.py          # DEMO_VECTORS, load_demo
    │   └── json_format.py        # j_s, j_vec, parse_vec, extract_str, extract_int, parse_body
    └── tests/
        ├── parity/
        │   ├── run_parity.py     # diff runner: C++ vs Python output
        │   └── fixtures/         # captured HTTP request/response pairs
        ├── test_distances.py     # (filled in P5)
        ├── test_bruteforce.py
        ├── test_kdtree.py
        ├── test_hnsw.py
        ├── test_chunker.py
        ├── test_db.py
        ├── test_document_db.py
        └── test_api.py
├── pyproject.toml                # package metadata + console script
└── requirements.txt              # pinned runtime + dev deps
```

Notes on layout choices:

- The Python sources sit under `py/` (not at the repo root) so the C++ build
  artefacts and the `index.html` stay at the top level unchanged.
- `main.py` is a **one-liner shim** at the project root, not the real entry
  point. It exists only so users who type `python main.py` (as the README
  currently shows) keep working. Real entry point is `python -m vectordb`.
- `tests/parity/` is a separate harness from the unit tests under
  `tests/test_*.py`; the parity runner compares actual C++ binary output
  against Python output for the same requests.

---

## 2. Per-Module Spec

Each module below lists its file path, the C++ source line ranges, the public
Python API, internal state (for classes), and dependencies on other Python
modules in this package.

### 2.1 `distances.py`

- **Path**: `py/vectordb/distances.py`
- **C++ source**: `main.cpp:33, 39-64`
- **Public API**:
  ```python
  from typing import Callable, Sequence

  DistFn = Callable[[Sequence[float], Sequence[float]], float]

  def euclidean(a: Sequence[float], b: Sequence[float]) -> float: ...
  def cosine(a: Sequence[float], b: Sequence[float]) -> float: ...
  def manhattan(a: Sequence[float], b: Sequence[float]) -> float: ...
  def get_dist_fn(metric: str) -> DistFn: ...
  ```
- **Internal state**: None (pure functions; `DistFn` is a `Callable` alias).
- **Dependencies**: stdlib only (`math`).

### 2.2 `bruteforce.py`

- **Path**: `py/vectordb/bruteforce.py`
- **C++ source**: `main.cpp:26-31` (VectorItem) and `main.cpp:70-91` (BruteForce)
- **Public API**:
  ```python
  from dataclasses import dataclass
  from typing import List, Sequence, Tuple

  @dataclass
  class VectorItem:
      id: int
      metadata: str
      category: str
      emb: List[float]

  class BruteForce:
      def __init__(self) -> None: ...
      def insert(self, v: VectorItem) -> None: ...
      def knn(self, q: Sequence[float], k: int, dist: DistFn) -> List[Tuple[float, int]]: ...
      def remove(self, item_id: int) -> None: ...
  ```
- **Internal state**:
  - `BruteForce.items: List[VectorItem]` — public, matches C++ `std::vector<VectorItem> items`.
  - `VectorItem` is an immutable-style dataclass (mutable `emb` list, but `id`/`metadata`/`category` are not rewritten after insert).
- **Dependencies**: `distances.DistFn` (type only).

### 2.3 `kdtree.py`

- **Path**: `py/vectordb/kdtree.py`
- **C++ source**: `main.cpp:97-159`
- **Public API**:
  ```python
  from typing import Optional, List, Sequence, Tuple

  class KDNode:
      def __init__(self, item: VectorItem) -> None: ...

  class KDTree:
      def __init__(self, dims: int) -> None: ...
      def insert(self, v: VectorItem) -> None: ...
      def knn(self, q: Sequence[float], k: int, dist: DistFn) -> List[Tuple[float, int]]: ...
      def rebuild(self, items: Sequence[VectorItem]) -> None: ...
  ```
- **Internal state**:
  - `KDTree.dims: int` — fixed at construction.
  - `KDTree.root: Optional[KDNode]` — `None` when empty.
  - `KDNode.item: VectorItem`, `KDNode.left: Optional[KDNode]`, `KDNode.right: Optional[KDNode]`.
- **Dependencies**: `bruteforce.VectorItem`, `distances.DistFn`.

### 2.4 `hnsw.py`

- **Path**: `py/vectordb/hnsw.py`
- **C++ source**: `main.cpp:165-331`
- **Public API**:
  ```python
  from typing import Dict, List, Sequence, Tuple, Any, Optional

  class HNSWNode:
      def __init__(self, item: VectorItem, max_lyr: int) -> None: ...

  class HNSW:
      def __init__(self, m: int = 16, ef_build: int = 200, seed: int = 42) -> None: ...
      def insert(self, item: VectorItem, dist: DistFn) -> None: ...
      def knn(self, q: Sequence[float], k: int, ef: int, dist: DistFn) -> List[Tuple[float, int]]: ...
      def remove(self, item_id: int) -> None: ...
      def get_info(self) -> Dict[str, Any]: ...  # GraphInfo dict
      def size(self) -> int: ...
  ```
  The C++ nested `HNSW::GraphInfo` is exposed as a plain `dict` with the
  schema in §2.4 of `04-algorithm-parity.md` (camelCase keys: `topLayer`,
  `nodeCount`, `nodesPerLayer`, `edgesPerLayer`, `nodes`, `edges`).
- **Internal state**:
  - `HNSW.G: Dict[int, HNSWNode]`, `HNSW.M: int`, `HNSW.M0: int`,
    `HNSW.ef_build: int`, `HNSW.mL: float`, `HNSW.top_layer: int` (starts `-1`),
    `HNSW.entry_pt: int` (starts `-1`), `HNSW.rng: random.Random`.
  - `HNSWNode.item: VectorItem`, `HNSWNode.max_lyr: int`,
    `HNSWNode.nbrs: List[List[int]]`.
- **Dependencies**: `bruteforce.VectorItem`, `distances.DistFn`, stdlib `math`/`random`/`heapq`.

### 2.5 `db.py`

- **Path**: `py/vectordb/db.py`
- **C++ source**: `main.cpp:337-429`
- **Public API**:
  ```python
  from dataclasses import dataclass
  from typing import List, Dict, Any
  import threading

  @dataclass
  class Hit:
      id: int
      meta: str
      cat: str
      emb: List[float]
      dist: float

  @dataclass
  class SearchOut:
      hits: List[Hit]
      us: int           # microseconds
      algo: str
      metric: str

  @dataclass
  class BenchOut:
      bf_us: int
      kd_us: int
      hnsw_us: int
      n: int

  class VectorDB:
      def __init__(self, dims: int) -> None: ...
      def insert(self, meta: str, cat: str, emb: List[float], dist: DistFn) -> int: ...
      def remove(self, item_id: int) -> bool: ...
      def search(self, q: List[float], k: int, metric: str, algo: str) -> SearchOut: ...
      def benchmark(self, q: List[float], k: int, metric: str) -> BenchOut: ...
      def all(self) -> List[VectorItem]: ...
      def hnsw_info(self) -> Dict[str, Any]: ...
      def size(self) -> int: ...
  ```
- **Internal state**:
  - `VectorDB.dims: int` (public, matches C++ `const int dims`).
  - `VectorDB._store: Dict[int, VectorItem]`.
  - `VectorDB._bf: BruteForce`, `VectorDB._kdt: KDTree`, `VectorDB._hnsw: HNSW`.
  - `VectorDB._mu: threading.Lock()`.
  - `VectorDB._next_id: int`.
- **Dependencies**: `bruteforce.BruteForce`, `bruteforce.VectorItem`,
  `kdtree.KDTree`, `hnsw.HNSW`, `distances.get_dist_fn`, stdlib `threading`.

### 2.6 `document_db.py`

- **Path**: `py/vectordb/document_db.py`
- **C++ source**: `main.cpp:644-711`
- **Public API**:
  ```python
  from dataclasses import dataclass
  from typing import List, Tuple
  import threading

  @dataclass
  class DocItem:
      id: int
      title: str
      text: str
      emb: List[float]

  class DocumentDB:
      def __init__(self) -> None: ...
      def insert(self, title: str, text: str, emb: List[float]) -> int: ...
      def search(self, q: List[float], k: int, max_dist: float = 0.7) -> List[Tuple[float, DocItem]]: ...
      def remove(self, item_id: int) -> bool: ...
      def all(self) -> List[DocItem]: ...
      def size(self) -> int: ...
      def get_dims(self) -> int: ...
  ```
- **Internal state**:
  - `DocumentDB._store: Dict[int, DocItem]`.
  - `DocumentDB._hnsw: HNSW`, `DocumentDB._bf: BruteForce`.
  - `DocumentDB._mu: threading.Lock()`.
  - `DocumentDB._next_id: int`, `DocumentDB._dims: int` (starts `0`).
- **Dependencies**: `bruteforce.BruteForce`, `bruteforce.VectorItem`,
  `hnsw.HNSW`, `distances.cosine`, stdlib `threading`.

### 2.7 `ollama.py`

- **Path**: `py/vectordb/ollama.py`
- **C++ source**: `main.cpp:561-638`
- **Public API**:
  ```python
  from typing import List

  class OllamaClient:
      embed_model: str = "nomic-embed-text"
      gen_model: str = "llama3.2"

      def __init__(self, host: str = "127.0.0.1", port: int = 11434) -> None: ...
      def is_available(self) -> bool: ...
      def embed(self, text: str) -> List[float]: ...
      def generate(self, prompt: str) -> str: ...
  ```
- **Internal state**:
  - `OllamaClient._host: str`, `OllamaClient._port: int`.
  - Per-call timeouts (passed to `requests`): 2.0s for `/api/tags`;
    (3.0s connect, 30.0s read) for `/api/embeddings`;
    (3.0s connect, 180.0s read) for `/api/generate`.
- **Dependencies**: third-party `requests`; stdlib `json`.
  Replaces the C++ `httplib::Client` calls. No module-internal deps.

### 2.8 `chunker.py`

- **Path**: `py/vectordb/chunker.py`
- **C++ source**: `main.cpp:531-552`
- **Public API**:
  ```python
  from typing import List

  def chunk_text(text: str, chunk_words: int = 250, overlap_words: int = 30) -> List[str]: ...
  ```
- **Internal state**: None (pure function).
- **Dependencies**: stdlib only.

### 2.9 `demo_data.py`

- **Path**: `py/vectordb/demo_data.py`
- **C++ source**: `main.cpp:717-760`
- **Public API**:
  ```python
  from typing import List, Dict, Any

  DEMO_VECTORS: List[Dict[str, Any]] = [...]  # 20 entries, see 04-algorithm-parity.md §7

  def load_demo(db: VectorDB) -> None: ...
  ```
  `DEMO_VECTORS` uses dict keys `"meta"`, `"cat"`, `"emb"` to match the
  C++ initializer list literal in `main.cpp:717-760`; `load_demo` walks the
  list and calls `db.insert(meta, cat, emb, get_dist_fn("cosine"))`.
- **Internal state**: Module-level constant `DEMO_VECTORS`.
- **Dependencies**: `db.VectorDB`, `distances.get_dist_fn`.

### 2.10 `json_format.py`

- **Path**: `py/vectordb/json_format.py`
- **C++ source**: `main.cpp:435-525`
- **Public API**:
  ```python
  from typing import Any, Tuple, List

  def j_s(s: str) -> str: ...
  def j_vec(v: List[float]) -> str: ...
  def parse_vec(s: str) -> List[float]: ...
  def extract_str(body: str, key: str) -> str: ...
  def extract_int(body: str, key: str, default: int = 0) -> int: ...
  def parse_body(body: str) -> Tuple[str, str, List[float]]: ...  # (metadata, category, embedding)
  def cors(headers: dict) -> dict: ...   # for Flask: returns CORS header dict
  ```
  `cors` returns a header `dict` to merge into a Flask `Response` instead of
  mutating an `httplib::Response` in place. The behaviour — `Access-Control-*`
  headers permitting all origins and methods — is identical.
- **Internal state**: None (pure functions).
- **Dependencies**: stdlib `re`, `json`. No module-internal deps.

### 2.11 `app.py`

- **Path**: `py/vectordb/app.py`
- **C++ source**: `main.cpp:766-1089` (the `main()` HTTP server setup)
- **Public API**:
  ```python
  from flask import Flask

  def create_app(db: VectorDB, doc_db: DocumentDB, ollama: OllamaClient) -> Flask: ...
  def run(host: str = "0.0.0.0", port: int = 8080) -> None: ...
  ```
  `create_app` registers the 14 routes listed in `01-architecture.md` §1.12.
  `run` builds the three service objects, calls `load_demo`, checks Ollama
  availability, and calls `app.run(host=..., port=...)`.
- **Internal state**: Per-process singletons (`db`, `doc_db`, `ollama`).
- **Dependencies**: `db.VectorDB`, `document_db.DocumentDB`, `ollama.OllamaClient`,
  `demo_data.load_demo`, `json_format.*`, `chunker.chunk_text`,
  `distances.get_dist_fn`, third-party `flask`.

### 2.12 `__main__.py`

- **Path**: `py/vectordb/__main__.py`
- **C++ source**: `main.cpp:766-1089` (entry point)
- **Public API**:
  ```python
  def main() -> None: ...
  ```
  Resolves the repo root (parent of `py/`), `os.chdir`s into it so that
  `GET /` finds `index.html` (replaces the C++ working-directory dependency
  described in `01-architecture.md` §8.8), and calls `app.run()`.
- **Internal state**: None.
- **Dependencies**: `app.run`, stdlib `os`, `pathlib`.

### 2.13 `__init__.py`

- **Path**: `py/vectordb/__init__.py`
- **Public API**: Re-exports the stable public surface so callers can do
  `from vectordb import VectorDB, BruteForce, euclidean, ...`:
  ```python
  from .distances import euclidean, cosine, manhattan, get_dist_fn, DistFn
  from .bruteforce import VectorItem, BruteForce
  from .kdtree import KDTree
  from .hnsw import HNSW
  from .db import VectorDB
  from .document_db import DocumentDB, DocItem
  from .ollama import OllamaClient
  from .chunker import chunk_text
  from .demo_data import load_demo
  ```
  The Flask app, `app.py`, and `__main__.py` are deliberately **not**
  re-exported — they are side-effecting entry points.

### 2.14 `main.py` (project root shim)

- **Path**: `main.py` (at `C:\Users\Samar\Documents\Your-OWN-AI\main.py`)
- **Public API**: `python main.py` invokes `vectordb.__main__.main`.
- **Content** (entire file, 4 lines):
  ```python
  """Back-compat shim. The real entry point is `python -m vectordb`."""
  from vectordb.__main__ import main

  if __name__ == "__main__":
      main()
  ```
- **Dependencies**: `vectordb` (installed via `pyproject.toml` or run with
  `PYTHONPATH=py`).

---

## 3. Class-to-Module Mapping

Every C++ `class`/`struct` from `01-architecture.md` mapped to its Python
target module and final identifier.

| C++ Symbol | C++ Lines | Python Module | Python Identifier | Notes |
|:---|:---|:---|:---|:---|
| `VectorItem` (struct) | `26-31` | `bruteforce.py` | `VectorItem` | Name unchanged. Becomes a `@dataclass`. |
| `DistFn` (using alias) | `33` | `distances.py` | `DistFn` | `Callable[[Sequence[float], Sequence[float]], float]`. |
| `BruteForce` (class) | `70-91` | `bruteforce.py` | `BruteForce` | Name unchanged. |
| `KDNode` (struct) | `97-102` | `kdtree.py` | `KDNode` | Name unchanged. |
| `KDTree` (class) | `104-159` | `kdtree.py` | `KDTree` | Name unchanged. |
| `HNSW` (class) | `165-331` | `hnsw.py` | `HNSW` | Name unchanged. |
| `HNSW::Node` (private struct) | `173-177` | `hnsw.py` | `HNSWNode` | Renamed from `Node` to avoid stdlib `typing.Node` shadowing. |
| `HNSW::GraphInfo` (public nested struct) | `154-161` | `hnsw.py` | (return value of `HNSW.get_info()`) | Replaced by a plain `dict` with camelCase keys; no struct class needed. |
| `HNSW::GraphInfo::NV` | `157` | `hnsw.py` | (inline dict) | Replaced by inline `{id, metadata, category, maxLyr}` dicts. |
| `HNSW::GraphInfo::EV` | `158` | `hnsw.py` | (inline dict) | Replaced by inline `{src, dst, lyr}` dicts. |
| `VectorDB` (class) | `337-429` | `db.py` | `VectorDB` | Name unchanged. Owns a `threading.Lock()`. |
| `VectorDB::Hit` | `203` | `db.py` | `Hit` | `@dataclass`. |
| `VectorDB::SearchOut` | `204` | `db.py` | `SearchOut` | `@dataclass`. |
| `VectorDB::BenchOut` | `205` | `db.py` | `BenchOut` | `@dataclass`. |
| `OllamaClient` (class) | `561-638` | `ollama.py` | `OllamaClient` | Name unchanged. |
| `DocItem` (struct) | `644-649` | `document_db.py` | `DocItem` | `@dataclass`. |
| `DocumentDB` (class) | `651-711` | `document_db.py` | `DocumentDB` | Name unchanged. Owns a `threading.Lock()`. |

**Total**: 14 distinct C++ classes/structs, all mapped. 1 rename
(`HNSW::Node` → `HNSWNode`) to avoid stdlib namespace collision; the rest
keep their names.

---

## 4. Free Function Mapping

Every free function in `main.cpp` mapped to its Python target module and
final identifier.

| C++ Function | C++ Lines | Python Module | Python Identifier | Notes |
|:---|:---|:---|:---|:---|
| `jS` | `435-446` | `json_format.py` | `j_s` | `s` is reserved-feeling; snake_case. |
| `jVec` | `448-455` | `json_format.py` | `j_vec` | snake_case. |
| `parseVec` | `457-463` | `json_format.py` | `parse_vec` | snake_case. |
| `extractStr` | `466-492` | `json_format.py` | `extract_str` | snake_case. |
| `extractInt` | `495-501` | `json_format.py` | `extract_int` | snake_case. |
| `parseBody` | `503-519` | `json_format.py` | `parse_body` | snake_case. |
| `cors` | `521-525` | `json_format.py` | `cors` | Name unchanged (already lowercase). Returns a header `dict` for Flask. |
| `chunkText` | `531-552` | `chunker.py` | `chunk_text` | snake_case. |
| `euclidean` | `39-43` | `distances.py` | `euclidean` | Name unchanged. |
| `cosine` | `45-52` | `distances.py` | `cosine` | Name unchanged. |
| `manhattan` | `54-58` | `distances.py` | `manhattan` | Name unchanged. |
| `getDistFn` | `60-64` | `distances.py` | `get_dist_fn` | snake_case. |
| `loadDemo` | `717-760` | `demo_data.py` | `load_demo` | snake_case. |

**Total**: 13 free functions, 9 renamed to snake_case, 4 already
lowercase. The `DEMO_VECTORS` constant in `demo_data.py` is the
Pythonic equivalent of the C++ initializer-list literal in `loadDemo`.

---

## 5. Naming Conventions

- **Functions and variables**: `snake_case` (Python PEP 8).
- **Classes**: `PascalCase` (Python PEP 8). All C++ class names are already
  `PascalCase`; none are renamed.
- **Module-level constants**: `UPPER_SNAKE_CASE` (e.g. `DEMO_VECTORS`).
- **Type aliases**: `PascalCase` (e.g. `DistFn`).

### Renames

| C++ Identifier | Python Identifier | Reason |
|:---|:---|:---|
| `jS` | `j_s` | snake_case |
| `jVec` | `j_vec` | snake_case |
| `parseVec` | `parse_vec` | snake_case |
| `extractStr` | `extract_str` | snake_case |
| `extractInt` | `extract_int` | snake_case |
| `parseBody` | `parse_body` | snake_case |
| `chunkText` | `chunk_text` | snake_case |
| `getDistFn` | `get_dist_fn` | snake_case |
| `loadDemo` | `load_demo` | snake_case |
| `HNSW::Node` | `HNSWNode` | avoid stdlib `typing.Node` shadowing |
| `cors` | `cors` | already lowercase |
| `euclidean`, `cosine`, `manhattan` | unchanged | already snake_case |
| `BruteForce`, `KDTree`, `HNSW`, `KDNode`, `VectorDB`, `DocumentDB`, `OllamaClient`, `VectorItem`, `DocItem`, `Hit`, `SearchOut`, `BenchOut` | unchanged | already PascalCase |

### HTTP API field names stay camelCase

The 14 HTTP routes return JSON with camelCase keys. These are the public
contract with the frontend (`index.html`) and **must not change**:

- `/search`, `/insert`, `/delete/:id`, `/items`, `/benchmark` — use
  `id`, `metadata`, `category`, `embedding`, `topLayer`, `nodeCount`,
  `nodesPerLayer`, `edgesPerLayer`, `nodes`, `edges`, `us`, `bfUs`,
  `kdUs`, `hnswUs`, `n`, `algo`, `metric`, `hits`.
- `/hnsw-info` — see `04-algorithm-parity.md` §4.3 for the exact schema.
- `/doc/insert`, `/doc/list`, `/doc/search`, `/doc/ask` — use
  `id`, `title`, `text`, `preview`, `wordCount`, `answer`, `count`.

Inside the Python modules, attribute names are snake_case (e.g.
`HNSW.top_layer`, `HNSW.entry_pt`, `HNSW.ef_build`); conversion to
camelCase happens at the JSON serialization boundary in `app.py`.

---

## 6. Entry Point

The recommended invocation is:

```bash
python -m vectordb
```

This runs `py/vectordb/__main__.py:main()`, which:

1. Computes the repo root (the parent of `py/`) and `os.chdir`s into it
   so that `GET /` finds `index.html` (fixing the working-directory
   dependency noted in `01-architecture.md` §8.8).
2. Constructs `VectorDB(16)`, `DocumentDB()`, `OllamaClient()`.
3. Calls `load_demo(db)`.
4. Calls `ollama.is_available()` and logs the result.
5. Calls `app.run(host="0.0.0.0", port=8080)`.

### Backward compatibility

The README currently shows `python main.py`. To preserve that, a
**one-liner shim** is created at the project root:

```python
# main.py
from vectordb.__main__ import main
if __name__ == "__main__":
    main()
```

Both invocations work:

- `python -m vectordb` — canonical.
- `python main.py` — legacy, kept working.

The `pyproject.toml` console-script entry (see §7) provides a third
invocation: `vectordb-server`.

---

## 7. `pyproject.toml` Draft

Minimum content for `py/vectordb` to install as a package:

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "vectordb"
version = "0.1.0"
description = "VectorDB & RAG engine migrated from main.cpp to Python 3.11+"
requires-python = ">=3.11"
dependencies = [
    "flask>=3.0,<4.0",
    "requests>=2.31,<3.0",
    "numpy>=1.26,<3.0",
    "hnswlib>=0.7,<1.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0,<9.0",
    "pytest-cov>=4.1,<6.0",
]

[project.scripts]
vectordb-server = "vectordb.__main__:main"

[tool.setuptools.packages.find]
where = ["py"]
```

`hnswlib` is included as a dependency per the engine decision in
`04-algorithm-parity.md` §4.4. If the team instead keeps the native
Python hand-port for HNSW (recommended for v1 parity), drop `hnswlib`
and add a comment in `08-risks-and-decisions.md` recording the choice.
See `08-risks-and-decisions.md` decision **D1** (HNSW engine).

---

## 8. `requirements.txt` Draft

```
# Runtime
flask>=3.0,<4.0
requests>=2.31,<3.0
numpy>=1.26,<3.0
# hnswlib see D1 in 08-risks.md
hnswlib>=0.7,<1.0

# Dev
pytest>=8.0,<9.0
pytest-cov>=4.1,<6.0
```

`hnswlib` is the only line with an open question — it is included on
the strength of `04-algorithm-parity.md` §4.4 (hand-port preferred for
v1) but the line is kept so the v2 swap is a single edit. See
`08-risks-and-decisions.md` **D1**.

---

## 9. Migration Order Mapping (from `07-execution-plan.md`)

For each phase, the files that phase will create (skeleton) or fill
(complete). The skeleton in P1 is a stub that imports nothing; later
phases add the real code.

| Phase | Files Created / Filled |
|:---|:---|
| **P1** (scaffold) | `py/vectordb/__init__.py` (empty), `py/vectordb/__main__.py` (stub `main()`), `main.py` (shim), `pyproject.toml`, `requirements.txt`, `README.md` (updated), `py/tests/__init__.py` (empty), `py/tests/parity/run_parity.py` (skeleton), `py/tests/parity/fixtures/.gitkeep` |
| **P2** (distances) | `py/vectordb/distances.py` (filled), `py/vectordb/__init__.py` (re-export), `py/tests/test_distances.py` (parity tests vs C++ values) |
| **P3** (indexes) | `py/vectordb/bruteforce.py`, `py/vectordb/kdtree.py`, `py/vectordb/hnsw.py`, `py/vectordb/__init__.py` (re-export), `py/tests/test_bruteforce.py`, `py/tests/test_kdtree.py`, `py/tests/test_hnsw.py` |
| **P4** (containers) | `py/vectordb/db.py`, `py/vectordb/document_db.py`, `py/vectordb/__init__.py` (re-export), `py/tests/test_db.py`, `py/tests/test_document_db.py` |
| **P5** (json_format, chunker) | `py/vectordb/json_format.py`, `py/vectordb/chunker.py`, `py/tests/test_chunker.py` |
| **P6** (ollama, demo) | `py/vectordb/ollama.py`, `py/vectordb/demo_data.py` (with `DEMO_VECTORS` constant and `load_demo`), `py/tests/test_api.py` (mocked Ollama) |
| **P7** (Flask app + cutover) | `py/vectordb/app.py`, `py/vectordb/__main__.py` (filled, replaces stub from P1), DELETE `main.cpp`, `README.md` (final update) |
| **P8** (parity + sign-off) | `py/tests/parity/run_parity.py` (filled — runs C++ binary and Python server side by side, diffs responses), `py/tests/parity/fixtures/` (captured request/response pairs) |

P1 and P8 span the whole package (scaffolding, then full integration);
the meat is in P2–P7.

---

## 10. What Is NOT in the Package

Explicit non-goals for the Python package — things that do not migrate
or do not live under `py/vectordb/`:

- **`httplib.h`** (16999-line vendored header): replaced by Flask. The
  HTTP server is `flask.Flask`; the HTTP client is `requests`. The file
  `httplib.h` is deleted when `main.cpp` is deleted in P7.
- **`main.cpp`**: deleted in P7 once `vectordb.app` and
  `vectordb.__main__` are running and parity is signed off.
- **Rufflo / Claude-Flow / `.claude/`, `.claude-flow/`, `.swarm/`**:
  untracked dev tooling, not part of the shipped package, and not
  installed by `pip install`.
- **`tests/`**: lives **outside** the `vectordb` package (`py/tests/`,
  not `py/vectordb/tests.py`) so that pytest can collect them without
  importing the package as a side effect, and so that the parity
  harness is a separate tool.
- **`index.html`**: stays at the project root (served by `GET /`); not
  bundled into the Python package.
- **`.gitignore`**: not migrated — the existing gitignore already
  covers `__pycache__/`, `.pytest_cache/`, `*.egg-info/`, etc. The
  migration only adds Python-specific lines if necessary.

---

## Summary

Module decomposition written — **14 modules, 14 classes mapped, 9 functions renamed.**
