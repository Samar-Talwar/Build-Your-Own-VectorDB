# Architecture Map: C++ VectorDB & RAG Engine (`main.cpp`)

**Source File**: `C:\Users\Samar\Documents\Your-OWN-AI\main.cpp` (1089 lines)  
**Target Migration**: Python 3.11+ async service

---

## 1. Module Map

### 1.1 Data Types & Aliases

#### `VectorItem` (struct)
- **Lines**: `main.cpp:26-31`
- **Purpose**: In-memory representation of an indexed vector item with metadata, category, and dense float vector.
- **Public Surface**:
  ```cpp
  struct VectorItem {
      int id;
      std::string metadata;
      std::string category;
      std::vector<float> emb;
  };
  ```
- **Private Members**: None (aggregate struct).
- **Mutex Ownership**: None.

#### `DistFn` (using alias)
- **Lines**: `main.cpp:33`
- **Purpose**: Function wrapper signature for pairwise vector distance metrics.
- **Public Surface**:
  ```cpp
  using DistFn = std::function<float(const std::vector<float>&, const std::vector<float>&)>;
  ```
- **Private Members**: None.
- **Mutex Ownership**: None.

---

### 1.2 Distance Metrics (Free Functions)

#### `euclidean`
- **Lines**: `main.cpp:39-43`
- **Purpose**: Computes $L_2$ Euclidean distance $\sqrt{\sum (a_i - b_i)^2}$.
- **Public Surface**:
  ```cpp
  float euclidean(const std::vector<float>& a, const std::vector<float>& b);
  ```
- **Private Members**: N/A.
- **Mutex Ownership**: None.

#### `cosine`
- **Lines**: `main.cpp:45-52`
- **Purpose**: Computes cosine distance $1.0 - \frac{a \cdot b}{\|a\| \|b\|}$; returns $1.0f$ if either norm is $< 10^{-9}$.
- **Public Surface**:
  ```cpp
  float cosine(const std::vector<float>& a, const std::vector<float>& b);
  ```
- **Private Members**: N/A.
- **Mutex Ownership**: None.

#### `manhattan`
- **Lines**: `main.cpp:54-58`
- **Purpose**: Computes $L_1$ Manhattan distance $\sum |a_i - b_i|$.
- **Public Surface**:
  ```cpp
  float manhattan(const std::vector<float>& a, const std::vector<float>& b);
  ```
- **Private Members**: N/A.
- **Mutex Ownership**: None.

#### `getDistFn`
- **Lines**: `main.cpp:60-64`
- **Purpose**: Factory dispatcher mapping metric names (`"cosine"`, `"manhattan"`, default `"euclidean"`) to function pointers.
- **Public Surface**:
  ```cpp
  DistFn getDistFn(const std::string& m);
  ```
- **Private Members**: N/A.
- **Mutex Ownership**: None.

---

### 1.3 Exact Search: `BruteForce` (class)

- **Lines**: `main.cpp:70-91`
- **Purpose**: Flat-array storage with exhaustive linear search for exact $k$-nearest neighbors.
- **Public Surface**:
  ```cpp
  class BruteForce {
  public:
      std::vector<VectorItem> items;
      void insert(const VectorItem& v);
      std::vector<std::pair<float,int>> knn(const std::vector<float>& q, int k, DistFn dist);
      void remove(int id);
  };
  ```
- **Private Members**: None (stores `items` publicly).
- **Mutex Ownership**: None (relies on calling container).

---

### 1.4 Tree Search: `KDNode` (struct) & `KDTree` (class)

#### `KDNode` (struct)
- **Lines**: `main.cpp:97-102`
- **Purpose**: Binary tree node for KD-tree spatial partitioning.
- **Public Surface**:
  ```cpp
  struct KDNode {
      VectorItem item;
      KDNode* left;
      KDNode* right;
      explicit KDNode(const VectorItem& v);
  };
  ```
- **Private Members**: None.
- **Mutex Ownership**: None.

#### `KDTree` (class)
- **Lines**: `main.cpp:104-159`
- **Purpose**: Space-partitioning $k$-d tree index for axis-aligned spatial search.
- **Public Surface**:
  ```cpp
  class KDTree {
  public:
      explicit KDTree(int d);
      ~KDTree();
      void insert(const VectorItem& v);
      std::vector<std::pair<float,int>> knn(const std::vector<float>& q, int k, DistFn dist);
      void rebuild(const std::vector<VectorItem>& items);
  };
  ```
- **Private Members**:
  ```cpp
  KDNode* root;
  int dims;
  void destroy(KDNode* n);
  KDNode* ins(KDNode* n, const VectorItem& v, int d);
  void knn(KDNode* n, const std::vector<float>& q, int k, int d, DistFn dist,
           std::priority_queue<std::pair<float,int>>& heap);
  ```
- **Mutex Ownership**: None.

---

### 1.5 Graph Search: `HNSW` (class)

- **Lines**: `main.cpp:165-331`
- **Purpose**: Multi-layer Hierarchical Navigable Small World graph index for approximate nearest neighbor search.
- **Public Surface**:
  ```cpp
  class HNSW {
  public:
      struct GraphInfo {
          int topLayer, nodeCount;
          std::vector<int> nodesPerLayer, edgesPerLayer;
          struct NV { int id; std::string metadata, category; int maxLyr; };
          struct EV { int src, dst, lyr; };
          std::vector<NV> nodes;
          std::vector<EV> edges;
      };

      HNSW(int m = 16, int efBuild = 200);
      void insert(const VectorItem& item, DistFn dist);
      std::vector<std::pair<float,int>> knn(const std::vector<float>& q, int k, int ef, DistFn dist);
      void remove(int id);
      GraphInfo getInfo();
      size_t size() const;
  };
  ```
- **Private Members**:
  ```cpp
  struct Node {
      VectorItem item;
      int maxLyr;
      std::vector<std::vector<int>> nbrs;
  };

  std::unordered_map<int, Node> G;
  int M, M0, ef_build;
  float mL;
  int topLayer;
  int entryPt;
  std::mt19937 rng;

  int randLevel();
  std::vector<std::pair<float,int>> searchLayer(const std::vector<float>& q, int ep, int ef, int lyr, DistFn dist);
  std::vector<int> selectNbrs(std::vector<std::pair<float,int>>& cands, int maxM);
  ```
- **Mutex Ownership**: None.

---

### 1.6 Vector Database Engine: `VectorDB` (class)

- **Lines**: `main.cpp:337-429`
- **Purpose**: Thread-safe unified multi-index (BruteForce + KDTree + HNSW) container for fixed-dimension vectors with query benchmarking and graph introspection.
- **Public Surface**:
  ```cpp
  class VectorDB {
  public:
      const int dims;
      struct Hit { int id; std::string meta, cat; std::vector<float> emb; float dist; };
      struct SearchOut { std::vector<Hit> hits; long long us; std::string algo, metric; };
      struct BenchOut { long long bfUs, kdUs, hnswUs; int n; };

      explicit VectorDB(int d);
      int insert(const std::string& meta, const std::string& cat, const std::vector<float>& emb, DistFn dist);
      bool remove(int id);
      SearchOut search(const std::vector<float>& q, int k, const std::string& metric, const std::string& algo);
      BenchOut benchmark(const std::vector<float>& q, int k, const std::string& metric);
      std::vector<VectorItem> all();
      HNSW::GraphInfo hnswInfo();
      size_t size();
  };
  ```
- **Private Members**:
  ```cpp
  std::unordered_map<int, VectorItem> store;
  BruteForce bf;
  KDTree kdt;
  HNSW hnsw;
  std::mutex mu;
  int nextId;
  ```
- **Mutex Ownership**: Owns `std::mutex mu`. Held via `std::lock_guard` across all public member functions.

---

### 1.7 JSON & HTTP Parsing Helpers (Free Functions)

- **`jS`** (`main.cpp:435-446`): JSON string escaping and quote wrapping.
- **`jVec`** (`main.cpp:448-455`): Float vector to JSON array formatting with 4 decimal places.
- **`parseVec`** (`main.cpp:457-463`): Parses comma-delimited numeric string into `std::vector<float>`.
- **`extractStr`** (`main.cpp:466-492`): Simple string scanner extracting JSON string fields with escape resolution.
- **`extractInt`** (`main.cpp:495-501`): Simple string scanner extracting integer JSON fields with fallback default.
- **`parseBody`** (`main.cpp:503-519`): Parses `metadata`, `category`, and `embedding` array from JSON body for `/insert`.
- **`cors`** (`main.cpp:521-525`): Injects CORS headers into `httplib::Response`.

---

### 1.8 Text Chunker (Free Function)

#### `chunkText`
- **Lines**: `main.cpp:531-552`
- **Purpose**: Splits raw text into whitespace-delimited word chunks with configurable sliding window overlap.
- **Public Surface**:
  ```cpp
  std::vector<std::string> chunkText(const std::string& text, int chunkWords = 250, int overlapWords = 30);
  ```
- **Private Members**: N/A.
- **Mutex Ownership**: None.

---

### 1.9 Ollama Client: `OllamaClient` (class)

- **Lines**: `main.cpp:561-638`
- **Purpose**: Synchronous HTTP client wrapping Ollama REST API endpoints (`/api/tags`, `/api/embeddings`, `/api/generate`).
- **Public Surface**:
  ```cpp
  class OllamaClient {
  public:
      std::string embedModel; // default "nomic-embed-text"
      std::string genModel;   // default "llama3.2"
      OllamaClient(const std::string& h = "127.0.0.1", int p = 11434);
      bool isAvailable();
      std::vector<float> embed(const std::string& text);
      std::string generate(const std::string& prompt);
  };
  ```
- **Private Members**:
  ```cpp
  std::string host;
  int port;
  std::string esc(const std::string& s);
  std::vector<float> parseEmbedding(const std::string& body);
  std::string parseResponse(const std::string& body);
  ```
- **Mutex Ownership**: None (instantiates per-call `httplib::Client`).

---

### 1.10 Document Store & Semantic Search: `DocItem` (struct) & `DocumentDB` (class)

#### `DocItem` (struct)
- **Lines**: `main.cpp:644-649`
- **Purpose**: Container for a text document chunk and its precomputed dense embedding.
- **Public Surface**:
  ```cpp
  struct DocItem {
      int id;
      std::string title;
      std::string text;
      std::vector<float> emb;
  };
  ```
- **Private Members**: None.
- **Mutex Ownership**: None.

#### `DocumentDB` (class)
- **Lines**: `main.cpp:651-711`
- **Purpose**: Thread-safe document repository providing cosine semantic search via HNSW (with BruteForce fallback for $< 10$ documents) and dynamic dimension discovery.
- **Public Surface**:
  ```cpp
  class DocumentDB {
  public:
      DocumentDB();
      int insert(const std::string& title, const std::string& text, const std::vector<float>& emb);
      std::vector<std::pair<float, DocItem>> search(const std::vector<float>& q, int k, float max_dist = 0.7f);
      bool remove(int id);
      std::vector<DocItem> all();
      size_t size();
      int getDims();
  };
  ```
- **Private Members**:
  ```cpp
  std::unordered_map<int, DocItem> store;
  HNSW hnsw;
  BruteForce bf;
  std::mutex mu;
  int nextId;
  int dims;
  ```
- **Mutex Ownership**: Owns `std::mutex mu`. Locked in `insert`, `search`, `remove`, `all`, `size`. Unlocked in `getDims` (read race).

---

### 1.11 Demo Data (Free Function)

#### `loadDemo`
- **Lines**: `main.cpp:717-760`
- **Purpose**: Seeds `VectorDB` with 20 predefined 16-dimensional vectors across 4 categories (`cs`, `math`, `food`, `sports`) using cosine distance.
- **Public Surface**:
  ```cpp
  void loadDemo(VectorDB& db);
  ```
- **Private Members**: N/A.
- **Mutex Ownership**: None.

---

### 1.12 Application Entry & HTTP Server: `main` (free function)

- **Lines**: `main.cpp:766-1089`
- **Purpose**: Instantiates `VectorDB`, `DocumentDB`, and `OllamaClient`, populates demo data, checks Ollama connectivity, binds 14 HTTP route handlers, and launches `httplib::Server` on port `8080`.
- **Public Surface**:
  ```cpp
  int main();
  ```
- **Routes Registered**:
  - `OPTIONS .*` (`main.cpp:785`): Global CORS preflight (204).
  - `GET /search` (`main.cpp:791`): Demo vector $k$-NN search.
  - `POST /insert` (`main.cpp:821`): Demo vector insertion.
  - `DELETE /delete/:id` (`main.cpp:831`): Demo vector deletion by regex match.
  - `GET /items` (`main.cpp:839`): List all demo vectors.
  - `GET /benchmark` (`main.cpp:855`): Runs side-by-side search over BruteForce, KDTree, HNSW.
  - `GET /hnsw-info` (`main.cpp:871`): Introspects HNSW graph topology (layers, nodes, edges).
  - `POST /doc/insert` (`main.cpp:905`): Chunks text, computes Ollama embeddings, saves to `DocumentDB`.
  - `DELETE /doc/delete/:id` (`main.cpp:940`): Deletes document chunk by ID.
  - `GET /doc/list` (`main.cpp:949`): Lists stored document chunks with preview text and word counts.
  - `POST /doc/search` (`main.cpp:971`): Semantic retrieval for UI visualizer.
  - `POST /doc/ask` (`main.cpp:1000`): Full RAG pipeline: embeds question, retrieves chunks, builds prompt, generates answer via Ollama.
  - `GET /status` (`main.cpp:1052`): System and model health check.
  - `GET /stats` (`main.cpp:1067`): Vector index summary statistics.
  - `GET /` (`main.cpp:1078`): Static file server for `index.html`.

---

## 2. Include Surface

| Line | Include Directive | Purpose / Usage in Code |
|:---|:---|:---|
| `main.cpp:1` | `#include "httplib.h"` | Single-header HTTP server (`httplib::Server`) and client (`httplib::Client`). |
| `main.cpp:2` | `#include <iostream>` | Terminal logging at startup (`std::cout`, `std::endl`). |
| `main.cpp:3` | `#include <vector>` | Dynamic arrays (`std::vector<float>` embeddings, item lists, neighbour lists). |
| `main.cpp:4` | `#include <string>` | Text manipulation, JSON parsing/formatting, route patterns (`std::string`). |
| `main.cpp:5` | `#include <algorithm>` | Algorithms: `std::sort`, `std::min`, `std::max`, `std::remove`, `std::remove_if`, `std::count`. |
| `main.cpp:6` | `#include <cmath>` | Math routines: `std::sqrt`, `std::abs`, `std::log`, `std::floor`. |
| `main.cpp:7` | `#include <random>` | Random level generation in HNSW (`std::mt19937`, `std::uniform_real_distribution`). |
| `main.cpp:8` | `#include <chrono>` | Microsecond search latency timing (`std::chrono::high_resolution_clock`). |
| `main.cpp:9` | `#include <mutex>` | Thread synchronization (`std::mutex`, `std::lock_guard`). |
| `main.cpp:10` | `#include <unordered_map>` | Key-value stores for `VectorDB`, `DocumentDB`, `HNSW::G`, and visited sets (`std::unordered_map`). |
| `main.cpp:11` | `#include <queue>` | Priority queues / min-max heaps for $k$-NN search (`std::priority_queue`). |
| `main.cpp:12` | `#include <set>` | **Unused include** (included but never referenced in code). |
| `main.cpp:13` | `#include <sstream>` | String stream construction for JSON responses (`std::ostringstream`, `std::istringstream`). |
| `main.cpp:14` | `#include <iomanip>` | Floating-point formatting formatting (`std::setprecision`, `std::fixed`). |
| `main.cpp:15` | `#include <functional>` | Distance function wrapper type (`std::function<...>`). |
| `main.cpp:16` | `#include <fstream>` | File stream reading for `index.html` (`std::ifstream`, `std::istreambuf_iterator`). |
| `main.cpp:17` | `#include <climits>` | **Unused include** (included but never referenced in code). |

---

## 3. Compile-Time & Runtime Constants

| Constant / Parameter | Value | Location | Description & Impact |
|:---|:---|:---|:---|
| `DIMS` | `16` | `main.cpp:19` | Global static dimension for demo vectors in `VectorDB`. |
| `HNSW::M` | `16` (default) | `main.cpp:227` | Max outgoing edges per node at layers $> 0$. |
| `HNSW::M0` | `32` (`2 * M`) | `main.cpp:228` | Max outgoing edges per node at layer 0 (ground layer). |
| `HNSW::ef_build` | `200` (default) | `main.cpp:227` | Search beam width during HNSW node insertion. |
| `HNSW::mL` | `1.0f / ln(16)` $\approx 0.36067$ | `main.cpp:229` | Normalization factor for random level distribution $- \ln(\text{uniform}(0,1)) \cdot mL$. |
| `HNSW::rng seed` | `42` | `main.cpp:229` | Fixed seed for `std::mt19937` pseudo-random number generator (deterministic graph levels). |
| `HNSW query ef` | `50` | `main.cpp:382, 408, 684` | Hardcoded search beam width `ef` in `VectorDB::search`, `VectorDB::benchmark`, and `DocumentDB::search`. |
| `cosine epsilon` | `1e-9f` | `main.cpp:50` | Norm threshold under which vectors are treated as zero vectors (returning distance `1.0f`). |
| `chunkWords` | `250` (default) | `main.cpp:532` | Word count per text chunk in `chunkText`. |
| `overlapWords` | `30` (default) | `main.cpp:532` | Word overlap between adjacent chunks in `chunkText`. |
| `Ollama host` | `"127.0.0.1"` | `main.cpp:604` | Default Ollama service hostname. |
| `Ollama port` | `11434` | `main.cpp:604` | Default Ollama service port. |
| `embedModel` | `"nomic-embed-text"` | `main.cpp:601` | Default embedding model name passed to Ollama `/api/embeddings`. |
| `genModel` | `"llama3.2"` | `main.cpp:602` | Default LLM generation model name passed to Ollama `/api/generate`. |
| `Ollama ping timeout` | `2s` connect | `main.cpp:609` | Timeout for `/api/tags` health check. |
| `Ollama embed timeout`| `3s` connect, `30s` read | `main.cpp:617-618` | Timeouts for embedding requests. |
| `Ollama gen timeout`  | `3s` connect, `180s` read | `main.cpp:628-629` | Timeouts for LLM generation requests. |
| `DocumentDB fallback` | `< 10` items | `main.cpp:682` | Threshold where `DocumentDB::search` uses `BruteForce::knn` instead of `HNSW::knn`. |
| `max_dist` | `0.7f` (default) | `main.cpp:678` | Distance cutoff threshold in `DocumentDB::search` (drops hits with distance $> 0.7$). |
| `preview length` | `120` chars | `main.cpp:957` | Truncation limit for text preview strings in `/doc/list`. |
| `default search k` | `5` (demo), `3` (doc) | `main.cpp:798, 974, 1003` | Default neighbor count returned when `k` parameter is omitted. |
| `HTTP listen host` | `"0.0.0.0"` | `main.cpp:1087` | Server binding IP interface. |
| `HTTP listen port` | `8080` | `main.cpp:1087` | Server binding TCP port. |

---

## 4. Call Graph (Internal Adjacency List)

```text
BruteForce::knn                  -> DistFn (dynamic dispatch: euclidean | cosine | manhattan)
BruteForce::remove               -> std::remove_if

KDTree::~KDTree                  -> KDTree::destroy
KDTree::destroy                  -> KDTree::destroy (recursive), delete
KDTree::insert                   -> KDTree::ins
KDTree::ins                      -> KDNode::KDNode, KDTree::ins (recursive)
KDTree::knn (public)             -> KDTree::knn (private)
KDTree::knn (private)            -> DistFn (dynamic dispatch), KDTree::knn (recursive)
KDTree::rebuild                  -> KDTree::destroy, KDTree::insert

HNSW::HNSW                       -> std::log
HNSW::randLevel                  -> std::uniform_real_distribution, std::log, std::floor
HNSW::selectNbrs                 -> std::min
HNSW::searchLayer                -> DistFn (dynamic dispatch)
HNSW::insert                     -> HNSW::randLevel, HNSW::searchLayer, HNSW::selectNbrs, DistFn (dynamic dispatch)
HNSW::knn                        -> HNSW::searchLayer
HNSW::remove                     -> std::remove
HNSW::getInfo                    -> (constructs GraphInfo)

VectorDB::VectorDB               -> KDTree::KDTree, HNSW::HNSW
VectorDB::insert                 -> VectorItem::VectorItem, BruteForce::insert, KDTree::insert, HNSW::insert
VectorDB::remove                 -> BruteForce::remove, HNSW::remove, KDTree::rebuild
VectorDB::search                 -> getDistFn, BruteForce::knn, KDTree::knn, HNSW::knn
VectorDB::benchmark              -> getDistFn, BruteForce::knn, KDTree::knn, HNSW::knn
VectorDB::all                    -> (iterates store)
VectorDB::hnswInfo               -> HNSW::getInfo
VectorDB::size                   -> (queries store.size)

OllamaClient::isAvailable        -> httplib::Client::Get
OllamaClient::embed              -> OllamaClient::esc, httplib::Client::Post, OllamaClient::parseEmbedding
OllamaClient::generate           -> OllamaClient::esc, httplib::Client::Post, OllamaClient::parseResponse
OllamaClient::parseEmbedding     -> parseVec
OllamaClient::parseResponse      -> extractStr

DocumentDB::DocumentDB           -> HNSW::HNSW
DocumentDB::insert               -> DocItem::DocItem, VectorItem::VectorItem, HNSW::insert, BruteForce::insert, cosine
DocumentDB::search               -> BruteForce::knn, HNSW::knn, cosine
DocumentDB::remove               -> HNSW::remove, BruteForce::remove
DocumentDB::all                  -> (iterates store)
DocumentDB::size                 -> (queries store.size)
DocumentDB::getDims              -> (reads dims)

loadDemo                         -> getDistFn, VectorDB::insert
parseBody                        -> extractStr, parseVec

main                             -> VectorDB::VectorDB, DocumentDB::DocumentDB, OllamaClient::OllamaClient
main                             -> loadDemo
main                             -> OllamaClient::isAvailable
main                             -> httplib::Server::Server, httplib::Server::Options, httplib::Server::Get, httplib::Server::Post, httplib::Server::Delete, httplib::Server::listen
main (route: /search)            -> cors, parseVec, VectorDB::search, jS, jVec
main (route: /insert)            -> cors, parseBody, getDistFn, VectorDB::insert
main (route: /delete/:id)        -> cors, VectorDB::remove
main (route: /items)             -> cors, VectorDB::all, jS, jVec
main (route: /benchmark)         -> cors, parseVec, getDistFn, VectorDB::benchmark
main (route: /hnsw-info)         -> cors, VectorDB::hnswInfo, jS
main (route: /doc/insert)        -> cors, extractStr, chunkText, OllamaClient::embed, DocumentDB::insert, DocumentDB::getDims
main (route: /doc/delete/:id)    -> cors, DocumentDB::remove
main (route: /doc/list)          -> cors, DocumentDB::all, jS
main (route: /doc/search)        -> cors, extractStr, extractInt, OllamaClient::embed, DocumentDB::search, jS
main (route: /doc/ask)           -> cors, extractStr, extractInt, OllamaClient::embed, DocumentDB::search, OllamaClient::generate, jS, DocumentDB::size
main (route: /status)            -> cors, OllamaClient::isAvailable, DocumentDB::size, DocumentDB::getDims, VectorDB::size, jS
main (route: /stats)             -> cors, VectorDB::size
main (route: /)                  -> std::ifstream (reads index.html)
```

---

## 5. State Ownership & Memory Lifecycle

```
main()
 ├── VectorDB db (stack, lives for process lifetime)
 │    ├── std::unordered_map<int, VectorItem> store (owns VectorItem copies by value)
 │    ├── BruteForce bf
 │    │    └── std::vector<VectorItem> items (owns duplicate VectorItem copies by value)
 │    ├── KDTree kdt
 │    │    └── KDNode* root (owns heap-allocated KDNode tree; each node contains VectorItem by value)
 │    └── HNSW hnsw
 │         └── std::unordered_map<int, Node> G (owns HNSW::Node copies by value; each contains VectorItem)
 │
 ├── DocumentDB docDB (stack, lives for process lifetime)
 │    ├── std::unordered_map<int, DocItem> store (owns DocItem by value, including text and emb)
 │    ├── HNSW hnsw
 │    │    └── std::unordered_map<int, Node> G (owns VectorItem copy with category="doc")
 │    └── BruteForce bf
 │         └── std::vector<VectorItem> items (owns VectorItem copy)
 │
 └── OllamaClient ollama (stack, stateless configuration)
```

### Allocation & Copy Semantics:
1. **Redundant Copies in `VectorDB`**:
   - Inserting an item into `VectorDB` results in **4 independent copies** of the `VectorItem` (including its `std::vector<float>` vector payload):
     1. In `VectorDB::store`
     2. In `VectorDB::bf.items`
     3. In `VectorDB::kdt` (`KDNode::item`)
     4. In `VectorDB::hnsw.G` (`HNSW::Node::item`)
2. **Redundant Copies in `DocumentDB`**:
   - Inserting a document chunk stores the full payload in `DocumentDB::store` as `DocItem`, and creates **2 duplicate `VectorItem` records** (one in `HNSW::G` and one in `BruteForce::items`).
3. **Parameter Passing**:
   - Query vectors `const std::vector<float>& q` and items `const VectorItem& v` are passed by `const &` into search/insert methods, but value-copied upon insertion into maps/vectors/nodes.
   - Return values from `all()`, `search()`, and `benchmark()` are returned by value (moving or copying).

---

## 6. Concurrency Surface

### 6.1 Mutex Inventory

| Mutex Identifier | Declared In | Type | Guard Scope / Protected Members |
|:---|:---|:---|:---|
| `VectorDB::mu` | `main.cpp:342` | `std::mutex` | Protects `store`, `bf`, `kdt`, `hnsw`, `nextId`. Acquired in `insert`, `remove`, `search`, `benchmark`, `all`, `hnswInfo`, `size`. |
| `DocumentDB::mu` | `main.cpp:655` | `std::mutex` | Protects `store`, `hnsw`, `bf`, `nextId`, `dims`. Acquired in `insert`, `search`, `remove`, `all`, `size`. |

### 6.2 Lock Scope Analysis & Blocking Operations

- **External Network Calls**:
  - `OllamaClient::embed` and `OllamaClient::generate` are **always invoked outside any mutex lock** in the HTTP route handlers (`/doc/insert`, `/doc/search`, `/doc/ask`).
  - Route handlers perform embedding/generation on the worker thread, then acquire `DocumentDB::mu` only during index insertion or index lookup.
- **File I/O**:
  - `GET /` opens and reads `index.html` via `std::ifstream` without any locks.
- **Heavy In-Lock Computations**:
  - `VectorDB::remove(id)` holds `VectorDB::mu` while executing `kdt.rebuild(rem)`, which tears down the entire KD-tree and reconstructs it from scratch via $O(N \log N)$ or $O(N^2)$ sequential insertions.
  - `VectorDB::benchmark(...)` holds `VectorDB::mu` while sequentially executing `BruteForce::knn`, `KDTree::knn`, and `HNSW::knn` back-to-back.
- **Lockless Access (Data Race)**:
  - `DocumentDB::getDims()` (`main.cpp:710`) reads member `dims` without acquiring `DocumentDB::mu`, whereas `DocumentDB::insert` writes to `dims` under `DocumentDB::mu`.

---

## 7. Source File Inventory

| File Path | Lines | Type / Role | Build Status |
|:---|:---|:---|:---|
| `C:\Users\Samar\Documents\Your-OWN-AI\main.cpp` | 1089 (1106 with trailing blank lines) | Primary C++ application source (server, indexes, RAG). | Standalone executable entry point. |
| `C:\Users\Samar\Documents\Your-OWN-AI\httplib.h` | 16999 | Vendored single-header library (`yhirose/cpp-httplib` v0.15.3). | Header-only dependency. |
| `C:\Users\Samar\Documents\Your-OWN-AI\index.html` | ~500 | Web frontend (HTML/JS UI visualizer). | Static asset served on `GET /`. |

*Note: There are **no build files** present in the repository (no `CMakeLists.txt`, `Makefile`, `meson.build`, `BUILD`, or Visual Studio project files). The original code was designed to be compiled directly via `g++ -O3 -std=c++17 main.cpp -lpthread -o server` or `clang++`.*

---

## 8. Architectural Quirks, Footguns & Migration Risks

1. **Unbalanced KD-Tree Degeneration**:
   - `KDTree::insert` inserts nodes sequentially with alternating axis cuts $d \pmod{\text{dims}}$. If vectors are inserted in correlated/sorted order, the KD-tree degenerates into an $O(N)$ linked list.
2. **KD-Tree Deletion Triggers Full Rebuild**:
   - `KDTree` lacks a node deletion or node rotation algorithm. `VectorDB::remove()` destroys the entire root and re-inserts all surviving nodes into a new tree while holding the primary mutex.
3. **HNSW Entry Point Orphan on Deletion**:
   - When deleting a node in `HNSW::remove(id)` (`main.cpp:292-296`), if `entryPt == id`, it reassigns `entryPt` to the first available key in map `G` (`for (auto& [nid, nd] : G) if (nid != id) { entryPt = nid; break; }`). It **never checks** if the new entry point resides on `topLayer`, nor does it update `topLayer`. If the deleted entry point was the sole node at `topLayer`, subsequent `searchLayer` operations will traverse empty layers.
4. **HNSW `topLayer` Monotonicity**:
   - `topLayer` starts at `-1` and only increases upon insertion; it never decreases on deletion.
5. **Ad-Hoc String JSON Parsing**:
   - Functions `extractStr`, `extractInt`, and `parseVec` parse JSON via naive string scans (`std::string::find`). They lack full RFC 8259 compliance and cannot handle nested objects, arbitrary whitespace around delimiters, or complex JSON escaping.
6. **Dynamic Dimension Mismatch Vulnerability in `DocumentDB`**:
   - `DocumentDB` locks in its vector dimension `dims = (int)emb.size()` on the first insertion. Subsequent insertions do not validate vector dimensionality. Inserting an embedding of different size causes undefined behavior or calculation errors in `cosine()`.
7. **Dual-Index Redundancy in `DocumentDB`**:
   - `DocumentDB` maintains both an `HNSW` graph and a `BruteForce` list. When query count $< 10$, it queries `BruteForce`; otherwise, it queries `HNSW` with hardcoded beam width `ef = 50`.
8. **Static Working Directory Dependency**:
   - The root route `GET /` opens `index.html` via a relative file path (`std::ifstream f("index.html")`). If the binary is started from any directory other than the project root, `GET /` returns HTTP 404.
9. **Prompt Template Hardcoding**:
   - The system prompt in `POST /doc/ask` (`main.cpp:1024-1031`) contains explicit instructions forbidding the LLM from referencing the word "context":
     ```text
     "You are a helpful assistant. Answer the user's question directly. Use the provided context if it contains relevant information. If it doesn't, just use your own general knowledge. IMPORTANT: Do NOT mention the 'context', 'provided text', or say things like 'the context doesn't mention'. Just answer the question naturally."
     ```
   - This exact system instruction must be preserved in the Python migration to maintain identical generation behavior.
