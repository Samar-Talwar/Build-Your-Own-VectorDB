# Risks & Open Decisions — C++ → Python Migration

> **Status:** Planning phase. No code has been written. The C++ source is untouched.
> **Companion docs:** [01-architecture](01-architecture.md), [02-dependencies](02-dependencies.md) (pending), [03-api-contract](03-api-contract.md) (pending), [04-algorithm-parity](04-algorithm-parity.md) (pending), [05-module-decomposition](05-module-decomposition.md) (pending), [06-test-strategy](06-test-strategy.md) (pending), [07-execution-plan](07-execution-plan.md) (pending).

---

## 1. Risk Register

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| R1 | **HNSW graph topology drift** when porting. The C++ uses a hand-rolled algorithm with a specific RNG, layer formula, and `selectNbrs` simple heuristic. If the port uses `hnswlib` (which has a different neighbor-selection heuristic and RNG), `/hnsw-info` will return a different graph shape and a different `topLayer` distribution. | **High** (if using `hnswlib`) / **Low** (if hand-port) | Medium. The frontend renders the graph, but it can be drawn for any valid HNSW. Answer correctness for `/search` is preserved. | Default: use `hnswlib`. Document the visual difference. If bit-exact graph is required, hand-port (≈200 lines). Provide a feature flag in code to swap. |
| R2 | **JSON wire-format precision drift.** The C++ uses `setprecision(4)` for embeddings, `setprecision(6)` for `/search` distances, `setprecision(4)` for `/doc/*` distances. Python's `json.dumps` will print `0.9` not `0.9000`, breaking byte-equality. | **High** (if naively used) | High. Frontend parses these numbers; `parseFloat` is forgiving but visual diffs and downstream consumers may not be. | Use `round(x, 4)` and `round(x, 6)` on every numeric field, then `json.dumps` with `separators=(',', ':')` to match `std::ostringstream` whitespace. Add a parity test that diffs response bodies. |
| R3 | **Demo data drift.** The 20 hand-tuned 16D vectors at `main.cpp:720-759` are the visible product (scatter plot). If a port regenerates them or rounds them, the cluster shape changes. | Medium | High (visual regression). | Copy the 20 vectors verbatim into a `demo_data.py` module. Do not regenerate. Verify with a byte-by-byte diff. |
| R4 | **Ollama blocking under load.** `/doc/insert` makes N sequential synchronous calls to `/api/embeddings` (one per chunk). For a long document (10 chunks × 1s each = 10s) this holds the HTTP worker thread. With multi-threaded Flask (`threaded=True`), two concurrent `/doc/insert` requests can race. | Medium | Medium. Behavior matches C++ (which is also single-threaded). | For v1, accept the same blocking behavior as C++. If multiple workers are needed, switch to async (FastAPI + httpx async) or thread pool. |
| R5 | **KD-Tree degeneration on edge-case inputs.** The C++ KD-Tree is axis-rotating but lacks rebalancing. With adversarial input it degenerates to O(N) per query. | Low (demo data is well-distributed) | Low (frontend won't notice at N=20) | Preserve as-is in v1. If the port uses `scipy.spatial.KDTree`, this is solved automatically (balanced construction). |
| R6 | **HNSW entry-point-orphan bug** (architect report #3). When the entry point is deleted, `topLayer` is not decremented. The C++ has this bug. | Low (only triggered by deleting a high-layer node) | Low. The C++ search still works because `topLayer` is only used as a starting range; `searchLayer` returns empty for empty layers. | **Do not fix in v1.** Preserve the bug to avoid behavioral drift. Optionally add a TODO. |
| R7 | **HNSW `topLayer` monotonicity bug** (architect report #4). `topLayer` only ever increases. | Low | Low. | Preserve. |
| R8 | **DocumentDB `getDims` data race** (architect report, §6.2). `getDims()` reads `dims` without a lock while `insert` writes it. | Low (rarely concurrent in practice) | Low. | The Python port naturally uses a single global `dims` attribute; either set it once at insert time under a lock, or accept the same race by using `nonlocal` with no lock. Match C++ semantics. |
| R9 | **Static working directory for `index.html`**. The C++ `GET /` opens `index.html` with a relative path; if the binary is started outside the project root, the UI 404s. The Python port inherits this. | Low | Low. | Preserve behavior. Document in README that the user must `cd` to the project root. |
| R10 | **CORS hard-coded to `*`.** The C++ allows all origins. | None | None (intentional). | Preserve. |
| R11 | **No persistence.** All state is in-memory; restart loses documents. | None | None (intentional, matches C++). | Preserve. Optional v2: add `pickle` snapshot on shutdown. |
| R12 | **httplib v0.42.0 vs Flask feature gaps.** httplib's regex routes (`/delete/(\d+)`) become Flask path converters (`/delete/<int:id>`). CORS, query params, body parsing, and content-type setting all map cleanly. | Low | Low. | None. |
| R13 | **Float32 vs float64 in numpy.** The C++ uses `float` (32-bit). numpy defaults to float64. Distance calculations will differ in the last few bits. | High | Low (within `setprecision(4)` rounding, irrelevant). | Use `np.float32` throughout the index to match. Or accept the drift — `setprecision(4)` masks it. |
| R14 | **RAG prompt template drift.** The prompt at `main.cpp:1024-1031` is hard-coded with a specific anti-meta-instruction ("Do NOT mention the 'context'"). Drift here will change answer style. | Medium | Medium. | Copy the prompt verbatim. Add a constant `RAG_SYSTEM_PROMPT`. |
| R15 | **Seed determinism in tests.** The C++ HNSW seed is 42. If the port doesn't seed identically, `/hnsw-info` will differ between runs of the C++ binary and the Python binary. | High | Low (the C++ is also deterministic, so the *Python* runs will be reproducible — just not matching the C++ graph). | Document. Make parity tests skip the HNSW graph-diff assertion, or only run parity tests against the C++ baseline captured once. |
| R16 | **Threading model change.** C++ uses a single-threaded `httplib::Server`. Flask dev server is also single-threaded by default but supports `threaded=True`. The lock semantics in `VectorDB` and `DocumentDB` are only needed in the multi-threaded case. | Low | Low. | Start with `threaded=False` to match C++. Add a `threading.Lock` only if `threaded=True` is enabled. |
| R17 | **MCP / Ruflo toolchain coupling.** The `.mcp.json` references `claude-flow` and `ruv-swarm` MCP servers. If the Python server is started in an environment where these are also bound, port conflicts are possible. | None | None. | MCP servers are not started by the VectorDB app. Unrelated. |
| R18 | **`/doc/ask` returns HTTP 200 even when `OllamaClient::generate` fails.** The error string is stuffed into the `answer` field; the frontend's `if (d.error)` check misses it. | High (LLM outages are routine) | High (silent corruption — UI shows model output that is actually an error message). | **Preserve the bug** in v1: return HTTP 200 with `answer: errorString` exactly. v2 candidate: detect at the route layer and return 502. Document with a `# ponytail: silent 200 on LLM failure — preserved to match C++` comment at the route. |
| R19 | **Two distinct Ollama error strings.** `/doc/insert` returns the long message with install instructions; `/doc/search` and `/doc/ask` return the short `"Ollama unavailable"`. Drift here breaks the frontend's parser. | Medium | Medium. | Define as constants `OLLAMA_ERROR_LONG` / `OLLAMA_ERROR_SHORT` in `ollama.py`. Parity test asserts the exact bytes. |

---

## 2. Open Decisions (must be locked before coding starts)

| # | Decision | Options | Default if no input | Rationale |
|---|---|---|---|---|
| **D1** | **HNSW implementation** | (a) `hnswlib` pip package, (b) hand-port from C++ (≈200 lines), (c) `nmslib` | **(b) hand-port** (per algorithm-spec-writer) | The algorithm-spec-writer recommends hand-port to preserve `/hnsw-info` JSON structure bit-exact (same keys, same edge-dedup rule, same `topLayer` semantics). The lead's initial default was `hnswlib` (90% less code, well-tested) but `/hnsw-info` numbers would drift. **Flag for user decision.** |
| **D2** | **Web framework** | (a) Flask, (b) FastAPI + uvicorn, (c) Starlette | **(a) Flask** | Closest spiritual match to `httplib::Server` (sync, simple, no opinion on async). FastAPI would force async-everywhere and change the blocking-Ollama pattern. |
| **D3** | **Python version** | (a) 3.11, (b) 3.12, (c) 3.13 | **(b) 3.12** | 3.11 is fine; 3.12 has `typing.override`, better error messages, and `numpy` 2.x compatibility. 3.13 is too new for `hnswlib` wheels. |
| **D4** | **HTTP client for Ollama** | (a) `requests`, (b) `httpx`, (c) the `ollama` PyPI package | **(a) `requests`** | 1:1 match for `httplib::Client`. `ollama` PyPI package would abstract away `/api/embeddings` but adds a dep and may not match the exact request shape. |
| **D5** | **KD-Tree** | (a) `scipy.spatial.KDTree`, (b) hand-port from C++ | **(a) `scipy.spatial.KDTree`** | ~10 lines vs ~60. KDTree results will be the same nearest neighbors for the demo data; tie-breaking may differ on adversarial input (irrelevant for the demo). |
| **D6** | **Brute force** | (a) numpy vectorized, (b) hand-port | **(a) numpy vectorized** | `np.linalg.norm` over rows. One-liner. |
| **D7** | **Project layout** | (a) flat (`main.py` only, like C++), (b) `py/vectordb/` package, (c) `src/vectordb/` | **(b) `py/vectordb/`** | The C++ is one ~1100-line file. The Python port will be ~600-800 lines across multiple files. A package is needed. |
| **D8** | **Persistence** | (a) none, (b) `pickle` on shutdown, (c) sqlite | **(a) none** | Matches C++ behavior exactly. v2 candidate. |
| **D9** | **Threading** | (a) `threaded=False`, (b) `threaded=True` with locks, (c) `gunicorn` with N workers | **(a) `threaded=False`** | Matches C++ behavior. Easier to reason about. Add (b) only if the user observes a bottleneck. |
| **D10** | **Tests** | (a) parity harness only, (b) parity + per-module unit tests, (c) full pytest suite | **(a) parity harness only** | The user explicitly asked for migration, not for adding test infrastructure. Parity tests directly verify the contract. Unit tests are a v2 add. |

---

## 3. Risk Heatmap (qualitative)

```
Impact ──►
  ▲
  │  R2 (JSON precision)   R1 (HNSW graph)   R3 (demo data)
  │  R13 (float32/double)  R4 (Ollama block) R14 (RAG prompt)
  │  R15 (seed determinism)
  │
  │  R5 (KD degeneracy)    R6 (entry orphan)
  │  R7 (topLayer mono)    R8 (dims race)
  │  R16 (threading)
  │
  │  R9 (CWD)  R10 (CORS)  R11 (persist)  R12 (Flask diffs)  R17 (MCP)
  └──────────────────────────────────────────────────────────────► Likelihood
```

The top-right quadrant (high impact, high likelihood) contains only R2 and R3. Both are mitigated by direct code copying — no new logic.

---

## 4. Out-of-scope (explicitly NOT migrated)

- C++17 language features (no need to enumerate)
- The C++ `httplib` source itself (replaced by Flask)
- The C++ KD-Tree degenerates-to-O(N) behavior (preserved as-is; not fixed)
- The C++ HNSW entry-point-orphan bug (preserved as-is)
- The C++ `DocumentDB::getDims` race (preserved as-is)
- The `index.html` frontend (NOT modified, NOT migrated)
- The C++ build system (Python doesn't need one; just a `requirements.txt` and a `python -m vectordb` entry)
- The `.claude/`, `.claude-flow/`, `.swarm/`, `CLAUDE.md`, `.mcp.json` files (dev tooling; not part of the app)
