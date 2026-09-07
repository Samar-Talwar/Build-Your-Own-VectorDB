# Dependency Map: C++ → Python Migration

**Source**: `C:\Users\Samar\Documents\Your-OWN-AI\main.cpp` (962 lines, single TU)
**Source header**: `C:\Users\Samar\Documents\Your-OWN-AI\httplib.h` (16,999 lines, vendored)
**Target**: Python 3.11+ async service (per `README.md` Step 1)

Cross-checked against `docs/migration/01-architecture.md`. `httplib` is **0.42.0** per the
source banner (`#define CPPHTTPLIB_VERSION "0.42.0"` at `httplib.h:11`), not the
0.15.3 the architect draft cited.

---

## 1. Build Toolchain (C++)

There is **no build system** in the repo. No `Makefile`, no `CMakeLists.txt`,
no `build.sh`, no `compile.bat`. Compilation is hand-invoked.

| Aspect | Value | Evidence |
|---|---|---|
| Language standard | C++17 (implied: `std::optional` not used; `std::filesystem` not used; structured bindings used at `main.cpp:198`) | `main.cpp` source |
| Source file count | 1 (`main.cpp`) | `Glob` of project root |
| Header | 1 vendored (`httplib.h`) | `Glob` of project root |
| Compiler (Windows) | MSVC or MinGW `g++` (per `README.md` troubleshooting: `undefined reference to WSA...` → `-lws2_32`) | `README.md:160` |
| Compiler (Linux) | `g++` or `clang++` | Standard for httplib |
| Link flags (Windows) | `-lws2_32` (Winsock), `-lpthread` (pthreads) | `README.md:160` + httplib docs |
| Link flags (Linux) | `-lpthread` | httplib docs |
| Optimization flag (mentioned) | `-O2` | `README.md:161` |
| Standard library | libstdc++ / MSVC STL | Implied by C++17 use |
| Build system | **none** | Repo has no `Makefile`, `CMakeLists.txt`, `meson.build`, etc. |

Implication for Python port: toolchain collapses to a `requirements.txt` + a
single `python main.py` command (per `README.md:155`). No make/CI recipe to
port.

---

## 2. Vendored C++ Deps

| Name | Version | License | Source URL | Why Vendored | Size (lines) | Where Used |
|---|---|---|---|---|---|---|
| cpp-httplib | 0.42.0 | MIT | https://github.com/yhirose/cpp-httplib | Single-header, no Boost, zero external deps — easier than adding a system lib on Windows | 16,999 | All HTTP server/client: `httplib::Server`, `httplib::Client`, `Request`, `Response` (`main.cpp:1`, `:604–637`, `:766–1088`) |

That is the **only** vendored dep. Everything else is stdlib.

---

## 3. Standard Library Headers

| `#include` | C++ Header | Python Stdlib / Runtime Equivalent | Substitute Needed? |
|---|---|---|---|
| `"httplib.h"` | (vendored) | `http.server` or FastAPI/Starlette/Flask | **Yes** — see §5 |
| `<iostream>` | C++ streams I/O | built-in `print()`, `logging` | No — stdlib has it |
| `<vector>` | dynamic array | built-in `list` | No |
| `<string>` | UTF-8 string | built-in `str` | No |
| `<algorithm>` | sort/find/transform | `sorted()`, `bisect`, custom | No |
| `<cmath>` | math (sqrt, abs, log) | `math` module | No — stdlib |
| `<random>` | PRNG (mt19937) | `random` module | No — stdlib |
| `<chrono>` | clocks/durations | `time.perf_counter_ns()`, `time.monotonic_ns()` (3.7+) | No — stdlib |
| `<mutex>` | mutual exclusion | `threading.Lock` / `asyncio.Lock` | No — stdlib |
| `<unordered_map>` | hash map | built-in `dict` (CPython 3.7+ preserves insertion order; not required for correctness here) | No |
| `<queue>` | priority queue | `heapq` | No — stdlib |
| `<set>` | ordered set | built-in `set` | No |
| `<sstream>` | stringstream | `io.StringIO` | No |
| `<iomanip>` | stream formatters (`setprecision`) | `f"{x:.6f}"` formatted strings | No |
| `<functional>` | `std::function` | first-class functions / `typing.Callable` | No |
| `<fstream>` | file I/O | `open()` / `pathlib.Path` | No |
| `<climits>` | integer limits (`INT_MAX` etc.) | `sys.maxsize`, `math.inf` | No |

**Substitute needed: only the HTTP layer.** Every other header maps 1:1 to a
Python stdlib module. This is what makes the port mechanical for ~95% of the
code; the HTTP / Ollama client is the only design decision.

---

## 4. Runtime Services

### 4.1 Ollama (local LLM host)

| Field | Value | Source |
|---|---|---|
| Host | `127.0.0.1` (localhost) | `main.cpp:604` `OllamaClient(host, port)` default |
| Port | `11434` | `main.cpp:604` |
| Embed model | `nomic-embed-text` (~274 MB) | `main.cpp:601`, `README.md:110` |
| Generate model | `llama3.2` (~2 GB) | `main.cpp:602`, `README.md:115` |
| Conn. timeout (embed) | 3 s | `main.cpp:617` |
| Conn. timeout (generate) | 3 s | `main.cpp:628` |
| Read timeout (embed) | 30 s | `main.cpp:618` |
| Read timeout (generate) | **180 s** (3 min) — LLMs are slow | `main.cpp:629` |
| Read timeout (health check) | 2 s | `main.cpp:609` |
| Embedding dims (output) | 768 (set by `nomic-embed-text`) | `README.md:18,30` |

### 4.2 Health check — `GET /api/tags`

- **Request**: `GET http://127.0.0.1:11434/api/tags`, no body.
- **Response**: JSON list of installed models. The C++ only checks `status == 200`; it does not parse the body (`main.cpp:607–612`).
- **Python port**: hit it with `httpx.get(..., timeout=2.0)`, treat 200 as up.

### 4.3 Embed — `POST /api/embeddings`

**Request body (exact C++ serialization at `main.cpp:619`):**

```json
{"model":"nomic-embed-text","prompt":"<text with JSON-escaped quotes/newlines>"}
```

Content-Type: `application/json`. The `esc()` helper JSON-escapes only `"` and
`\n` (manual, not full RFC-8259). This works for the inputs the server feeds
in (chunked document text, user questions) but is **not** general-purpose.
Python port should use `json.dumps({"model": ..., "prompt": text})` — safer and
behaves identically for in-spec inputs.

**Response shape (Ollama native, see Ollama docs; C++ parses with a hand-rolled
bracket-matcher at `main.cpp:580–593`):**

```json
{
  "embedding": [0.123, -0.456, ..., 0.789]   // ~768 floats
}
```

### 4.4 Generate — `POST /api/generate`

**Request body (exact at `main.cpp:630–632`):**

```json
{
  "model":"llama3.2",
  "prompt":"<full RAG prompt, JSON-escaped>",
  "stream":false
}
```

**Response shape (Ollama native, C++ pulls only the `response` field at `main.cpp:596–598`):**

```json
{"response":"<assistant text>"}
```

### 4.5 Prompt template (RAG)

Verbatim from `main.cpp:1023–1031` — the Python port must reproduce this
exactly for parity:

```
You are a helpful assistant. Answer the user's question directly.
Use the provided context if it contains relevant information.
If it doesn't, just use your own general knowledge.
IMPORTANT: Do NOT mention the 'context', 'provided text', or say things like 'the context doesn't mention'.
Just answer the question naturally.

Context:
[1] <title1>:
<text1>

[2] <title2>:
<text2>

...

Question: <user question>

Answer:
```

### 4.6 No other runtime services

- No database.
- No message queue.
- No auth server.
- Static `index.html` is served from disk on `GET /` (`main.cpp:1078–1085`).

---

## 5. Python Target Stack (recommendation)

| C++ construct | Recommendation | Justification |
|---|---|---|
| **`httplib::Server` (HTTP server)** | **FastAPI** (Uvicorn underneath) | Async-native (matches the Ollama 180 s timeout story without blocking a thread), OpenAPI for free (lets you sanity-test endpoints with Swagger UI), first-class Pydantic models for request/response bodies (the C++ body parsing at `main.cpp:524–579` is exactly the validation work Pydantic replaces). Flask is simpler but its WSGI thread model blocks on long calls — wrong for `/doc/ask`. Starlette is the lower layer underneath FastAPI; using it directly skips the schema-validations win for ~20% less code. FastAPI wins. |
| **`httplib::Client` (HTTP client → Ollama)** | **`httpx`** | Async client with the same `timeout=` / `post()` API as `httpx` (familiar). Sync mode works too if you want to mirror the C++ flow 1:1. `requests` is sync-only and would force the whole server into threads. The `ollama` PyPI package wraps the same endpoints but adds a layer of opinion and version coupling we don't need; we already know the exact request shape. **`httpx` keeps the contract visible in our code, which is the whole point of this exercise.** |
| **`std::vector<float>` + `<cmath>` (HNSW inner loop, distances)** | **NumPy** (vectors) + **pure Python** (control flow) | 768-dim dot products and L2 norms are the hot path. NumPy vectorizes them at ~100× CPython. The HNSW *graph traversal* (priority queue, neighbor selection) stays in pure Python because it's pointer-chasing and NumPy doesn't help. **No SciPy needed** — `scipy.spatial.distance` would do cosine/L2, but the HNSW hot path also reuses the distance as a Python callable (the C++ `DistFn = std::function<...>` at `main.cpp:33`), which is awkward over SciPy and trivial as `def dist(a, b): return ...`. |
| **HNSW implementation** | **Hand-port from `main.cpp:165–320`** | The C++ class is ~150 lines and self-contained. A faithful hand-port preserves the educational value (per the README's "from scratch" goal, `README.md:6`) and gives identical numbers for the parity harness. `hnswlib` (Apache-2.0) is faster and more correct in edge cases, but it is a black box — defeats the "from scratch" premise. **Trade-off accepted:** the hand-port won't match `hnswlib`'s recall on adversarial inputs, but for the demo corpus it will match the C++. Add `hnswlib` later if a customer brings a real workload. |
| **KD-Tree** | **Hand-port from `main.cpp:97–159`** | Same reason: 60 lines, matches the C++ output, no library that would do it more simply without pulling numpy/sklearn. |
| **Brute Force** | **Hand-port from `main.cpp:70–91`** | Trivial; one for-loop + `heapq.nsmallest` (5 lines). |
| **JSON serialization** | **stdlib `json`** | The C++ builds JSON by hand with `ostringstream` (`main.cpp:804–818`). Python `json.dumps` is faster, safer, and adds nothing to `requirements.txt`. |
| **Distance functions** | **Pure Python** (`math.sqrt`, sum-comprehensions) | Three functions, 15 lines total. No need for a library. |
| **PRNG (`std::mt19937`)** | **`random` module** | HNSW `randLevel()` uses `std::uniform_real_distribution`. `random.random()` is the 1:1 substitute; same statistical properties are not required for the parity harness. |
| **Thread-safety (`std::mutex`)** | **`threading.Lock`** (per-store) | The C++ guards `DocumentDB::store` and uses one global-ish model. FastAPI is async-single-thread per worker; if you need true parallelism, use `asyncio.Lock` (since the critical sections are I/O-light). **Pick `asyncio.Lock` — FastAPI workers are coroutine-cooperative and blocking locks would defeat the async model.** |
| **CORS** | **FastAPI `CORSMiddleware`** | Replaces the C++ `cors()` helper (`main.cpp` pre-handler) with one `add_middleware(...)` call. |
| **Static file serving (`index.html`)** | **FastAPI `StaticFiles`** or `FileResponse` | One line: `@app.get("/")` returning `FileResponse("index.html")`. |

---

## 6. Dependency Graph (post-port)

```
                    ┌──────────────────────┐
                    │   main.py (entry)    │
                    │   uvicorn / FastAPI  │
                    └──────────┬───────────┘
                               │ imports
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
       ┌─────────────┐  ┌──────────────┐  ┌──────────────┐
       │  api/routes │  │ ollama/client│  │  core/index  │
       │  .py        │  │ .py (httpx)  │  │  .py         │
       └──────┬──────┘  └──────┬───────┘  └──────┬───────┘
              │                │                 │
              │                │                 ├─► core/distance.py
              │                │                 ├─► core/brute_force.py
              │                │                 ├─► core/kdtree.py
              │                │                 └─► core/hnsw.py
              │                │                       │
              ▼                ▼                       ▼
       ┌─────────────┐  ┌──────────────┐         ┌──────────┐
       │ schemas.py  │  │   httpx      │         │  numpy   │
       │ (Pydantic)  │  │  (stdlib     │         │ (vectors)│
       └─────────────┘  │   + PyPI)    │         └──────────┘
                        └──────────────┘
                              │
                              ▼
                       ┌──────────────┐
                       │  Ollama at   │
                       │ 127.0.0.1:11434
                       └──────────────┘
```

Files, end to end:

- `main.py` — FastAPI app, lifespan, `uvicorn.run(...)`.
- `api/routes.py` — the 12 endpoints (`/search`, `/insert`, `/delete/{id}`,
  `/items`, `/benchmark`, `/hnsw-info`, `/doc/insert`, `/doc/delete/{id}`,
  `/doc/list`, `/doc/search`, `/doc/ask`, `/status`, `/stats`, `/`).
- `api/schemas.py` — Pydantic models for the 5 POST bodies
  (`/insert`, `/doc/insert`, `/doc/search`, `/doc/ask`, plus query params).
- `core/distance.py` — `euclidean`, `cosine`, `manhattan`, `get_metric`.
- `core/brute_force.py` — hand-ported `BruteForce`.
- `core/kdtree.py` — hand-ported `KDTree`.
- `core/hnsw.py` — hand-ported `HNSW` (the big one).
- `core/index.py` — `VectorDB` + `DocumentDB` facades, mirrors the C++ classes.
- `ollama/client.py` — `httpx`-backed `OllamaClient` with the same timeouts and
  request bodies as `main.cpp:600–637`.
- `index.html` — unchanged, served via `FileResponse`.

---

## 7. License & Supply-Chain Notes

| Package | License | Permissive? | Notes |
|---|---|---|---|
| cpp-httplib (vendored, replaced) | MIT | Yes | Replaced wholesale — no obligation on Python side. |
| fastapi | MIT | Yes | OK for any use. |
| uvicorn | BSD-3-Clause | Yes | OK. |
| pydantic | MIT | Yes | OK. |
| httpx | BSD-3-Clause | Yes | OK. |
| numpy | BSD-3-Clause | Yes | OK. |
| starlette (transitive via fastapi) | BSD-3-Clause | Yes | OK. |
| anyio (transitive) | MIT | Yes | OK. |
| h11 (transitive) | MIT | Yes | OK. |
| hnswlib (NOT used) | Apache-2.0 | Yes | Not added; listed for reference. |

**No GPL, AGPL, LGPL, or SSPL dependencies.** All licenses are MIT or
BSD-3-Clause, both of which permit closed-source distribution with
attribution. No copyleft concerns for any downstream use, including
commercial SaaS.

The `ollama` server itself is MIT-licensed; we are a *client* of it, so its
license does not bind our distribution.

No native extensions other than NumPy (which is BSD and ships wheels for
Windows / macOS / Linux on PyPI). No `gcc` / Visual Studio Build Tools
required to install dependencies on any of the three target platforms.

---

## 8. `requirements.txt` Draft

Pin to **major versions only** — narrow pins cause needless breakage on
security updates and don't matter for the parity harness (we test against the
live behavior of each library, not a specific patch release).

```text
# Web framework
fastapi>=0.110,<1.0
uvicorn[standard]>=0.27,<1.0

# HTTP client (Ollama)
httpx>=0.27,<1.0

# Vector math
numpy>=1.26,<3.0

# Request/response models (pydantic v2 is the current major)
pydantic>=2.5,<3.0
```

Notes:

- `uvicorn[standard]` pulls in `httptools`, `uvloop` (Linux/macOS), `websockets`,
  and `watchfiles` — none of these are required by *us* but they're the
  recommended install and harmless.
- No `hnswlib` (hand-port per §5).
- No `flask`, no `requests`, no `starlette` direct, no `scipy`.
- Python version floor is implicit (project requires 3.11+; we'll encode that
  in `pyproject.toml` instead, where it belongs — `requirements.txt` does
  not have a standard way to express it).

---

## 9. What Is NOT a Dependency

These live in the repo and are easy to mistake for app dependencies. They
are **not**:

| Path | What it is | Why it's not a dep |
|---|---|---|
| `.claude/` | Claude Code session helpers, agent prompts, slash-command definitions | Dev tooling — never imported by `main.py` |
| `.claude-flow/` | Ruflo / Claude Flow coordination state and history | Dev tooling, agent orchestration cache |
| `.swarm/` | Swarm coordination scratch space | Dev tooling, ephemeral |
| `CLAUDE.md` | Repo-level instructions for Claude Code agents | Documentation, not code |
| `.mcp.json` | MCP server config for the dev harness | Dev tooling |
| `.gitignore` | Git exclusions | n/a |
| `index.html` | The web UI served at `/` | First-party static asset, not a dependency |

The Python port's runtime will import from stdlib + `requirements.txt`
**only**. The above files are reproducible from the repo and irrelevant to
the deployed service.
