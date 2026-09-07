# Migration Execution Plan & Task Graph

> **Status:** Planning phase. No code has been written. The C++ source is untouched.
> **Companion docs:** [01-architecture](01-architecture.md), [02-dependencies](02-dependencies.md) (pending), [03-api-contract](03-api-contract.md) (pending), [04-algorithm-parity](04-algorithm-parity.md) (pending), [05-module-decomposition](05-module-decomposition.md) (pending), [06-test-strategy](06-test-strategy.md) (pending), [08-risks](08-risks-and-decisions.md).

---

## 1. Migration Philosophy

The migration is structured as a **strangler-fig with a parity gate at every phase**:

1. **Every phase ends in a runnable Python server** that exposes the same routes.
2. **After every phase, a parity harness** (see [06-test-strategy](06-test-strategy.md)) compares the new server's response to the still-running C++ baseline.
3. **The C++ binary is the source of truth** until the final cutover phase.
4. **No phase** may modify `main.cpp` or `httplib.h`.

This means a partial migration is always bootable. We never have a window where the project is broken.

---

## 2. Phased Task Graph

```
P0 (scaffold) ──► P1 (BF + demo) ──► P2 (KD-Tree) ──┐
                                                    ├─► P3 (HNSW) ──► P4 (chunker + Ollama + DocDB) ──► P5 (RAG) ──► P6 (cutover) ──► P7 (cleanup)
P0 (scaffold) ────────────────────────────────────► P8 (parity harness running alongside) ◄─── tests every phase
```

**Parallelism map:**

| Phase | Independent work inside the phase |
|---|---|
| P0 | Scaffold all module files (empty stubs); `requirements.txt`; Flask app skeleton returning 501. |
| P1 | `distances.py`, `bruteforce.py`, `demo_data.py`, `db.py` (VectorDB wrapper for BF only), `app.py` routes for `/search` (BF), `/insert`, `/delete`, `/items`, `/stats`. |
| P2 | `kdtree.py`, route `/search` now accepts `algo=kdtree`. |
| P3 | `hnsw.py` (or `hnswlib` adapter), `db.py` extended, routes `/search` (hnsw), `/benchmark`, `/hnsw-info`. |
| P4 | `chunker.py`, `ollama.py`, `document_db.py`, `app.py` routes for `/doc/insert`, `/doc/delete`, `/doc/list`, `/doc/search`, `/status`. |
| P5 | Route `/doc/ask` + RAG prompt template. |
| P6 | Smoke test against the C++ baseline. Switch `python -m vectordb` to be the default; remove C++ from `README.md` Quick Start. |
| P7 | Delete `main.cpp`, `httplib.h`. Update `README.md` to remove the "compile g++" troubleshooting section. |

**Within P0**, all module stubs can be authored in parallel.
**Within P1–P5**, the algorithm module and the route are tightly coupled (they ship together).
**P8** is cross-cutting — it ships alongside P0 and is updated each phase.

---

## 3. Phase Detail

### P0 — Scaffold (entry: now; exit: when `python -m vectordb` runs and returns 501 on all routes)

**Files created:**
```
py/vectordb/__init__.py
py/vectordb/__main__.py         # entry: python -m vectordb
py/vectordb/app.py              # Flask app with route stubs
py/vectordb/distances.py        # stub
py/vectordb/bruteforce.py       # stub
py/vectordb/kdtree.py           # stub
py/vectordb/hnsw.py             # stub
py/vectordb/db.py               # VectorDB stub
py/vectordb/document_db.py      # DocumentDB stub
py/vectordb/ollama.py           # OllamaClient stub
py/vectordb/chunker.py          # chunkText stub
py/vectordb/demo_data.py        # stub
py/vectordb/json_format.py      # wire-format helpers
requirements.txt
tests/parity/run_parity.py      # parity harness (can be a script)
docs/migration/                 # this folder
```

**Entry criteria:** `01-architecture.md` exists. **C++ untouched.**
**Exit criteria:** `pip install -r requirements.txt && python -m vectordb` starts on port 8080. `GET /` returns 404 (no `index.html` in py/ — that's a v1.1 add, see P0.5 below). All other routes return 501 with `{"error":"not implemented"}` and the right CORS headers. C++ binary still runs and is the source of truth.

> **P0.5 (deferred — optional):** Copy `index.html` into the repo root and have Flask serve it on `GET /`. This makes parity testing easier. **Do this in P0 if simple; otherwise defer to P6.**

**Parity check:** `python -m vectordb &` then `curl -i http://localhost:8080/search?v=...` returns 501; `curl -i http://localhost:8081/search?v=...` (C++ on 8081) returns 200. CORS headers match.

---

### P1 — Brute Force + demo data + search BF path (entry: P0 done; exit: `/search?algo=bruteforce` returns byte-identical JSON to C++)

**Files filled in:**
- `py/vectordb/distances.py` — `euclidean`, `cosine`, `manhattan`, `get_dist_fn(name)`
- `py/vectordb/bruteforce.py` — `BruteForce` class
- `py/vectordb/demo_data.py` — the 20 vectors from `main.cpp:720-759`, copied verbatim
- `py/vectordb/db.py` — `VectorDB` (BF only for now)
- `py/vectordb/json_format.py` — `j_s`, `j_vec`, `parse_vec`, `extract_str`, `extract_int`, `parse_body`, `cors`
- `py/vectordb/app.py` — routes `/search` (BF only), `/insert`, `/delete/<int:id>`, `/items`, `/stats`, `/`

**Verification:**
- `curl "http://localhost:8080/search?v=0.9,0.85,0.72,0.68,0.12,0.08,0.15,0.10,0.05,0.08,0.06,0.09,0.07,0.11,0.08,0.06&k=3&metric=cosine&algo=bruteforce"` must match the C++ output **byte-for-byte** (modulo float last-bit).
- **Float precision table** (per `03-api-contract.md` §3.4 — must NOT use a global formatter):
  - `/search` `results[].distance` → 6 decimals
  - `/search` `results[].embedding[]` → 4 decimals
  - `/items` `[].embedding[]` → 4 decimals
  - `/doc/search` and `/doc/ask` `contexts[].distance` → 4 decimals
  - All IDs, latencies, counts → integer
  - `json.dumps` with `separators=(',', ':')` to match C++ whitespace.
- `curl "http://localhost:8080/insert" -d '{"metadata":"x","category":"y","embedding":[0.5,0.5,...]}'` matches C++.
- `curl "http://localhost:8080/items"` matches C++.
- `curl "http://localhost:8080/stats"` matches C++.

**Risk gates:** R2 (JSON precision), R3 (demo data), R13 (float32). All mitigated by direct copying.

---

### P2 — KD-Tree (entry: P1 done; exit: `/search?algo=kdtree` returns same nearest-neighbor IDs as C++)

**Files filled in:**
- `py/vectordb/kdtree.py` — `KDTree` class (either hand-port or wrap `scipy.spatial.KDTree`)
- `py/vectordb/db.py` — `VectorDB` adds `kdt` member
- `py/vectordb/app.py` — route `/search` accepts `algo=kdtree`

**Verification:**
- `curl "...&algo=kdtree&k=3&metric=cosine"` returns the same top-3 IDs (and same distances within `setprecision(6)`).

**Risk gate:** R5 (KD degeneracy). Not triggered by demo data. OK.

---

### P3 — HNSW (entry: P2 done; exit: `/search?algo=hnsw`, `/benchmark`, `/hnsw-info` all return valid JSON matching C++ field structure)

**Files filled in:**
- `py/vectordb/hnsw.py` — `HNSW` class (either `hnswlib` adapter or hand-port; default: `hnswlib` adapter with `M=16`, `ef_construction=200`, seed=42, `space='cosine'`)
- `py/vectordb/db.py` — `VectorDB` adds `hnsw` member and `hnsw_info()` method
- `py/vectordb/app.py` — routes `/search` accepts `algo=hnsw`, `/benchmark`, `/hnsw-info`

**Verification:**
- `curl "...&algo=hnsw&k=3"` returns valid JSON with same field structure.
- `curl "/benchmark?..."` returns `{bruteforceUs, kdtreeUs, hnswUs, itemCount}` with the same keys.
- `curl "/hnsw-info"` returns `{topLayer, nodeCount, nodesPerLayer, edgesPerLayer, nodes, edges}` with the same field names and array shapes. **The numeric values for `topLayer`, `nodesPerLayer`, `edgesPerLayer` may differ from the C++ binary** (per R1). This is acceptable; the field names and types must match.

**Risk gates:** R1 (graph drift), R6/R7 (entry-point bugs — not present in `hnswlib`; OK), R15 (seed determinism — capture the C++ `/hnsw-info` once as a snapshot fixture, then assert the Python output is *internally* consistent across runs).

---

### P4 — Chunker + Ollama + DocumentDB (entry: P3 done; exit: `/doc/insert`, `/doc/list`, `/doc/delete`, `/doc/search`, `/status` all return valid JSON; requires Ollama running for the full path)

**Files filled in:**
- `py/vectordb/chunker.py` — `chunk_text(text, 250, 30)` copied from `main.cpp:531-552`
- `py/vectordb/ollama.py` — `OllamaClient` class using `requests` with the same timeouts (2s/30s/180s) and the same request bodies
- `py/vectordb/document_db.py` — `DocumentDB` class
- `py/vectordb/app.py` — routes `/doc/insert`, `/doc/delete/<int:id>`, `/doc/list`, `/doc/search`, `/status`

**Verification (requires Ollama at 127.0.0.1:11434 with `nomic-embed-text` pulled):**
- `curl -X POST -d '{"title":"test","text":"..."}' /doc/insert` returns `{ids, chunks, dims}` with `dims=768`.
- `curl /doc/list` returns the document with the 120-char preview.
- `curl -X POST -d '{"question":"hello","k":3}' /doc/search` returns `{contexts:[{id,title,distance}]}`.

**Risk gates:** R4 (Ollama blocking — match C++ by sequential calls), R8 (`getDims` race — match C++ by not locking).

---

### P5 — RAG (`/doc/ask`) (entry: P4 done; exit: `/doc/ask` returns `{answer, model, contexts, docCount}` matching C++ field structure)

**Files filled in:**
- `py/vectordb/app.py` — route `/doc/ask` with the exact prompt template from `main.cpp:1024-1031`

**Verification:**
- `curl -X POST -d '{"question":"What is HNSW?","k":3}' /doc/ask` returns a JSON with `answer` field. The `answer` text will not match the C++ binary's `answer` text (LLM stochasticity), but the **structure** must match. Parity test must assert on field names and types, not on `answer` content.

**Risk gate:** R14 (RAG prompt drift) — mitigated by copying the prompt verbatim.

---

### P6 — Cutover (entry: P5 done; exit: Python is the default; C++ binary is removed from `README.md` Quick Start)

**Steps:**
1. Update `README.md`: change "Step 5 — Run the Python Server" to be the only run path; remove the `g++` troubleshooting section.
2. Add a `Makefile` or `pyproject.toml` (decide in P0) so the project has a single canonical entry point.
3. Add a `Procfile` for `gunicorn` if D9 changes.
4. Run the full smoke test (every route) end-to-end against the Python server. C++ binary is **not** required to be running anymore for the smoke test.
5. Run the parity harness one last time. If the diff is empty (or only the documented acceptable drifts), the migration is complete.

**Entry criteria:** P1–P5 all done. Parity harness passes.
**Exit criteria:** Smoke test passes. README is updated.

---

### P7 — Cleanup (entry: P6 done; exit: `main.cpp` and `httplib.h` are removed from the repo)

**Steps:**
1. `git rm main.cpp httplib.h`
2. `git commit -m "Remove C++ implementation after Python migration complete"`
3. Confirm `index.html` is still served correctly.
4. Update `README.md` to remove the section that references `main.cpp` and the `g++` step.

**Out of scope:** Removing `.claude/`, `.claude-flow/`, `.swarm/`, `CLAUDE.md`, `.mcp.json` (dev tooling).

---

## 4. Per-Phase Acceptance Gates

Before advancing to the next phase, all of these must hold:

- [ ] All routes added in this phase return 2xx for valid input and 4xx for invalid input matching the C++ error message.
- [ ] The parity harness script (`tests/parity/run_parity.py`) diff is empty for the routes in this phase.
- [ ] `python -m vectordb` starts and survives a `kill -TERM` (or `Ctrl+C`).
- [ ] The C++ source files (`main.cpp`, `httplib.h`) are byte-identical to their pre-migration state (`git status` shows them unchanged).
- [ ] `index.html` is unchanged.
- [ ] `README.md` is updated only in the "Quick Start" section if the run command changes.

---

## 5. Parallelism Within a Phase

The maximum useful parallelism for a single phase is bounded by the file dependency graph. In P0, all stub files are independent and can be authored in parallel by separate agents. In P1–P5, the algorithm module and the route handler are coupled — they ship together.

For the **current planning phase**, the maximum useful parallelism is the fan-out of 3 agents:
1. **dependency-mapper** → `02-dependencies.md`
2. **api-contract-freezer** → `03-api-contract.md`
3. **algorithm-spec-writer** → `04-algorithm-parity.md`

All three are independent of each other (they only depend on `01-architecture.md`).

After those three land:
4. **module-decomposer** → `05-module-decomposition.md` (depends on the above)
5. **test-strategist** → `06-test-strategy.md` (depends on `03` and `04`)

Then the lead integrates them into `07-execution-plan.md` (this document) and `08-risks.md`.

---

## 6. Cutover Checklist (final, pre-deletion)

- [ ] `pip install -r requirements.txt` works on a fresh venv.
- [ ] `python -m vectordb` starts on `:8080`.
- [ ] All 14 routes return 2xx for valid input.
- [ ] The frontend (`index.html`) works end-to-end: search, embed, ask.
- [ ] Parity harness shows zero diff for all STRICT routes.
- [ ] Parity harness shows only acceptable diffs (HNSW graph numbers) for STRUCTURAL routes.
- [ ] `git log` shows clean commits per phase, not one giant commit.
- [ ] `README.md` Quick Start is Python-only.
- [ ] `requirements.txt` is committed.
- [ ] `.gitignore` excludes `__pycache__/`, `*.pyc`, `.venv/`.

Then and only then: `git rm main.cpp httplib.h`.
