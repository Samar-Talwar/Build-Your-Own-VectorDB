# Test Strategy & Parity Harness

> **Status:** Planning phase. No code has been written. The C++ source is untouched.
> **Companion docs:** [01-architecture](01-architecture.md), [03-api-contract](03-api-contract.md) (pending), [04-algorithm-parity](04-algorithm-parity.md), [07-execution-plan](07-execution-plan.md), [08-risks](08-risks-and-decisions.md).

---

## 1. Strategy Overview

The migration uses a **strangler-fig parity harness**: the C++ binary stays up as the source of truth, and every Python response is diffed against the C++ response for the same input. The C++ binary is *only* retired in P6 (cutover), and only after the harness reports zero unacceptable diffs.

The harness script is **`tests/parity/run_parity.py`** and lives alongside the Python package from P0. It is a single Python file (no third-party deps — stdlib only: `urllib`, `json`, `argparse`, `difflib`, `subprocess`). The harness boots two HTTP servers (one C++, one Python), fires a fixed catalog of fixtures at both, normalizes each pair of responses according to per-route rules, and reports a per-route pass/fail.

```
                   +-------------------+
                   |  run_parity.py    |
                   +---------+---------+
                             |
                +------------+------------+
                |                         |
        curl fixtures             curl fixtures
                |                         |
                v                         v
        +---------------+         +---------------+
        | C++ binary    |         | Python server |
        | :8081         |         | :8080         |
        +---------------+         +---------------+
                |                         |
                +----+----------+---------+
                     |          |
                     v          v
              normalize,   normalize,
              diff         diff
                     |          |
                     +----+-----+
                          v
                   per-route pass/fail
                   exit 0 if all STRICT pass
                   exit 1 if any STRICT fails
                   exit 0 (with warnings) if only LOOSE/STRUCTURAL fail
```

The harness is invoked manually from P0 onward and is the acceptance gate between phases. It is **not** CI-integrated in v1 (deferred to v2; see §9).

---

## 2. Harness Design

### 2.1 Process Layout

The harness is a single Python entry point. It assumes two long-running server processes are already up; it does **not** spawn them itself (keeping the harness simple, reproducible, and easy to debug).

| Process | Port | Command |
|:---|:---|:---|
| C++ baseline | `8081` | `g++ -O2 -std=c++17 main.cpp -o /tmp/vectordb_cpp -lws2_32 && /tmp/vectordb_cpp --port 8081` *(see §8 — the C++ binary hard-codes `:8080`, so one of three workarounds is required)* |
| Python SUT | `8080` | `python -m vectordb --port 8080` |
| Harness | n/a | `python tests/parity/run_parity.py --cpp http://localhost:8081 --py http://localhost:8080` |

> **P0 simplification:** in P0, the C++ binary is on `:8080` and the Python stubs are on `:8081`. Once the Python server is on `:8080` and fully feature-complete (P5+), the C++ moves to `:8081` via one of the workarounds in §8.

### 2.2 Per-Fixture Flow

For each fixture in the catalog (see §4):

1. **Send** the request to `http://cpp:8081{path}?{query}` with `{method, headers, body}` from the fixture.
2. **Send** the same request to `http://py:8080{path}?{query}`.
3. **Capture** both responses: status, headers, body.
4. **Normalize** both bodies through the per-route normalizer (see §3).
5. **Diff** the normalized forms. If they differ:
   - For `STRICT` routes: report a **FAIL** with the diff snippet.
   - For `STRUCTURAL` routes: report a **FAIL** only on structural mismatch (see §3.2); log numeric drift as a warning.
   - For `LOOSE` routes: report a **PASS** as long as the `error` field is absent and required fields are present.

### 2.3 Per-Route Result Line

```
[ STRICT  ] GET /search?algo=bruteforce  k=3   PASS  (0.000s)
[ STRICT  ] POST /insert                          FAIL  diff:
{"id":3} vs {"id":4}
[ STRUCT  ] GET /hnsw-info                         PASS  warn: topLayer 2 vs 3
[ LOOSE   ] POST /doc/ask                          PASS
```

### 2.4 Exit Codes

| Code | Meaning |
|:---:|:---|
| `0` | All STRICT routes passed. STRUCTURAL/LOOSE may have soft warnings. |
| `1` | At least one STRICT route failed, or any fixture returned an unexpected HTTP status (e.g., 5xx instead of 2xx). |
| `2` | Harness error (could not connect to either server, bad fixture file, etc.). |

### 2.5 Skipped Fixtures

A fixture is **skipped** (not failed) when:
- Both servers report `ollamaAvailable: false` (for `/doc/insert`, `/doc/search`, `/doc/ask`, `/status`).
- The fixture targets a route not yet implemented in the Python SUT (returns `501`); the harness reports `SKIP (not yet implemented)`.
- `index.html` is missing in the Python repo (skips `GET /` until P0.5 lands).

A skipped fixture never causes a non-zero exit. The harness prints a summary line: `12 passed, 3 failed, 2 skipped, 1 warned`.

---

## 3. Normalization Rules

Three normalization classes are applied per route, in this priority order: STRICT > STRUCTURAL > LOOSE.

### 3.1 STRICT (byte-identical)

No normalization. The two response bodies are compared as raw strings.

- `Content-Type: application/json` is set on both.
- Whitespace, key order, and number formatting must match exactly.
- This is the strongest contract; failures here mean a real bug in the Python port.

### 3.2 STRUCTURAL (parse + compare)

Both bodies are `json.loads`'d, then:

- **Keys**: the union of keys from both responses must match (or be a documented subset; see per-route rules below). Missing-key failures are FATAL.
- **Floats**: compared with `abs(a - b) < 1e-5` per numeric leaf.
- **Ints, strings, bools**: compared with `==`.
- **Arrays**: same length, same order (because the C++ emits results sorted by `(distance asc, id asc)` and the Python must do the same).
- **Objects**: recursive structural diff.

### 3.3 LOOSE (existence only)

Only two checks:
1. `error` field is absent (or empty).
2. Required fields are present (`answer` must exist; the harness does not read its content).

LOOSE routes exist for LLM-stochastic responses where exact text matching is impossible.

### 3.4 Per-Route Rules

| Route | Class | Special Rules |
|:---|:---:|:---|
| `GET /search` | **STRICT** | Compare entire body. Float distances tolerate `1e-6` (last-bit; see R13). |
| `POST /insert` | **STRICT** | Compare entire body. IDs must match (this implies consistent `nextId` counter). |
| `DELETE /delete/<id>` | **STRICT** | Compare entire body. |
| `GET /items` | **STRICT** | Compare entire body. Array of 20+ items. |
| `GET /benchmark` | **STRUCTURAL** | Compare **`itemCount` only**. Do not compare `bruteforceUs`, `kdtreeUs`, `hnswUs` (timings are not comparable across runtimes; per R13). |
| `GET /hnsw-info` | **STRUCTURAL** | Compare `topLayer`, `nodeCount`, field-name set, `len(nodesPerLayer)`, `len(edgesPerLayer)`, `len(nodes)`, `len(edges)`. **Do not** compare `nodesPerLayer[i]` or `edgesPerLayer[i]` numerically if `D1 = hnswlib`; compare exactly if `D1 = hand-port` (per D1 and R1). |
| `POST /doc/insert` | **STRUCTURAL** | Compare `len(ids) == chunks`, `dims` integer match, `len(ids) > 0`. Skip if Ollama is offline on either server. |
| `DELETE /doc/delete/<id>` | **STRICT** | Compare entire body. |
| `GET /doc/list` | **STRUCTURAL** | Compare array length, each item's `id`/`title`/`words` exactly, `preview` first 100 chars (last 20 chars may be `…` truncation noise; tolerate). Skip if Ollama offline. |
| `POST /doc/search` | **STRUCTURAL** | Compare `len(contexts) == k`, each context's `id`/`title` exactly, `distance` within `1e-4`. Skip if Ollama offline. |
| `POST /doc/ask` | **LOOSE** | Only check `error` absent and `answer` field present. Do not compare `answer` text. Do not compare `model`. Compare `len(contexts) == k` and each context's `id` exactly. Skip if Ollama offline. |
| `GET /status` | **STRUCTURAL** | Compare `docDims` (if non-zero), `demoCount`, `demoDims` exactly. **Skip** `ollamaAvailable` (depends on transient Ollama state). |
| `GET /stats` | **STRICT** | Compare entire body. |
| `GET /` | **STRICT** | Compare entire body byte-for-byte. Skip if `index.html` missing in Python repo. |

> **The default for unknown new fields is "must match"**. If a new field appears in the C++ response and not in the Python response (or vice versa), that's a STRUCTURAL fail. The harness prints the union-difference.

---

## 4. Fixture Catalog

All fixtures live in `tests/parity/fixtures.json` as a single JSON array. Each entry has:

```json
{
  "route": "GET /search",
  "class": "STRICT",
  "query": "v=0.9,0.85,0.72,0.68,0.12,0.08,0.15,0.10,0.05,0.08,0.06,0.09,0.07,0.11,0.08,0.06&k=3&metric=cosine&algo=bruteforce",
  "body": null,
  "requires": "demo-loaded"
}
```

The `requires` field is one of `"demo-loaded"`, `"ollama"`, `"index-html"`, or `null`.

### 4.1 `GET /search` — 3 fixtures

1. **BF with demo vector as query** — uses the first demo vector (id=1, the Linked List row) as the query, `k=3`, `metric=cosine`. Exercises the exact-match (distance=0) path.
2. **KD-Tree with random query** — `k=5`, `metric=cosine`, `algo=kdtree`. Vector is a random 16D value seeded with `42` so it's reproducible.
3. **HNSW with random query** — `k=10`, `metric=euclidean`, `algo=hnsw`. Different metric to ensure the dispatcher routes correctly.

The metric x algo coverage from these three is `cosine × {BF, kdtree}` and `euclidean × hnsw`. `manhattan` is exercised by ad-hoc debug runs (not in the default catalog) since its behavior is symmetric to the others.

### 4.2 `POST /insert` — 1 fixture

A fresh vector with `metadata="parity-harness-test"`, `category="test"`, and an embedding of all `0.5`s. Expected: `{"id": 21}` (assuming the 20 demo items are present).

### 4.3 `DELETE /delete/<id>` — 1 fixture

Deletes the vector inserted in §4.2. Expected: `{"ok": true}`.

### 4.4 `GET /items` — 1 fixture

The initial state — 20 items. Asserts the array length and the 20 IDs.

### 4.5 `GET /benchmark` — 1 fixture

Same query as the BF fixture in §4.1, `k=3`, `metric=cosine`. Asserts only `itemCount == 20`.

### 4.6 `GET /hnsw-info` — 1 fixture

Asserts `nodeCount == 20`, `len(nodesPerLayer) >= 1`, `len(edgesPerLayer) == len(nodesPerLayer)`.

### 4.7 `GET /stats` — 1 fixture

`{"count":20,"dims":16,"algorithms":[...],"metrics":[...]}`.

### 4.8 `GET /status` — 1 fixture *(conditional)*

Only fired if `ollamaAvailable == true` on **both** servers. Otherwise: skip with a note in the harness output.

### 4.9 `POST /doc/insert` — 2 fixtures *(conditional)*

Both skipped if Ollama is not running on either server. The harness prints `SKIP: ollama offline` for these.

1. **Short text** — 200-word passage (single chunk, no chunking). Asserts `chunks == 1`, `dims == 768` (or whatever `nomic-embed-text` reports), `len(ids) == 1`.
2. **Long text** — 1000-word passage (4 chunks: 250, 220, 220, 220, last is 90). Asserts `chunks == 5` (250, 220, 220, 220, 90), `len(ids) == 5`, `dims` matches the first fixture.

### 4.10 `GET /doc/list` — 1 fixture

Fired after the `/doc/insert` long-text fixture. Asserts `len(docs) == 5` and that each `preview` is at most 121 characters.

### 4.11 `DELETE /doc/delete/<id>` — 1 fixture

Deletes the first ID from the `/doc/insert` long-text fixture.

### 4.12 `POST /doc/search` — 1 fixture

Fired after `/doc/insert` short-text. `{"question": "linked list pointers", "k": 2}`. Asserts `len(contexts) <= 2`.

### 4.13 `POST /doc/ask` — 1 fixture

Fired after `/doc/insert` short-text. `{"question": "What is a linked list?", "k": 2}`. **LOOSE** — only checks `error` absent and `answer` present.

### 4.14 `GET /` — 1 fixture

Asserts `index.html` body byte-for-byte. Skipped if `index.html` is absent in the Python repo root (P0.5).

### 4.15 Catalog Summary

| Route | # Fixtures | Class |
|:---|:---:|:---|
| `GET /search` | 3 | STRICT |
| `POST /insert` | 1 | STRICT |
| `DELETE /delete/<id>` | 1 | STRICT |
| `GET /items` | 1 | STRICT |
| `GET /benchmark` | 1 | STRUCTURAL |
| `GET /hnsw-info` | 1 | STRUCTURAL |
| `GET /stats` | 1 | STRICT |
| `GET /status` | 1 (conditional) | STRUCTURAL |
| `POST /doc/insert` | 2 (conditional) | STRUCTURAL |
| `GET /doc/list` | 1 (conditional) | STRUCTURAL |
| `DELETE /doc/delete/<id>` | 1 (conditional) | STRICT |
| `POST /doc/search` | 1 (conditional) | STRUCTURAL |
| `POST /doc/ask` | 1 (conditional) | LOOSE |
| `GET /` | 1 (conditional) | STRICT |
| **Total** | **18** | **11 STRICT, 6 STRUCTURAL, 1 LOOSE** |

---

## 5. Per-Phase Acceptance Test

Each phase in [07-execution-plan](07-execution-plan.md) has a named set of fixtures that must pass to advance. The harness is run with `--phase <P0|P1|...|P6>` and only the fixtures for that phase are executed.

| Phase | Required-to-Pass Fixtures |
|:---|:---|
| **P0** | All 14 routes return `501` (or `404` for `/`) on the Python server; CORS headers match. This is a smoke test that the harness plumbing works. No STRICT fixture must fail. |
| **P1** | `GET /search?algo=bruteforce` × 1, `POST /insert`, `DELETE /delete/<id>`, `GET /items`, `GET /stats`, `GET /` — all STRICT. |
| **P2** | P1's set, plus `GET /search?algo=kdtree` (the second search fixture). |
| **P3** | P2's set, plus `GET /search?algo=hnsw` (the third search fixture), `GET /benchmark`, `GET /hnsw-info`. All STRUCTURAL for `/hnsw-info` and `/benchmark`. |
| **P4** | P3's set, plus `POST /doc/insert` × 2, `GET /doc/list`, `DELETE /doc/delete/<id>`, `POST /doc/search`, `GET /status` — all conditional on Ollama. |
| **P5** | P4's set, plus `POST /doc/ask` (LOOSE). |
| **P6** | All 18 fixtures, no skips allowed except the Ollama-conditional ones when offline. **Cutover gate.** |
| **P7** | All 18 fixtures must pass. This is the final regression sweep before `git rm main.cpp`. |

Each phase-end run produces a report like:

```
=== P3 parity report ===
[PASS] GET /search?algo=bruteforce  k=3
[PASS] POST /insert
[PASS] DELETE /delete/<id>
[PASS] GET /items
[PASS] GET /stats
[PASS] GET /search?algo=kdtree  k=5
[PASS] GET /search?algo=hnsw   k=10
[PASS] GET /benchmark
[WARN] GET /hnsw-info  drift: topLayer 2 vs 3  (acceptable per D1=hnswlib)
[SKIP] POST /doc/insert  (ollama offline)
[SKIP] GET /doc/list     (ollama offline)
[SKIP] DELETE /doc/delete/<id>  (ollama offline)
[SKIP] POST /doc/search  (ollama offline)
[SKIP] GET /status       (ollama offline)
[SKIP] POST /doc/ask     (ollama offline)
[SKIP] GET /             (no index.html in python repo)

Result: P3 ADVANCE (0 STRICT failures, 1 acceptable STRUCTURAL warning)
```

---

## 6. Algorithm Unit Tests

These are **small pytest-based tests** that exercise the algorithm modules directly without needing the HTTP server. They live in `tests/test_*.py` and are run with `pytest tests/`. They are **not** a replacement for the parity harness — they catch regressions at the algorithm layer so a parity failure can be localized to "the Python distance function is wrong" vs. "the JSON formatting is wrong."

The harness invokes them as part of the per-phase check (`python -m pytest tests/test_distances.py tests/test_bruteforce.py ...`), but they can also be run in isolation.

### 6.1 `tests/test_distances.py`

| Test | Input | Expected |
|:---|:---|:---|
| `test_euclidean_basic` | `euclidean([0,0,0], [3,4,0])` | `5.0` |
| `test_euclidean_empty` | `euclidean([], [])` | `0.0` |
| `test_cosine_identical` | `cosine([1,0], [1,0])` | `0.0` |
| `test_cosine_orthogonal` | `cosine([1,0], [0,1])` | `1.0` |
| `test_cosine_opposite` | `cosine([1,0], [-1,0])` | `2.0` |
| `test_cosine_zero_vector` | `cosine([0,0], [1,0])` | `1.0` *(must return 1.0, not 0.0 — the 1e-9 quirk)* |
| `test_manhattan_basic` | `manhattan([1,2,3], [4,5,6])` | `9.0` |
| `test_get_dist_fn` | `get_dist_fn("cosine") is cosine`, `get_dist_fn("manhattan") is manhattan`, `get_dist_fn("euclidean") is euclidean`, `get_dist_fn("") is euclidean`, `get_dist_fn("garbage") is euclidean` | identity check |

### 6.2 `tests/test_bruteforce.py`

```python
def test_insert_and_knn_returns_all_sorted():
    bf = BruteForce()
    for i, v in enumerate(DEMO_VECTORS):
        bf.insert(VectorItem(id=i+1, metadata=v["meta"], category=v["cat"], emb=v["emb"]))
    results = bf.knn(DEMO_VECTORS[0]["emb"], k=5, dist=cosine)
    assert len(results) == 5
    assert results[0][1] == 1  # nearest is self
    # sorted by (distance asc, id asc)
    for a, b in zip(results, results[1:]):
        assert (a[0], a[1]) <= (b[0], b[1])

def test_remove():
    bf = BruteForce()
    for i in range(3): bf.insert(VectorItem(id=i+1, metadata="", category="", emb=[float(i)]))
    bf.remove(2)
    assert [v.id for v in bf.items] == [1, 3]
```

### 6.3 `tests/test_kdtree.py`

```python
def test_kdtree_matches_bruteforce_top1():
    bf, kdt = BruteForce(), KDTree(16)
    for v in DEMO_VECTORS:
        item = VectorItem(id=..., metadata=..., category=..., emb=v["emb"])
        bf.insert(item); kdt.insert(item)
    q = random_vector(16, seed=42)
    bf_top = bf.knn(q, k=1, dist=cosine)[0]
    kd_top = kdt.knn(q, k=1, dist=cosine)[0]
    assert bf_top[1] == kd_top[1]  # same nearest-neighbor ID
    assert abs(bf_top[0] - kd_top[0]) < 1e-6  # same distance within last-bit
```

### 6.4 `tests/test_hnsw.py`

```python
def test_hnsw_matches_bruteforce_top1_fuzzy():
    # HNSW with ef=50 is approximate, so allow either top-1 or top-2
    hnsw = HNSW(m=16, ef_build=200, seed=42)
    bf = BruteForce()
    for v in DEMO_VECTORS:
        item = VectorItem(id=..., metadata=..., category=..., emb=v["emb"])
        bf.insert(item); hnsw.insert(item, cosine)
    q = random_vector(16, seed=42)
    bf_top1 = bf.knn(q, k=1, dist=cosine)[0][1]
    bf_top2 = bf.knn(q, k=2, dist=cosine)
    hnsw_top1 = hnsw.knn(q, k=1, ef=50, dist=cosine)[0][1]
    assert hnsw_top1 in {bf_top1, bf_top2[1][1]}
```

### 6.5 `tests/test_chunker.py`

| Test | Input | Expected |
|:---|:---|:---|
| `test_short_text_returns_original` | `chunk_text("hello world")` | `["hello world"]` *(unchanged input, not re-joined)* |
| `test_exact_250_words_returns_original` | `chunk_text(" ".join(["w"]*250))` | `["w w ... w"]` *(250 words, one chunk, original string)* |
| `test_long_text_chunks_correctly` | `chunk_text(" ".join(["w"]*700))` | 4 chunks: 250, 220, 220, 10 words |
| `test_chunk_step_is_220` | `chunk_text(" ".join(["w"]*1000))` | Chunks of 250, 220, 220, 220, 90 — step is 220 (250-30) |
| `test_empty_text` | `chunk_text("")` | `[]` |
| `test_whitespace_splitting` | `chunk_text("a  b\tc\nd")` | `["a  b\tc\nd"]` *(5 chars, 4 words per C++ `>>`)* |

### 6.6 `tests/test_demo_data.py`

```python
def test_all_vectors_are_16d():
    for v in DEMO_VECTORS:
        assert len(v["emb"]) == 16

def test_demo_matches_cpp_source():
    # Compare against a snapshot file generated from main.cpp:720-759
    cpp_snapshot = json.loads(Path("tests/fixtures/demo_data_cpp.json").read_text())
    assert DEMO_VECTORS == cpp_snapshot  # bit-exact: 20 entries, exact float values
```

`tests/fixtures/demo_data_cpp.json` is a fixture file committed to the repo, generated by a one-time script that reads `main.cpp:720-759` and emits the 20 vectors. This file is the source of truth for R3.

### 6.7 `tests/test_json_format.py`

| Test | Input | Expected |
|:---|:---|:---|
| `test_j_vec_formatting` | `j_vec([0.9, 0.85])` | `"[0.9000,0.8500]"` *(4 decimal places, no spaces — matches `std::setprecision(4)`)* |
| `test_j_s_escape_quote` | `j_s('a"b')` | `'"a\\"b"'` |
| `test_j_s_escape_backslash` | `j_s('a\\b')` | `'"a\\\\b"'` |
| `test_j_s_newline` | `j_s('a\nb')` | `'"a\\nb"'` |
| `test_parse_vec_basic` | `parse_vec("0.5,0.6,0.7")` | `[0.5, 0.6, 0.7]` |
| `test_parse_vec_empty` | `parse_vec("")` | `[]` |
| `test_extract_str_basic` | `extract_str('{"key":"val"}', "key")` | `"val"` |
| `test_extract_int_basic` | `extract_int('{"k":42}', "k", 0)` | `42` |
| `test_extract_int_default` | `extract_int('{}', "missing", 99)` | `99` |

These tests are the R2 mitigation: the wire format must match the C++ exactly or `parseFloat` consumers and visual diffs will break.

### 6.8 Total Unit Tests

| File | # Tests |
|:---|:---:|
| `test_distances.py` | 8 |
| `test_bruteforce.py` | 2 |
| `test_kdtree.py` | 1 |
| `test_hnsw.py` | 1 |
| `test_chunker.py` | 6 |
| `test_demo_data.py` | 2 |
| `test_json_format.py` | 9 |
| **Total** | **29** |

---

## 7. What the Harness Does NOT Cover

Explicitly out of scope for v1:

1. **LLM answer text** — `/doc/ask` is LOOSE. The harness checks the `answer` field exists and `error` is absent, but does not validate the generated text. The `model` field may also differ (e.g., if the user has a different `llama3.2` build). Per R14.
2. **HNSW graph numeric drift** — if `D1 = hnswlib`, the harness does not compare `nodesPerLayer[i]`, `edgesPerLayer[i]`, or individual `edges[].{src,dst,lyr}` numerically. Only the schema is checked. Per R1.
3. **Race conditions under high concurrency** — neither the C++ nor the v1 Python server is tested with concurrent requests. The C++ has a documented `getDims` race (R8) and the Python matches it. D9 keeps `threaded=False` for v1.
4. **Persistence** — neither server has persistence. Restart loses state. R11 confirms this is intentional.
5. **Ollama behavior** — the harness does not test Ollama itself. It only tests the *VectorDB code's interaction with Ollama* (the request bodies, the response parsing, the error messages).
6. **`httplib` specifics** — the Python server is Flask, not `httplib`. The harness does not test behavior specific to `httplib` (e.g., the regex `(\d+)` matching style). R12.
7. **CORS preflight** — the harness does not fire `OPTIONS` requests. The CORS headers are checked on the `GET /` response, but preflight (R10) is not tested.
8. **Frontend rendering** — `index.html` is byte-compared, but its JavaScript is not executed. The harness does not test that the UI works end-to-end.

---

## 8. Running the Harness

### 8.1 The Port Problem

The C++ binary hard-codes `0.0.0.0:8080` at `main.cpp:1087`:

```cpp
svr.listen("0.0.0.0", 8080);
```

For the harness to work, the C++ baseline must listen on `8081` (or any non-8080 port) so the Python server can have `8080`. Three workarounds are available, in order of preference:

1. **Edit `main.cpp:1087` during the planning phase only** (per §8.2 below). The harness and the developer workflow become trivial. The diff is reverted before any phase begins (P0's entry criteria: "C++ untouched").
2. **Use `socat` or `netsh portproxy` to forward**: a `socat TCP-LISTEN:8081,fork TCP:127.0.0.1:8080` on Linux forwards 8081 to 8080, and the C++ binary still binds 8080. The harness then hits the C++ at `:8081` (which is really just a forwarder). This is platform-specific (`socat` is not standard on Windows; `netsh` requires admin).
3. **Run the C++ on a separate machine/VM** with port `8080` exposed. The harness hits the C++ at the remote host. Heavy, but clean.

**Recommendation**: use option (1) for development, option (2) or (3) for CI (when v2 lands; see §9). The edit at `main.cpp:1087` is a 1-line change and the file is reverted before each phase ships (per the per-phase entry criteria in [07-execution-plan](07-execution-plan.md) §3 P0: "C++ untouched").

> **Workaround for v1 testing during planning**: run the C++ on `:8080` and the Python on `:8081`, then pass `--cpp http://localhost:8080 --py http://localhost:8081` to the harness. The convention `--cpp :8081, --py :8080` is the *target* convention; the inverse is acceptable during the planning phase.

### 8.2 Exact Commands

```bash
# Terminal 1: C++ baseline (after the 1-line edit at main.cpp:1087)
g++ -O2 -std=c++17 main.cpp -o /tmp/vectordb_cpp -lws2_32
/tmp/vectordb_cpp &
# Now listening on :8081

# Terminal 2: Python SUT
python -m vectordb --port 8080
# Now listening on :8080

# Terminal 3: harness
python tests/parity/run_parity.py \
    --cpp http://localhost:8081 \
    --py  http://localhost:8080 \
    --phase P3   # or omit to run all phases
```

The harness prints a per-fixture report and exits with code `0` (all STRICT pass) or `1` (any STRICT fail).

### 8.3 Plumbing Requirements

- `g++` (MinGW or MSYS2 on Windows; system GCC on Linux/macOS).
- Python 3.11+ (matches D3).
- The Python server must implement `--port` to override the default `:8080`.
- The C++ binary must be modified to accept a `--port` flag (or have the port hard-coded to `:8081` for the planning phase). A 1-line edit:

  ```cpp
  // main.cpp:1087 — replace
  svr.listen("0.0.0.0", 8080);
  // with
  int port = 8081;
  if (argc > 2 && std::string(argv[1]) == "--port") port = std::atoi(argv[2]);
  svr.listen("0.0.0.0", port);
  ```

  This is the **only** edit to `main.cpp` during planning. It is reverted before P0 begins.

---

## 9. CI Integration (Deferred)

For v1, the harness is a **manual script** invoked from the developer's terminal. It is the human's job to:

1. Build the C++ binary.
2. Start both servers in two terminals.
3. Run the harness after each phase.
4. Read the per-route report and decide whether to advance.

For v2, integrate with GitHub Actions:

- **Workflow file**: `.github/workflows/parity.yml`.
- **Job**: `parity-check` runs on `pull_request` to `main`.
- **Steps**:
  1. `actions/checkout@v4`
  2. `actions/setup-python@v5` (Python 3.12)
  3. `actions/setup-go` (or any C++ toolchain step) — for `g++`
  4. Build the C++ binary: `g++ -O2 -std=c++17 main.cpp -o /tmp/vectordb_cpp -lws2_32`
  5. `pip install -r requirements.txt`
  6. Start C++ in the background: `/tmp/vectordb_cpp --port 8081 &`
  7. Start Python in the background: `python -m vectordb --port 8080 &`
  8. `sleep 2` (wait for both servers to bind)
  9. Run the harness: `python tests/parity/run_parity.py --cpp http://localhost:8081 --py http://localhost:8080`
  10. On non-zero exit, post the diff to the PR as a comment using `actions/github-script@v7`.

This is a v2 deliverable. v1 ships without CI integration to keep the scope tight (per D10: "parity harness only", and per the v1 philosophy: "the user explicitly asked for migration, not for adding test infrastructure").

---

**Test strategy written — 18 fixtures, 29 unit tests, 11 STRICT, 6 STRUCTURAL, 1 LOOSE.**
