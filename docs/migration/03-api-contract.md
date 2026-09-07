# API Contract: C++ VectorDB & RAG HTTP Service

**Source**: `C:\Users\Samar\Documents\Your-OWN-AI\main.cpp` (lines 766-1089)
**Public contract authority**: `C:\Users\Samar\Documents\Your-OWN-AI\index.html` (frontend `fetch(...)` calls)
**Frozen**: 2026-09-07

This document freezes every observable behavior of the HTTP API. The Python port MUST match these specifications for routes marked **STRICT** byte-for-byte, and match keys/types/structure for **STRUCTURAL** routes. The frontend is the source of truth; any deviation breaks the UI.

---

## 1. Global Behaviors

| Property | Value | Source |
|----------|-------|--------|
| Bind host | `0.0.0.0` | `main.cpp:1087` |
| Bind port | `8080` | `main.cpp:1087` |
| CORS `Access-Control-Allow-Origin` | `*` | `main.cpp:522` (via `cors()`) |
| CORS `Access-Control-Allow-Methods` | `GET, POST, DELETE, OPTIONS` | `main.cpp:523` |
| CORS `Access-Control-Allow-Headers` | `Content-Type` | `main.cpp:524` |
| CORS preflight (`OPTIONS .*`) | HTTP 204, headers from `cors()` | `main.cpp:785-787` |
| Default response Content-Type | `application/json` | every `res.set_content(..., "application/json")` |
| Error body format | `{"error":"<literal>"}` (single key) | consistent across handlers |
| JSON string escaping | `"`, `\`, `\n`, `\r`, `\t` escaped via `jS()` | `main.cpp:435-446` |
| Float format | `std::fixed` + `std::setprecision(N)`; never scientific notation | `main.cpp:14` (iomanip include) |
| Field ordering | Insertion order in `std::ostringstream` (matches C++ source order) | implicit |
| Array element separator | `,` (no space after) | all `for` loops in handlers |
| `jVec` array format | `[0.9000,0.8500,0.7000,...]` — no spaces | `main.cpp:448-455` |

---

## 2. Route Specifications

### 2.1 `OPTIONS .*` — CORS Preflight

- **Method + path**: `OPTIONS` any path (regex `.*`)
- **Handler**: `main.cpp:785-787`
- **Request body**: none
- **Response headers**: from `cors()` (see Global Behaviors)
- **Response body**: empty
- **Status**: `204`
- **Parity class**: **STRICT** (every response includes these headers; the frontend never actually sends preflight in this codebase but the CORS machinery must remain identical for browser compatibility)

---

### 2.2 `GET /search` — Demo Vector k-NN Search

- **Method + path**: `GET /search`
- **Handler**: `main.cpp:791-819`

**Query parameters:**

| Param | Type | Default | Validation |
|-------|------|---------|------------|
| `v` | comma-delimited floats | required | length MUST equal `DIMS` (16) |
| `k` | int | `5` | parsed via `std::stoi` (throws → keeps default) |
| `metric` | string | `"cosine"` | empty → `"cosine"`; passed to `getDistFn` |
| `algo` | string | `"hnsw"` | empty → `"hnsw"`; routed in `VectorDB::search` |

**Validation error response (HTTP 200, `application/json`):**
```json
{"error":"need 16D vector"}
```

**Success response shape (key order, types, precision):**
```json
{
  "results": [
    {
      "id": 7,
      "metadata": "string",
      "category": "string",
      "distance": 0.123456,        // setprecision(6), std::fixed
      "embedding": [0.9000,0.8500,...]  // setprecision(4) via jVec, 16 elements
    }
  ],
  "latencyUs": 42,                  // long long int
  "algo": "hnsw",
  "metric": "cosine"
}
```

**Float precision table:**

| Field | Precision | Notes |
|-------|-----------|-------|
| `results[].distance` | `setprecision(6)` | `std::fixed`; e.g. `0.123456` |
| `results[].embedding[]` | `setprecision(4)` | via `jVec`; 16 elements when full VectorDB |
| `latencyUs` | int (no decimal) | microseconds |

**Status codes:** 200 always (validation errors return 200 with error body, NOT 400)
**Parity class**: **STRICT**
**Frontend reads** (`index.html:537-541`): `data.results[]` (id, metadata, category, distance), `data.latencyUs`

---

### 2.3 `POST /insert` — Insert Demo Vector

- **Method + path**: `POST /insert`
- **Handler**: `main.cpp:821-829`

**Request body** (`Content-Type: application/json`):
```json
{
  "metadata": "string (required, non-empty)",
  "category": "string (required, non-empty)",
  "embedding": [f0, f1, ...]  // 16 floats, required
}
```

**Validation error response (HTTP 200):**
```json
{"error":"invalid body"}
```
(returned if `parseBody` fails OR `embedding.size() != 16`)

**Success response shape:**
```json
{"id": 7}
```

**Status codes:** 200 only
**Parity class**: **STRICT**
**Frontend reads** (`index.html:623`): no fields read; just discards response and reloads items

---

### 2.4 `DELETE /delete/:id` — Delete Demo Vector

- **Method + path**: `DELETE /delete/<digits>` (regex `/delete/(\d+)`)
- **Handler**: `main.cpp:831-837`

**Path parameter:** `:id` — integer parsed from regex match group 1

**Response shape (always):**
```json
{"ok": true}    // if item existed
{"ok": false}   // if id not found
```

**Status codes:** 200 always
**Parity class**: **STRICT**
**Frontend reads** (`index.html:631`): no fields read; just calls then reloads

---

### 2.5 `GET /items` — List All Demo Vectors

- **Method + path**: `GET /items`
- **Handler**: `main.cpp:839-853`

**Query parameters:** none

**Response shape** (JSON array — top-level is array, NOT object):
```json
[
  {
    "id": 1,
    "metadata": "string",
    "category": "string",
    "embedding": [0.9000,0.8500,...]  // setprecision(4) via jVec
  }
]
```

**Float precision table:**

| Field | Precision | Notes |
|-------|-----------|-------|
| `embedding[]` | `setprecision(4)` | via `jVec` |

**Status codes:** 200 only
**Parity class**: **STRICT** (top-level array is not wrapped in an object)
**Frontend reads** (`index.html:509-519`): `id`, `embedding` (for PCA), `category`, `metadata`

---

### 2.6 `GET /benchmark` — Compare All Algorithms

- **Method + path**: `GET /benchmark`
- **Handler**: `main.cpp:855-869`

**Query parameters:**

| Param | Type | Default | Validation |
|-------|------|---------|------------|
| `v` | comma-delimited floats | required | length MUST equal 16 |
| `k` | int | `5` | parsed via `std::stoi` (throws → keeps default) |
| `metric` | string | `"cosine"` | empty → `"cosine"` |

**Validation error response (HTTP 200):**
```json
{"error":"need 16D vector"}
```

**Success response shape (no floats; all int):**
```json
{
  "bruteforceUs": 123,
  "kdtreeUs": 45,
  "hnswUs": 12,
  "itemCount": 20
}
```

**Float precision table:** N/A (all values are `long long` / `int`)

**Status codes:** 200 only
**Parity class**: **STRICT**
**Frontend reads** (`index.html:592-603`): `bruteforceUs`, `kdtreeUs`, `hnswUs`

---

### 2.7 `GET /hnsw-info` — HNSW Graph Topology

- **Method + path**: `GET /hnsw-info`
- **Handler**: `main.cpp:871-899`

**Query parameters:** none

**Response shape:**
```json
{
  "topLayer": 3,
  "nodeCount": 20,
  "nodesPerLayer": [1, 3, 8, 8],
  "edgesPerLayer": [0, 4, 16, 32],
  "nodes": [
    {
      "id": 1,
      "metadata": "string",
      "category": "string",
      "maxLyr": 2
    }
  ],
  "edges": [
    {
      "src": 1,
      "dst": 2,
      "lyr": 0
    }
  ]
}
```

**Float precision table:** N/A (all ints)

**Status codes:** 200 only
**Parity class**: **STRICT**
**Frontend reads** (`index.html:610-614`): `nodesPerLayer[]`, `edgesPerLayer[]` (parallel arrays by index)

---

### 2.8 `POST /doc/insert` — Insert Document (Chunked + Embedded)

- **Method + path**: `POST /doc/insert`
- **Handler**: `main.cpp:905-938`

**Request body** (`Content-Type: application/json`):
```json
{
  "title": "string (required, non-empty)",
  "text": "string (required, non-empty)"
}
```

**Validation error responses (HTTP 200, in this order):**

1. Missing title or text:
   ```json
   {"error":"need title and text"}
   ```

2. Ollama embed failure (returned immediately on first failed chunk):
   ```json
   {"error":"Ollama unavailable. Install from https://ollama.com then run: ollama pull nomic-embed-text && ollama pull llama3.2"}
   ```
   (Exact literal — see Special Case below)

**Success response shape:**
```json
{
  "ids": [1, 2, 3],
  "chunks": 3,
  "dims": 768
}
```

**Float precision table:** N/A (all ints)

**Status codes:** 200 only
**Parity class**: **STRICT**
**Frontend reads** (`index.html:680-684`): `d.chunks`, `d.dims` (success); `d.error` (failure)

**Title formatting for multi-chunk documents** (`main.cpp:926-928`): if `chunks.size() > 1`, each chunk title becomes `"<original> [i/N]"` where `i` is 1-indexed and `N` is total chunk count.

---

### 2.9 `DELETE /doc/delete/:id` — Delete Document Chunk

- **Method + path**: `DELETE /doc/delete/<digits>` (regex `/doc/delete/(\d+)`)
- **Handler**: `main.cpp:941-947`

**Path parameter:** `:id` — integer parsed from regex match group 1

**Response shape (always):**
```json
{"ok": true}    // if id existed
{"ok": false}   // if id not found
```

**Status codes:** 200 only
**Parity class**: **STRICT**
**Frontend reads** (`index.html:725`): no fields read; just calls then reloads

---

### 2.10 `GET /doc/list` — List Stored Document Chunks

- **Method + path**: `GET /doc/list`
- **Handler**: `main.cpp:950-967`

**Query parameters:** none

**Response shape** (JSON array — top-level is array):
```json
[
  {
    "id": 1,
    "title": "string",
    "preview": "string (truncated to 120 chars + '…' if longer)",
    "words": 247
  }
]
```

**Truncation rule** (`main.cpp:957-958`): `text.substr(0, 120)`; if `text.size() > 120`, append the literal U+2026 `…` character (3 bytes in UTF-8: `0xE2 0x80 0xA6`).

**`words` field calculation** (`main.cpp:962`): `(int)std::count(text.begin(), text.end(), ' ') + 1` — counts space characters, NOT whitespace, and assumes at least 1 word (even for empty text this returns 1, but empty text cannot exist in DocumentDB).

**Float precision table:** N/A (all ints)

**Status codes:** 200 only
**Parity class**: **STRICT**
**Frontend reads** (`index.html:711-718`): `id`, `title`, `preview`, `words`

---

### 2.11 `POST /doc/search` — Semantic Retrieval

- **Method + path**: `POST /doc/search`
- **Handler**: `main.cpp:971-996`

**Request body:**
```json
{
  "question": "string (required, non-empty)",
  "k": 3    // int, default 3
}
```

**Validation error responses (HTTP 200):**

1. Missing/empty question:
   ```json
   {"error":"need question"}
   ```

2. Ollama embed failure:
   ```json
   {"error":"Ollama unavailable"}
   ```

**Success response shape:**
```json
{
  "contexts": [
    {
      "id": 1,
      "title": "string",
      "distance": 0.1234    // setprecision(4), std::fixed
    }
  ]
}
```

**Float precision table:**

| Field | Precision | Notes |
|-------|-----------|-------|
| `contexts[].distance` | `setprecision(4)` | `std::fixed`; e.g. `0.1234` |

**Status codes:** 200 only
**Parity class**: **STRICT**
**Frontend reads** (`index.html:765-775`): `contexts[].title`, `contexts[].id` (via title prefix match against `pcaPoints`)

---

### 2.12 `POST /doc/ask` — RAG Pipeline (Embed + Retrieve + Generate)

- **Method + path**: `POST /doc/ask`
- **Handler**: `main.cpp:1000-1050`

**Request body:**
```json
{
  "question": "string (required, non-empty)",
  "k": 3    // int, default 3
}
```

**Validation error responses (HTTP 200):**

1. Missing/empty question:
   ```json
   {"error":"need question"}
   ```

2. Ollama embed failure:
   ```json
   {"error":"Ollama unavailable"}
   ```

**Success response shape:**
```json
{
  "answer": "string (LLM-generated)",
  "model": "llama3.2",
  "contexts": [
    {
      "id": 1,
      "title": "string",
      "text": "string (full chunk text, NOT truncated)",
      "distance": 0.1234    // setprecision(4)
    }
  ],
  "docCount": 7
}
```

**Float precision table:**

| Field | Precision | Notes |
|-------|-----------|-------|
| `contexts[].distance` | `setprecision(4)` | `std::fixed` |
| `docCount` | int | |
| `answer` | string (LLM output) | NOT numeric |

**Status codes:** 200 only
**Parity class**: **STRUCTURAL** for the `answer` field (LLM output is non-deterministic); **STRICT** for everything else (contexts, model name, docCount)
**Frontend reads** (`index.html:801-815`): `d.answer`, `d.model`, `d.contexts[].title`, `d.contexts[].text`, `d.contexts[].distance`

**Prompt template (CRITICAL — must be preserved)** (`main.cpp:1023-1031`):
```
You are a helpful assistant. Answer the user's question directly. Use the provided context if it contains relevant information. If it doesn't, just use your own general knowledge. IMPORTANT: Do NOT mention the 'context', 'provided text', or say things like 'the context doesn't mention'. Just answer the question naturally.

Context:
[1] <title1>:
<text1>

[2] <title2>:
<text2>

Question: <user question>

Answer:
```

---

### 2.13 `GET /status` — System Health

- **Method + path**: `GET /status`
- **Handler**: `main.cpp:1053-1065`

**Query parameters:** none

**Response shape:**
```json
{
  "ollamaAvailable": true,
  "embedModel": "nomic-embed-text",
  "genModel": "llama3.2",
  "docCount": 7,
  "docDims": 768,
  "demoDims": 16,
  "demoCount": 20
}
```

**Float precision table:** N/A (booleans + ints + strings)

**Status codes:** 200 only
**Parity class**: **STRICT**
**Frontend reads** (`index.html:645-660`): `ollamaAvailable`, `embedModel`, `genModel`, `docDims`, `docCount`

---

### 2.14 `GET /stats` — Vector Index Summary

- **Method + path**: `GET /stats`
- **Handler**: `main.cpp:1067-1075`

**Query parameters:** none

**Response shape:**
```json
{
  "count": 20,
  "dims": 16,
  "algorithms": ["bruteforce","kdtree","hnsw"],
  "metrics": ["euclidean","cosine","manhattan"]
}
```

**Float precision table:** N/A (ints + string arrays)

**Status codes:** 200 only
**Parity class**: **STRICT**
**Frontend reads**: NOT directly — the badge text reads from `/items` not `/stats` (the `#statsLabel` element shows `allItems.length` + DIMS from `loadItems`).

---

### 2.15 `GET /` — Static `index.html`

- **Method + path**: `GET /`
- **Handler**: `main.cpp:1078-1085`

**Response shape:** raw HTML file contents
**Content-Type:** `text/html`
**Status codes:** 200 (success) or 404 (file not found — when `index.html` cannot be opened from the working directory)
**Parity class**: **STRUCTURAL** (file content, but Content-Type and 404-on-missing-file behavior must match)
**Frontend reads:** the entire single-page app

---

## 3. Special Case: Exact `error` Message Strings

The C++ returns these exact literals. The Python port MUST preserve them byte-for-byte so the frontend can detect them via `d.error` truthiness checks.

| Route | Exact string |
|-------|--------------|
| `GET /search` (bad vector dims) | `need 16D vector` |
| `GET /benchmark` (bad vector dims) | `need 16D vector` |
| `POST /insert` (bad body or bad dims) | `invalid body` |
| `POST /doc/insert` (missing fields) | `need title and text` |
| `POST /doc/insert` (Ollama embed failure) | `Ollama unavailable. Install from https://ollama.com then run: ollama pull nomic-embed-text && ollama pull llama3.2` |
| `POST /doc/search` (missing question) | `need question` |
| `POST /doc/search` (Ollama embed failure) | `Ollama unavailable` |
| `POST /doc/ask` (missing question) | `need question` |
| `POST /doc/ask` (Ollama embed failure) | `Ollama unavailable` |
| `OllamaClient::generate` failure (returned in `answer` field of `/doc/ask`, NOT as top-level error) | `ERROR: Ollama unavailable. Run: ollama serve` |

**Distinction between embed and generate error messages**:
- The **embed** error path returns `{"error": "Ollama unavailable"}` (or the long install message on `/doc/insert`).
- The **generate** error path returns the error string in the `answer` field with the full response still shaped normally: `{"answer": "ERROR: Ollama unavailable. Run: ollama serve", "model": "...", "contexts": [...], "docCount": N}`.

**Distinction for `/doc/insert`**: the embed failure message is the LONG version (with install instructions) — different from the short `Ollama unavailable` used by `/doc/search` and `/doc/ask`.

---

## 4. Field Ordering, JSON Formatting, and Number Formatting

**Field ordering**: The C++ `std::ostringstream` writes keys in C++ source order (not alphabetical, not insertion order of any map). The Python port must emit keys in the exact order documented in Section 2 per route. Example for `/search` results:
```json
{"id":7,"metadata":"...","category":"...","distance":0.123456,"embedding":[...]}
```
NOT `{"category":"...","distance":...,...}` (alphabetical).

**Number formatting**:
- All floats use `std::fixed` (no scientific notation, no `e+05` form).
- `setprecision(N)` is set per-field, NOT globally — so each route's `setprecision` choice must be applied at the exact point of emission.
- `jVec` always uses `setprecision(4)`.
- `latencyUs` is a `long long` integer, not float.

**Array format**: `jVec` produces `[0.9000,0.8500,0.7000,...]` — no spaces between elements, no spaces after commas, no trailing zero-stripping. The `0.9000` style (trailing zeros preserved) is mandatory.

**String escaping** (via `jS`): only `"`, `\`, `\n`, `\r`, `\t` are escaped. Other characters pass through unchanged. The frontend embeds titles and previews directly into HTML via template literals, so characters like `<`, `>`, `&` are NOT escaped by the API (the frontend is responsible for its own XSS protection).

**Booleans**: written as `true` / `false` literals (not `1` / `0`).

**Integers**: no quotes, no decimal point, negative numbers include a leading `-` (e.g. `topLayer: -1` if HNSW is empty, though this is unlikely with seeded demo data).

---

## 5. Truncation Rules

| Route | Field | Rule |
|-------|-------|------|
| `GET /doc/list` | `preview` | `text.substr(0, 120)`; if `text.size() > 120`, append U+2026 `…` (single character, 3 UTF-8 bytes). The ellipsis is NOT a separate field. |
| `GET /doc/ask` | `contexts[].text` | FULL text (not truncated) — the frontend uses this for the expandable context panel. |
| `GET /doc/list` | `words` | `(int)std::count(text.begin(), text.end(), ' ') + 1` — counts ONLY ASCII space (U+0020), not tabs or newlines. Always at least 1. |

---

## 6. ID Assignment

All IDs are **1-indexed integers**, assigned monotonically at insertion time via the `nextId` member of each store:

- `VectorDB` (`main.cpp:342`, mutated in `insert` at `main.cpp:362`): IDs 1, 2, 3, ... across all demo vector inserts (including the 20 seeded by `loadDemo`).
- `DocumentDB` (`main.cpp:655`, mutated in `insert` at `main.cpp:670`): IDs 1, 2, 3, ... per chunk (so a 3-chunk document inserted today gets IDs 1-3, then a 2-chunk insert gets 4-5, etc.).

IDs are NOT reused after deletion. Deleting ID 5 and inserting again yields the next sequential ID (6), not 5.

---

## 7. What the Frontend Reads (per route)

| Route | Fields the frontend parses | Implication |
|-------|---------------------------|-------------|
| `GET /search` | `results[].id`, `results[].metadata`, `results[].category`, `results[].distance`; `latencyUs` | All keys + types must match |
| `POST /insert` | (none) | Body shape unimportant; only HTTP success matters |
| `DELETE /delete/:id` | (none) | Body shape unimportant |
| `GET /items` | `id`, `category`, `metadata`, `embedding` (full array for PCA) | All keys + types must match; top-level must be array |
| `GET /benchmark` | `bruteforceUs`, `kdtreeUs`, `hnswUs` | All keys + types must match |
| `GET /hnsw-info` | `nodesPerLayer[]`, `edgesPerLayer[]` (parallel arrays) | All keys + types must match |
| `POST /doc/insert` | `error` (failure); `chunks`, `dims` (success) | All keys + types must match |
| `DELETE /doc/delete/:id` | (none) | Body shape unimportant |
| `GET /doc/list` | `id`, `title`, `preview`, `words` | All keys + types must match; top-level must be array |
| `POST /doc/search` | `contexts[].title` (for prefix match against `pcaPoints`) | `title` field is the critical contract |
| `POST /doc/ask` | `answer`, `model`, `contexts[].title`, `contexts[].text`, `contexts[].distance` | `answer` is LLM-generated (loose); everything else strict |
| `GET /status` | `ollamaAvailable`, `embedModel`, `genModel`, `docDims`, `docCount` | All keys + types must match |
| `GET /stats` | (not called by frontend) | Informational; no contract pressure |
| `GET /` | HTML | File content |

---

## 8. Parity Classification

| Route | Class | Rationale |
|-------|-------|-----------|
| `OPTIONS .*` | **STRICT** | CORS headers must match |
| `GET /search` | **STRICT** | Frontend reads all fields; lat/distance formatting is displayed |
| `POST /insert` | **STRICT** | Body shape trivial but must remain JSON-valid |
| `DELETE /delete/:id` | **STRICT** | Body shape trivial |
| `GET /items` | **STRICT** | Top-level array + 4 fields per item are all consumed |
| `GET /benchmark` | **STRICT** | 4 fields all read |
| `GET /hnsw-info` | **STRICT** | Complex nested structure, 6 top-level keys |
| `POST /doc/insert` | **STRICT** | 3 top-level keys all read |
| `DELETE /doc/delete/:id` | **STRICT** | Body shape trivial |
| `GET /doc/list` | **STRICT** | 4 fields all read; truncation rule must match |
| `POST /doc/search` | **STRICT** | `contexts[].title` is matched against frontend's `pcaPoints` |
| `POST /doc/ask` | **STRUCTURAL** (mostly) | `answer` is LLM-generated (loose); contexts/model/docCount are strict |
| `GET /status` | **STRICT** | 7 fields all read |
| `GET /stats` | **STRICT** | Informational; not called by frontend but keep parity |
| `GET /` | **STRUCTURAL** | Static file; content matches; Content-Type and 404 behavior strict |

**Counts**: 14 HTTP routes (excluding `OPTIONS` preflight which is a CORS mechanism, not a feature route, but documented in the same 14 as the architect reported — if `OPTIONS` is counted, then 15; the freeze considers it route #1 in §2.1). For parity tally:
- **STRICT**: 12 routes (`OPTIONS`, `/search`, `/insert`, `/delete/:id`, `/items`, `/benchmark`, `/hnsw-info`, `/doc/insert`, `/doc/delete/:id`, `/doc/list`, `/doc/search`, `/status`, `/stats`)
- **STRUCTURAL**: 2 routes (`/doc/ask` for the `answer` field; `/` for the file)
- **LOOSE**: 0 routes (no route is entirely loose; even `/doc/ask` has strict metadata)

If the architect's "14 routes" excludes `OPTIONS` and `/`, then: 12 routes total, 11 STRICT, 1 STRUCTURAL (`/doc/ask`), 0 LOOSE. If we count all 14: 12 STRICT, 2 STRUCTURAL, 0 LOOSE.

---

**API contract frozen — 14 routes documented, 12 strict, 2 structural, 0 loose.**
