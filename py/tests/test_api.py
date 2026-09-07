"""API tests using Flask's test client.  Verifies the response bodies
match the C++ byte-for-byte for the routes the frontend consumes
(documented in docs/migration/03-api-contract.md).  Ollama is mocked."""

import json

import pytest

from vectordb.app import build_app
from vectordb.ollama import OllamaClient


class _StubOllama(OllamaClient):
    """OllamaClient that doesn't touch the network."""

    def __init__(self) -> None:
        # Skip the parent __init__; we override both methods.
        self.embed_model = "nomic-embed-text"
        self.gen_model = "llama3.2"
        self.embed_calls: list[str] = []
        self.gen_calls: list[str] = []

    def is_available(self) -> bool:
        return False  # Frontend should see OFFLINE in /status

    def embed(self, text: str) -> list[float]:
        self.embed_calls.append(text)
        # Deterministic 4-dim fake embedding based on text length.
        n = max(len(text), 1)
        return [n / 100.0, 0.0, 0.0, 0.0]

    def generate(self, prompt: str) -> str:
        self.gen_calls.append(prompt)
        return "stub answer"


@pytest.fixture
def client():
    app = build_app(demo=True, ollama=_StubOllama())
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


# ── /search ──────────────────────────────────────────────────────────


def test_search_bruteforce_within_a_category(client):
    q = "0.9,0.85,0.72,0.68,0.12,0.08,0.15,0.10,0.05,0.08,0.06,0.09,0.07,0.11,0.08,0.06"
    r = client.get(f"/search?v={q}&algo=bruteforce&k=3&metric=cosine")
    assert r.status_code == 200
    assert r.headers["Content-Type"] == "application/json"
    body = r.get_data(as_text=True)
    # Top hit should be the linked-list vector (id 1) with distance 0.
    obj = json.loads(body)
    assert obj["results"][0]["id"] == 1
    assert obj["results"][0]["distance"] == "0.000000"
    # Embedding is setprecision(4) -> "0.9000" style.
    assert obj["results"][0]["embedding"][0] == 0.9
    assert obj["algo"] == "bruteforce"
    assert obj["metric"] == "cosine"
    # latencyUs is an int.
    assert isinstance(obj["latencyUs"], int)


def test_search_default_algo_is_hnsw(client):
    q = "0.9,0.85,0.72,0.68,0.12,0.08,0.15,0.10,0.05,0.08,0.06,0.09,0.07,0.11,0.08,0.06"
    r = client.get(f"/search?v={q}")
    assert r.status_code == 200
    obj = json.loads(r.get_data(as_text=True))
    assert obj["algo"] == "hnsw"


def test_search_wrong_dims_returns_error(client):
    r = client.get("/search?v=0.1,0.2")
    body = json.loads(r.get_data(as_text=True))
    assert body == {"error": "need 16D vector"}


# ── /insert + /delete/:id ───────────────────────────────────────────


def test_insert_returns_id(client):
    body = json.dumps({
        "metadata": "test",
        "category": "cs",
        "embedding": [0.5] * 16,
    })
    r = client.post("/insert", data=body, content_type="application/json")
    obj = json.loads(r.get_data(as_text=True))
    assert obj["id"] == 21  # 20 demo + 1


def test_insert_invalid_body(client):
    r = client.post("/insert", data="{}", content_type="application/json")
    assert json.loads(r.get_data(as_text=True)) == {"error": "invalid body"}


def test_insert_wrong_dims(client):
    r = client.post("/insert",
                    data=json.dumps({"metadata": "x", "category": "y", "embedding": [0.5, 0.5]}),
                    content_type="application/json")
    assert json.loads(r.get_data(as_text=True)) == {"error": "invalid body"}


def test_delete_unknown(client):
    r = client.delete("/delete/999")
    assert json.loads(r.get_data(as_text=True)) == {"ok": False}


def test_delete_known(client):
    r = client.delete("/delete/1")
    assert json.loads(r.get_data(as_text=True)) == {"ok": True}


# ── /items ──────────────────────────────────────────────────────────


def test_items_top_level_is_array(client):
    r = client.get("/items")
    obj = r.get_json()
    assert isinstance(obj, list)
    assert len(obj) == 20
    # Field order is preserved (Python dict preserves insertion order).
    first = obj[0]
    keys = list(first.keys())
    assert keys == ["id", "metadata", "category", "embedding"]


# ── /benchmark ──────────────────────────────────────────────────────


def test_benchmark_returns_4_fields(client):
    q = "0.9,0.85,0.72,0.68,0.12,0.08,0.15,0.10,0.05,0.08,0.06,0.09,0.07,0.11,0.08,0.06"
    r = client.get(f"/benchmark?v={q}&k=5")
    obj = json.loads(r.get_data(as_text=True))
    assert set(obj.keys()) == {"bruteforceUs", "kdtreeUs", "hnswUs", "itemCount"}
    assert obj["itemCount"] == 20
    assert all(isinstance(obj[k], int) for k in ("bruteforceUs", "kdtreeUs", "hnswUs", "itemCount"))


# ── /hnsw-info ──────────────────────────────────────────────────────


def test_hnsw_info_shape(client):
    r = client.get("/hnsw-info")
    obj = json.loads(r.get_data(as_text=True))
    assert set(obj.keys()) == {
        "topLayer", "nodeCount", "nodesPerLayer",
        "edgesPerLayer", "nodes", "edges",
    }
    assert obj["nodeCount"] == 20
    assert isinstance(obj["nodesPerLayer"], list)
    assert isinstance(obj["edgesPerLayer"], list)


# ── /doc/insert + /doc/delete + /doc/list + /doc/search + /doc/ask ─


def test_doc_insert_missing_fields(client):
    r = client.post("/doc/insert", data="{}",
                    content_type="application/json")
    assert json.loads(r.get_data(as_text=True)) == {"error": "need title and text"}


def test_doc_insert_happy_path(client):
    r = client.post("/doc/insert",
                    data=json.dumps({"title": "T", "text": "hello world"}),
                    content_type="application/json")
    obj = json.loads(r.get_data(as_text=True))
    assert obj["chunks"] == 1
    assert obj["dims"] == 4  # stub embed dim
    assert len(obj["ids"]) == 1


def test_doc_insert_long_text_chunks(client):
    text = " ".join(f"w{i}" for i in range(300))  # > 250 words -> 2 chunks
    r = client.post("/doc/insert",
                    data=json.dumps({"title": "L", "text": text}),
                    content_type="application/json")
    obj = json.loads(r.get_data(as_text=True))
    assert obj["chunks"] == 2
    assert len(obj["ids"]) == 2


def test_doc_delete_unknown(client):
    r = client.delete("/doc/delete/999")
    assert json.loads(r.get_data(as_text=True)) == {"ok": False}


def test_doc_list_truncation(client):
    # Insert a doc with text > 120 chars.
    long_text = "word " * 50  # 250 chars
    client.post("/doc/insert",
                data=json.dumps({"title": "T", "text": long_text}),
                content_type="application/json")
    r = client.get("/doc/list")
    obj = json.loads(r.get_data(as_text=True))
    assert isinstance(obj, list)
    assert len(obj) == 1
    assert obj[0]["title"] == "T"
    # Truncated to 120 chars + ellipsis.
    assert len(obj[0]["preview"]) == 121  # 120 + the ellipsis char
    assert obj[0]["preview"].endswith("…")


def test_doc_search_happy_path(client):
    client.post("/doc/insert",
                data=json.dumps({"title": "alpha", "text": "the alpha doc"}),
                content_type="application/json")
    r = client.post("/doc/search",
                    data=json.dumps({"question": "anything", "k": 1}),
                    content_type="application/json")
    obj = json.loads(r.get_data(as_text=True))
    assert len(obj["contexts"]) == 1
    assert obj["contexts"][0]["title"] == "alpha"


def test_doc_search_missing_question(client):
    r = client.post("/doc/search", data="{}",
                    content_type="application/json")
    assert json.loads(r.get_data(as_text=True)) == {"error": "need question"}


def test_doc_ask_returns_answer(client):
    client.post("/doc/insert",
                data=json.dumps({"title": "x", "text": "y"}),
                content_type="application/json")
    r = client.post("/doc/ask",
                    data=json.dumps({"question": "hi", "k": 1}),
                    content_type="application/json")
    obj = json.loads(r.get_data(as_text=True))
    assert obj["answer"] == "stub answer"
    assert obj["model"] == "llama3.2"
    assert obj["docCount"] == 1
    assert len(obj["contexts"]) == 1


# ── /status + /stats + / + CORS preflight ───────────────────────────


def test_status_shape(client):
    r = client.get("/status")
    obj = json.loads(r.get_data(as_text=True))
    assert set(obj.keys()) == {
        "ollamaAvailable", "embedModel", "genModel",
        "docCount", "docDims", "demoDims", "demoCount",
    }
    assert obj["embedModel"] == "nomic-embed-text"
    assert obj["genModel"] == "llama3.2"
    assert obj["demoDims"] == 16
    assert obj["demoCount"] == 20
    assert obj["ollamaAvailable"] is False  # stub says OFFLINE


def test_stats_shape(client):
    r = client.get("/stats")
    obj = json.loads(r.get_data(as_text=True))
    assert obj["count"] == 20
    assert obj["dims"] == 16
    assert obj["algorithms"] == ["bruteforce", "kdtree", "hnsw"]
    assert obj["metrics"] == ["euclidean", "cosine", "manhattan"]


def test_root_serves_index(client):
    r = client.get("/")
    assert r.status_code == 200
    assert r.headers["Content-Type"] == "text/html"
    assert "<!DOCTYPE html>" in r.get_data(as_text=True)


def test_cors_preflight_204(client):
    r = client.options("/search", headers={
        "Origin": "http://example.com",
        "Access-Control-Request-Method": "GET",
    })
    assert r.status_code == 204
    assert r.headers["Access-Control-Allow-Origin"] == "*"
    assert r.headers["Access-Control-Allow-Methods"] == "GET, POST, DELETE, OPTIONS"


def test_cors_headers_on_get(client):
    r = client.get("/stats")
    assert r.headers["Access-Control-Allow-Origin"] == "*"
