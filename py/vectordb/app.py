"""Flask application wiring all 14 HTTP routes.

Faithful Python port of the C++ `main()` function in main.cpp:766-1089.
Each handler mirrors its C++ counterpart line-for-line; the comments
cross-reference the original.  The Flask dev server is configured
single-threaded (D9 in 08-risks-and-decisions.md) to preserve the
C++'s request-handling semantics.

Routes:
  GET    /                 — static index.html
  OPTIONS *                 — CORS preflight
  GET    /search            — demo vector k-NN
  POST   /insert            — insert demo vector
  DELETE /delete/<id>       — delete demo vector
  GET    /items             — list all demo vectors
  GET    /benchmark         — time all 3 algorithms
  GET    /hnsw-info         — HNSW graph topology
  POST   /doc/insert        — chunk + embed + insert
  DELETE /doc/delete/<id>   — delete doc chunk
  GET    /doc/list          — list doc chunks
  POST   /doc/search        — semantic retrieval
  POST   /doc/ask           — RAG (embed + retrieve + generate)
  GET    /status            — health / counts
  GET    /stats             — index summary
"""

from __future__ import annotations

import os
import re
from typing import Optional

from flask import Flask, Response, request

from . import DIMS
from .chunker import chunk_text
from .db import VectorDB
from .demo_data import load_demo
from .document_db import DocumentDB
from .json_format import (
    CORS_HEADERS,
    err,
    extract_int,
    extract_str,
    f4,
    f4q,
    f6,
    f6q,
    j_s,
    j_vec,
    ok,
    parse_body,
    parse_vec,
)
from .ollama import OLLAMA_ERROR_LONG, OLLAMA_ERROR_SHORT, OllamaClient


# RAG prompt template (verbatim from main.cpp:1023-1031).
RAG_PROMPT_TEMPLATE = (
    "You are a helpful assistant. Answer the user's question directly. "
    "Use the provided context if it contains relevant information. "
    "If it doesn't, just use your own general knowledge. "
    "IMPORTANT: Do NOT mention the 'context', 'provided text', or say things like 'the context doesn't mention'. "
    "Just answer the question naturally.\n\n"
    "Context:\n{context}\n"
    "Question: {question}\n\n"
    "Answer:"
)


def _cors(res: Response) -> None:
    """Apply the three CORS headers to a Flask response."""
    for k, v in CORS_HEADERS.items():
        res.headers[k] = v


def _json(res: Response, body: str, status: int = 200) -> Response:
    """Set Content-Type and body; the C++ always returns 200 (even on
    validation errors) — callers pass status=200 by default."""
    res.status = status
    res.headers["Content-Type"] = "application/json"
    res.set_data(body)
    return res


def build_app(
    db: Optional[VectorDB] = None,
    doc_db: Optional[DocumentDB] = None,
    ollama: Optional[OllamaClient] = None,
    demo: bool = True,
) -> Flask:
    """Construct the Flask app.  Used by the test suite to inject mocks.

    `demo=True` populates the 20 demo vectors at startup; the test suite
    passes `demo=False` to start with an empty database.
    """
    app = Flask(__name__)
    # Disable Flask's auto-OPTIONS so our explicit handler (below) is the
    # only responder.  Without this, every GET/POST route also gets a
    # default OPTIONS handler that returns 200 — which breaks the CORS
    # parity contract (the C++ returns 204).
    app.config["PROVIDE_AUTOMATIC_OPTIONS"] = False
    db = db or VectorDB(DIMS)
    doc_db = doc_db or DocumentDB()
    ollama = ollama or OllamaClient()
    if demo:
        load_demo(db)

    # -- CORS preflight (handled in before_request so it short-circuits) -

    @app.before_request
    def _cors_preflight() -> Optional[Response]:
        if request.method == "OPTIONS":
            res = Response(status=204)
            _cors(res)
            return res
        return None

    # -- Demo vector endpoints ------------------------------------------

    @app.get("/search")
    def _search() -> Response:
        res = Response()
        _cors(res)
        q = parse_vec(request.args.get("v", ""))
        if len(q) != DIMS:
            return _json(res, err(f"need {DIMS}D vector"))
        try:
            k = int(request.args.get("k", "5"))
        except (TypeError, ValueError):
            k = 5
        metric = request.args.get("metric") or "cosine"
        algo = request.args.get("algo") or "hnsw"
        out = db.search(q, k, metric, algo)
        parts: list[str] = ['{"results":[']
        for i, h in enumerate(out.hits):
            if i:
                parts.append(",")
            parts.append(
                '{"id":' + str(h.id)
                + ',"metadata":' + j_s(h.metadata)
                + ',"category":' + j_s(h.category)
                + ',"distance":' + f6q(h.dist)
                + ',"embedding":' + j_vec(h.emb) + "}"
            )
        parts.append(
            '],"latencyUs":' + str(out.us)
            + ',"algo":' + j_s(out.algo)
            + ',"metric":' + j_s(out.metric) + "}"
        )
        return _json(res, "".join(parts))

    @app.post("/insert")
    def _insert() -> Response:
        res = Response()
        _cors(res)
        parsed = parse_body(request.get_data(as_text=True))
        if parsed is None or len(parsed[2]) != DIMS:
            return _json(res, err("invalid body"))
        meta, cat, emb = parsed
        from .distances import get_dist_fn
        id_ = db.insert(meta, cat, emb, get_dist_fn("cosine"))
        return _json(res, '{"id":' + str(id_) + "}")

    @app.delete("/delete/<int:id_>")
    def _delete(id_: int) -> Response:
        res = Response()
        _cors(res)
        return _json(res, ok(db.remove(id_)))

    @app.get("/items")
    def _items() -> Response:
        res = Response()
        _cors(res)
        items = db.all()
        parts: list[str] = ["["]
        for i, v in enumerate(items):
            if i:
                parts.append(",")
            parts.append(
                '{"id":' + str(v.id)
                + ',"metadata":' + j_s(v.metadata)
                + ',"category":' + j_s(v.category)
                + ',"embedding":' + j_vec(v.emb) + "}"
            )
        parts.append("]")
        return _json(res, "".join(parts))

    @app.get("/benchmark")
    def _benchmark() -> Response:
        res = Response()
        _cors(res)
        q = parse_vec(request.args.get("v", ""))
        if len(q) != DIMS:
            return _json(res, err(f"need {DIMS}D vector"))
        try:
            k = int(request.args.get("k", "5"))
        except (TypeError, ValueError):
            k = 5
        metric = request.args.get("metric") or "cosine"
        b = db.benchmark(q, k, metric)
        return _json(
            res,
            '{"bruteforceUs":' + str(b.bf_us)
            + ',"kdtreeUs":' + str(b.kd_us)
            + ',"hnswUs":' + str(b.hnsw_us)
            + ',"itemCount":' + str(b.n) + "}",
        )

    @app.get("/hnsw-info")
    def _hnsw_info() -> Response:
        res = Response()
        _cors(res)
        gi = db.hnsw_info()
        parts: list[str] = [
            '{"topLayer":' + str(gi.top_layer)
            + ',"nodeCount":' + str(gi.node_count)
            + ',"nodesPerLayer":['
        ]
        for i, n in enumerate(gi.nodes_per_layer):
            if i:
                parts.append(",")
            parts.append(str(n))
        parts.append('],"edgesPerLayer":[')
        for i, n in enumerate(gi.edges_per_layer):
            if i:
                parts.append(",")
            parts.append(str(n))
        parts.append('],"nodes":[')
        for i, n in enumerate(gi.nodes):
            if i:
                parts.append(",")
            parts.append(
                '{"id":' + str(n.id)
                + ',"metadata":' + j_s(n.metadata)
                + ',"category":' + j_s(n.category)
                + ',"maxLyr":' + str(n.max_lyr) + "}"
            )
        parts.append('],"edges":[')
        for i, e in enumerate(gi.edges):
            if i:
                parts.append(",")
            parts.append('{"src":' + str(e.src) + ',"dst":' + str(e.dst) + ',"lyr":' + str(e.lyr) + "}")
        parts.append("]}")
        return _json(res, "".join(parts))

    # -- Document + RAG endpoints ----------------------------------------

    @app.post("/doc/insert")
    def _doc_insert() -> Response:
        res = Response()
        _cors(res)
        body = request.get_data(as_text=True)
        title = extract_str(body, "title")
        text = extract_str(body, "text")
        if not title or not text:
            return _json(res, err("need title and text"))
        chunks = chunk_text(text, 250, 30)
        ids: list[int] = []
        for i, chunk in enumerate(chunks):
            emb = ollama.embed(chunk)
            if not emb:
                return _json(res, err(OLLAMA_ERROR_LONG))
            chunk_title = (
                f"{title} [{i + 1}/{len(chunks)}]" if len(chunks) > 1 else title
            )
            ids.append(doc_db.insert(chunk_title, chunk, emb))
        parts: list[str] = ['{"ids":[']
        for i, id_ in enumerate(ids):
            if i:
                parts.append(",")
            parts.append(str(id_))
        parts.append(
            '],"chunks":' + str(len(chunks))
            + ',"dims":' + str(doc_db.get_dims()) + "}"
        )
        return _json(res, "".join(parts))

    @app.delete("/doc/delete/<int:id_>")
    def _doc_delete(id_: int) -> Response:
        res = Response()
        _cors(res)
        return _json(res, ok(doc_db.remove(id_)))

    @app.get("/doc/list")
    def _doc_list() -> Response:
        res = Response()
        _cors(res)
        docs = doc_db.all()
        parts: list[str] = ["["]
        for i, d in enumerate(docs):
            if i:
                parts.append(",")
            preview = d.text[:120]
            if len(d.text) > 120:
                preview += "…"  # U+2026 HORIZONTAL ELLIPSIS
            words = d.text.count(" ") + 1
            parts.append(
                '{"id":' + str(d.id)
                + ',"title":' + j_s(d.title)
                + ',"preview":' + j_s(preview)
                + ',"words":' + str(words) + "}"
            )
        parts.append("]")
        return _json(res, "".join(parts))

    @app.post("/doc/search")
    def _doc_search() -> Response:
        res = Response()
        _cors(res)
        body = request.get_data(as_text=True)
        question = extract_str(body, "question")
        k = extract_int(body, "k", 3)
        if not question:
            return _json(res, err("need question"))
        q_emb = ollama.embed(question)
        if not q_emb:
            return _json(res, err(OLLAMA_ERROR_SHORT))
        hits = doc_db.search(q_emb, k)
        parts: list[str] = ['{"contexts":[']
        for i, (dist, d) in enumerate(hits):
            if i:
                parts.append(",")
            parts.append(
                '{"id":' + str(d.id)
                + ',"title":' + j_s(d.title)
                + ',"distance":' + f4q(dist) + "}"
            )
        parts.append("]}")
        return _json(res, "".join(parts))

    @app.post("/doc/ask")
    def _doc_ask() -> Response:
        res = Response()
        _cors(res)
        body = request.get_data(as_text=True)
        question = extract_str(body, "question")
        k = extract_int(body, "k", 3)
        if not question:
            return _json(res, err("need question"))
        q_emb = ollama.embed(question)
        if not q_emb:
            return _json(res, err(OLLAMA_ERROR_SHORT))
        hits = doc_db.search(q_emb, k)
        ctx_parts: list[str] = []
        for i, (_, d) in enumerate(hits):
            ctx_parts.append(f"[{i + 1}] {d.title}:\n{d.text}\n\n")
        prompt = RAG_PROMPT_TEMPLATE.format(
            context="".join(ctx_parts), question=question
        )
        answer = ollama.generate(prompt)
        parts: list[str] = [
            '{"answer":' + j_s(answer)
            + ',"model":' + j_s(ollama.gen_model)
            + ',"contexts":['
        ]
        for i, (dist, d) in enumerate(hits):
            if i:
                parts.append(",")
            parts.append(
                '{"id":' + str(d.id)
                + ',"title":' + j_s(d.title)
                + ',"text":' + j_s(d.text)
                + ',"distance":' + f4q(dist) + "}"
            )
        parts.append('],"docCount":' + str(doc_db.size()) + "}")
        return _json(res, "".join(parts))

    @app.get("/status")
    def _status() -> Response:
        res = Response()
        _cors(res)
        up = ollama.is_available()
        return _json(
            res,
            '{"ollamaAvailable":' + ("true" if up else "false")
            + ',"embedModel":' + j_s(ollama.embed_model)
            + ',"genModel":' + j_s(ollama.gen_model)
            + ',"docCount":' + str(doc_db.size())
            + ',"docDims":' + str(doc_db.get_dims())
            + ',"demoDims":' + str(DIMS)
            + ',"demoCount":' + str(db.size()) + "}",
        )

    @app.get("/stats")
    def _stats() -> Response:
        res = Response()
        _cors(res)
        return _json(
            res,
            '{"count":' + str(db.size())
            + ',"dims":' + str(DIMS)
            + ',"algorithms":["bruteforce","kdtree","hnsw"]'
            + ',"metrics":["euclidean","cosine","manhattan"]}',
        )

    @app.route("/", methods=["GET", "OPTIONS"])
    def _root() -> Response:
        # Static index.html
        candidates = [
            os.path.join(os.path.dirname(__file__), "..", "..", "index.html"),
            "index.html",
        ]
        for path in candidates:
            path = os.path.abspath(path)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    res = Response(f.read())
                    res.headers["Content-Type"] = "text/html"
                    _cors(res)
                    return res
            except OSError:
                continue
        res = Response(status=404)
        _cors(res)
        return res

    return app


def run_server(host: str = "0.0.0.0", port: int = 8080) -> int:  # pragma: no cover
    """Entry point used by `python -m vectordb`."""
    from .ollama import OllamaClient
    app = build_app()
    ollama = OllamaClient()
    up = ollama.is_available()
    print("=== VectorDB Engine ===")
    print(f"http://localhost:{port}")
    print(f"{DIMS} dims | HNSW+KD-Tree+BruteForce")
    print(f"Ollama: {'ONLINE' if up else 'OFFLINE (install from ollama.com)'}")
    if up:
        print(f"  embed model: {ollama.embed_model}  gen model: {ollama.gen_model}")
    # Single-threaded (D9) so request handling matches the C++.
    app.run(host=host, port=port, threaded=False, use_reloader=False)
    return 0
