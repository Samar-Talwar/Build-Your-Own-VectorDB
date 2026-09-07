"""Tests for the Ollama client.  Uses a stub HTTP server (built with
the stdlib) so the test doesn't depend on a real Ollama instance."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from vectordb.ollama import OLLAMA_ERROR_LONG, OLLAMA_GENERATE_ERROR, OLLAMA_ERROR_SHORT, OllamaClient


class _Handler(BaseHTTPRequestHandler):
    """Simple in-process Ollama stub.  Routes by path."""
    embedding_dim = 4

    def do_GET(self):
        if self.path == "/api/tags":
            self._ok(b'{"models":[]}')
        else:
            self._not_found()

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        _ = self.rfile.read(length)
        if self.path == "/api/embeddings":
            payload = json.dumps({"embedding": [0.1] * _Handler.embedding_dim}).encode()
            self._ok(payload)
        elif self.path == "/api/generate":
            payload = json.dumps({"response": "ok"}).encode()
            self._ok(payload)
        else:
            self._not_found()

    def _ok(self, body: bytes):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _not_found(self):
        self.send_response(404)
        self.end_headers()

    def log_message(self, fmt, *args):  # silence stderr
        pass


@pytest.fixture
def ollama_stub():
    srv = HTTPServer(("127.0.0.1", 0), _Handler)
    port = srv.server_address[1]
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    yield f"127.0.0.1:{port}"
    srv.shutdown()
    srv.server_close()


def test_is_available_true(ollama_stub):
    host, port = ollama_stub.split(":")
    c = OllamaClient(host=host, port=int(port))
    assert c.is_available() is True


def test_embed_returns_list(ollama_stub):
    host, port = ollama_stub.split(":")
    c = OllamaClient(host=host, port=int(port))
    out = c.embed("hello")
    assert out == [0.1] * 4


def test_generate_returns_string(ollama_stub):
    host, port = ollama_stub.split(":")
    c = OllamaClient(host=host, port=int(port))
    assert c.generate("hi") == "ok"


def test_unavailable_returns_short_error_message():
    # Point at a closed port -> everything fails fast.
    c = OllamaClient(host="127.0.0.1", port=1)  # unprivileged, no listener
    assert c.is_available() is False
    assert c.embed("x") == []  # embed returns empty on failure
    assert c.generate("x") == OLLAMA_GENERATE_ERROR


def test_error_string_constants_are_distinct():
    # The frontend parses these by exact match — they must not drift.
    assert OLLAMA_ERROR_LONG != OLLAMA_ERROR_SHORT
    assert "ollama.com" in OLLAMA_ERROR_LONG
    assert "ollama.com" not in OLLAMA_ERROR_SHORT
    assert OLLAMA_GENERATE_ERROR.startswith("ERROR: ")
