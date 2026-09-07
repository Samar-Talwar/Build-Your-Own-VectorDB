"""Client for the local Ollama REST API.

Port of main.cpp:561-638 (`OllamaClient`).  Hits three endpoints:

  * `GET /api/tags`             — health probe (timeout 2s).
  * `POST /api/embeddings`      — embed a string (timeouts 3s/30s).
  * `POST /api/generate`        — prompt the LLM (timeouts 3s/180s).

`generate` returns the error string in-band (matching the C++ which
stuffs the LLM failure message into the `answer` field of `/doc/ask`).
`embed` returns an empty list on any failure, with two exception
message strings exposed for the routes that surface them:

  * `OLLAMA_ERROR_LONG`  — for `/doc/insert` (with install instructions).
  * `OLLAMA_ERROR_SHORT` — for `/doc/search` and `/doc/ask`.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import List, Optional


# Exact strings the C++ uses; the frontend parses these byte-exact.
OLLAMA_ERROR_LONG = (
    "Ollama unavailable. Install from https://ollama.com then run: "
    "ollama pull nomic-embed-text && ollama pull llama3.2"
)
OLLAMA_ERROR_SHORT = "Ollama unavailable"

# generate() returns this in the `answer` field when the LLM call fails
# — NOT in the top-level `error` field.  See /doc/ask contract.
OLLAMA_GENERATE_ERROR = "ERROR: Ollama unavailable. Run: ollama serve"


class OllamaClient:
    """Thin urllib-based wrapper around the local Ollama server."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 11434,
        embed_model: str = "nomic-embed-text",
        gen_model: str = "llama3.2",
    ) -> None:
        self.host = host
        self.port = port
        self.embed_model = embed_model
        self.gen_model = gen_model

    # -- low-level request helper ---------------------------------------

    def _request(
        self,
        method: str,
        path: str,
        body: Optional[dict] = None,
        connect_timeout: float = 3.0,
        read_timeout: float = 30.0,
    ) -> Optional[bytes]:
        url = f"http://{self.host}:{self.port}{path}"
        data: Optional[bytes] = None
        headers: dict[str, str] = {}
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=read_timeout) as resp:
                return resp.read()
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
            return None

    # -- public API ------------------------------------------------------

    def is_available(self) -> bool:
        """Health probe — 2-second connect timeout, GET /api/tags."""
        url = f"http://{self.host}:{self.port}/api/tags"
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=2):
                return True
        except Exception:
            return False

    def embed(self, text: str) -> List[float]:
        """Return the embedding for `text`, or [] on failure.

        The C++ uses `httplib::Client` with separate connect (3s) and
        read (30s) timeouts; `urllib` only exposes a single timeout, so
        we use the sum.  Failures (any URLError, status != 200) return
        an empty list, matching main.cpp:621-622.
        """
        body = {"model": self.embed_model, "prompt": text}
        raw = self._request("POST", "/api/embeddings", body=body,
                            connect_timeout=3.0, read_timeout=30.0)
        if not raw:
            return []
        try:
            obj = json.loads(raw)
            emb = obj.get("embedding", [])
            return [float(x) for x in emb]
        except (json.JSONDecodeError, TypeError, ValueError):
            return []

    def generate(self, prompt: str) -> str:
        """Call the LLM.  Returns `OLLAMA_GENERATE_ERROR` on failure.

        Matches the C++ which stuffs the error string into the
        `/doc/ask` `answer` field (rather than returning a top-level
        `{"error":...}`).  See R18 in 08-risks-and-decisions.md —
        this is a preserved quirk, not a bug.
        """
        body = {"model": self.gen_model, "prompt": prompt, "stream": False}
        raw = self._request("POST", "/api/generate", body=body,
                            connect_timeout=3.0, read_timeout=180.0)
        if not raw:
            return OLLAMA_GENERATE_ERROR
        try:
            obj = json.loads(raw)
            return str(obj.get("response", OLLAMA_GENERATE_ERROR))
        except (json.JSONDecodeError, TypeError):
            return OLLAMA_GENERATE_ERROR
