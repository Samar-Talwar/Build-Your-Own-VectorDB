"""JSON wire-format helpers that match the C++ byte-exact.

The frontend parses the response bodies via `JSON.parse`, so the field
order, float precision, and string escaping must match the C++ output
exactly.  This module centralises the building blocks:

  * `j_s(s)` — quote-escape a string (matches `jS` in main.cpp:435-446).
  * `j_vec(v)` — comma-joined floats with setprecision(4) (main.cpp:448-455).
  * `f4(x)` / `f6(x)` — pre-formatted float strings (setprecision(4) / (6)).
  * `parse_vec(s)` — parse a comma-delimited float string (main.cpp:457-463).
  * `extract_str(body, key)` / `extract_int(body, key, def)` — minimal
    JSON field extractors tolerant of un-escaped newlines in the body
    (matches `extractStr` / `extractInt` at main.cpp:466-501).
  * `parse_body(b, meta, cat, emb)` — convenience wrapper used by
    `/insert` (main.cpp:503-519).
  * `cors_headers()` — CORS header dict (main.cpp:521-525).

We do NOT use `json.dumps` for response bodies because the C++ emits keys
in C++ source order with no spaces.  `json.dumps` would sort keys
alphabetically and add whitespace.  Instead the route handlers build the
JSON string by hand, calling these helpers per field.
"""

from __future__ import annotations

import json
from typing import List, Tuple

# CORS headers (main.cpp:521-525)
CORS_HEADERS: dict[str, str] = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, POST, DELETE, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type",
}


def j_s(s: str) -> str:
    """Quote-escape a string the way the C++ `jS` does.  Only the
    characters `"`, `\\`, `\\n`, `\\r`, `\\t` are escaped.  Other
    characters (including `<`, `>`, `&`) pass through unchanged —
    the frontend is responsible for its own XSS protection.
    """
    out = ['"']
    for c in s:
        if c == '"':
            out.append('\\"')
        elif c == "\\":
            out.append("\\\\")
        elif c == "\n":
            out.append("\\n")
        elif c == "\r":
            out.append("\\r")
        elif c == "\t":
            out.append("\\t")
        else:
            out.append(c)
    out.append('"')
    return "".join(out)


def f4(x: float) -> str:
    """Format a float as `setprecision(4)` does — `0.1234` style with
    fixed decimal point and trailing zeros preserved."""
    return f"{x:.4f}"


def f6(x: float) -> str:
    """Format a float as `setprecision(6)` does — `0.123456` style."""
    return f"{x:.6f}"


def f6q(x: float) -> str:
    """f6 wrapped in quotes — used for the `distance` field in JSON
    responses, which the C++ emits as a string (the frontend parses it)."""
    return f'"{f6(x)}"'


def f4q(x: float) -> str:
    """f4 wrapped in quotes — used for the `distance` field in /doc/* responses."""
    return f'"{f4(x)}"'


def j_vec(v: List[float]) -> str:
    """Render a vector as `[0.9000,0.8500,...]` with no spaces.  Port of
    main.cpp:448-455."""
    return "[" + ",".join(f4(x) for x in v) + "]"


def parse_vec(s: str) -> List[float]:
    """Parse a comma-delimited float string.  Tolerant of empty values
    and trailing commas.  Port of main.cpp:457-463."""
    if not s:
        return []
    out: List[float] = []
    for t in s.split(","):
        t = t.strip()
        if not t:
            continue
        try:
            out.append(float(t))
        except ValueError:
            # Matches the C++ which swallows the parse exception silently.
            continue
    return out


def _find_key(body: str, key: str) -> int:
    """Return the index just after the colon following `"<key>"`, or -1."""
    needle = '"' + key + '"'
    p = body.find(needle)
    if p < 0:
        return -1
    p = body.find(":", p + len(needle))
    if p < 0:
        return -1
    p += 1
    while p < len(body) and body[p] in " \t":
        p += 1
    return p


def extract_str(body: str, key: str) -> str:
    """Read a string field by key.  Tolerant of escape sequences
    (`\\"`, `\\\\`, `\\n`, `\\r`, `\\t`).  Port of main.cpp:466-492."""
    p = _find_key(body, key)
    if p < 0 or p >= len(body) or body[p] != '"':
        return ""
    p += 1
    out: List[str] = []
    while p < len(body):
        c = body[p]
        if c == '"':
            break
        if c == "\\" and p + 1 < len(body):
            p += 1
            nxt = body[p]
            if nxt == '"':
                out.append('"')
            elif nxt == "\\":
                out.append("\\")
            elif nxt == "n":
                out.append("\n")
            elif nxt == "r":
                out.append("\r")
            elif nxt == "t":
                out.append("\t")
            else:
                out.append(nxt)
        else:
            out.append(c)
        p += 1
    return "".join(out)


def extract_int(body: str, key: str, default: int = 0) -> int:
    """Read an int field by key.  Port of main.cpp:495-501."""
    p = _find_key(body, key)
    if p < 0:
        return default
    end = p
    while end < len(body) and (body[end].isdigit() or body[end] == "-"):
        end += 1
    if end == p:
        return default
    try:
        return int(body[p:end])
    except ValueError:
        return default


def parse_body(
    body: str,
) -> Tuple[str, str, List[float]] | None:
    """Parse `{metadata, category, embedding:[...]}` from a request body.
    Returns None if the body is missing required fields.  Port of
    main.cpp:503-519 — the C++ requires non-empty `metadata` and a
    non-empty `embedding`; we follow the same rule.
    """
    meta = extract_str(body, "metadata")
    cat = extract_str(body, "category")
    p = body.find('"embedding"')
    if p < 0:
        emb: List[float] = []
    else:
        p = body.find("[", p)
        if p < 0:
            emb = []
        else:
            e = body.find("]", p)
            if e < 0:
                emb = []
            else:
                emb = parse_vec(body[p + 1:e])
    if not meta or not emb:
        return None
    return meta, cat, emb


def ok(ok_: bool) -> str:
    """Render `{"ok":true}` / `{"ok":false}` byte-exact."""
    return '{"ok":' + ("true" if ok_ else "false") + "}"


def err(msg: str) -> str:
    """Render `{"error":"<literal>"}` byte-exact (used by every 4xx-ish
    response — note the C++ always returns HTTP 200, even for errors)."""
    return '{"error":' + j_s(msg) + "}"


def dumps(data) -> str:
    """For the few responses that DON'T need hand-built JSON (none in
    the C++; this is a placeholder for future Python-only routes)."""
    return json.dumps(data, separators=(",", ":"))
