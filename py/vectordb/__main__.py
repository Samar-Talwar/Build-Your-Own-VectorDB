"""Entry point: ``python -m vectordb`` starts the HTTP server on :8080.

Mirrors the behaviour of the original C++ binary (main.cpp:1087) which binds
0.0.0.0:8080.  We use Flask's development server in single-threaded mode to
preserve the C++'s request-handling semantics (see D9 in
docs/migration/08-risks-and-decisions.md).
"""

from __future__ import annotations

import sys

from .app import build_app, run_server


def main() -> int:
    return run_server()


if __name__ == "__main__":
    sys.exit(main())
