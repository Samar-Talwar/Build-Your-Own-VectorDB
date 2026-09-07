"""Shim that mirrors the C++ `main.cpp` entry point: running this
file starts the VectorDB & RAG server on port 8080.

The implementation lives in `py/vectordb/app.py`.  The package is
imported through the `py/` directory's `__init__.py` (which adds
itself to `sys.path`).

Run with:
    python main.py
"""

from py.vectordb import run_server


if __name__ == "__main__":
    run_server()
