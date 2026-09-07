"""VectorDB & RAG engine — Python port of main.cpp.

A small in-memory vector database and retrieval-augmented generation service.
Three search algorithms (BruteForce, KD-Tree, HNSW), three distance metrics
(Euclidean, Cosine, Manhattan), and a RAG pipeline backed by a local Ollama
server.  See README.md and docs/migration/ for design notes.
"""

__version__ = "2.0.0"
__all__ = ["DIMS", "OLLAMA_HOST", "OLLAMA_PORT"]

# Demo vectors are 16-dimensional; document embeddings are determined at
# runtime by whichever Ollama model is in use.
DIMS = 16

# Default Ollama endpoint.
OLLAMA_HOST = "127.0.0.1"
OLLAMA_PORT = 11434
