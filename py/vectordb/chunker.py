"""Text chunker — direct port of main.cpp:531-552.

Splits a long string into overlapping word-windows.  The defaults
(250 words, 30-word overlap) match the C++ exactly and are referenced
by `/doc/insert`.

The C++ has a deliberate short-circuit: if the input has fewer than
`chunk_words` words, the original text is returned as a single chunk.
The port preserves that behaviour bit-for-bit — including the empty-
input edge case (returns an empty list).
"""

from __future__ import annotations

from typing import List


def chunk_text(
    text: str,
    chunk_words: int = 250,
    overlap_words: int = 30,
) -> List[str]:
    """Sliding-window chunker.  Port of main.cpp:531-552.

    Returns a single-element list containing the original text when the
    document fits within one window; otherwise returns overlapping
    chunks joined by single spaces.  The `step` (window advance) is
    `chunk_words - overlap_words`, and the final chunk is truncated
    at the end of the document (no padding).
    """
    words = text.split()
    if not words:
        return []
    if len(words) <= chunk_words:
        return [text]
    chunks: List[str] = []
    step = chunk_words - overlap_words
    i = 0
    n = len(words)
    while i < n:
        end = min(i + chunk_words, n)
        chunks.append(" ".join(words[i:end]))
        if end == n:
            break
        i += step
    return chunks
