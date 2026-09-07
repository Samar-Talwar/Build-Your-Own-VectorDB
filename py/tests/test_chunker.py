"""Tests for the text chunker.  Verifies the C++ short-circuit
(returns the original text when it fits in one window) and the
sliding-window step."""

from vectordb.chunker import chunk_text


def test_empty_input():
    assert chunk_text("") == []


def test_short_input_returns_single_chunk():
    text = "hello world foo"
    out = chunk_text(text, chunk_words=250, overlap_words=30)
    assert out == [text]


def test_long_input_produces_overlapping_chunks():
    words = [f"w{i}" for i in range(300)]
    out = chunk_text(" ".join(words), chunk_words=250, overlap_words=30)
    # step = 220, so indices 0, 220 -> 2 chunks.
    assert len(out) == 2
    # First chunk is 250 words.
    assert len(out[0].split()) == 250
    # Last chunk is whatever remains (300 - 220 = 80).
    assert len(out[1].split()) == 80


def test_step_equals_window_minus_overlap():
    # 100 words, window 10, overlap 2 -> step 8, ceil(100/8) = 13 chunks.
    text = " ".join(f"w{i}" for i in range(100))
    out = chunk_text(text, chunk_words=10, overlap_words=2)
    # First chunk has 10 words, then step 8, last chunk has the tail.
    assert out[0].split()[0] == "w0"
    assert out[0].split()[-1] == "w9"
    # The second chunk starts 8 words after the first.
    assert out[1].split()[0] == "w8"
