"""Tests for the JSON wire-format helpers.

These verify the byte-exact properties the frontend depends on:
  * `j_s` escapes only `"`, `\\`, `\\n`, `\\r`, `\\t`.
  * `j_vec` uses setprecision(4) with no spaces.
  * `f4` / `f6` use fixed-point with the specified precision.
  * `parse_vec` is tolerant of empty / malformed values.
  * `extract_str` / `extract_int` handle the same hand-rolled JSON the
    C++ accepts (and reject nothing legitimate).
  * `err` / `ok` produce the exact literal bodies.
"""

from vectordb.json_format import (
    err,
    extract_int,
    extract_str,
    f4,
    f6,
    j_s,
    j_vec,
    ok,
    parse_vec,
)


def test_j_s_escapes_quotes():
    assert j_s('a"b') == '"a\\"b"'


def test_j_s_escapes_backslash():
    assert j_s("a\\b") == '"a\\\\b"'


def test_j_s_escapes_newline():
    assert j_s("a\nb") == '"a\\nb"'


def test_j_s_preserves_angle_brackets():
    # The C++ does NOT escape <, >, & — the frontend is responsible
    # for its own XSS.  The port must match.
    assert j_s("<script>") == '"<script>"'


def test_j_vec_precision_4():
    assert j_vec([0.9, 0.85, 0.72]) == "[0.9000,0.8500,0.7200]"


def test_f4_trailing_zeros_preserved():
    assert f4(0.9) == "0.9000"
    assert f4(0.123456) == "0.1235"  # rounded, still 4 decimals


def test_f6_precision():
    assert f6(0.123456) == "0.123456"
    assert f6(0.5) == "0.500000"


def test_parse_vec_basic():
    assert parse_vec("0.9,0.85,0.72") == [0.9, 0.85, 0.72]


def test_parse_vec_tolerates_spaces():
    assert parse_vec(" 0.9 , 0.85 , 0.72 ") == [0.9, 0.85, 0.72]


def test_parse_vec_empty_string():
    assert parse_vec("") == []


def test_parse_vec_skips_garbage():
    # The C++ swallows parse exceptions silently; the port does the same.
    assert parse_vec("0.9,abc,0.5") == [0.9, 0.5]


def test_extract_str_simple():
    assert extract_str('{"title":"hello","x":1}', "title") == "hello"


def test_extract_str_with_escapes():
    body = '{"title":"hello\\"world","x":1}'
    assert extract_str(body, "title") == 'hello"world'


def test_extract_str_missing_returns_empty():
    assert extract_str('{"x":1}', "title") == ""


def test_extract_int():
    assert extract_int('{"k":42}', "k") == 42
    assert extract_int('{"k":42}', "k", 99) == 42
    assert extract_int('{}', "k", 7) == 7
    assert extract_int('{"k":-5}', "k") == -5


def test_ok_true_and_false():
    assert ok(True) == '{"ok":true}'
    assert ok(False) == '{"ok":false}'


def test_err_with_message():
    assert err("invalid body") == '{"error":"invalid body"}'
    assert err("Ollama unavailable") == '{"error":"Ollama unavailable"}'
