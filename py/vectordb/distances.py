"""Distance metrics and dispatcher.

Direct port of main.cpp:39-64.  All metrics operate on equal-length lists of
floats; the dispatcher returns one of three callables keyed by metric name.
Unknown names fall back to Euclidean (matches the C++ `getDistFn` default).
"""

from __future__ import annotations

import math
from typing import Callable, List

Vector = List[float]
DistFn = Callable[[Vector, Vector], float]


def euclidean(a: Vector, b: Vector) -> float:
    """L2 (Euclidean) distance — direct port of main.cpp:39-43."""
    s = 0.0
    for ai, bi in zip(a, b):
        d = ai - bi
        s += d * d
    return math.sqrt(s)


def cosine(a: Vector, b: Vector) -> float:
    """1 - cosine similarity — direct port of main.cpp:45-52.

    Returns 1.0 if either vector has zero norm (the C++ sentinel for "no
    similarity can be computed" without raising).  Clamps negative roundoff
    to 0 so equal-direction vectors sort before non-equal ones.
    """
    dot = 0.0
    na = 0.0
    nb = 0.0
    for ai, bi in zip(a, b):
        dot += ai * bi
        na += ai * ai
        nb += bi * bi
    if na < 1e-9 or nb < 1e-9:
        return 1.0
    d = 1.0 - dot / (math.sqrt(na) * math.sqrt(nb))
    if d < 0.0:
        return 0.0
    return d


def manhattan(a: Vector, b: Vector) -> float:
    """L1 (Manhattan) distance — direct port of main.cpp:54-58."""
    return sum(abs(ai - bi) for ai, bi in zip(a, b))


_DISPATCH = {
    "cosine": cosine,
    "manhattan": manhattan,
    # "euclidean" is the default fallback; no key needed.
}


def get_dist_fn(name: str) -> DistFn:
    """Return a distance function by name; default to Euclidean.

    Direct port of main.cpp:60-64.  Empty / unknown names both return
    `euclidean` to match the C++ switch's fallthrough behaviour.
    """
    return _DISPATCH.get(name, euclidean)
