"""Information gain helpers."""

from __future__ import annotations

import math
from typing import Iterable, List


def entropy(counts: List[int]) -> float:
    total = sum(counts)
    if total == 0:
        return 0.0
    ent = 0.0
    for c in counts:
        if c:
            p = c / total
            ent -= p * math.log2(p)
    return ent


def information_gain(total_worlds: int, partitions: List[int]) -> float:
    """Entropy drop from partitioning a world set into the given sizes."""
    if total_worlds == 0:
        return 0.0
    prior = math.log2(total_worlds)
    post = 0.0
    for c in partitions:
        if c:
            p = c / total_worlds
            post += p * math.log2(c)
    return prior - post
