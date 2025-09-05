"""Constraint primitives and helpers (stub)."""

from __future__ import annotations

from typing import Callable, Iterable

from .world import World

Constraint = Callable[[World], bool]


def apply_constraints(worlds: Iterable[World], constraints: Iterable[Constraint]) -> list[World]:
    """Filter worlds that satisfy every constraint.

    This is a lightweight helper; the real engine will integrate constraint
    propagation directly during search.
    """
    result = []
    for w in worlds:
        if all(c(w) for c in constraints):
            result.append(w)
    return result
