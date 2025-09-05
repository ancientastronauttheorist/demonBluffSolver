"""Observation DSL (stub)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Callable

from .world import World


@dataclass(frozen=True)
class Observation:
    type: str
    data: Dict[str, Any]


Filter = Callable[[World], bool]


def apply_observations(worlds: Iterable[World], observations: Iterable[Observation], resolver: Callable[[Observation], Filter]) -> list[World]:
    """Apply observation filters using a resolver provided by role adapters."""
    result = list(worlds)
    for obs in observations:
        flt = resolver(obs)
        result = [w for w in result if flt(w)]
    return result
