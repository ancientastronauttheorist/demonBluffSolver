from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .model import Assignment, Seat


@dataclass(frozen=True)
class World:
    """Immutable mapping from seat to assignment."""
    assignments: Mapping[Seat, Assignment]

    def __hash__(self) -> int:  # pragma: no cover - trivial
        # sort by seat to get stable hash
        items = tuple(sorted(self.assignments.items()))
        return hash(items)

    def seat(self, seat: Seat) -> Assignment:
        return self.assignments[seat]
