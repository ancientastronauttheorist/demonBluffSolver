from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import FrozenSet, Mapping


class Alignment(str, Enum):
    """Basic team classification for a role."""
    GOOD = "good"
    EVIL = "evil"
    OUTCAST = "outcast"
    PUPPET = "puppet"
    UNKNOWN = "unknown"


class Status(str, Enum):
    """Mutable conditions that may affect a seat."""
    CORRUPTED = "corrupted"
    POISONED = "poisoned"
    DRUNK = "drunk"
    DISGUISED = "disguised"


@dataclass(frozen=True)
class Assignment:
    """A concrete role/alignment/status bundle for a seat."""
    role: str
    alignment: Alignment
    statuses: FrozenSet[Status] = field(default_factory=frozenset)
    revealed: bool = False


Seat = int
WorldMapping = Mapping[Seat, Assignment]
