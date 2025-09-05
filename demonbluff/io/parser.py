from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Any

import yaml


@dataclass
class Puzzle:
    seats: int
    deck: List[str]
    evils_in_play: int | None = None
    flipped: Dict[int, str] = field(default_factory=dict)
    claims: List[Dict[str, Any]] = field(default_factory=list)
    executions: List[int] = field(default_factory=list)
    observations: List[Dict[str, Any]] = field(default_factory=list)
    options: Dict[str, Any] = field(default_factory=dict)


def load_puzzle(path: str | Path) -> Puzzle:
    """Load a YAML puzzle description into a Puzzle object."""
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    seats = int(data["seats"])
    deck = list(data.get("deck", []))
    flipped = {int(k): v for k, v in data.get("flipped", {}).items()}
    claims = list(data.get("claims", [])) or list(data.get("info_log", []))
    executions = list(data.get("executions", []))
    observations = list(data.get("observations", []))
    options = data.get("options", {})
    evils_in_play = options.get("evils_in_play", data.get("evils_in_play"))

    return Puzzle(
        seats=seats,
        deck=deck,
        evils_in_play=evils_in_play,
        flipped=flipped,
        claims=claims,
        executions=executions,
        observations=observations,
        options=options,
    )
