"""Utility functions for the Alchemist role in Demon Bluff.

The Alchemist cures corruption on nearby villagers.  This module provides
helpers to calculate how many villagers are cured given a seating
arrangement and the position of the Alchemist.

The script expects a YAML file describing the seating arrangement.  Each
seat is a mapping with ``role`` and ``corrupted`` keys.  Example::

    seating:
      - role: "knitter"
        corrupted: true
      - role: "alchemist"
        corrupted: false
      - role: "scout"
        corrupted: true

Running ``python alchemist.py example.yaml`` will report how many
corrupted villagers were cured by the Alchemist in the example seating.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import yaml

# ---------------------------------------------------------------------------
# Load villager roles from characters.yaml so the script stays in sync with
# the official role definitions.
# ---------------------------------------------------------------------------

_CHARACTERS_FILE = Path(__file__).with_name("characters.yaml")

try:
    _CHAR_DATA = yaml.safe_load(_CHARACTERS_FILE.read_text(encoding="utf-8"))
except FileNotFoundError as exc:  # pragma: no cover - repository guarantees file
    raise SystemExit(f"Missing characters file: {_CHARACTERS_FILE}") from exc

_VILLAGER_ROLES = {
    role["name"]
    for role in _CHAR_DATA.get("roles", [])
    if "villager" in role.get("attributes", [])
}


@dataclass
class Seat:
    """Representation of a seat at the table."""

    role: str
    corrupted: bool = False


# ---------------------------------------------------------------------------
# Core calculation
# ---------------------------------------------------------------------------

def cured_villagers(seating: List[Seat], alchemist_index: int) -> Tuple[int, List[int]]:
    """Return the number of villagers cured and their indices.

    Parameters
    ----------
    seating:
        Full table in clockwise order.
    alchemist_index:
        Index of the Alchemist within ``seating`` (0-based).

    Returns
    -------
    count, indices:
        Number of villagers cured and a list of their indices.
    """

    n = len(seating)
    cured_indices: List[int] = []
    for offset in (-2, -1, 1, 2):
        idx = (alchemist_index + offset) % n
        seat = seating[idx]
        if seat.role in _VILLAGER_ROLES and seat.corrupted:
            cured_indices.append(idx)
    return len(cured_indices), cured_indices


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------

def _load_seating(path: Path) -> List[Seat]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    seats_data: Iterable[dict] = data.get("seating", [])
    seating = [Seat(role=s["role"].lower(), corrupted=bool(s.get("corrupted", False))) for s in seats_data]
    return seating


def main(argv: List[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Calculate cures by the Alchemist.")
    parser.add_argument("seating_file", type=Path, help="YAML file describing seating")
    parser.add_argument(
        "--alchemist",
        type=int,
        default=None,
        help="0-based index of the Alchemist seat.  If omitted, the first seat with role 'alchemist' is used.",
    )

    args = parser.parse_args(argv)
    seating = _load_seating(args.seating_file)

    try:
        idx = args.alchemist if args.alchemist is not None else next(
            i for i, seat in enumerate(seating) if seat.role == "alchemist"
        )
    except StopIteration as exc:  # pragma: no cover - handled by SystemExit below
        raise SystemExit("No Alchemist seat found in seating file") from exc

    count, indices = cured_villagers(seating, idx)
    print(f"Alchemist at seat {idx} cured {count} villager(s): {indices}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
