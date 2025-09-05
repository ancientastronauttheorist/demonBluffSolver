"""Command-line interface stubs."""

from __future__ import annotations

import argparse
from pathlib import Path

from . import parser


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Demon Bluff Solver (skeleton)")
    ap.add_argument("puzzle", nargs="?", default="puzzle1.yaml", help="Path to puzzle YAML")
    args = ap.parse_args(argv)

    puz = parser.load_puzzle(Path(args.puzzle))
    print(f"Loaded puzzle with {puz.seats} seats and deck of {len(puz.deck)} roles.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
