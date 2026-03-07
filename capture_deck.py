"""Capture a robust CURRENT DECK screenshot for manual verification."""

import sys

import game_utils


def main() -> int:
    name = sys.argv[1] if len(sys.argv) > 1 else "deck_view"
    path = game_utils.hold_tab_screenshot(name)
    print(f"Saved full deck screenshot to {path}")
    print(f"Saved enhanced crop to {path.rsplit('.', 1)[0]}_crop.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
