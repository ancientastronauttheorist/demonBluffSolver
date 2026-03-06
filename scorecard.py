"""Scorecard — cumulative win/loss tracker for Demon Bluff solver.

Stores results in scorecard.json. Auto-called from game_over in game_loop.py.

Usage:
    python scorecard.py                  # Show scorecard
    python scorecard.py record win 8 asc6_v11 "clean solve"
    python scorecard.py record loss 0 asc6_v12 "trusted corrupted ability"
    python scorecard.py streak           # Show current streak
    python scorecard.py reset            # Clear all records
"""

import json
import os
import sys
from datetime import datetime

SCORECARD_FILE = os.path.join(os.path.dirname(__file__), "scorecard.json")


def load() -> dict:
    if os.path.exists(SCORECARD_FILE):
        with open(SCORECARD_FILE) as f:
            return json.load(f)
    return {"games": []}


def save(data: dict):
    with open(SCORECARD_FILE, "w") as f:
        json.dump(data, f, indent=2)


def record(result: str, hp: int, name: str = "", notes: str = ""):
    data = load()
    data["games"].append({
        "result": result,
        "hp": hp,
        "name": name,
        "notes": notes,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
    })
    save(data)


def show():
    data = load()
    games = data.get("games", [])
    if not games:
        print("No games recorded yet.")
        return

    wins = [g for g in games if g["result"] == "win"]
    losses = [g for g in games if g["result"] == "loss"]
    total = len(games)
    win_count = len(wins)
    loss_count = len(losses)
    win_rate = win_count / total * 100 if total else 0

    # Current streak
    streak_type, streak_len = current_streak(games)

    # HP stats (wins only)
    avg_hp = sum(g["hp"] for g in wins) / win_count if wins else 0
    perfect = sum(1 for g in wins if g["hp"] == 10)

    print(f"=== Demon Bluff Scorecard ===")
    print(f"Games:    {total}  ({win_count}W / {loss_count}L)")
    print(f"Win rate: {win_rate:.0f}%")
    print(f"Streak:   {streak_len} {streak_type}{'s' if streak_len != 1 else ''}")
    print(f"Avg HP (wins): {avg_hp:.1f}  |  Perfect (10 HP): {perfect}")
    print()

    # Last 10 games
    recent = games[-10:]
    print(f"Last {len(recent)} games:")
    for g in recent:
        marker = "W" if g["result"] == "win" else "L"
        hp_str = f"HP={g['hp']}" if g.get("hp") is not None else ""
        name_str = g.get("name", "")
        print(f"  [{marker}] {name_str:20s} {hp_str:6s}  {g.get('date', '')}")


def current_streak(games: list) -> tuple[str, int]:
    if not games:
        return ("win", 0)
    streak_type = games[-1]["result"]
    count = 0
    for g in reversed(games):
        if g["result"] == streak_type:
            count += 1
        else:
            break
    return (streak_type, count)


def reset():
    save({"games": []})
    print("Scorecard reset.")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "show"

    if cmd == "show" or cmd not in ("record", "streak", "reset"):
        show()
    elif cmd == "record":
        result = sys.argv[2] if len(sys.argv) > 2 else "unknown"
        hp = int(sys.argv[3]) if len(sys.argv) > 3 else 0
        name = sys.argv[4] if len(sys.argv) > 4 else ""
        notes = sys.argv[5] if len(sys.argv) > 5 else ""
        record(result, hp, name, notes)
        print(f"Recorded: {result.upper()} (HP={hp}) {name}")
        show()
    elif cmd == "streak":
        data = load()
        st, sl = current_streak(data.get("games", []))
        print(f"Current streak: {sl} {st}{'s' if sl != 1 else ''}")
    elif cmd == "reset":
        reset()
