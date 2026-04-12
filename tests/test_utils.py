"""Shared test utilities for Demon Bluff solver."""

import json
import os


CASES_DIR = os.path.join(os.path.dirname(__file__), "cases_v2")


def save_test_case(session_path: str, name: str, true_evil_positions: dict[int, str],
                   notes: str = ""):
    """Save current game session as a regression test case.

    true_evil_positions: {pos: role_name} e.g. {7: "Baa", 2: "Chancellor"}
    """
    with open(session_path) as f:
        session_data = json.load(f)

    case = dict(session_data)
    case["name"] = name
    case["true_evil_positions"] = {str(k): v for k, v in true_evil_positions.items()}
    case["notes"] = notes

    # Defensive: night-killed positions must NOT appear in executed.
    # executed = day executions only; night_kills = night deaths. Disjoint sets.
    nk_set = set(case.get("night_kills", []))
    case["executed"] = [p for p in case.get("executed", []) if p not in nk_set]

    # Nest deck fields into "deck" key if stored flat (session format)
    if "deck" not in case and "villagers" in case:
        case["deck"] = {
            "villagers": case.pop("villagers"),
            "outcasts": case.pop("outcasts"),
            "minions": case.pop("minions"),
            "demons": case.pop("demons"),
        }

    os.makedirs(CASES_DIR, exist_ok=True)
    out_path = os.path.join(CASES_DIR, f"{name}.json")
    with open(out_path, "w") as f:
        json.dump(case, f, indent=2)
    print(f"Test case saved: {out_path}")
    return out_path
