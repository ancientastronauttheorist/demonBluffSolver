"""Regression test runner for Demon Bluff solver.

Each test case is a JSON file in tests/cases/ representing a completed game.
The runner replays the game state through the solver and verifies:
1. The true evil positions were in the surviving scenarios
2. Definite-evil positions are actually evil
3. Definite-good positions are actually good

Usage:
    python -m tests.test_regression          # Run all tests
    python -m tests.test_regression asc4_v4  # Run specific test
"""

import json
import os
import sys
import glob

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from solver import CardInfo, DeckComposition, GameState, SolverResult, solve


CASES_DIR = os.path.join(os.path.dirname(__file__), "cases")


def load_test_case(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def build_game_state(case: dict) -> GameState:
    """Build a GameState from a test case JSON."""
    deck = DeckComposition(
        villagers=case["deck"]["villagers"],
        outcasts=case["deck"]["outcasts"],
        minions=case["deck"]["minions"],
        demons=case["deck"]["demons"],
    )
    cards = [
        CardInfo(c["position"], c["apparent_role"], c.get("info_text", ""),
                 c.get("info_parsed", {}))
        for c in case.get("cards", [])
    ]
    # Convert executed_evil_roles keys from str (JSON) to int
    raw_eer = case.get("executed_evil_roles", {})
    eer = {int(k): v for k, v in raw_eer.items()}

    return GameState(
        n_cards=case["n_cards"],
        deck=deck,
        cards=cards,
        n_evil=case.get("n_evil", 0),
        executed=case.get("executed", []),
        confirmed_evil=case.get("confirmed_evil", []),
        confirmed_good=case.get("confirmed_good", []),
        pd_corruption_target=case.get("pd_corruption_target"),
        executed_evil_roles=eer,
        slayer_results=case.get("slayer_results", []),
        night_kills=case.get("night_kills", []),
        night_kill_evil_count=case.get("night_kill_evil_count", 0),
        hp=case.get("hp", 10),
        wrong_exec_cost=case.get("wrong_exec_cost", 2),
    )


def run_test(case: dict, verbose: bool = True) -> tuple[bool, list[str]]:
    """Run a single test case. Returns (passed, list_of_messages)."""
    name = case.get("name", "unnamed")
    true_evils = case.get("true_evil_positions", {})
    # true_evils is {pos_str: role_name} e.g. {"7": "Baa", "2": "Chancellor"}
    true_evil_set = {int(k) for k in true_evils.keys()}

    state = build_game_state(case)
    result = solve(state)

    messages = []
    passed = True

    # Check 1: Were there any surviving scenarios?
    if result.n_surviving == 0:
        messages.append(f"FAIL: 0 surviving scenarios (solver found no valid solution)")
        passed = False
    else:
        messages.append(f"OK: {result.n_surviving} surviving scenarios (from {result.n_scenarios})")

    # Check 2: Is the true solution among surviving scenarios?
    if result.n_surviving > 0:
        found_match = False
        for s in result.surviving_scenarios:
            scenario_evil_set = set(s.evil_positions.keys())
            if s.puppet_position:
                scenario_evil_set.add(s.puppet_position)
            # Check if this scenario's evil positions match true evils
            # (only check non-executed positions since executed are already resolved)
            non_executed_true = true_evil_set - set(state.executed)
            non_executed_scenario = scenario_evil_set - set(state.executed)
            if non_executed_true == non_executed_scenario:
                found_match = True
                break
        if found_match:
            messages.append(f"OK: True solution found in surviving scenarios")
        else:
            messages.append(f"FAIL: True solution NOT in surviving scenarios!")
            messages.append(f"  True evils: {true_evils}")
            messages.append(f"  Surviving evil sets: {[dict(s.evil_positions) for s in result.surviving_scenarios[:5]]}")
            passed = False

    # Check 3: Definite evil positions should actually be evil
    for pos in result.definite_evil:
        if pos in true_evil_set:
            messages.append(f"OK: #{pos} correctly identified as definite evil")
        else:
            messages.append(f"FAIL: #{pos} marked definite evil but was actually Good!")
            passed = False

    # Check 4: Definite good positions should actually be good
    for pos in result.definite_good:
        if pos not in true_evil_set:
            pass  # Correct, don't spam
        else:
            messages.append(f"FAIL: #{pos} marked definite good but was actually Evil!")
            passed = False

    return passed, messages


def run_all_tests(filter_name: str = None, verbose: bool = True) -> bool:
    """Run all test cases. Returns True if all passed."""
    if not os.path.isdir(CASES_DIR):
        print(f"No test cases directory: {CASES_DIR}")
        return True

    case_files = sorted(glob.glob(os.path.join(CASES_DIR, "*.json")))
    if not case_files:
        print("No test cases found.")
        return True

    all_passed = True
    for path in case_files:
        name = os.path.basename(path).replace(".json", "")
        if filter_name and filter_name not in name:
            continue

        case = load_test_case(path)
        passed, messages = run_test(case, verbose)

        status = "PASS" if passed else "FAIL"
        print(f"\n[{status}] {name}")
        if verbose or not passed:
            for msg in messages:
                print(f"  {msg}")

        if not passed:
            all_passed = False

    return all_passed


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


if __name__ == "__main__":
    filter_name = sys.argv[1] if len(sys.argv) > 1 else None
    all_passed = run_all_tests(filter_name)
    sys.exit(0 if all_passed else 1)
