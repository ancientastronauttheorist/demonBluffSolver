"""Replay-style regression tests for Demon Bluff solver.

Unlike test_regression.py which only checks the final state, this test
replays each game step-by-step and verifies the solver at key decision points:

Phase 1 (reveals): Add cards one at a time — track scenario counts only.
  Truth may be temporarily absent (e.g. corruption source not yet revealed).
Phase 2 (abilities): Apply slayer/PD results — ASSERT truth in set.
  After all info is entered, the solver must have the true solution.
Phase 3 (executions): Step through executions — ASSERT truth in set +
  definite-evil/good correctness after each execution.

Usage:
    python -m tests.test_replay              # Run all tests
    python -m tests.test_replay asc26_v5     # Run specific test
    python -m tests.test_replay --verbose    # Show every step
    python -m tests.test_replay --strict     # Also assert truth during reveals
"""

import json
import os
import sys
import glob
from dataclasses import dataclass, field

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from solver import GameState, CardInfo, DeckComposition
from rust_solver import rust_solve_to_objects


CASES_DIR_LEGACY = os.path.join(os.path.dirname(__file__), "cases")
CASES_DIR = os.path.join(os.path.dirname(__file__), "cases_v2")


@dataclass
class ReplayStep:
    """A single step in the game replay."""
    step_type: str  # "reveal", "ability", "night_kill", "execute", "all_revealed"
    description: str
    n_surviving: int = 0
    truth_in_set: bool = True
    definite_evil_correct: bool = True
    definite_good_correct: bool = True
    bad_definite_evil: list[int] = field(default_factory=list)
    bad_definite_good: list[int] = field(default_factory=list)
    error: str = ""


@dataclass
class ReplayResult:
    """Full result of replaying a game."""
    name: str
    passed: bool
    steps: list[ReplayStep]
    failure_reason: str = ""


def _get_reveal_order(case: dict) -> list[int]:
    """Get reveal order from case, falling back to cards list order."""
    ro = case.get("reveal_order", [])
    if ro:
        return list(ro)
    return [c["position"] for c in case.get("cards", [])]


def _build_card_lookup(case: dict) -> dict[int, dict]:
    """Build position -> card_dict lookup."""
    return {c["position"]: c for c in case.get("cards", [])}


def _get_execution_steps(case: dict) -> list[dict]:
    """Build execution step info from the case."""
    executed = case.get("executed", [])
    confirmed_evil = set(case.get("confirmed_evil", []))
    executed_evil_roles = {int(k): v for k, v in case.get("executed_evil_roles", {}).items()}
    executed_good_corrupted = {int(k): v for k, v in case.get("executed_good_corrupted", {}).items()}
    night_kills_set = set(case.get("night_kills", []))

    steps = []
    for pos in executed:
        if pos in night_kills_set:
            continue
        was_evil = pos in confirmed_evil or pos in executed_evil_roles
        evil_role = executed_evil_roles.get(pos)
        was_corrupted = executed_good_corrupted.get(pos, False)
        steps.append({
            "position": pos,
            "was_evil": was_evil,
            "evil_role": evil_role,
            "was_corrupted": was_corrupted,
        })
    return steps


def _check_truth_in_scenarios(result, true_evil_set: set[int], executed: list[int]) -> bool:
    """Check if the true evil assignment is among surviving scenarios."""
    if result.n_surviving == 0:
        return False
    non_executed_true = true_evil_set - set(executed)
    for s in result.surviving_scenarios:
        scenario_evil_set = set(s.evil_positions.keys())
        if s.puppet_position:
            scenario_evil_set.add(s.puppet_position)
        non_executed_scenario = scenario_evil_set - set(executed)
        if non_executed_true == non_executed_scenario:
            return True
    return False


def _check_definite_evil(result, true_evil_set: set[int]) -> list[int]:
    """Return list of positions incorrectly marked as definite evil."""
    return [pos for pos in result.definite_evil if pos not in true_evil_set]


def _check_definite_good(result, true_evil_set: set[int]) -> list[int]:
    """Return list of positions incorrectly marked as definite good."""
    return [pos for pos in result.definite_good if pos in true_evil_set]


def _determine_night_kill_timing(case: dict) -> dict[int, list[int]]:
    """Determine when night kills happen relative to reveals.

    Lilis triggers night every 4 reveals. Returns {reveal_count: [killed_positions]}.
    """
    night_kills = case.get("night_kills", [])
    if not night_kills:
        return {}

    reveal_order = _get_reveal_order(case)
    n_nights = len(reveal_order) // 4
    timing = {}

    kill_idx = 0
    for night_num in range(1, n_nights + 1):
        after_reveal = night_num * 4
        if kill_idx < len(night_kills):
            timing.setdefault(after_reveal, []).append(night_kills[kill_idx])
            kill_idx += 1

    while kill_idx < len(night_kills):
        last_trigger = max(timing.keys()) if timing else 4
        timing.setdefault(last_trigger, []).append(night_kills[kill_idx])
        kill_idx += 1

    return timing


def replay_game(case: dict, verbose: bool = False, strict: bool = False) -> ReplayResult:
    """Replay a game step-by-step and check the solver at key decision points.

    Args:
        strict: If True, also assert truth-in-set during reveals (not just
                after all reveals). Useful for finding corruption-related pruning.
    """
    name = case.get("name", "unnamed")
    true_evils = case.get("true_evil_positions", {})
    true_evil_set = {int(k) for k in true_evils.keys()}

    deck_data = case.get("deck")
    if deck_data is None:
        deck_data = {
            "villagers": case.get("villagers", []),
            "outcasts": case.get("outcasts", []),
            "minions": case.get("minions", []),
            "demons": case.get("demons", []),
        }

    reveal_order = _get_reveal_order(case)
    card_lookup = _build_card_lookup(case)
    exec_steps = _get_execution_steps(case)
    night_kill_timing = _determine_night_kill_timing(case)

    steps = []
    passed = True
    failure_reason = ""

    # Incremental state
    current_cards = []
    current_executed = []
    current_confirmed_evil = []
    current_confirmed_good = []
    current_executed_evil_roles = {}
    current_slayer_results = []
    current_pd_results = []
    current_night_kills = []
    current_night_kill_evil_count = 0
    current_executed_good_corrupted = {}
    current_reveal_order = []

    def make_state() -> GameState:
        return GameState(
            n_cards=case["n_cards"],
            deck=DeckComposition.from_dict(deck_data),
            cards=[CardInfo.from_dict(c) for c in current_cards],
            n_evil=case.get("n_evil", 0),
            executed=list(current_executed),
            confirmed_evil=list(current_confirmed_evil),
            confirmed_good=list(current_confirmed_good),
            pd_corruption_target=case.get("pd_corruption_target"),
            executed_evil_roles=dict(current_executed_evil_roles),
            slayer_results=list(current_slayer_results),
            pd_ability_results=list(current_pd_results),
            blocked_positions=list(case.get("blocked_positions", [])),
            night_kills=list(current_night_kills),
            night_kill_evil_count=current_night_kill_evil_count,
            hp=case.get("hp", 10),
            wrong_exec_cost=case.get("wrong_exec_cost", 2),
            board_villager_count=case.get("board_villager_count"),
            board_outcast_count=case.get("board_outcast_count"),
            board_minion_count=case.get("board_minion_count"),
            board_demon_count=case.get("board_demon_count"),
            reveal_order=list(current_reveal_order),
            executed_good_corrupted=dict(current_executed_good_corrupted),
        )

    def run_solver_step(step_type: str, description: str, assert_truth: bool = False) -> ReplayStep:
        """Run solver and build a ReplayStep. Only fails the test if assert_truth=True."""
        nonlocal passed, failure_reason
        state = make_state()
        try:
            result = rust_solve_to_objects(state)
            if result is None:
                raise RuntimeError("Rust solver binary not found — run `cargo build --release`")
        except Exception as e:
            step = ReplayStep(step_type=step_type, description=description,
                              n_surviving=0, truth_in_set=False, error=str(e))
            if passed:
                passed = False
                failure_reason = f"Solver crashed at: {description} ({e})"
            return step

        truth_ok = _check_truth_in_scenarios(result, true_evil_set, current_executed)
        bad_evil = _check_definite_evil(result, true_evil_set)
        bad_good = _check_definite_good(result, true_evil_set)

        step = ReplayStep(
            step_type=step_type,
            description=description,
            n_surviving=result.n_surviving,
            truth_in_set=truth_ok,
            definite_evil_correct=len(bad_evil) == 0,
            definite_good_correct=len(bad_good) == 0,
            bad_definite_evil=bad_evil,
            bad_definite_good=bad_good,
        )

        if assert_truth and passed:
            if not truth_ok:
                passed = False
                failure_reason = f"True solution eliminated at: {description} ({result.n_surviving} scenarios)"
            elif bad_evil:
                passed = False
                failure_reason = f"Wrong definite evil {bad_evil} at: {description}"
            elif bad_good:
                passed = False
                failure_reason = f"Wrong definite good {bad_good} at: {description}"

        return step

    # === Phase 1: Reveal cards one by one ===
    # Only track scenario counts; truth may be temporarily absent due to
    # corruption sources not yet revealed. Use --strict to also assert here.
    reveals_done = 0
    for pos in reveal_order:
        card_data = card_lookup.get(pos)
        if card_data is None:
            continue

        current_cards.append(card_data)
        current_reveal_order.append(pos)
        reveals_done += 1

        desc = f"reveal #{pos} ({card_data['apparent_role']})"
        step = run_solver_step("reveal", desc, assert_truth=strict)
        steps.append(step)

        if verbose:
            status = "OK" if step.truth_in_set else ("FAIL" if strict else "---")
            print(f"  [{status}] {desc}: {step.n_surviving} scenarios")
            if step.error:
                print(f"        ERROR: {step.error}")

        # Night kill after every 4 reveals (Lilis games)
        if reveals_done in night_kill_timing:
            for nk_pos in night_kill_timing[reveals_done]:
                current_night_kills.append(nk_pos)
                if nk_pos in true_evil_set:
                    current_night_kill_evil_count += 1
                if nk_pos not in current_executed:
                    current_executed.append(nk_pos)
                desc = f"night_kill #{nk_pos} (after {reveals_done} reveals)"
                step = run_solver_step("night_kill", desc, assert_truth=strict)
                steps.append(step)
                if verbose:
                    status = "OK" if step.truth_in_set else ("FAIL" if strict else "---")
                    print(f"  [{status}] {desc}: {step.n_surviving} scenarios")

        if not passed:
            return ReplayResult(name=name, passed=False, steps=steps, failure_reason=failure_reason)

    # Add any card data not in reveal_order (e.g. night-killed cards with info
    # from abilities like Medium). These exist in the JSON but weren't player-revealed.
    revealed_positions = set(current_reveal_order)
    for card_data in case.get("cards", []):
        if card_data["position"] not in revealed_positions:
            current_cards.append(card_data)

    # === Phase 2: Apply ability results, then check ===
    # Preload execution results from test case (kills reveal roles in-game)
    case_evil_roles = {int(k): v for k, v in case.get("executed_evil_roles", {}).items()}
    case_confirmed_good = set(case.get("confirmed_good", []))

    ability_descs = []
    for sr in case.get("slayer_results", []):
        current_slayer_results.append(sr)
        if sr.get("killed"):
            tp = sr["target_pos"]
            if tp not in current_executed:
                current_executed.append(tp)
                # Slayer kills Wretch → confirmed good (Wretch registers as evil
                # for abilities but is actually good). Check case's final data.
                if tp in case_confirmed_good:
                    current_confirmed_good.append(tp)
                else:
                    current_confirmed_evil.append(tp)
                # Use evil_role from slayer result, or fall back to case data
                evil_role = sr.get("evil_role") or case_evil_roles.get(tp)
                if evil_role:
                    current_executed_evil_roles[tp] = evil_role
        ability_descs.append(f"slayer #{sr['slayer_pos']}->#{sr['target_pos']} ({'kill' if sr.get('killed') else 'fail'})")

    for pd in case.get("pd_ability_results", []):
        current_pd_results.append(pd)
        ability_descs.append(f"pd #{pd['pd_pos']}->#{pd['target']} ({'corrupted' if pd.get('is_corrupted') else 'clean'})")

    # After all reveals + abilities: this is the first real decision point.
    # The solver MUST have the true solution here.
    if ability_descs:
        desc = "abilities: " + ", ".join(ability_descs)
    else:
        desc = "all info entered (no abilities)"

    step = run_solver_step("all_revealed", desc, assert_truth=True)
    steps.append(step)
    if verbose:
        status = "OK" if step.truth_in_set else "FAIL"
        print(f"  [{status}] {desc}: {step.n_surviving} scenarios")
        if step.error:
            print(f"        ERROR: {step.error}")

    if not passed:
        return ReplayResult(name=name, passed=False, steps=steps, failure_reason=failure_reason)

    # === Phase 3: Execute one at a time ===
    for exec_info in exec_steps:
        pos = exec_info["position"]
        if pos in current_executed:
            continue

        current_executed.append(pos)
        if exec_info["was_evil"]:
            current_confirmed_evil.append(pos)
            if exec_info["evil_role"]:
                current_executed_evil_roles[pos] = exec_info["evil_role"]
        else:
            current_confirmed_good.append(pos)
            if exec_info["was_corrupted"]:
                current_executed_good_corrupted[pos] = True

        evil_desc = f"evil {exec_info['evil_role'] or '?'}" if exec_info["was_evil"] else "good"
        desc = f"execute #{pos} ({evil_desc})"
        step = run_solver_step("execute", desc, assert_truth=True)
        steps.append(step)

        if verbose:
            status = "OK" if step.truth_in_set else "FAIL"
            print(f"  [{status}] {desc}: {step.n_surviving} scenarios")

        if not passed:
            return ReplayResult(name=name, passed=False, steps=steps, failure_reason=failure_reason)

    return ReplayResult(name=name, passed=passed, steps=steps, failure_reason=failure_reason)


def run_all_replays(filter_name: str = None, verbose: bool = False,
                    strict: bool = False, v2_only: bool = False) -> bool:
    """Run all replay tests. Returns True if all passed."""
    case_files = []
    dirs = [CASES_DIR] if v2_only else [CASES_DIR_LEGACY, CASES_DIR]
    for d in dirs:
        if os.path.isdir(d):
            case_files.extend(glob.glob(os.path.join(d, "*.json")))
    case_files.sort(key=lambda p: os.path.basename(p))
    if not case_files:
        print("No test cases found.")
        return True

    all_passed = True
    n_pass = 0
    n_fail = 0
    failures = []

    for path in case_files:
        name = os.path.basename(path).replace(".json", "")
        if filter_name and filter_name not in name:
            continue

        with open(path) as f:
            case = json.load(f)

        result = replay_game(case, verbose=verbose, strict=strict)

        if result.passed:
            n_pass += 1
            if verbose:
                print(f"[PASS] {name} ({len(result.steps)} steps)")
            else:
                print(f"[PASS] {name}")
        else:
            n_fail += 1
            all_passed = False
            failures.append(result)
            total_steps = len(result.steps)
            fail_idx = next((i for i, s in enumerate(result.steps)
                             if s.error or (not s.truth_in_set and s.step_type in ("all_revealed", "execute"))
                             or s.bad_definite_evil or s.bad_definite_good), -1)
            print(f"[FAIL] {name} (step {fail_idx + 1}/{total_steps}): {result.failure_reason}")

    print(f"\n{'='*60}")
    print(f"Results: {n_pass} passed, {n_fail} failed, {n_pass + n_fail} total")

    if failures:
        print(f"\nFailures:")
        for r in failures:
            print(f"  {r.name}: {r.failure_reason}")

    return all_passed


if __name__ == "__main__":
    args = sys.argv[1:]
    verbose = "--verbose" in args or "-v" in args
    strict = "--strict" in args
    v2_only = "--v2-only" in args or "--v2" in args
    args = [a for a in args if a not in ("--verbose", "-v", "--strict", "--v2-only", "--v2")]
    filter_name = args[0] if args else None

    all_passed = run_all_replays(filter_name, verbose=verbose, strict=strict, v2_only=v2_only)
    sys.exit(0 if all_passed else 1)
