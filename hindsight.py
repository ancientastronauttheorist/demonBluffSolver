"""Hindsight replay: simulate the solver playing each game from scratch.

For every test case, enters all card info with no executions, then lets
the solver pick targets one at a time. Shows a step-by-step table of
what would have happened.

Usage:
    python hindsight.py                  # All games
    python hindsight.py asc26_v6         # Single game
    python hindsight.py --losses-only    # Only show games the solver loses
"""

import json
import os
import sys
import glob
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(__file__))

from solver import GameState, CardInfo, DeckComposition
from rust_solver import rust_solve_to_objects
from knowledge_base import execution_cost_for

CASES_DIR_LEGACY = os.path.join(os.path.dirname(__file__), "tests", "cases")
CASES_DIR = os.path.join(os.path.dirname(__file__), "tests", "cases_v2")


@dataclass
class ExecStep:
    step_num: int
    position: int
    prob_evil: float
    was_evil: bool
    evil_role: str  # role name if evil, "good" if good, "Drunk" if Drunk
    scenarios_before: int
    scenarios_after: int
    definite_evil_before: list[int] = field(default_factory=list)
    hp_cost: int = 0
    note: str = ""


@dataclass
class HindsightResult:
    name: str
    n_cards: int
    n_evil: int
    true_evils: dict[int, str]  # pos -> role
    steps: list[ExecStep]
    won: bool
    hp_start: int
    hp_end: int
    wrong_exec_cost: int
    evils_found: int


def _is_evil_in_scenario(pos, scenario):
    if pos in scenario.evil_positions:
        return True
    if pos == scenario.puppet_position:
        return True
    return False


def _evil_probs(result, n_cards, executed):
    probs = {}
    for pos in range(1, n_cards + 1):
        if pos in executed:
            continue
        if result.n_surviving == 0:
            probs[pos] = 0.0
        else:
            count = sum(1 for s in result.surviving_scenarios
                        if _is_evil_in_scenario(pos, s))
            probs[pos] = count / result.n_surviving
    return probs


def _pick_target(result, probs, bombardier_positions, wretch_positions):
    """Pick the best execution target (mirrors strategy.py logic)."""
    # Definite evil first (Bombardiers/Wretch safe to execute if definite evil)
    for pos in sorted(result.definite_evil):
        if pos not in probs:
            continue  # already executed
        return pos

    # Never voluntarily execute Wretch — it's always good, abilities just
    # "see" it as evil. Executing Wretch = guaranteed wrong exec penalty.
    # Only exception: definite evil (evil disguised as Wretch, handled above).
    safe = {p: pr for p, pr in probs.items() if p not in wretch_positions}

    # Bombardier: risky but not suicidal. Allow execution when it's the
    # top candidate and clearly the best choice (prob > 0.5 and beats
    # non-Bombardier alternatives by a margin).
    non_bomb = {p: pr for p, pr in safe.items() if p not in bombardier_positions}
    if non_bomb:
        best_non_bomb = max(non_bomb, key=non_bomb.get)
        best_non_bomb_pr = non_bomb[best_non_bomb]

        # Check if a Bombardier clearly dominates
        bomb_candidates = {p: pr for p, pr in safe.items()
                          if p in bombardier_positions and pr > 0.5}
        if bomb_candidates:
            best_bomb = max(bomb_candidates, key=bomb_candidates.get)
            best_bomb_pr = bomb_candidates[best_bomb]
            # Pick Bombardier if it's significantly better
            if best_bomb_pr > best_non_bomb_pr + 0.1:
                return best_bomb

        return best_non_bomb

    # Only Bombardier candidates left (all non-Bombardier/Wretch exhausted)
    bomb_only = {p: pr for p, pr in safe.items() if p in bombardier_positions}
    if bomb_only:
        return max(bomb_only, key=bomb_only.get)

    # Absolute fallback: even Wretch
    if probs:
        return max(probs, key=probs.get)
    return None


def _reconstruct_starting_hp(case: dict) -> int:
    """Reconstruct the game's starting HP from saved state.

    Test cases store HP at save time (after executions). Reverse-engineer
    the original HP by adding back execution costs:
    - Wrong exec (good, non-Drunk): wrong_exec_cost
    - Drunk exec: 2
    - Lilis night kills: 2 each
    """
    hp = case.get("hp", 10)
    wrong_exec_cost = case.get("wrong_exec_cost", 2)
    true_evil_set = {int(k) for k in case.get("true_evil_positions", {}).keys()}
    confirmed_good = set(case.get("confirmed_good", []))
    executed = case.get("executed", [])
    night_kills = set(case.get("night_kills", []))
    exec_good_corrupted = case.get("executed_good_corrupted", {})
    exec_good_roles = {
        int(k): v for k, v in case.get("executed_good_roles", {}).items()
    }

    # Cards executed as wrong (good, not night-killed)
    # Use confirmed_good if available; otherwise use true_evil_set to detect
    # wrong execs (older test cases may not track confirmed_good).
    confirmed_evil = set(case.get("confirmed_evil", []))
    exec_evil_roles = {int(k) for k in case.get("executed_evil_roles", {}).keys()}
    slayer_killed = {sr["target_pos"] for sr in case.get("slayer_results", [])
                     if sr.get("killed")}

    for pos in executed:
        if pos in night_kills or pos in slayer_killed:
            continue  # no execution HP cost
        if pos in confirmed_evil or pos in exec_evil_roles or pos in true_evil_set:
            continue  # correct execution, no HP cost

        # This was a wrong execution
        card = next((c for c in case.get("cards", [])
                     if c["position"] == pos), None)
        apparent_role = card.get("apparent_role", "") if card else None
        execution_role = exec_good_roles.get(pos, apparent_role)
        recorded_corrupted = exec_good_corrupted.get(
            str(pos), exec_good_corrupted.get(pos, False)
        )
        # For non-Drunk this is the active status. Drunk is always persisted
        # clean, so false cannot distinguish ordinary statused Drunk from an
        # Alchemist-resistant generated one. Use the conservative 2-HP base;
        # never fabricate Knight's +4 without separate active-status evidence.
        hp += execution_cost_for(
            execution_role,
            apparent_role=apparent_role,
            was_corrupted=bool(recorded_corrupted),
            was_killable=True,
            default=wrong_exec_cost,
        )

    # Lilis night kills cost 2 HP each
    hp += len(night_kills) * 2

    return hp


def replay_hindsight(case: dict) -> HindsightResult:
    name = case.get("name", "unnamed")
    true_evils = {int(k): v for k, v in case.get("true_evil_positions", {}).items()}
    true_evil_set = set(true_evils.keys())
    n_cards = case["n_cards"]
    n_evil = case.get("n_evil", len(true_evils))
    wrong_exec_cost = case.get("wrong_exec_cost", 2)
    hp_start = _reconstruct_starting_hp(case)
    hp = hp_start

    deck_data = case.get("deck", {
        "villagers": case.get("villagers", []),
        "outcasts": case.get("outcasts", []),
        "minions": case.get("minions", []),
        "demons": case.get("demons", []),
    })

    # Pre-apply abilities (slayer, PD) that happen before execution phase
    pre_executed = []
    pre_confirmed_evil = []
    pre_confirmed_good = []
    pre_exec_evil_roles = {}
    pre_exec_good_corrupted = {}
    pre_exec_good_roles = {}

    case_confirmed_good = set(case.get("confirmed_good", []))
    case_evil_roles = {int(k): v for k, v in case.get("executed_evil_roles", {}).items()}
    case_good_roles = {int(k): v for k, v in case.get("executed_good_roles", {}).items()}

    for sr in case.get("slayer_results", []):
        if sr.get("killed"):
            tp = sr["target_pos"]
            pre_executed.append(tp)
            if tp in case_confirmed_good:
                pre_confirmed_good.append(tp)
                if tp in case_good_roles:
                    pre_exec_good_roles[tp] = case_good_roles[tp]
            else:
                pre_confirmed_evil.append(tp)
                evil_role = sr.get("evil_role") or case_evil_roles.get(tp)
                if evil_role:
                    pre_exec_evil_roles[tp] = evil_role

    # Night kills are also pre-executed
    night_kills = case.get("night_kills", [])
    night_kill_evil_count = case.get("night_kill_evil_count", 0)
    for nk in night_kills:
        if nk not in pre_executed:
            pre_executed.append(nk)

    # Build base state (all cards, pre-applied abilities)
    def make_state(executed, confirmed_evil, confirmed_good,
                   exec_evil_roles, exec_good_corrupted, exec_good_roles):
        return GameState(
            n_cards=n_cards,
            deck=DeckComposition.from_dict(deck_data),
            cards=[CardInfo.from_dict(c) for c in case.get("cards", [])],
            n_evil=n_evil,
            executed=list(executed),
            confirmed_evil=list(confirmed_evil),
            confirmed_good=list(confirmed_good),
            pd_corruption_target=case.get("pd_corruption_target"),
            executed_evil_roles=dict(exec_evil_roles),
            slayer_results=list(case.get("slayer_results", [])),
            pd_ability_results=list(case.get("pd_ability_results", [])),
            blocked_positions=list(case.get("blocked_positions", [])),
            night_kills=list(night_kills),
            night_kill_evil_count=night_kill_evil_count,
            hp=hp,
            wrong_exec_cost=wrong_exec_cost,
            board_villager_count=case.get("board_villager_count"),
            board_outcast_count=case.get("board_outcast_count"),
            board_minion_count=case.get("board_minion_count"),
            board_demon_count=case.get("board_demon_count"),
            board_count_provenance=case.get("board_count_provenance", "legacy_unknown"),
            reveal_order=list(case.get("reveal_order", [])),
            executed_good_corrupted=dict(exec_good_corrupted),
            executed_good_roles=dict(exec_good_roles),
        )

    executed = list(pre_executed)
    confirmed_evil = list(pre_confirmed_evil)
    confirmed_good = list(pre_confirmed_good)
    exec_evil_roles = dict(pre_exec_evil_roles)
    exec_good_corrupted = dict(pre_exec_good_corrupted)
    exec_good_roles = dict(pre_exec_good_roles)
    evils_found = len([p for p in confirmed_evil if p in true_evil_set])
    evils_found += night_kill_evil_count

    # Dangerous positions to avoid executing
    bombardier_positions = set()
    wretch_positions = set()
    for c in case.get("cards", []):
        if c.get("apparent_role") == "Bombardier":
            bombardier_positions.add(c["position"])
        if c.get("apparent_role") == "Wretch":
            wretch_positions.add(c["position"])

    steps = []
    step_num = 0
    evils_needed = n_evil - evils_found

    while evils_needed > 0 and hp > 0:
        state = make_state(executed, confirmed_evil, confirmed_good,
                           exec_evil_roles, exec_good_corrupted, exec_good_roles)
        try:
            result = rust_solve_to_objects(state)
        except Exception as e:
            steps.append(ExecStep(
                step_num=step_num, position=0, prob_evil=0,
                was_evil=False, evil_role="ERROR",
                scenarios_before=0, scenarios_after=0,
                note=f"Solver crashed: {e}",
            ))
            break

        if result.n_surviving == 0:
            steps.append(ExecStep(
                step_num=step_num, position=0, prob_evil=0,
                was_evil=False, evil_role="0 scenarios",
                scenarios_before=0, scenarios_after=0,
                note="0 scenarios — solver bug",
            ))
            break

        probs = _evil_probs(result, n_cards, set(executed))
        target = _pick_target(result, probs, bombardier_positions, wretch_positions)

        if target is None:
            steps.append(ExecStep(
                step_num=step_num, position=0, prob_evil=0,
                was_evil=False, evil_role="no target",
                scenarios_before=result.n_surviving, scenarios_after=0,
                note="No valid execution target",
            ))
            break

        scenarios_before = result.n_surviving
        def_evil_before = sorted(result.definite_evil)
        prob = probs.get(target, 0)

        # Execute
        was_evil = target in true_evil_set
        executed.append(target)

        if was_evil:
            confirmed_evil.append(target)
            evil_role = true_evils[target]
            exec_evil_roles[target] = evil_role
            hp_cost = 0
            evils_found += 1
            evils_needed -= 1
        else:
            confirmed_good.append(target)
            card_data = next((c for c in case.get("cards", [])
                              if c["position"] == target), None)
            apparent = card_data["apparent_role"] if card_data else "?"

            # Only assert corruption status if we KNOW it (card was executed
            # in the real game). For alternate-path executions, leave it unknown
            # so the solver doesn't falsely reject corrupted-card scenarios.
            real_corrupted = case.get("executed_good_corrupted") or {}
            if str(target) in real_corrupted or target in real_corrupted:
                was_corrupted = real_corrupted.get(str(target), real_corrupted.get(target, False))
                exec_good_corrupted[target] = was_corrupted
            if target in case_good_roles:
                exec_good_roles[target] = case_good_roles[target]

            execution_role = case_good_roles.get(target, apparent)
            recorded_corrupted = exec_good_corrupted.get(target, False)
            hp_cost = execution_cost_for(
                execution_role,
                apparent_role=apparent,
                # Drunk=false is observed-clean evidence, so this remains a
                # conservative base-cost fallback in historical replays.
                was_corrupted=bool(recorded_corrupted),
                was_killable=True,
                default=wrong_exec_cost,
            )
            evil_role = execution_role if target in case_good_roles else "good"
            hp -= hp_cost

        # Run solver again to get scenarios_after
        state_after = make_state(executed, confirmed_evil, confirmed_good,
                                 exec_evil_roles, exec_good_corrupted, exec_good_roles)
        try:
            result_after = rust_solve_to_objects(state_after)
            scenarios_after = result_after.n_surviving
        except Exception:
            scenarios_after = -1

        note = ""
        if target in result.definite_evil:
            note = "definite evil"
        elif prob >= 0.8:
            note = "high confidence"

        steps.append(ExecStep(
            step_num=step_num,
            position=target,
            prob_evil=prob,
            was_evil=was_evil,
            evil_role=evil_role if was_evil else evil_role,
            scenarios_before=scenarios_before,
            scenarios_after=scenarios_after,
            definite_evil_before=def_evil_before,
            hp_cost=hp_cost,
            note=note,
        ))
        step_num += 1

    won = evils_needed <= 0 and hp > 0

    return HindsightResult(
        name=name, n_cards=n_cards, n_evil=n_evil,
        true_evils=true_evils, steps=steps, won=won,
        hp_start=hp_start, hp_end=hp, wrong_exec_cost=wrong_exec_cost,
        evils_found=evils_found,
    )


def print_result(r: HindsightResult):
    outcome = "WIN" if r.won else "LOSS"
    print(f"\n{'='*72}")
    print(f"  {r.name}  |  {r.n_cards} cards, {r.n_evil} evil  |  "
          f"HP: {r.hp_start}->{r.hp_end}  |  {outcome}")
    true_str = ", ".join(f"#{p}={r.true_evils[p]}" for p in sorted(r.true_evils))
    print(f"  True evils: {true_str}")
    print(f"{'='*72}")

    if not r.steps:
        print("  (no execution steps — all evils found by abilities/night kills)")
        return

    # Table header
    print(f"  {'Step':>4}  {'Action':<22}  {'Prob':>5}  {'Result':<16}  "
          f"{'HP':>4}  {'Scenarios':<14}  {'Note'}")
    print(f"  {'----':>4}  {'------':<22}  {'-----':>5}  {'------':<16}  "
          f"{'----':>4}  {'---------':<14}  {'----'}")

    running_hp = r.hp_start
    for s in r.steps:
        if s.position == 0:
            action = s.evil_role  # error message
            prob_str = ""
            result_str = ""
            hp_str = ""
            scen_str = f"{s.scenarios_before}"
        else:
            action = f"Execute #{s.position}"
            prob_str = f"{s.prob_evil*100:.0f}%"
            if s.was_evil:
                result_str = f"Evil ({s.evil_role})"
            elif s.evil_role == "Drunk":
                result_str = "good (Drunk -2)"
                running_hp -= 2
            else:
                result_str = f"WRONG (-{r.wrong_exec_cost})"
                running_hp -= r.wrong_exec_cost
            if s.was_evil:
                hp_str = f"{running_hp}"
            else:
                hp_str = f"{running_hp}"
            scen_str = f"{s.scenarios_before}->{s.scenarios_after}"

        print(f"  {s.step_num:>4}  {action:<22}  {prob_str:>5}  {result_str:<16}  "
              f"{hp_str:>4}  {scen_str:<14}  {s.note}")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    flags = {a for a in sys.argv[1:] if a.startswith("-")}
    losses_only = "--losses-only" in flags or "-l" in flags
    filter_name = args[0] if args else None

    case_files = []
    for d in (CASES_DIR_LEGACY, CASES_DIR):
        if os.path.isdir(d):
            case_files.extend(glob.glob(os.path.join(d, "*.json")))
    case_files.sort(key=lambda p: os.path.basename(p))
    if not case_files:
        print("No test cases found.")
        return

    results = []
    for path in case_files:
        name = os.path.basename(path).replace(".json", "")
        if filter_name and filter_name not in name:
            continue

        with open(path) as f:
            case = json.load(f)

        r = replay_hindsight(case)
        results.append(r)

    # Print results
    for r in results:
        if losses_only and r.won:
            continue
        print_result(r)

    # Summary
    wins = sum(1 for r in results if r.won)
    losses = sum(1 for r in results if not r.won)
    total = len(results)

    wrong_execs = sum(
        sum(1 for s in r.steps if not s.was_evil and s.position > 0)
        for r in results
    )
    total_execs = sum(
        sum(1 for s in r.steps if s.position > 0)
        for r in results
    )
    correct_execs = total_execs - wrong_execs

    # Categorize losses
    zero_scenario_losses = []
    hp_losses = []
    for r in results:
        if r.won:
            continue
        hit_zero = any(s.note and "0 scenarios" in s.note for s in r.steps)
        if hit_zero:
            zero_scenario_losses.append(r.name)
        else:
            hp_losses.append(r.name)

    # Perfect games (all correct, no wrong execs)
    perfect = sum(1 for r in results
                  if r.won and all(s.was_evil for s in r.steps if s.position > 0))

    # First-pick accuracy (was the very first execution correct?)
    first_correct = sum(1 for r in results
                        if r.steps and r.steps[0].position > 0 and r.steps[0].was_evil)

    print(f"\n{'='*72}")
    print(f"  SUMMARY: {wins}/{total} wins ({wins/total*100:.0f}%)")
    print(f"  Executions: {correct_execs}/{total_execs} correct "
          f"({correct_execs/max(total_execs,1)*100:.0f}%), "
          f"{wrong_execs} wrong")
    print(f"  Perfect games (0 wrong execs): {perfect}/{total} "
          f"({perfect/max(total,1)*100:.0f}%)")
    print(f"  First pick correct: {first_correct}/{total} "
          f"({first_correct/max(total,1)*100:.0f}%)")
    if losses > 0:
        print(f"  Losses: {losses} total")
        if zero_scenario_losses:
            print(f"    0-scenario crash: {len(zero_scenario_losses)} "
                  f"({', '.join(zero_scenario_losses)})")
        if hp_losses:
            print(f"    HP exhausted: {len(hp_losses)} "
                  f"({', '.join(hp_losses)})")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()
