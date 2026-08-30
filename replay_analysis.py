#!/usr/bin/env python3
"""Generate scenario-narrowing analysis tables for v2 test cases.

Usage:
    python replay_analysis.py --all           # all v2 cases (overwrites file)
    python replay_analysis.py asc26_v8        # single case (appends to file)
"""

import json
import sys
import os
import io
import contextlib
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from solver import GameState, CardInfo, DeckComposition, slayer_revealed_role
from knowledge_base import Alignment, get_card
from rust_solver import rust_solve_to_objects

OUTPUT_FILE = "scenario_analysis.txt"
CASES_DIR = "tests/cases_v2"


def quiet_solve(state):
    with contextlib.redirect_stdout(io.StringIO()):
        return rust_solve_to_objects(state)


def replay_case(case):
    """Replay a test case and return (header, steps).

    Each step: (action_description, n_surviving, definite_evil, definite_good).
    """
    n_cards = case["n_cards"]
    n_evil = case["n_evil"]
    name = case.get("name", "unknown")
    deck_data = case.get("deck", {})
    card_list = case.get("cards", [])
    reveal_order = case.get("reveal_order", [c["position"] for c in card_list])
    true_evil = case.get("true_evil_positions", {})
    exec_evil_roles = case.get("executed_evil_roles", {})
    confirmed_good = set(case.get("confirmed_good", []))

    # Header
    evil_desc = " + ".join(
        role for _, role in sorted(true_evil.items(), key=lambda x: int(x[0]))
    )
    header_parts = [f"{name} \u2014 {n_cards} cards, {n_evil} evil ({evil_desc})"]
    if case.get("blocked_positions"):
        bp = ", ".join(f"#{p}" for p in case["blocked_positions"])
        header_parts.append(f"Witch-blocked: {bp}")
    if case.get("night_kills"):
        header_parts.append(f"{len(case['night_kills'])} night kill(s)")
    header = ", ".join(header_parts)

    # Ability result lookups
    slayer_by_pos = {sr["slayer_pos"]: sr for sr in case.get("slayer_results", [])}
    pd_by_pos = {pr["pd_pos"]: pr for pr in case.get("pd_ability_results", [])}
    slayer_killed = {
        sr["target_pos"]
        for sr in case.get("slayer_results", [])
        if sr.get("killed", False)
    }
    card_lookup = {c["position"]: c for c in card_list}

    # Incremental state
    cur_executed = []
    cur_confirmed_evil = []
    cur_confirmed_good = []
    cur_exec_evil_roles = {}
    cur_exec_good_corrupted = {}
    cur_exec_good_roles = {}
    cur_slayer_results = []
    cur_pd_results = []

    def make_state():
        return GameState(
            n_cards=n_cards,
            deck=DeckComposition.from_dict(deck_data),
            cards=[CardInfo.from_dict(c) for c in card_list],
            n_evil=n_evil,
            executed=list(cur_executed),
            confirmed_evil=list(cur_confirmed_evil),
            confirmed_good=list(cur_confirmed_good),
            pd_corruption_target=case.get("pd_corruption_target"),
            executed_evil_roles={int(k): v for k, v in cur_exec_evil_roles.items()},
            slayer_results=list(cur_slayer_results),
            pd_ability_results=list(cur_pd_results),
            blocked_positions=list(case.get("blocked_positions", [])),
            night_kills=list(case.get("night_kills", [])),
            night_kill_evil_count=case.get("night_kill_evil_count", 0),
            hp=case.get("hp", 10),
            wrong_exec_cost=case.get("wrong_exec_cost", 5),
            board_villager_count=case.get("board_villager_count"),
            board_outcast_count=case.get("board_outcast_count"),
            board_count_provenance=case.get("board_count_provenance", "legacy_unknown"),
            rambler_rule_version=case.get("rambler_rule_version"),
            rambler_shut_up_observations=[
                dict(observation)
                for observation in case.get("rambler_shut_up_observations", [])
            ],
            baker_rule_version=case.get("baker_rule_version"),
            reveal_order=list(reveal_order),
            executed_good_corrupted=dict(cur_exec_good_corrupted),
            executed_good_roles=dict(cur_exec_good_roles),
        )

    def solve_step():
        r = quiet_solve(make_state())
        return r.n_surviving, list(r.definite_evil), list(r.definite_good)

    steps = []

    # Step 1: All cards revealed
    ns, de, dg = solve_step()
    nk = case.get("night_kills", [])
    label = "All cards revealed"
    if nk:
        label += f" + {len(nk)} night kill(s)"
    steps.append((label, ns, de, dg))

    # Abilities in used_abilities order
    for pos in case.get("used_abilities", []):
        if pos in pd_by_pos:
            pr = pd_by_pos[pos]
            cur_pd_results.append(pr)
            if pr["is_corrupted"]:
                desc = f"PD #{pos} checks #{pr['target']}: corrupted, evil=#{pr['evil_revealed']}"
            else:
                desc = f"PD #{pos} checks #{pr['target']}: clean"
            ns, de, dg = solve_step()
            steps.append((desc, ns, de, dg))

        elif pos in slayer_by_pos:
            sr = slayer_by_pos[pos]
            cur_slayer_results.append(sr)
            target = sr["target_pos"]
            killed = sr.get("killed", False)

            if killed:
                cur_executed.append(target)
                revealed_role = slayer_revealed_role(sr)
                role_def = get_card(revealed_role) if revealed_role else None
                recorded_roles = case.get("executed_good_roles", {})
                recorded_good_role = recorded_roles.get(
                    str(target), recorded_roles.get(target)
                )
                target_was_good = (
                    role_def is not None and role_def.alignment == Alignment.GOOD
                ) or (
                    target in confirmed_good
                    and str(target) not in exec_evil_roles
                    and target not in exec_evil_roles
                )
                if not target_was_good:
                    role = (
                        revealed_role
                        or exec_evil_roles.get(str(target), exec_evil_roles.get(target))
                        or "?"
                    )
                    cur_confirmed_evil.append(target)
                    cur_exec_evil_roles[str(target)] = role
                    desc = f"Slayer #{pos} kills #{target} ({role})"
                else:
                    cur_confirmed_good.append(target)
                    recorded_corruption = case.get("executed_good_corrupted", {})
                    if str(target) in recorded_corruption or target in recorded_corruption:
                        cur_exec_good_corrupted[target] = recorded_corruption.get(
                            str(target), recorded_corruption.get(target)
                        )
                    role = revealed_role or recorded_good_role
                    if role:
                        cur_exec_good_roles[target] = role
                    shown_role = role or card_lookup.get(target, {}).get("apparent_role", "?")
                    desc = f"Slayer #{pos} kills #{target} ({shown_role})"
            else:
                desc = f"Slayer #{pos} targets #{sr['target_pos']}: survives"

            ns, de, dg = solve_step()
            steps.append((desc, ns, de, dg))
        # else: informational ability (Judge, Druid, etc.) — already in card data

    # Executions (non-slayer)
    evil_found = len(cur_confirmed_evil)
    for exec_pos in case.get("executed", []):
        if exec_pos in slayer_killed or exec_pos in cur_executed:
            continue

        cur_executed.append(exec_pos)

        if str(exec_pos) in exec_evil_roles:
            role = exec_evil_roles[str(exec_pos)]
            cur_confirmed_evil.append(exec_pos)
            cur_exec_evil_roles[str(exec_pos)] = role
            evil_found += 1
            suffix = " \u2014 WIN" if evil_found >= n_evil else ""
            desc = f"Execute #{exec_pos}: evil ({role}){suffix}"
        else:
            cur_confirmed_good.append(exec_pos)
            role = card_lookup.get(exec_pos, {}).get("apparent_role", "?")
            corr = case.get("executed_good_corrupted", {})
            if str(exec_pos) in corr:
                desc = f"Execute #{exec_pos}: good corrupted ({role})"
            else:
                desc = f"Execute #{exec_pos}: good ({role})"
            if str(exec_pos) in corr or exec_pos in corr:
                cur_exec_good_corrupted[exec_pos] = corr.get(
                    str(exec_pos), corr.get(exec_pos)
                )
            recorded_roles = case.get("executed_good_roles", {})
            observed_role = recorded_roles.get(str(exec_pos), recorded_roles.get(exec_pos))
            if observed_role:
                cur_exec_good_roles[exec_pos] = observed_role

        ns, de, dg = solve_step()
        steps.append((desc, ns, de, dg))

    return header, steps


def format_table(header, steps):
    """Format as Unicode box-drawing table."""
    w_step = 4
    w_action = max((len(s[0]) for s in steps), default=6)
    w_action = max(w_action, 6)
    w_scen = 9

    def fmt_list(lst, n_surv):
        if n_surv == 0:
            return "\u2014"
        return str(lst) if lst else "[]"

    evil_strs = [fmt_list(s[2], s[1]) for s in steps]
    good_strs = [fmt_list(s[3], s[1]) for s in steps]

    w_evil = max(max((len(s) for s in evil_strs), default=2), 13)
    w_good = max(max((len(s) for s in good_strs), default=2), 13)

    def hline(left, mid, right):
        return (
            f"{left}\u2500{'─' * (w_step + 1)}{mid}\u2500{'─' * (w_action + 1)}"
            f"{mid}\u2500{'─' * (w_scen + 1)}{mid}\u2500{'─' * (w_evil + 1)}"
            f"{mid}\u2500{'─' * (w_good + 1)}{right}"
        )

    def row(c1, c2, c3, c4, c5):
        return (
            f"\u2502 {c1:<{w_step}} \u2502 {c2:<{w_action}} "
            f"\u2502 {c3:<{w_scen}} \u2502 {c4:<{w_evil}} "
            f"\u2502 {c5:<{w_good}} \u2502"
        )

    lines = [header, ""]
    lines.append(hline("\u250c", "\u252c", "\u2510"))
    lines.append(row("Step", "Action", "Scenarios", "Definite Evil", "Definite Good"))
    lines.append(hline("\u251c", "\u253c", "\u2524"))

    for i, (action, ns, de, dg) in enumerate(steps):
        lines.append(row(str(i + 1), action, str(ns), evil_strs[i], good_strs[i]))
        if i < len(steps) - 1:
            lines.append(hline("\u251c", "\u253c", "\u2524"))

    lines.append(hline("\u2514", "\u2534", "\u2518"))
    return "\n".join(lines)


def process_case(case_path):
    with open(case_path) as f:
        case = json.load(f)
    header, steps = replay_case(case)
    return format_table(header, steps)


def main():
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <case_name|--all>")
        print(f"  python {sys.argv[0]} asc26_v8       # single case (appends)")
        print(f"  python {sys.argv[0]} --all           # all v2 cases (overwrites)")
        sys.exit(1)

    cases_dir = Path(CASES_DIR)

    if sys.argv[1] == "--all":
        case_files = sorted(cases_dir.glob("*.json"))
    else:
        name = sys.argv[1]
        if not name.endswith(".json"):
            name += ".json"
        p = cases_dir / name
        if not p.exists():
            print(f"Case not found: {p}")
            sys.exit(1)
        case_files = [p]

    tables = []
    for cf in case_files:
        print(f"Processing {cf.name}...", end=" ", flush=True)
        try:
            table = process_case(cf)
            tables.append(table)
            print("OK")
        except Exception as e:
            print(f"ERROR: {e}")
            tables.append(f"{cf.stem} \u2014 ERROR: {e}")

    output = "\n\n".join(tables) + "\n"

    if sys.argv[1] == "--all":
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            f.write(output)
        print(f"\nWrote {len(tables)} tables to {OUTPUT_FILE}")
    else:
        mode = "a" if os.path.exists(OUTPUT_FILE) else "w"
        with open(OUTPUT_FILE, mode, encoding="utf-8") as f:
            if mode == "a":
                f.write("\n\n")
            f.write(output)
        print(f"\nAppended to {OUTPUT_FILE}")

    # Print last table
    if tables:
        print()
        try:
            print(tables[-1])
        except UnicodeEncodeError:
            print(tables[-1].encode("utf-8", errors="replace").decode("utf-8"))


if __name__ == "__main__":
    main()
