"""Post-game decision analysis for Demon Bluff Solver.

Replays completed games with ground truth, annotates each decision:
- Re-runs solver + strategy at each step
- Compares recommendation against action actually taken
- Classifies wrong decisions by category
- Detects cross-game failure patterns

Usage:
    python decision_analysis.py analyze <case_name>    # Analyze one game
    python decision_analysis.py analyze_all            # Analyze all v2 cases
    python decision_analysis.py failure_report          # Top failure patterns
"""

import json
import os
import sys
import glob
from dataclasses import dataclass, field, asdict
from collections import Counter

sys.path.insert(0, os.path.dirname(__file__))

from solver import (
    GameState,
    CardInfo,
    DeckComposition,
    SolverResult,
    slayer_revealed_role,
)
from knowledge_base import Alignment, get_card
from rust_solver import rust_solve_to_objects
from strategy import (
    _compute_confidence,
    _is_terminal_loss_role,
    _public_terminal_loss_position,
    evil_probabilities,
    recommend_action,
)


CASES_DIR = os.path.join(os.path.dirname(__file__), "tests", "cases_v2")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "decision_analysis")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# Data Structures
# ============================================================

@dataclass
class DecisionPoint:
    """A single decision point in a game."""
    step: int
    phase: str           # "reveal", "execute", "ability"
    description: str
    n_surviving: int
    recommended_action: str    # "execute #3", "reveal #5", etc.
    recommended_position: int | None
    recommended_confidence: float
    actual_action: str         # what was actually done
    actual_position: int | None
    correct: bool              # did recommendation match actual?
    category: str = ""         # classification if wrong
    evil_probs: dict = field(default_factory=dict)
    definite_evil: list = field(default_factory=list)
    definite_good: list = field(default_factory=list)
    roles_in_deck: list = field(default_factory=list)


@dataclass
class GameAnalysis:
    """Full analysis of one game."""
    name: str
    n_cards: int
    n_evil: int
    result: str  # "win" or "loss"
    decisions: list[DecisionPoint]
    summary: dict = field(default_factory=dict)


# ============================================================
# Core Analysis
# ============================================================

def analyze_game(case: dict) -> GameAnalysis:
    """Replay a game step-by-step, comparing solver recommendations against actual play."""
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

    reveal_order = case.get("reveal_order", [])
    if not reveal_order:
        reveal_order = [c["position"] for c in case.get("cards", [])]

    card_lookup = {c["position"]: c for c in case.get("cards", [])}
    night_kills = case.get("night_kills", [])
    executed = case.get("executed", [])
    confirmed_evil = set(case.get("confirmed_evil", []))
    confirmed_good = set(case.get("confirmed_good", []))
    executed_evil_roles = {int(k): v for k, v in case.get("executed_evil_roles", {}).items()}
    executed_current_roles = {
        int(k): v for k, v in case.get("executed_current_roles", {}).items()
    }
    slayer_results = case.get("slayer_results", [])
    slayer_killed = {
        sr["target_pos"] for sr in slayer_results if sr.get("killed")
    }
    pd_results = case.get("pd_ability_results", [])

    # Determine game result
    game_result = "win"
    wrong_execs = [
        p for p in executed
        if p not in confirmed_evil
        and p not in night_kills
        and p not in slayer_killed
    ]
    hp = case.get("hp", 10)
    cost = case.get("wrong_exec_cost", 5)
    remaining_hp = hp - len(wrong_execs) * cost
    terminal_loss_role = case.get("terminal_loss_role")
    if (
        isinstance(terminal_loss_role, str)
        and terminal_loss_role.strip().replace("_", " ").casefold()
        == "bombardier"
    ) or remaining_hp <= 0:
        game_result = "loss"

    # Build incremental state
    current_cards = []
    current_executed = []
    current_confirmed_evil = []
    current_confirmed_good = []
    current_executed_evil_roles = {}
    current_slayer_results = []
    current_pd_results = []
    current_night_kills = []
    current_night_kill_evil_count = 0
    current_reveal_order = []
    current_executed_good_corrupted = {}
    current_executed_good_roles = {}
    current_executed_current_roles = {}
    used_abilities = []

    decisions = []
    step_num = 0

    all_roles = (
        deck_data.get("villagers", []) + deck_data.get("outcasts", []) +
        deck_data.get("minions", []) + deck_data.get("demons", [])
    )

    def make_state() -> GameState:
        revealed_positions = {
            card["position"]
            for card in current_cards
        }
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
            board_count_provenance=case.get("board_count_provenance", "legacy_unknown"),
            rambler_rule_version=case.get("rambler_rule_version"),
            rambler_shut_up_observations=[
                dict(observation)
                for observation in case.get("rambler_shut_up_observations", [])
                if observation.get("speaker_position") in revealed_positions
            ],
            baker_rule_version=case.get("baker_rule_version"),
            doppel_drunk_rule_version=case.get("doppel_drunk_rule_version"),
            fortune_teller_rule_version=case.get("fortune_teller_rule_version"),
            terminal_loss_role=(
                terminal_loss_role
                if terminal_loss_role == "Bombardier"
                and all(pos in current_executed for pos in executed)
                else None
            ),
            executed_current_roles=dict(current_executed_current_roles),
            reveal_order=list(current_reveal_order),
            executed_good_corrupted=dict(current_executed_good_corrupted),
            executed_good_roles=dict(current_executed_good_roles),
        )

    def analyze_at_step(phase: str, description: str, actual_action: str,
                        actual_pos: int | None) -> DecisionPoint:
        nonlocal step_num
        step_num += 1

        state = make_state()
        result = rust_solve_to_objects(state)
        if result is None:
            return DecisionPoint(
                step=step_num, phase=phase, description=description,
                n_surviving=0, recommended_action="error",
                recommended_position=None, recommended_confidence=0,
                actual_action=actual_action, actual_position=actual_pos,
                correct=False, category="solver_unavailable",
            )

        probs = evil_probabilities(state, result)
        action = recommend_action(state, result, used_abilities)
        action.confidence = _compute_confidence(action, state, result, probs)

        rec_str = action.action_type
        if action.position:
            rec_str += f" #{action.position}"

        # Compare recommendation vs actual
        correct = _actions_match(action, actual_action, actual_pos)
        category = ""
        if not correct:
            category = _classify_mismatch(
                action, actual_action, actual_pos, result,
                true_evil_set, probs, state
            )

        return DecisionPoint(
            step=step_num, phase=phase, description=description,
            n_surviving=result.n_surviving,
            recommended_action=rec_str,
            recommended_position=action.position,
            recommended_confidence=action.confidence,
            actual_action=actual_action,
            actual_position=actual_pos,
            correct=correct,
            category=category,
            evil_probs={p: round(prob, 3) for p, prob in probs.items()
                        if 0 < prob < 1 and p not in state.executed},
            definite_evil=list(result.definite_evil),
            definite_good=list(result.definite_good),
            roles_in_deck=all_roles,
        )

    # Night kill timing (Lilis triggers every 4 reveals)
    night_kill_timing = {}
    if night_kills:
        kill_idx = 0
        n_nights = len(reveal_order) // 4
        for night_num in range(1, n_nights + 1):
            after_reveal = night_num * 4
            if kill_idx < len(night_kills):
                night_kill_timing.setdefault(after_reveal, []).append(night_kills[kill_idx])
                kill_idx += 1

    # ── Phase 1: Reveals ──
    for i, pos in enumerate(reveal_order, 1):
        card_data = card_lookup.get(pos)
        if not card_data:
            continue

        current_cards.append(card_data)
        current_reveal_order.append(pos)

        # Check for night kills after this reveal
        kills_now = night_kill_timing.get(i, [])
        for kp in kills_now:
            current_night_kills.append(kp)
            current_executed.append(kp)
            if kp in true_evil_set:
                current_night_kill_evil_count += 1

    # ── Analyze after all reveals (pre-execution) ──
    # Add slayer/PD results
    for sr in slayer_results:
        current_slayer_results.append(sr)
        if sr.get("killed"):
            t = sr["target_pos"]
            if t not in current_executed:
                current_executed.append(t)
            revealed_role = slayer_revealed_role(sr)
            role_def = get_card(revealed_role) if revealed_role else None
            recorded_evil_role = executed_evil_roles.get(t)
            if t in confirmed_evil or recorded_evil_role is not None:
                target_was_good = False
            elif t in confirmed_good:
                target_was_good = True
            elif _is_terminal_loss_role(revealed_role):
                target_was_good = None
            elif role_def is not None:
                target_was_good = role_def.alignment == Alignment.GOOD
            else:
                target_was_good = None
            if target_was_good is True:
                if t not in current_confirmed_good:
                    current_confirmed_good.append(t)
                observed_corruption = case.get("executed_good_corrupted", {})
                if str(t) in observed_corruption or t in observed_corruption:
                    current_executed_good_corrupted[t] = observed_corruption.get(
                        str(t), observed_corruption.get(t)
                    )
                observed_roles = case.get("executed_good_roles", {})
                observed_role = (
                    revealed_role
                    or observed_roles.get(str(t), observed_roles.get(t))
                )
                if observed_role:
                    current_executed_good_roles[t] = observed_role
            elif target_was_good is False:
                if t not in current_confirmed_evil:
                    current_confirmed_evil.append(t)
                if recorded_evil_role:
                    current_executed_evil_roles[t] = recorded_evil_role

    for pd in pd_results:
        current_pd_results.append(pd)

    dp = analyze_at_step(
        "all_revealed", "All cards revealed + abilities applied",
        "solve", None,
    )
    dp.correct = True  # Informational snapshot, not a scored decision
    dp.category = ""
    decisions.append(dp)

    # ── Phase 2: Executions ──
    night_kills_set = set(night_kills)
    for pos in executed:
        if pos in night_kills_set or pos in slayer_killed:
            continue

        was_evil = pos in confirmed_evil or pos in executed_evil_roles
        evil_role = executed_evil_roles.get(pos)

        # What was the actual action?
        actual = f"execute #{pos}"

        dp = analyze_at_step(
            "execute", f"Execute #{pos} ({'evil' if was_evil else 'good'})",
            actual, pos,
        )
        decisions.append(dp)

        # Apply the execution
        current_executed.append(pos)
        current_role = executed_current_roles.get(pos)
        if current_role:
            current_executed_current_roles[pos] = current_role
        if was_evil:
            if pos not in current_confirmed_evil:
                current_confirmed_evil.append(pos)
            if evil_role:
                current_executed_evil_roles[pos] = evil_role
        else:
            if pos not in current_confirmed_good:
                current_confirmed_good.append(pos)
            observed_corruption = case.get("executed_good_corrupted", {})
            if str(pos) in observed_corruption or pos in observed_corruption:
                current_executed_good_corrupted[pos] = observed_corruption.get(
                    str(pos), observed_corruption.get(pos)
                )
            observed_roles = case.get("executed_good_roles", {})
            observed_role = observed_roles.get(str(pos), observed_roles.get(pos))
            if observed_role:
                current_executed_good_roles[pos] = observed_role

    # A legacy/offline case may lack the explicit terminal marker while still
    # carrying authoritative public death evidence. Classify the completed
    # game from the fully applied state; this does not leak final evidence into
    # any earlier decision snapshot.
    final_state = make_state()
    if (
        _is_terminal_loss_role(final_state.terminal_loss_role)
        or _public_terminal_loss_position(final_state) is not None
    ):
        game_result = "loss"

    # ── Summary ──
    total = len(decisions)
    correct = sum(1 for d in decisions if d.correct)
    wrong = total - correct
    categories = Counter(d.category for d in decisions if not d.correct)

    summary = {
        "total_decisions": total,
        "correct": correct,
        "wrong": wrong,
        "accuracy": round(correct / max(total, 1), 3),
        "wrong_categories": dict(categories),
    }

    return GameAnalysis(
        name=name, n_cards=case["n_cards"], n_evil=case.get("n_evil", 0),
        result=game_result, decisions=decisions, summary=summary,
    )


def _actions_match(recommended, actual_action: str, actual_pos: int | None) -> bool:
    """Check if recommended action matches what was actually done."""
    if recommended.action_type == "execute" and actual_action.startswith("execute"):
        return recommended.position == actual_pos
    if recommended.action_type == "win":
        return True  # Any action is fine when game is won
    if recommended.action_type == "error":
        return False  # 0 scenarios is always wrong
    # For reveals/abilities, we're mostly checking execution decisions
    # so treat non-execute recommendations as "correct" for now
    if recommended.action_type in ("reveal", "use_ability") and not actual_action.startswith("execute"):
        return True
    return False


def _classify_mismatch(recommended, actual_action: str, actual_pos: int | None,
                       result: SolverResult, true_evil_set: set,
                       probs: dict, state: GameState) -> str:
    """Classify why recommendation didn't match actual."""
    # 0 scenarios = solver bug
    if result.n_surviving == 0:
        return "zero_scenarios"

    # Recommended execute but actual was different target
    if recommended.action_type == "execute" and actual_action.startswith("execute"):
        rec_is_evil = recommended.position in true_evil_set
        act_is_evil = actual_pos in true_evil_set
        if rec_is_evil and not act_is_evil:
            return "wrong_target_choice"  # solver was right, player chose wrong
        if not rec_is_evil and act_is_evil:
            return "bad_heuristic"  # solver recommended wrong target
        if not rec_is_evil and not act_is_evil:
            return "both_wrong"  # neither was evil
        return "different_valid_target"  # both evil, different order

    # Solver recommended non-execute but player executed
    if recommended.action_type != "execute" and actual_action.startswith("execute"):
        if actual_pos in true_evil_set:
            return "premature_but_correct"  # player jumped ahead, got lucky
        return "premature_wrong_exec"  # player jumped ahead, wrong target

    # Solver recommended execute but player did something else
    if recommended.action_type == "execute" and not actual_action.startswith("execute"):
        return "missed_execution"  # solver found definite evil, player didn't act

    return "uncategorized"


# ============================================================
# Cross-Game Pattern Detection
# ============================================================

def load_all_analyses() -> list[GameAnalysis]:
    """Load all saved analysis JSONs."""
    results = []
    for path in sorted(glob.glob(os.path.join(OUTPUT_DIR, "*.json"))):
        with open(path) as f:
            data = json.load(f)
        decisions = [DecisionPoint(**d) for d in data["decisions"]]
        analysis = GameAnalysis(
            name=data["name"], n_cards=data["n_cards"], n_evil=data["n_evil"],
            result=data["result"], decisions=decisions, summary=data["summary"],
        )
        results.append(analysis)
    return results


def failure_report(analyses: list[GameAnalysis] = None, top_n: int = 20) -> list[dict]:
    """Find recurring failure patterns across games.

    Groups wrong decisions by {category, roles_combo_hash, n_cards} and
    returns patterns with 2+ occurrences, sorted by frequency.
    """
    if analyses is None:
        analyses = load_all_analyses()

    patterns: dict[str, list] = {}

    for game in analyses:
        for dp in game.decisions:
            if dp.correct:
                continue

            # Create pattern key
            key_parts = [
                dp.category,
                str(game.n_cards),
                str(game.n_evil),
            ]
            key = "|".join(key_parts)

            if key not in patterns:
                patterns[key] = []
            patterns[key].append({
                "game": game.name,
                "step": dp.step,
                "description": dp.description,
                "recommended": dp.recommended_action,
                "actual": dp.actual_action,
                "n_surviving": dp.n_surviving,
                "confidence": dp.recommended_confidence,
            })

    # Filter to 2+ occurrences and sort by frequency
    recurring = [
        {"pattern": key, "count": len(instances), "instances": instances}
        for key, instances in patterns.items()
        if len(instances) >= 2
    ]
    recurring.sort(key=lambda x: -x["count"])
    return recurring[:top_n]


# ============================================================
# CLI
# ============================================================

def _load_case(name: str) -> dict | None:
    """Load a test case by name."""
    path = os.path.join(CASES_DIR, f"{name}.json")
    if not os.path.exists(path):
        print(f"Test case not found: {path}")
        return None
    with open(path) as f:
        case = json.load(f)
    case["name"] = name
    return case


def _save_analysis(analysis: GameAnalysis):
    """Save analysis to JSON."""
    path = os.path.join(OUTPUT_DIR, f"{analysis.name}.json")
    data = {
        "name": analysis.name,
        "n_cards": analysis.n_cards,
        "n_evil": analysis.n_evil,
        "result": analysis.result,
        "decisions": [asdict(d) for d in analysis.decisions],
        "summary": analysis.summary,
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def cmd_analyze(name: str):
    """Analyze a single game and print results."""
    case = _load_case(name)
    if not case:
        return

    analysis = analyze_game(case)
    _save_analysis(analysis)

    print(f"\n=== Decision Analysis: {name} ===")
    print(f"  Result: {analysis.result}, {analysis.n_cards} cards, {analysis.n_evil} evil")
    print(f"  Decisions: {analysis.summary['correct']}/{analysis.summary['total_decisions']} correct")
    print()

    for dp in analysis.decisions:
        marker = "OK" if dp.correct else "XX"
        print(f"  [{marker}] Step {dp.step}: {dp.description}")
        print(f"       Recommended: {dp.recommended_action} ({dp.recommended_confidence:.0%})")
        print(f"       Actual:      {dp.actual_action}")
        if not dp.correct:
            print(f"       Category:    {dp.category}")
        print(f"       Scenarios:   {dp.n_surviving}")
        if dp.evil_probs:
            top_probs = sorted(dp.evil_probs.items(), key=lambda x: -x[1])[:5]
            print(f"       Evil probs:  {', '.join(f'#{p}={prob:.0%}' for p, prob in top_probs)}")
        print()


def cmd_analyze_all():
    """Analyze all v2 test cases."""
    cases = sorted(glob.glob(os.path.join(CASES_DIR, "*.json")))
    total = len(cases)
    print(f"Analyzing {total} test cases...")

    all_analyses = []
    for i, path in enumerate(cases, 1):
        name = os.path.splitext(os.path.basename(path))[0]
        with open(path) as f:
            case = json.load(f)
        case["name"] = name

        analysis = analyze_game(case)
        _save_analysis(analysis)
        all_analyses.append(analysis)

        status = "OK" if analysis.summary["wrong"] == 0 else f"{analysis.summary['wrong']} wrong"
        if (i % 20 == 0) or i == total:
            print(f"  [{i}/{total}] {name}: {status}")

    # Summary
    total_decisions = sum(a.summary["total_decisions"] for a in all_analyses)
    total_correct = sum(a.summary["correct"] for a in all_analyses)
    total_wrong = total_decisions - total_correct
    all_categories = Counter()
    for a in all_analyses:
        all_categories.update(a.summary.get("wrong_categories", {}))

    print(f"\n=== ANALYSIS SUMMARY ===")
    print(f"  Games: {total}")
    print(f"  Total decisions: {total_decisions}")
    print(f"  Correct: {total_correct} ({total_correct/max(total_decisions,1):.1%})")
    print(f"  Wrong: {total_wrong}")
    if all_categories:
        print(f"  Categories:")
        for cat, count in all_categories.most_common():
            print(f"    {cat}: {count}")

    # Run failure pattern detection
    print()
    patterns = failure_report(all_analyses)
    if patterns:
        print(f"=== RECURRING FAILURE PATTERNS ===")
        for p in patterns[:10]:
            parts = p["pattern"].split("|")
            cat = parts[0] if parts else "?"
            print(f"  [{p['count']}x] {cat} — {p['pattern']}")
            for inst in p["instances"][:3]:
                print(f"       {inst['game']}: {inst['description']}")
    else:
        print("No recurring failure patterns found.")


def cmd_failure_report():
    """Show top failure patterns from saved analyses."""
    patterns = failure_report()
    if not patterns:
        print("No analyses found. Run: python decision_analysis.py analyze_all")
        return

    print(f"\n=== FAILURE PATTERNS (2+ occurrences) ===")
    for p in patterns:
        parts = p["pattern"].split("|")
        cat = parts[0] if parts else "?"
        n_cards = parts[1] if len(parts) > 1 else "?"
        n_evil = parts[2] if len(parts) > 2 else "?"
        print(f"\n  [{p['count']}x] {cat} ({n_cards} cards, {n_evil} evil)")
        for inst in p["instances"]:
            print(f"    {inst['game']}: {inst['description']}")
            print(f"      Recommended: {inst['recommended']}, Actual: {inst['actual']}")


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python decision_analysis.py analyze <case_name>")
        print("  python decision_analysis.py analyze_all")
        print("  python decision_analysis.py failure_report")
        return

    cmd = sys.argv[1].lower()
    if cmd == "analyze":
        if len(sys.argv) < 3:
            print("Usage: python decision_analysis.py analyze <case_name>")
            return
        cmd_analyze(sys.argv[2])
    elif cmd in ("analyze_all", "all"):
        cmd_analyze_all()
    elif cmd in ("failure_report", "failures", "report"):
        cmd_failure_report()
    else:
        print(f"Unknown command: {cmd}")


if __name__ == "__main__":
    main()
