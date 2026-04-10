"""Self-Improvement Analytics for Demon Bluff Solver.

Analyzes scorecard and test case data to surface loss patterns,
win rates by deck configuration, recurring errors, and rolling
win-rate trends.

Usage:
    python analytics.py              # Full report
    python analytics.py losses       # Loss patterns only
    python analytics.py configs      # Win rate by config only
    python analytics.py errors       # Recurring errors only
    python analytics.py rolling      # Rolling win rate only
"""

import json
import os
import sys
import glob

SCORECARD_FILE = os.path.join(os.path.dirname(__file__), "scorecard.json")
CASES_DIR_V2 = os.path.join(os.path.dirname(__file__), "tests", "cases_v2")
CASES_DIR_LEGACY = os.path.join(os.path.dirname(__file__), "tests", "cases")

# Demon role names for config extraction
DEMON_NAMES = {"Baa", "Pooka", "Lilis"}

# Lilis internal name variants that might appear in deck lists
LILIS_NAMES = {"Lilis", "Lillith", "Striga"}

# Priority-ordered loss buckets: (category, keywords)
LOSS_BUCKETS = [
    ("solver_0_scenarios", ["0 scenario", "0-scenario", "SOLVER BUG", "zero scenario"]),
    ("bombardier", ["bombardier", "blew up", "Bombardier"]),
    ("wrong_exec_corrupted", ["corrupted"]),
    ("wrong_exec_50_50", ["50/50", "coin flip", "guess", "1-in-3"]),
    ("data_entry", ["data entry", "misclass", "wrong faction", "wrong role"]),
    ("click_failure", ["misclick", "focus", "click fail"]),
    ("hp_attrition", ["HP", "attrition", "hp ran out"]),
]


def _load_scorecard(scorecard_path=None):
    path = scorecard_path or SCORECARD_FILE
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"games": []}


def _classify_loss(notes):
    """Classify a loss by checking notes against priority-ordered buckets."""
    if not notes:
        return "uncategorized"
    for category, keywords in LOSS_BUCKETS:
        for kw in keywords:
            if kw in notes:
                return category
    return "uncategorized"


def loss_patterns(scorecard_path=None):
    """Parse loss notes with priority-ordered keyword matcher.

    Returns: {category: [game_names]}
    """
    data = _load_scorecard(scorecard_path)
    games = data.get("games", [])
    losses = [g for g in games if g.get("result") == "loss"]

    buckets = {}
    for g in losses:
        notes = g.get("notes", "") or g.get("loss_cause", "") or ""
        category = _classify_loss(notes)
        buckets.setdefault(category, []).append(g.get("name", "unnamed"))

    return buckets


def _load_all_cases(cases_dirs=None):
    """Load test cases from both directories, keyed by name."""
    dirs = cases_dirs or [CASES_DIR_LEGACY, CASES_DIR_V2]
    cases = {}
    for d in dirs:
        if not os.path.isdir(d):
            continue
        for path in glob.glob(os.path.join(d, "*.json")):
            name = os.path.basename(path).replace(".json", "")
            with open(path) as f:
                case = json.load(f)
            cases[name] = case
    return cases


def _get_demon_role(case):
    """Extract demon role from a test case."""
    # Check true_evil_positions first
    true_evils = case.get("true_evil_positions", {})
    for pos, role in true_evils.items():
        if role in DEMON_NAMES:
            return role

    # Fall back to deck demons list
    deck = case.get("deck", {})
    demons = deck.get("demons", case.get("demons", []))
    if demons:
        return demons[0]

    return "unknown"


def _has_role_in_deck(case, role_names):
    """Check if any of the given role names appear in the deck."""
    deck = case.get("deck", {})
    all_roles = []
    for key in ("villagers", "outcasts", "minions", "demons"):
        all_roles.extend(deck.get(key, case.get(key, []) or []))
    return any(r in role_names for r in all_roles)


def accuracy_by_config(cases_dirs=None, scorecard_path=None):
    """Join test cases to scorecard by name for win/loss.

    Group by (n_cards, demon_role, has_witch, has_lilis).
    Return sorted list of config dicts with win_rate.
    """
    data = _load_scorecard(scorecard_path)
    games = data.get("games", [])
    result_by_name = {g.get("name"): g.get("result") for g in games}

    cases = _load_all_cases(cases_dirs)

    # Group by config
    configs = {}
    for name, case in cases.items():
        result = result_by_name.get(name)
        if result is None:
            continue  # no scorecard entry for this case

        n_cards = case.get("n_cards", 0)
        demon_role = _get_demon_role(case)
        has_witch = _has_role_in_deck(case, {"Witch"})
        has_lilis = _has_role_in_deck(case, LILIS_NAMES)

        key = (n_cards, demon_role, has_witch, has_lilis)
        configs.setdefault(key, {"wins": 0, "losses": 0, "games": []})
        if result == "win":
            configs[key]["wins"] += 1
        else:
            configs[key]["losses"] += 1
        configs[key]["games"].append(name)

    # Build sorted result list
    results = []
    for (n_cards, demon, has_witch, has_lilis), stats in configs.items():
        total = stats["wins"] + stats["losses"]
        win_rate = stats["wins"] / total * 100 if total else 0
        results.append({
            "n_cards": n_cards,
            "demon": demon,
            "has_witch": has_witch,
            "has_lilis": has_lilis,
            "wins": stats["wins"],
            "losses": stats["losses"],
            "total": total,
            "win_rate": win_rate,
            "games": stats["games"],
        })

    results.sort(key=lambda r: (r["win_rate"], -r["total"]))
    return results


def recurring_errors(scorecard_path=None, window=20):
    """Find error signatures that appear >2 times in last N games.

    Returns sorted list by count descending.
    """
    data = _load_scorecard(scorecard_path)
    games = data.get("games", [])
    recent = games[-window:] if len(games) >= window else games
    losses = [g for g in recent if g.get("result") == "loss"]

    # Build signatures: (loss_cause, first 40 chars of notes)
    sigs = {}
    for g in losses:
        notes = g.get("notes", "") or ""
        cause = g.get("loss_cause") or _classify_loss(notes)
        sig_notes = notes[:40].strip()
        sig = (cause, sig_notes)
        if sig not in sigs:
            sigs[sig] = {
                "cause": cause,
                "signature": sig_notes,
                "count": 0,
                "first_seen": g.get("date", ""),
                "last_seen": g.get("date", ""),
                "game_names": [],
            }
        sigs[sig]["count"] += 1
        sigs[sig]["last_seen"] = g.get("date", "")
        sigs[sig]["game_names"].append(g.get("name", "unnamed"))

    # Filter to >2 occurrences
    results = [v for v in sigs.values() if v["count"] > 2]
    results.sort(key=lambda r: -r["count"])
    return results


def rolling_winrate(scorecard_path=None, window=20):
    """Compute rolling win rate and detect regression.

    Returns dict with all_time_rate, rolling_rate, is_declining.
    is_declining is True if 2+ consecutive windows show decline.
    """
    data = _load_scorecard(scorecard_path)
    games = data.get("games", [])

    total = len(games)
    wins = sum(1 for g in games if g.get("result") == "win")
    all_time_rate = wins / total * 100 if total else 0

    if total < window:
        return {
            "all_time_rate": all_time_rate,
            "rolling_rate": all_time_rate,
            "is_declining": False,
            "window": window,
            "total_games": total,
            "note": f"Not enough games for {window}-game window",
        }

    # Current rolling window
    recent = games[-window:]
    rolling_wins = sum(1 for g in recent if g.get("result") == "win")
    rolling_rate = rolling_wins / window * 100

    # Check for declining trend: compute rolling rates for last 3 windows
    is_declining = False
    if total >= window * 3:
        rates = []
        for i in range(3):
            start = total - window * (3 - i)
            end = total - window * (2 - i)
            chunk = games[start:end]
            chunk_wins = sum(1 for g in chunk if g.get("result") == "win")
            rates.append(chunk_wins / window * 100)

        # Declining if each window is lower than the previous
        is_declining = rates[1] < rates[0] and rates[2] < rates[1]

    return {
        "all_time_rate": all_time_rate,
        "rolling_rate": rolling_rate,
        "is_declining": is_declining,
        "window": window,
        "total_games": total,
    }


def report(scorecard_path=None):
    """Print full analytics report."""
    path = scorecard_path or SCORECARD_FILE

    print("=" * 60)
    print("  DEMON BLUFF SOLVER -- SELF-IMPROVEMENT ANALYTICS")
    print("=" * 60)

    # --- Rolling Win Rate ---
    print("\n--- Rolling Win Rate ---")
    rw = rolling_winrate(path)
    trend = "DECLINING" if rw["is_declining"] else "STABLE"
    print(f"  All-time:  {rw['all_time_rate']:.0f}% ({rw['total_games']} games)")
    print(f"  Last {rw['window']:>3}:  {rw['rolling_rate']:.0f}% [{trend}]")
    if rw.get("note"):
        print(f"  Note: {rw['note']}")

    # --- Loss Patterns ---
    print("\n--- Loss Patterns ---")
    patterns = loss_patterns(path)
    if not patterns:
        print("  No losses recorded.")
    else:
        total_losses = sum(len(v) for v in patterns.values())
        print(f"  Total losses: {total_losses}")
        for category, names in sorted(patterns.items(),
                                       key=lambda x: -len(x[1])):
            pct = len(names) / total_losses * 100 if total_losses else 0
            print(f"  {category:.<30s} {len(names):>2} ({pct:4.0f}%)  "
                  f"{', '.join(names[:5])}"
                  f"{'...' if len(names) > 5 else ''}")

    # --- Win Rate by Config ---
    print("\n--- Win Rate by Deck Config ---")
    configs = accuracy_by_config(scorecard_path=path)
    if not configs:
        print("  No matched test cases with scorecard entries.")
    else:
        print(f"  {'Config':<40s} {'W':>3} {'L':>3} {'Tot':>4} {'Rate':>5}")
        print(f"  {'-'*40} {'-'*3} {'-'*3} {'-'*4} {'-'*5}")
        for c in configs:
            witch_str = "+Witch" if c["has_witch"] else ""
            lilis_str = "+Lilis" if c["has_lilis"] else ""
            label = (f"{c['n_cards']}cards {c['demon']}"
                     f"{witch_str}{lilis_str}")
            print(f"  {label:<40s} {c['wins']:>3} {c['losses']:>3} "
                  f"{c['total']:>4} {c['win_rate']:>4.0f}%")

    # --- Recurring Errors ---
    print("\n--- Recurring Errors (last 20 games) ---")
    errors = recurring_errors(path)
    if not errors:
        print("  No recurring error signatures (>2 occurrences).")
    else:
        for e in errors:
            print(f"  [{e['count']}x] {e['cause']}: \"{e['signature']}\"")
            print(f"       First: {e['first_seen']}  Last: {e['last_seen']}")
            print(f"       Games: {', '.join(e['game_names'])}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "report"

    if cmd == "losses":
        patterns = loss_patterns()
        for cat, names in sorted(patterns.items(), key=lambda x: -len(x[1])):
            print(f"{cat}: {', '.join(names)}")
    elif cmd == "configs":
        configs = accuracy_by_config()
        for c in configs:
            print(f"{c['n_cards']}cards {c['demon']} "
                  f"witch={c['has_witch']} lilis={c['has_lilis']}: "
                  f"{c['wins']}W/{c['losses']}L = {c['win_rate']:.0f}%")
    elif cmd == "errors":
        errors = recurring_errors()
        if not errors:
            print("No recurring errors found.")
        else:
            for e in errors:
                print(f"[{e['count']}x] {e['cause']}: {e['signature']}")
    elif cmd == "rolling":
        rw = rolling_winrate()
        trend = "DECLINING" if rw["is_declining"] else "STABLE"
        print(f"All-time: {rw['all_time_rate']:.0f}% | "
              f"Last {rw['window']}: {rw['rolling_rate']:.0f}% [{trend}]")
    else:
        report()
