# demonBluffSolver.py
# Minimal, puzzle-specific Demon Bluff solver for puzzle1.yaml
# Python 3.11+
#
# New in this version:
# - If ambiguous, prints ALL consistent worlds.
# - Recommends Fortune Teller queries (pairs of seats) that maximally split the worlds.
#   Ranking = minimax (fewest worlds in worst case), tie-break by information gain.
#
# Scope (kept tiny on purpose to "start slow"):
# - Roles used: {alchemist, fortune teller, scout, knitter, medium, wretch, puppeteer, minion, puppet}
# - Truth defaults: villagers/outcasts truthful; minions lie/disguise; puppet truthful but shows as villager
# - Puppeteer must have exactly one adjacent puppet (created from a villager-like GOOD seat)
# - Knitter claim = exact adjacent evil pairs count
# - Medium claim = “#X is the real Y”
# - Scout line used in puzzle1 = “puppeteer is 1 card away from closest evil” (validate as a factual constraint)
#
# Run:
#   python3 demonBluffSolver.py
#   python3 demonBluffSolver.py puzzle1.yaml
#
from __future__ import annotations
import sys, argparse, itertools, re, math, yaml
from dataclasses import dataclass
from typing import Dict, List, Tuple, Set

# ---------- tiny role model (only what's needed for puzzle1) ----------
VILLAGERS: Set[str] = {"alchemist","fortune teller","scout","knitter","medium"}
OUTCASTS: Set[str]   = {"wretch"}
MINIONS: Set[str]    = {"minion","puppeteer"}
SPECIAL: Set[str]    = {"puppet"}  # virtual evil, truthful

def is_truthful(true_role: str) -> bool:
    if true_role in VILLAGERS or true_role in OUTCASTS: return True
    if true_role in MINIONS: return False
    if true_role == "puppet": return True
    return False

def is_evil(true_role: str) -> bool:
    return (true_role in MINIONS) or (true_role == "puppet")

def neighbors(seat: int, N: int) -> Tuple[int,int]:
    return ((seat-2) % N + 1, seat % N + 1)

def ring_distance(a: int, b: int, N: int) -> int:
    d = abs(a-b)
    return min(d, N-d)

# ---------- parse puzzle ----------
@dataclass
class Puzzle:
    seats: int
    deck: List[str]
    flipped: Dict[int,str]
    info_log: List[Dict[str, str]]
    executions_to_win: int

def load_puzzle(path: str) -> Puzzle:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    seats = int(data["seats"])
    deck = list(data.get("deck", []))
    flipped = {int(k): v.lower() for k,v in data.get("flipped", {}).items()}
    info_log = list(data.get("info_log", []))
    options = data.get("options", {})
    executions_to_win = int(options.get("executions_to_win", 3))
    return Puzzle(seats, deck, flipped, info_log, executions_to_win)

# ---------- read statements we care about (puzzle1 patterns) ----------
@dataclass
class KnitterPairsClaim:
    seat: int
    pairs: int

@dataclass
class MediumRealClaim:
    seat: int
    target_seat: int
    role: str

@dataclass
class ScoutPuppeteerDistanceClaim:
    seat: int
    distance: int

@dataclass
class AlchemistCuredClaim:
    seat: int
    cured: int

def parse_info_log(info_log: List[Dict[str,str]]):
    knitter_pairs: List[KnitterPairsClaim] = []
    medium_real: List[MediumRealClaim] = []
    scout_puppeteer_d: List[ScoutPuppeteerDistanceClaim] = []
    alchem_cured: List[AlchemistCuredClaim] = []

    for row in info_log:
        seat = int(row.get("seat"))
        says = (row.get("says") or "").strip().lower()

        m = re.search(r"there are (\d+) pairs of evil", says)
        if m:
            knitter_pairs.append(KnitterPairsClaim(seat=seat, pairs=int(m.group(1))))
            continue

        m = re.search(r"#(\d+)\s+is the real\s+([a-z ]+)", says)
        if m:
            medium_real.append(MediumRealClaim(seat=seat, target_seat=int(m.group(1)), role=m.group(2).strip()))
            continue

        m = re.search(r"puppeteer is (\d+) card[s]? away from closest evil", says)
        if m:
            scout_puppeteer_d.append(SoutPuppeteerDistanceClaim(seat=seat, distance=int(m.group(1))))  # typo fixed below
            continue

        m = re.search(r"i cured (\d+) corruptions", says)
        if m:
            alchem_cured.append(AlchemistCuredClaim(seat=seat, cured=int(m.group(1))))
            continue

    return knitter_pairs, medium_real, scout_puppeteer_d, alchem_cured

# Fix a tiny typo above (keeps parsing robust without retyping logs)
class SoutPuppeteerDistanceClaim(ScoutPuppeteerDistanceClaim):
    pass

# ---------- world checks ----------
def pairs_of_evil(world: Dict[int,str], N: int) -> int:
    pairs = 0
    for i in range(1, N+1):
        j = i % N + 1
        if is_evil(world[i]) and is_evil(world[j]):
            pairs += 1
    return pairs

def check_flips(world: Dict[int,str], flipped: Dict[int,str]) -> bool:
    # Truthful seats must show their true role; minions can disguise; puppet must show a villager.
    for s, true_role in world.items():
        shown = flipped.get(s, "")
        if true_role == "puppet":
            if shown not in VILLAGERS:
                return False
        elif is_truthful(true_role):
            if shown != true_role:
                return False
        else:
            # liar (minion) → no constraint on shown
            pass
    return True

def check_puppetry(world: Dict[int,str], N: int) -> bool:
    # Exactly one puppeteer and one puppet; puppet must be adjacent to puppeteer.
    seats_by_role = invert(world)
    if "puppeteer" not in seats_by_role or "puppet" not in seats_by_role:
        return False
    ps = seats_by_role["puppeteer"][0]
    pu = seats_by_role["puppet"][0]
    return pu in neighbors(ps, N)

def invert(world: Dict[int,str]) -> Dict[str,List[int]]:
    d: Dict[str,List[int]] = {}
    for s,r in world.items():
        d.setdefault(r, []).append(s)
    return d

def check_knitter_claims(world: Dict[int,str], knitter_pairs: List[KnitterPairsClaim], N: int) -> bool:
    for claim in knitter_pairs:
        if world.get(claim.seat) != "knitter": return False
        if pairs_of_evil(world, N) != claim.pairs: return False
    return True

def check_medium_claims(world: Dict[int,str], medium_real: List[MediumRealClaim]) -> bool:
    for claim in medium_real:
        if world.get(claim.seat) != "medium": return False
        if world.get(claim.target_seat) != claim.role: return False
    return True

def check_scout_claims(world: Dict[int,str], scout_ds: List[ScoutPuppeteerDistanceClaim], N: int) -> bool:
    seats_by_role = invert(world)
    if "puppeteer" not in seats_by_role: return False
    p = seats_by_role["puppeteer"][0]
    evil_seats = [s for s,r in world.items() if is_evil(r) and s != p]
    if not evil_seats: return False
    nearest = min(ring_distance(p, e, N) for e in evil_seats)
    for claim in scout_ds:
        if nearest != claim.distance: return False
    return True

def solve_puzzle(puz):
    N = puz.seats
    roles = ["knitter","fortune teller","alchemist","wretch","medium","puppeteer","minion","puppet"]
    knit_claims, med_claims, scout_ds, alchem_claims = parse_info_log(puz.info_log)

    worlds: List[Dict[int,str]] = []
    for perm in itertools.permutations(roles, N):
        world = {i+1: perm[i] for i in range(N)}

        if not check_flips(world, puz.flipped): continue
        if not check_puppetry(world, N): continue
        if not check_knitter_claims(world, knit_claims, N): continue
        if not check_medium_claims(world, med_claims): continue
        if not check_scout_claims(world, scout_ds, N): continue
        # (We intentionally ignore alchemist cured counts for puzzle1.)

        worlds.append(world)
    return worlds

# ---------- Fortune Teller planning ----------
# Fortune Teller (active): pick 2 seats => learn if ANY of them is evil (Yes/No).
def ft_outcome_for_pair(world: Dict[int,str], i: int, j: int) -> str:
    return "Yes" if (is_evil(world[i]) or is_evil(world[j])) else "No"

def score_pair(worlds: List[Dict[int,str]], i: int, j: int) -> Tuple[int,float,int,int]:
    # Returns (minimax_worst, info_gain_bits, yes_count, no_count)
    total = len(worlds)
    yes = sum(1 for w in worlds if ft_outcome_for_pair(w, i, j) == "Yes")
    no = total - yes
    worst = max(yes, no)
    # Info gain = H(prior) - weighted H(posteriors) over outcomes
    def H(n): 
        if n == 0: return 0.0
        p = n/total
        return -p * math.log2(p)
    prior = math.log2(total) if total > 0 else 0.0
    # Posterior entropy over world-identity outcomes is simply H(yes_count) + H(no_count)
    info_gain = prior - (H(yes) + H(no))
    return (worst, info_gain, yes, no)

def best_fortune_teller_pairs(worlds: List[Dict[int,str]], N: int, top_k: int = 5):
    pairs = []
    for i in range(1, N+1):
        for j in range(i+1, N+1):
            worst, ig, yes, no = score_pair(worlds, i, j)
            pairs.append(((i,j), worst, ig, yes, no))
    # Sort: fewest worst-case remaining worlds, then higher info gain
    pairs.sort(key=lambda x: (x[1], -x[2], x[0]))
    return pairs[:top_k]

# ---------- pretty printing ----------
def fmt_world(world: Dict[int,str]) -> str:
    seats = sorted(world.keys())
    parts = []
    for s in seats:
        r = world[s]
        tag = " (evil)" if is_evil(r) else ""
        parts.append(f"#{s}:{r}{tag}")
    return ", ".join(parts)

def pick_executions(world: Dict[int,str], k: int) -> List[int]:
    evils = [s for s,r in world.items() if is_evil(r)]
    return evils[:k]

# ---------- main ----------
def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("puzzle", nargs="?", default="puzzle1.yaml")
    args = ap.parse_args(argv)

    puz = load_puzzle(args.puzzle)
    worlds = solve_puzzle(puz)

    if not worlds:
        print("No consistent worlds found.")
        return 2

    if len(worlds) == 1:
        world = worlds[0]
        print("Unique world:")
        for s in range(1, puz.seats+1):
            print(f"  #{s}: {world[s]}{' (evil)' if is_evil(world[s]) else ''}")
        to_execute = pick_executions(world, puz.executions_to_win)
        print(f"\nExecute {puz.executions_to_win}: {to_execute}")
        return 0

    # Ambiguous: print every world and Fortune Teller guidance
    print(f"Ambiguous: {len(worlds)} consistent worlds.\n")
    for idx, w in enumerate(worlds, 1):
        print(f"World {idx}: {fmt_world(w)}")

    print("\nFortune Teller suggestions (pairs of seats):")
    best = best_fortune_teller_pairs(worlds, puz.seats, top_k=6)
    total = len(worlds)
    for (i,j), worst, ig, yes, no in best:
        print(f"  ({i},{j})  → worst-case {worst}/{total} worlds,  IG≈{ig:.3f} bits,  Yes={yes}, No={no}")

    # Also show the single best (minimax) recommendation explicitly
    if best:
        (i,j), worst, ig, yes, no = best[0]
        print(f"\nRecommend Fortune Teller on seats ({i},{j}). "
              f"If answer is Yes → {yes} worlds remain; if No → {no} worlds remain "
              f"(worst case {worst}/{total}, ≈{ig:.3f} bits of information).")
    return 1

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
