# demonBluffSolver.py
# Minimal, puzzle-specific Demon Bluff solver for puzzle1.yaml
# Python 3.11+

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

# ---------- info_log patterns used in puzzle1 ----------
@dataclass
class KnitterPairsClaim: seat:int; pairs:int
@dataclass
class MediumRealClaim: seat:int; target_seat:int; role:str
@dataclass
class ScoutPuppeteerDistanceClaim: seat:int; distance:int
@dataclass
class AlchemistCuredClaim: seat:int; cured:int

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
            knitter_pairs.append(KnitterPairsClaim(seat, int(m.group(1))))
            continue

        m = re.search(r"#(\d+)\s+is the real\s+([a-z ]+)", says)
        if m:
            medium_real.append(MediumRealClaim(seat, int(m.group(1)), m.group(2).strip()))
            continue

        m = re.search(r"puppeteer is (\d+) card[s]? away from closest evil", says)
        if m:
            scout_puppeteer_d.append(ScoutPuppeteerDistanceClaim(seat, int(m.group(1))))
            continue

        m = re.search(r"i cured (\d+) corruptions", says)
        if m:
            alchem_cured.append(AlchemistCuredClaim(seat, int(m.group(1))))
            continue

    return knitter_pairs, medium_real, scout_puppeteer_d, alchem_cured

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
    by = invert(world)
    if "puppeteer" not in by or "puppet" not in by: return False
    ps = by["puppeteer"][0]
    pu = by["puppet"][0]
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
    # Validate just the factual content of the statement (as in the puzzle screenshot).
    by = invert(world)
    if "puppeteer" not in by: return False
    p = by["puppeteer"][0]
    evil_seats = [s for s,r in world.items() if is_evil(r) and s != p]
    if not evil_seats: return False
    nearest = min(ring_distance(p, e, N) for e in evil_seats)
    for claim in scout_ds:
        if nearest != claim.distance: return False
    return True

def solve_puzzle(puz):
    N = puz.seats
    # Roles actually used for puzzle1 reasoning
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
        # (Alchemist cured counts are ignored for puzzle1.)
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
    # Correct IG: H_prior - E[H_posterior (over world identity)]
    # H_prior = log2(total)
    # E[H_post] = (yes/total)*log2(yes) + (no/total)*log2(no)   [with 0→0 guards]
    def lg(n): return math.log2(n) if n > 0 else 0.0
    prior = lg(total)
    p_yes = yes/total
    p_no  = no/total
    expected_post = p_yes*lg(yes) + p_no*lg(no)
    info_gain = prior - expected_post
    return (worst, info_gain, yes, no)

def best_fortune_teller_pairs(worlds: List[Dict[int,str]], N: int, top_k: int = 5):
    pairs = []
    for i in range(1, N+1):
        for j in range(i+1, N+1):
            worst, ig, yes, no = score_pair(worlds, i, j)
            pairs.append(((i,j), worst, ig, yes, no))
    # Sort: fewest worst-case remaining worlds, then higher info gain
    pairs.sort(key=lambda x: (x[1], -x[2], x[0]))
    return pairs[:top_k], pairs

# ---------- helpers ----------
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

def evil_sets(worlds: List[Dict[int,str]]) -> Tuple[Set[int], Set[int], bool]:
    sets = [set(s for s,r in w.items() if is_evil(r)) for w in worlds]
    inter = set.intersection(*sets) if sets else set()
    uni   = set.union(*sets) if sets else set()
    all_same = all(s == sets[0] for s in sets)
    return inter, uni, all_same

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

    # Ambiguous: print every world
    print(f"Ambiguous: {len(worlds)} consistent worlds.\n")
    for idx, w in enumerate(worlds, 1):
        print(f"World {idx}: {fmt_world(w)}")

    # If the set of EVIL seats is invariant and already solves the objective, print executions now
    inter, uni, all_same = evil_sets(worlds)
    if all_same:
        evils_list = sorted(next(iter(set(s for s,r in w.items() if is_evil(r)) for w in worlds)))
        if len(evils_list) >= puz.executions_to_win:
            print(f"\n✅ The evil seats are identical across all worlds: {evils_list}")
            print(f"Execute {puz.executions_to_win}: {evils_list[:puz.executions_to_win]}")
            print("(Fortune Teller cannot further disambiguate who is minion/puppet/puppeteer; it only answers 'any evil?')")
            return 0

    # Otherwise, offer FT only if it actually splits worlds
    print("\nFortune Teller suggestions (pairs that split the worlds):")
    best, all_pairs = best_fortune_teller_pairs(worlds, puz.seats, top_k=10)
    total = len(worlds)
    any_split = False
    for (i,j), worst, ig, yes, no in best:
        if worst < total:  # actually splits
            any_split = True
            print(f"  ({i},{j}) → worst-case {worst}/{total}, IG≈{ig:.3f} bits, Yes={yes}, No={no}")
    if not any_split:
        print("  (none) — every FT query leaves all worlds possible; skip FT and execute the invariant evil seats.")
        if all_same:
            evils_list = sorted(inter)
            print(f"\nExecute {puz.executions_to_win}: {evils_list[:puz.executions_to_win]}")
            return 0
    else:
        (i,j), worst, ig, yes, no = min(((p[0], p[1], p[2], p[3], p[4]) for p in all_pairs), key=lambda x: (x[1], -x[2], x[0]))
        print(f"\nRecommend FT on seats ({i},{j}). If Yes → {yes} worlds remain; if No → {no} worlds remain "
              f"(worst case {worst}/{total}, ≈{ig:.3f} bits).")

    return 1

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
