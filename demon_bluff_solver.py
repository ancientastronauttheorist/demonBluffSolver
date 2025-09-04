# demonBluffSolver.py
# Minimal but extensible Demon Bluff solver for puzzle1 + puzzle2.
# Python 3.11+

from __future__ import annotations
import sys, argparse, itertools, re, math, yaml
from dataclasses import dataclass
from typing import Dict, List, Tuple, Set, Optional

# ---------- role sets (expanded) ----------
VILLAGERS: Set[str] = {
    "alchemist","fortune teller","scout","knitter","medium",
    "druid","baker","bard","enlightened"
}
OUTCASTS: Set[str]   = {"wretch","plague doctor","drunk"}
MINIONS: Set[str]    = {"minion","puppeteer","counsellor","shaman"}
SPECIAL: Set[str]    = {"puppet"}  # virtual evil, truthful

def is_truthful(true_role: str) -> bool:
    # Villagers & outcasts are truthful by default, except Drunk who lies.
    if true_role == "drunk": return False
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
    info_log: List[Dict[str,str]]
    executions_to_win: int

def load_puzzle(path: str) -> Puzzle:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    seats = int(data["seats"])
    deck = [str(x).lower() for x in data.get("deck", [])]
    flipped = {int(k): v.lower() for k,v in data.get("flipped", {}).items()}
    info_log = list(data.get("info_log", []))
    options = data.get("options", {})
    executions_to_win = int(options.get("executions_to_win", 3))
    return Puzzle(seats, deck, flipped, info_log, executions_to_win)

# ---------- info_log patterns ----------
@dataclass
class KnitterPairsClaim: seat:int; pairs:int
@dataclass
class MediumRealClaim: seat:int; target_seat:int; role:str
@dataclass
class ScoutPuppeteerDistanceClaim: seat:int; distance:int
@dataclass
class AlchemistCuredClaim: seat:int; cured:int
@dataclass
class BakerOriginalClaim: seat:int
@dataclass
class BakerWasClaim: seat:int; was_role:str

def parse_info_log(info_log: List[Dict[str,str]]):
    knitter_pairs: List[KnitterPairsClaim] = []
    medium_real: List[MediumRealClaim] = []
    scout_puppeteer_d: List[ScoutPuppeteerDistanceClaim] = []
    alchem_cured: List[AlchemistCuredClaim] = []
    baker_original: List[BakerOriginalClaim] = []
    baker_was: List[BakerWasClaim] = []

    for row in info_log:
        seat = int(row.get("seat"))
        says = (row.get("says") or "").strip().lower()

        m = re.search(r"there (?:are|is) (\d+) pairs? of evil", says)
        if m:
            knitter_pairs.append(KnitterPairsClaim(seat, int(m.group(1)))); continue

        m = re.search(r"#\s*(\d+)\s+is the real\s+([a-z ]+)", says)
        if m:
            medium_real.append(MediumRealClaim(seat, int(m.group(1)), m.group(2).strip())); continue

        m = re.search(r"puppeteer is (\d+) card[s]? away from closest evil", says)
        if m:
            scout_puppeteer_d.append(ScoutPuppeteerDistanceClaim(seat, int(m.group(1)))); continue

        m = re.search(r"i cured\s+(\d+)\s+corruptions?", says)
        if m:
            alchem_cured.append(AlchemistCuredClaim(seat, int(m.group(1)))); continue

        if "i am the original baker" in says:
            baker_original.append(BakerOriginalClaim(seat)); continue

        m = re.search(r"i was a[n]?\s+([a-z ]+)", says)
        if m and "baker" not in m.group(1):
            baker_was.append(BakerWasClaim(seat, m.group(1).strip())); continue

    return knitter_pairs, medium_real, scout_puppeteer_d, alchem_cured, baker_original, baker_was

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
    # Drunk lies and Disguises as a Villager (not necessarily in play).
    for s, true_role in world.items():
        shown = flipped.get(s, "")
        if true_role == "puppet":
            if shown not in VILLAGERS:
                return False
        elif true_role == "drunk":
            if shown not in VILLAGERS:
                return False
        elif is_truthful(true_role):
            if shown != true_role:
                return False
        else:
            # liar (minion) → any shown is fine
            pass
    return True

def check_puppetry(world: Dict[int,str], N: int) -> bool:
    # If a puppeteer exists, there can be at most one puppet adjacent.
    if "puppeteer" not in world.values():
        return True
    puppet_seats = [s for s,r in world.items() if r == "puppet"]
    if len(puppet_seats) > 1: return False
    if len(puppet_seats) == 1:
        p = [s for s,r in world.items() if r == "puppeteer"]
        if not p: return False
        p = p[0]
        n1,n2 = neighbors(p, N)
        if puppet_seats[0] not in (n1,n2): return False
    return True

def invert(world: Dict[int,str]) -> Dict[str,List[int]]:
    d: Dict[str,List[int]] = {}
    for s,r in world.items():
        d.setdefault(r, []).append(s)
    return d

def check_knitter_claims(world: Dict[int,str], knitter_pairs: List[KnitterPairsClaim], N: int) -> bool:
    for claim in knitter_pairs:
        if world.get(claim.seat) != "knitter": 
            # the claim came from a liar; don't enforce
            continue
        if pairs_of_evil(world, N) != claim.pairs: return False
    return True

def check_medium_claims(world: Dict[int,str], medium_real: List[MediumRealClaim]) -> bool:
    for claim in medium_real:
        if world.get(claim.seat) != "medium": 
            # liar; ignore
            continue
        if world.get(claim.target_seat) != claim.role: return False
    return True

def check_scout_claims(world: Dict[int,str], scout_ds: List[ScoutPuppeteerDistanceClaim], N: int) -> bool:
    if not scout_ds: return True
    by = invert(world)
    if "puppeteer" not in by: 
        # If someone said something about Puppeteer distance, they must be lying if there's no puppeteer.
        # Allow lying statements.
        return True
    p = by["puppeteer"][0]
    evil_seats = [s for s,r in world.items() if is_evil(r) and s != p]
    if not evil_seats: return False
    nearest = min(ring_distance(p, e, N) for e in evil_seats)
    for claim in scout_ds:
        if world.get(claim.seat) != "scout": 
            # liar; ignore
            continue
        if nearest != claim.distance: return False
    return True

def check_counsellor_adjacent_outcast(world: Dict[int,str], N:int) -> bool:
    # If a counsellor exists, ensure it sits next to at least one OUTCAST seat.
    cs = [s for s,r in world.items() if r == "counsellor"]
    if not cs: return True
    c = cs[0]
    n1,n2 = neighbors(c, N)
    return world.get(n1) in OUTCASTS or world.get(n2) in OUTCASTS

def check_drunk_not_in_play(world: Dict[int,str], flipped: Dict[int,str]) -> bool:
    # If a seat is drunk and showing X, then X must be a villager role that is NOT actually in play.
    for s,r in world.items():
        if r == "drunk":
            shown = flipped.get(s, "")
            if shown in VILLAGERS and shown not in world.values():
                continue
            else:
                return False
    return True

def check_baker_claims_and_counts(world: Dict[int,str], baker_original: List[BakerOriginalClaim], baker_was: List[BakerWasClaim]) -> bool:
    # Exactly one 'original' baker if someone claimed it (otherwise allow any one).
    baker_seats = [s for s,r in world.items() if r == "baker"]
    if not baker_seats: 
        return False
    if baker_original:
        # All 'original' claims must come from the same seat; and that seat must be a baker.
        orig_claimers = {c.seat for c in baker_original}
        if len(orig_claimers) != 1: 
            return False
        if list(orig_claimers)[0] not in baker_seats:
            return False
    # "I was X" → if truthful baker said it, X is NOT in play (converted away).
    for c in baker_was:
        if world.get(c.seat) == "baker" and is_truthful("baker"):
            if c.was_role in world.values():
                return False
    return True

# ---------- solving ----------
def solve_puzzle1(puz: Puzzle):
    N = puz.seats
    roles = ["knitter","fortune teller","alchemist","wretch","medium","puppeteer","minion","puppet"]
    knit_claims, med_claims, scout_ds, alchem_claims, baker_orig, baker_was = parse_info_log(puz.info_log)

    worlds: List[Dict[int,str]] = []
    for perm in itertools.permutations(roles, N):
        world = {i+1: perm[i] for i in range(N)}
        if not check_flips(world, puz.flipped): continue
        if not check_puppetry(world, N): continue
        if not check_knitter_claims(world, knit_claims, N): continue
        if not check_medium_claims(world, med_claims): continue
        if not check_scout_claims(world, scout_ds, N): continue
        worlds.append(world)
    return worlds

def solve_puzzle2(puz: Puzzle):
    N = puz.seats
    # For puzzle2, the true-role multiset must accommodate multiple Bakers and two Minions.
    multiset = ["druid","plague doctor","drunk","counsellor","shaman","baker","baker","baker"]
    knit_claims, med_claims, scout_ds, alchem_claims, baker_orig, baker_was = parse_info_log(puz.info_log)

    worlds: List[Dict[int,str]] = []
    for perm in set(itertools.permutations(multiset, N)):
        world = {i+1: perm[i] for i in range(N)}
        if not check_flips(world, puz.flipped): continue
        if not check_counsellor_adjacent_outcast(world, N): continue
        if not check_drunk_not_in_play(world, puz.flipped): continue
        if not check_baker_claims_and_counts(world, baker_orig, baker_was): continue
        worlds.append(world)
    return worlds

def choose_solver(puz: Puzzle):
    # Heuristic: if puzzle deck contains 'counsellor' AND 'shaman' AND many 'baker' flips, use puzzle2 solver.
    flip_roles = [r for r in puz.flipped.values()]
    if "counsellor" in puz.deck and "shaman" in puz.deck and flip_roles.count("baker") >= 3:
        return solve_puzzle2
    return solve_puzzle1

# ---------- execution picks & FT advice (unchanged) ----------
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
    all_same = all(s == sets[0] for s in sets) if sets else False
    return inter, uni, all_same

def best_fortune_teller_pairs(worlds: List[Dict[int,str]], N: int, top_k: int = 5):
    # Score each pair (i,j) by worst-case remaining worlds and information gain if FT says "Yes".
    def score_pair(worlds: List[Dict[int,str]], i: int, j: int):
        yes = [w for w in worlds if is_evil(w[i]) or is_evil(w[j])]
        no  = [w for w in worlds if not (is_evil(w[i]) or is_evil(w[j]))]
        total = len(worlds)
        worst = max(len(yes), len(no))
        # IG in bits
        def H(p): 
            return 0 if p == 0 or p == 1 else -(p*math.log2(p)+(1-p)*math.log2(1-p))
        p = len(yes)/total if total else 0
        ig = H(0.5) - H(p)  # relative to 1 bit baseline
        return worst, ig, len(yes), len(no)
    pairs = []
    for i in range(1, N+1):
        for j in range(i+1, N+1):
            worst, ig, yes, no = score_pair(worlds, i, j)
            pairs.append(((i,j), worst, ig, yes, no))
    pairs.sort(key=lambda x: (x[1], -x[2], x[0]))
    return pairs[:top_k], pairs

# ---------- main flow ----------
def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("puzzle", nargs="?", default="puzzle2.yaml")
    args = ap.parse_args(argv)

    puz = load_puzzle(args.puzzle)
    solver = choose_solver(puz)
    worlds = solver(puz)

    if not worlds:
        print("No consistent worlds found.")
        return 2

    N = puz.seats
    print(f"{len(worlds)} consistent world(s).")
    for idx,w in enumerate(worlds[:3], 1):
        print(f"World {idx}: {fmt_world(w)}")

    inter, uni, all_same = evil_sets(worlds)
    if all_same:
        evils = sorted(inter)
        print(f"\nEvils invariant across all worlds → execute seats: {evils[:puz.executions_to_win]}")
        return 0

    best, all_pairs = best_fortune_teller_pairs(worlds, N, top_k=10)
    total = len(worlds)
    any_split = False
    for (i,j), worst, ig, yes, no in best:
        if worst < total:
            any_split = True
            print(f"  ({i},{j}) → worst-case {worst}/{total}, IG≈{ig:.3f} bits, Yes={yes}, No={no}")
    if not any_split:
        print("  (none) — FT queries don't split; execute invariant evil seats if any.")
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
