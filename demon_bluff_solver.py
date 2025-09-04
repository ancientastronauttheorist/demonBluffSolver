import argparse
import itertools
import random
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import yaml


def load_yaml_robust(path: str):
    """Load YAML file, trimming trailing non-YAML lines."""
    with open(path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()
    trimmed = list(lines)
    while trimmed:
        try:
            return yaml.safe_load("\n".join(trimmed))
        except yaml.YAMLError:
            trimmed.pop()
    raise ValueError(f"Failed to parse YAML at {path}")


@dataclass
class Claim:
    seat: int  # 0-indexed
    claimed_role: str
    type: str
    value: object


@dataclass
class World:
    roles: List[str]
    alignments: List[str]
    corrupted: List[bool]
    puppet_seat: Optional[int]


def parse_characters(data) -> Dict[str, Dict]:
    roles = {}
    for entry in data.get("roles", []):
        name = entry.get("name").strip()
        attrs = [a.strip() for a in entry.get("attributes", [])]
        roles[name] = {"attributes": attrs}
    return roles


def parse_info_log(info_log: List[Dict]) -> List[Claim]:
    claims: List[Claim] = []
    KNITTER_RE = re.compile(r"there are (\d+) pairs of evil")
    ALCHEMIST_RE = re.compile(r"i cured (\d+) corruptions")
    MEDIUM_RE = re.compile(r"#(\d+) is the real (\w+)")
    SCOUT_RE = re.compile(r"(\w+) is (\d+) card(?:s)? away from closest evil")
    for item in info_log:
        seat = item["seat"] - 1
        role = item.get("role", "").strip()
        text = item.get("says", "").lower()
        m = KNITTER_RE.fullmatch(text)
        if m:
            claims.append(Claim(seat, role, "knitter_pairs", int(m.group(1))))
            continue
        m = ALCHEMIST_RE.fullmatch(text)
        if m:
            claims.append(Claim(seat, role, "alchemist_cured", int(m.group(1))))
            continue
        m = MEDIUM_RE.fullmatch(text)
        if m:
            t = int(m.group(1)) - 1
            r = m.group(2)
            claims.append(Claim(seat, role, "medium_identity", (t, r)))
            continue
        m = SCOUT_RE.fullmatch(text)
        if m:
            r = m.group(1)
            d = int(m.group(2))
            claims.append(Claim(seat, role, "scout_distance", (r, d)))
            continue
    return claims


def distance(a: int, b: int, N: int) -> int:
    d = abs(a - b)
    return min(d, N - d)


def count_evil_pairs(align: List[str]) -> int:
    N = len(align)
    evil = {"minion", "demon", "puppet"}
    cnt = 0
    for i in range(N):
        j = (i + 1) % N
        if align[i] in evil and align[j] in evil:
            cnt += 1
    return cnt


def corrupted_in_range(world: World, seat: int, r: int) -> int:
    N = len(world.roles)
    count = 0
    for i in range(seat - r, seat + r + 1):
        j = i % N
        if world.corrupted[j]:
            count += 1
    return count


def apply_claims(world: World, claims: List[Claim], roles_info: Dict[str, Dict], truth_map: Dict[str, bool]) -> bool:
    N = len(world.roles)
    evil_align = {"minion", "demon", "puppet"}
    good_align = {"villager", "outcast"}
    for cl in claims:
        s = cl.seat
        role_actual = world.roles[s]
        alignment = world.alignments[s]
        truthful = truth_map.get(alignment, False)
        if world.corrupted[s]:
            truthful = False
        if truthful and role_actual != cl.claimed_role:
            return False
        if role_actual != cl.claimed_role:
            continue  # can't evaluate
        if cl.type == "knitter_pairs":
            pc = count_evil_pairs(world.alignments)
            if truthful:
                if pc != cl.value:
                    return False
            else:
                if pc == cl.value:
                    return False
        elif cl.type == "alchemist_cured":
            cnt = corrupted_in_range(world, s, 2)
            if truthful:
                if cnt != cl.value:
                    return False
            else:
                if cnt == cl.value:
                    return False
        elif cl.type == "medium_identity":
            t, role_t = cl.value
            cond = world.roles[t] == role_t and world.alignments[t] in good_align
            if truthful:
                if not cond:
                    return False
            else:
                if cond:
                    return False
        elif cl.type == "scout_distance":
            role_t, dist_needed = cl.value
            if role_t in world.roles:
                seat_t = world.roles.index(role_t)
                distances = [distance(seat_t, j, N) for j in range(N) if j != seat_t and world.alignments[j] in evil_align]
                d = min(distances) if distances else None
            else:
                d = None
            if truthful:
                if d is None or d != dist_needed:
                    return False
            else:
                if d is not None and d == dist_needed:
                    return False
    return True


def enumerate_worlds(puzzle, roles_info, claims, rules):
    N = puzzle["seats"]
    deck = list(puzzle["deck"])
    truth_map = rules["terms"]["default_truth_by_alignment"].copy()
    # Puppets speak truthfully unless corrupted or rules override
    truth_map["puppet"] = True
    flipped_claims = {int(k) - 1: v.strip() for k, v in puzzle.get("flipped", {}).items()}
    # Optional requirement on total evils
    evils_required = puzzle.get("evils_in_play")
    if evils_required is None:
        evils_required = puzzle.get("options", {}).get("executions_to_win")

    worlds: List[World] = []
    for perm in itertools.permutations(deck, N):
        roles = list(perm)
        puppet_variants: List[Tuple[List[str], Optional[int], List[str]]] = []
        if "puppeteer" in roles:
            p = roles.index("puppeteer")
            neighbors = [(p - 1) % N, (p + 1) % N]
            options = []
            for n in neighbors:
                r = roles[n]
                attrs = roles_info.get(r, {}).get("attributes", [])
                if "villager" in attrs and "good" in attrs:
                    options.append(n)
            if options:
                for n in options:
                    align = []
                    for idx, r in enumerate(roles):
                        if idx == n:
                            align.append("puppet")
                        else:
                            attrs = roles_info.get(r, {}).get("attributes", [])
                            if "minion" in attrs:
                                align.append("minion")
                            elif "demon" in attrs:
                                align.append("demon")
                            elif "outcast" in attrs:
                                align.append("outcast")
                            else:
                                align.append("villager")
                    puppet_variants.append((roles, n, align))
            else:
                align = []
                for r in roles:
                    attrs = roles_info.get(r, {}).get("attributes", [])
                    if "minion" in attrs:
                        align.append("minion")
                    elif "demon" in attrs:
                        align.append("demon")
                    elif "outcast" in attrs:
                        align.append("outcast")
                    else:
                        align.append("villager")
                puppet_variants.append((roles, None, align))
        else:
            align = []
            for r in roles:
                attrs = roles_info.get(r, {}).get("attributes", [])
                if "minion" in attrs:
                    align.append("minion")
                elif "demon" in attrs:
                    align.append("demon")
                elif "outcast" in attrs:
                    align.append("outcast")
                else:
                    align.append("villager")
            puppet_variants.append((roles, None, align))

        for roles_list, puppet_seat, align in puppet_variants:
            candidates = [i for i, a in enumerate(align) if a == "villager"]
            if not candidates:
                continue
            # Allow corruption only if a role capable of causing it is present
            corruption_roles = {"plague doctor", "poisoner", "pooka"}
            if any(r in corruption_roles for r in roles_list):
                corrupt_options = [None] + candidates
            else:
                corrupt_options = [None]
            for corrupt in corrupt_options:
                corrupted = [False] * N
                if corrupt is not None:
                    corrupted[corrupt] = True
                world = World(list(roles_list), list(align), corrupted, puppet_seat)
                ok = True
                for seat_idx, claimed_role in flipped_claims.items():
                    alignment = world.alignments[seat_idx]
                    truthful = truth_map.get(alignment, False)
                    if corrupted[seat_idx]:
                        truthful = False
                    if truthful and world.roles[seat_idx] != claimed_role:
                        ok = False
                        break
                if ok and apply_claims(world, claims, roles_info, truth_map):
                    if evils_required is not None:
                        evil_count = sum(1 for a in world.alignments if a in {"minion", "demon", "puppet"})
                        if evil_count != evils_required:
                            continue
                    worlds.append(world)
    return worlds


def action_advisor(worlds: List[World], N: int, claims: List[Claim], max_target_combos: int):
    used_fortune = any(cl.claimed_role == "fortune teller" for cl in claims)
    roles_present = {r for w in worlds for r in w.roles}
    actions = []
    if not used_fortune and "fortune teller" in roles_present:
        evil_align = {"minion", "demon", "puppet"}
        pairs = list(itertools.combinations(range(N), 2))[:max_target_combos]
        total = len(worlds)
        for i, j in pairs:
            counts = {True: 0, False: 0}
            for w in worlds:
                outcome = (w.alignments[i] in evil_align) or (w.alignments[j] in evil_align)
                counts[outcome] += 1
            partitions = [counts[True], counts[False]]
            max_part = max(partitions)
            worst = 1 - max_part / total
            expected = 1 - sum(c * c for c in partitions) / (total * total)
            actions.append({
                "ability": "fortune teller",
                "targets": (i + 1, j + 1),
                "counts": counts,
                "worst": worst,
                "expected": expected,
            })
        actions.sort(key=lambda a: (-a["worst"], -a["expected"], a["targets"]))
    return actions[:5]


def main():
    parser = argparse.ArgumentParser(description="Demon Bluff Solver")
    parser.add_argument("puzzle")
    parser.add_argument("--rules", required=True)
    parser.add_argument("--chars", required=True)
    parser.add_argument("--advise", action="store_true")
    parser.add_argument("--explain", action="store_true")
    parser.add_argument("--all-worlds", action="store_true")
    parser.add_argument("--max-target-combos", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)

    rules = load_yaml_robust(args.rules)
    chars = load_yaml_robust(args.chars)
    puzzle = load_yaml_robust(args.puzzle)

    roles_info = parse_characters(chars)
    claims = parse_info_log(puzzle.get("info_log", []))

    worlds = enumerate_worlds(puzzle, roles_info, claims, rules)
    print(f"Consistent worlds: {len(worlds)}")
    if args.explain and worlds:
        sample = worlds[0]
        mapping = [f"{i+1}:{r}[{a}{',C' if c else ''}]" for i, (r, a, c) in enumerate(zip(sample.roles, sample.alignments, sample.corrupted))]
        print("Sample world:", ", ".join(mapping))
    if args.all_worlds:
        for idx, w in enumerate(worlds[:50]):
            mapping = [f"{i+1}:{r}[{a}{',C' if c else ''}]" for i, (r, a, c) in enumerate(zip(w.roles, w.alignments, w.corrupted))]
            print(f"World {idx+1}: ", ", ".join(mapping))
        if len(worlds) > 50:
            print("...")
    if args.advise and worlds:
        actions = action_advisor(worlds, puzzle["seats"], claims, args.max_target_combos)
        if not actions:
            print("No available actions to advise.")
        else:
            print("Action Advisor:")
            for act in actions:
                (i, j) = act["targets"]
                counts = act["counts"]
                worst = act["worst"] * 100
                expected = act["expected"] * 100
                print(f" Fortune Teller -> seats {i} & {j}: outcomes {counts}, worst-case -{worst:.1f}% exp -{expected:.1f}%")


if __name__ == "__main__":
    main()
