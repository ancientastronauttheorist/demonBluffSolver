import json
import re

# Load character definitions
with open('characters.json') as f:
    CHARACTERS = {r['name']: r for r in json.load(f)['roles']}

COUNTS_AS_EVIL = {'minion', 'demon', 'puppet'}

def alignment(role: str) -> str:
    if role == 'puppet':
        return 'puppet'
    attrs = CHARACTERS[role]['attributes']
    if 'demon' in attrs:
        return 'minion'
    if 'minion' in attrs:
        return 'minion'
    if 'outcast' in attrs:
        return 'outcast'
    return 'good'

def is_villager(role: str) -> bool:
    return 'villager' in CHARACTERS[role]['attributes']

def can_be_corrupted(role: str) -> bool:
    return role != 'puppet'

def cannot_be_cured(role: str) -> bool:
    return role == 'drunk'

def can_disguise(role: str) -> bool:
    return alignment(role) in {'minion', 'demon'} or role in {'puppet', 'doppelganger', 'drunk'}

def is_truthful(role: str, corrupted: bool) -> bool:
    if role == 'puppet':
        return True
    if role == 'confessor':
        return True
    if corrupted:
        return False
    return alignment(role) not in {'minion', 'demon'} and role != 'drunk'

def is_evil_for_observer(role: str) -> bool:
    if role == 'wretch':
        return True
    return alignment(role) in COUNTS_AS_EVIL

def neighbors(seat: int, seats: int):
    left = seat - 1 if seat > 1 else seats
    right = seat + 1 if seat < seats else 1
    return left, right

def range2(seat: int, seats: int):
    res = []
    for d in (-2, -1, 1, 2):
        idx = ((seat - 1 + d) % seats) + 1
        res.append(idx)
    return res

def count_adjacent_evil_pairs(roles: dict, seats: int) -> int:
    count = 0
    for i in range(1, seats + 1):
        j = i + 1 if i < seats else 1
        if is_evil_for_observer(roles[i]) and is_evil_for_observer(roles[j]):
            count += 1
    return count

def nearest_evil_distance(roles: dict, seat: int, seats: int) -> int:
    evil_seats = [i for i in range(1, seats + 1) if i != seat and is_evil_for_observer(roles[i])]
    if not evil_seats:
        return None
    dists = [min(abs(seat - e), seats - abs(seat - e)) for e in evil_seats]
    return min(dists)

def count_cures_for_seat(seat: int, roles: dict, corrupted: set, seats: int) -> int:
    total = 0
    for j in range2(seat, seats):
        r = roles[j]
        if j in corrupted and is_villager(r) and not cannot_be_cured(r):
            total += 1
    return total

def simulate_corruption(roles: dict, seats: int):
    base = set()
    # Pooka
    for i, r in roles.items():
        if r == 'pooka':
            for n in neighbors(i, seats):
                if is_villager(roles[n]) and can_be_corrupted(roles[n]):
                    base.add(n)
    scenarios = [base]
    # Poisoner
    for i, r in roles.items():
        if r == 'poisoner':
            new = []
            for s in scenarios:
                raw_targets = [n for n in neighbors(i, seats) if is_villager(roles[n]) and can_be_corrupted(roles[n])]
                targets = [n for n in raw_targets if n not in s]
                if not targets:
                    if raw_targets:
                        continue  # invalid: had targets but all already corrupted
                    else:
                        new.append(set(s))
                else:
                    for t in targets:
                        ns = set(s)
                        ns.add(t)
                        new.append(ns)
            scenarios = new
    # Plague doctor
    for i, r in roles.items():
        if r == 'plague doctor':
            targets = [j for j in range(1, seats + 1) if is_villager(roles[j]) and alignment(roles[j]) == 'good' and can_be_corrupted(roles[j])]
            new = []
            if targets:
                for t in targets:
                    for s in scenarios:
                        ns = set(s)
                        ns.add(t)
                        new.append(ns)
                scenarios = new
    for corr in scenarios:
        # cure counts before cures
        cure_counts = {i: count_cures_for_seat(i, roles, corr, seats) for i in range(1, seats + 1)}
        final_corr = set(corr)
        for i, r in roles.items():
            if r == 'alchemist' and i not in corr:
                for j in range2(i, seats):
                    if j in final_corr and is_villager(roles[j]) and not cannot_be_cured(roles[j]):
                        final_corr.remove(j)
        yield final_corr, cure_counts

def evaluate_statements(puzzle, roles, corrupted_set, cure_counts):
    seats = puzzle['seats']
    flipped = puzzle['flipped']
    for i in range(1, seats + 1):
        role = roles[i]
        display = flipped[str(i)]['role']
        says = flipped[str(i)].get('says')
        corrupted = i in corrupted_set
        truth = is_truthful(role, corrupted)
        if display == 'confessor':
            said_dizzy = (says or '').strip().lower() == 'i am dizzy'
            cond = alignment(role) in {'minion', 'demon'} or corrupted
            if said_dizzy != cond:
                return False
            continue
        if not says:
            continue
        text = says.lower()
        if display == 'knitter':
            m = re.search(r'(\d+)', text)
            if not m:
                return False
            said = int(m.group(1))
            actual = count_adjacent_evil_pairs(roles, seats)
            cond = (said == actual)
            if cond != truth:
                return False
        elif display == 'alchemist':
            m = re.search(r'(\d+)', text)
            if not m:
                return False
            said = int(m.group(1))
            actual = cure_counts[i]
            cond = (said == actual)
            if cond != truth:
                return False
        elif display == 'medium':
            m = re.search(r'#(\d+) is (?:a|the) real ([a-z ]+)', text)
            if not m:
                return False
            tgt = int(m.group(1))
            role_name = m.group(2).strip()
            cond = roles[tgt] == role_name
            if cond != truth:
                return False
        elif display == 'scout':
            m = re.search(r'([a-z ]+) is (\d+) card', text)
            if not m:
                return False
            role_name = m.group(1).strip()
            dist_claim = int(m.group(2))
            seat_target = None
            for j in range(1, seats + 1):
                if roles[j] == role_name:
                    seat_target = j
                    break
            if seat_target is None:
                return False
            dist = nearest_evil_distance(roles, seat_target, seats)
            cond = dist == dist_claim
            if cond != truth:
                return False
        elif display == 'oracle':
            m = re.search(r'#(\d+) or #(\d+) is a ([a-z ]+)', text)
            if not m:
                return False
            a = int(m.group(1))
            b = int(m.group(2))
            role_name = m.group(3).strip()
            if truth:
                cond = (roles[a] == role_name) ^ (roles[b] == role_name)
                if not cond:
                    return False
            else:
                if not (alignment(roles[a]) == 'good' and alignment(roles[b]) == 'good'):
                    return False
        else:
            return False
    return True

def check_global_constraints(puzzle, roles):
    seats = puzzle['seats']
    flipped = puzzle['flipped']
    puppeteer_seats = [i for i, r in roles.items() if r == 'puppeteer']
    puppet_seats = [i for i, r in roles.items() if r == 'puppet']
    if puppeteer_seats:
        if len(puppet_seats) != 1:
            return False
        pupt = puppeteer_seats[0]
        pup = puppet_seats[0]
        if min(abs(pupt - pup), seats - abs(pupt - pup)) != 1:
            return False
        disp = flipped[str(pup)]['role']
        if not is_villager(disp):
            return False
    else:
        if puppet_seats:
            return False
    # doppelganger check
    for i, r in roles.items():
        if r == 'doppelganger':
            disp = flipped[str(i)]['role']
            if alignment(disp) != 'good':
                return False
            if not any(j != i and roles[j] == disp for j in range(1, seats + 1)):
                return False
    return True

def solve_puzzle(puzzle_path: str):
    with open(puzzle_path) as f:
        puzzle = json.load(f)
    seats = puzzle['seats']
    flipped = puzzle['flipped']
    deck = puzzle['deck']
    roles_available = deck[:]
    if 'puppeteer' in deck:
        roles_available.append('puppet')

    best = None

    def backtrack(seat_idx, used, assignment):
        nonlocal best
        if best is not None:
            return
        if seat_idx > seats:
            if not check_global_constraints(puzzle, assignment):
                return
            for final_corr, cure_counts in simulate_corruption(assignment, seats):
                if evaluate_statements(puzzle, assignment, final_corr, cure_counts):
                    best = (assignment.copy(), final_corr)
                    break
            return
        disp = flipped[str(seat_idx)]['role']
        for role in roles_available:
            if role in used:
                continue
            if role != disp and not can_disguise(role):
                continue
            assignment[seat_idx] = role
            used.add(role)
            backtrack(seat_idx + 1, used, assignment)
            used.remove(role)
            del assignment[seat_idx]

    backtrack(1, set(), {})
    if best is None:
        raise ValueError('No solution found')
    roles, corrupted = best
    solution = {
        'seats': seats,
        'evils_in_play': sum(1 for r in roles.values() if alignment(r) in COUNTS_AS_EVIL),
        'solution': {
            str(i): {
                'role': roles[i],
                'alignment': alignment(roles[i]),
                'corrupted': i in corrupted
            } for i in range(1, seats + 1)
        }
    }
    return solution

def main():
    import sys
    for path in sys.argv[1:]:
        sol = solve_puzzle(path)
        print(json.dumps(sol, indent=2))

if __name__ == '__main__':
    main()
