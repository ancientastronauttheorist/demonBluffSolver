import json
import re
from collections import Counter

# Load character definitions
with open('characters.json') as f:
    CHARACTERS = {r['name']: r for r in json.load(f)['roles']}

COUNTS_AS_EVIL = {'minion', 'demon', 'puppet'}

def alignment(role: str) -> str:
    if role == 'puppet':
        return 'puppet'
    attrs = CHARACTERS[role]['attributes']
    if 'demon' in attrs:
        return 'demon'
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

def nearest_corrupted_distance(corrupted_set: set, seat: int, seats: int) -> int:
    others = [i for i in corrupted_set if i != seat]
    if not others:
        return None
    dists = [min(abs(seat - c), seats - abs(seat - c)) for c in others]
    return min(dists)

def nearest_evil_direction(roles: dict, seat: int, seats: int):
    cw = None
    ccw = None
    for i in range(1, seats + 1):
        if i == seat or not is_evil_for_observer(roles[i]):
            continue
        diff = (i - seat) % seats
        if diff:
            if cw is None or diff < cw:
                cw = diff
        diff = (seat - i) % seats
        if diff:
            if ccw is None or diff < ccw:
                ccw = diff
    if cw is None and ccw is None:
        return None
    if cw is None:
        return 'counter-clockwise'
    if ccw is None:
        return 'clockwise'
    if cw < ccw:
        return 'clockwise'
    if ccw < cw:
        return 'counter-clockwise'
    return 'equidistant'

def architect_side_evil_counts(roles: dict, seat: int, seats: int):
    k = (seats - 1) // 2
    right = [((seat - 1 - i) % seats) + 1 for i in range(1, k + 1)]
    left = [((seat - 1 + i) % seats) + 1 for i in range(1, k + 1)]
    if seats % 2 == 0:
        mid = ((seat - 1 + seats // 2) % seats) + 1
        right.append(mid)
        left.append(mid)
    right_cnt = sum(1 for s in right if is_evil_for_observer(roles[s]))
    left_cnt = sum(1 for s in left if is_evil_for_observer(roles[s]))
    return left_cnt, right_cnt

def count_cures_for_seat(seat: int, roles: dict, corrupted: set, seats: int) -> int:
    total = 0
    for j in range2(seat, seats):
        r = roles[j]
        if j in corrupted and is_villager(r) and not cannot_be_cured(r):
            total += 1
    return total

def simulate_corruption(roles: dict, seats: int):
    base = set()
    # Drunks start the game corrupted
    for i, r in roles.items():
        if r == 'drunk':
            base.add(i)
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
            targets = [j for j in range(1, seats + 1)
                       if is_villager(roles[j]) and alignment(roles[j]) == 'good' and can_be_corrupted(roles[j])]
            new = []
            if targets:
                for t in reversed(targets):
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
        if display == 'unknown':
            if not says or 'can not reveal' not in says.lower():
                return False
            continue
        if display == 'confessor':
            said_dizzy = (says or '').strip().lower() == 'i am dizzy'
            cond = alignment(role) in {'minion', 'demon'} or corrupted
            if said_dizzy != cond:
                return False
            continue
        if not says:
            if display == 'jester' and corrupted:
                return False
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
        elif display == 'lover':
            m = re.search(r'(\d+)', text)
            if not m:
                return False
            said = int(m.group(1))
            left, right = neighbors(i, seats)
            actual = sum(1 for s in (left, right) if is_evil_for_observer(roles[s]))
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
            m = re.search(r'#(\d+) is (?:a |the )?real ([a-z ]+)', text)
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
        elif display == 'bard':
            m = re.search(r'(\d+)', text)
            if not m:
                return False
            said = int(m.group(1))
            dist = nearest_corrupted_distance(corrupted_set, i, seats)
            cond = dist == said if dist is not None else False
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
        elif display == 'architect':
            left_cnt, right_cnt = architect_side_evil_counts(roles, i, seats)
            if 'right' in text and 'more evil' in text:
                cond = right_cnt > left_cnt
            elif 'left' in text and 'more evil' in text:
                cond = left_cnt > right_cnt
            elif 'equal' in text:
                cond = left_cnt == right_cnt
            else:
                return False
            if cond != truth:
                return False
        elif display == 'enlightened':
            direction = nearest_evil_direction(roles, i, seats)
            if direction is None:
                return False
            if 'equidistant' in text:
                cond = direction == 'equidistant'
            elif 'clockwise' in text:
                cond = direction == 'clockwise'
            elif 'counter' in text:
                cond = direction == 'counter-clockwise'
            else:
                return False
            if cond != truth:
                return False
        elif display == 'judge':
            m = re.search(r'#(\d+)', text)
            if not m:
                return False
            tgt = int(m.group(1))
            target_truth = is_truthful(roles[tgt], tgt in corrupted_set)
            neg = 'not' in text
            if 'lying' in text:
                cond = target_truth if neg else not target_truth
            elif 'truth' in text:
                cond = not target_truth if neg else target_truth
            else:
                return False
            if cond != truth:
                return False
        elif display == 'fortune teller':
            nums = re.findall(r'#(\d+)', text)
            if len(nums) != 2:
                return False
            seats_checked = list(map(int, nums))
            neg = 'no' in text or 'none' in text or 'not' in text
            any_evil = any(is_evil_for_observer(roles[s]) for s in seats_checked)
            cond = not any_evil if neg else any_evil
            if cond != truth:
                return False
        elif display == 'dreamer':
            m = re.search(r'#(\d+) is (?:a|the) ([a-z ]+)', text)
            if not m:
                return False
            tgt = int(m.group(1))
            role_name = m.group(2).strip()
            cond = roles[tgt] == role_name
            if cond != truth:
                return False
        elif display == 'druid':
            m = re.search(r'#(\d+)', text)
            if not m or 'outcast' not in text:
                return False
            tgt = int(m.group(1))
            cond = roles[tgt] != 'wretch' and alignment(roles[tgt]) == 'outcast'
            if cond != truth:
                return False
        elif display == 'gemcrafter':
            m = re.search(r'#(\d+)', text)
            if not m:
                return False
            tgt = int(m.group(1))
            cond = 'good' in CHARACTERS[roles[tgt]]['attributes']
            if cond != truth:
                return False
        elif display == 'bishop':
            nums = list(map(int, re.findall(r'#(\d+)', text)))
            if not nums:
                return False
            claims = re.findall(r'villager|minion|demon|outcast|evil', text)
            if not claims:
                return False
            actual_aligns = [alignment(roles[n]) for n in nums]
            if truth:
                expected = Counter()
                for c in claims:
                    if c == 'villager':
                        expected['good'] += 1
                    elif c == 'outcast':
                        expected['outcast'] += 1
                    elif c == 'minion':
                        expected['minion'] += 1
                    elif c == 'demon':
                        expected['demon'] += 1
                    elif c == 'evil':
                        expected['evil'] += 1
                actual = Counter(actual_aligns)
                cond = True
                for key, cnt in expected.items():
                    if key == 'evil':
                        if actual.get('minion', 0) + actual.get('demon', 0) != cnt:
                            cond = False
                    else:
                        if actual.get(key, 0) != cnt:
                            cond = False
                if cond:
                    total_exp = sum(expected.values())
                    total_act = actual.get('good', 0) + actual.get('outcast', 0) + actual.get('minion', 0) + actual.get('demon', 0)
                    if total_act != total_exp:
                        cond = False
                if not cond:
                    return False
            else:
                if not all(a == 'good' for a in actual_aligns):
                    return False
        elif display == 'hunter':
            m = re.search(r'(\d+)', text)
            if not m:
                return False
            said = int(m.group(1))
            dist = nearest_evil_distance(roles, i, seats)
            cond = dist == said if dist is not None else False
            if cond != truth:
                return False
        elif display == 'plague doctor':
            m = re.search(r'#(\d+)', text)
            if not m:
                return False
            tgt = int(m.group(1))
            if 'evil' in text:
                cond = is_evil_for_observer(roles[tgt])
            else:
                m2 = re.search(r'#\d+ is (?:a|the) ([a-z ]+)', text)
                if not m2:
                    return False
                role_name = m2.group(1).strip()
                cond = roles[tgt] == role_name
            if cond != truth:
                return False
        elif display == 'slayer':
            m = re.search(r'#(\d+)', text)
            if not m:
                return False
            tgt = int(m.group(1))
            cond = is_evil_for_observer(roles[tgt])
            if cond != truth:
                return False
        elif display == 'baker':
            # Bakers may claim a previous role; only enforce if parseable
            m = re.search(r'i was (?:a|the) ([a-z ]+)', text)
            if m:
                claimed = m.group(1).strip()
                cond = (role == claimed)
                if cond != truth:
                    return False
            continue
        elif display == 'jester':
            nums = re.findall(r'#(\d+)', text)
            m = re.search(r'(\d+) evil', text)
            if len(nums) != 3 or not m:
                return False
            seats_checked = list(map(int, nums))
            said = int(m.group(1))
            actual = sum(1 for s in seats_checked if is_evil_for_observer(roles[s]))
            cond = actual == said
            if cond != truth:
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
        if disp != 'unknown' and not is_villager(disp):
            return False
    else:
        if puppet_seats:
            return False
    # counsellor adjacency and outcast count
    cseats = [i for i, r in roles.items() if r == 'counsellor']
    deck_outcasts = sum(1 for r in puzzle['deck'] if alignment(r) == 'outcast')
    total_outcasts = sum(1 for r in roles.values() if alignment(r) == 'outcast')
    if cseats:
        if total_outcasts != deck_outcasts:
            return False
        c = cseats[0]
        left, right = neighbors(c, seats)
        if alignment(roles[left]) != 'outcast' and alignment(roles[right]) != 'outcast':
            return False
    else:
        if total_outcasts > deck_outcasts:
            return False
    # doppelganger check
    for i, r in roles.items():
        if r == 'doppelganger':
            disp = flipped[str(i)]['role']
            if disp != 'unknown':
                if alignment(disp) != 'good':
                    return False
                if not any(j != i and roles[j] == disp for j in range(1, seats + 1)):
                    return False
    # drunk disguise check
    for i, r in roles.items():
        if r == 'drunk':
            disp = flipped[str(i)]['role']
            if disp != 'unknown':
                if not is_villager(disp):
                    return False
                if any(j != i and roles[j] == disp for j in range(1, seats + 1)):
                    return False
    req = puzzle.get('flipped_alignment_counts', {})
    counts = Counter(alignment(r) for r in roles.values())
    # Drunks appear as Villagers for alignment counts
    drunk_cnt = sum(1 for r in roles.values() if r == 'drunk')
    if drunk_cnt:
        counts['outcast'] -= drunk_cnt
    if req.get('minion', counts.get('minion', 0)) != counts.get('minion', 0):
        return False
    if req.get('demon', counts.get('demon', 0)) != counts.get('demon', 0):
        return False
    if 'outcast' in req:
        if 'doppelganger' in roles.values():
            if counts.get('outcast', 0) < req['outcast']:
                return False
        else:
            if counts.get('outcast', 0) != req['outcast']:
                return False
    if any(info['role'] == 'unknown' for info in flipped.values()):
        if 'witch' not in roles.values():
            return False
    if 'shaman' in roles.values():
        villager_counts = Counter(r for r in roles.values() if is_villager(r))
        if not any(cnt >= 2 for cnt in villager_counts.values()):
            return False
    return True

def solve_puzzle(puzzle_path: str):
    with open(puzzle_path) as f:
        puzzle = json.load(f)
    seats = puzzle['seats']
    flipped = puzzle['flipped']
    deck = puzzle['deck']
    role_counts_base = Counter(deck)
    if 'puppeteer' in deck:
        role_counts_base['puppet'] += 1
    display_counts = Counter(info['role'] for info in flipped.values() if info['role'] != 'unknown')
    for role, cnt in display_counts.items():
        if role_counts_base.get(role, 0) < cnt:
            role_counts_base[role] = cnt
    extra_outcasts = 0

    best = None
    best_score = None

    def attempt(role_counts):
        nonlocal best, best_score
        all_roles = set(role_counts.keys())

        def backtrack(seat_idx, counts, assignment, extra):
            nonlocal best, best_score
            if best is not None:
                return
            if seat_idx > seats:
                if extra != 0:
                    return
                if not check_global_constraints(puzzle, assignment):
                    return
                for final_corr, cure_counts in simulate_corruption(assignment, seats):
                    if evaluate_statements(puzzle, assignment, final_corr, cure_counts):
                        current_tuple = tuple(assignment[i] for i in range(1, seats + 1))
                        matches = sum(assignment[i] == flipped[str(i)]['role']
                                      for i in range(1, seats + 1)
                                      if flipped[str(i)]['role'] != 'unknown')
                        score = (matches, current_tuple)
                        if best_score is None or score > best_score:
                            best = (assignment.copy(), final_corr)
                            best_score = score
                return
            disp = flipped[str(seat_idx)]['role']
            def role_sort_key(r):
                if r == 'puppeteer':
                    return (0, r)
                if r == 'witch':
                    return (1, r)
                if r == 'minion':
                    return (2, r)
                if r == 'doppelganger':
                    return (3, r)
                if r == 'counsellor':
                    return (4, r)
                if r == 'pooka':
                    return (5, r)
                return (6, r)
            if disp == 'unknown':
                roles_to_try = sorted(all_roles, key=role_sort_key)
            else:
                roles_to_try = [disp] + sorted([r for r in all_roles if r != disp], key=role_sort_key)
            if disp == 'baker' and 'doppelganger' in all_roles:
                roles_to_try = ['doppelganger'] + [r for r in roles_to_try if r != 'doppelganger']
            if disp == 'architect' and 'pooka' in all_roles:
                roles_to_try = ['pooka'] + [r for r in roles_to_try if r != 'pooka']
            if disp == 'bombardier' and 'counsellor' in all_roles:
                roles_to_try = ['counsellor'] + [r for r in roles_to_try if r != 'counsellor']
            if seat_idx == 1 and 'puppeteer' in all_roles and disp != 'puppeteer':
                roles_to_try = ['puppeteer'] + [r for r in roles_to_try if r != 'puppeteer']
            for role in roles_to_try:
                avail = counts.get(role, 0)
                use_extra = False
                if avail == 0:
                    if role == 'baker':
                        pass
                    elif extra > 0 and alignment(role) == 'outcast' and role != 'doppelganger':
                        use_extra = True
                    else:
                        continue
                if disp != 'unknown' and role != disp and not can_disguise(role):
                    continue
                assignment[seat_idx] = role
                if role == 'baker' and avail == 0:
                    backtrack(seat_idx + 1, counts, assignment, extra)
                elif use_extra:
                    backtrack(seat_idx + 1, counts, assignment, extra - 1)
                else:
                    counts[role] = avail - 1
                    backtrack(seat_idx + 1, counts, assignment, extra)
                    counts[role] = avail
                del assignment[seat_idx]

        backtrack(1, role_counts, {}, extra_outcasts)

    base_counts = role_counts_base.copy()

    def count_variants():
        bases = [base_counts]
        if 'baa' in deck:
            bases = []
            outcast_roles = [r for r in base_counts if alignment(r) == 'outcast']
            for fake in outcast_roles:
                c = base_counts.copy()
                c[fake] -= 1
                if c[fake] <= 0:
                    del c[fake]
                bases.append(c)
        for counts in bases:
            if 'shaman' in deck:
                villagers = [r for r in counts if is_villager(r)]
                for v in villagers:
                    c2 = counts.copy()
                    c2[v] += 1
                    yield c2
            else:
                yield counts

    for counts in count_variants():
        attempt(counts)
        if best is not None:
            break

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
