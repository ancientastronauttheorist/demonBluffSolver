"""Demon Bluff constraint-satisfaction solver.

Enumerates all possible Evil placements, filters by info consistency,
reports positions that are Evil in ALL surviving scenarios.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from itertools import combinations, product
from typing import Optional
from knowledge_base import get_card, Role, Alignment, CARDS_BY_NAME


# ============================================================
# Circle Geometry
# ============================================================

def circle_distance(a: int, b: int, n: int) -> int:
    """Shortest distance between positions a and b on a circle of size n.
    Positions are 1-indexed."""
    diff = abs(a - b)
    return min(diff, n - diff)


def circle_direction(from_pos: int, to_pos: int, n: int) -> str:
    """Direction from from_pos to to_pos on a circle (CW or CCW).
    Returns 'Equidistant' if exactly opposite. Positions 1-indexed.
    CW means increasing position numbers (1->2->3...)."""
    if from_pos == to_pos:
        return "Equidistant"
    cw_dist = (to_pos - from_pos) % n
    ccw_dist = (from_pos - to_pos) % n
    if cw_dist < ccw_dist:
        return "CW"
    elif ccw_dist < cw_dist:
        return "CCW"
    else:
        return "Equidistant"


def adjacent_positions(pos: int, n: int) -> list[int]:
    """Return the two positions adjacent to pos on a circle of size n. 1-indexed."""
    left = ((pos - 2) % n) + 1
    right = (pos % n) + 1
    return [left, right]


def positions_in_range(pos: int, range_val: int, n: int) -> list[int]:
    """Return all positions within range_val steps of pos (excluding pos itself)."""
    result = []
    for d in range(1, range_val + 1):
        cw = ((pos - 1 + d) % n) + 1
        ccw = ((pos - 1 - d) % n) + 1
        result.append(cw)
        if cw != ccw:
            result.append(ccw)
    return result


# ============================================================
# Data Model
# ============================================================

class TruthStatus(Enum):
    TRUTHFUL = "truthful"
    LYING = "lying"


@dataclass
class CardInfo:
    """A revealed card's info as seen in the game."""
    position: int           # 1-indexed position in circle
    apparent_role: str      # What role it appears as (may be disguise)
    info_text: str = ""     # Raw info text from the card
    info_parsed: dict = field(default_factory=dict)  # Structured info (type-specific)


@dataclass
class DeckComposition:
    """Roles known to be in play (from deck view)."""
    villagers: list[str]    # e.g. ["Enlightened", "Knitter", "Confessor"]
    outcasts: list[str]     # e.g. ["Plague Doctor"]
    minions: list[str]      # e.g. ["Puppeteer"]
    demons: list[str]       # e.g. ["Pooka"]

    @property
    def evil_roles(self) -> list[str]:
        return self.minions + self.demons

    @property
    def all_roles(self) -> list[str]:
        return self.villagers + self.outcasts + self.minions + self.demons


@dataclass
class GameState:
    """Full state of a game in progress."""
    n_cards: int                    # Total cards in circle
    deck: DeckComposition
    cards: list[CardInfo]           # Revealed cards (may not be all)
    n_evil: int = 0                 # Total evil to find
    executed: list[int] = field(default_factory=list)  # Already executed positions
    confirmed_evil: list[int] = field(default_factory=list)
    confirmed_good: list[int] = field(default_factory=list)
    pd_corruption_target: Optional[int] = None  # If PD target is known


@dataclass
class Scenario:
    """A hypothetical assignment of evil roles to positions."""
    evil_positions: dict[int, str]  # pos -> evil role name
    puppet_position: Optional[int] = None  # If Puppeteer in play
    corrupted: set[int] = field(default_factory=set)  # Corrupted positions
    pd_corrupted: Optional[int] = None  # Plague Doctor corruption target


@dataclass
class SolverResult:
    """Output of the solver."""
    definite_evil: list[int]        # Evil in ALL surviving scenarios
    definite_good: list[int]        # Good in ALL surviving scenarios
    bombardier_positions: list[int] # Never execute these
    n_scenarios: int                # Total scenarios checked
    n_surviving: int                # Scenarios that passed all checks
    surviving_scenarios: list[Scenario] = field(default_factory=list)
    reasoning: list[str] = field(default_factory=list)


# ============================================================
# Scenario Generation
# ============================================================

def _generate_evil_placements(state: GameState) -> list[dict[int, str]]:
    """Generate all ways to assign evil roles to card positions."""
    n = state.n_cards
    evil_roles = state.deck.evil_roles[:]
    all_positions = list(range(1, n + 1))

    # Remove already-executed evil roles from what we need to place
    # (they're confirmed evil and already dealt with)
    for pos in state.confirmed_evil:
        if pos in state.executed:
            # This evil is done — remove one evil role from the pool
            # We don't know which role it was, but reduce count
            # Better: if we know the role, remove specifically
            pass

    # Remove confirmed-evil-and-executed from the roles to place
    remaining_evil_roles = evil_roles[:]
    executed_evil_positions = [p for p in state.confirmed_evil if p in state.executed]
    # Remove N roles for N executed evils (we don't track which role was executed)
    for _ in executed_evil_positions:
        if remaining_evil_roles:
            remaining_evil_roles.pop(0)
    evil_roles = remaining_evil_roles

    # Remove already-executed positions from candidates
    available = [p for p in all_positions if p not in state.executed]

    # Positions confirmed good can't be evil
    available = [p for p in available if p not in state.confirmed_good]

    # Confirmed evil positions (not yet executed) must get an evil role
    has_puppeteer = "Puppeteer" in evil_roles

    # If Puppeteer is present, we need an extra slot for Puppet
    if has_puppeteer:
        # Puppet isn't in evil_roles from deck — it's created at game start
        # We need to place Puppeteer + Puppet + other evils
        base_evil = [r for r in evil_roles if r != "Puppeteer"]
        placements = []
        # Choose positions for base evils (non-Puppeteer, non-Puppet)
        n_base = len(base_evil)
        # Puppeteer position + Puppet position + base evil positions
        for puppeteer_pos in available:
            adj = adjacent_positions(puppeteer_pos, n)
            puppet_candidates = [a for a in adj if a in available and a != puppeteer_pos]
            for puppet_pos in puppet_candidates:
                remaining = [p for p in available
                             if p != puppeteer_pos and p != puppet_pos]
                if n_base == 0:
                    placements.append({
                        puppeteer_pos: "Puppeteer",
                        puppet_pos: "Puppet",
                    })
                else:
                    for combo in combinations(remaining, n_base):
                        placement = {puppeteer_pos: "Puppeteer", puppet_pos: "Puppet"}
                        for role_perms in _permutations_of(base_evil):
                            p = dict(placement)
                            for i, pos in enumerate(combo):
                                p[pos] = role_perms[i]
                            placements.append(p)
        return placements

    # No Puppeteer — straightforward combinations
    n_evil = len(evil_roles)
    placements = []
    for combo in combinations(available, n_evil):
        for role_perms in _permutations_of(evil_roles):
            p = {}
            for i, pos in enumerate(combo):
                p[pos] = role_perms[i]
            placements.append(p)
    return placements


def _permutations_of(roles: list[str]) -> list[list[str]]:
    """Generate distinct permutations of roles (handling duplicates)."""
    if not roles:
        return [[]]
    seen = set()
    result = []
    for i, role in enumerate(roles):
        rest = roles[:i] + roles[i+1:]
        for perm in _permutations_of(rest):
            t = tuple([role] + perm)
            if t not in seen:
                seen.add(t)
                result.append(list(t))
    return result


# ============================================================
# Constraint Filtering
# ============================================================

def _apply_placement_constraints(placement: dict[int, str],
                                  state: GameState) -> bool:
    """Check structural constraints on evil placement."""
    n = state.n_cards

    # Chancellor must be adjacent to an outcast position
    # (Chancellor converts a Villager to Outcast, sits next to it)
    for pos, role in placement.items():
        if role == "Chancellor":
            adj = adjacent_positions(pos, n)
            # At least one adjacent must NOT be evil (can be the converted outcast)
            has_non_evil_adj = any(a not in placement for a in adj)
            if not has_non_evil_adj:
                return False

    # Confirmed evil (not yet executed) must be in the placement
    for pos in state.confirmed_evil:
        if pos not in state.executed and pos not in placement:
            return False

    return True


def _compute_corruption(placement: dict[int, str], state: GameState,
                        pd_target: Optional[int] = None) -> set[int]:
    """Compute which positions are corrupted given evil placement."""
    n = state.n_cards
    corrupted = set()

    for pos, role in placement.items():
        if role == "Pooka":
            # Pooka corrupts adjacent Villagers
            for adj in adjacent_positions(pos, n):
                if adj not in placement:  # Not evil
                    card = _get_card_at(adj, state)
                    if card and _is_villager_role(card.apparent_role, state):
                        corrupted.add(adj)
        elif role == "Poisoner":
            # Poisoner corrupts 1 adjacent Villager
            for adj in adjacent_positions(pos, n):
                if adj not in placement:
                    card = _get_card_at(adj, state)
                    if card and _is_villager_role(card.apparent_role, state):
                        corrupted.add(adj)
                        break  # Only 1

    # Plague Doctor corruption
    if pd_target and pd_target not in placement:
        corrupted.add(pd_target)

    # Drunk is always corrupted (and can't be cured)
    for card in state.cards:
        if card.apparent_role == "Drunk" and card.position not in placement:
            corrupted.add(card.position)

    return corrupted


def _get_card_at(pos: int, state: GameState) -> Optional[CardInfo]:
    """Get revealed card at position, or None."""
    for card in state.cards:
        if card.position == pos:
            return card
    return None


def _is_villager_role(role_name: str, state: GameState) -> bool:
    """Check if a role name is a Villager (corruption target)."""
    card_def = get_card(role_name)
    if card_def:
        return card_def.role == Role.VILLAGER
    return role_name in state.deck.villagers


def _get_real_role(pos: int, scenario: Scenario, state: GameState) -> str:
    """Get the real role of a position in a scenario."""
    if pos in scenario.evil_positions:
        return scenario.evil_positions[pos]
    if pos == scenario.puppet_position:
        return "Puppet"
    card = _get_card_at(pos, state)
    if card:
        return card.apparent_role
    return "Unknown"


def _is_evil_in_scenario(pos: int, scenario: Scenario) -> bool:
    """Check if a position is evil in this scenario (includes Puppet)."""
    return pos in scenario.evil_positions or pos == scenario.puppet_position


def _effective_alignment(pos: int, scenario: Scenario, state: GameState) -> Alignment:
    """Effective alignment for ability purposes. Wretch registers as Evil."""
    if _is_evil_in_scenario(pos, scenario):
        return Alignment.EVIL
    card = _get_card_at(pos, state)
    if card and card.apparent_role == "Wretch":
        return Alignment.EVIL  # Wretch registers as Evil to abilities
    return Alignment.GOOD


def _truth_status(pos: int, scenario: Scenario, state: GameState) -> TruthStatus:
    """Determine if a card tells truth or lies in this scenario."""
    # Evil characters lie (except Puppet)
    if pos in scenario.evil_positions:
        role = scenario.evil_positions[pos]
        if role == "Puppet":
            return TruthStatus.TRUTHFUL
        return TruthStatus.LYING

    if pos == scenario.puppet_position:
        return TruthStatus.TRUTHFUL  # Puppet can't lie

    # Corrupted characters lie
    if pos in scenario.corrupted:
        return TruthStatus.LYING

    # Good uncorrupted = truthful
    return TruthStatus.TRUTHFUL


# ============================================================
# Info Validators
# ============================================================

def _validate_enlightened(card: CardInfo, scenario: Scenario,
                          state: GameState) -> bool:
    """Enlightened: 'CW', 'CCW', or 'Equidistant' to nearest Evil."""
    info = card.info_parsed
    if "direction" not in info:
        return True  # Can't validate without parsed info

    claimed = info["direction"]  # "CW", "CCW", or "Equidistant"
    pos = card.position
    n = state.n_cards
    truth = _truth_status(pos, scenario, state)

    # Find closest evil(s) and their direction (exclude self)
    evil_positions = [p for p in range(1, n + 1)
                      if _is_evil_in_scenario(p, scenario) and p != pos]
    if not evil_positions:
        return True

    min_dist = min(circle_distance(pos, ep, n) for ep in evil_positions)
    closest = [ep for ep in evil_positions if circle_distance(pos, ep, n) == min_dist]

    if len(closest) >= 2:
        # Check if they're in different directions
        dirs = set(circle_direction(pos, ep, n) for ep in closest)
        if "CW" in dirs and "CCW" in dirs:
            real_answer = "Equidistant"
        else:
            real_answer = circle_direction(pos, closest[0], n)
    else:
        real_answer = circle_direction(pos, closest[0], n)

    if truth == TruthStatus.TRUTHFUL:
        return claimed == real_answer
    else:
        # Evil MUST lie — claimed must NOT equal real answer
        return claimed != real_answer


def _validate_knitter(card: CardInfo, scenario: Scenario,
                      state: GameState) -> bool:
    """Knitter: 'N pairs of Evil adjacent to each other'."""
    info = card.info_parsed
    if "evil_pairs" not in info:
        return True

    claimed = info["evil_pairs"]
    n = state.n_cards
    truth = _truth_status(pos := card.position, scenario, state)

    # Count pairs of adjacent evil positions (Wretch registers as Evil)
    evil_set = {p for p in range(1, n + 1)
                if _effective_alignment(p, scenario, state) == Alignment.EVIL}
    pairs = 0
    for p in evil_set:
        for adj in adjacent_positions(p, n):
            if adj in evil_set and adj > p:  # Count each pair once
                pairs += 1

    if truth == TruthStatus.TRUTHFUL:
        return claimed == pairs
    else:
        return claimed != pairs


def _validate_confessor(card: CardInfo, scenario: Scenario,
                        state: GameState) -> bool:
    """Confessor: 'I am dirty' if Evil or Corrupted. Can't lie."""
    info = card.info_parsed
    if "dirty" not in info:
        return True

    claimed_dirty = info["dirty"]  # True = "I am dirty", False = "I am clean"
    pos = card.position

    # Confessor CAN'T LIE — always tells truth regardless of corruption/evil
    is_evil = _is_evil_in_scenario(pos, scenario)
    is_corrupted = pos in scenario.corrupted
    actually_dirty = is_evil or is_corrupted

    return claimed_dirty == actually_dirty


def _validate_gemcrafter(card: CardInfo, scenario: Scenario,
                         state: GameState) -> bool:
    """Gemcrafter: 'X is Good'. Wretch registers as Evil."""
    info = card.info_parsed
    if "good_position" not in info:
        return True

    claimed_pos = info["good_position"]
    pos = card.position
    truth = _truth_status(pos, scenario, state)

    # Check effective alignment (Wretch = Evil for ability purposes)
    actual_good = _effective_alignment(claimed_pos, scenario, state) == Alignment.GOOD

    if truth == TruthStatus.TRUTHFUL:
        return actual_good
    else:
        return not actual_good


def _validate_lover(card: CardInfo, scenario: Scenario,
                    state: GameState) -> bool:
    """Lover: 'N Evil adjacent to me'."""
    info = card.info_parsed
    if "evil_adjacent" not in info:
        return True

    claimed = info["evil_adjacent"]
    pos = card.position
    n = state.n_cards
    truth = _truth_status(pos, scenario, state)

    adj = adjacent_positions(pos, n)
    actual = sum(1 for a in adj
                 if _effective_alignment(a, scenario, state) == Alignment.EVIL)

    if truth == TruthStatus.TRUTHFUL:
        return claimed == actual
    else:
        return claimed != actual


def _validate_scout(card: CardInfo, scenario: Scenario,
                    state: GameState) -> bool:
    """Scout: 'Evil X is N away from nearest Evil'."""
    info = card.info_parsed
    if "evil_role" not in info or "distance" not in info:
        return True

    evil_role = info["evil_role"]
    claimed_dist = info["distance"]
    pos = card.position
    n = state.n_cards
    truth = _truth_status(pos, scenario, state)

    # Find position of the named evil role
    target_pos = None
    for p, role in scenario.evil_positions.items():
        if role == evil_role:
            target_pos = p
            break
    if target_pos is None:
        return True  # Can't validate

    # Find distance to closest OTHER evil
    other_evil = [p for p in range(1, n + 1)
                  if _is_evil_in_scenario(p, scenario) and p != target_pos]
    if not other_evil:
        return True

    actual_dist = min(circle_distance(target_pos, oe, n) for oe in other_evil)

    if truth == TruthStatus.TRUTHFUL:
        return claimed_dist == actual_dist
    else:
        return claimed_dist != actual_dist


def _validate_bard(card: CardInfo, scenario: Scenario,
                   state: GameState) -> bool:
    """Bard: 'N cards from closest Corrupted'."""
    info = card.info_parsed
    if "corruption_distance" not in info:
        return True

    claimed = info["corruption_distance"]
    pos = card.position
    n = state.n_cards
    truth = _truth_status(pos, scenario, state)

    if not scenario.corrupted:
        # No corrupted cards — Bard would say... 0? or max?
        # "How far from closest corrupted" — if none, this is tricky
        # Treat as: no corrupted = can't match any distance claim
        # A truthful Bard with no corrupted targets is edge case
        actual = 0  # Game likely shows 0 or doesn't fire
    else:
        actual = min(circle_distance(pos, c, n) for c in scenario.corrupted)

    if truth == TruthStatus.TRUTHFUL:
        return claimed == actual
    else:
        return claimed != actual


def _validate_fortune_teller(card: CardInfo, scenario: Scenario,
                             state: GameState) -> bool:
    """Fortune Teller: 'Pick 2 chars, any Evil?' -> Yes/No."""
    info = card.info_parsed
    if "targets" not in info or "has_evil" not in info:
        return True

    targets = info["targets"]  # list of 2 positions
    claimed_evil = info["has_evil"]  # True/False
    pos = card.position
    truth = _truth_status(pos, scenario, state)

    actual_evil = any(_effective_alignment(t, scenario, state) == Alignment.EVIL
                      for t in targets)

    if truth == TruthStatus.TRUTHFUL:
        return claimed_evil == actual_evil
    else:
        return claimed_evil != actual_evil


def _validate_oracle(card: CardInfo, scenario: Scenario,
                     state: GameState) -> bool:
    """Oracle: '1 of positions [a,b] is role X (Minion)'."""
    info = card.info_parsed
    if "targets" not in info or "minion_role" not in info:
        return True

    targets = info["targets"]  # list of 2 positions
    minion_role = info["minion_role"]  # e.g. "Shaman"
    pos = card.position
    truth = _truth_status(pos, scenario, state)

    # Check if either target actually IS that minion role
    actual = any(scenario.evil_positions.get(t) == minion_role for t in targets)

    if truth == TruthStatus.TRUTHFUL:
        return actual
    else:
        return not actual


def _validate_medium(card: CardInfo, scenario: Scenario,
                     state: GameState) -> bool:
    """Medium: 'X is Good Y role'."""
    info = card.info_parsed
    if "good_position" not in info or "good_role" not in info:
        return True

    claimed_pos = info["good_position"]
    claimed_role = info["good_role"]
    pos = card.position
    truth = _truth_status(pos, scenario, state)

    # Check if the claimed position is actually that good role
    is_good = _effective_alignment(claimed_pos, scenario, state) == Alignment.GOOD
    actual_role = _get_real_role(claimed_pos, scenario, state)
    actual_match = is_good and actual_role == claimed_role

    if truth == TruthStatus.TRUTHFUL:
        return actual_match
    else:
        return not actual_match


def _validate_hunter(card: CardInfo, scenario: Scenario,
                     state: GameState) -> bool:
    """Hunter: 'Nearest Evil is N away'."""
    info = card.info_parsed
    if "distance" not in info:
        return True

    claimed = info["distance"]
    pos = card.position
    n = state.n_cards
    truth = _truth_status(pos, scenario, state)

    evil_positions = [p for p in range(1, n + 1)
                      if _is_evil_in_scenario(p, scenario) and p != pos]
    if not evil_positions:
        return True

    actual = min(circle_distance(pos, ep, n) for ep in evil_positions)

    if truth == TruthStatus.TRUTHFUL:
        return claimed == actual
    else:
        return claimed != actual


def _validate_architect(card: CardInfo, scenario: Scenario,
                        state: GameState) -> bool:
    """Architect: 'Left'/'Right'/'Equal' — which side has more Evil."""
    info = card.info_parsed
    if "side" not in info:
        return True

    claimed = info["side"]  # "Left", "Right", "Equal"
    pos = card.position
    n = state.n_cards
    truth = _truth_status(pos, scenario, state)

    # CW = right side, CCW = left side
    left_count = 0
    right_count = 0
    for p in range(1, n + 1):
        if p == pos:
            continue
        if _is_evil_in_scenario(p, scenario):
            d = circle_direction(pos, p, n)
            if d == "CW":
                right_count += 1
            elif d == "CCW":
                left_count += 1
            else:  # Equidistant — count for both? Skip?
                left_count += 0.5
                right_count += 0.5

    if left_count > right_count:
        actual = "Left"
    elif right_count > left_count:
        actual = "Right"
    else:
        actual = "Equal"

    if truth == TruthStatus.TRUTHFUL:
        return claimed == actual
    else:
        return claimed != actual


def _validate_empress(card: CardInfo, scenario: Scenario,
                      state: GameState) -> bool:
    """Empress: 'Among [a,b,c], only 1 is Evil'."""
    info = card.info_parsed
    if "targets" not in info:
        return True

    targets = info["targets"]  # list of 3 positions
    pos = card.position
    truth = _truth_status(pos, scenario, state)

    evil_count = sum(1 for t in targets
                     if _effective_alignment(t, scenario, state) == Alignment.EVIL)

    if truth == TruthStatus.TRUTHFUL:
        return evil_count == 1
    else:
        return evil_count != 1


def _validate_witness(card: CardInfo, scenario: Scenario,
                      state: GameState) -> bool:
    """Witness: 'X was affected by Evil ability'."""
    info = card.info_parsed
    if "affected_position" not in info:
        return True

    claimed_pos = info["affected_position"]
    pos = card.position
    truth = _truth_status(pos, scenario, state)

    # "Affected by evil ability" = corrupted, puppeted, or adjacent to Chancellor conversion
    actually_affected = (
        claimed_pos in scenario.corrupted or
        claimed_pos == scenario.puppet_position
    )

    if truth == TruthStatus.TRUTHFUL:
        return actually_affected
    else:
        return not actually_affected


def _validate_jester(card: CardInfo, scenario: Scenario,
                     state: GameState) -> bool:
    """Jester: 'Pick 3 chars, N are Evil'."""
    info = card.info_parsed
    if "targets" not in info or "evil_count" not in info:
        return True

    targets = info["targets"]
    claimed = info["evil_count"]
    pos = card.position
    truth = _truth_status(pos, scenario, state)

    actual = sum(1 for t in targets
                 if _effective_alignment(t, scenario, state) == Alignment.EVIL)

    if truth == TruthStatus.TRUTHFUL:
        return claimed == actual
    else:
        return claimed != actual


def _validate_dreamer(card: CardInfo, scenario: Scenario,
                      state: GameState) -> bool:
    """Dreamer: 'Pick 1 char, learn an Evil role. If Evil picked, correct.'"""
    info = card.info_parsed
    if "target" not in info or "evil_role" not in info:
        return True

    target = info["target"]
    claimed_role = info["evil_role"]
    pos = card.position
    truth = _truth_status(pos, scenario, state)

    target_is_evil = _is_evil_in_scenario(target, scenario)

    if truth == TruthStatus.TRUTHFUL:
        if target_is_evil:
            # Must show correct evil role
            actual_role = scenario.evil_positions.get(target)
            return claimed_role == actual_role
        else:
            # Shows a random evil role (any evil role is valid)
            return claimed_role in [r for r in scenario.evil_positions.values()]
    else:
        # Lying — info is false
        if target_is_evil:
            actual_role = scenario.evil_positions.get(target)
            return claimed_role != actual_role
        else:
            return True  # Hard to validate lies on non-evil target


def _validate_judge(card: CardInfo, scenario: Scenario,
                    state: GameState) -> bool:
    """Judge: 'Pick 1 char — are they Lying?'"""
    info = card.info_parsed
    if "target" not in info or "is_lying" not in info:
        return True

    target = info["target"]
    claimed_lying = info["is_lying"]
    pos = card.position
    truth = _truth_status(pos, scenario, state)

    actual_lying = _truth_status(target, scenario, state) == TruthStatus.LYING

    if truth == TruthStatus.TRUTHFUL:
        return claimed_lying == actual_lying
    else:
        return claimed_lying != actual_lying


def _validate_alchemist(card: CardInfo, scenario: Scenario,
                        state: GameState) -> bool:
    """Alchemist: 'Cured N' — cures corruption in range 2."""
    info = card.info_parsed
    if "cured_count" not in info:
        return True

    claimed = info["cured_count"]
    pos = card.position
    n = state.n_cards
    truth = _truth_status(pos, scenario, state)

    # Alchemist cures villagers in range 2 of corruption
    # But if Alchemist itself is corrupted, it can't cure anyone
    if pos in scenario.corrupted:
        actual = 0
    else:
        in_range = positions_in_range(pos, 2, n)
        actual = sum(1 for p in in_range if p in scenario.corrupted)

    if truth == TruthStatus.TRUTHFUL:
        return claimed == actual
    else:
        return claimed != actual


def _validate_druid(card: CardInfo, scenario: Scenario,
                    state: GameState) -> bool:
    """Druid: 'Pick 3 chars, learn 1 Outcast among them'. Wretch=Evil to Druid."""
    info = card.info_parsed
    if "targets" not in info:
        return True

    targets = info["targets"]
    found_outcast = info.get("found_outcast")  # role name or None
    pos = card.position
    truth = _truth_status(pos, scenario, state)

    # Check which targets are actually outcasts (Wretch registers as Evil, not Outcast)
    actual_outcasts = []
    for t in targets:
        if _is_evil_in_scenario(t, scenario):
            continue
        card_at = _get_card_at(t, state)
        if card_at:
            card_def = get_card(card_at.apparent_role)
            if card_def and card_def.role == Role.OUTCAST:
                if card_at.apparent_role == "Wretch":
                    continue  # Wretch registers as Evil to abilities
                actual_outcasts.append(card_at.apparent_role)

    has_outcast = len(actual_outcasts) > 0

    if truth == TruthStatus.TRUTHFUL:
        if found_outcast:
            return has_outcast and found_outcast in actual_outcasts
        else:
            return not has_outcast
    else:
        if found_outcast:
            return not has_outcast or found_outcast not in actual_outcasts
        else:
            return has_outcast


def _validate_bishop(card: CardInfo, scenario: Scenario,
                     state: GameState) -> bool:
    """Bishop: 'Learn 3 chars: 1 Villager, 1 Outcast, 1 Evil if possible'."""
    info = card.info_parsed
    if "targets" not in info:
        return True
    # Bishop's info is structural — hard to validate without knowing which is which
    # For now, just check that at least one target is Evil
    targets = info["targets"]
    pos = card.position
    truth = _truth_status(pos, scenario, state)

    has_evil = any(_effective_alignment(t, scenario, state) == Alignment.EVIL
                   for t in targets)

    if truth == TruthStatus.TRUTHFUL:
        return has_evil  # Should contain 1 Evil if possible
    else:
        return True  # Hard to validate lying Bishop


# ============================================================
# Validator Registry
# ============================================================

VALIDATORS = {
    "Enlightened": _validate_enlightened,
    "Knitter": _validate_knitter,
    "Confessor": _validate_confessor,
    "Gemcrafter": _validate_gemcrafter,
    "Lover": _validate_lover,
    "Scout": _validate_scout,
    "Bard": _validate_bard,
    "Fortune Teller": _validate_fortune_teller,
    "Oracle": _validate_oracle,
    "Medium": _validate_medium,
    "Hunter": _validate_hunter,
    "Architect": _validate_architect,
    "Empress": _validate_empress,
    "Witness": _validate_witness,
    "Jester": _validate_jester,
    "Dreamer": _validate_dreamer,
    "Judge": _validate_judge,
    "Alchemist": _validate_alchemist,
    "Druid": _validate_druid,
    "Bishop": _validate_bishop,
}


# ============================================================
# Main Solver
# ============================================================

def _build_scenarios(state: GameState) -> list[Scenario]:
    """Build all candidate scenarios from evil placements."""
    placements = _generate_evil_placements(state)
    scenarios = []

    # Find Plague Doctor position (if any) for corruption
    pd_pos = None
    for card in state.cards:
        if card.apparent_role == "Plague Doctor":
            if card.position not in []:  # Not evil (check later per scenario)
                pd_pos = card.position

    for placement in placements:
        if not _apply_placement_constraints(placement, state):
            continue

        puppet_pos = None
        if "Puppeteer" in placement.values():
            for pos, role in placement.items():
                if role == "Puppet":
                    puppet_pos = pos

        # Determine PD corruption targets
        # PD corrupts 1 random Good Villager — we need to try all possibilities
        # For simplicity, if PD is in play and not evil, try each villager position
        pd_targets = [None]
        if pd_pos and pd_pos not in placement:
            # PD is good — it corrupted someone
            if "Plague Doctor" in state.deck.outcasts:
                if state.pd_corruption_target is not None:
                    # Known PD target — only use that
                    pd_targets = [state.pd_corruption_target]
                else:
                    candidates = []
                    for p in range(1, state.n_cards + 1):
                        if p == pd_pos or p in placement:
                            continue
                        c = _get_card_at(p, state)
                        if c and _is_villager_role(c.apparent_role, state):
                            candidates.append(p)
                    if candidates:
                        pd_targets = candidates
                    else:
                        pd_targets = [None]

        for pd_t in pd_targets:
            corrupted = _compute_corruption(placement, state, pd_t)
            scenario = Scenario(
                evil_positions=dict(placement),
                puppet_position=puppet_pos,
                corrupted=corrupted,
                pd_corrupted=pd_t,
            )
            scenarios.append(scenario)

    return scenarios


def _check_scenario(scenario: Scenario, state: GameState) -> bool:
    """Check if a scenario is consistent with all revealed card info."""
    for card in state.cards:
        if card.position in scenario.evil_positions:
            # Evil card — its apparent role is a disguise, skip role-based validation
            # But we can still check if its info (as a lie) is consistent
            # The evil card is LYING — its info must NOT match truth
            role = card.apparent_role
            if role in VALIDATORS:
                if not VALIDATORS[role](card, scenario, state):
                    return False
            continue

        if card.position == scenario.puppet_position:
            # Puppet — disguised as a villager, tells truth
            role = card.apparent_role
            if role in VALIDATORS:
                if not VALIDATORS[role](card, scenario, state):
                    return False
            continue

        # Good card (possibly corrupted)
        role = card.apparent_role
        if role in VALIDATORS:
            if not VALIDATORS[role](card, scenario, state):
                return False

    return True


def solve(state: GameState) -> SolverResult:
    """Main solver entry point."""
    scenarios = _build_scenarios(state)
    reasoning = [f"Generated {len(scenarios)} candidate scenarios"]

    surviving = [s for s in scenarios if _check_scenario(s, state)]
    reasoning.append(f"{len(surviving)} scenarios survived validation")

    if not surviving:
        return SolverResult(
            definite_evil=[], definite_good=[],
            bombardier_positions=[], n_scenarios=len(scenarios),
            n_surviving=0, surviving_scenarios=[],
            reasoning=reasoning + ["NO VALID SCENARIOS — check input data"],
        )

    n = state.n_cards
    all_positions = set(range(1, n + 1))

    # Positions evil in ALL surviving scenarios
    definite_evil = []
    for pos in all_positions:
        if all(_is_evil_in_scenario(pos, s) for s in surviving):
            definite_evil.append(pos)

    # Positions good in ALL surviving scenarios
    definite_good = []
    for pos in all_positions:
        if all(not _is_evil_in_scenario(pos, s) for s in surviving):
            definite_good.append(pos)

    # Bombardier positions — never execute
    bombardier_positions = []
    for card in state.cards:
        if card.apparent_role == "Bombardier" and card.position in definite_good:
            bombardier_positions.append(card.position)

    for pos in sorted(definite_evil):
        roles = set()
        for s in surviving:
            if pos in s.evil_positions:
                roles.add(s.evil_positions[pos])
        reasoning.append(f"  #{pos} is DEFINITELY EVIL (possible roles: {roles})")

    for pos in sorted(definite_good):
        reasoning.append(f"  #{pos} is DEFINITELY GOOD")

    uncertain = all_positions - set(definite_evil) - set(definite_good)
    if uncertain:
        reasoning.append(f"  Uncertain: {sorted(uncertain)}")

    return SolverResult(
        definite_evil=sorted(definite_evil),
        definite_good=sorted(definite_good),
        bombardier_positions=sorted(bombardier_positions),
        n_scenarios=len(scenarios),
        n_surviving=len(surviving),
        surviving_scenarios=surviving,
        reasoning=reasoning,
    )


if __name__ == "__main__":
    # Quick smoke test with a simple scenario
    print("Solver module loaded successfully")
    print(f"Circle distance 1->4 on 7: {circle_distance(1, 4, 7)}")
    print(f"Circle direction 1->3 on 7: {circle_direction(1, 3, 7)}")
    print(f"Adjacent to 1 on 7: {adjacent_positions(1, 7)}")
