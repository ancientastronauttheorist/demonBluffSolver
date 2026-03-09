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
    CW means increasing position numbers (1->2->3...) matching the game's
    visual clockwise layout."""
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

    def to_dict(self) -> dict:
        return {
            "position": self.position,
            "apparent_role": self.apparent_role,
            "info_text": self.info_text,
            "info_parsed": dict(self.info_parsed),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CardInfo":
        return cls(
            data["position"],
            data["apparent_role"],
            data.get("info_text", ""),
            data.get("info_parsed", {}),
        )


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

    def to_dict(self) -> dict:
        return {
            "villagers": list(self.villagers),
            "outcasts": list(self.outcasts),
            "minions": list(self.minions),
            "demons": list(self.demons),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DeckComposition":
        return cls(
            villagers=list(data.get("villagers", [])),
            outcasts=list(data.get("outcasts", [])),
            minions=list(data.get("minions", [])),
            demons=list(data.get("demons", [])),
        )


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
    executed_evil_roles: dict[int, str] = field(default_factory=dict)  # pos -> evil role name (e.g. {2: "Chancellor"})
    slayer_results: list[dict] = field(default_factory=list)  # [{slayer_pos, target_pos, killed}]
    night_kills: list[int] = field(default_factory=list)  # Positions killed by Lilis night (unrevealed)
    night_kill_evil_count: int = 0  # How many of the night kills were evil
    hp: int = 10                    # Current health points
    wrong_exec_cost: int = 2        # HP lost per wrong execution (varies by ascension)
    pd_ability_results: list[dict] = field(default_factory=list)  # [{"pd_pos": N, "target": N, "is_corrupted": bool, "evil_revealed": N|None}]
    blocked_positions: list[int] = field(default_factory=list)  # Positions blocked from reveal (Witch)
    board_villager_count: Optional[int] = None  # Actual villagers on board (when pool > board)
    board_outcast_count: Optional[int] = None   # Actual outcasts on board (when pool > board)
    board_minion_count: Optional[int] = None    # Actual minions on board (when pool > board)
    board_demon_count: Optional[int] = None     # Actual demons on board (when pool > board)
    reveal_order: list[int] = field(default_factory=list)  # Order positions were flipped (for Baker)

    def to_dict(self, *, nest_deck: bool = True) -> dict:
        data = {
            "n_cards": self.n_cards,
            "n_evil": self.n_evil,
            "cards": [card.to_dict() for card in self.cards],
            "executed": list(self.executed),
            "confirmed_evil": list(self.confirmed_evil),
            "confirmed_good": list(self.confirmed_good),
            "pd_corruption_target": self.pd_corruption_target,
            "executed_evil_roles": {str(k): v for k, v in self.executed_evil_roles.items()},
            "slayer_results": list(self.slayer_results),
            "pd_ability_results": list(self.pd_ability_results),
            "blocked_positions": list(self.blocked_positions),
            "night_kills": list(self.night_kills),
            "night_kill_evil_count": self.night_kill_evil_count,
            "hp": self.hp,
            "wrong_exec_cost": self.wrong_exec_cost,
            "board_villager_count": self.board_villager_count,
            "board_outcast_count": self.board_outcast_count,
            "board_minion_count": self.board_minion_count,
            "board_demon_count": self.board_demon_count,
            "reveal_order": list(self.reveal_order),
        }
        if nest_deck:
            data["deck"] = self.deck.to_dict()
        else:
            data.update(self.deck.to_dict())
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "GameState":
        deck_data = data.get("deck")
        if deck_data is None:
            deck_data = {
                "villagers": data.get("villagers", []),
                "outcasts": data.get("outcasts", []),
                "minions": data.get("minions", []),
                "demons": data.get("demons", []),
            }

        raw_eer = data.get("executed_evil_roles", {})
        executed_evil_roles = {int(k): v for k, v in raw_eer.items()}

        return cls(
            n_cards=data["n_cards"],
            deck=DeckComposition.from_dict(deck_data),
            cards=[CardInfo.from_dict(c) for c in data.get("cards", [])],
            n_evil=data.get("n_evil", 0),
            executed=list(data.get("executed", [])),
            confirmed_evil=list(data.get("confirmed_evil", [])),
            confirmed_good=list(data.get("confirmed_good", [])),
            pd_corruption_target=data.get("pd_corruption_target"),
            executed_evil_roles=executed_evil_roles,
            slayer_results=list(data.get("slayer_results", [])),
            pd_ability_results=list(data.get("pd_ability_results", [])),
            blocked_positions=list(data.get("blocked_positions", [])),
            night_kills=list(data.get("night_kills", [])),
            night_kill_evil_count=data.get("night_kill_evil_count", 0),
            hp=data.get("hp", 10),
            wrong_exec_cost=data.get("wrong_exec_cost", 2),
            board_villager_count=data.get("board_villager_count"),
            board_outcast_count=data.get("board_outcast_count"),
            board_minion_count=data.get("board_minion_count"),
            board_demon_count=data.get("board_demon_count"),
            reveal_order=list(data.get("reveal_order", [])),
        )


@dataclass
class Scenario:
    """A hypothetical assignment of evil roles to positions."""
    evil_positions: dict[int, str]  # pos -> evil role name
    puppet_position: Optional[int] = None  # If Puppeteer in play
    corrupted: set[int] = field(default_factory=set)  # Corrupted positions
    pd_corrupted: Optional[int] = None  # Plague Doctor corruption target
    doppelganger_position: Optional[int] = None  # Doppelganger pos (real role != apparent)
    drunk_position: Optional[int] = None  # Drunk pos (disguised as Villager, always corrupted)
    alchemist_cures: dict = field(default_factory=dict)  # alch_pos -> cure count (pre-cure)


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

    # Puppet is created dynamically by Puppeteer's game-start ability, not
    # independently placed.  When both are in the deck, remove Puppet from
    # evil_roles to avoid double-counting (the Puppeteer branch creates it).
    puppet_in_deck = "Puppet" in evil_roles and "Puppeteer" in evil_roles
    if puppet_in_deck:
        evil_roles.remove("Puppet")

    # Remove already-executed evil roles from what we need to place
    remaining_evil_roles = evil_roles[:]
    for pos, role in state.executed_evil_roles.items():
        # Remove the specific role that was executed (e.g., "Chancellor")
        if role in remaining_evil_roles:
            remaining_evil_roles.remove(role)
        elif role == "Puppet":
            pass  # Puppet isn't in evil_roles, it's generated by Puppeteer

    # Fallback for old-style confirmed_evil without role info
    executed_evil_without_role = [
        p for p in state.confirmed_evil
        if p in state.executed and p not in state.executed_evil_roles
    ]
    for _ in executed_evil_without_role:
        if remaining_evil_roles:
            remaining_evil_roles.pop(0)  # Best-effort: remove first available

    evil_roles = remaining_evil_roles

    # How many evil positions remain to be placed?
    # Old sessions may only know that an executed position was evil, not which
    # role it was. Those positions still reduce the number of remaining evils.
    n_executed_evil = len(state.executed_evil_roles) + len(executed_evil_without_role)
    expected_remaining = state.n_evil - n_executed_evil

    # Pre-compute valid placement sizes: when Puppet ambiguity exists
    # (game dialog may or may not count Puppet in n_evil), accept +1
    valid_sizes = {expected_remaining, expected_remaining + 1} if puppet_in_deck else {expected_remaining}

    # Remove already-executed positions from candidates
    # Night-killed positions stay eligible (evil could have been among them)
    player_executed = [p for p in state.executed if p not in state.night_kills]
    available = [p for p in all_positions if p not in player_executed]

    # Positions confirmed good can't be evil
    available = [p for p in available if p not in state.confirmed_good]

    # Confirmed evil positions (not yet executed) must get an evil role
    has_puppeteer = "Puppeteer" in evil_roles

    # Check if Puppeteer was already executed but Puppet still needs placing
    puppeteer_executed_pos = None
    puppet_still_alive = False
    for ex_pos_str, ex_role in state.executed_evil_roles.items():
        if ex_role == "Puppeteer":
            puppeteer_executed_pos = int(ex_pos_str)
    if puppeteer_executed_pos is not None and not has_puppeteer:
        # Puppeteer was executed — check if Puppet was also executed
        puppet_executed = any(r == "Puppet" for r in state.executed_evil_roles.values())
        if not puppet_executed:
            puppet_still_alive = True

    # If Puppet might still be alive (Puppeteer executed, Puppet not)
    if puppet_still_alive:
        adj = adjacent_positions(puppeteer_executed_pos, n)
        puppet_candidates = [a for a in adj if a in available]
        placements = []
        # Case 1: Puppet exists at an adjacent position
        for puppet_pos in puppet_candidates:
            remaining_avail = [p for p in available if p != puppet_pos]
            n_other = len(evil_roles)
            if n_other == 0:
                placements.append({puppet_pos: "Puppet"})
            else:
                for combo in combinations(remaining_avail, n_other):
                    for role_perms in _permutations_of(evil_roles):
                        p = {puppet_pos: "Puppet"}
                        for i, pos in enumerate(combo):
                            p[pos] = role_perms[i]
                        placements.append(p)
        # Case 2: No Puppet was created (both adjacent were evil/non-Villager)
        n_other = len(evil_roles)
        for combo in combinations(available, n_other):
            for role_perms in _permutations_of(evil_roles):
                p = {}
                for i, pos in enumerate(combo):
                    p[pos] = role_perms[i]
                placements.append(p)
        return [p for p in placements if len(p) in valid_sizes]

    # Check if Puppet was already executed but Puppeteer is still alive
    puppet_executed_pos = None
    for ex_pos, ex_role in state.executed_evil_roles.items():
        if ex_role == "Puppet":
            puppet_executed_pos = int(ex_pos)

    # If Puppet was executed, Puppeteer must be adjacent to the Puppet's position
    if puppet_executed_pos is not None and has_puppeteer:
        base_evil = [r for r in evil_roles if r != "Puppeteer"]
        placements = []
        n_base = len(base_evil)
        # Puppeteer must be adjacent to the executed Puppet
        puppeteer_candidates = [a for a in adjacent_positions(puppet_executed_pos, n)
                                if a in available]
        for puppeteer_pos in puppeteer_candidates:
            remaining = [p for p in available if p != puppeteer_pos]
            if n_base == 0:
                placements.append({puppeteer_pos: "Puppeteer"})
            else:
                for combo in combinations(remaining, n_base):
                    for role_perms in _permutations_of(base_evil):
                        p = {puppeteer_pos: "Puppeteer"}
                        for i, pos in enumerate(combo):
                            p[pos] = role_perms[i]
                        placements.append(p)
        return [p for p in placements if len(p) in valid_sizes]

    # If Puppeteer is present, we need an extra slot for Puppet
    if has_puppeteer:
        # Puppet isn't in evil_roles from deck — it's created at game start
        # We need to place Puppeteer + Puppet + other evils
        # BUT: Puppet creation is optional ("if possible") — if both adjacent
        # positions are evil or non-Villager (Outcast), no Puppet is created.
        base_evil = [r for r in evil_roles if r != "Puppeteer"]
        placements = []
        n_base = len(base_evil)
        for puppeteer_pos in available:
            adj = adjacent_positions(puppeteer_pos, n)
            puppet_candidates = [a for a in adj if a in available and a != puppeteer_pos]

            # Puppeteer targets Villagers only when possible (patch v0.131a).
            # If any adjacent candidate appears as a Villager, restrict to
            # Villagers + unrevealed (which might be Villagers).
            villager_or_unknown = []
            has_known_villager = False
            for pc in puppet_candidates:
                card_at = _get_card_at(pc, state)
                if card_at is None:
                    villager_or_unknown.append(pc)  # unrevealed, could be Villager
                elif _is_villager_role(card_at.apparent_role, state):
                    villager_or_unknown.append(pc)
                    has_known_villager = True
            if has_known_villager:
                puppet_candidates = villager_or_unknown

            # Case 1: Puppet IS created (at each adjacent candidate)
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

            # Case 2: Puppet NOT created (Puppeteer without Puppet)
            remaining = [p for p in available if p != puppeteer_pos]
            if n_base == 0:
                placements.append({puppeteer_pos: "Puppeteer"})
            else:
                for combo in combinations(remaining, n_base):
                    placement = {puppeteer_pos: "Puppeteer"}
                    for role_perms in _permutations_of(base_evil):
                        p = dict(placement)
                        for i, pos in enumerate(combo):
                            p[pos] = role_perms[i]
                        placements.append(p)

        return [p for p in placements if len(p) in valid_sizes]

    # No Puppeteer — straightforward combinations
    n_evil = len(evil_roles)

    # When evil pool > board evil count (Asc10+), generate all valid subsets
    # of evil roles that match board minion/demon counts.
    if n_evil > expected_remaining and expected_remaining > 0:
        evil_role_subsets = _evil_role_subsets(
            evil_roles, state, expected_remaining
        )
    elif n_evil == expected_remaining:
        evil_role_subsets = [evil_roles]
    elif not puppet_in_deck:
        assert n_evil == expected_remaining, \
            f"Evil role count {n_evil} != expected remaining {expected_remaining}"
        evil_role_subsets = [evil_roles]
    else:
        evil_role_subsets = [evil_roles] if evil_roles else []

    placements = []
    if expected_remaining > 0:
        for role_set in evil_role_subsets:
            n_roles = len(role_set)
            for combo in combinations(available, n_roles):
                for role_perms in _permutations_of(role_set):
                    p = {}
                    for i, pos in enumerate(combo):
                        p[pos] = role_perms[i]
                    placements.append(p)
    elif expected_remaining <= 0:
        placements.append({})  # No evil left to place
    return placements


def _evil_role_subsets(
    evil_roles: list[str], state: GameState, expected_remaining: int
) -> list[list[str]]:
    """Generate all valid subsets of evil roles when pool > board count.

    Uses board_minion_count and board_demon_count to pick the right number
    of minions and demons from the pool.  Falls back to plain combinations
    when counts are not available.
    """
    from knowledge_base import CARDS_BY_NAME, Role
    minion_pool = [r for r in evil_roles
                   if r in CARDS_BY_NAME and CARDS_BY_NAME[r].role == Role.MINION]
    demon_pool  = [r for r in evil_roles
                   if r in CARDS_BY_NAME and CARDS_BY_NAME[r].role == Role.DEMON]

    # Determine how many of each to pick
    bm = state.board_minion_count
    bd = state.board_demon_count

    # Subtract already-executed evil roles from counts
    for _pos, role in state.executed_evil_roles.items():
        if role in minion_pool:
            if bm is not None:
                bm -= 1
            minion_pool = [r for r in minion_pool if r != role]  # remove first
        elif role in demon_pool:
            if bd is not None:
                bd -= 1
            demon_pool = [r for r in demon_pool if r != role]

    if bm is not None and bd is not None:
        # Pick bm minions from pool and bd demons from pool
        subsets = []
        for m_combo in combinations(minion_pool, max(bm, 0)):
            for d_combo in combinations(demon_pool, max(bd, 0)):
                subset = list(m_combo) + list(d_combo)
                if len(subset) == expected_remaining:
                    subsets.append(subset)
        return subsets if subsets else [evil_roles[:expected_remaining]]

    # Fallback: try all combinations of expected_remaining from evil_roles
    subsets = []
    for combo in combinations(evil_roles, expected_remaining):
        subsets.append(list(combo))
    return subsets


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

def _normalize_role(name: str) -> str:
    """Normalize role names: lowercase, remove spaces and underscores."""
    return name.lower().replace("_", "").replace(" ", "")


def _is_plague_doctor(name: str) -> bool:
    """Case-insensitive check for any variant of Plague Doctor."""
    return _normalize_role(name) == "plaguedoctor"


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
                        pd_target: Optional[int] = None,
                        doppelganger_pos: Optional[int] = None,
                        poisoner_target: Optional[int] = None,
                        drunk_pos: Optional[int] = None) -> set[int]:
    """Compute which positions are corrupted given evil placement.

    doppelganger_pos: if set, this position is Doppelganger (Outcast) and
    immune to Pooka/Poisoner corruption (they only target Villagers).
    drunk_pos: if set, this position is Drunk (always corrupted, can't be cured).
    """
    n = state.n_cards
    corrupted = set()

    for pos, role in placement.items():
        if role == "Pooka":
            # Pooka corrupts adjacent Villagers (Doppelganger immune)
            for adj in adjacent_positions(pos, n):
                if adj not in placement and adj != doppelganger_pos:
                    card = _get_card_at(adj, state)
                    if card and _is_villager_role(card.apparent_role, state):
                        corrupted.add(adj)
                    elif card is None and _unrevealed_must_be_villager(
                            adj, placement, state, doppelganger_pos, drunk_pos):
                        corrupted.add(adj)
        elif role == "Poisoner":
            # Poisoner corrupts 1 adjacent Villager (Doppelganger immune)
            # Target is enumerated externally via poisoner_target parameter
            if poisoner_target and poisoner_target not in placement and poisoner_target != doppelganger_pos:
                corrupted.add(poisoner_target)

    # Plague Doctor corruption
    if pd_target and pd_target not in placement:
        corrupted.add(pd_target)

    # Drunk is always corrupted (can't be cured by Alchemist)
    # Two cases: (1) explicitly revealed as "Drunk" via Medium, or
    # (2) enumerated drunk_pos (Drunk disguised as Villager)
    if drunk_pos is not None and drunk_pos not in placement:
        corrupted.add(drunk_pos)
    for card in state.cards:
        if card.apparent_role == "Drunk" and card.position not in placement:
            corrupted.add(card.position)

    return corrupted


def _apply_post_corruption(corrupted: set[int], placement: dict[int, str],
                           state: GameState,
                           puppet_pos: Optional[int] = None,
                           drunk_pos: Optional[int] = None
                           ) -> tuple[set[int], dict[int, int]]:
    """Apply setup-order post-processing: Puppet cure + Alchemist cures.

    Returns (final_corrupted, alchemist_cures) where alchemist_cures maps
    Alchemist position -> number of positions cured (for validator use).
    """
    result = set(corrupted)

    # 1. Puppeteer removes corruption from Puppet
    # If Puppeteer converts a corrupted Villager, corruption is removed
    if puppet_pos is not None and puppet_pos in result:
        result.discard(puppet_pos)

    # 2. Alchemist cures corruption (acts after all corruption sources)
    # Find all apparent Alchemist positions
    alchemist_positions = []
    for card in state.cards:
        if card.apparent_role == "Alchemist":
            alchemist_positions.append(card.position)

    # Reverse seat order (higher seat acts first)
    alchemist_positions.sort(reverse=True)
    alch_cures = {}

    for alch_pos in alchemist_positions:
        # Evil Alchemist can't cure
        if alch_pos in placement:
            alch_cures[alch_pos] = 0
            continue
        # Corrupted Alchemist can't cure
        if alch_pos in result:
            alch_cures[alch_pos] = 0
            continue
        # Cure positions in range 2 (except Drunk — can't be cured)
        in_range = positions_in_range(alch_pos, 2, state.n_cards)
        count = 0
        for p in in_range:
            if p in result and p != drunk_pos:
                # Also skip explicitly revealed Drunk
                card_at = _get_card_at(p, state)
                if card_at and card_at.apparent_role == "Drunk":
                    continue
                result.discard(p)
                count += 1
        alch_cures[alch_pos] = count

    return result, alch_cures


def _corruption_variants(placement: dict[int, str], state: GameState,
                         pd_target: Optional[int] = None,
                         poisoner_target: Optional[int] = None) -> list[set[int]]:
    """Generate corruption set variants accounting for Doppelganger immunity.

    If Doppelganger is in the deck, any apparent Villager adjacent to a
    corruption source could be Doppelganger (Outcast, immune). Returns all
    distinct corruption sets to try.
    """
    has_doppelganger = "Doppelganger" in state.deck.outcasts
    if not has_doppelganger:
        return [_compute_corruption(placement, state, pd_target, poisoner_target=poisoner_target)]

    n = state.n_cards
    # Find positions near corruption sources that appear as Villagers
    corruption_nearby = set()
    for pos, role in placement.items():
        if role == "Pooka":
            for adj in adjacent_positions(pos, n):
                if adj not in placement:
                    card = _get_card_at(adj, state)
                    if card and _is_villager_role(card.apparent_role, state):
                        corruption_nearby.add(adj)
        elif role == "Poisoner":
            for p in adjacent_positions(pos, n):
                if p not in placement:
                    card = _get_card_at(p, state)
                    if card and _is_villager_role(card.apparent_role, state):
                        corruption_nearby.add(p)

    # Try Doppelganger at each relevant position (or nowhere relevant)
    dopp_options = [None] + list(corruption_nearby)
    seen = set()
    variants = []
    for dopp_pos in dopp_options:
        corrupted = _compute_corruption(placement, state, pd_target, dopp_pos, poisoner_target)
        key = frozenset(corrupted)
        if key not in seen:
            seen.add(key)
            variants.append(corrupted)
    return variants


def _get_card_at(pos: int, state: GameState) -> Optional[CardInfo]:
    """Get revealed card at position, or None."""
    for card in state.cards:
        if card.position == pos:
            return card
    return None


def get_card_at(pos: int, state: GameState) -> Optional[CardInfo]:
    """Public query helper for revealed card lookup."""
    return _get_card_at(pos, state)


def _is_villager_role(role_name: str, state: GameState) -> bool:
    """Check if a role name is a Villager (corruption target)."""
    card_def = get_card(role_name)
    if card_def:
        return card_def.role == Role.VILLAGER
    return role_name in state.deck.villagers


def _unrevealed_must_be_villager(pos: int, evil_positions: dict[int, str],
                                  state: GameState,
                                  doppelganger_pos: Optional[int] = None,
                                  drunk_pos: Optional[int] = None) -> bool:
    """Check if an unrevealed good position must be a Villager.

    When all Outcast slots on the board are occupied by other positions,
    any remaining unrevealed good position must be a Villager.
    evil_positions should include ALL evil positions (remaining + executed).
    """
    max_outcasts = state.board_outcast_count
    if max_outcasts is None:
        max_outcasts = len(state.deck.outcasts)

    occupied = 0
    for card in state.cards:
        if card.position in evil_positions or card.position == pos:
            continue
        card_def = get_card(card.apparent_role)
        if card_def and card_def.role == Role.OUTCAST:
            occupied += 1

    # Doppelganger/Drunk take outcast slots but appear as Villagers
    if doppelganger_pos is not None and doppelganger_pos != pos and doppelganger_pos not in evil_positions:
        occupied += 1
    if drunk_pos is not None and drunk_pos != pos and drunk_pos not in evil_positions:
        occupied += 1

    return occupied >= max_outcasts


def _hidden_outcast_presence_flags(role_name: str, state: GameState) -> tuple[bool, bool]:
    """Return (can_be_on_board, can_be_absent) for a hidden outcast role.

    Uses board_outcast_count when available. This matters for hidden outcasts
    like Doppelganger/Drunk: if the board header says there is exactly one
    outcast and the pool's only outcast is Doppelganger, then Doppelganger
    cannot be treated as optional.
    """
    if role_name not in state.deck.outcasts:
        return (False, True)

    slots = state.board_outcast_count
    if slots is None:
        return (True, True)

    other_outcasts = len(state.deck.outcasts) - 1
    can_be_on = slots > 0
    can_be_absent = other_outcasts >= slots
    return (can_be_on, can_be_absent)


def _get_position_type(pos: int, scenario: Scenario, state: GameState) -> Optional[str]:
    """Get the type category of a position: 'Villager', 'Outcast', 'Minion', or 'Demon'.
    Returns None if type can't be determined (unrevealed good card)."""
    role = _known_evil_role(pos, scenario, state)
    if role is not None:
        if role in state.deck.demons:
            return "Demon"
        if role == "Unknown":
            return None
        return "Minion"  # Minion, Puppet, Witch, etc.

    if pos == scenario.doppelganger_position or pos == scenario.drunk_position:
        return "Outcast"

    card = _get_card_at(pos, state)
    if card is None:
        return None  # Unrevealed, can't determine

    # Wretch registers as Evil Minion to abilities (wiki: Bishop sees Wretch as "Minion")
    if card.apparent_role == "Wretch":
        return "Minion"

    role_name = card.apparent_role
    # Check knowledge base
    card_def = get_card(role_name)
    if card_def:
        return card_def.role.value  # "Villager", "Outcast", "Minion", "Demon"
    # Fallback: check deck lists
    norm = role_name.replace(" ", "_")
    for lst, typ in [(state.deck.outcasts, "Outcast"), (state.deck.villagers, "Villager"),
                     (state.deck.minions, "Minion"), (state.deck.demons, "Demon")]:
        if role_name in lst or norm in lst:
            return typ
    return "Villager"


def _get_real_role(pos: int, scenario: Scenario, state: GameState) -> str:
    """Get the real role of a position in a scenario."""
    known_evil_role = _known_evil_role(pos, scenario, state)
    if known_evil_role is not None:
        return known_evil_role
    if pos == scenario.doppelganger_position:
        return "Doppelganger"
    if pos == scenario.drunk_position:
        return "Drunk"
    card = _get_card_at(pos, state)
    if card:
        return card.apparent_role
    return "Unknown"


def get_real_role(pos: int, scenario: Scenario, state: GameState) -> str:
    """Public query helper for a position's real role in a scenario."""
    return _get_real_role(pos, scenario, state)


def _known_evil_role(pos: int, scenario: Scenario, state: GameState) -> Optional[str]:
    """Return the evil role at a position, including already executed evil cards."""
    if pos in scenario.evil_positions:
        return scenario.evil_positions[pos]
    if pos == scenario.puppet_position:
        return "Puppet"
    if pos in state.executed_evil_roles:
        return state.executed_evil_roles[pos]
    if pos in state.confirmed_evil and pos in state.executed:
        return "Unknown"
    return None


def _is_evil_in_board_state(pos: int, scenario: Scenario, state: GameState) -> bool:
    """Check if a position should still count as evil for clue validation."""
    return _known_evil_role(pos, scenario, state) is not None


def _is_evil_in_scenario(pos: int, scenario: Scenario) -> bool:
    """Check if a position is evil in this scenario (includes Puppet)."""
    return pos in scenario.evil_positions or pos == scenario.puppet_position


def scenario_is_evil(pos: int, scenario: Scenario) -> bool:
    """Public query helper for scenario evil membership."""
    return _is_evil_in_scenario(pos, scenario)


def _effective_alignment(pos: int, scenario: Scenario, state: GameState) -> Alignment:
    """Effective alignment for ability purposes. Wretch registers as Evil."""
    if _is_evil_in_board_state(pos, scenario, state):
        return Alignment.EVIL
    card = _get_card_at(pos, state)
    if card and card.apparent_role == "Wretch":
        return Alignment.EVIL  # Wretch registers as Evil to abilities
    return Alignment.GOOD


def effective_alignment(pos: int, scenario: Scenario, state: GameState) -> Alignment:
    """Public query helper for effective alignment in a scenario."""
    return _effective_alignment(pos, scenario, state)


def _truth_status(pos: int, scenario: Scenario, state: GameState) -> TruthStatus:
    """Determine if a card tells truth or lies in this scenario."""
    # Confessor can't lie — always truthful regardless of Evil/Corrupted status.
    # This affects Judge validation: Judge sees Confessor as "truthful" even if Evil.
    card = _get_card_at(pos, state)
    if card and card.apparent_role == "Confessor":
        return TruthStatus.TRUTHFUL

    # Evil characters lie (except Puppet)
    role = _known_evil_role(pos, scenario, state)
    if role is not None:
        if role == "Puppet":
            return TruthStatus.TRUTHFUL
        return TruthStatus.LYING

    # Corrupted characters lie
    if pos in scenario.corrupted:
        return TruthStatus.LYING

    # Good uncorrupted = truthful
    return TruthStatus.TRUTHFUL


def truth_status(pos: int, scenario: Scenario, state: GameState) -> TruthStatus:
    """Public query helper for whether a position tells the truth in a scenario."""
    return _truth_status(pos, scenario, state)


# ============================================================
# Info Validators
# ============================================================

def _validate_enlightened(card: CardInfo, scenario: Scenario,
                          state: GameState) -> bool:
    """Enlightened: 'CW', 'CCW', or 'Equidistant' to nearest Evil."""
    info = card.info_parsed
    if "direction" not in info:
        return True  # Can't validate without parsed info

    raw = info["direction"].lower()
    claimed = {"cw": "CW", "ccw": "CCW", "equidistant": "Equidistant", "equal": "Equidistant"}.get(raw, raw.capitalize())
    pos = card.position
    n = state.n_cards
    truth = _truth_status(pos, scenario, state)

    # Find closest evil(s) and their direction (exclude self)
    # Wretch counts as Evil for Enlightened (wiki confirmed)
    evil_positions = [p for p in range(1, n + 1)
                      if _effective_alignment(p, scenario, state) == Alignment.EVIL and p != pos]
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
    # Don't exclude executed positions — Knitter info reflects board at reveal time
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
    """Confessor: 'I am dizzy' if Evil or Corrupted. Can't lie."""
    info = card.info_parsed
    # Support both old "dirty" key and new "dizzy" key
    if "dizzy" in info:
        claimed_dizzy = info["dizzy"]
    elif "dirty" in info:
        claimed_dizzy = info["dirty"]
    else:
        return True

    pos = card.position

    # Confessor CAN'T LIE — always tells truth regardless of corruption/evil
    is_evil = _is_evil_in_board_state(pos, scenario, state)
    is_corrupted = pos in scenario.corrupted
    actually_dizzy = is_evil or is_corrupted

    return claimed_dizzy == actually_dizzy


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
        if not actual_good:
            return False
        # Wiki: Gemcrafter can't point at self unless no other Good targets exist
        if claimed_pos == pos:
            other_good = any(
                _effective_alignment(p, scenario, state) == Alignment.GOOD
                for p in range(1, state.n_cards + 1)
                if p != pos
            )
            if other_good:
                return False  # Self-pointing invalid when other Good targets exist
        return True
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
    for p in range(1, n + 1):
        if _known_evil_role(p, scenario, state) == evil_role:
            target_pos = p
            break
    if target_pos is None:
        return True  # Can't validate

    # Find distance to closest OTHER evil (Wretch registers as Evil)
    other_evil = [p for p in range(1, n + 1)
                  if _effective_alignment(p, scenario, state) == Alignment.EVIL
                  and p != target_pos]
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

    # -1 sentinel means "no corrupted characters exist" (binary claim)
    if claimed == -1:
        no_corrupted = len(scenario.corrupted) == 0
        if truth == TruthStatus.TRUTHFUL:
            return no_corrupted
        else:
            return not no_corrupted

    # Distance claim: "I am N cards from closest corrupted"
    if not scenario.corrupted:
        # Claimed a distance but no corrupted exist — truthful would say "none" (−1)
        if truth == TruthStatus.TRUTHFUL:
            return False
        else:
            return True  # Liar claimed a distance when there are none
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

    # Check if either target actually IS that minion role.
    # Wretch registers as a random Evil Minion — Oracle can identify Wretch
    # as any Minion role (wiki: "Bishop can only see Wretch as a Minion").
    def _target_matches(t):
        if _known_evil_role(t, scenario, state) == minion_role:
            return True
        # Wretch registers as any Evil Minion to Oracle
        card_at = _get_card_at(t, state)
        if card_at and card_at.apparent_role == "Wretch":
            # Wretch can match any minion role in the deck
            return minion_role in state.deck.minions
        return False

    actual = any(_target_matches(t) for t in targets)

    if truth == TruthStatus.TRUTHFUL:
        return actual
    else:
        # Wiki: lying Oracle can't include two Evil characters — both targets must be Good
        if not all(_effective_alignment(t, scenario, state) == Alignment.GOOD for t in targets):
            return False
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
    # Normalize spaces/underscores for comparison (e.g., "Fortune Teller" vs "Fortune_Teller")
    actual_match = is_good and actual_role.replace(" ", "_") == claimed_role.replace(" ", "_")

    if truth == TruthStatus.TRUTHFUL:
        return actual_match
    else:
        # Wiki: lying Medium points at Evil, Drunk, or Doppelganger (not normal Good)
        target_is_evil = _is_evil_in_board_state(claimed_pos, scenario, state)
        target_is_drunk = claimed_pos == scenario.drunk_position
        target_is_dopp = claimed_pos == scenario.doppelganger_position
        if not (target_is_evil or target_is_drunk or target_is_dopp):
            return False
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

    # Wretch registers as Evil for Hunter (wiki confirmed)
    evil_positions = [p for p in range(1, n + 1)
                      if _effective_alignment(p, scenario, state) == Alignment.EVIL and p != pos]
    if not evil_positions:
        return True

    actual = min(circle_distance(pos, ep, n) for ep in evil_positions)

    if truth == TruthStatus.TRUTHFUL:
        return claimed == actual
    else:
        return claimed != actual


def _architect_sides(n: int) -> tuple[set[int], set[int], set[int]]:
    """Return (left, right, both) position sets for n-card circle.

    The board has a fixed vertical line down the center.
    Card #N is always at top (12 o'clock), cards go CW: N, 1, 2, ...
    Right side = CW half (positions 1 .. floor(N/2)-1 for even, 1..floor(N/2) for odd)
    Left side = CCW half (positions floor(N/2)+1 .. N-1)
    Both = top (#N), and for even N also bottom (#N/2).
    Positions in 'both' count toward BOTH left and right evil totals.
    """
    half = n // 2
    both = {n}
    right = set()
    left = set()

    if n % 2 == 0:
        both.add(half)
        for i in range(1, half):
            right.add(i)
        for i in range(half + 1, n):
            left.add(i)
    else:
        for i in range(1, half + 1):
            right.add(i)
        for i in range(half + 1, n):
            left.add(i)

    return left, right, both


def _validate_architect(card: CardInfo, scenario: Scenario,
                        state: GameState) -> bool:
    """Architect: 'Left'/'Right'/'Equal' — which side has more Evil.

    The board is split by a fixed vertical line (not relative to Architect).
    Card #N at top and #N/2 at bottom (even N) are on the line = BOTH sides.
    Wretch counts as Evil (wiki confirmed).
    """
    info = card.info_parsed
    if "side" not in info:
        return True

    raw_side = info["side"].lower()
    claimed = {"left": "Left", "cw": "Left", "right": "Right", "ccw": "Right",
               "equal": "Equal", "equidistant": "Equal", "both": "Equal"}.get(
               raw_side, raw_side.capitalize())
    pos = card.position
    n = state.n_cards
    truth = _truth_status(pos, scenario, state)

    left_set, right_set, both_set = _architect_sides(n)

    left_count = 0
    right_count = 0
    for p in range(1, n + 1):
        if _effective_alignment(p, scenario, state) == Alignment.EVIL:
            if p in both_set:
                left_count += 1
                right_count += 1
            elif p in left_set:
                left_count += 1
            elif p in right_set:
                right_count += 1

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

    # Wiki: Empress can't include self in targets unless she is a Puppet
    if pos in targets and pos != scenario.puppet_position:
        return False

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

    # "Affected by evil ability" = corrupted, puppeted, or killed by Lilis
    actually_affected = (
        claimed_pos in scenario.corrupted or
        claimed_pos == scenario.puppet_position or
        claimed_pos in state.night_kills
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

    target_is_evil = _is_evil_in_board_state(target, scenario, state)
    # Wretch registers as Evil Minion — Dreamer sees "Cabbage" for Wretch
    target_card = _get_card_at(target, state)
    target_is_wretch = target_card and target_card.apparent_role == "Wretch"

    evil_roles = [
        role for p in range(1, state.n_cards + 1)
        if (role := _known_evil_role(p, scenario, state)) not in (None, "Puppet", "Unknown")
    ]

    if truth == TruthStatus.TRUTHFUL:
        if target_is_wretch:
            # Wiki: "When targeting a Wretch, a truthful Dreamer always reports 'a Cabbage'"
            return claimed_role.lower() == "cabbage"
        elif target_is_evil:
            # Must show correct evil role
            actual_role = _known_evil_role(target, scenario, state)
            return claimed_role == actual_role
        else:
            # Shows a random evil role (any evil role is valid)
            return claimed_role in evil_roles
    else:
        # Lying — info is false
        if target_is_wretch:
            # Lying Dreamer on Wretch: would NOT say Cabbage
            return claimed_role.lower() != "cabbage"
        elif target_is_evil:
            actual_role = _known_evil_role(target, scenario, state)
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

    # Corrupted Judge's ability "doesn't work" — result is unreliable,
    # NOT a clean inversion like an evil Judge. Skip validation entirely.
    if pos in scenario.corrupted:
        return True

    truth = _truth_status(pos, scenario, state)

    actual_lying = _truth_status(target, scenario, state) == TruthStatus.LYING

    if truth == TruthStatus.TRUTHFUL:
        return claimed_lying == actual_lying
    else:
        return claimed_lying != actual_lying


def _validate_alchemist(card: CardInfo, scenario: Scenario,
                        state: GameState) -> bool:
    """Alchemist: 'Cured N' — cures corruption in range 2.

    Uses pre-computed cure count from scenario.alchemist_cures (computed
    during setup-order post-processing, before cures are applied to the
    corruption set).
    """
    info = card.info_parsed
    if "cured_count" not in info:
        return True

    claimed = info["cured_count"]
    pos = card.position
    truth = _truth_status(pos, scenario, state)

    # Use pre-computed cure count (accounts for setup order, Drunk immunity,
    # corrupted/evil Alchemist can't cure, reverse seat order)
    actual = scenario.alchemist_cures.get(pos, 0)

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
    has_unrevealed_good_target = False
    for t in targets:
        if _is_evil_in_board_state(t, scenario, state):
            continue
        if t == scenario.doppelganger_position:
            actual_outcasts.append("Doppelganger")
            continue
        if t == scenario.drunk_position:
            actual_outcasts.append("Drunk")
            continue
        card_at = _get_card_at(t, state)
        if card_at:
            card_def = get_card(card_at.apparent_role)
            if card_def and card_def.role == Role.OUTCAST:
                if card_at.apparent_role == "Wretch":
                    continue  # Wretch registers as Evil to abilities
                actual_outcasts.append(card_at.apparent_role)
        else:
            # Unrevealed Good target — could be any Good role including Outcast
            has_unrevealed_good_target = True

    has_outcast = len(actual_outcasts) > 0

    if truth == TruthStatus.TRUTHFUL:
        if found_outcast:
            # Outcast found: must be among targets. If unrevealed Good targets exist,
            # the Outcast could plausibly be one of them.
            return has_outcast and found_outcast in actual_outcasts or has_unrevealed_good_target
        else:
            # No outcast found: no revealed outcasts AND no unrevealed Good targets
            # that could be an outcast (conservative: allow if unrevealed exist)
            return not has_outcast and not has_unrevealed_good_target
    else:
        if found_outcast:
            # Lying about finding outcast: truth is no outcast among targets
            # Only valid if no revealed outcasts AND no unrevealed Good targets
            return not has_outcast and not has_unrevealed_good_target
        else:
            # Lying about no outcast: truth is there IS an outcast
            return has_outcast or has_unrevealed_good_target


def _validate_bishop(card: CardInfo, scenario: Scenario,
                     state: GameState) -> bool:
    """Bishop: 'Learn 3 chars' — reports their type categories (Villager/Outcast/Minion/Demon)."""
    info = card.info_parsed
    if "targets" not in info:
        return True

    targets = info["targets"]
    pos = card.position
    truth = _truth_status(pos, scenario, state)
    claimed_types = info.get("types")

    # Full type validation if claimed types are provided
    if claimed_types:
        actual_types = []
        for t in targets:
            t_type = _get_position_type(t, scenario, state)
            if t_type is None:
                # Can't determine type (unrevealed good card) — fall through to weak check
                actual_types = None
                break
            actual_types.append(t_type)

        if actual_types is not None:
            types_match = sorted(actual_types) == sorted(claimed_types)
            if truth == TruthStatus.TRUTHFUL:
                return types_match
            else:
                return not types_match

    # Weak validation fallback (no claimed types or unrevealed targets)
    has_evil = any(_effective_alignment(t, scenario, state) == Alignment.EVIL
                   for t in targets)
    if truth == TruthStatus.TRUTHFUL:
        return has_evil
    else:
        return True


def _validate_bounty_hunter(card: CardInfo, scenario: Scenario,
                            state: GameState) -> bool:
    """Pseudo-validator for Poet's direct evil-call clue variant."""
    info = card.info_parsed
    if "evil_position" not in info:
        return True

    claimed_pos = info["evil_position"]
    pos = card.position
    truth = _truth_status(pos, scenario, state)

    actual_evil = _effective_alignment(claimed_pos, scenario, state) == Alignment.EVIL

    if truth == TruthStatus.TRUTHFUL:
        return actual_evil
    else:
        return not actual_evil


def _validate_slayer_results(scenario: Scenario, state: GameState) -> bool:
    """Validate Slayer ability results against a scenario.

    Each result is {slayer_pos, target_pos, killed}.
    - "killed": Slayer must be good+uncorrupted AND target must be evil.
    - "couldn't kill": Inconsistent ONLY if Slayer is good+uncorrupted AND target is evil
      (a working Slayer facing an evil target MUST kill).
    """
    for result in state.slayer_results:
        slayer_pos = result["slayer_pos"]
        target_pos = result["target_pos"]
        killed = result["killed"]

        # Cache evil role lookup (avoids duplicate traversal via _is_evil_in_board_state)
        slayer_evil_role = _known_evil_role(slayer_pos, scenario, state)
        slayer_is_evil = slayer_evil_role is not None
        # Puppet disguised as Slayer can use the ability (Puppet can't lie,
        # so its abilities work).  Other evil Slayers cannot.
        slayer_is_puppet = slayer_evil_role == "Puppet"
        slayer_is_corrupted = slayer_pos in scenario.corrupted
        target_is_evil = _effective_alignment(target_pos, scenario, state) == Alignment.EVIL

        if killed:
            # Kill succeeded: Slayer must be real (good or Puppet) + uncorrupted + target evil
            if slayer_is_evil and not slayer_is_puppet:
                return False  # Evil (non-Puppet) can't really use Slayer ability
            if slayer_is_corrupted:
                return False  # Corrupted Slayer can't kill
            if not target_is_evil:
                return False  # Can't kill a good target
        else:
            # Couldn't kill: inconsistent if a working Slayer faces an evil target
            slayer_works = (not slayer_is_evil or slayer_is_puppet) and not slayer_is_corrupted
            if slayer_works and target_is_evil:
                return False  # Working Slayer MUST kill evil target

    return True


def _validate_pd_ability(scenario: Scenario, state: GameState) -> bool:
    """Validate Plague Doctor active ability results.

    PD picks a target and checks if they're corrupted.
    - Good PD + corrupted target: says "Is Corrupted" + reveals an evil position (truthful)
    - Good PD + clean target: says "Not Corrupted" (truthful)
    - Evil PD + corrupted target: lies, says "Not Corrupted"
    - Evil PD + clean target: lies, says "Is Corrupted" + points to a good position as evil

    PD is an Outcast, immune to corruption, so no corrupted-PD case.
    """
    for result in state.pd_ability_results:
        pd_pos = result["pd_pos"]
        target = result["target"]
        claimed_corrupted = result["is_corrupted"]
        evil_revealed = result.get("evil_revealed")

        pd_is_evil = _is_evil_in_board_state(pd_pos, scenario, state)

        actual_corrupted = target in scenario.corrupted

        if pd_is_evil:
            # Evil PD lies about corruption status
            if claimed_corrupted == actual_corrupted:
                return False
            # Evil PD lies about evil reveal: claimed evil should be good
            if claimed_corrupted and evil_revealed is not None:
                if _is_evil_in_board_state(evil_revealed, scenario, state):
                    return False
        else:
            # Good PD tells truth about corruption
            if claimed_corrupted != actual_corrupted:
                return False
            # Good PD reveals a real evil
            if claimed_corrupted and evil_revealed is not None:
                if not _is_evil_in_board_state(evil_revealed, scenario, state):
                    return False

    return True


def _validate_baker(card: CardInfo, scenario: Scenario,
                    state: GameState) -> bool:
    """Baker: 'I am the original Baker' or 'I was a <Role>'.

    When a Baker is revealed, a random unrevealed Good Villager becomes a Baker
    and learns its original role. The 'original' Baker says 'I am the original Baker'.
    Converted Bakers say 'I was a <Role>'.

    Validation:
    - Truthful Baker claiming 'original': Baker must be in the deck (always true if Baker appears)
    - Truthful Baker claiming role X: X must be a Villager in the deck
    - Lying Baker: claimed info doesn't match reality (inverted checks)
    """
    info = card.info_parsed
    if "original_role" not in info:
        return True

    claimed = info["original_role"]
    pos = card.position
    truth = _truth_status(pos, scenario, state)

    if claimed.lower() == "original":
        # Claims to be the original Baker (not converted).
        # The original Baker is immune to corruption, and corrupted converted
        # Bakers say "I was [wrong role]", not "I am the original Baker".
        # So a good corrupted position claiming "original" is impossible.
        # Note: evil Bakers CAN claim "original" in-game (observed), despite
        # wiki saying otherwise. So we only reject corrupted good, not evil.
        if truth == TruthStatus.LYING:
            if not _is_evil_in_board_state(pos, scenario, state) and pos in scenario.corrupted:
                return False  # Corrupted good card can't claim "I am the original Baker"
        return True

    # Claims "I was a <Role>" — the role they were before Baker conversion
    # Normalize the claimed role for lookup
    claimed_card = get_card(claimed)
    if not claimed_card:
        return True  # Unknown role, can't validate

    is_villager = claimed_card.role == Role.VILLAGER

    if truth == TruthStatus.TRUTHFUL:
        # Must claim a real Villager role from the deck
        return is_villager and claimed_card.name in state.deck.villagers
    else:
        # Lying: claimed role is NOT a valid Villager from the deck,
        # OR the claim is simply wrong. Since we can't verify the exact
        # original role, we only reject if a truthful claim would be invalid.
        # A lie could claim anything, so allow it.
        return True


def _validate_poet(card: CardInfo, scenario: Scenario,
                    state: GameState) -> bool:
    """Poet: learns one random clue pattern.

    The clue type is encoded in `copied_role` for validator reuse, but it is
    not constrained to roles currently in play.
    """
    info = card.info_parsed
    copied_role = info.get("copied_role")
    if not copied_role:
        return True  # No copied role identified — can't validate

    # Look up the copied role's validator
    # (use late binding since VALIDATORS dict is defined below)
    validator = VALIDATORS.get(copied_role)
    if not validator:
        return True  # No validator for this copied role

    # Delegate: the card keeps its original position (for truth status)
    # but we pass the copied role's info fields through info_parsed
    return validator(card, scenario, state)


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
    "Bounty Hunter": _validate_bounty_hunter,
    "Baker": _validate_baker,
    "Poet": _validate_poet,
    # NOTE: "Knight" has no info to validate — Knight's ability is "I can't die"
    # (execution immunity). See EXECUTION_IMMUNE_ROLES below.
}

# Roles with execution immunity when Good and not corrupted.
# Evil disguised as Knight CAN be executed (immunity doesn't transfer).
# Corrupted Knight LOSES immunity. Doppelganger-as-Knight DOES block execution.
EXECUTION_IMMUNE_ROLES = {"Knight"}


# ============================================================
# Main Solver
# ============================================================

def _build_scenarios(state: GameState) -> list[Scenario]:
    """Build all candidate scenarios from evil placements."""
    placements = _generate_evil_placements(state)
    scenarios = []

    # Find ALL Plague Doctor positions (there may be 2 if evil disguised as PD)
    all_pd_positions = []
    for card in state.cards:
        if _is_plague_doctor(card.apparent_role):
            all_pd_positions.append(card.position)

    for placement in placements:
        if not _apply_placement_constraints(placement, state):
            continue

        puppet_pos = None
        for pos, role in placement.items():
            if role == "Puppet":
                puppet_pos = pos
                break

        # Build full evil set including executed evils (needed for corruption
        # computation, validator evil counting, and target enumeration)
        full_evil = dict(placement)
        for ex_pos, ex_role in state.executed_evil_roles.items():
            full_evil[ex_pos] = ex_role
        for ex_pos in state.confirmed_evil:
            if ex_pos in state.executed and ex_pos not in full_evil:
                full_evil[ex_pos] = "Unknown"

        # Determine PD corruption targets
        # PD corrupts 1 random Good Villager — we need to try all possibilities
        # For simplicity, if PD is in play and not evil, try each villager position
        # When multiple PD cards exist, use the one that's NOT evil in this placement
        pd_pos = None
        for p in all_pd_positions:
            if p not in placement:
                pd_pos = p
                break  # First non-evil PD is the real one

        pd_targets = [None]
        if pd_pos:
            # PD is good — it corrupted someone
            if any(_is_plague_doctor(o) for o in state.deck.outcasts):
                if state.pd_corruption_target is not None:
                    # Known PD target — only use that
                    pd_targets = [state.pd_corruption_target]
                else:
                    candidates = []
                    for p in range(1, state.n_cards + 1):
                        if p == pd_pos or p in full_evil:
                            continue
                        c = _get_card_at(p, state)
                        if c and _is_villager_role(c.apparent_role, state):
                            candidates.append(p)
                        elif c is None and _unrevealed_must_be_villager(
                                p, full_evil, state):
                            candidates.append(p)
                    if candidates:
                        pd_targets = candidates
                    else:
                        pd_targets = [None]

        # Compute Pooka-corrupted positions (for Poisoner to skip)
        pooka_corrupted = set()
        for pos, role in full_evil.items():
            if role == "Pooka":
                for adj in adjacent_positions(pos, state.n_cards):
                    if adj not in full_evil:
                        card = _get_card_at(adj, state)
                        if card and _is_villager_role(card.apparent_role, state):
                            pooka_corrupted.add(adj)
                        elif card is None and _unrevealed_must_be_villager(
                                adj, full_evil, state):
                            pooka_corrupted.add(adj)

        # Determine Poisoner corruption targets (adjacent, 1 Villager)
        # Poisoner acts AFTER Pooka — won't target already-corrupted positions
        poisoner_targets = [None]
        for pos, role in full_evil.items():
            if role == "Poisoner":
                candidates = []
                for p in adjacent_positions(pos, state.n_cards):
                    if p in full_evil:
                        continue
                    if p in pooka_corrupted:
                        continue  # Poisoner skips Pooka's victims
                    card = _get_card_at(p, state)
                    if card and _is_villager_role(card.apparent_role, state):
                        candidates.append(p)
                    elif card is None and _unrevealed_must_be_villager(
                            p, full_evil, state):
                        candidates.append(p)
                if candidates:
                    poisoner_targets = candidates
                break  # Only one Poisoner

        # Determine possible Doppelganger positions
        has_doppelganger = "Doppelganger" in state.deck.outcasts
        can_have_dopp, can_skip_dopp = _hidden_outcast_presence_flags("Doppelganger", state)
        dopp_candidates = [None] if can_skip_dopp else []
        if has_doppelganger and can_have_dopp:
            for p in range(1, state.n_cards + 1):
                if p in full_evil or p == puppet_pos:
                    continue
                # Don't skip executed positions — an executed Good card
                # could have been the Doppelganger
                card = _get_card_at(p, state)
                if card and _is_villager_role(card.apparent_role, state):
                    dopp_candidates.append(p)

        # Determine possible Drunk positions
        # Drunk (Outcast) disguises as a not-in-play Villager, is always
        # corrupted, and can't be cured. Enumerate possible positions.
        has_drunk = "Drunk" in state.deck.outcasts
        # Check if Drunk is already explicitly revealed (e.g., by Medium)
        drunk_already_known = any(
            c.apparent_role == "Drunk" for c in state.cards
        )
        can_have_drunk, can_skip_drunk = _hidden_outcast_presence_flags("Drunk", state)
        drunk_candidates = [None] if (drunk_already_known or can_skip_drunk) else []
        if has_drunk and not drunk_already_known and can_have_drunk:
            for p in range(1, state.n_cards + 1):
                if p in full_evil or p == puppet_pos:
                    continue
                # Don't skip executed positions — an executed Good card
                # could have been the Drunk, and its pre-execution info
                # still needs to be validated as corrupted
                card = _get_card_at(p, state)
                # Drunk appears as a Villager, so it could be at any
                # position showing as a Villager, OR any unrevealed position
                if card is None or _is_villager_role(card.apparent_role, state):
                    drunk_candidates.append(p)

        for pd_t in pd_targets:
            for pois_t in poisoner_targets:
                seen = set()
                for dopp_pos in dopp_candidates:
                    for drunk_pos in drunk_candidates:
                        # Drunk and Doppelganger can't be the same position
                        if drunk_pos is not None and drunk_pos == dopp_pos:
                            continue
                        raw_corrupted = _compute_corruption(
                            full_evil, state, pd_t, dopp_pos, pois_t,
                            drunk_pos)
                        # Apply setup-order post-processing:
                        # Puppet cure + Alchemist cures
                        final_corrupted, alch_cures = _apply_post_corruption(
                            raw_corrupted, full_evil, state,
                            puppet_pos, drunk_pos)
                        # Deduplicate scenarios with same corruption+dopp+drunk combo
                        key = (frozenset(final_corrupted), dopp_pos, drunk_pos)
                        if key in seen:
                            continue
                        seen.add(key)
                        scenario = Scenario(
                            evil_positions=dict(full_evil),
                            puppet_position=puppet_pos,
                            corrupted=final_corrupted,
                            pd_corrupted=pd_t,
                            doppelganger_position=dopp_pos,
                            drunk_position=drunk_pos,
                            alchemist_cures=alch_cures,
                        )
                        scenarios.append(scenario)

    return scenarios


def _validate_role_counts(scenario: Scenario, state: GameState) -> bool:
    """Check that apparent roles among Good positions don't exceed deck counts.

    If deck has 1 Hunter but 2 Good positions show Hunter, the excess must be
    explainable by disguising Outcasts (Drunk/Doppelganger) that are Good in
    this scenario. Evil positions are excluded (their apparent role is a disguise
    that doesn't consume a deck slot).

    Also checks Outcast role counts: Good positions showing an Outcast role
    can't exceed that role's deck count (no disguiser can fake an Outcast role).
    """
    from collections import Counter

    # Count apparent roles among Good (non-evil, non-puppet) revealed positions,
    # PLUS puppet (Puppet keeps its apparent Villager role, consuming a deck slot)
    good_villager_counts = Counter()  # villager_role -> count
    good_outcast_counts = Counter()   # outcast_role -> count

    deck_villager_set = set(state.deck.villagers)
    # Normalize outcast names for comparison (spaces vs underscores)
    deck_outcast_set = set(state.deck.outcasts) | set(o.replace("_", " ") for o in state.deck.outcasts)

    for card in state.cards:
        pos = card.position
        # Evil positions use a disguise — don't consume a Good deck slot
        if _known_evil_role(pos, scenario, state) is not None:
            continue
        # Puppet is Evil but keeps its original Villager appearance,
        # consuming a Villager deck slot
        role = card.apparent_role
        if _is_villager_role(role, state):
            good_villager_counts[role] += 1
        elif role in deck_outcast_set:
            good_outcast_counts[role] += 1

    # Count how many disguising Outcasts are Good in this scenario.
    # Drunk and Doppelganger appear as Villagers on the board but are
    # actually Outcasts — each can explain 1 extra apparent Villager.
    n_disguisers = 0
    disguise_outcasts = {"Drunk", "Doppelganger"}
    for outcast in state.deck.outcasts:
        if outcast not in disguise_outcasts:
            continue
        # Only hidden outcasts that are actually present in this scenario can
        # explain an extra apparent Villager copy. `None` means absent here,
        # not "present but unknown."
        if outcast == "Doppelganger" and scenario.doppelganger_position is not None:
            if scenario.doppelganger_position not in scenario.evil_positions:
                n_disguisers += 1
        elif outcast == "Drunk" and scenario.drunk_position is not None:
            if scenario.drunk_position not in scenario.evil_positions:
                n_disguisers += 1

    # Check Villager roles: excess Good appearances over deck count
    deck_v_counts = Counter(_normalize_role(v) for v in state.deck.villagers)
    total_excess = 0
    for role, count in good_villager_counts.items():
        # Baker converts other villagers into Bakers (cascading), so any
        # number of Good Bakers is valid when Baker is in the deck
        if _normalize_role(role) == "baker" and any(_normalize_role(v) == "baker" for v in state.deck.villagers):
            continue
        deck_count = deck_v_counts.get(_normalize_role(role), 0)
        if count > deck_count:
            total_excess += count - deck_count

    # Shaman creates "2 same Villager roles" — allows 1 extra copy of a Villager
    shaman_allowance = 0
    if "Shaman" in state.deck.minions:
        for p in range(1, state.n_cards + 1):
            role = _known_evil_role(p, scenario, state)
            if role == "Shaman":
                shaman_allowance = 1
                break

    # Shaman REQUIRES a duplicate: if Shaman is on the board, there MUST be
    # at least one Villager role appearing 2+ times among Good positions.
    # Note: total_excess skips Baker (conversion allows unlimited), so check
    # raw counts directly to catch Baker duplicates satisfying Shaman.
    # Only enforce when all Good positions are visible — the duplicate could
    # be at an unrevealed or night-killed position.
    if shaman_allowance > 0:
        has_any_duplicate = any(count >= 2 for count in good_villager_counts.values())
        if not has_any_duplicate:
            revealed_positions = {c.position for c in state.cards}
            # Night-killed positions are unrevealed — could hold the duplicate
            unrevealed_good = sum(1 for p in range(1, state.n_cards + 1)
                                 if p not in revealed_positions
                                 and _known_evil_role(p, scenario, state) is None)
            if unrevealed_good == 0:
                return False  # All Good positions visible, no villager duplicate

    if total_excess > n_disguisers + shaman_allowance:
        return False

    # Check Outcast roles: no disguiser can fake an Outcast appearance,
    # so Good Outcast appearances can't exceed deck counts
    deck_o_counts = Counter(_normalize_role(o) for o in state.deck.outcasts)
    for role, count in good_outcast_counts.items():
        if count > deck_o_counts.get(_normalize_role(role), 0):
            return False

    # Extra roles mechanic (Asc10+): pool has more roles than board positions.
    # Total good outcasts on board can't exceed actual board outcast count.
    if state.board_outcast_count is not None:
        total_good_outcasts = sum(good_outcast_counts.values())
        if total_good_outcasts > state.board_outcast_count:
            return False

    # Similarly, total good villager appearances can't exceed board villager
    # count + disguiser allowance (Drunk/Doppelganger appear as Villagers).
    if state.board_villager_count is not None:
        total_good_villagers = sum(good_villager_counts.values())
        if total_good_villagers > state.board_villager_count + n_disguisers + shaman_allowance:
            return False

    return True


def _check_scenario(scenario: Scenario, state: GameState) -> bool:
    """Check if a scenario is consistent with all revealed card info."""
    # Structural check: role counts must be explainable by deck + disguisers
    if not _validate_role_counts(scenario, state):
        return False

    for card in state.cards:
        # Skip only executed cards whose role/info we don't know at all.
        # Executed evil cards still constrain the board via the clue they gave
        # before dying, so keep validating them as liars when their apparent
        # role/info are known.
        if card.position in state.executed:
            if card.position in state.confirmed_evil and card.position not in state.executed_evil_roles:
                continue

        if _known_evil_role(card.position, scenario, state) is not None:
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

    # Validate Slayer ability results
    if state.slayer_results:
        if not _validate_slayer_results(scenario, state):
            return False

    # Validate Plague Doctor active ability results
    if state.pd_ability_results:
        if not _validate_pd_ability(scenario, state):
            return False

    # Validate Lilis night kill constraint: exactly N evils among night-killed positions
    if state.night_kills:
        evil_in_night_kills = sum(1 for p in state.night_kills
                                  if p in scenario.evil_positions)
        if evil_in_night_kills != state.night_kill_evil_count:
            return False

        # Knight immunity: Lilis can't kill uncorrupted Knight.
        # If Knight is in the deck, it can't be at a night-killed uncorrupted
        # Good position.  Reject scenarios where Knight has no valid placement.
        if "Knight" in state.deck.villagers:
            # Knight already revealed at a Good position → accounted for
            knight_revealed = any(
                c.apparent_role == "Knight" and c.position not in scenario.evil_positions
                for c in state.cards
            )
            if not knight_revealed:
                revealed_positions = {c.position for c in state.cards}
                # Unrevealed Good positions where Knight could be
                valid_for_knight = [
                    p for p in range(1, state.n_cards + 1)
                    if p not in scenario.evil_positions
                    and p not in revealed_positions
                    and not (p in state.night_kills and p not in scenario.corrupted)
                ]
                # In pool > board games, Knight might not be on the board
                pool_gt_board = (state.board_villager_count is not None
                                 and len(state.deck.villagers) > state.board_villager_count)
                if not valid_for_knight and not pool_gt_board:
                    return False

    return True


def _diagnose_zero_scenarios(scenarios: list[Scenario], state: GameState) -> list[str]:
    """When 0 scenarios survive, diagnose which card's info is causing the failure.

    Tries removing each card's info one at a time and reports which removals
    allow scenarios to survive — pinpointing likely misread info or missing mechanics.
    """
    diag = ["", "=== ZERO-SCENARIO DIAGNOSTICS ==="]

    if not scenarios:
        diag.append("No candidate scenarios were generated (structural constraint failure)")
        diag.append("Check: confirmed_evil/confirmed_good, deck composition, n_evil count")
        return diag

    # Count how many scenarios each validator rejects
    validator_rejections: dict[str, int] = {}
    for s in scenarios:
        for card in state.cards:
            role = card.apparent_role
            if role in VALIDATORS:
                if not VALIDATORS[role](card, s, state):
                    key = f"#{card.position} {role}"
                    validator_rejections[key] = validator_rejections.get(key, 0) + 1

    if validator_rejections:
        diag.append("Rejection counts (card -> how many scenarios it rejected):")
        for key, count in sorted(validator_rejections.items(), key=lambda x: -x[1]):
            pct = count / len(scenarios) * 100
            diag.append(f"  {key}: rejected {count}/{len(scenarios)} ({pct:.0f}%)")

    # Try removing each card's info one at a time
    diag.append("")
    diag.append("Leave-one-out analysis (removing each card's info):")
    cards_with_info = [c for c in state.cards if c.apparent_role in VALIDATORS]

    for skip_card in cards_with_info:
        # Build a modified state without this card's validator
        modified_cards = []
        for c in state.cards:
            if c.position == skip_card.position:
                # Replace with no_info version (skip validation)
                modified_cards.append(CardInfo(c.position, c.apparent_role, c.info_text, {}))
            else:
                modified_cards.append(c)
        modified_state = GameState(
            n_cards=state.n_cards, deck=state.deck, cards=modified_cards,
            n_evil=state.n_evil, executed=state.executed,
            confirmed_evil=state.confirmed_evil, confirmed_good=state.confirmed_good,
            pd_corruption_target=state.pd_corruption_target,
            executed_evil_roles=state.executed_evil_roles,
            hp=state.hp, wrong_exec_cost=state.wrong_exec_cost,
        )
        n_surviving = sum(1 for s in scenarios if _check_scenario(s, modified_state))
        if n_surviving > 0:
            diag.append(f"  WITHOUT #{skip_card.position} {skip_card.apparent_role}: "
                        f"{n_surviving} scenarios survive  <-- SUSPECT")
        else:
            diag.append(f"  WITHOUT #{skip_card.position} {skip_card.apparent_role}: still 0")

    return diag


def solve(state: GameState) -> SolverResult:
    """Main solver entry point."""
    scenarios = _build_scenarios(state)
    reasoning = [f"Generated {len(scenarios)} candidate scenarios"]

    surviving = [s for s in scenarios if _check_scenario(s, state)]
    reasoning.append(f"{len(surviving)} scenarios survived validation")

    if not surviving:
        diag = _diagnose_zero_scenarios(scenarios, state)
        return SolverResult(
            definite_evil=[], definite_good=[],
            bombardier_positions=[], n_scenarios=len(scenarios),
            n_surviving=0, surviving_scenarios=[],
            reasoning=reasoning + ["NO VALID SCENARIOS — check input data"] + diag,
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

    # Bombardier positions — never execute unless confirmed evil
    # (executing a good Bombardier = instant loss, so exclude ALL non-confirmed-evil Bombardiers)
    bombardier_positions = []
    for card in state.cards:
        if card.apparent_role == "Bombardier" and card.position not in definite_evil:
            bombardier_positions.append(card.position)

    # Knight (execution-immune) positions — can't execute, but confirms Good
    for card in state.cards:
        if card.apparent_role in EXECUTION_IMMUNE_ROLES and card.position not in definite_good:
            # If we attempted execution and it was blocked, the card is confirmed Good
            # (evil disguise wouldn't have the immunity)
            # Add to definite_good if not already there
            if all(not _is_evil_in_scenario(card.position, s) for s in surviving):
                if card.position not in definite_good:
                    definite_good.append(card.position)
                    reasoning.append(f"  #{card.position} is execution-immune ({card.apparent_role}) — confirmed GOOD")

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
