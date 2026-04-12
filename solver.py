# Types and helpers only — solve engine is in Rust (crates/solver-core).
# Python solve() has been removed. Use rust_solver.rust_solve_to_objects() instead.

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
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
    executed_good_corrupted: dict[int, bool] = field(default_factory=dict)  # Corruption status of executed good cards

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
            "executed_good_corrupted": {str(k): v for k, v in self.executed_good_corrupted.items()},
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
            executed_good_corrupted={int(k): v for k, v in data.get("executed_good_corrupted", {}).items()},
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
    chancellor_conversion: Optional[int] = None  # Position converted to Outcast by Chancellor


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


# Roles with execution immunity when Good and not corrupted.
# Evil disguised as Knight CAN be executed (immunity doesn't transfer).
# Corrupted Knight LOSES immunity. Doppelganger-as-Knight DOES block execution.
EXECUTION_IMMUNE_ROLES = {"Knight"}


# ============================================================
# Query Helpers (used by strategy.py, game_loop.py, rust_solver.py)
# ============================================================

# Module-level cache for the current state's card lookup
_card_lookup: dict[int, CardInfo] = {}
_card_lookup_id: int = -1  # id of the state.cards list we built from


def _build_card_lookup(state: GameState) -> dict[int, CardInfo]:
    """Build position -> CardInfo lookup dict for O(1) access."""
    return {card.position: card for card in state.cards}


def _get_card_at(pos: int, state: GameState) -> Optional[CardInfo]:
    """Get revealed card at position, or None. Uses cached dict lookup."""
    global _card_lookup, _card_lookup_id
    cards_id = id(state.cards)
    if cards_id != _card_lookup_id:
        _card_lookup = _build_card_lookup(state)
        _card_lookup_id = cards_id
    return _card_lookup.get(pos)


def get_card_at(pos: int, state: GameState) -> Optional[CardInfo]:
    """Public query helper for revealed card lookup."""
    return _get_card_at(pos, state)


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


if __name__ == "__main__":
    # Quick smoke test
    print("Solver module loaded successfully (types + helpers only)")
    print(f"Circle distance 1->4 on 7: {circle_distance(1, 4, 7)}")
    print(f"Circle direction 1->3 on 7: {circle_direction(1, 3, 7)}")
    print(f"Adjacent to 1 on 7: {adjacent_positions(1, 7)}")
