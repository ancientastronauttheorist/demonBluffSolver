# Types and helpers only — solve engine is in Rust (crates/solver-core).
# Python solve() has been removed. Use rust_solver.rust_solve_to_objects() instead.

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from knowledge_base import get_card, Role, Alignment, CARDS_BY_NAME


RAMBLER_RULE_VERSION = "rambler2_shut_up"
BAKER_RULE_VERSION = "baker_day_reveal_v1"
DOPPEL_DRUNK_RULE_VERSION = "doppel_drunk_reveal_v1"
FORTUNE_TELLER_RULE_VERSION = "fortune_teller_native_v1"
POET_VARIANT = "public_current"
TWIN_RECIPIENT_BLUFF_RULE_VERSION = "twin_recipient_bluff_native_v1"
TWIN_RECIPIENT_BLUFF_PREFIX_RULE_VERSION = (
    "twin_recipient_bluff_one_lilis_prefix_native_v1"
)

# Native Gossip constructor order.  These are canonical public clue-provider
# names, not a list of every Villager whose text happens to resemble a Poet
# result.  Bounty Hunter is a retained provider distinct from current Hunter.
POET_PROVIDER_ROLES = (
    "Lover",
    "Scout",
    "Oracle",
    "Bounty Hunter",
    "Medium",
    "Knitter",
    "Hunter",
    "Enlightened",
    "Empress",
    "Bishop",
    "Gemcrafter",
    "Bard",
)


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
class TwinRecipientBluffContext:
    """Offline-only hidden pool state at a moved Twin recipient's bluff draw."""

    rule_version: str
    recipient_position: int
    acquisition_ordinal: int
    duplicate_pool: list[str] = field(default_factory=list)
    unique_pool: list[str] = field(default_factory=list)
    bluff_must_include_at_recipient: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "rule_version": self.rule_version,
            "recipient_position": self.recipient_position,
            "acquisition_ordinal": self.acquisition_ordinal,
            "duplicate_pool": list(self.duplicate_pool),
            "unique_pool": list(self.unique_pool),
            "bluff_must_include_at_recipient": list(
                self.bluff_must_include_at_recipient
            ),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TwinRecipientBluffContext":
        if type(data) is not dict:
            raise TypeError("twin_recipient_bluff_context must be an exact dict")

        rule_version = data.get("rule_version")
        recipient_position = data.get("recipient_position")
        acquisition_ordinal = data.get("acquisition_ordinal")
        if type(rule_version) is not str or not rule_version.strip():
            raise ValueError(
                "twin_recipient_bluff_context.rule_version must be a nonempty str"
            )
        if (
            type(recipient_position) is not int
            or not 1 <= recipient_position <= 255
        ):
            raise ValueError(
                "twin_recipient_bluff_context.recipient_position "
                "must be an exact nonzero u8"
            )
        if type(acquisition_ordinal) is not int or not 0 <= acquisition_ordinal <= 65535:
            raise ValueError(
                "twin_recipient_bluff_context.acquisition_ordinal "
                "must be an exact u16"
            )

        pools = {}
        for key in (
            "duplicate_pool",
            "unique_pool",
            "bluff_must_include_at_recipient",
        ):
            raw_pool = data.get(key, [])
            if type(raw_pool) is not list:
                raise TypeError(f"twin_recipient_bluff_context.{key} must be a list")
            if any(type(role) is not str or not role.strip() for role in raw_pool):
                raise ValueError(
                    f"twin_recipient_bluff_context.{key} roles must be nonempty strs"
                )
            pools[key] = list(raw_pool)

        return cls(
            rule_version=rule_version,
            recipient_position=recipient_position,
            acquisition_ordinal=acquisition_ordinal,
            **pools,
        )


@dataclass(frozen=True)
class DelayedRevealAcquisitionEvent:
    """One successful hidden delayed-Reveal bluff acquisition."""

    position: int
    acquisition_ordinal: int

    def to_dict(self) -> dict:
        return {
            "position": self.position,
            "acquisition_ordinal": self.acquisition_ordinal,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DelayedRevealAcquisitionEvent":
        if type(data) is not dict or set(data) != {
            "position",
            "acquisition_ordinal",
        }:
            raise TypeError(
                "delayed-Reveal acquisition event must be an exact dict"
            )
        position = data["position"]
        acquisition_ordinal = data["acquisition_ordinal"]
        if type(position) is not int or not 1 <= position <= 255:
            raise ValueError(
                "delayed-Reveal acquisition position must be an exact nonzero u8"
            )
        if (
            type(acquisition_ordinal) is not int
            or not 0 <= acquisition_ordinal <= 65535
        ):
            raise ValueError(
                "delayed-Reveal acquisition ordinal must be an exact u16"
            )
        return cls(position, acquisition_ordinal)


@dataclass
class TwinRecipientBluffPrefixContext:
    """Offline-only one-Lilis acquisition prefix before a moved Twin draw."""

    rule_version: str
    acquisition_order: list[DelayedRevealAcquisitionEvent] = field(
        default_factory=list
    )
    bluff_must_include_before_prefix: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "rule_version": self.rule_version,
            "acquisition_order": [
                event.to_dict() for event in self.acquisition_order
            ],
            "bluff_must_include_before_prefix": list(
                self.bluff_must_include_before_prefix
            ),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TwinRecipientBluffPrefixContext":
        expected = {
            "rule_version",
            "acquisition_order",
            "bluff_must_include_before_prefix",
        }
        if type(data) is not dict or set(data) != expected:
            raise TypeError(
                "twin_recipient_bluff_prefix_context must be an exact dict"
            )
        rule_version = data["rule_version"]
        if type(rule_version) is not str or not rule_version.strip():
            raise ValueError(
                "twin_recipient_bluff_prefix_context.rule_version must be "
                "a nonempty str"
            )
        raw_order = data["acquisition_order"]
        if type(raw_order) is not list:
            raise TypeError(
                "twin_recipient_bluff_prefix_context.acquisition_order "
                "must be a list"
            )
        raw_pool = data["bluff_must_include_before_prefix"]
        if type(raw_pool) is not list:
            raise TypeError(
                "twin_recipient_bluff_prefix_context."
                "bluff_must_include_before_prefix must be a list"
            )
        if any(type(role) is not str or not role.strip() for role in raw_pool):
            raise ValueError(
                "twin_recipient_bluff_prefix_context."
                "bluff_must_include_before_prefix roles must be nonempty strs"
            )
        return cls(
            rule_version=rule_version,
            acquisition_order=[
                DelayedRevealAcquisitionEvent.from_dict(event)
                for event in raw_order
            ],
            bluff_must_include_before_prefix=list(raw_pool),
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
    # Hidden native Start history. Keep null during live solving; populated
    # only by an explicit offline fixture or post-mortem reconstruction.
    pd_corruption_target: Optional[int] = None
    executed_evil_roles: dict[int, str] = field(default_factory=dict)  # pos -> evil role name (e.g. {2: "Chancellor"})
    slayer_results: list[dict] = field(default_factory=list)  # [{slayer_pos, target_pos, killed, revealed_role?}]
    night_kills: list[int] = field(default_factory=list)  # Positions killed by Lilis night (unrevealed)
    night_kill_evil_count: int = 0  # How many of the night kills were evil
    hp: int = 10                    # Current health points
    wrong_exec_cost: int = 2        # HP lost per wrong execution (varies by ascension)
    pd_ability_results: list[dict] = field(default_factory=list)  # [{"pd_pos": N, "target": N, "is_corrupted": bool, "evil_revealed": N|None}]
    blocked_positions: list[int] = field(default_factory=list)  # Positions blocked from reveal (Witch)
    board_villager_count: Optional[int] = None  # Normalized pre-Start header V count
    board_outcast_count: Optional[int] = None   # Normalized pre-Start header O count
    board_minion_count: Optional[int] = None    # Actual minions on board (when pool > board)
    board_demon_count: Optional[int] = None     # Actual demons on board (when pool > board)
    reveal_order: list[int] = field(default_factory=list)  # Order positions were flipped (for Baker)
    executed_good_corrupted: dict[int, bool] = field(default_factory=dict)  # Corruption status of executed good cards
    executed_good_roles: dict[int, str] = field(default_factory=dict)  # Public current roles of executed good cards
    board_count_provenance: str = "legacy_unknown"  # Appended for positional ABI compatibility
    # Missing means an archived pre-audit fixture.  Fresh live sessions opt in
    # explicitly so absence of a shut-up observation can constrain Rambler2.
    rambler_rule_version: Optional[str] = None
    # Ordered public Rambler2 replacements.  The scalar stored on each card is
    # only the latest-value compatibility alias; this ledger survives later
    # ResetAfterNight results on the same speaker.
    rambler_shut_up_observations: list[dict] = field(default_factory=list)
    # Missing means an archived pre-audit Baker fixture/session. Fresh live
    # sessions opt in explicitly to the shipped Day/reveal-order semantics.
    baker_rule_version: Optional[str] = None
    # Missing means an archived pre-audit Doppelganger/Drunk fixture/session.
    # Fresh live sessions opt into the shipped delayed-Reveal source rules.
    doppel_drunk_rule_version: Optional[str] = None
    # Missing means an archived pre-audit Fortune Teller fixture/session.
    # Fresh live sessions retain the exact native speech/reference history.
    fortune_teller_rule_version: Optional[str] = None
    # Public terminal marker set only after a non-Night death reveals the
    # canonical current CharacterData role Bombardier. Missing preserves
    # legacy fixtures and in-progress games.
    terminal_loss_role: Optional[str] = None
    # Exact public current CharacterData revealed by successful ordinary
    # executions. Appended for positional ABI compatibility; original Evil
    # identity remains in executed_evil_roles.
    executed_current_roles: dict[int, str] = field(default_factory=dict)
    # Exact public current role later revealed for a hidden night victim (for
    # example by Medium). Kept separate from ordinary-execution evidence.
    revealed_night_current_roles: dict[int, str] = field(default_factory=dict)
    # Offline/post-mortem pool snapshot for one exact moved Twin recipient.
    # Live play must leave this absent because the pool state is hidden.
    twin_recipient_bluff_context: Optional[TwinRecipientBluffContext] = None
    # Offline-only explicit delayed-Reveal prefix. Never inferred from the
    # public player click order and never populated by a live GameSession.
    twin_recipient_bluff_prefix_context: Optional[
        TwinRecipientBluffPrefixContext
    ] = None

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
            "board_count_provenance": self.board_count_provenance,
            "reveal_order": list(self.reveal_order),
            "executed_good_corrupted": {str(k): v for k, v in self.executed_good_corrupted.items()},
            "executed_good_roles": {str(k): v for k, v in self.executed_good_roles.items()},
            "executed_current_roles": {
                str(k): v for k, v in self.executed_current_roles.items()
            },
            "revealed_night_current_roles": {
                str(k): v
                for k, v in self.revealed_night_current_roles.items()
            },
            "rambler_shut_up_observations": [
                dict(observation)
                for observation in self.rambler_shut_up_observations
            ],
        }
        if self.rambler_rule_version is not None:
            data["rambler_rule_version"] = self.rambler_rule_version
        if self.baker_rule_version is not None:
            data["baker_rule_version"] = self.baker_rule_version
        if self.doppel_drunk_rule_version is not None:
            data["doppel_drunk_rule_version"] = self.doppel_drunk_rule_version
        if self.fortune_teller_rule_version is not None:
            data["fortune_teller_rule_version"] = self.fortune_teller_rule_version
        if self.terminal_loss_role is not None:
            data["terminal_loss_role"] = self.terminal_loss_role
        if self.twin_recipient_bluff_context is not None:
            data["twin_recipient_bluff_context"] = (
                self.twin_recipient_bluff_context.to_dict()
            )
        if self.twin_recipient_bluff_prefix_context is not None:
            data["twin_recipient_bluff_prefix_context"] = (
                self.twin_recipient_bluff_prefix_context.to_dict()
            )
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
            board_count_provenance=data.get("board_count_provenance", "legacy_unknown"),
            reveal_order=list(data.get("reveal_order", [])),
            executed_good_corrupted={int(k): v for k, v in data.get("executed_good_corrupted", {}).items()},
            executed_good_roles={int(k): v for k, v in data.get("executed_good_roles", {}).items()},
            rambler_rule_version=data.get("rambler_rule_version"),
            rambler_shut_up_observations=[
                dict(observation)
                for observation in data.get("rambler_shut_up_observations", [])
            ],
            baker_rule_version=data.get("baker_rule_version"),
            doppel_drunk_rule_version=data.get("doppel_drunk_rule_version"),
            fortune_teller_rule_version=data.get("fortune_teller_rule_version"),
            terminal_loss_role=data.get("terminal_loss_role"),
            executed_current_roles={
                int(k): v
                for k, v in data.get("executed_current_roles", {}).items()
            },
            revealed_night_current_roles={
                int(k): v
                for k, v in data.get(
                    "revealed_night_current_roles", {}
                ).items()
            },
            twin_recipient_bluff_context=(
                TwinRecipientBluffContext.from_dict(
                    data["twin_recipient_bluff_context"]
                )
                if data.get("twin_recipient_bluff_context") is not None
                else None
            ),
            twin_recipient_bluff_prefix_context=(
                TwinRecipientBluffPrefixContext.from_dict(
                    data["twin_recipient_bluff_prefix_context"]
                )
                if data.get("twin_recipient_bluff_prefix_context") is not None
                else None
            ),
        )


def slayer_revealed_role(result: dict) -> Optional[str]:
    """Return a Slayer kill's public role, accepting historical saves."""
    return result.get("revealed_role") or result.get("evil_role")


@dataclass
class ChancellorTrace:
    """Probability-safe projection of Chancellor's native Start relocation.

    ``original_positions`` and ``affected_anchor_positions`` are grouped native
    history alternatives that converge to one represented final board.  Anchor
    positions are provenance only: ``Scenario.messed_up_by_evil`` is the
    authoritative set of markers that survived resistance and later Start
    actions.
    """
    original_positions: list[int] = field(default_factory=list)
    added_outcast_position: int = 0
    added_outcast_role: str = ""
    # Appended to preserve the positional constructor used by older callers.
    affected_anchor_positions: list[int] = field(default_factory=list)


@dataclass
class ShamanTrace:
    """Current-role projection of Shaman's ordered Start overwrite."""

    source_position: int
    target_position: int
    copied_role: str
    target_previous_roles: list[str] = field(default_factory=list)


class TwinNeighborSide(str, Enum):
    """Exact occurrence selected from native ``[previous, next]``."""

    PREVIOUS = "previous"
    NEXT = "next"


class TwinStartKind(str, Enum):
    """Serialized Rust variant tag for one exact Twin Start outcome."""

    NO_DEMON = "no_demon"
    SWAP = "swap"


@dataclass(frozen=True)
class TwinStartOutcome:
    """One exact native Twin Start outcome returned by the Rust solver."""

    kind: TwinStartKind
    demon_occurrence_index: Optional[int] = None
    demon_anchor_position: Optional[int] = None
    neighbor_side: Optional[TwinNeighborSide] = None
    neighbor_position: Optional[int] = None
    neighbor_pre_swap_role: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", TwinStartKind(self.kind))
        if self.neighbor_side is not None:
            object.__setattr__(
                self,
                "neighbor_side",
                TwinNeighborSide(self.neighbor_side),
            )

        swap_fields = (
            self.demon_occurrence_index,
            self.demon_anchor_position,
            self.neighbor_side,
            self.neighbor_position,
            self.neighbor_pre_swap_role,
        )
        if self.kind is TwinStartKind.NO_DEMON:
            if any(value is not None for value in swap_fields):
                raise ValueError("no_demon Twin outcome cannot carry swap fields")
        elif any(value is None for value in swap_fields):
            raise ValueError("swap Twin outcome requires every swap field")

    def to_dict(self) -> dict:
        result = {"kind": self.kind.value}
        if self.kind is TwinStartKind.SWAP:
            result.update({
                "demon_occurrence_index": self.demon_occurrence_index,
                "demon_anchor_position": self.demon_anchor_position,
                "neighbor_side": self.neighbor_side.value,
                "neighbor_position": self.neighbor_position,
                "neighbor_pre_swap_role": self.neighbor_pre_swap_role,
            })
        return result


@dataclass(frozen=True)
class TwinTrace:
    """Generated ordered Twin current-data history for one scenario."""

    actor_position: int
    outcome: TwinStartOutcome

    def to_dict(self) -> dict:
        return {
            "actor_position": self.actor_position,
            "outcome": self.outcome.to_dict(),
        }


class BluffAcquisitionSourceKind(str, Enum):
    """Tagged native source used for a moved Twin recipient's live bluff."""

    DUPLICATE_POOL = "duplicate_pool"
    UNIQUE_POOL = "unique_pool"
    BLUFF_MUST_INCLUDE = "bluff_must_include"


@dataclass(frozen=True)
class BluffAcquisitionSource:
    """Occurrence-sensitive source of one installed recipient bluff."""

    kind: BluffAcquisitionSourceKind
    occurrence_index: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", BluffAcquisitionSourceKind(self.kind))

    def to_dict(self) -> dict:
        return {
            "kind": self.kind.value,
            "occurrence_index": self.occurrence_index,
        }


@dataclass(frozen=True)
class RevealBluffAcquisitionTrace:
    """Exact earlier bluff installation in the hidden Reveal order."""

    position: int
    acquisition_ordinal: int
    current_role: str
    bluff_role: str
    source: BluffAcquisitionSource

    def to_dict(self) -> dict:
        return {
            "position": self.position,
            "acquisition_ordinal": self.acquisition_ordinal,
            "current_role": self.current_role,
            "bluff_role": self.bluff_role,
            "source": self.source.to_dict(),
        }


@dataclass(frozen=True)
class TwinRecipientBluffTrace:
    """Exact Minion bluff installed on a runtime-Good moved Twin recipient."""

    recipient_position: int
    acquisition_ordinal: int
    bluff_role: str
    source: BluffAcquisitionSource
    prior_acquisitions: list[RevealBluffAcquisitionTrace] = field(
        default_factory=list
    )

    def to_dict(self) -> dict:
        result = {
            "recipient_position": self.recipient_position,
            "acquisition_ordinal": self.acquisition_ordinal,
            "bluff_role": self.bluff_role,
            "source": self.source.to_dict(),
        }
        if self.prior_acquisitions:
            result["prior_acquisitions"] = [
                trace.to_dict() for trace in self.prior_acquisitions
            ]
        return result


class PuppeteerNeighborSide(str, Enum):
    """Exact occurrence selected from native ``[previous, next]``."""

    PREVIOUS = "previous"
    NEXT = "next"


class PuppeteerStartKind(str, Enum):
    """Serialized Rust variant tag for one exact Puppeteer Start outcome."""

    NO_CANDIDATE = "no_candidate"
    CONVERTED = "converted"


@dataclass(frozen=True)
class PuppeteerStartOutcome:
    """One exact native Puppeteer Start outcome returned by the Rust solver."""

    kind: PuppeteerStartKind
    candidate_occurrence_index: Optional[int] = None
    neighbor_side: Optional[PuppeteerNeighborSide] = None
    target_position: Optional[int] = None
    erased_villager_role: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", PuppeteerStartKind(self.kind))
        if self.neighbor_side is not None:
            object.__setattr__(
                self,
                "neighbor_side",
                PuppeteerNeighborSide(self.neighbor_side),
            )

        converted_fields = (
            self.candidate_occurrence_index,
            self.neighbor_side,
            self.target_position,
            self.erased_villager_role,
        )
        if self.kind is PuppeteerStartKind.NO_CANDIDATE:
            if any(value is not None for value in converted_fields):
                raise ValueError(
                    "no_candidate Puppeteer outcome cannot carry conversion fields"
                )
        elif any(value is None for value in converted_fields):
            raise ValueError(
                "converted Puppeteer outcome requires every conversion field"
            )

    def to_dict(self) -> dict:
        result = {"kind": self.kind.value}
        if self.kind is PuppeteerStartKind.CONVERTED:
            result.update({
                "candidate_occurrence_index": self.candidate_occurrence_index,
                "neighbor_side": self.neighbor_side.value,
                "target_position": self.target_position,
                "erased_villager_role": self.erased_villager_role,
            })
        return result


@dataclass(frozen=True)
class PuppeteerTrace:
    """Generated ordered Puppeteer current-data history for one scenario."""

    actor_position: int
    outcome: PuppeteerStartOutcome

    def to_dict(self) -> dict:
        return {
            "actor_position": self.actor_position,
            "outcome": self.outcome.to_dict(),
        }


@dataclass
class Scenario:
    """A hypothetical assignment of evil roles to positions."""
    evil_positions: dict[int, str]  # pos -> evil role name
    puppet_position: Optional[int] = None  # If Puppeteer in play
    corrupted: set[int] = field(default_factory=set)  # Corrupted positions
    pd_corrupted: Optional[int] = None  # Plague Doctor corruption target
    doppelganger_position: Optional[int] = None  # Doppelganger pos (real role != apparent)
    drunk_position: Optional[int] = None  # Drunk pos (disguised as Villager, always lies)
    alchemist_cures: dict = field(default_factory=dict)  # alch_pos -> cure count (pre-cure)
    # Keep the legacy scalar in its historical positional-constructor slot.
    chancellor_conversion: Optional[int] = None
    messed_up_by_evil: set[int] = field(default_factory=set)
    chancellor_trace: Optional[ChancellorTrace] = None
    # Appended to preserve every historical positional Scenario constructor.
    shaman_trace: Optional[ShamanTrace] = None
    # Generated solver output only; legacy scenarios intentionally omit it.
    twin_trace: Optional[TwinTrace] = None
    # Exact pre-first-writer CharacterData map for atomically replayed slices.
    pre_twin_current_roles: dict[int, str] = field(default_factory=dict)
    # Exact post-Twin Puppeteer conversion and erased-role provenance.
    puppeteer_trace: Optional[PuppeteerTrace] = None
    # Exact offline-only bluff outcome on the runtime-Good moved Twin seat.
    twin_recipient_bluff_trace: Optional[TwinRecipientBluffTrace] = None

    def chancellor_original_villager_positions(self) -> list[int]:
        """Return possible physical seats of Chancellor's erased Villager.

        Native does not retain or mark this first target.  For each grouped
        original Chancellor seat ``c``, identity flow gives ``v = f`` when
        ``c == a`` and ``v = a`` otherwise, where ``a`` is the generated
        Outcast's final home and ``f`` is the final Chancellor seat.
        """
        trace = self.chancellor_trace
        if trace is None:
            return []

        final_chancellors = [
            position
            for position, role in self.evil_positions.items()
            if role.lower() == "chancellor"
        ]
        if len(final_chancellors) != 1:
            return []

        final_chancellor_position = final_chancellors[0]
        return sorted({
            final_chancellor_position
            if original_position == trace.added_outcast_position
            else trace.added_outcast_position
            for original_position in trace.original_positions
        })


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

# Module-level cache for the current state's card lookup. Keep a strong
# reference to the exact list rather than only its ``id``: CPython can recycle
# an id as soon as a short-lived GameState is collected, which previously let
# a new state inherit the prior state's position map. The signature also
# catches in-place list/position edits made by live-session code.
_card_lookup: dict[int, CardInfo] = {}
_card_lookup_cards: Optional[list[CardInfo]] = None
_card_lookup_signature: tuple[tuple[int, int], ...] = ()


def _build_card_lookup(state: GameState) -> dict[int, CardInfo]:
    """Build position -> CardInfo lookup dict for O(1) access."""
    return {card.position: card for card in state.cards}


def _get_card_at(pos: int, state: GameState) -> Optional[CardInfo]:
    """Get revealed card at position, or None. Uses cached dict lookup."""
    global _card_lookup, _card_lookup_cards, _card_lookup_signature
    signature = tuple((id(card), card.position) for card in state.cards)
    if (
        state.cards is not _card_lookup_cards
        or signature != _card_lookup_signature
    ):
        _card_lookup = _build_card_lookup(state)
        _card_lookup_cards = state.cards
        _card_lookup_signature = signature
    return _card_lookup.get(pos)


def get_card_at(pos: int, state: GameState) -> Optional[CardInfo]:
    """Public query helper for revealed card lookup."""
    return _get_card_at(pos, state)


def _known_evil_role(pos: int, scenario: Scenario, state: GameState) -> Optional[str]:
    """Return the stable Evil role, falling back to generated Puppet identity.

    ``puppet_position`` is a later current-data overlay.  On the supported
    Twin/Puppet overlap, the stable origin remains Twin Minion and current-role
    consumers must apply the Puppet writer separately.
    """
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


def _pre_twin_current_role_at(
    pos: int,
    scenario: Scenario,
    state: GameState,
) -> Optional[str]:
    """Current CharacterData immediately before the first exact writer."""
    if pos in scenario.pre_twin_current_roles:
        return scenario.pre_twin_current_roles[pos]

    stable = scenario.evil_positions.get(pos)
    if stable is not None:
        generated_puppet = (
            pos == scenario.puppet_position
            and stable.lower().replace(" ", "").replace("_", "") == "puppet"
        )
        if not generated_puppet:
            return stable
    if (
        scenario.chancellor_trace is not None
        and pos == scenario.chancellor_trace.added_outcast_position
    ):
        return scenario.chancellor_trace.added_outcast_role
    if pos == scenario.doppelganger_position:
        return "Doppelganger"
    if pos == scenario.drunk_position:
        return "Drunk"
    card = _get_card_at(pos, state)
    return card.apparent_role if card else None


def _role_after_twin_at(
    pos: int,
    before: Optional[str],
    trace: TwinTrace,
) -> Optional[str]:
    """Replay one exact Twin current-data swap at a physical position."""
    if trace.outcome.kind is TwinStartKind.NO_DEMON:
        return before
    actor = trace.actor_position
    neighbor = trace.outcome.neighbor_position
    if actor == neighbor:
        return before
    if pos == actor:
        return trace.outcome.neighbor_pre_swap_role
    if pos == neighbor:
        return "Twin Minion"
    return before


def puppet_erased_role_at(pos: int, scenario: Scenario) -> Optional[str]:
    """Saved Villager identity displayed by an exactly replayed Puppet."""
    trace = scenario.puppeteer_trace
    if trace is None or trace.outcome.kind is not PuppeteerStartKind.CONVERTED:
        return None
    if trace.outcome.target_position != pos:
        return None
    return trace.outcome.erased_villager_role


def effective_role_at(pos: int, scenario: Scenario, state: GameState) -> Optional[str]:
    """True represented role, including generated and hidden Outcasts."""
    if (
        scenario.twin_trace is not None
        or scenario.pre_twin_current_roles
        or scenario.puppeteer_trace is not None
    ):
        current = _pre_twin_current_role_at(pos, scenario, state)
        if scenario.twin_trace is not None:
            current = _role_after_twin_at(pos, current, scenario.twin_trace)

        if scenario.puppeteer_trace is not None:
            if (
                scenario.puppeteer_trace.outcome.kind
                is PuppeteerStartKind.CONVERTED
                and scenario.puppeteer_trace.outcome.target_position == pos
            ):
                current = "Puppet"
        elif pos == scenario.puppet_position:
            current = "Puppet"

        if (
            scenario.shaman_trace is not None
            and pos in {
                scenario.shaman_trace.source_position,
                scenario.shaman_trace.target_position,
            }
        ):
            current = scenario.shaman_trace.copied_role
        return current

    # Shaman overwrites the destination's current dataRef while preserving its
    # physical runtime alignment. The source already owns the copied role.
    # Current-role consumers must therefore prefer this trace over the
    # endpoints' original Evil/Drunk/Doppelganger identities.
    if (
        scenario.shaman_trace is not None
        and pos in {
            scenario.shaman_trace.source_position,
            scenario.shaman_trace.target_position,
        }
    ):
        return scenario.shaman_trace.copied_role
    # Puppeteer acts after Twin and fully reinitializes the selected current
    # CharacterData as Puppet.  Keep this overlay ahead of the stable Evil map:
    # a Twin body can therefore have a stable Twin origin and current Puppet
    # data at the same physical position.
    if pos == scenario.puppet_position:
        return "Puppet"
    evil_role = _known_evil_role(pos, scenario, state)
    if evil_role is not None:
        return evil_role
    if (
        scenario.chancellor_trace is not None
        and pos == scenario.chancellor_trace.added_outcast_position
    ):
        return scenario.chancellor_trace.added_outcast_role
    if pos == scenario.doppelganger_position:
        return "Doppelganger"
    if pos == scenario.drunk_position:
        return "Drunk"
    card = _get_card_at(pos, state)
    return card.apparent_role if card else None


def _effective_alignment(pos: int, scenario: Scenario, state: GameState) -> Alignment:
    """Effective alignment for ability purposes. Wretch registers as Evil."""
    if _is_evil_in_board_state(pos, scenario, state):
        return Alignment.EVIL
    role = effective_role_at(pos, scenario, state)
    if role and role.lower().replace(" ", "").replace("_", "") == "wretch":
        return Alignment.EVIL  # Wretch registers as Evil to abilities
    return Alignment.GOOD


def effective_alignment(pos: int, scenario: Scenario, state: GameState) -> Alignment:
    """Public query helper for effective alignment in a scenario."""
    return _effective_alignment(pos, scenario, state)


def _truth_status(pos: int, scenario: Scenario, state: GameState) -> TruthStatus:
    """Apply native CheckLying precedence to the scenario abstraction.

    The model has no general bluff-data pointer or status collection. Clean
    Puppet/Doppelganger scenarios represent HealthyBluff, while Drunk represents
    a non-null bluff without it. Other arbitrary good bluff holders cannot be
    represented without extending Scenario. Lying/Appear are intentionally not
    inferred from apparent roles; native CheckLying does not consult them.
    """
    # Corruption wins over HealthyBluff and over cant_lie roles such as
    # Confessor. Drunk is normally in this set, but its explicit bluff mapping
    # below also keeps hand-built scenarios faithful when it is omitted.
    if pos in scenario.corrupted:
        return TruthStatus.LYING

    evil_role = _known_evil_role(pos, scenario, state)
    effective_role = effective_role_at(pos, scenario, state)
    effective_role_key = (
        effective_role.lower().replace(" ", "").replace("_", "")
        if effective_role else None
    )

    # Both roles apply HealthyBluff in the represented clean runtime cases.
    modeled_healthy_bluff = (
        pos == scenario.puppet_position
        or evil_role == "Puppet"
        or pos == scenario.doppelganger_position
        or effective_role_key == "doppelganger"
    )
    if modeled_healthy_bluff:
        return TruthStatus.TRUTHFUL

    # Runtime Evil lies regardless of bluff data. Drunk and Doppelganger are
    # the model's explicit non-null-bluff positions; clean Doppelganger already
    # returned through HealthyBluff above.
    modeled_non_null_bluff = (
        pos == scenario.drunk_position
        or pos == scenario.doppelganger_position
        or effective_role_key in {"drunk", "doppelganger"}
    )
    if evil_role is not None or modeled_non_null_bluff:
        return TruthStatus.LYING

    return TruthStatus.TRUTHFUL


def truth_status(pos: int, scenario: Scenario, state: GameState) -> TruthStatus:
    """Public query helper for whether a position tells the truth in a scenario."""
    return _truth_status(pos, scenario, state)


def _shaman_copied_confessor_status_at(pos: int, scenario: Scenario) -> bool:
    """Return the exact shipped copied-Confessor appearance-status fact."""
    trace = scenario.shaman_trace
    return bool(
        trace is not None
        and trace.copied_role.lower().replace(" ", "").replace("_", "")
        == "confessor"
        and pos in {trace.source_position, trace.target_position}
    )


def _truth_appearance_status(
    pos: int,
    scenario: Scenario,
    state: GameState,
) -> TruthStatus:
    """Model native ``CharacterHelper.CheckLyingAppearance``.

    Confessor applies ``AppearTruthfull`` from both its real and bluff-role
    initialization paths. Shaman's exact copied-Confessor endpoints retain the
    same physical status after later no-reset presentation changes. The
    scenario model does not carry arbitrary appearance statuses, so every
    other apparent role falls back to the actual native lie predicate
    represented by :func:`_truth_status`.
    """
    card = _get_card_at(pos, state)
    if (
        _shaman_copied_confessor_status_at(pos, scenario)
        or card is not None
        and card.apparent_role.lower().replace(" ", "").replace("_", "") == "confessor"
    ):
        return TruthStatus.TRUTHFUL

    return _truth_status(pos, scenario, state)


def truth_appearance_status(
    pos: int,
    scenario: Scenario,
    state: GameState,
) -> TruthStatus:
    """Public query for the truthfulness a Judge perceives at a position."""
    return _truth_appearance_status(pos, scenario, state)


if __name__ == "__main__":
    # Quick smoke test
    print("Solver module loaded successfully (types + helpers only)")
    print(f"Circle distance 1->4 on 7: {circle_distance(1, 4, 7)}")
    print(f"Circle direction 1->3 on 7: {circle_direction(1, 3, 7)}")
    print(f"Adjacent to 1 on 7: {adjacent_positions(1, 7)}")
