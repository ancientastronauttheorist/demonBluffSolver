"""Game loop adapter: bridges Claude's vision reads to the constraint solver.

Card builder functions, session tracking, CLI interface.
"""

from __future__ import annotations
import atexit
from collections import Counter
import copy
import json
import os
import re
import sys
import time
from datetime import datetime
from typing import Optional

from knowledge_base import get_card
from solver import (
    BAKER_RULE_VERSION,
    CardInfo,
    DeckComposition,
    DOPPEL_DRUNK_RULE_VERSION,
    FORTUNE_TELLER_RULE_VERSION,
    GameState,
    POET_PROVIDER_ROLES,
    POET_VARIANT,
    RAMBLER_RULE_VERSION,
    SolverResult,
    slayer_revealed_role,
)
from rust_solver import rust_solve_to_objects
from strategy import (
    evil_probabilities,
    ordinary_execution_bombardier_positions,
    print_recommendation,
    recommend_action,
)


# ============================================================
# Card Builder Functions
# ============================================================

def _enlightened_native_text(direction: str) -> str:
    """Return Shugenja's exact shipped public clue text."""
    try:
        return {
            "CW": "Closest Evil is:\nClockwise",
            "CCW": "Closest Evil is:\nCounter-clockwise",
            "Equidistant": "Closest Evil is equidistant",
        }[direction]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            "Enlightened direction must be CW, CCW, or Equidistant"
        ) from exc


def _parse_enlightened_native_text(info_text: str) -> Optional[str]:
    """Parse only one exact current Shugenja sentence."""
    if not isinstance(info_text, str):
        return None
    for direction in ("CW", "CCW", "Equidistant"):
        if info_text == _enlightened_native_text(direction):
            return direction
    return None


def _enlightened_runtime_matches(runtime_data, direction: str) -> bool:
    """Validate native runtime provenance, allowing an unreadable object."""
    if runtime_data is None:
        return True
    return (
        isinstance(runtime_data, dict)
        and runtime_data.get("type") == "direction"
        and runtime_data.get("direction") == direction
    )


def _canonical_enlightened_direction(direction: str) -> str:
    """Canonicalize the supported manual direction spellings."""
    if not isinstance(direction, str):
        raise ValueError(
            "Enlightened direction must be CW, CCW, or Equidistant"
        )
    key = direction.strip().casefold()
    canonical = {
        "cw": "CW",
        "clockwise": "CW",
        "ccw": "CCW",
        "counter-clockwise": "CCW",
        "counterclockwise": "CCW",
        "equidistant": "Equidistant",
    }.get(key)
    if canonical is None:
        raise ValueError(
            "Enlightened direction must be CW, CCW, or Equidistant"
        )
    return canonical


def card_enlightened(
    pos: int,
    direction: str,
    *,
    info_text: str = "",
    enlightened_variant: Optional[str] = None,
) -> CardInfo:
    """Build an Enlightened observation while preserving legacy callers."""
    info = {"direction": direction}
    if enlightened_variant is not None:
        info["enlightened_variant"] = enlightened_variant
        expected_text = _enlightened_native_text(direction)
        if info_text and info_text != expected_text:
            raise ValueError("Current Enlightened text must match its direction")
        info_text = expected_text
    return CardInfo(pos, "Enlightened", info_text=info_text, info_parsed=info)

def _knitter_native_text(evil_pairs: int) -> str:
    """Return Knitter's exact shipped public clue text."""
    if type(evil_pairs) is not int or evil_pairs < 0:
        raise ValueError("Knitter pair count must be a non-negative integer")
    if evil_pairs == 0:
        return "Evils are not adjacent to eachother"
    if evil_pairs == 1:
        return "There is only 1 pair of Evil"
    return f"There are {evil_pairs} pairs of Evil"


def _parse_knitter_native_text(info_text: str) -> Optional[int]:
    """Parse one exact current Knitter sentence into its pair count."""
    if not isinstance(info_text, str):
        return None
    if info_text == _knitter_native_text(0):
        return 0
    if info_text == _knitter_native_text(1):
        return 1
    match = re.fullmatch(r"There are ([1-9]\d*) pairs of Evil", info_text)
    if match is None:
        return None
    evil_pairs = int(match.group(1))
    if evil_pairs < 2 or info_text != _knitter_native_text(evil_pairs):
        return None
    return evil_pairs


def card_knitter(
    pos: int,
    evil_pairs: int,
    *,
    info_text: str = "",
    knitter_variant: Optional[str] = None,
) -> CardInfo:
    """Build a Knitter observation while preserving unmarked legacy callers."""
    info = {"evil_pairs": evil_pairs}
    if knitter_variant is not None:
        info["knitter_variant"] = knitter_variant
        if not info_text:
            info_text = _knitter_native_text(evil_pairs)
    return CardInfo(pos, "Knitter", info_text=info_text, info_parsed=info)

def _confessor_native_text(dizzy: bool) -> str:
    """Return Confessor.ConjourInfo's exact shipped public sentence."""
    if type(dizzy) is not bool:
        raise ValueError("Confessor dizzy claim must be a boolean")
    return "I am dizzy" if dizzy else "I am Good"


def _parse_confessor_native_text(info_text) -> Optional[bool]:
    """Parse only one exact current Confessor sentence."""
    if not isinstance(info_text, str):
        return None
    if info_text == _confessor_native_text(False):
        return False
    if info_text == _confessor_native_text(True):
        return True
    return None


def _canonical_confessor_claim(value: str) -> bool:
    """Canonicalize one explicit manual Confessor result token."""
    if not isinstance(value, str):
        raise ValueError("Confessor result must be Good or dizzy")
    key = value.strip().casefold()
    if key in {"dizzy", "dirty", "true", "1", "yes"}:
        return True
    if key in {"good", "clean", "false", "0", "no"}:
        return False
    raise ValueError("Confessor result must be Good or dizzy")


def card_confessor(
    pos: int,
    dizzy: bool,
    *,
    info_text: str = "",
    confessor_variant: Optional[str] = None,
) -> CardInfo:
    """Build a Confessor observation while preserving unmarked archives."""
    info = {"dizzy": dizzy}
    if confessor_variant is not None:
        if confessor_variant != _PUBLIC_CURRENT_VARIANT:
            raise ValueError("Unsupported Confessor variant")
        if type(pos) is not int or pos <= 0:
            raise ValueError("Current Confessor position must be positive")
        if type(dizzy) is not bool:
            raise ValueError("Current Confessor dizzy claim must be a boolean")
        if not isinstance(info_text, str):
            raise ValueError("Current Confessor text must be a string")
        expected_text = _confessor_native_text(dizzy)
        if info_text and info_text != expected_text:
            raise ValueError("Current Confessor text must match its claim")
        info["confessor_variant"] = confessor_variant
        info_text = expected_text
    return CardInfo(
        pos,
        "Confessor",
        info_text=info_text,
        info_parsed=info,
    )

def _gemcrafter_native_text(good_position: int) -> str:
    """Return Archivist.ConjourInfo's exact shipped public sentence."""
    if type(good_position) is not int or good_position <= 0:
        raise ValueError("Gemcrafter target must be a positive integer")
    return f"#{good_position} is Good"


def _parse_gemcrafter_native_text(info_text) -> Optional[int]:
    """Parse only one exact current Archivist sentence."""
    if not isinstance(info_text, str):
        return None
    match = re.fullmatch(r"#([1-9]\d*) is Good", info_text)
    if match is None:
        return None
    try:
        good_position = int(match.group(1))
    except ValueError:
        # Python caps pathological decimal conversions; malformed memory text
        # must fail closed rather than escape the live parser.
        return None
    return (
        good_position
        if info_text == _gemcrafter_native_text(good_position)
        else None
    )


def card_gemcrafter(
    pos: int,
    good_position: int,
    *,
    info_text: str = "",
    gemcrafter_variant: Optional[str] = None,
) -> CardInfo:
    """Build a Gemcrafter observation while preserving unmarked legacy data."""
    info = {"good_position": good_position}
    if gemcrafter_variant is not None:
        if gemcrafter_variant != _PUBLIC_CURRENT_VARIANT:
            raise ValueError("Unsupported Gemcrafter variant")
        if type(pos) is not int or pos <= 0:
            raise ValueError("Current Gemcrafter position must be positive")
        if not isinstance(info_text, str):
            raise ValueError("Current Gemcrafter text must be a string")
        expected_text = _gemcrafter_native_text(good_position)
        if info_text and info_text != expected_text:
            raise ValueError("Current Gemcrafter text must match its target")
        info["gemcrafter_variant"] = gemcrafter_variant
        info_text = expected_text
    return CardInfo(
        pos,
        "Gemcrafter",
        info_text=info_text,
        info_parsed=info,
    )

def card_lover(
    pos: int,
    evil_adjacent: int,
    *,
    info_text: str = "",
    lover_variant: Optional[str] = None,
) -> CardInfo:
    """Build a Lover observation.

    Provenance remains opt-in so archived unmarked fixtures preserve their
    historical predicate. Live auto/manual entry supplies ``public_current``
    only after validating the native clue and adjacent-character references.
    """
    info = {"evil_adjacent": evil_adjacent}
    if lover_variant is not None:
        info["lover_variant"] = lover_variant
    return CardInfo(pos, "Lover", info_text=info_text, info_parsed=info)

_PUBLIC_CURRENT_VARIANT = "public_current"


def _lover_native_text(evil_adjacent: int) -> str:
    """Return Empath.ConjourInfo's exact shipped public sentence."""
    try:
        return {
            0: "NO Evils\nadjacent to me",
            1: "1 Evil\nadjacent to me",
            2: "2 Evils\nadjacent to me",
        }[evil_adjacent]
    except (KeyError, TypeError) as exc:
        raise ValueError("Lover evil count must be 0, 1, or 2") from exc


def _parse_lover_native_text(clue: str) -> Optional[int]:
    """Parse only one exact Empath.ConjourInfo output."""
    for evil_adjacent in range(3):
        if clue == _lover_native_text(evil_adjacent):
            return evil_adjacent
    return None


def _current_lover_refs(position: int, n_cards: int) -> list[int]:
    """Native Characters.GetAdjacentCharacters order: previous, then next."""
    if type(position) is not int or type(n_cards) is not int:
        raise ValueError("Lover adjacency requires integer board coordinates")
    if n_cards <= 0 or not 1 <= position <= n_cards:
        raise ValueError("Lover position is outside the current board")
    return [
        ((position - 2) % n_cards) + 1,
        (position % n_cards) + 1,
    ]


def card_scout(
    pos: int,
    evil_role: str,
    distance: int,
    *,
    info_text: str = "",
    scout_variant: Optional[str] = None,
) -> CardInfo:
    """Build a Scout observation.

    The optional provenance marker is deliberately opt-in so archived direct
    Scout fixtures keep their historical predicate. Live auto/manual entry
    supplies ``public_current`` after validating the native sentence.
    """
    info = {"evil_role": evil_role, "distance": distance}
    if scout_variant is not None:
        info["scout_variant"] = scout_variant
    return CardInfo(pos, "Scout", info_text=info_text, info_parsed=info)


def _card_scout_one_evil(
    pos: int,
    *,
    info_text: str = "",
    scout_variant: Optional[str] = None,
) -> CardInfo:
    """Build Scout's exact native one-Evil sentinel observation."""
    info = {"one_evil": True}
    if scout_variant is not None:
        info["scout_variant"] = scout_variant
    return CardInfo(pos, "Scout", info_text=info_text, info_parsed=info)

def _bard_native_text(corruption_distance: int) -> str:
    """Return Acrobat2.ConjourInfo's exact shipped public sentence.

    Current payloads normalize native distance zero to ``-1`` so it remains
    distinct from historical numeric-zero observations.
    """
    if type(corruption_distance) is not int:
        raise ValueError("Bard corruption distance must be an integer")
    if corruption_distance == -1:
        return "There are no Corrupted characters"
    if corruption_distance == 1:
        return "I am 1 card away from Corrupted character"
    if corruption_distance >= 2:
        return (
            f"I am {corruption_distance} cards away from Corrupted character"
        )
    raise ValueError("Bard corruption distance must be -1 or positive")


def _parse_bard_native_text(info_text) -> Optional[int]:
    """Parse only one exact current Acrobat2 sentence."""
    if not isinstance(info_text, str):
        return None
    if info_text == _bard_native_text(-1):
        return -1
    if info_text == _bard_native_text(1):
        return 1
    match = re.fullmatch(
        r"I am ([1-9]\d*) cards away from Corrupted character",
        info_text,
    )
    if match is None:
        return None
    try:
        corruption_distance = int(match.group(1))
    except ValueError:
        # Python caps pathological decimal conversions. Treat malformed memory
        # text as untrusted input and fail closed.
        return None
    return (
        corruption_distance
        if info_text == _bard_native_text(corruption_distance)
        else None
    )


def _valid_current_bard_distance(
    corruption_distance: int,
    n_cards: Optional[int],
) -> bool:
    """Whether a normalized Bard claim can be emitted in the current build."""
    if type(corruption_distance) is not int:
        return False
    if corruption_distance == -1:
        return True
    if type(n_cards) is not int or n_cards <= 0:
        return False
    # Truth can report the half-circle distance. Bluff draws from the fixed
    # public domain 0..3, so 1..3 remains possible even on a tiny board.
    return 1 <= corruption_distance <= max(3, n_cards // 2)


def _current_bard_refs(
    pos: int,
    corruption_distance: int,
    n_cards: int,
) -> list[int]:
    """Return Acrobat2.GetCharactersAtRange order, preserving duplicates."""
    if corruption_distance in {-1, 0} or corruption_distance > n_cards - 1:
        return []
    forward = ((pos - 1 + corruption_distance) % n_cards) + 1
    backward = ((pos - 1 - corruption_distance) % n_cards) + 1
    return [forward, backward]


def card_bard(
    pos: int,
    corruption_distance: int,
    *,
    info_text: str = "",
    bard_variant: Optional[str] = None,
) -> CardInfo:
    """Build a Bard observation while preserving unmarked archive callers."""
    if bard_variant is not None and type(corruption_distance) is not int:
        raise ValueError("Current Bard distance must be an integer")
    # Existing manual/archive convention: zero means no Corrupted characters.
    if corruption_distance == 0:
        corruption_distance = -1
    info = {"corruption_distance": corruption_distance}
    if bard_variant is not None:
        if bard_variant != _PUBLIC_CURRENT_VARIANT:
            raise ValueError("Unsupported Bard variant")
        if type(pos) is not int or pos <= 0:
            raise ValueError("Current Bard position must be positive")
        # The builder owns scalar/current schema safety; board-specific upper
        # bounds are enforced by manual and memory ingestion.
        if corruption_distance != -1 and corruption_distance < 1:
            raise ValueError("Current Bard distance must be -1 or positive")
        if not isinstance(info_text, str):
            raise ValueError("Current Bard text must be a string")
        expected_text = _bard_native_text(corruption_distance)
        if info_text and info_text != expected_text:
            raise ValueError("Current Bard text must match its distance")
        info["bard_variant"] = bard_variant
        info_text = expected_text
    return CardInfo(
        pos,
        "Bard",
        info_text=info_text,
        info_parsed=info,
    )

def card_fortune_teller(
    pos: int,
    targets: list[int],
    has_evil: bool,
    *,
    info_text: str = "",
    observations: Optional[list[dict]] = None,
) -> CardInfo:
    info = {"targets": list(targets), "has_evil": has_evil}
    if observations is not None:
        info["observations"] = [dict(observation) for observation in observations]
    return CardInfo(
        pos,
        "Fortune Teller",
        info_text=info_text,
        info_parsed=info,
    )

def _oracle_native_text(targets: list[int], minion_role: str) -> str:
    """Return Investigator's exact positive public sentence."""
    return f"#{targets[0]} or #{targets[1]} is a {minion_role}"


def card_oracle(
    pos: int,
    targets: list[int],
    minion_role: str,
    *,
    info_text: str = "",
    oracle_variant: Optional[str] = None,
) -> CardInfo:
    """Build an Oracle observation while preserving unmarked legacy callers."""
    info = {"targets": list(targets), "minion_role": minion_role}
    if oracle_variant is not None:
        info["oracle_variant"] = oracle_variant
        if not info_text:
            info_text = _oracle_native_text(targets, minion_role)
    return CardInfo(pos, "Oracle", info_text=info_text, info_parsed=info)


def _card_oracle_no_minions(
    pos: int,
    *,
    info_text: str = "There are no minions",
    oracle_variant: Optional[str] = None,
) -> CardInfo:
    """Build Investigator's exact truthful empty-Minions sentinel."""
    info = {"no_minions": True}
    if oracle_variant is not None:
        info["oracle_variant"] = oracle_variant
    return CardInfo(pos, "Oracle", info_text=info_text, info_parsed=info)

def _medium_native_text(good_position: int, good_role: str) -> str:
    """Return Lookout's exact shipped public clue text."""
    qualifier = "actually a" if good_role == "Drunk" else "a real"
    return f"#{good_position} is {qualifier}\n{good_role}"


def _parse_medium_native_text(info_text: str) -> Optional[tuple[int, str]]:
    """Parse one exact current Lookout result into canonical public values."""
    if not isinstance(info_text, str):
        return None
    match = re.fullmatch(
        r"#([1-9]\d*) is (?:a real|actually a)\n([^\r\n]+)",
        info_text,
    )
    if match is None:
        return None
    good_position = int(match.group(1))
    good_role = get_card(match.group(2))
    if good_role is None:
        return None
    if info_text != _medium_native_text(good_position, good_role.name):
        return None
    return good_position, good_role.name


def card_medium(
    pos: int,
    good_position: int,
    good_role: str,
    *,
    info_text: str = "",
    medium_variant: Optional[str] = None,
) -> CardInfo:
    """Build a Medium observation while preserving unmarked legacy callers."""
    info = {"good_position": good_position, "good_role": good_role}
    if medium_variant is not None:
        info["medium_variant"] = medium_variant
        if not info_text:
            info_text = _medium_native_text(good_position, good_role)
    return CardInfo(pos, "Medium", info_text=info_text, info_parsed=info)

def card_hunter(
    pos: int,
    distance: int,
    *,
    info_text: str = "",
    hunter_variant: Optional[str] = None,
) -> CardInfo:
    """Build a Hunter observation, optionally marking current provenance."""
    info = {"distance": distance}
    if hunter_variant is not None:
        info["hunter_variant"] = hunter_variant
    return CardInfo(pos, "Hunter", info_text=info_text, info_parsed=info)


def _valid_current_scout_distance(
    distance: int,
    n_cards: Optional[int],
) -> bool:
    """Whether a numeric Scout result is reachable in the current build."""
    if type(distance) is not int or distance < 1:
        return False
    if n_cards is None:
        return True
    # Truth reaches half the circle. Bluff draws a false zero-based gap from
    # 0..2 and displays it as 1..3, so the public range is their union.
    return distance <= max(3, n_cards // 2)


def _valid_current_hunter_distance(
    distance: int,
    n_cards: Optional[int],
) -> bool:
    """Whether a Hunter result is reachable in the current build."""
    if type(distance) is not int:
        return False
    if n_cards is None:
        return distance >= 1
    if n_cards == 1:
        return distance == 0
    return 1 <= distance <= n_cards // 2 or distance == n_cards - 1


def _current_hunter_refs(pos: int, distance: int, n_cards: int) -> list[int]:
    """Return native GetCharactersAtRange order, preserving duplicates."""
    if distance == 0:
        return []
    forward = ((pos - 1 + distance) % n_cards) + 1
    backward = ((pos - 1 - distance) % n_cards) + 1
    return [forward, backward]

def card_architect(pos: int, side: str) -> CardInfo:
    """side: 'Left', 'Right', or 'Equal'"""
    return CardInfo(pos, "Architect", info_parsed={"side": side})

def _validate_current_empress_targets(targets) -> list[int]:
    """Validate Empress's sorted three-reference current-build payload."""
    if (
        not isinstance(targets, list)
        or len(targets) != 3
        or any(type(target) is not int or target <= 0 for target in targets)
        or len(set(targets)) != len(targets)
        or targets != sorted(targets)
    ):
        raise ValueError(
            "Current Empress targets must be three ascending unique positions"
        )
    return list(targets)


def _empress_native_text(targets: list[int]) -> str:
    """Return Empress's exact shipped public sentence."""
    targets = _validate_current_empress_targets(targets)
    return (
        f"One is Evil:\n#{targets[0]}, #{targets[1]} or #{targets[2]}"
    )


def _parse_empress_native_text(info_text) -> Optional[list[int]]:
    """Parse only one exact current Empress sentence."""
    if not isinstance(info_text, str):
        return None
    match = re.fullmatch(
        r"One is Evil:\n#([1-9]\d*), #([1-9]\d*) or #([1-9]\d*)",
        info_text,
    )
    if match is None:
        return None
    targets = [int(match.group(index)) for index in range(1, 4)]
    try:
        expected = _empress_native_text(targets)
    except ValueError:
        return None
    return targets if info_text == expected else None


def card_empress(
    pos: int,
    targets: list[int],
    *,
    info_text: str = "",
    empress_variant: Optional[str] = None,
) -> CardInfo:
    """Build an Empress observation while preserving unmarked legacy data."""
    info = {"targets": targets}
    if empress_variant is not None:
        if empress_variant != _PUBLIC_CURRENT_VARIANT:
            raise ValueError("Unsupported Empress variant")
        if type(pos) is not int or pos <= 0:
            raise ValueError("Current Empress position must be positive")
        if not isinstance(info_text, str):
            raise ValueError("Current Empress text must be a string")
        current_targets = _validate_current_empress_targets(targets)
        expected_text = _empress_native_text(current_targets)
        if info_text and info_text != expected_text:
            raise ValueError("Current Empress text must match its targets")
        info["targets"] = current_targets
        info["empress_variant"] = empress_variant
        info_text = expected_text
    return CardInfo(pos, "Empress", info_text=info_text, info_parsed=info)

def card_witness(pos: int, affected_position: int) -> CardInfo:
    return CardInfo(pos, "Witness", info_parsed={"affected_position": affected_position})

def _validate_current_jester_targets(
    targets,
    *,
    n_cards: Optional[int] = None,
) -> list[int]:
    """Validate the three physical selections retained in click order."""
    if (
        not isinstance(targets, list)
        or len(targets) != 3
        or any(type(target) is not int or target <= 0 for target in targets)
        or len(set(targets)) != 3
        or (
            n_cards is not None
            and (
                type(n_cards) is not int
                or n_cards <= 0
                or any(target > n_cards for target in targets)
            )
        )
    ):
        raise ValueError(
            "Current Jester targets must be three distinct current-board positions"
        )
    return list(targets)


def _validate_jester_reference_ids(
    references,
    *,
    n_cards: Optional[int] = None,
) -> list[int]:
    """Validate native Character IDs without inventing object uniqueness.

    Juggler's picker guarantees three distinct physical Character objects, but
    ActedInfo retains only their integer display IDs. Different objects may
    carry the same ID, so duplicate references are valid even though solver
    ``targets`` (physical board positions) must remain distinct.
    """
    if (
        not isinstance(references, list)
        or len(references) != 3
        or any(type(reference) is not int or reference <= 0 for reference in references)
        or (
            n_cards is not None
            and (
                type(n_cards) is not int
                or n_cards <= 0
                or any(reference > n_cards for reference in references)
            )
        )
    ):
        raise ValueError(
            "Current Jester references must be three current-board display IDs"
        )
    return list(references)


def _jester_native_text(reference_ids: list[int], evil_count: int) -> str:
    """Return Juggler's exact shipped result text."""
    references = _validate_jester_reference_ids(reference_ids)
    if type(evil_count) is not int or not 0 <= evil_count <= 3:
        raise ValueError("Current Jester evil_count must be an integer from 0 to 3")
    displayed = sorted(references)
    result = (
        "There is 1 Evil"
        if evil_count == 1
        else f"There are {evil_count} Evils"
    )
    return (
        f"Among:\n#{displayed[0]}, #{displayed[1]}, #{displayed[2]}:\n"
        f"{result}"
    )


def _parse_jester_native_text(
    info_text,
) -> Optional[tuple[list[int], int]]:
    """Parse only one byte-exact current Jester result sentence."""
    if not isinstance(info_text, str):
        return None
    match = re.fullmatch(
        r"Among:\n#([1-9]\d*), #([1-9]\d*), #([1-9]\d*):\n"
        r"There (?:is (1) Evil|are ([023]) Evils)",
        info_text,
    )
    if match is None:
        return None
    displayed = [int(match.group(index)) for index in range(1, 4)]
    evil_count = int(match.group(4) or match.group(5))
    try:
        displayed = _validate_jester_reference_ids(displayed)
        expected = _jester_native_text(displayed, evil_count)
    except ValueError:
        return None
    if displayed != sorted(displayed) or info_text != expected:
        return None
    return displayed, evil_count


def card_jester(
    pos: int,
    targets: list[int],
    evil_count: int,
    *,
    info_text: str = "",
    jester_variant: Optional[str] = None,
) -> CardInfo:
    """Build Jester evidence while retaining unmarked archive compatibility."""
    info = {"targets": targets, "evil_count": evil_count}
    if jester_variant is not None:
        if jester_variant != _PUBLIC_CURRENT_VARIANT:
            raise ValueError("Unsupported Jester variant")
        if type(pos) is not int or pos <= 0:
            raise ValueError("Current Jester position must be positive")
        current_targets = _validate_current_jester_targets(targets)
        if type(evil_count) is not int or not 0 <= evil_count <= 3:
            raise ValueError(
                "Current Jester evil_count must be an integer from 0 to 3"
            )
        expected_text = _jester_native_text(current_targets, evil_count)
        if not isinstance(info_text, str) or (
            info_text and info_text != expected_text
        ):
            raise ValueError("Current Jester text must match its targets and count")
        info = {
            "targets": current_targets,
            "evil_count": evil_count,
            "jester_variant": jester_variant,
        }
        info_text = expected_text
    return CardInfo(pos, "Jester", info_text=info_text, info_parsed=info)


def _card_current_jester_no_info(pos: int) -> CardInfo:
    """Mark a live Juggler shell without pretending its ability has fired."""
    if type(pos) is not int or pos <= 0:
        raise ValueError("Current Jester position must be positive")
    return CardInfo(
        pos,
        "Jester",
        info_parsed={"jester_variant": _PUBLIC_CURRENT_VARIANT},
    )

def card_jester_silenced(
    pos: int,
    targets: Optional[list[int]] = None,
    shut_up_target: Optional[int] = None,
    info_text: str = "",
) -> CardInfo:
    """Jester whose emitted result was replaced by Rambler2 interference.

    Current native interference rewrites the emitted ``ActedInfo`` reference
    list to the Rambler target, so the live parser must not treat that list as
    Jester's original picks. ``targets`` remains optional only for archived,
    explicitly reconstructed observations.
    """
    info = {"silenced": True}
    if targets is not None:
        info["targets"] = list(targets)
    if shut_up_target is not None:
        info["shut_up_target"] = shut_up_target
    return CardInfo(pos, "Jester", info_text=info_text, info_parsed=info)

def card_rambler(pos: int, silenced: bool, silenced_by: Optional[int] = None) -> CardInfo:
    """Build the archived, pre-Rambler2 observation shape."""
    info = {"silenced": silenced}
    if silenced_by is not None:
        info["silenced_by"] = silenced_by
    return CardInfo(pos, "Rambler", info_parsed=info)


def card_rambler_quote(pos: int, info_text: str) -> CardInfo:
    """Current Rambler2 Day output when it was not interrupted."""
    return CardInfo(
        pos,
        "Rambler",
        info_text=info_text,
        info_parsed={"quote_observed": True},
    )

def card_shut_up(
    pos: int,
    role: str,
    target: int,
    info_text: str = "",
) -> CardInfo:
    """A Rambler-redesign clue: this card said "#target shut up!"."""
    if type(target) is not int or target <= 0:
        raise ValueError("Rambler shut-up target must be a positive integer")
    return CardInfo(
        pos,
        _normalize_role_name(role),
        info_text=info_text,
        info_parsed={"shut_up_target": target},
    )

def card_dreamer(
    pos: int,
    target: int,
    evil_role: str,
    info_text: str = "",
) -> CardInfo:
    return CardInfo(
        pos,
        "Dreamer",
        info_text=info_text,
        info_parsed={"target": target, "evil_role": evil_role},
    )


def _validate_dreamer_targets(targets) -> list[int]:
    try:
        normalized = list(targets)
    except TypeError as exc:
        raise ValueError("Dreamer requires exactly 2 integer targets") from exc
    if len(normalized) != 2 or any(type(target) is not int for target in normalized):
        raise ValueError("Dreamer requires exactly 2 integer targets")
    return normalized


def _validate_dreamer_role_options(role_options) -> list[str]:
    try:
        normalized = [
            option.strip() if isinstance(option, str) else option
            for option in role_options
        ]
    except TypeError as exc:
        raise ValueError("Dreamer requires exactly 2 nonempty distinct role options") from exc
    if (
        len(normalized) != 2
        or any(not isinstance(option, str) or not option for option in normalized)
        or _dreamer_role_key(normalized[0]) == _dreamer_role_key(normalized[1])
    ):
        raise ValueError("Dreamer requires exactly 2 nonempty distinct role options")
    return normalized


def card_dreamer_ambiguous(
    pos: int,
    targets: list[int],
    evil_role_options: list[str],
    info_text: str = "",
) -> CardInfo:
    """Public Dreamer output: "Among #X, #Y there is: R1 or R2".

    The Rust solver handles this shape as `{targets, evil_role_options}` and
    tests the observation against the actor's truthful or lying native output
    support. Role order does not map to target order.
    """
    normalized_targets = _validate_dreamer_targets(targets)
    normalized_options = _validate_dreamer_role_options(evil_role_options)
    return CardInfo(
        pos,
        "Dreamer",
        info_text=info_text,
        info_parsed={
            "targets": normalized_targets,
            "evil_role_options": normalized_options,
            "dreamer_variant": "public_current",
        },
    )


def card_dreamer_cabbage(
    pos: int,
    targets: list[int],
    info_text: str = "",
) -> CardInfo:
    """Public Dreamer's Wretch clue: one selected target is a Cabbage."""
    normalized_targets = _validate_dreamer_targets(targets)
    return CardInfo(
        pos,
        "Dreamer",
        info_text=info_text,
        info_parsed={
            "targets": normalized_targets,
            "cabbage": True,
            "dreamer_variant": "public_current",
        },
    )


def _dreamer_role_key(role: str) -> str:
    """Canonical comparison key for the two native role-name options."""
    return "".join(character for character in role.casefold() if character.isalnum())


def _has_active_clue_result(card: CardInfo) -> bool:
    """True when an active ability entry contains a real clue result."""
    role = card.apparent_role.lower().replace(" ", "_")
    info = card.info_parsed or {}
    if type(info.get("shut_up_target")) is int:
        # Rambler2 replaces the normal result, but the active use was consumed.
        return True
    if role == "dreamer":
        return bool(info.get("target") or info.get("targets"))
    if role == "jester":
        return bool(
            info.get("callback_events")
            or (
                "targets" in info
                and "evil_count" in info
            )
        )
    if role in {"fortune_teller", "druid", "judge"}:
        return bool(info)
    return False


def _fortune_teller_native_text(targets: list[int], has_evil: bool) -> str:
    """Return the exact current-build Fortune Teller result sentence."""
    return (
        f"Is #{targets[0]} or #{targets[1]} Evil?: "
        f"{'True' if has_evil else 'False'}"
    )


def _fortune_teller_targets(
    value,
    *,
    label: str,
    n_cards: Optional[int],
    require_ascending: bool,
) -> list[int]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{label} must contain exactly two targets")
    if any(type(target) is not int for target in value):
        raise ValueError(f"{label} targets must be integers")
    targets = list(value)
    if targets[0] == targets[1]:
        raise ValueError(f"{label} targets must be distinct")
    if any(
        target <= 0 or (n_cards is not None and target > n_cards)
        for target in targets
    ):
        suffix = f"1..{n_cards}" if n_cards is not None else "positive"
        raise ValueError(f"{label} targets must be within {suffix}")
    if require_ascending and targets[0] >= targets[1]:
        raise ValueError(f"{label} targets must be stored in ascending ID order")
    return targets


def _fortune_teller_observation_history(
    info: dict,
    *,
    n_cards: Optional[int] = None,
    strict_native: bool,
) -> list[dict]:
    """Validate and return chronological normal Fortune Teller observations."""
    if not isinstance(info, dict):
        raise ValueError("Fortune Teller info_parsed must be an object")

    def validate_observation(
        observation,
        label: str,
        *,
        require_text: bool,
    ) -> dict:
        if not isinstance(observation, dict):
            raise ValueError(f"{label} must be an object")
        if "targets" not in observation or "has_evil" not in observation:
            raise ValueError(
                f"{label} must contain both targets and has_evil"
            )
        targets = _fortune_teller_targets(
            observation["targets"],
            label=label,
            n_cards=n_cards,
            require_ascending=strict_native,
        )
        has_evil = observation["has_evil"]
        if type(has_evil) is not bool:
            raise ValueError(f"{label}.has_evil must be a boolean")
        normalized = {"targets": targets, "has_evil": has_evil}
        if strict_native and require_text:
            text = observation.get("text")
            if not isinstance(text, str):
                raise ValueError(f"{label}.text must be a string")
            expected = _fortune_teller_native_text(targets, has_evil)
            if text != expected:
                raise ValueError(
                    f"{label}.text must exactly equal {expected!r}"
                )
            normalized["text"] = text
        elif isinstance(observation.get("text"), str):
            normalized["text"] = observation["text"]
        return normalized

    has_targets = "targets" in info
    has_result = "has_evil" in info
    top_level = None
    if has_targets != has_result:
        raise ValueError(
            "Fortune Teller info_parsed must contain both targets and "
            "has_evil, or neither"
        )
    if has_targets:
        top_level = validate_observation(
            info,
            "Fortune Teller top-level observation",
            require_text=False,
        )

    if "observations" in info:
        raw_observations = info["observations"]
        if not isinstance(raw_observations, list):
            raise ValueError("Fortune Teller observations must be an array")
        observations = [
            validate_observation(
                observation,
                f"Fortune Teller observations[{index}]",
                require_text=strict_native,
            )
            for index, observation in enumerate(raw_observations)
        ]
        if observations:
            if top_level is not None:
                latest = observations[-1]
                if (
                    top_level["targets"] != latest["targets"]
                    or top_level["has_evil"] != latest["has_evil"]
                ):
                    raise ValueError(
                        "Fortune Teller latest-value alias must match the "
                        "last chronological observation"
                    )
            return observations
        if strict_native and top_level is not None:
            raise ValueError(
                "Current Fortune Teller latest-value alias cannot use an "
                "explicitly empty observations array"
            )
        if (
            strict_native
            and top_level is None
            and type(info.get("shut_up_target")) is not int
        ):
            raise ValueError(
                "Current Fortune Teller empty observations require a newest "
                "Rambler interruption"
            )

    return [top_level] if top_level is not None else []


def _prepare_current_fortune_teller_card(
    card: CardInfo,
    *,
    n_cards: int,
) -> list[dict]:
    """Normalize manual input, then enforce current native evidence shape."""
    info = card.info_parsed
    if not isinstance(info, dict):
        raise ValueError("Fortune Teller info_parsed must be an object")

    has_targets = "targets" in info
    has_result = "has_evil" in info
    if has_targets != has_result:
        raise ValueError(
            "Fortune Teller info_parsed must contain both targets and "
            "has_evil, or neither"
        )
    if has_targets and "shut_up_target" in info:
        raise ValueError(
            "Current Fortune Teller evidence cannot combine a normal result "
            "with a Rambler interruption"
        )

    # The manual CLI supplies the public Boolean rather than raw memory. Give
    # that current-session input the exact native sorted alias/history shape.
    if has_targets and "observations" not in info:
        targets = _fortune_teller_targets(
            info["targets"],
            label="Fortune Teller manual result",
            n_cards=n_cards,
            require_ascending=False,
        )
        has_evil = info["has_evil"]
        if type(has_evil) is not bool:
            raise ValueError("Fortune Teller has_evil must be a boolean")
        targets.sort()
        text = _fortune_teller_native_text(targets, has_evil)
        if card.info_text and card.info_text != text:
            raise ValueError(
                "Fortune Teller info_text does not match the native result: "
                f"{card.info_text!r} != {text!r}"
            )
        card.info_text = text
        info["targets"] = targets
        info["observations"] = [
            {"targets": list(targets), "has_evil": has_evil, "text": text}
        ]

    history = _fortune_teller_observation_history(
        info,
        n_cards=n_cards,
        strict_native=True,
    )
    top_level_present = "targets" in info and "has_evil" in info
    interrupted = type(info.get("shut_up_target")) is int
    if history:
        if top_level_present:
            latest = history[-1]
            if card.info_text != latest["text"]:
                raise ValueError(
                    "Fortune Teller info_text must match the newest normal "
                    "observation"
                )
        elif not interrupted:
            raise ValueError(
                "Fortune Teller history without a latest-value alias is only "
                "valid when Rambler interrupted the newest use"
            )
    elif top_level_present:
        raise ValueError(
            "Fortune Teller latest-value alias requires a chronological "
            "observation"
        )
    elif interrupted:
        info.setdefault("observations", [])
    return history


def _judge_observation_history(
    info: dict,
    *,
    n_cards: Optional[int] = None,
) -> list[dict]:
    """Validate and return Judge-only evidence.

    Rambler interference may coexist with an empty Judge history, but a
    present Judge observation must be complete and typed.  Raising a focused
    ``ValueError`` here keeps malformed manual/session data from becoming an
    opaque Rust zero-scenario result (or a Python ``TypeError``).
    """
    if not isinstance(info, dict):
        raise ValueError("Judge info_parsed must be an object")

    def validate_observation(observation, label: str) -> dict:
        if not isinstance(observation, dict):
            raise ValueError(f"{label} must be an object")
        if "target" not in observation or "is_lying" not in observation:
            raise ValueError(
                f"{label} must contain both target and is_lying"
            )
        target = observation["target"]
        is_lying = observation["is_lying"]
        if type(target) is not int:
            raise ValueError(f"{label}.target must be an integer")
        if target <= 0 or (n_cards is not None and target > n_cards):
            suffix = f"1..{n_cards}" if n_cards is not None else "positive"
            raise ValueError(f"{label}.target must be within {suffix}")
        if type(is_lying) is not bool:
            raise ValueError(f"{label}.is_lying must be a boolean")
        return {"target": target, "is_lying": is_lying}

    has_target = "target" in info
    has_is_lying = "is_lying" in info
    top_level = None
    if has_target != has_is_lying:
        raise ValueError(
            "Judge info_parsed must contain both target and is_lying, or neither"
        )
    if has_target:
        top_level = validate_observation(info, "Judge top-level observation")

    if "observations" in info:
        raw_observations = info["observations"]
        if not isinstance(raw_observations, list):
            raise ValueError("Judge observations must be an array")
        observations = [
            validate_observation(
                observation,
                f"Judge observations[{index}]",
            )
            for index, observation in enumerate(raw_observations)
        ]
        if observations:
            return observations

    return [top_level] if top_level is not None else []


def _latest_acted_event_fingerprint(card: Optional[dict]):
    """Stable fingerprint of the newest public event, including history size."""
    if not isinstance(card, dict):
        return None
    infos = card.get("acted_infos")
    if not isinstance(infos, list) or not infos:
        return None
    try:
        newest = json.dumps(
            infos[-1],
            sort_keys=True,
            separators=(",", ":"),
            default=repr,
        )
    except (TypeError, ValueError):
        newest = repr(infos[-1])
    return len(infos), newest


def _acted_history_fingerprint(card: Optional[dict]):
    """Stable fingerprint of an actor's complete native callback list."""
    if not isinstance(card, dict):
        return None
    infos = card.get("acted_infos")
    if not isinstance(infos, list):
        return None
    try:
        encoded = json.dumps(
            infos,
            sort_keys=True,
            separators=(",", ":"),
            default=repr,
        )
    except (TypeError, ValueError):
        encoded = repr(infos)
    return len(infos), encoded


def _acted_history_snapshot(card: Optional[dict]) -> Optional[tuple[str, ...]]:
    """Canonical append-only native event prefix, or None when unreadable."""
    if not isinstance(card, dict):
        return None
    infos = card.get("acted_infos")
    if not isinstance(infos, list):
        return None
    snapshots = []
    for info in infos:
        try:
            snapshots.append(json.dumps(
                info,
                sort_keys=True,
                separators=(",", ":"),
                default=repr,
            ))
        except (TypeError, ValueError):
            return None
    return tuple(snapshots)


def _has_new_coherent_acted_suffix(
    card: Optional[dict],
    before: Optional[tuple[str, ...]],
) -> bool:
    """Require a strict event-list extension whose newest text is public."""
    if before is None or not isinstance(card, dict):
        return False
    after = _acted_history_snapshot(card)
    if (
        after is None
        or len(after) <= len(before)
        or after[:len(before)] != before
    ):
        return False
    infos = card.get("acted_infos")
    latest = infos[-1] if infos else None
    clue = card.get("clue_text")
    return (
        isinstance(latest, dict)
        and isinstance(latest.get("desc"), str)
        and bool(latest["desc"])
        and isinstance(clue, str)
        and clue == latest["desc"]
    )


def _newest_coherent_acted_refs(card: Optional[dict]) -> Optional[list[int]]:
    """Return click-order refs owned by the newest visible native event."""
    if not isinstance(card, dict):
        return None
    infos = card.get("acted_infos")
    latest = (
        infos[-1]
        if isinstance(infos, list) and infos and isinstance(infos[-1], dict)
        else None
    )
    if latest is None or latest.get("desc") != card.get("clue_text"):
        return None
    refs = latest.get("targets")
    if not isinstance(refs, list) or any(type(ref) is not int for ref in refs):
        return None
    return list(refs)


def _active_result_refs_match_clicks(
    card: Optional[dict],
    expected_targets: list[int],
    *,
    n_cards: int,
) -> bool:
    """Bind a normal Dreamer/Jester result to exact click chronology.

    Rambler replaces the original reference list, so a coherent shut-up event
    is authenticated by the strict append boundary instead.
    """
    refs = _newest_coherent_acted_refs(card)
    clue = card.get("clue_text") if isinstance(card, dict) else None
    if refs is None or not isinstance(clue, str):
        return False
    if _parse_shut_up_target_text(clue, n_cards=n_cards) is not None:
        return len(refs) == 1
    return refs == expected_targets


def _pickable_uses_remaining(card: Optional[dict]) -> Optional[int]:
    """Read native remaining Day-callback budget without legacy coercion."""
    if not isinstance(card, dict):
        return None
    remaining = card.get("pickable_uses_remaining")
    return remaining if type(remaining) is int else None


def _active_cycle_is_spent(card: Optional[dict]) -> bool:
    """Whether native current-cycle pickability is conclusively unavailable."""
    remaining = _pickable_uses_remaining(card)
    return remaining is not None and remaining <= 0


def _observed_active_role_key(card: Optional[dict]) -> str:
    """Normalize the public active identity carried by one memory snapshot."""
    if not isinstance(card, dict):
        return ""
    raw_role = (
        card.get("disguise")
        or card.get("current_role")
        or card.get("true_role")
        or ""
    )
    key = str(raw_role).strip().casefold().replace("-", "_").replace(" ", "_")
    return {
        "dreamer2": "dreamer",
        "juggler": "jester",
        "judge2": "judge",
        "librarian": "druid",
        "puzzlemaster": "plague_doctor",
        "rangedempath": "druid",
    }.get(key, key)


def _local_repeatable_event_expectation(
    card: CardInfo,
    *,
    n_cards: int,
    rambler_observations: list[dict],
    fortune_teller_rule_version: Optional[str],
) -> Optional[tuple[int, str, list[dict], list[int]]]:
    """Return the local normal/interruption history expected in native memory.

    ResetAfterNight memory is append-only, while the session retains normal
    observations plus Rambler replacements in a separate ledger.  Reconcile
    those two local surfaces before clicking so a stale-shorter pre-snapshot
    cannot make an old recovered event look new.
    """
    role_key = card.apparent_role.lower().replace(" ", "_")
    if role_key not in {"fortune_teller", "judge"}:
        return None
    info = card.info_parsed
    if not isinstance(info, dict):
        raise ValueError("local repeatable evidence must be an object")

    if role_key == "fortune_teller":
        normal_history = _fortune_teller_observation_history(
            info,
            n_cards=n_cards,
            strict_native=(
                fortune_teller_rule_version
                == FORTUNE_TELLER_RULE_VERSION
            ),
        )
    else:
        normal_history = _judge_observation_history(
            info,
            n_cards=n_cards,
        )

    interruption_targets = []
    for observation in rambler_observations:
        if (
            isinstance(observation, dict)
            and observation.get("speaker_position") == card.position
        ):
            target = observation.get("shut_up_target")
            if type(target) is not int or not 1 <= target <= n_cards:
                raise ValueError(
                    "local Rambler interruption history is malformed"
                )
            interruption_targets.append(target)

    interrupted = "shut_up_target" in info
    if interrupted:
        target = info.get("shut_up_target")
        if type(target) is not int or not 1 <= target <= n_cards:
            raise ValueError("local Rambler interruption target is malformed")
        expected_desc = f"#{target} shut up!"
        expected_targets = [target]
        # Older imported states may carry the latest interruption on the card
        # without its parallel ledger entry. The card still proves one event.
        if not interruption_targets:
            interruption_targets.append(target)
        elif interruption_targets[-1] != target:
            raise ValueError(
                "local latest interruption disagrees with the Rambler ledger"
            )
    elif normal_history:
        latest = normal_history[-1]
        if role_key == "fortune_teller":
            expected_targets = list(latest["targets"])
            expected_desc = _fortune_teller_native_text(
                expected_targets,
                latest["has_evil"],
            )
        else:
            expected_targets = [latest["target"]]
            expected_desc = (
                f"#{latest['target']} is\n"
                f"{'Lying' if latest['is_lying'] else 'saying Truth'}"
            )
    else:
        if card.info_text:
            raise ValueError(
                "local repeatable clue text has no structured event history"
            )
        return None

    if card.info_text and card.info_text != expected_desc:
        raise ValueError(
            "local repeatable clue text disagrees with structured evidence"
        )

    minimum_count = len(normal_history) + len(interruption_targets)
    expected = _latest_acted_event_fingerprint({
        "acted_infos": [{
            "desc": expected_desc,
            "targets": expected_targets,
        }],
    })
    normal_projection = [
        (
            {
                "targets": list(observation["targets"]),
                "has_evil": observation["has_evil"],
            }
            if role_key == "fortune_teller"
            else {
                "target": observation["target"],
                "is_lying": observation["is_lying"],
            }
        )
        for observation in normal_history
    ]
    return (
        minimum_count,
        expected[1],
        normal_projection,
        interruption_targets,
    )


def _repeatable_memory_history_projection(
    card: dict,
    *,
    role_key: str,
    n_cards: int,
) -> tuple[list[dict], list[int]]:
    """Project an exact FT/Judge native prefix into its two local ledgers."""
    raw_infos = card.get("acted_infos")
    if not isinstance(raw_infos, list):
        raise ValueError("pre-click acted-info history is unreadable")
    if raw_infos:
        if not isinstance(raw_infos[-1], dict):
            raise ValueError("newest acted-info history entry must be an object")
        if card.get("clue_text") != raw_infos[-1].get("desc"):
            raise ValueError(
                "pre-click savedAct does not match the newest acted-info event"
            )

    normal_history: list[dict] = []
    interruption_targets: list[int] = []
    for index, event in enumerate(raw_infos):
        if not isinstance(event, dict):
            raise ValueError(f"acted_infos[{index}] must be an object")
        desc = event.get("desc")
        refs = event.get("targets")
        if not isinstance(desc, str):
            raise ValueError(f"acted_infos[{index}].desc must be a string")
        shut_up_target = _parse_shut_up_target_text(desc, n_cards=n_cards)
        if shut_up_target is not None:
            if refs != [shut_up_target]:
                raise ValueError(
                    f"acted_infos[{index}] Rambler reference is malformed"
                )
            interruption_targets.append(shut_up_target)
            continue

        if role_key == "fortune_teller":
            if (
                not isinstance(refs, list)
                or len(refs) != 2
                or any(type(target) is not int for target in refs)
                or len(set(refs)) != 2
                or refs != sorted(refs)
                or any(not 1 <= target <= n_cards for target in refs)
            ):
                raise ValueError(
                    f"acted_infos[{index}] Fortune Teller references are malformed"
                )
            match = re.fullmatch(
                r"Is #(\d+) or #(\d+) Evil\?: (False|True)",
                desc,
            )
            if match is None or [int(match.group(1)), int(match.group(2))] != refs:
                raise ValueError(
                    f"acted_infos[{index}] Fortune Teller text/reference mismatch"
                )
            normal_history.append({
                "targets": list(refs),
                "has_evil": match.group(3) == "True",
            })
            continue

        if (
            not isinstance(refs, list)
            or len(refs) != 1
            or type(refs[0]) is not int
            or not 1 <= refs[0] <= n_cards
        ):
            raise ValueError(
                f"acted_infos[{index}] Judge reference is malformed"
            )
        match = re.fullmatch(r"#(\d+) is\n(saying Truth|Lying)", desc)
        if match is None or int(match.group(1)) != refs[0]:
            raise ValueError(
                f"acted_infos[{index}] Judge text/reference mismatch"
            )
        normal_history.append({
            "target": refs[0],
            "is_lying": match.group(2) == "Lying",
        })

    return normal_history, interruption_targets


def _classify_repeatable_memory_capture(
    existing: Optional[CardInfo],
    memory_card: dict,
    *,
    n_cards: int,
    rambler_observations: list[dict],
    fortune_teller_rule_version: Optional[str],
) -> tuple[str, Optional[str]]:
    """Classify an FT/Judge raw history as stale, extending, or conflicting."""
    role_key = _observed_active_role_key(memory_card)
    if role_key not in {"fortune_teller", "judge"}:
        return "error", "repeatable history classifier received another role"
    try:
        memory_normal, memory_interruptions = (
            _repeatable_memory_history_projection(
                memory_card,
                role_key=role_key,
                n_cards=n_cards,
            )
        )
        local = (
            _local_repeatable_event_expectation(
                existing,
                n_cards=n_cards,
                rambler_observations=rambler_observations,
                fortune_teller_rule_version=fortune_teller_rule_version,
            )
            if existing is not None else None
        )
    except ValueError as exc:
        return "error", str(exc)

    if local is None:
        if existing is not None and (existing.info_parsed or existing.info_text):
            return "error", (
                "local repeatable evidence cannot be projected into native history"
            )
        return "update", None

    _, expected_latest, local_normal, local_interruptions = local
    memory_latest = _latest_acted_event_fingerprint(memory_card)
    if (
        memory_normal == local_normal
        and memory_interruptions == local_interruptions
        and memory_latest is not None
        and memory_latest[1] == expected_latest
    ):
        return "stale", None
    if (
        len(memory_normal) >= len(local_normal)
        and memory_normal[:len(local_normal)] == local_normal
        and len(memory_interruptions) >= len(local_interruptions)
        and memory_interruptions[:len(local_interruptions)]
        == local_interruptions
        and (
            len(memory_normal) + len(memory_interruptions)
            > len(local_normal) + len(local_interruptions)
        )
    ):
        return "update", None
    return "error", (
        "raw repeatable history does not preserve the local ordered normal/"
        "interruption subsequences"
    )


def _parse_ambiguous_among(clue: Optional[str]) -> Optional[tuple[list[int], list[str]]]:
    """Parse "Among #X, #Y there is: R1 or R2" into (targets, role_options).

    Returns None if the clue is not in ambiguous-among form. Matches both
    newline-separated (game memory) and space-separated (human-typed) forms.
    Requires "or" between the two role names — rejects Oracle's "is a X" and
    Bishop's faction-list output.
    """
    if not clue:
        return None
    import re
    m = re.fullmatch(
        r'\s*Among\s+#\s*(\d+)\s*,\s*#\s*(\d+)\s+'
        r'there\s+is\s*:?\s*'
        r"([A-Za-z][A-Za-z _'-]*?)\s+or\s+"
        r"([A-Za-z][A-Za-z _'-]*?)\s*[.!]?\s*",
        clue,
        re.IGNORECASE | re.DOTALL,
    )
    if not m:
        return None
    targets = [int(m.group(1)), int(m.group(2))]
    options = [m.group(3).strip(), m.group(4).strip()]
    if any(re.search(r'\bor\b', option, re.IGNORECASE) for option in options):
        return None
    try:
        return (
            _validate_dreamer_targets(targets),
            _validate_dreamer_role_options(options),
        )
    except ValueError:
        return None


def _parse_cabbage_between(clue: Optional[str]) -> Optional[list[int]]:
    """Parse "Between #X, #Y there is: a Cabbage" into two target IDs."""
    if not clue:
        return None
    import re
    m = re.fullmatch(
        r'\s*Between\s+#\s*(\d+)\s*,\s*#\s*(\d+)\s+'
        r'there\s+is\s*:\s*a\s+Cabbage\s*[.!]?\s*',
        clue,
        re.IGNORECASE | re.DOTALL,
    )
    if not m:
        return None
    try:
        return _validate_dreamer_targets([int(m.group(1)), int(m.group(2))])
    except ValueError:
        return None

def card_judge(pos: int, target: int, is_lying: bool) -> CardInfo:
    return CardInfo(pos, "Judge", info_parsed={"target": target, "is_lying": is_lying})

def card_alchemist(pos: int, corrupted_count: int) -> CardInfo:
    """Post-patch: clue is # Corrupted around me [Range 2] at start of Round (before Cure)."""
    return CardInfo(pos, "Alchemist", info_parsed={"corrupted_count": corrupted_count})

def _validate_current_druid_targets(
    targets,
    *,
    n_cards: Optional[int] = None,
) -> list[int]:
    """Validate Druid's three original-click-order Character references."""
    if (
        not isinstance(targets, list)
        or len(targets) != 3
        or any(type(target) is not int or target <= 0 for target in targets)
        or len(set(targets)) != 3
        or (
            n_cards is not None
            and (
                type(n_cards) is not int
                or n_cards <= 0
                or any(target > n_cards for target in targets)
            )
        )
    ):
        raise ValueError(
            "Current Druid targets must be three distinct current-board positions"
        )
    return list(targets)


def _canonical_druid_outcast(found_outcast) -> Optional[str]:
    """Return the solver's canonical token for one public Outcast name."""
    if found_outcast is None:
        return None
    if not isinstance(found_outcast, str) or not found_outcast:
        raise ValueError("Druid result must be none or a canonical Outcast role")
    role = get_card(found_outcast)
    if role is None or role.role.value != "Outcast":
        raise ValueError("Druid result must be none or a canonical Outcast role")
    return role.name.replace(" ", "_")


def _druid_native_text(
    targets: list[int],
    found_outcast: Optional[str],
) -> str:
    """Return Librarian's exact result, sorting IDs but not stored references."""
    click_order = _validate_current_druid_targets(targets)
    displayed = sorted(click_order)
    prefix = f"Among #{displayed[0]}, #{displayed[1]}, #{displayed[2]}\n"
    canonical = _canonical_druid_outcast(found_outcast)
    if canonical is None:
        return prefix + "there are NO Outcasts"
    public_name = get_card(canonical).name
    return prefix + f"there is: {public_name}"


def _parse_druid_native_text(info_text) -> Optional[tuple[list[int], Optional[str]]]:
    """Parse only one exact current Druid result sentence."""
    if not isinstance(info_text, str):
        return None
    none_match = re.fullmatch(
        r"Among #([1-9]\d*), #([1-9]\d*), #([1-9]\d*)\n"
        r"there are NO Outcasts",
        info_text,
    )
    positive_match = re.fullmatch(
        r"Among #([1-9]\d*), #([1-9]\d*), #([1-9]\d*)\n"
        r"there is: ([^\r\n]+)",
        info_text,
    )
    match = none_match or positive_match
    if match is None:
        return None
    try:
        displayed = [int(match.group(index)) for index in range(1, 4)]
        displayed = _validate_current_druid_targets(displayed)
        found_outcast = (
            None
            if none_match is not None
            else _canonical_druid_outcast(match.group(4))
        )
    except (ValueError, TypeError):
        return None
    if displayed != sorted(displayed):
        return None
    try:
        expected = _druid_native_text(displayed, found_outcast)
    except ValueError:
        return None
    return (displayed, found_outcast) if info_text == expected else None


def card_druid(
    pos: int,
    targets: list[int],
    found_outcast: Optional[str] = None,
    *,
    info_text: str = "",
    druid_variant: Optional[str] = None,
) -> CardInfo:
    """Build a Druid observation while preserving unmarked archive payloads."""
    info = {
        "targets": list(targets) if isinstance(targets, list) else targets,
        "found_outcast": found_outcast,
    }
    if druid_variant is not None:
        if druid_variant != _PUBLIC_CURRENT_VARIANT:
            raise ValueError("Unsupported Druid variant")
        if type(pos) is not int or pos <= 0:
            raise ValueError("Current Druid position must be positive")
        current_targets = _validate_current_druid_targets(targets)
        current_outcast = _canonical_druid_outcast(found_outcast)
        if not isinstance(info_text, str):
            raise ValueError("Current Druid text must be a string")
        expected_text = _druid_native_text(current_targets, current_outcast)
        if info_text and info_text != expected_text:
            raise ValueError("Current Druid text must match its targets and result")
        info = {
            "targets": current_targets,
            "found_outcast": current_outcast,
            "druid_variant": druid_variant,
        }
        info_text = expected_text
    return CardInfo(pos, "Druid", info_text=info_text, info_parsed=info)


def _druid_shut_up_text(target: int) -> str:
    """Return the exact native Rambler replacement shown by Druid."""
    if type(target) is not int or target <= 0:
        raise ValueError("Druid Rambler target must be a positive integer")
    return f"#{target}\nshut up!"


def _card_current_druid_interruption(
    pos: int,
    target: int,
    *,
    n_cards: Optional[int] = None,
) -> CardInfo:
    """Build one provenance-marked current Druid interruption surface."""
    if type(pos) is not int or pos <= 0:
        raise ValueError("Current Druid position must be positive")
    if type(target) is not int or target <= 0:
        raise ValueError("Druid Rambler target must be a positive integer")
    if n_cards is not None and (pos > n_cards or target > n_cards):
        raise ValueError("Current Druid interruption is outside the board")
    return CardInfo(
        pos,
        "Druid",
        info_text=_druid_shut_up_text(target),
        info_parsed={
            "shut_up_target": target,
            "druid_variant": _PUBLIC_CURRENT_VARIANT,
        },
    )


_ORDERED_CALLBACK_LEDGER_VARIANT = "ordered_callbacks_v1"
_CALLBACK_COMMON_FIELDS = {
    "activation_id",
    "activation_evidence",
    "callback_index",
    "dispatch_path",
    "event_kind",
    "reset_generation",
    "text",
    "references",
    "settled_reveal_count",
}
_CALLBACK_ACTIVATION_EVIDENCE = {
    "single_callback_suffix",
    "auto_use_click",
    "session_reset_generation",
    "same_activation_extension",
}
_DRUID_RESULT_CALLBACK_FIELDS = _CALLBACK_COMMON_FIELDS | {
    "targets",
    "found_outcast",
}
_DRUID_INTERRUPTION_CALLBACK_FIELDS = _CALLBACK_COMMON_FIELDS | {
    "shut_up_target",
}


def _validate_callback_references(
    references,
    *,
    n_cards: int,
    label: str,
):
    """Validate the public ActedInfo reference surface for one callback."""
    if references is None:
        return None
    if (
        not isinstance(references, list)
        or any(
            type(reference) is not int
            or not 1 <= reference <= n_cards
            for reference in references
        )
    ):
        raise ValueError(
            f"{label}.references must be null or current-board positions"
        )
    return list(references)


def _druid_scalar_callback(info: dict, *, n_cards: int) -> dict:
    """Validate one pre-ledger current result and return its raw callback."""
    if not isinstance(info, dict):
        raise ValueError("Druid info_parsed must be an object")
    if info.get("druid_variant") != _PUBLIC_CURRENT_VARIANT:
        raise ValueError("Current Druid evidence requires public_current provenance")
    if "observations" in info:
        raise ValueError(
            "Legacy marked Druid observations cannot be safely resumed; "
            "restart the village from verified reveal history"
        )
    if "targets" in info or "found_outcast" in info:
        if set(info) != {"druid_variant", "targets", "found_outcast"}:
            raise ValueError("Current scalar Druid result has unsupported fields")
        targets = _validate_current_druid_targets(
            info.get("targets"),
            n_cards=n_cards,
        )
        found_outcast = _canonical_druid_outcast(info.get("found_outcast"))
        if info.get("found_outcast") != found_outcast:
            raise ValueError("Current Druid found_outcast must be canonical")
        return {
            "event_kind": "druid_result",
            "text": _druid_native_text(targets, found_outcast),
            "references": list(targets),
            "targets": list(targets),
            "found_outcast": found_outcast,
        }
    if "shut_up_target" in info:
        raise ValueError(
            "A current Druid interruption cannot be persisted from manual "
            "scalar input; recover its authenticated raw callback history"
        )
    raise ValueError("Current scalar Druid evidence has no callback result")


def _validate_ordered_callback_groups(
    events,
    *,
    actor_position: int,
    n_cards: int,
    reveal_order: Optional[list[int]] = None,
    baker_rule_version: Optional[str] = None,
) -> list[dict]:
    """Validate role-agnostic activation grouping and public boundaries."""
    if not isinstance(events, list) or not events:
        raise ValueError("Current callback_events must be a nonempty array")
    if reveal_order is not None:
        if baker_rule_version != BAKER_RULE_VERSION:
            raise ValueError(
                "Current callback history requires verified reveal_order/Baker "
                "provenance"
            )
        if (
            len(reveal_order) > n_cards
            or any(
                type(position) is not int
                or not 1 <= position <= n_cards
                for position in reveal_order
            )
            or len(set(reveal_order)) != len(reveal_order)
        ):
            raise ValueError("Verified reveal_order is malformed")

    normalized = []
    groups: list[list[dict]] = []
    previous_activation = 0
    previous_boundary = 0
    previous_reset_generation = -1
    for index, raw_event in enumerate(events):
        label = f"Druid callback_events[{index}]"
        if not isinstance(raw_event, dict):
            raise ValueError(f"{label} must be an object")
        event = copy.deepcopy(raw_event)
        activation_id = event.get("activation_id")
        callback_index = event.get("callback_index")
        activation_evidence = event.get("activation_evidence")
        dispatch_path = event.get("dispatch_path")
        reset_generation = event.get("reset_generation")
        settled = event.get("settled_reveal_count")
        if type(activation_id) is not int or activation_id <= 0:
            raise ValueError(f"{label}.activation_id must be positive")
        if type(callback_index) is not int or callback_index < 0:
            raise ValueError(f"{label}.callback_index must be nonnegative")
        if dispatch_path not in {"either", "real", "raw"}:
            raise ValueError(f"{label}.dispatch_path is unsupported")
        if activation_evidence not in _CALLBACK_ACTIVATION_EVIDENCE:
            raise ValueError(f"{label}.activation_evidence is unsupported")
        if type(reset_generation) is not int or reset_generation < 0:
            raise ValueError(f"{label}.reset_generation must be nonnegative")
        if (
            type(settled) is not int
            or settled < 1
            or settled > n_cards
        ):
            raise ValueError(
                f"{label}.settled_reveal_count must be within 1..{n_cards}"
            )
        if reveal_order is not None:
            if settled > len(reveal_order):
                raise ValueError(
                    f"{label}.settled_reveal_count exceeds verified reveals"
                )
            if actor_position not in reveal_order[:settled]:
                raise ValueError(
                    f"Druid actor #{actor_position} is absent from the "
                    f"verified reveal prefix for {label}"
                )
        event["references"] = _validate_callback_references(
            event.get("references"),
            n_cards=n_cards,
            label=label,
        )
        if activation_id != previous_activation:
            if activation_id != previous_activation + 1 or callback_index != 0:
                raise ValueError(
                    "Druid activation IDs and callback indices must be "
                    "contiguous from activation 1"
                )
            if settled < previous_boundary:
                raise ValueError(
                    "Druid settled reveal boundaries must be nondecreasing"
                )
            if reset_generation <= previous_reset_generation:
                raise ValueError(
                    "Druid reset generations must increase between activations"
                )
            groups.append([])
            previous_activation = activation_id
            previous_boundary = settled
            previous_reset_generation = reset_generation
        else:
            if not groups or callback_index != len(groups[-1]):
                raise ValueError(
                    "Druid callback indices must be contiguous within an activation"
                )
            if settled != previous_boundary:
                raise ValueError(
                    "Every callback in one Druid activation must share one "
                    "settled reveal boundary"
                )
            if reset_generation != previous_reset_generation:
                raise ValueError(
                    "Every callback in one Druid activation must share one "
                    "reset generation"
                )
            if activation_evidence != groups[-1][0]["activation_evidence"]:
                raise ValueError(
                    "Every callback in one Druid activation must share one "
                    "activation evidence value"
                )
        groups[-1].append(event)
        normalized.append(event)

    for activation_id, group in enumerate(groups, start=1):
        paths = [event["dispatch_path"] for event in group]
        activation_evidence = group[0]["activation_evidence"]
        if len(group) == 1:
            if paths != ["either"]:
                raise ValueError(
                    f"Druid activation {activation_id} single callback must "
                    "use dispatch_path either"
                )
            if activation_evidence == "same_activation_extension":
                raise ValueError(
                    f"Druid activation {activation_id} cannot use "
                    "same_activation_extension for one callback"
                )
        elif len(group) == 2:
            if paths != ["real", "raw"]:
                raise ValueError(
                    f"Druid activation {activation_id} must dispatch real then raw"
                )
            if activation_evidence == "single_callback_suffix":
                raise ValueError(
                    f"Druid activation {activation_id} cannot use "
                    "single_callback_suffix for two callbacks"
                )
        else:
            raise ValueError(
                f"Druid activation {activation_id} has more than two callbacks"
            )
    return normalized


def _validate_druid_callback_event(
    event: dict,
    *,
    n_cards: int,
    label: str,
) -> dict:
    """Validate one kind-specific callback after common group validation."""
    kind = event.get("event_kind")
    if kind == "druid_result":
        if set(event) != _DRUID_RESULT_CALLBACK_FIELDS:
            raise ValueError(f"{label} has unsupported Druid-result fields")
        targets = _validate_current_druid_targets(
            event.get("targets"),
            n_cards=n_cards,
        )
        found_outcast = _canonical_druid_outcast(event.get("found_outcast"))
        if event.get("found_outcast") != found_outcast:
            raise ValueError(f"{label}.found_outcast must be canonical")
        expected_text = _druid_native_text(targets, found_outcast)
        if event.get("text") != expected_text:
            raise ValueError(f"{label}.text must exactly equal {expected_text!r}")
        if event.get("references") != targets:
            raise ValueError(f"{label}.references must equal click-order targets")
        event["targets"] = targets
        event["found_outcast"] = found_outcast
        return event
    if kind == "rambler_interruption":
        if set(event) != _DRUID_INTERRUPTION_CALLBACK_FIELDS:
            raise ValueError(f"{label} has unsupported interruption fields")
        target = event.get("shut_up_target")
        if type(target) is not int or not 1 <= target <= n_cards:
            raise ValueError(f"{label}.shut_up_target is outside the board")
        if event.get("text") != _druid_shut_up_text(target):
            raise ValueError(f"{label}.text is not the exact interruption text")
        if event.get("references") != [target]:
            raise ValueError(f"{label}.references must be [{target}]")
        return event
    if kind == "opaque_real":
        if set(event) != _CALLBACK_COMMON_FIELDS:
            raise ValueError(f"{label} has unsupported opaque callback fields")
        if not isinstance(event.get("text"), str) or not event["text"]:
            raise ValueError(f"{label}.text must preserve a nonempty callback")
        return event
    raise ValueError(f"{label}.event_kind is unsupported")


def _druid_callback_ledger(
    info: dict,
    *,
    actor_position: int,
    n_cards: int,
    reveal_order: Optional[list[int]] = None,
    baker_rule_version: Optional[str] = None,
) -> list[dict]:
    """Validate the strict ordered current-Druid callback ledger."""
    if not isinstance(info, dict):
        raise ValueError("Druid info_parsed must be an object")
    if info.get("druid_variant") != _PUBLIC_CURRENT_VARIANT:
        raise ValueError("Current Druid ledger requires public_current provenance")
    if info.get("callback_ledger_variant") != _ORDERED_CALLBACK_LEDGER_VARIANT:
        if "observations" in info:
            raise ValueError(
                "Legacy marked Druid observations cannot be safely resumed; "
                "restart the village from verified reveal history"
            )
        raise ValueError(
            "Scalar-only current Druid evidence cannot be safely resumed; "
            "restart the village from verified reveal history"
        )
    events = _validate_ordered_callback_groups(
        info.get("callback_events"),
        actor_position=actor_position,
        n_cards=n_cards,
        reveal_order=reveal_order,
        baker_rule_version=baker_rule_version,
    )
    events = [
        _validate_druid_callback_event(
            event,
            n_cards=n_cards,
            label=f"Druid callback_events[{index}]",
        )
        for index, event in enumerate(events)
    ]
    for group_id in range(1, events[-1]["activation_id"] + 1):
        group = [event for event in events if event["activation_id"] == group_id]
        if group[0]["event_kind"] == "opaque_real" and (
            len(group) != 2
            or group[0]["dispatch_path"] != "real"
            or group[1]["dispatch_path"] != "raw"
            or group[1]["event_kind"] == "opaque_real"
        ):
            raise ValueError(
                "An opaque real callback must be followed by one raw public "
                "Druid callback in the same activation"
            )
        if group[-1]["event_kind"] == "opaque_real":
            raise ValueError("A Druid activation cannot end with an opaque callback")
        if len(group) == 2:
            interruption_flags = [
                event["event_kind"] == "rambler_interruption"
                for event in group
            ]
            if interruption_flags[0] != interruption_flags[1]:
                raise ValueError(
                    "Both callbacks in one Druid activation must either be "
                    "Rambler interruptions or both remain non-interruptions"
                )
            if all(interruption_flags) and (
                group[0]["shut_up_target"] != group[1]["shut_up_target"]
            ):
                raise ValueError(
                    "Both Rambler callbacks in one Druid activation must name "
                    "the same physical target"
                )
            if all(
                event["event_kind"] == "druid_result" for event in group
            ) and group[0]["targets"] != group[1]["targets"]:
                raise ValueError(
                    "Both Druid-result callbacks in one activation must share "
                    "the same click-order targets"
                )

    latest = events[-1]
    common_fields = {
        "druid_variant",
        "callback_ledger_variant",
        "callback_events",
    }
    if latest["event_kind"] == "druid_result":
        if set(info) != common_fields | {"targets", "found_outcast"}:
            raise ValueError("Current Druid latest normal alias is malformed")
        if (
            info.get("targets") != latest["targets"]
            or info.get("found_outcast") != latest["found_outcast"]
        ):
            raise ValueError(
                "Current Druid latest alias must match the final callback"
            )
    else:
        if set(info) != common_fields | {"shut_up_target"}:
            raise ValueError("Current Druid latest interruption alias is malformed")
        if info.get("shut_up_target") != latest["shut_up_target"]:
            raise ValueError(
                "Current Druid interruption alias must match the final callback"
            )
    return events


def _stamp_druid_callback_group(
    callbacks: list[dict],
    *,
    activation_id: int,
    activation_evidence: str,
    reset_generation: int,
    settled_reveal_count: int,
) -> list[dict]:
    """Stamp one provable one- or two-dispatch activation."""
    if not 1 <= len(callbacks) <= 2:
        raise ValueError(
            "A newly captured Druid activation must have one or two callbacks"
        )
    if callbacks[-1].get("event_kind") == "opaque_real":
        raise ValueError("A Druid activation is still awaiting its public callback")
    if activation_evidence not in _CALLBACK_ACTIVATION_EVIDENCE:
        raise ValueError("Druid activation evidence is unsupported")
    if (
        activation_evidence == "single_callback_suffix"
        and len(callbacks) != 1
    ):
        raise ValueError(
            "single_callback_suffix can authenticate only one callback"
        )
    if (
        activation_evidence == "same_activation_extension"
        and len(callbacks) != 2
    ):
        raise ValueError(
            "same_activation_extension requires exactly two callbacks"
        )
    if type(reset_generation) is not int or reset_generation < 0:
        raise ValueError("Druid reset generation must be nonnegative")
    stamped = []
    for callback_index, callback in enumerate(callbacks):
        event = copy.deepcopy(callback)
        event.update({
            "activation_id": activation_id,
            "activation_evidence": activation_evidence,
            "callback_index": callback_index,
            "dispatch_path": (
                "either"
                if len(callbacks) == 1
                else "real" if callback_index == 0 else "raw"
            ),
            "reset_generation": reset_generation,
            "settled_reveal_count": settled_reveal_count,
        })
        stamped.append(event)
    if stamped[0]["event_kind"] == "opaque_real" and len(stamped) != 2:
        raise ValueError("Opaque real callback has no matching raw Druid callback")
    return stamped


def _druid_callback_signature(event: dict) -> dict:
    """Return exactly the two public fields present in raw ActedInfo."""
    return {
        "desc": event["text"],
        "targets": copy.deepcopy(event["references"]),
    }


def _druid_interruption_records(
    events: list[dict],
    *,
    speaker_position: int,
) -> list[dict]:
    """Project every ordered Druid interruption into global Rambler shape."""
    return [
        {
            "speaker_position": speaker_position,
            "shut_up_target": event["shut_up_target"],
        }
        for event in events
        if event.get("event_kind") == "rambler_interruption"
    ]


def _validate_druid_rambler_sync(
    events: list[dict],
    *,
    speaker_position: int,
    rambler_observations,
) -> None:
    """Require exact per-speaker parity with the global Rambler ledger."""
    if not isinstance(rambler_observations, list):
        raise ValueError("Global Rambler evidence must be an array")
    expected = _druid_interruption_records(
        events,
        speaker_position=speaker_position,
    )
    actual = []
    for observation in rambler_observations:
        if (
            isinstance(observation, dict)
            and observation.get("speaker_position") == speaker_position
        ):
            if set(observation) != {"speaker_position", "shut_up_target"}:
                raise ValueError(
                    "Same-speaker global Druid/Rambler evidence has unsupported "
                    "fields"
                )
            actual.append(dict(observation))
    if actual != expected:
        raise ValueError(
            "Persisted Druid interruptions disagree with the exact same-speaker "
            "global Rambler evidence"
        )


def _apply_druid_callback_ledger(card: CardInfo, events: list[dict]) -> None:
    """Install one validated ledger and its exact latest-value alias."""
    latest = events[-1]
    info = {
        "druid_variant": _PUBLIC_CURRENT_VARIANT,
        "callback_ledger_variant": _ORDERED_CALLBACK_LEDGER_VARIANT,
        "callback_events": copy.deepcopy(events),
    }
    if latest["event_kind"] == "druid_result":
        info["targets"] = list(latest["targets"])
        info["found_outcast"] = latest["found_outcast"]
    elif latest["event_kind"] == "rambler_interruption":
        info["shut_up_target"] = latest["shut_up_target"]
    else:
        raise ValueError("Current Druid latest callback is not publicly usable")
    card.info_parsed = info
    card.info_text = latest["text"]


_DRUID_RAW_RESULT_FIELDS = {
    "event_kind", "text", "references", "targets", "found_outcast",
}
_DRUID_RAW_INTERRUPTION_FIELDS = {
    "event_kind", "text", "references", "shut_up_target",
}
_DRUID_RAW_OPAQUE_FIELDS = {"event_kind", "text", "references"}
_DRUID_PENDING_FIELDS = {
    "activation_id",
    "expected_targets",
    "prior_callback_count",
    "reset_generation",
    "settled_reveal_count",
}


def _validate_raw_druid_callbacks(callbacks, *, n_cards: int) -> list[dict]:
    """Revalidate transient callbacks before trusting a session join."""
    if not isinstance(callbacks, list) or not callbacks:
        raise ValueError("Druid raw callback history must be a nonempty array")
    normalized = []
    for index, raw_callback in enumerate(callbacks):
        label = f"Druid raw callback[{index}]"
        if not isinstance(raw_callback, dict):
            raise ValueError(f"{label} must be an object")
        callback = copy.deepcopy(raw_callback)
        kind = callback.get("event_kind")
        if kind == "druid_result":
            if set(callback) != _DRUID_RAW_RESULT_FIELDS:
                raise ValueError(f"{label} has unsupported result fields")
            targets = _validate_current_druid_targets(
                callback.get("targets"),
                n_cards=n_cards,
            )
            found = _canonical_druid_outcast(callback.get("found_outcast"))
            if callback.get("found_outcast") != found:
                raise ValueError(f"{label}.found_outcast must be canonical")
            if callback.get("references") != targets:
                raise ValueError(f"{label}.references must equal click order")
            if callback.get("text") != _druid_native_text(targets, found):
                raise ValueError(f"{label}.text is not exact")
            callback["targets"] = targets
        elif kind == "rambler_interruption":
            if set(callback) != _DRUID_RAW_INTERRUPTION_FIELDS:
                raise ValueError(f"{label} has unsupported interruption fields")
            target = callback.get("shut_up_target")
            if type(target) is not int or not 1 <= target <= n_cards:
                raise ValueError(f"{label}.shut_up_target is outside the board")
            if callback.get("references") != [target]:
                raise ValueError(f"{label}.references must equal [{target}]")
            if callback.get("text") != _druid_shut_up_text(target):
                raise ValueError(f"{label}.text is not exact")
        elif kind == "opaque_real":
            if set(callback) != _DRUID_RAW_OPAQUE_FIELDS:
                raise ValueError(f"{label} has unsupported opaque fields")
            if not isinstance(callback.get("text"), str) or not callback["text"]:
                raise ValueError(f"{label}.text must be nonempty")
            callback["references"] = _validate_callback_references(
                callback.get("references"),
                n_cards=n_cards,
                label=label,
            )
        else:
            raise ValueError(f"{label}.event_kind is unsupported")
        normalized.append(callback)
    if normalized[-1]["event_kind"] == "opaque_real":
        raise ValueError("Druid raw history is still awaiting its raw callback")
    return normalized


def _validate_druid_pending_token(
    token,
    *,
    actor_position: int,
    n_cards: int,
    reveal_order: list[int],
    reset_generation: int,
    prior_callback_count: int,
    next_activation_id: int,
) -> dict:
    """Validate a persisted automated-click token without mutating it."""
    if not isinstance(token, dict) or set(token) != _DRUID_PENDING_FIELDS:
        raise ValueError("Pending Druid auto-use token is malformed")
    settled = token.get("settled_reveal_count")
    if (
        type(settled) is not int
        or not 1 <= settled <= len(reveal_order)
        or actor_position not in reveal_order[:settled]
    ):
        raise ValueError("Pending Druid auto-use reveal boundary is invalid")
    if token.get("reset_generation") != reset_generation:
        raise ValueError("Pending Druid auto-use reset generation is stale")
    if token.get("prior_callback_count") != prior_callback_count:
        raise ValueError("Pending Druid auto-use callback prefix is stale")
    if token.get("activation_id") != next_activation_id:
        raise ValueError("Pending Druid auto-use activation ID is stale")
    targets = _validate_current_druid_targets(
        token.get("expected_targets"),
        n_cards=n_cards,
    )
    normalized = copy.deepcopy(token)
    normalized["expected_targets"] = targets
    return normalized


_JESTER_RESULT_CALLBACK_FIELDS = _CALLBACK_COMMON_FIELDS | {
    "targets",
    "evil_count",
}
_JESTER_INTERRUPTION_CALLBACK_FIELDS = _CALLBACK_COMMON_FIELDS | {
    "shut_up_target",
}
_JESTER_RAW_RESULT_FIELDS = {
    "event_kind", "text", "references", "evil_count",
}
_JESTER_RAW_INTERRUPTION_FIELDS = {
    "event_kind", "text", "references", "shut_up_target",
}
_JESTER_RAW_OPAQUE_FIELDS = {"event_kind", "text", "references"}
_JESTER_PENDING_FIELDS = {
    "activation_id",
    "expected_targets",
    "prior_callback_count",
    "reset_generation",
    "settled_reveal_count",
}


def _jester_shut_up_text(target: int) -> str:
    """Return the exact shared Rambler replacement text."""
    if type(target) is not int or target <= 0:
        raise ValueError("Jester Rambler target must be a positive integer")
    return f"#{target}\nshut up!"


def _card_current_jester_interruption(
    pos: int,
    target: int,
    *,
    n_cards: Optional[int] = None,
) -> CardInfo:
    """Build one provenance-marked current Jester interruption surface."""
    if type(pos) is not int or pos <= 0:
        raise ValueError("Current Jester position must be positive")
    if type(target) is not int or target <= 0:
        raise ValueError("Jester Rambler target must be a positive integer")
    if n_cards is not None and (pos > n_cards or target > n_cards):
        raise ValueError("Current Jester interruption is outside the board")
    return CardInfo(
        pos,
        "Jester",
        info_text=_jester_shut_up_text(target),
        info_parsed={
            "shut_up_target": target,
            "jester_variant": _PUBLIC_CURRENT_VARIANT,
        },
    )


def _jester_scalar_callback(info: dict, *, n_cards: int) -> dict:
    """Validate one non-resumable current scalar result."""
    if not isinstance(info, dict):
        raise ValueError("Jester info_parsed must be an object")
    if info.get("jester_variant") != _PUBLIC_CURRENT_VARIANT:
        raise ValueError("Current Jester evidence requires public_current provenance")
    if set(info) != {"jester_variant", "targets", "evil_count"}:
        if "shut_up_target" in info:
            raise ValueError(
                "A current Jester interruption cannot be persisted from manual "
                "scalar input; recover its authenticated raw callback history"
            )
        raise ValueError("Current scalar Jester result has unsupported fields")
    targets = _validate_current_jester_targets(
        info.get("targets"),
        n_cards=n_cards,
    )
    evil_count = info.get("evil_count")
    if type(evil_count) is not int or not 0 <= evil_count <= 3:
        raise ValueError("Current Jester evil_count must be an integer from 0 to 3")
    return {
        "event_kind": "jester_result",
        "text": _jester_native_text(targets, evil_count),
        "references": list(targets),
        "targets": list(targets),
        "evil_count": evil_count,
    }


def _validate_jester_callback_event(
    event: dict,
    *,
    n_cards: int,
    label: str,
) -> dict:
    """Validate one Jester-specific event after common group validation."""
    kind = event.get("event_kind")
    if kind == "jester_result":
        if set(event) != _JESTER_RESULT_CALLBACK_FIELDS:
            raise ValueError(f"{label} has unsupported Jester-result fields")
        targets = _validate_current_jester_targets(
            event.get("targets"),
            n_cards=n_cards,
        )
        evil_count = event.get("evil_count")
        if type(evil_count) is not int or not 0 <= evil_count <= 3:
            raise ValueError(f"{label}.evil_count must be an integer from 0 to 3")
        references = _validate_jester_reference_ids(
            event.get("references"),
            n_cards=n_cards,
        )
        expected = _jester_native_text(references, evil_count)
        if event.get("text") != expected:
            raise ValueError(f"{label}.text must exactly equal {expected!r}")
        event["targets"] = targets
        event["references"] = references
        return event
    if kind == "rambler_interruption":
        if set(event) != _JESTER_INTERRUPTION_CALLBACK_FIELDS:
            raise ValueError(f"{label} has unsupported interruption fields")
        target = event.get("shut_up_target")
        if type(target) is not int or not 1 <= target <= n_cards:
            raise ValueError(f"{label}.shut_up_target is outside the board")
        if event.get("text") != _jester_shut_up_text(target):
            raise ValueError(f"{label}.text is not the exact interruption text")
        if event.get("references") != [target]:
            raise ValueError(f"{label}.references must be [{target}]")
        return event
    if kind == "opaque_real":
        if set(event) != _CALLBACK_COMMON_FIELDS:
            raise ValueError(f"{label} has unsupported opaque callback fields")
        text = event.get("text")
        if not isinstance(text, str) or not text:
            raise ValueError(f"{label}.text must preserve a nonempty callback")
        if _parse_jester_native_text(text) is not None or _looks_like_shut_up_text(text):
            raise ValueError(f"{label}.text cannot hide a Jester/Rambler callback")
        return event
    raise ValueError(f"{label}.event_kind is unsupported")


def _jester_callback_ledger(
    info: dict,
    *,
    actor_position: int,
    n_cards: int,
    reveal_order: Optional[list[int]] = None,
    baker_rule_version: Optional[str] = None,
) -> list[dict]:
    """Validate the strict ordered current-Jester callback ledger."""
    if not isinstance(info, dict):
        raise ValueError("Jester info_parsed must be an object")
    if info.get("jester_variant") != _PUBLIC_CURRENT_VARIANT:
        raise ValueError("Current Jester ledger requires public_current provenance")
    if info.get("callback_ledger_variant") != _ORDERED_CALLBACK_LEDGER_VARIANT:
        raise ValueError(
            "Scalar-only current Jester evidence cannot be safely resumed; "
            "restart the village from verified reveal history"
        )
    events = _validate_ordered_callback_groups(
        info.get("callback_events"),
        actor_position=actor_position,
        n_cards=n_cards,
        reveal_order=reveal_order,
        baker_rule_version=baker_rule_version,
    )
    events = [
        _validate_jester_callback_event(
            event,
            n_cards=n_cards,
            label=f"Jester callback_events[{index}]",
        )
        for index, event in enumerate(events)
    ]
    for activation_id in range(1, events[-1]["activation_id"] + 1):
        group = [event for event in events if event["activation_id"] == activation_id]
        if group[0]["event_kind"] == "opaque_real" and (
            len(group) != 2
            or group[0]["dispatch_path"] != "real"
            or group[1]["dispatch_path"] != "raw"
            or group[1]["event_kind"] == "opaque_real"
        ):
            raise ValueError(
                "An opaque real callback must be followed by one raw public "
                "Jester callback in the same activation"
            )
        if group[-1]["event_kind"] == "opaque_real":
            raise ValueError("A Jester activation cannot end with an opaque callback")
        normal = [event for event in group if event["event_kind"] == "jester_result"]
        if len(normal) == 2:
            if normal[0]["targets"] != normal[1]["targets"]:
                raise ValueError(
                    "Both Jester-result callbacks in one activation must share "
                    "the same physical click-order targets"
                )
            if normal[0]["references"] != normal[1]["references"]:
                raise ValueError(
                    "Both Jester-result callbacks in one activation must share "
                    "the same native reference-ID chronology"
                )

    latest = events[-1]
    common_fields = {
        "jester_variant",
        "callback_ledger_variant",
        "callback_events",
    }
    if latest["event_kind"] == "jester_result":
        if set(info) != common_fields | {"targets", "evil_count"}:
            raise ValueError("Current Jester latest normal alias is malformed")
        if (
            info.get("targets") != latest["targets"]
            or info.get("evil_count") != latest["evil_count"]
        ):
            raise ValueError("Current Jester latest alias must match the final callback")
    else:
        if set(info) != common_fields | {"shut_up_target"}:
            raise ValueError("Current Jester latest interruption alias is malformed")
        if info.get("shut_up_target") != latest["shut_up_target"]:
            raise ValueError(
                "Current Jester interruption alias must match the final callback"
            )
    return events


def _stamp_jester_callback_group(
    callbacks: list[dict],
    *,
    activation_id: int,
    activation_evidence: str,
    reset_generation: int,
    settled_reveal_count: int,
) -> list[dict]:
    """Stamp one provable one- or two-dispatch Jester activation."""
    if not 1 <= len(callbacks) <= 2:
        raise ValueError("A newly captured Jester activation needs one or two callbacks")
    if callbacks[-1].get("event_kind") == "opaque_real":
        raise ValueError("A Jester activation is still awaiting its public callback")
    if activation_evidence not in _CALLBACK_ACTIVATION_EVIDENCE:
        raise ValueError("Jester activation evidence is unsupported")
    if activation_evidence == "single_callback_suffix" and len(callbacks) != 1:
        raise ValueError("single_callback_suffix can authenticate only one callback")
    if activation_evidence == "same_activation_extension" and len(callbacks) != 2:
        raise ValueError("same_activation_extension requires exactly two callbacks")
    if type(reset_generation) is not int or reset_generation < 0:
        raise ValueError("Jester reset generation must be nonnegative")
    stamped = []
    for callback_index, callback in enumerate(callbacks):
        event = copy.deepcopy(callback)
        event.update({
            "activation_id": activation_id,
            "activation_evidence": activation_evidence,
            "callback_index": callback_index,
            "dispatch_path": (
                "either"
                if len(callbacks) == 1
                else "real" if callback_index == 0 else "raw"
            ),
            "reset_generation": reset_generation,
            "settled_reveal_count": settled_reveal_count,
        })
        stamped.append(event)
    if stamped[0]["event_kind"] == "opaque_real" and len(stamped) != 2:
        raise ValueError("Opaque real callback has no matching raw Jester callback")
    return stamped


def _bind_jester_physical_targets(
    callbacks: list[dict],
    targets,
    *,
    n_cards: int,
) -> list[dict]:
    """Attach one proven physical picker chronology to normal callbacks."""
    physical_targets = _validate_current_jester_targets(
        targets,
        n_cards=n_cards,
    )
    bound = copy.deepcopy(callbacks)
    for callback in bound:
        if callback.get("event_kind") == "jester_result":
            callback["targets"] = list(physical_targets)
    return bound


def _infer_jester_physical_targets(
    callbacks: list[dict],
    *,
    n_cards: int,
) -> list[dict]:
    """Use reference IDs as seats only when that mapping is unambiguous."""
    normal = [
        callback
        for callback in callbacks
        if callback.get("event_kind") == "jester_result"
    ]
    if not normal:
        return copy.deepcopy(callbacks)
    references = normal[0].get("references")
    try:
        physical_targets = _validate_current_jester_targets(
            references,
            n_cards=n_cards,
        )
    except ValueError as exc:
        raise ValueError(
            "Jester native reference IDs do not uniquely identify the three "
            "physical picker targets; authenticated click provenance is required"
        ) from exc
    if any(callback.get("references") != references for callback in normal[1:]):
        raise ValueError(
            "Jester normal callbacks disagree on their native reference-ID "
            "chronology"
        )
    return _bind_jester_physical_targets(
        callbacks,
        physical_targets,
        n_cards=n_cards,
    )


def _jester_callback_signature(event: dict) -> dict:
    """Return exactly the public fields stored in one raw ActedInfo."""
    return {
        "desc": event["text"],
        "targets": copy.deepcopy(event["references"]),
    }


def _jester_interruption_records(
    events: list[dict],
    *,
    speaker_position: int,
) -> list[dict]:
    """Project each independently replaced callback into global Rambler state."""
    return [
        {
            "speaker_position": speaker_position,
            "shut_up_target": event["shut_up_target"],
        }
        for event in events
        if event.get("event_kind") == "rambler_interruption"
    ]


def _validate_jester_rambler_sync(
    events: list[dict],
    *,
    speaker_position: int,
    rambler_observations,
) -> None:
    """Require exact event-local parity with the shared Rambler ledger."""
    if not isinstance(rambler_observations, list):
        raise ValueError("Global Rambler evidence must be an array")
    expected = _jester_interruption_records(
        events,
        speaker_position=speaker_position,
    )
    actual = []
    for observation in rambler_observations:
        if (
            isinstance(observation, dict)
            and observation.get("speaker_position") == speaker_position
        ):
            if set(observation) != {"speaker_position", "shut_up_target"}:
                raise ValueError(
                    "Same-speaker global Jester/Rambler evidence has unsupported fields"
                )
            actual.append(dict(observation))
    if actual != expected:
        raise ValueError(
            "Persisted Jester interruptions disagree with the exact same-speaker "
            "global Rambler evidence"
        )


def _apply_jester_callback_ledger(card: CardInfo, events: list[dict]) -> None:
    """Install one validated ledger and its exact newest-value alias."""
    latest = events[-1]
    info = {
        "jester_variant": _PUBLIC_CURRENT_VARIANT,
        "callback_ledger_variant": _ORDERED_CALLBACK_LEDGER_VARIANT,
        "callback_events": copy.deepcopy(events),
    }
    if latest["event_kind"] == "jester_result":
        info["targets"] = list(latest["targets"])
        info["evil_count"] = latest["evil_count"]
    elif latest["event_kind"] == "rambler_interruption":
        info["shut_up_target"] = latest["shut_up_target"]
    else:
        raise ValueError("Current Jester latest callback is not publicly usable")
    card.info_parsed = info
    card.info_text = latest["text"]


def _validate_raw_jester_callbacks(callbacks, *, n_cards: int) -> list[dict]:
    """Revalidate transient Jester callbacks before a session join."""
    if not isinstance(callbacks, list):
        raise ValueError("Jester raw callback history must be an array")
    normalized = []
    for index, raw_callback in enumerate(callbacks):
        label = f"Jester raw callback[{index}]"
        if not isinstance(raw_callback, dict):
            raise ValueError(f"{label} must be an object")
        callback = copy.deepcopy(raw_callback)
        kind = callback.get("event_kind")
        if kind == "jester_result":
            if set(callback) != _JESTER_RAW_RESULT_FIELDS:
                raise ValueError(f"{label} has unsupported result fields")
            references = _validate_jester_reference_ids(
                callback.get("references"),
                n_cards=n_cards,
            )
            evil_count = callback.get("evil_count")
            if type(evil_count) is not int or not 0 <= evil_count <= 3:
                raise ValueError(f"{label}.evil_count must be an integer from 0 to 3")
            if callback.get("text") != _jester_native_text(references, evil_count):
                raise ValueError(f"{label}.text is not exact")
            callback["references"] = references
        elif kind == "rambler_interruption":
            if set(callback) != _JESTER_RAW_INTERRUPTION_FIELDS:
                raise ValueError(f"{label} has unsupported interruption fields")
            target = callback.get("shut_up_target")
            if type(target) is not int or not 1 <= target <= n_cards:
                raise ValueError(f"{label}.shut_up_target is outside the board")
            if callback.get("references") != [target]:
                raise ValueError(f"{label}.references must equal [{target}]")
            if callback.get("text") != _jester_shut_up_text(target):
                raise ValueError(f"{label}.text is not exact")
        elif kind == "opaque_real":
            if set(callback) != _JESTER_RAW_OPAQUE_FIELDS:
                raise ValueError(f"{label} has unsupported opaque fields")
            text = callback.get("text")
            if not isinstance(text, str) or not text:
                raise ValueError(f"{label}.text must be nonempty")
            if _parse_jester_native_text(text) is not None or _looks_like_shut_up_text(text):
                raise ValueError(f"{label}.text cannot hide a Jester/Rambler callback")
            callback["references"] = _validate_callback_references(
                callback.get("references"),
                n_cards=n_cards,
                label=label,
            )
        else:
            raise ValueError(f"{label}.event_kind is unsupported")
        normalized.append(callback)
    if normalized and normalized[-1]["event_kind"] == "opaque_real":
        raise ValueError("Jester raw history is still awaiting its raw callback")
    return normalized


def _validate_jester_pending_token(
    token,
    *,
    actor_position: int,
    n_cards: int,
    reveal_order: list[int],
    reset_generation: int,
    prior_callback_count: int,
    next_activation_id: int,
) -> dict:
    """Validate a persisted automated Jester click token."""
    if not isinstance(token, dict) or set(token) != _JESTER_PENDING_FIELDS:
        raise ValueError("Pending Jester auto-use token is malformed")
    settled = token.get("settled_reveal_count")
    if (
        type(settled) is not int
        or not 1 <= settled <= len(reveal_order)
        or actor_position not in reveal_order[:settled]
    ):
        raise ValueError("Pending Jester auto-use reveal boundary is invalid")
    if token.get("reset_generation") != reset_generation:
        raise ValueError("Pending Jester auto-use reset generation is stale")
    if token.get("prior_callback_count") != prior_callback_count:
        raise ValueError("Pending Jester auto-use callback prefix is stale")
    if token.get("activation_id") != next_activation_id:
        raise ValueError("Pending Jester auto-use activation ID is stale")
    targets = _validate_current_jester_targets(
        token.get("expected_targets"),
        n_cards=n_cards,
    )
    normalized = copy.deepcopy(token)
    normalized["expected_targets"] = targets
    return normalized

_BISHOP_PUBLIC_TYPES = ("Villager", "Outcast", "Minion", "Demon")


def _bishop_native_text(targets: list[int], types: list[str]) -> str:
    """Return Bishop's exact shipped public sentence."""
    if (
        not isinstance(targets, list)
        or not 1 <= len(targets) <= 3
        or any(type(target) is not int or target <= 0 for target in targets)
        or len(set(targets)) != len(targets)
        or targets != sorted(targets)
    ):
        raise ValueError(
            "Current Bishop targets must be one to three ascending unique positions"
        )
    if (
        not isinstance(types, list)
        or len(types) != len(targets)
        or any(role_type not in _BISHOP_PUBLIC_TYPES for role_type in types)
    ):
        raise ValueError(
            "Current Bishop types must match its targets and use canonical factions"
        )

    if len(targets) == 1:
        return f"#{targets[0]} is a {types[0]}"
    ids = ", ".join(f"#{target}" for target in targets)
    if len(types) == 2:
        type_text = f"{types[0]} and {types[1]}"
    else:
        type_text = f"{types[0]}, {types[1]} and {types[2]}"
    return f"Between\n{ids}\nthere is:\n{type_text}"


def _parse_bishop_native_text(
    info_text: str,
) -> Optional[tuple[list[int], list[str]]]:
    """Parse only one exact current Bishop sentence."""
    if not isinstance(info_text, str):
        return None
    type_pattern = "(?:Villager|Outcast|Minion|Demon)"
    patterns = (
        rf"#([1-9]\d*) is a ({type_pattern})",
        rf"Between\n#([1-9]\d*), #([1-9]\d*)\nthere is:\n"
        rf"({type_pattern}) and ({type_pattern})",
        rf"Between\n#([1-9]\d*), #([1-9]\d*), #([1-9]\d*)\nthere is:\n"
        rf"({type_pattern}), ({type_pattern}) and ({type_pattern})",
    )
    for count, pattern in enumerate(patterns, start=1):
        match = re.fullmatch(pattern, info_text)
        if match is None:
            continue
        targets = [int(match.group(index)) for index in range(1, count + 1)]
        types = [
            match.group(index)
            for index in range(count + 1, (count * 2) + 1)
        ]
        try:
            expected = _bishop_native_text(targets, types)
        except ValueError:
            return None
        if info_text == expected:
            return targets, types
    return None


def _bishop_refs_match(
    displayed: list[int],
    refs,
    *,
    n_cards: int,
) -> bool:
    """Whether native shuffled ActedInfo refs are the displayed ID set."""
    return (
        isinstance(refs, list)
        and len(refs) == len(displayed)
        and all(
            type(ref) is int and 1 <= ref <= n_cards
            for ref in refs
        )
        and len(set(refs)) == len(refs)
        and set(refs) == set(displayed)
    )


def card_bishop(
    pos: int,
    targets: list[int],
    types: list[str] = None,
    *,
    info_text: str = "",
    bishop_variant: Optional[str] = None,
) -> CardInfo:
    """Build a Bishop observation while preserving unmarked legacy callers."""
    info = {"targets": targets}
    if types:
        info["types"] = types
    if bishop_variant is not None:
        expected_text = _bishop_native_text(targets, types)
        if info_text and info_text != expected_text:
            raise ValueError("Current Bishop text must match its targets and types")
        info["bishop_variant"] = bishop_variant
        info_text = expected_text
    return CardInfo(pos, "Bishop", info_text=info_text, info_parsed=info)

def _canonical_poet_provider(provider: str) -> str:
    """Return one exact current-build Poet provider in canonical public form."""
    if not isinstance(provider, str):
        raise ValueError("Poet copied role must be a provider name")
    key = re.sub(r"[^a-z0-9]", "", provider.casefold())
    for canonical in POET_PROVIDER_ROLES:
        if key == re.sub(r"[^a-z0-9]", "", canonical.casefold()):
            return canonical
    raise ValueError(
        f"Unsupported current Poet provider {provider!r}; expected one of "
        + ", ".join(POET_PROVIDER_ROLES)
    )


def _card_current_poet(
    pos: int,
    provider: str,
    info_parsed: dict,
    *,
    info_text: str = "",
) -> CardInfo:
    """Build a provenance-marked current Poet observation."""
    info = dict(info_parsed)
    info["copied_role"] = _canonical_poet_provider(provider)
    info["poet_variant"] = POET_VARIANT
    return CardInfo(pos, "Poet", info_text=info_text, info_parsed=info)


def _bounty_hunter_native_text(evil_position: int) -> str:
    """Return Bounty Hunter's exact shipped public clue text."""
    return f"#{evil_position}\nis Evil"


def card_bounty_hunter(
    pos: int,
    evil_position: int,
    *,
    info_text: Optional[str] = None,
) -> CardInfo:
    """Poet's retained Bounty Hunter direct-evil provider."""
    if type(evil_position) is not int or evil_position <= 0:
        raise ValueError("Bounty Hunter Poet clue requires a positive position")
    if info_text is None:
        info_text = _bounty_hunter_native_text(evil_position)
    return _card_current_poet(
        pos,
        "Bounty Hunter",
        {"evil_position": evil_position},
        info_text=info_text,
    )


def card_poet_with_info(
    pos: int,
    copied_role: str,
    copied_args: list[str],
    *,
    n_cards: Optional[int] = None,
) -> CardInfo:
    """Poet clue parser.

    Usage: card poet <pos> <copied_role> <copied_role_args...>
    Examples:
        card poet 5 knitter 0          (Poet gave Knitter-style clue)
        card poet 3 lover 2            (Poet gave Lover-style clue)
        card poet 2 gemcrafter 5       (Poet gave Gemcrafter-style clue)
        card poet 4 bard 1             (Poet gave Bard-style clue)
        card poet 1 bounty_hunter 6    (Poet directly named #6 as Evil)
    """
    canonical_provider = _canonical_poet_provider(copied_role)
    if n_cards is not None:
        if type(n_cards) is not int or n_cards <= 0:
            raise ValueError("Poet board size must be a positive integer")
        if type(pos) is not int or not 1 <= pos <= n_cards:
            raise ValueError("Poet position is outside the current board")

    def require_args(count: int) -> None:
        if len(copied_args) != count:
            raise ValueError(
                f"{canonical_provider} Poet clue requires exactly {count} "
                f"argument{'s' if count != 1 else ''}"
            )

    def integer(index: int, field: str) -> int:
        try:
            return int(copied_args[index])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{canonical_provider} Poet {field} must be an integer"
            ) from exc

    def position(index: int, field: str) -> int:
        value = integer(index, field)
        if value <= 0 or (n_cards is not None and value > n_cards):
            raise ValueError(
                f"{canonical_provider} Poet {field} must be a current-board position"
            )
        return value

    def positions(
        index: int,
        count: int,
        *,
        allow_duplicates: bool = False,
    ) -> list[int]:
        try:
            values = [int(value.strip()) for value in copied_args[index].split(',')]
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError(
                f"{canonical_provider} Poet targets must be comma-separated positions"
            ) from exc
        if len(values) != count or any(
            value <= 0 or (n_cards is not None and value > n_cards)
            for value in values
        ):
            raise ValueError(
                f"{canonical_provider} Poet clue requires exactly {count} current-board "
                f"target{'s' if count != 1 else ''}"
            )
        if not allow_duplicates and len(set(values)) != len(values):
            raise ValueError(f"{canonical_provider} Poet targets must be distinct")
        return values

    def canonical_role(
        index: int,
        field: str,
        *,
        factions: Optional[set[str]] = None,
    ) -> str:
        role = get_card(copied_args[index])
        if role is None or (factions is not None and role.role.value not in factions):
            allowed = (
                ""
                if factions is None
                else " " + "/".join(sorted(factions))
            )
            raise ValueError(
                f"{canonical_provider} Poet {field} must be a canonical{allowed} role"
            )
        return role.name

    if canonical_provider == "Lover":
        require_args(1)
        evil_adjacent = integer(0, "evil count")
        if evil_adjacent not in {0, 1, 2}:
            raise ValueError("Lover Poet evil count must be 0, 1, or 2")
        return _card_current_poet(
            pos,
            "Lover",
            {"evil_adjacent": evil_adjacent},
            info_text=_lover_native_text(evil_adjacent),
        )
    elif canonical_provider == "Scout":
        sentinel_key = re.sub(
            r"[^a-z0-9]",
            "",
            " ".join(copied_args).casefold(),
        )
        if sentinel_key in {"oneevil", "thereisonly1evil"}:
            payload = {"one_evil": True}
        else:
            require_args(2)
            distance = integer(1, "distance")
            if not _valid_current_scout_distance(distance, n_cards):
                raise ValueError("Scout Poet distance is outside the native range")
            payload = {
                # Truth names GetRegisterAs(), while bluff names current
                # dataRef. Either can be a Good role after an identity move.
                "evil_role": canonical_role(0, "named role"),
                "distance": distance,
            }
    elif canonical_provider == "Oracle":
        sentinel_key = re.sub(
            r"[^a-z0-9]",
            "",
            " ".join(copied_args).casefold(),
        )
        if sentinel_key in {"nominions", "therearenominions"}:
            return _card_current_poet(
                pos,
                "Oracle",
                {"no_minions": True},
                info_text="There are no minions",
            )
        require_args(2)
        oracle_targets = positions(0, 2, allow_duplicates=True)
        if oracle_targets != sorted(oracle_targets):
            raise ValueError("Oracle Poet targets must be in ascending ID order")
        minion_role = canonical_role(
            1,
            "minion role",
            factions={"Minion"},
        )
        return _card_current_poet(
            pos,
            "Oracle",
            {
                "targets": oracle_targets,
                "minion_role": minion_role,
            },
            info_text=_oracle_native_text(oracle_targets, minion_role),
        )
    elif canonical_provider == "Bounty Hunter":
        require_args(1)
        return card_bounty_hunter(pos, position(0, "evil target"))
    elif canonical_provider == "Medium":
        require_args(2)
        good_position = position(0, "good target")
        good_role = canonical_role(1, "good role")
        return _card_current_poet(
            pos,
            "Medium",
            {
                "good_position": good_position,
                "good_role": good_role,
            },
            info_text=_medium_native_text(good_position, good_role),
        )
    elif canonical_provider == "Knitter":
        if n_cards is None:
            raise ValueError("Current Knitter Poet entry requires session board size")
        require_args(1)
        evil_pairs = integer(0, "pair count")
        if evil_pairs < 0 or evil_pairs > n_cards:
            raise ValueError("Knitter Poet pair count is outside the current board")
        return _card_current_poet(
            pos,
            "Knitter",
            {"evil_pairs": evil_pairs},
            info_text=_knitter_native_text(evil_pairs),
        )
    elif canonical_provider == "Hunter":
        require_args(1)
        distance = integer(0, "distance")
        if not _valid_current_hunter_distance(distance, n_cards):
            raise ValueError("Hunter Poet distance is outside the native range")
        payload = {"distance": distance}
    elif canonical_provider == "Enlightened":
        if n_cards is None:
            raise ValueError(
                "Current Enlightened Poet entry requires session board size"
            )
        require_args(1)
        direction = _canonical_enlightened_direction(copied_args[0])
        return _card_current_poet(
            pos,
            "Enlightened",
            {"direction": direction},
            info_text=_enlightened_native_text(direction),
        )
    elif canonical_provider == "Empress":
        if n_cards is None:
            raise ValueError(
                "Current Empress Poet entry requires session board size"
            )
        require_args(1)
        empress_targets = sorted(positions(0, 3))
        empress_targets = _validate_current_empress_targets(empress_targets)
        return _card_current_poet(
            pos,
            "Empress",
            {"targets": empress_targets},
            info_text=_empress_native_text(empress_targets),
        )
    elif canonical_provider == "Bishop":
        if n_cards is None:
            raise ValueError(
                "Current Bishop Poet entry requires session board size"
            )
        require_args(2)
        target_values = [value.strip() for value in copied_args[0].split(',')]
        if not 1 <= len(target_values) <= 3:
            raise ValueError("Bishop Poet clue requires one to three targets")
        target_count = len(target_values)
        target_positions = sorted(positions(0, target_count))
        type_values = [value.strip().casefold() for value in copied_args[1].split(',')]
        type_names = {
            "villager": "Villager",
            "outcast": "Outcast",
            "minion": "Minion",
            "demon": "Demon",
        }
        if len(type_values) != target_count or any(
            value not in type_names for value in type_values
        ):
            raise ValueError(
                "Bishop Poet types must match its targets and be Villager, "
                "Outcast, Minion, or Demon"
            )
        bishop_types = [type_names[value] for value in type_values]
        return _card_current_poet(
            pos,
            "Bishop",
            {"targets": target_positions, "types": bishop_types},
            info_text=_bishop_native_text(target_positions, bishop_types),
        )
    elif canonical_provider == "Gemcrafter":
        if n_cards is None:
            raise ValueError(
                "Current Gemcrafter Poet entry requires session board size"
            )
        require_args(1)
        good_position = position(0, "good target")
        return _card_current_poet(
            pos,
            "Gemcrafter",
            {"good_position": good_position},
            info_text=_gemcrafter_native_text(good_position),
        )
    elif canonical_provider == "Bard":
        if n_cards is None:
            raise ValueError("Current Bard Poet entry requires session board size")
        require_args(1)
        corruption_distance = integer(0, "corruption distance")
        if corruption_distance == 0:
            corruption_distance = -1
        if not _valid_current_bard_distance(corruption_distance, n_cards):
            raise ValueError("Bard Poet distance is outside the native range")
        return _card_current_poet(
            pos,
            "Bard",
            {"corruption_distance": corruption_distance},
            info_text=_bard_native_text(corruption_distance),
        )
    else:
        raise ValueError(f"Unsupported current Poet provider {canonical_provider!r}")

    return _card_current_poet(pos, canonical_provider, payload)


def _canonical_baker_original_role(original_role: str) -> str:
    """Validate and canonicalize the scalar stored by a Baker clue.

    ``original`` is a dedicated public-text sentinel. A literal ``Baker`` is
    an ordinary canonical role claim and must never alias that sentinel.
    """
    if not isinstance(original_role, str):
        raise ValueError("Baker original role must be a role name or 'original'")
    candidate = original_role.strip()
    if not candidate or candidate.casefold() in {"none", "unknown", "?"}:
        raise ValueError("Baker original role must be a known role or 'original'")
    if candidate.casefold() == "original":
        return "original"
    role_def = get_card(candidate)
    if role_def is None:
        raise ValueError(f"Unknown Baker original role: {original_role!r}")
    return role_def.name


def card_baker(
    pos: int,
    original_role: str,
    info_text: str = "",
) -> CardInfo:
    """Baker: 'I am the original Baker' or 'I was a <Role>'.

    original_role: 'original' for the first Baker, or the Villager role name
    the Baker claims to have been before conversion.
    """
    return CardInfo(
        pos,
        "Baker",
        info_text=info_text,
        info_parsed={
            "original_role": _canonical_baker_original_role(original_role),
        },
    )


def _normalize_role_name(role: str) -> str:
    """Normalize a role name to its canonical form using the knowledge base."""
    from knowledge_base import get_card
    card_def = get_card(role)
    if card_def:
        return card_def.name.replace(" ", "_")
    return role


def _execution_role_key(role: str | None) -> str:
    """Normalize role/status text for post-execution identity checks."""
    return (role or "").strip().replace("_", " ").replace("-", " ").casefold()


def _canonical_terminal_loss_role(role: str | None) -> Optional[str]:
    """Return the sole public CharacterData role that ends the game on death.

    Managed ``Saint`` is the implementation class behind public Bombardier;
    public CharacterData named Saint is a different role and must not alias it.
    """
    if _execution_role_key(role) == "bombardier":
        return "Bombardier"
    return None


def _consensus_original_evil_role(
    position: int,
    result,
    current_role: str | None,
) -> Optional[str]:
    """Keep runtime-Evil identity separate from a transformed current role.

    Solver worlds retain the original Evil assignment after current-data
    mutation. Current CharacterData is never an origin fallback: zero worlds
    invokes recovery/manual entry instead of hidden-memory guessing.
    """
    scenarios = list(getattr(result, "surviving_scenarios", []) or [])
    if not scenarios:
        return None

    roles: set[str] = set()
    for scenario in scenarios:
        role = scenario.evil_positions.get(position)
        if role is None and scenario.puppet_position == position:
            role = "Puppet"
        if role is None:
            return None
        roles.add(_normalize_role_name(role))
    if len(roles) == 1:
        return next(iter(roles))
    return None


def _observed_current_role(observed: dict | None) -> str | None:
    """Read current CharacterData with compatibility for older reader output."""
    if not observed:
        return None
    return observed.get("current_role") or observed.get("true_role")


def _observed_status_keys(observed: dict | None) -> set[str]:
    """Return normalized live status names from a post-action observation."""
    if not observed:
        return set()
    return {
        _execution_role_key(str(status)).replace(" ", "")
        for status in observed.get("statuses", [])
    }


def _is_known_role(role: str | None) -> bool:
    """Whether a role value is precise enough for persistent bookkeeping."""
    return bool(
        role
        and role.strip().casefold() not in {"unknown", "?", "none"}
    )


def _execution_apparent_role(observed: dict | None,
                             fallback_role: str | None = None) -> str | None:
    """Return the displayed role from a post-action memory observation.

    The live bluff pointer is preferred.  A card entry is UI-derived and is a
    safe fallback when memory has no bluff object; current data is last.
    """
    if observed:
        return observed.get("disguise") or fallback_role or _observed_current_role(observed)
    return fallback_role


def _observed_knight_immunity(observed: dict | None,
                              fallback_role: str | None = None,
                              *,
                              current_identity_may_have_moved: bool = False,
                              ) -> bool:
    """Whether a just-attempted execution is natively consistent with immunity.

    This is intentionally a post-action validator, never a pre-click decision
    helper.  A clean good true Knight is protected.  A Doppelganger showing as
    Knight is protected only while HealthyBluff makes it delegate protection to
    the bluff role.  Drunk-as-Knight and other merely apparent Knights remain
    killable and must not be auto-labelled immune from deck/card data alone.
    """
    # After a current-data mover, HealthyBluff Knight protection can live on a
    # runtime-Evil physical card. Survival is then alignment-neutral; hidden
    # memory alignment must never turn it into confirmed-Good evidence.
    if current_identity_may_have_moved:
        return False
    if not observed or observed.get("state") not in ("Alive", "Revealed"):
        return False
    if observed.get("is_evil") is not False:
        return False
    apparent_role = _execution_apparent_role(observed, fallback_role)
    if _execution_role_key(apparent_role) != "knight":
        return False

    true_role = _execution_role_key(_observed_current_role(observed))
    statuses = _observed_status_keys(observed)
    healthy_bluff = "healthybluff" in statuses
    corrupted = "corrupted" in statuses
    if true_role in ("knight", "immortal"):
        return healthy_bluff or not corrupted
    if true_role in ("doppelganger", "doppleganger"):
        return healthy_bluff
    return False


def _clamped_post_damage_hp(current_hp: int, damage: int) -> int:
    """Mirror CurrentMaxValue.Reduce's lower clamp for local bookkeeping."""
    return max(0, current_hp - damage)


def card_no_info(pos: int, role: str) -> CardInfo:
    """For cards with no deduction info: Slayer, Knight, Bombardier, Wretch, etc."""
    role = _normalize_role_name(role)
    return CardInfo(pos, role, info_parsed={})


SESSION_FILE = os.path.join(os.path.dirname(__file__), "game_session.json")
SCREENSHOTS_DIR = os.path.join(os.path.dirname(__file__), "screenshots")


def cleanup_screenshots(keep: int = 20):
    """Delete old screenshots, keeping only the most recent `keep` files.

    Issue #11: Prevents disk fill over 100+ games (~4MB/game).
    """
    if not os.path.isdir(SCREENSHOTS_DIR):
        return 0
    files = []
    for f in os.listdir(SCREENSHOTS_DIR):
        path = os.path.join(SCREENSHOTS_DIR, f)
        if os.path.isfile(path) and f.lower().endswith(('.png', '.jpg', '.jpeg')):
            files.append((os.path.getmtime(path), path))
    files.sort(reverse=True)  # newest first
    to_delete = files[keep:]
    for _, path in to_delete:
        try:
            os.remove(path)
        except OSError:
            pass
    if to_delete:
        print(f"[cleanup] Deleted {len(to_delete)} old screenshots (kept {min(keep, len(files))})")
    return len(to_delete)
DECISION_LOG = os.path.join(os.path.dirname(__file__), "game_session_state.md")
_SESSION_LOCK_HANDLE = None
_SESSION_LOCK_PATH: Optional[str] = None


def _release_session_lock():
    global _SESSION_LOCK_HANDLE, _SESSION_LOCK_PATH
    if _SESSION_LOCK_HANDLE is None:
        return

    handle = _SESSION_LOCK_HANDLE
    path = _SESSION_LOCK_PATH or ""
    try:
        if os.name == "nt":
            import msvcrt
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        handle.close()
        _SESSION_LOCK_HANDLE = None
        _SESSION_LOCK_PATH = None


def _acquire_session_lock(path: str = SESSION_FILE, timeout_s: float = 5.0):
    """Acquire a per-command lock for the session file.

    Returns the lock handle. Caller MUST call _release_session_lock() when done.
    For in-process/REPL use, this is called per save()/load() and released
    immediately after the IO completes, preventing deadlocks on reentrant calls.
    """
    global _SESSION_LOCK_HANDLE, _SESSION_LOCK_PATH
    if _SESSION_LOCK_HANDLE is not None:
        if _SESSION_LOCK_PATH == path:
            return _SESSION_LOCK_HANDLE
        _release_session_lock()

    lock_path = f"{path}.lock"
    os.makedirs(os.path.dirname(lock_path), exist_ok=True)
    handle = open(lock_path, "a+b")
    deadline = time.time() + timeout_s

    while True:
        try:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except OSError:
            if time.time() >= deadline:
                handle.close()
                raise TimeoutError(f"Timed out acquiring session lock for {path}")
            time.sleep(0.05)

    _SESSION_LOCK_HANDLE = handle
    _SESSION_LOCK_PATH = path
    return handle


atexit.register(_release_session_lock)


# ============================================================
# Decision Log
# ============================================================

class DecisionLog:
    """Append-only markdown log of every decision in the current game."""

    @staticmethod
    def _ts() -> str:
        return datetime.now().strftime("%H:%M:%S")

    @staticmethod
    def start_game(n_cards: int, n_evil: int, hp: int, cost: int):
        with open(DECISION_LOG, "a") as f:
            f.write(f"\n---\n\n")
            f.write(f"# New Game — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Cards: {n_cards}, Evil: {n_evil}, HP: {hp}, Wrong exec cost: {cost}\n\n")

    @staticmethod
    def log_deck(villagers, outcasts, minions, demons):
        with open(DECISION_LOG, "a") as f:
            f.write(f"## Deck\n")
            f.write(f"- Villagers: {', '.join(villagers)}\n")
            f.write(f"- Outcasts: {', '.join(outcasts)}\n")
            f.write(f"- Minions: {', '.join(minions)}\n")
            f.write(f"- Demons: {', '.join(demons)}\n\n")

    @staticmethod
    def log_card(card: CardInfo):
        with open(DECISION_LOG, "a") as f:
            f.write(f"### [{DecisionLog._ts()}] Revealed #{card.position} {card.apparent_role}\n")
            f.write(f"Info: {card.info_parsed}\n\n")

    @staticmethod
    def log_solver_output(result: SolverResult, state: GameState):
        with open(DECISION_LOG, "a") as f:
            f.write(f"#### [{DecisionLog._ts()}] Solver Output\n")
            f.write(f"Scenarios: {result.n_surviving}/{result.n_scenarios}\n")
            if result.definite_evil:
                f.write(f"Definite evil: {['#'+str(p) for p in result.definite_evil]}\n")
            if result.definite_good:
                f.write(f"Definite good: {['#'+str(p) for p in result.definite_good]}\n")
            if result.n_surviving > 0:
                probs = evil_probabilities(state, result)
                uncertain = {p: prob for p, prob in probs.items()
                             if 0 < prob < 1 and p not in state.executed}
                if uncertain:
                    f.write(f"Evil probabilities: " +
                            ", ".join(f"#{p}={prob:.0%}" for p, prob in
                                      sorted(uncertain.items(), key=lambda x: -x[1])) + "\n")
            for line in result.reasoning:
                f.write(f"  {line}\n")
            f.write("\n")

    @staticmethod
    def log_recommendation(action):
        with open(DECISION_LOG, "a") as f:
            f.write(f"#### [{DecisionLog._ts()}] Recommendation\n")
            f.write(f"Action: **{action.action_type.upper()}**")
            if action.position:
                f.write(f" #{action.position}")
            if action.ability_name:
                f.write(f" ({action.ability_name})")
            if action.targets:
                f.write(f" -> targets {['#'+str(t) for t in action.targets]}")
            f.write(f"\nReason: {action.reasoning}\n")
            for w in action.warnings:
                f.write(f"WARNING: {w}\n")
            f.write("\n")

    @staticmethod
    def log_execution(pos: int, was_evil, evil_role):
        with open(DECISION_LOG, "a") as f:
            f.write(f"### [{DecisionLog._ts()}] Executed #{pos}")
            if evil_role:
                f.write(f" -> {evil_role} (EVIL)")
            elif was_evil is True:
                f.write(f" -> EVIL")
            elif was_evil is False:
                f.write(f" -> GOOD (WRONG!)")
            f.write("\n\n")

    @staticmethod
    def log_ability_used(pos: int):
        with open(DECISION_LOG, "a") as f:
            f.write(f"### [{DecisionLog._ts()}] Ability used at #{pos}\n\n")

    @staticmethod
    def log_slayer_result(slayer_pos: int, target_pos: int, killed: bool,
                          revealed_role: Optional[str] = None):
        with open(DECISION_LOG, "a") as f:
            outcome = "killed" if killed else "could not kill"
            role = f" -> {revealed_role}" if revealed_role else ""
            f.write(
                f"### [{DecisionLog._ts()}] Slayer #{slayer_pos} {outcome} "
                f"#{target_pos}{role}\n\n"
            )

    @staticmethod
    def log_game_over(result: str, hp: int, notes: str = ""):
        """Log game outcome: 'win' or 'loss'."""
        with open(DECISION_LOG, "a") as f:
            f.write(f"## [{DecisionLog._ts()}] GAME OVER — {result.upper()}\n")
            f.write(f"Final HP: {hp}\n")
            if notes:
                f.write(f"Notes: {notes}\n")
            f.write("\n")

    @staticmethod
    def log_custom(label: str, text: str):
        """For Claude to log its own reasoning."""
        with open(DECISION_LOG, "a") as f:
            f.write(f"#### [{DecisionLog._ts()}] {label}\n")
            f.write(f"{text}\n\n")


# ============================================================
# GameSession
# ============================================================

def _prepare_current_jester_session_capture(
    session,
    card: CardInfo,
    existing: Optional[CardInfo],
    existing_role_key: Optional[str],
) -> Optional[dict]:
    """Validate and join one current Jester card without mutating the session."""
    if (
        card.apparent_role.lower().replace(" ", "_") != "jester"
        or not isinstance(card.info_parsed, dict)
        or card.info_parsed.get("jester_variant") != _PUBLIC_CURRENT_VARIANT
    ):
        return None

    existing_current = (
        existing is not None
        and existing_role_key == "jester"
        and isinstance(existing.info_parsed, dict)
        and existing.info_parsed.get("jester_variant") == _PUBLIC_CURRENT_VARIANT
    )
    result = {
        "noop": False,
        "event_observed": False,
        "raw_capture": False,
        "generation_to_store": None,
        "consume_pending": False,
        "new_rambler_records": [],
    }
    raw_value = getattr(card, "_jester_raw_callbacks", None)
    has_explicit_ledger = (
        card.info_parsed.get("callback_ledger_variant") is not None
        or "callback_events" in card.info_parsed
    )
    marker_only = set(card.info_parsed) == {"jester_variant"}

    if raw_value is None:
        if has_explicit_ledger:
            incoming = _jester_callback_ledger(
                card.info_parsed,
                actor_position=card.position,
                n_cards=session.n_cards,
                reveal_order=session.reveal_order,
                baker_rule_version=session.baker_rule_version,
            )
            if not existing_current:
                raise ValueError(
                    "An ordered Jester ledger requires authenticated raw "
                    "callback provenance"
                )
            persisted = _jester_callback_ledger(
                existing.info_parsed,
                actor_position=card.position,
                n_cards=session.n_cards,
                reveal_order=session.reveal_order,
                baker_rule_version=session.baker_rule_version,
            )
            _validate_jester_rambler_sync(
                persisted,
                speaker_position=card.position,
                rambler_observations=session.rambler_shut_up_observations,
            )
            if incoming != persisted:
                raise ValueError(
                    "An ordered Jester ledger cannot be replaced without "
                    "authenticated appended callbacks"
                )
            result["noop"] = True
            return result
        if marker_only:
            if existing_current and existing.info_parsed.get(
                "callback_ledger_variant"
            ) == _ORDERED_CALLBACK_LEDGER_VARIANT:
                result["noop"] = True
            return result

        scalar = _jester_scalar_callback(card.info_parsed, n_cards=session.n_cards)
        if card.info_text != scalar["text"]:
            raise ValueError("Current scalar Jester text must exactly match its payload")
        if existing_current and existing.info_parsed.get(
            "callback_ledger_variant"
        ) == _ORDERED_CALLBACK_LEDGER_VARIANT:
            raise ValueError(
                "A strict Jester callback ledger cannot be replaced by "
                "non-resumable scalar compatibility evidence"
            )
        result["event_observed"] = True
        return result

    result["raw_capture"] = True
    if session.baker_rule_version != BAKER_RULE_VERSION:
        raise ValueError(
            "Current Jester callback capture requires verified "
            "reveal_order/Baker provenance"
        )
    if (
        not session.reveal_order
        or len(session.reveal_order) > session.n_cards
        or len(set(session.reveal_order)) != len(session.reveal_order)
        or any(
            type(position) is not int
            or not 1 <= position <= session.n_cards
            for position in session.reveal_order
        )
        or card.position not in session.reveal_order
    ):
        raise ValueError("Current Jester actor must be in the verified reveal prefix")

    raw_callbacks = _validate_raw_jester_callbacks(
        raw_value,
        n_cards=session.n_cards,
    )
    prior_events: list[dict] = []
    if existing_current:
        if existing.info_parsed.get(
            "callback_ledger_variant"
        ) == _ORDERED_CALLBACK_LEDGER_VARIANT:
            prior_events = _jester_callback_ledger(
                existing.info_parsed,
                actor_position=card.position,
                n_cards=session.n_cards,
                reveal_order=session.reveal_order,
                baker_rule_version=session.baker_rule_version,
            )
        elif set(existing.info_parsed) != {"jester_variant"}:
            raise ValueError(
                "Scalar current Jester evidence cannot join an ordered callback ledger"
            )
    elif existing is not None and (
        existing_role_key != "jester"
        or bool(existing.info_text)
        or bool(existing.info_parsed)
    ):
        raise ValueError(
            "Authenticated current Jester callbacks may replace only an empty "
            "same-role placeholder"
        )

    prior_signatures = [_jester_callback_signature(event) for event in prior_events]
    raw_signatures = [_jester_callback_signature(event) for event in raw_callbacks]
    if (
        len(raw_signatures) < len(prior_signatures)
        or raw_signatures[:len(prior_signatures)] != prior_signatures
    ):
        raise ValueError(
            "Raw Jester callback history does not preserve the persisted prefix"
        )
    if raw_signatures == prior_signatures:
        _validate_jester_rambler_sync(
            prior_events,
            speaker_position=card.position,
            rambler_observations=session.rambler_shut_up_observations,
        )
        if not prior_events and not existing_current:
            # A readable empty/passive history is an authenticated marker for
            # a newly observed Jester, not an event ledger.
            delattr(card, "_jester_raw_callbacks")
            result["raw_capture"] = False
            return result
        result["noop"] = True
        return result

    suffix = raw_callbacks[len(prior_events):]
    if len(suffix) > 2:
        raise ValueError(
            "New Jester callback suffix cannot be grouped into one provable activation"
        )
    has_generation = card.position in session.jester_reset_generations
    if prior_events and not has_generation:
        raise ValueError(
            "Persisted Jester callback history has no matching session "
            "reset-generation provenance"
        )
    session_generation = session.jester_reset_generations.get(
        card.position,
        session.lilis_nights_resolved,
    )
    if prior_events and prior_events[-1]["reset_generation"] > session_generation:
        raise ValueError("Persisted Jester reset generation exceeds session history")
    next_activation_id = prior_events[-1]["activation_id"] + 1 if prior_events else 1
    pending = session.jester_pending_activations.get(card.position)
    if pending is not None:
        token = _validate_jester_pending_token(
            pending,
            actor_position=card.position,
            n_cards=session.n_cards,
            reveal_order=session.reveal_order,
            reset_generation=session_generation,
            prior_callback_count=len(prior_events),
            next_activation_id=next_activation_id,
        )
        if any(
            callback["event_kind"] == "jester_result"
            for callback in suffix
        ):
            suffix = _bind_jester_physical_targets(
                suffix,
                token["expected_targets"],
                n_cards=session.n_cards,
            )
        new_group = _stamp_jester_callback_group(
            suffix,
            activation_id=next_activation_id,
            activation_evidence="auto_use_click",
            reset_generation=session_generation,
            settled_reveal_count=token["settled_reveal_count"],
        )
        merged_events = prior_events + new_group
        result["consume_pending"] = True
    else:
        final_group = (
            [
                event for event in prior_events
                if event["activation_id"] == prior_events[-1]["activation_id"]
            ]
            if prior_events else []
        )
        delayed_extension = (
            len(suffix) == 1
            and len(final_group) == 1
            and final_group[0]["dispatch_path"] == "either"
            and final_group[0]["reset_generation"] == session_generation
            and card.position in session.used_abilities
        )
        if delayed_extension:
            group_start = len(prior_events) - 1
            extended_raw = raw_callbacks[group_start:]
            if len(extended_raw) != 2:
                raise ValueError(
                    "Delayed Jester callback extension is not exactly the "
                    "persisted event plus one appended callback"
                )
            extended_raw = _bind_jester_physical_targets(
                extended_raw,
                final_group[0].get("targets"),
                n_cards=session.n_cards,
            )
            new_group = _stamp_jester_callback_group(
                extended_raw,
                activation_id=final_group[0]["activation_id"],
                activation_evidence="same_activation_extension",
                reset_generation=session_generation,
                settled_reveal_count=final_group[0]["settled_reveal_count"],
            )
            merged_events = prior_events[:-1] + new_group
        else:
            last_generation = (
                prior_events[-1]["reset_generation"] if prior_events else -1
            )
            if session_generation <= last_generation:
                raise ValueError(
                    "New Jester callback suffix has no unconsumed reset generation"
                )
            if not prior_events and len(raw_callbacks) != 1:
                raise ValueError(
                    "An initial Jester ledger attachment must contain exactly "
                    "one raw callback"
                )
            generation_gap = session_generation - last_generation
            if len(suffix) == 2 and generation_gap > 1:
                raise ValueError(
                    "A two-callback Jester suffix after skipped reset generations "
                    "is ambiguous"
                )
            if not prior_events and session_generation == 0:
                if len(suffix) != 1:
                    raise ValueError(
                        "Initial Jester history cannot prove a multi-callback activation"
                    )
                evidence = "single_callback_suffix"
            else:
                evidence = "session_reset_generation"
            suffix = _infer_jester_physical_targets(
                suffix,
                n_cards=session.n_cards,
            )
            new_group = _stamp_jester_callback_group(
                suffix,
                activation_id=next_activation_id,
                activation_evidence=evidence,
                reset_generation=session_generation,
                settled_reveal_count=len(session.reveal_order),
            )
            merged_events = prior_events + new_group

    _apply_jester_callback_ledger(card, merged_events)
    validated_events = _jester_callback_ledger(
        card.info_parsed,
        actor_position=card.position,
        n_cards=session.n_cards,
        reveal_order=session.reveal_order,
        baker_rule_version=session.baker_rule_version,
    )
    _validate_jester_rambler_sync(
        prior_events,
        speaker_position=card.position,
        rambler_observations=session.rambler_shut_up_observations,
    )
    result["new_rambler_records"] = _jester_interruption_records(
        validated_events[len(prior_events):],
        speaker_position=card.position,
    )
    result["generation_to_store"] = session_generation
    result["event_observed"] = True
    delattr(card, "_jester_raw_callbacks")
    return result


class GameSession:
    def __init__(self, n_cards: int, n_evil: int):
        self.n_cards = n_cards
        self.n_evil = n_evil
        self.villagers: list[str] = []
        self.outcasts: list[str] = []
        self.minions: list[str] = []
        self.demons: list[str] = []
        self.cards: list[CardInfo] = []
        self.executed: list[int] = []
        self.confirmed_evil: list[int] = []
        self.confirmed_good: list[int] = []
        self.pd_corruption_target: Optional[int] = None
        self.used_abilities: list[int] = []
        self.executed_evil_roles: dict[int, str] = {}  # pos -> evil role name
        self.slayer_results: list[dict] = []  # [{slayer_pos, target_pos, killed, revealed_role?}]
        self.night_kills: list[int] = []  # Positions killed by Lilis night
        self.night_kill_evil_count: int = 0  # How many night kills were evil
        self.hp: int = 10
        self.wrong_exec_cost: int = 5  # Asc4+ default (Drunk=2, Lilis=2 are exceptions)
        self.pd_ability_results: list[dict] = []  # [{"pd_pos": N, "target": N, "is_corrupted": bool, "evil_revealed": N|None}]
        self.blocked_positions: list[int] = []  # Positions blocked from reveal (e.g. Witch)
        self.executed_good_corrupted: dict[int, bool] = {}  # pos -> was corrupted (from execution observation)
        self.executed_good_roles: dict[int, str] = {}  # pos -> public current role after a wrong execution
        self.board_villager_count: Optional[int] = None  # Normalized pre-Start header V count
        self.board_outcast_count: Optional[int] = None   # Normalized pre-Start header O count
        self.board_count_provenance: str = "legacy_unknown"
        self.rambler_rule_version: Optional[str] = RAMBLER_RULE_VERSION
        self.rambler_shut_up_observations: list[dict] = []
        self.baker_rule_version: Optional[str] = BAKER_RULE_VERSION
        self.doppel_drunk_rule_version: Optional[str] = DOPPEL_DRUNK_RULE_VERSION
        self.fortune_teller_rule_version: Optional[str] = FORTUNE_TELLER_RULE_VERSION
        # Session-only provenance for ResetAfterNight Druid activations. The
        # generation advances once per completed Night while the actor is
        # known, even if its prior use bit was already clear. Pending tokens
        # are persisted before an automated click so a crash cannot orphan the
        # strongest available grouping evidence.
        self.druid_reset_generations: dict[int, int] = {}
        self.druid_pending_activations: dict[int, dict] = {}
        self.jester_reset_generations: dict[int, int] = {}
        self.jester_pending_activations: dict[int, dict] = {}
        self.terminal_loss_role: Optional[str] = None
        self.executed_current_roles: dict[int, str] = {}
        self.revealed_night_current_roles: dict[int, str] = {}
        self.reveal_order: list[int] = []  # Order positions were flipped (for Baker)
        self.lilis_batch_index: int = 0  # Explicit Lilis batch counter (don't derive from reveal_order)
        # Trigger/result synchronization is live-session bookkeeping only.
        # Historical solver fixtures do not retain enough timing state to
        # reconstruct no-kill outcomes safely.
        self.lilis_nights_resolved: int = 0
        # Authoritative live-only work queue. Unlike batch/resolved history,
        # this is never reconstructed from legacy saves or final board state.
        self.pending_lilis_nights: int = 0

        # Clear solver cache on new game
        try:
            from rust_solver import clear_solver_cache
            clear_solver_cache()
        except ImportError:
            pass

    # -- Deck --

    def has_role_in_deck(self, role_name: str) -> bool:
        """Check if a role (by normalized name) is in any deck faction."""
        norm = _normalize_role_name(role_name)
        return any(
            _normalize_role_name(v) == norm
            for faction in [self.villagers, self.outcasts,
                            self.minions, self.demons]
            for v in faction
        )

    def current_identity_may_have_moved(self) -> bool:
        """Whether shipped Start effects can separate data from origin."""
        return any(
            self.has_role_in_deck(role)
            for role in ("Chancellor", "Twin Minion", "Shaman")
        )

    def twin_live_solver_unsafe(self) -> bool:
        """Whether live solving lacks the ordered current-data trace.

        Twin can move Minion data onto a runtime-Good body before delayed
        Reveal assigns a bluff. That apparent-Good clue then lies, while the
        pre-trace solver treats the physical Good seat as truthful. Until the
        ordered Twin permutation is modeled, every authored Twin game must
        stop before Rust or strategy can produce a live action.
        """
        return self.has_role_in_deck("Twin Minion")

    def _death_current_role(
        self,
        current_role: Optional[str],
        original_evil_role: Optional[str] = None,
    ) -> Optional[str]:
        """Resolve death identity without treating a moved origin as current."""
        if _is_known_role(current_role):
            return current_role
        if (
            not self.current_identity_may_have_moved()
            and _is_known_role(original_evil_role)
        ):
            return original_evil_role
        return None

    def _known_dead_current_roles(self) -> dict[int, str]:
        """Collect exact public current-role death evidence by physical seat."""
        roles = dict(self.executed_current_roles)
        for position, role in self.executed_good_roles.items():
            roles.setdefault(position, role)
        for result in self.slayer_results:
            if not result.get("killed"):
                continue
            role = slayer_revealed_role(result)
            if role:
                roles.setdefault(result["target_pos"], role)
        for position, role in self.revealed_night_current_roles.items():
            roles.setdefault(position, role)
        if not self.current_identity_may_have_moved():
            for position, role in self.executed_evil_roles.items():
                roles.setdefault(position, role)
        return roles

    def _apply_current_death_hooks(
        self,
        pos: int,
        current_role: Optional[str],
        original_evil_role: Optional[str] = None,
        *,
        allow_terminal: bool = True,
    ) -> Optional[str]:
        """Apply native role-on-death bookkeeping from current data only."""
        role = self._death_current_role(current_role, original_evil_role)
        if role is None:
            return None
        if allow_terminal:
            self.mark_terminal_loss(role)
        role_key = _execution_role_key(role)
        if role_key == "witch":
            self.release_witch_blocks(f"current-role Witch death at #{pos}")
        if role_key == "baa":
            _baa_post_death_deck_refresh(self)
        return role

    def full_reset(self):
        """Clear ALL mutable state for between-game isolation.

        Call this between games in batch mode to prevent state leaks.
        Clears: cards, executed, confirmed, abilities, night kills, blocked,
                reveal order, HP, deck, solver cache, Rust daemon.
        """
        self.cards.clear()
        self.executed.clear()
        self.confirmed_evil.clear()
        self.confirmed_good.clear()
        self.pd_corruption_target = None
        self.used_abilities.clear()
        self.executed_evil_roles.clear()
        self.slayer_results.clear()
        self.night_kills.clear()
        self.night_kill_evil_count = 0
        self.hp = 10
        self.wrong_exec_cost = 5
        self.pd_ability_results.clear()
        self.blocked_positions.clear()
        self.executed_good_corrupted.clear()
        self.executed_good_roles.clear()
        self.board_villager_count = None
        self.board_outcast_count = None
        self.board_count_provenance = "legacy_unknown"
        self.rambler_rule_version = RAMBLER_RULE_VERSION
        self.rambler_shut_up_observations.clear()
        self.baker_rule_version = BAKER_RULE_VERSION
        self.doppel_drunk_rule_version = DOPPEL_DRUNK_RULE_VERSION
        self.fortune_teller_rule_version = FORTUNE_TELLER_RULE_VERSION
        self.druid_reset_generations.clear()
        self.druid_pending_activations.clear()
        self.jester_reset_generations.clear()
        self.jester_pending_activations.clear()
        self.terminal_loss_role = None
        self.executed_current_roles.clear()
        self.revealed_night_current_roles.clear()
        self.reveal_order.clear()
        self.lilis_batch_index = 0
        self.lilis_nights_resolved = 0
        self.pending_lilis_nights = 0
        self.villagers.clear()
        self.outcasts.clear()
        self.minions.clear()
        self.demons.clear()

        # Clear solver cache
        try:
            from rust_solver import clear_solver_cache, shutdown_daemon
            clear_solver_cache()
            shutdown_daemon()
        except ImportError:
            pass

        print("[full_reset] All session state cleared, solver cache + daemon reset")

    def lilis_deck_count(self) -> int:
        """Return the number of authored Lilis records in the public deck."""
        return sum(
            _normalize_role_name(role) == "Lilis"
            for faction in [self.villagers, self.outcasts,
                            self.minions, self.demons]
            for role in faction
        )

    def has_lilis_night_rule(self) -> bool:
        """Whether the authored roster installs the persistent Night rule."""
        return self.lilis_deck_count() > 0

    def has_duplicate_lilis(self) -> bool:
        """Whether authored live Night effects exceed the one-actor model."""
        return self.lilis_deck_count() > 1

    def lilis_actor_state(self) -> str:
        """Return ``active``, ``inactive``, or ``unknown`` from public facts.

        The NightModeRule persists after every Lilis dies, so callers deciding
        reveal batching must use ``has_lilis_night_rule`` instead. Chancellor
        and Twin swap current data without changing the number of Lilis data
        records. Shaman can erase or duplicate the current Lilis actor before
        Lilis Start, so this checkpoint leaves its actor state unknown and
        hard-pauses live reveal/Night automation for that deck combination.
        """
        deck_count = self.lilis_deck_count()
        if deck_count == 0:
            return "inactive"
        if self.has_role_in_deck("Shaman"):
            return "unknown"
        known_dead = sum(
            _normalize_role_name(role) == "Lilis"
            for role in self._known_dead_current_roles().values()
        )
        if known_dead >= deck_count:
            return "inactive"
        return "active"

    def is_lilis_alive(self) -> Optional[bool]:
        """Compatibility view of :meth:`lilis_actor_state`.

        ``None`` is intentional: treating unknown as active would invent 2 HP
        damage, while treating it as inactive would skip a real Lilis action.
        """
        state = self.lilis_actor_state()
        if state == "unknown":
            return None
        return state == "active"

    def schedule_lilis_night(self) -> None:
        """Atomically add one verified every-four-reveals Night transition."""
        if not self.has_lilis_night_rule():
            raise ValueError("no Lilis Night rule exists in this deck")
        if self.has_duplicate_lilis():
            raise ValueError(
                "duplicate Lilis live nights are unsupported: multiple actors "
                "can charge HP while colliding on one delayed victim"
            )
        self.lilis_batch_index += 1
        self.pending_lilis_nights += 1

    def is_witch_known_dead(self) -> bool:
        """Whether any known Witch death released the ordinary shared quota."""
        return self.has_role_in_deck("Witch") and any(
            _normalize_role_name(role) == "Witch"
            for role in self._known_dead_current_roles().values()
        )

    def release_witch_blocks(self, reason: str) -> list[int]:
        """Drop current block markers after a death may have released the quota.

        Cipher owns a global hidden-card quota, not a status on a particular
        character. Clearing these markers never reveals a card or asserts the
        hidden Witch's identity; it only permits a verified public click probe.
        """
        released = list(dict.fromkeys(self.blocked_positions))
        if released:
            self.blocked_positions.clear()
            print(
                "  [Witch] Released current block marker(s) "
                f"{['#' + str(position) for position in released]} ({reason}); "
                "the card still needs a verified reveal click."
            )
        return released

    def set_deck(self, villagers: list[str], outcasts: list[str],
                 minions: list[str], demons: list[str]):
        self.villagers = villagers
        self.outcasts = outcasts
        self.minions = minions
        self.demons = demons

    # -- Cards --

    def add_card(
        self,
        card: CardInfo,
        *,
        mark_active_result: bool = True,
    ):
        role_key = card.apparent_role.lower().replace(" ", "_")
        variant_field = {
            "druid": "druid_variant",
            "jester": "jester_variant",
        }.get(role_key)
        if (
            variant_field is not None
            and isinstance(card.info_parsed, dict)
            and card.info_parsed.get(variant_field) == _PUBLIC_CURRENT_VARIANT
        ):
            # Normalize current callback evidence on a private copy. A later
            # chronology/reveal-boundary rejection must leave the caller's
            # CardInfo byte-for-byte unchanged for recovery and diagnostics.
            source_card = card
            card = CardInfo(
                source_card.position,
                source_card.apparent_role,
                source_card.info_text,
                copy.deepcopy(source_card.info_parsed),
            )
            for attribute in ("_druid_raw_callbacks", "_jester_raw_callbacks"):
                if hasattr(source_card, attribute):
                    setattr(
                        card,
                        attribute,
                        copy.deepcopy(getattr(source_card, attribute)),
                    )
        existing = next(
            (previous for previous in self.cards if previous.position == card.position),
            None,
        )
        existing_role_key = (
            existing.apparent_role.lower().replace(" ", "_")
            if existing is not None else None
        )

        medium_target_placeholder = None
        if role_key == "medium":
            medium_target = card.info_parsed.get("good_position")
            medium_role = card.info_parsed.get("good_role")
            if (
                medium_target
                and medium_role
                and medium_target in self.night_kills
            ):
                target_entry = next(
                    (
                        previous
                        for previous in self.cards
                        if previous.position == medium_target
                    ),
                    None,
                )
                if target_entry is not None:
                    target_role_key = _execution_role_key(
                        target_entry.apparent_role
                    )
                    medium_role_key = _execution_role_key(medium_role)
                    placeholder = (
                        target_role_key in {"", "?", "unknown", "none", "no info"}
                        and not target_entry.info_text
                        and not target_entry.info_parsed
                    )
                    if placeholder:
                        medium_target_placeholder = target_entry
                    elif target_role_key != medium_role_key:
                        raise ValueError(
                            "Medium public death-role reveal conflicts with "
                            f"existing #{medium_target} CardInfo "
                            f"({target_entry.apparent_role!r} != {medium_role!r}); "
                            "refusing contradictory solver input"
                        )

        incoming_shut_up_target = card.info_parsed.get("shut_up_target")
        if "shut_up_target" in card.info_parsed:
            if type(incoming_shut_up_target) is not int:
                raise ValueError("Rambler shut-up target must be an integer")
            if not 1 <= incoming_shut_up_target <= self.n_cards:
                raise ValueError(
                    f"Rambler shut-up target #{incoming_shut_up_target} "
                    f"is outside 1..{self.n_cards}"
                )

        # Current Fortune Teller evidence is exact and chronological. Promote
        # manual scalar input to that shape before mutating any session state;
        # archived unmarked sessions retain their historical scalar behavior.
        current_fortune_history: list[dict] = []
        prior_fortune_history: list[dict] = []
        current_fortune_rules = (
            self.fortune_teller_rule_version
            == FORTUNE_TELLER_RULE_VERSION
        )
        if role_key == "fortune_teller" and current_fortune_rules:
            current_fortune_history = _prepare_current_fortune_teller_card(
                card,
                n_cards=self.n_cards,
            )
            if existing is not None and existing_role_key == "fortune_teller":
                prior_fortune_history = _fortune_teller_observation_history(
                    existing.info_parsed,
                    n_cards=self.n_cards,
                    strict_native=True,
                )

        jester_capture = _prepare_current_jester_session_capture(
            self,
            card,
            existing,
            existing_role_key,
        )
        if jester_capture is not None and jester_capture["noop"]:
            return
        jester_event_observed = bool(
            jester_capture and jester_capture["event_observed"]
        )
        jester_raw_capture = bool(
            jester_capture and jester_capture["raw_capture"]
        )

        # Current Druid raw callbacks are reconciled against an immutable
        # persisted prefix before any session field is changed. Manual scalar
        # input remains explicit compatibility evidence and is intentionally
        # non-resumable: it cannot certify an old reveal boundary or native
        # real/raw grouping.
        current_druid_rules = (
            role_key == "druid"
            and isinstance(card.info_parsed, dict)
            and card.info_parsed.get("druid_variant")
            == _PUBLIC_CURRENT_VARIANT
        )
        druid_event_observed = False
        reset_druid_event = False
        druid_raw_capture = False
        druid_generation_to_store = None
        druid_consume_pending = False
        druid_new_rambler_records: list[dict] = []
        if current_druid_rules:
            existing_current_druid = (
                existing is not None
                and existing_role_key == "druid"
                and isinstance(existing.info_parsed, dict)
                and existing.info_parsed.get("druid_variant")
                == _PUBLIC_CURRENT_VARIANT
            )
            raw_callbacks_value = getattr(card, "_druid_raw_callbacks", None)
            has_explicit_ledger = (
                card.info_parsed.get("callback_ledger_variant")
                is not None
                or "callback_events" in card.info_parsed
            )
            if raw_callbacks_value is None:
                if has_explicit_ledger:
                    incoming_events = _druid_callback_ledger(
                        card.info_parsed,
                        actor_position=card.position,
                        n_cards=self.n_cards,
                        reveal_order=self.reveal_order,
                        baker_rule_version=self.baker_rule_version,
                    )
                    if not existing_current_druid:
                        raise ValueError(
                            "An ordered Druid ledger requires authenticated raw "
                            "callback provenance"
                        )
                    persisted_events = _druid_callback_ledger(
                        existing.info_parsed,
                        actor_position=card.position,
                        n_cards=self.n_cards,
                        reveal_order=self.reveal_order,
                        baker_rule_version=self.baker_rule_version,
                    )
                    _validate_druid_rambler_sync(
                        persisted_events,
                        speaker_position=card.position,
                        rambler_observations=(
                            self.rambler_shut_up_observations
                        ),
                    )
                    if incoming_events != persisted_events:
                        raise ValueError(
                            "An ordered Druid ledger cannot be replaced without "
                            "authenticated appended callbacks"
                        )
                    return

                scalar_callback = _druid_scalar_callback(
                    card.info_parsed,
                    n_cards=self.n_cards,
                )
                if card.info_text != scalar_callback["text"]:
                    raise ValueError(
                        "Current scalar Druid text must exactly match its payload"
                    )
                if (
                    existing_current_druid
                    and existing.info_parsed.get("callback_ledger_variant")
                    == _ORDERED_CALLBACK_LEDGER_VARIANT
                ):
                    raise ValueError(
                        "A strict Druid callback ledger cannot be replaced by "
                        "non-resumable scalar compatibility evidence"
                    )
                druid_event_observed = True
            else:
                druid_raw_capture = True
                if self.baker_rule_version != BAKER_RULE_VERSION:
                    raise ValueError(
                        "Current Druid callback capture requires verified "
                        "reveal_order/Baker provenance"
                    )
                if (
                    not self.reveal_order
                    or len(self.reveal_order) > self.n_cards
                    or len(set(self.reveal_order)) != len(self.reveal_order)
                    or any(
                        type(position) is not int
                        or not 1 <= position <= self.n_cards
                        for position in self.reveal_order
                    )
                    or card.position not in self.reveal_order
                ):
                    raise ValueError(
                        "Current Druid actor must be in the verified reveal prefix"
                    )

                raw_callbacks = _validate_raw_druid_callbacks(
                    raw_callbacks_value,
                    n_cards=self.n_cards,
                )
                prior_events: list[dict] = []
                if existing_current_druid:
                    prior_events = _druid_callback_ledger(
                        existing.info_parsed,
                        actor_position=card.position,
                        n_cards=self.n_cards,
                        reveal_order=self.reveal_order,
                        baker_rule_version=self.baker_rule_version,
                    )
                elif existing is not None and (
                    existing_role_key != "druid"
                    or bool(existing.info_text)
                    or bool(existing.info_parsed)
                ):
                    raise ValueError(
                        "Authenticated current Druid callbacks may replace only "
                        "an empty same-role placeholder"
                    )

                prior_signatures = [
                    _druid_callback_signature(event) for event in prior_events
                ]
                raw_signatures = [
                    _druid_callback_signature(event) for event in raw_callbacks
                ]
                if (
                    len(raw_signatures) < len(prior_signatures)
                    or raw_signatures[:len(prior_signatures)] != prior_signatures
                ):
                    raise ValueError(
                        "Raw Druid callback history does not preserve the "
                        "persisted prefix"
                    )
                if raw_signatures == prior_signatures:
                    _validate_druid_rambler_sync(
                        prior_events,
                        speaker_position=card.position,
                        rambler_observations=(
                            self.rambler_shut_up_observations
                        ),
                    )
                    return
                suffix = raw_callbacks[len(prior_events):]
                if len(suffix) > 2:
                    raise ValueError(
                        "New Druid callback suffix cannot be grouped into one "
                        "provable activation"
                    )

                has_generation = card.position in self.druid_reset_generations
                if prior_events and not has_generation:
                    raise ValueError(
                        "Persisted Druid callback history has no matching "
                        "session reset-generation provenance"
                    )
                session_generation = self.druid_reset_generations.get(
                    card.position,
                    self.lilis_nights_resolved,
                )
                if prior_events and (
                    prior_events[-1]["reset_generation"] > session_generation
                ):
                    raise ValueError(
                        "Persisted Druid reset generation exceeds session history"
                    )
                next_activation_id = (
                    prior_events[-1]["activation_id"] + 1
                    if prior_events else 1
                )
                pending = self.druid_pending_activations.get(card.position)
                new_group: list[dict]
                if pending is not None:
                    token = _validate_druid_pending_token(
                        pending,
                        actor_position=card.position,
                        n_cards=self.n_cards,
                        reveal_order=self.reveal_order,
                        reset_generation=session_generation,
                        prior_callback_count=len(prior_events),
                        next_activation_id=next_activation_id,
                    )
                    if len(suffix) > 2:
                        raise ValueError(
                            "Automated Druid activation emitted too many callbacks"
                        )
                    if any(
                        callback["event_kind"] == "druid_result"
                        and callback["targets"] != token["expected_targets"]
                        for callback in suffix
                    ):
                        raise ValueError(
                            "Druid raw callback targets disagree with the "
                            "persisted auto-use click token"
                        )
                    new_group = _stamp_druid_callback_group(
                        suffix,
                        activation_id=next_activation_id,
                        activation_evidence="auto_use_click",
                        reset_generation=session_generation,
                        settled_reveal_count=token["settled_reveal_count"],
                    )
                    merged_events = prior_events + new_group
                    druid_consume_pending = True
                else:
                    final_group = (
                        [
                            event for event in prior_events
                            if event["activation_id"]
                            == prior_events[-1]["activation_id"]
                        ]
                        if prior_events else []
                    )
                    delayed_extension = (
                        len(suffix) == 1
                        and len(final_group) == 1
                        and final_group[0]["dispatch_path"] == "either"
                        and final_group[0]["reset_generation"]
                        == session_generation
                        and card.position in self.used_abilities
                    )
                    if delayed_extension:
                        group_start = len(prior_events) - 1
                        extended_raw = raw_callbacks[group_start:]
                        if len(extended_raw) != 2:
                            raise ValueError(
                                "Delayed Druid callback extension is not exactly "
                                "the persisted event plus one appended callback"
                            )
                        new_group = _stamp_druid_callback_group(
                            extended_raw,
                            activation_id=final_group[0]["activation_id"],
                            activation_evidence="same_activation_extension",
                            reset_generation=session_generation,
                            settled_reveal_count=final_group[0][
                                "settled_reveal_count"
                            ],
                        )
                        merged_events = prior_events[:-1] + new_group
                    else:
                        last_generation = (
                            prior_events[-1]["reset_generation"]
                            if prior_events else -1
                        )
                        if session_generation <= last_generation:
                            raise ValueError(
                                "New Druid callback suffix has no unconsumed "
                                "reset generation"
                            )
                        if not prior_events and len(raw_callbacks) != 1:
                            raise ValueError(
                                "An initial Druid ledger attachment must contain "
                                "exactly one raw callback"
                            )
                        generation_gap = session_generation - last_generation
                        if len(suffix) == 2 and generation_gap > 1:
                            raise ValueError(
                                "A two-callback Druid suffix after skipped reset "
                                "generations is ambiguous"
                            )
                        if not prior_events and session_generation == 0:
                            if len(suffix) != 1:
                                raise ValueError(
                                    "Initial Druid history cannot prove a "
                                    "multi-callback activation"
                                )
                            activation_evidence = "single_callback_suffix"
                        else:
                            activation_evidence = "session_reset_generation"
                        new_group = _stamp_druid_callback_group(
                            suffix,
                            activation_id=next_activation_id,
                            activation_evidence=activation_evidence,
                            reset_generation=session_generation,
                            settled_reveal_count=len(self.reveal_order),
                        )
                        merged_events = prior_events + new_group

                _apply_druid_callback_ledger(card, merged_events)
                validated_events = _druid_callback_ledger(
                    card.info_parsed,
                    actor_position=card.position,
                    n_cards=self.n_cards,
                    reveal_order=self.reveal_order,
                    baker_rule_version=self.baker_rule_version,
                )
                _validate_druid_rambler_sync(
                    prior_events,
                    speaker_position=card.position,
                    rambler_observations=self.rambler_shut_up_observations,
                )
                druid_new_rambler_records = _druid_interruption_records(
                    validated_events[len(prior_events):],
                    speaker_position=card.position,
                )
                druid_generation_to_store = session_generation
                druid_event_observed = True
                delattr(card, "_druid_raw_callbacks")
        # Validate Judge evidence before mutating reveal order or any session
        # list, so a malformed history is rejected atomically.
        current_judge_history: list[dict] = []
        prior_judge_history: list[dict] = []
        if role_key == "judge":
            current_judge_history = _judge_observation_history(
                card.info_parsed,
                n_cards=self.n_cards,
            )
            if existing is not None and existing_role_key == "judge":
                prior_judge_history = _judge_observation_history(
                    existing.info_parsed,
                    n_cards=self.n_cards,
                )

        # Compatibility fallback for manual/legacy flows that enter a card
        # without first memory-verifying its flip. This order is useful but no
        # longer authoritative for the current Baker model.
        if card.position not in self.reveal_order:
            self.reveal_order.append(card.position)
            # ``baker_rule_version`` certifies that the entire order came from
            # memory verification. Once compatibility entry order invents a
            # missing seat, later flips cannot upgrade it back to current.
            self.baker_rule_version = None
            # Warn if entry order doesn't match expected #1->#N sequence
            expected_next = len(self.reveal_order)  # 1st entry should be pos 1, 2nd pos 2, etc.
            if card.position != expected_next:
                # Check if it's just not sequential (e.g., entering #3 as 2nd card)
                print(f"  WARNING: Card #{card.position} entered as reveal #{len(self.reveal_order)}, "
                      f"but sequential order expects #{expected_next}.")
                print(f"  Current reveal_order: {self.reveal_order}")
                print(f"  If cards were flipped out of #1->#N order, this is correct.")
                print(f"  If this is a mistake, fix now — reveal_order affects Baker validation.")
        # Judge is ResetAfterNight. A same-round reread corrects the one
        # current event; only a post-Night use extends the chronological
        # history. Native memory may supply either just the newest result or
        # the full normal-result history, so merge both shapes deliberately.
        judge_event_observed = (
            role_key == "judge" and _has_active_clue_result(card)
        )
        same_judge_event = (
            judge_event_observed
            and existing is not None
            and existing_role_key == "judge"
            and card.position in self.used_abilities
        )
        reset_judge_event = (
            judge_event_observed
            and existing is not None
            and existing_role_key == "judge"
            and card.position not in self.used_abilities
        )
        fortune_event_observed = (
            current_fortune_rules
            and role_key == "fortune_teller"
            and _has_active_clue_result(card)
        )
        same_fortune_event = (
            fortune_event_observed
            and existing is not None
            and existing_role_key == "fortune_teller"
            and card.position in self.used_abilities
        )
        reset_fortune_event = (
            fortune_event_observed
            and existing is not None
            and existing_role_key == "fortune_teller"
            and card.position not in self.used_abilities
        )
        existing_shut_up_target = (
            existing.info_parsed.get("shut_up_target")
            if existing is not None else None
        )

        def merge_reset_history(
            older: list[dict],
            incoming: list[dict],
        ) -> list[dict]:
            if not incoming:
                return list(older)
            if (
                len(incoming) > len(older)
                and incoming[:len(older)] == older
            ):
                return list(incoming)
            return list(older) + [dict(incoming[-1])]

        if judge_event_observed and existing is not None and existing_role_key == "judge":
            incoming_is_shut_up = type(incoming_shut_up_target) is int
            existing_is_shut_up = type(existing_shut_up_target) is int

            if same_judge_event:
                # If the existing event was a normal Judge result, its last
                # observation is the current round and must be replaced. A
                # shut-up event has no normal observation, so all retained
                # entries are older rounds.
                older_rounds = (
                    prior_judge_history
                    if existing_is_shut_up
                    else prior_judge_history[:-1]
                )
                observations = (
                    list(older_rounds)
                    if incoming_is_shut_up
                    else merge_reset_history(
                        list(older_rounds),
                        current_judge_history,
                    )
                )
            elif reset_judge_event:
                observations = (
                    list(prior_judge_history)
                    if incoming_is_shut_up
                    else merge_reset_history(
                        prior_judge_history,
                        current_judge_history,
                    )
                )
            else:
                observations = list(current_judge_history)

            if len(observations) > 1 or (
                incoming_is_shut_up and observations
            ):
                card.info_parsed["observations"] = observations
            else:
                card.info_parsed.pop("observations", None)

        if (
            fortune_event_observed
            and existing is not None
            and existing_role_key == "fortune_teller"
        ):
            incoming_is_shut_up = type(incoming_shut_up_target) is int
            existing_is_shut_up = type(existing_shut_up_target) is int

            if same_fortune_event:
                older_rounds = (
                    prior_fortune_history
                    if existing_is_shut_up
                    else prior_fortune_history[:-1]
                )
                observations = (
                    list(older_rounds)
                    if incoming_is_shut_up
                    else merge_reset_history(
                        list(older_rounds),
                        current_fortune_history,
                    )
                )
            elif reset_fortune_event:
                observations = (
                    list(prior_fortune_history)
                    if incoming_is_shut_up
                    else merge_reset_history(
                        prior_fortune_history,
                        current_fortune_history,
                    )
                )
            else:
                observations = list(current_fortune_history)

            # Current Fortune Teller always carries an explicit normal-result
            # ledger, including an empty/older-only ledger when the newest use
            # was replaced by Rambler.
            card.info_parsed["observations"] = observations

        # The ledger is chronological public-event state, not an audit log of
        # parser corrections. Editing a non-reset event replaces/removes its
        # current record in place, preserving global event order. A later
        # ResetAfterNight Judge/Fortune Teller/Druid events append new records even
        # when the public result is identical.
        incoming_is_shut_up = type(incoming_shut_up_target) is int
        existing_is_shut_up = type(existing_shut_up_target) is int
        incoming_is_event = (
            (role_key != "druid" or not druid_raw_capture)
            and (role_key != "jester" or not jester_raw_capture)
            and (
            role_key not in {"druid", "fortune_teller", "jester", "judge"}
            or judge_event_observed
            or fortune_event_observed
            or druid_event_observed
            or jester_event_observed
            )
        )
        reset_reusable_event = (
            reset_judge_event
            or reset_fortune_event
            or reset_druid_event
        )

        if druid_generation_to_store is not None:
            self.druid_reset_generations[card.position] = (
                druid_generation_to_store
            )
        if druid_consume_pending:
            self.druid_pending_activations.pop(card.position, None)
        if druid_new_rambler_records:
            self.rambler_shut_up_observations.extend(
                druid_new_rambler_records
            )
        if jester_capture is not None:
            generation = jester_capture["generation_to_store"]
            if generation is not None:
                self.jester_reset_generations[card.position] = generation
            if jester_capture["consume_pending"]:
                self.jester_pending_activations.pop(card.position, None)
            if jester_capture["new_rambler_records"]:
                self.rambler_shut_up_observations.extend(
                    jester_capture["new_rambler_records"]
                )

        if incoming_is_event:
            new_record = (
                {
                    "speaker_position": card.position,
                    "shut_up_target": incoming_shut_up_target,
                }
                if incoming_is_shut_up else None
            )
            if existing is None or reset_reusable_event:
                if new_record is not None:
                    self.rambler_shut_up_observations.append(new_record)
            else:
                current_record_index = None
                if existing_is_shut_up:
                    for index in range(
                        len(self.rambler_shut_up_observations) - 1,
                        -1,
                        -1,
                    ):
                        observation = self.rambler_shut_up_observations[index]
                        if (
                            observation.get("speaker_position") == card.position
                            and observation.get("shut_up_target")
                            == existing_shut_up_target
                        ):
                            current_record_index = index
                            break
                if current_record_index is not None:
                    if new_record is None:
                        self.rambler_shut_up_observations.pop(
                            current_record_index
                        )
                    else:
                        self.rambler_shut_up_observations[
                            current_record_index
                        ] = new_record
                elif new_record is not None:
                    self.rambler_shut_up_observations.append(new_record)

        # Replace if same position already exists (re-read)
        self.cards = [c for c in self.cards if c.position != card.position]
        self.cards.append(card)
        self.cards.sort(key=lambda c: c.position)
        # Auto-mark active abilities used when a manual card entry contains
        # their real result. PD and Slayer keep dedicated result commands.
        active_result_roles = {
            "dreamer",
            "druid",
            "fortune_teller",
            "jester",
            "judge",
            "plague_doctor",
            "slayer",
        }
        if (
            mark_active_result
            and role_key in active_result_roles
            and _has_active_clue_result(card)
        ):
            self.mark_ability_used(card.position)
        # Medium reveals a dead card's role — auto-create card entry for
        # night-killed positions so the solver can track PD corruption etc.
        if card.apparent_role == "Medium":
            gp = card.info_parsed.get("good_position")
            gr = card.info_parsed.get("good_role")
            if gp and gr and gp in self.night_kills:
                existing = [c for c in self.cards if c.position == gp]
                if not existing:
                    dead_card = CardInfo(gp, gr, info_parsed={})
                    self.cards.append(dead_card)
                    self.cards.sort(key=lambda c: c.position)
                    print(f"  [auto] Created card entry for dead #{gp} ({gr}) from Medium info")
                elif medium_target_placeholder is not None:
                    medium_target_placeholder.apparent_role = _normalize_role_name(gr)
                    print(
                        f"  [auto] Updated dead #{gp} placeholder to the "
                        f"public Medium role {gr}"
                    )

                # Medium is a public current-role reveal for this hidden night
                # victim even when a placeholder CardInfo already exists. Run
                # death hooks only on the first coherent public observation so
                # repeated entry cannot duplicate Baa refresh output.
                canonical_current = _normalize_role_name(gr)
                previous_current = self.revealed_night_current_roles.get(gp)
                if previous_current is None:
                    self.revealed_night_current_roles[gp] = canonical_current
                    self._apply_current_death_hooks(
                        gp,
                        canonical_current,
                        allow_terminal=False,
                    )
                elif (
                    _execution_role_key(previous_current)
                    != _execution_role_key(canonical_current)
                ):
                    print(
                        "  [RECOVERY] Conflicting Medium role for hidden death "
                        f"#{gp}: kept {previous_current}, ignored {canonical_current}"
                    )

    def mark_executed(self, pos: int, was_evil: Optional[bool] = None,
                      evil_role: Optional[str] = None,
                      was_corrupted: Optional[bool] = None,
                      true_role: Optional[str] = None,
                      record_current_role: bool = True):
        """Record one execution with stable origin and current data separate.

        ``evil_role`` is the stable/original Evil assignment. ``true_role`` is
        the legacy parameter name for the publicly observed current
        CharacterData role.
        """
        if pos not in self.executed:
            self.executed.append(pos)
        self._apply_current_death_hooks(pos, true_role, evil_role)
        if (
            record_current_role
            and _is_known_role(true_role)
        ):
            self.executed_current_roles[pos] = true_role.replace(' ', '_')
        if was_evil is True and pos not in self.confirmed_evil:
            self.confirmed_evil.append(pos)
        elif was_evil is False and pos not in self.confirmed_good:
            self.confirmed_good.append(pos)
        if evil_role:
            self.executed_evil_roles[pos] = evil_role.replace(' ', '_')
        # Execution bookkeeping exposes Drunk as clean even when its active
        # Corrupted status drove role effects such as Knight's +4 damage.
        if was_evil is False and was_corrupted is not None:
            observed_corrupted = (
                False if _execution_role_key(true_role) == "drunk"
                else was_corrupted
            )
            self.executed_good_corrupted[pos] = observed_corrupted
        if was_evil is False and _is_known_role(true_role):
            self.executed_good_roles[pos] = true_role.replace(' ', '_')

    def mark_terminal_loss(self, current_role: Optional[str]) -> bool:
        """Persist a public non-Night terminal-role death, if applicable."""
        role = _canonical_terminal_loss_role(current_role)
        if role is None:
            return False
        self.terminal_loss_role = role
        return True

    def record_execution_blocked(self, pos: int,
                                 reason: str = "Knight immunity") -> None:
        """Persist a confirmed-good execution attempt that left the card alive."""
        if self.current_identity_may_have_moved():
            raise ValueError(
                "Survived execution is alignment-neutral after current-data "
                "movement; blocked Good bookkeeping is unsafe"
            )
        if pos not in self.confirmed_good:
            self.confirmed_good.append(pos)
        # A protected card is alive and must never enter the executed list.
        self.save()
        DecisionLog.log_custom(
            "Execution Blocked",
            f"#{pos} {reason} — confirmed good, no HP loss",
        )

    def add_pd_ability_result(self, pd_pos: int, target: int, is_corrupted: bool,
                              evil_revealed: Optional[int] = None):
        if any(result.get("pd_pos") == pd_pos for result in self.pd_ability_results):
            raise ValueError(f"Plague Doctor #{pd_pos} already has a recorded result")
        actor = next((card for card in self.cards if card.position == pd_pos), None)
        actor_role = (
            actor.apparent_role.lower().replace(" ", "_")
            if actor is not None else None
        )
        if actor_role != "plague_doctor":
            shown = actor.apparent_role if actor is not None else "unrevealed"
            raise ValueError(
                f"Position #{pd_pos} is {shown}, not an apparent Plague Doctor"
            )
        self.pd_ability_results.append({
            "pd_pos": pd_pos,
            "target": target,
            "is_corrupted": is_corrupted,
            "evil_revealed": evil_revealed,
        })
        self.mark_ability_used(pd_pos)

    def clear_pd_ability_result(self, pd_pos: int) -> int:
        """Remove recorded PD evidence so a mistaken UI entry can be corrected."""
        before = len(self.pd_ability_results)
        self.pd_ability_results = [
            result
            for result in self.pd_ability_results
            if result.get("pd_pos") != pd_pos
        ]
        removed = before - len(self.pd_ability_results)
        if removed and pd_pos in self.used_abilities:
            self.used_abilities.remove(pd_pos)
        return removed

    def mark_ability_used(self, pos: int):
        if pos not in self.used_abilities:
            self.used_abilities.append(pos)

    def _require_no_pending_druid_activation(self, operation: str) -> None:
        """Block Night/reset mutation while a callback click is unresolved."""
        pending_by_role = {
            "Druid": self.druid_pending_activations,
            "Jester": self.jester_pending_activations,
        }
        active = {
            role: pending
            for role, pending in pending_by_role.items()
            if pending
        }
        if not active:
            return
        positions = "; ".join(
            f"{role} " + ", ".join(
                f"#{position}"
                for position in sorted(pending, key=lambda value: str(value))
            )
            for role, pending in active.items()
        )
        raise ValueError(
            f"Cannot {operation} while auto-use callback recovery is pending "
            f"at {positions}; run auto_card before Night/reset"
        )

    def reset_after_night_abilities(
        self,
        *,
        completed_nights: int = 1,
    ) -> list[int]:
        """Apply shipped ResetAfterNight usage to the session model.

        The current public roster audit has proven this usage mode for Judge,
        Fortune Teller, and Druid. Keep accumulated clue evidence, but make each
        apparent resettable actor available again after a completed night.
        """
        if type(completed_nights) is not int or completed_nights <= 0:
            raise ValueError("completed_nights must be a positive integer")
        self._require_no_pending_druid_activation(
            "reset ResetAfterNight abilities"
        )

        from knowledge_base import get_card

        resettable = set()
        for card in self.cards:
            card_def = get_card(card.apparent_role)
            if card_def and card_def.ability_resets_after_night:
                resettable.add(card.position)
            if _execution_role_key(card.apparent_role) == "druid":
                self.druid_reset_generations[card.position] = (
                    max(
                        self.druid_reset_generations.get(card.position, 0)
                        + completed_nights,
                        self.lilis_nights_resolved,
                    )
                )
            if _execution_role_key(card.apparent_role) == "jester":
                self.jester_reset_generations[card.position] = (
                    max(
                        self.jester_reset_generations.get(card.position, 0)
                        + completed_nights,
                        self.lilis_nights_resolved,
                    )
                )
        reset = sorted(resettable.intersection(self.used_abilities))
        if reset:
            self.used_abilities = [
                position
                for position in self.used_abilities
                if position not in resettable
            ]
        return reset

    def record_lilis_night_result(
        self,
        killed_positions: list[int],
        n_evil_among_killed: int = 0,
    ) -> dict:
        """Atomically record one or more pending native Lilis nights.

        Native selects at most one victim per night, so ``N`` unique victims
        are a catch-up recording for ``N`` already-triggered nights. An empty
        list records one no-kill night. Every resolved night deals 2 HP whether
        its victim died, was protected, or did not exist.
        """
        self._require_no_pending_druid_activation("record a Lilis Night")
        if self.has_duplicate_lilis():
            raise ValueError(
                "duplicate Lilis live nights are unsupported: multiple actors "
                "can charge HP while colliding on one delayed victim"
            )
        if not self.has_lilis_night_rule():
            raise ValueError("no Lilis Night rule exists in this deck")
        if self.has_role_in_deck("Shaman"):
            raise ValueError(
                "Lilis+Shaman actor multiplicity is unsupported; no Night "
                "bookkeeping was changed"
            )
        if not self.is_lilis_alive():
            raise ValueError(
                "Lilis is known dead; resolve this rule-only Night with "
                "night_no_kill"
            )

        positions = list(killed_positions)
        if any(not isinstance(position, int) or isinstance(position, bool)
               for position in positions):
            raise ValueError("Lilis victim positions must be integers")
        if len(positions) != len(set(positions)):
            raise ValueError("Lilis victim positions must be unique")
        if any(not 1 <= position <= self.n_cards for position in positions):
            raise ValueError(
                f"Lilis victim positions must be within 1..={self.n_cards}"
            )
        if (not isinstance(n_evil_among_killed, int)
                or isinstance(n_evil_among_killed, bool)
                or not 0 <= n_evil_among_killed <= len(positions)):
            raise ValueError(
                "Lilis evil-victim count must be between 0 and the number "
                "of killed positions"
            )
        already_dead = set(self.executed) | set(self.night_kills)
        repeated_dead = sorted(set(positions) & already_dead)
        if repeated_dead:
            raise ValueError(f"Lilis victim(s) already dead: {repeated_dead}")
        already_revealed = (
            set(self.reveal_order)
            | {card.position for card in self.cards}
        )
        revealed_victims = sorted(set(positions) & already_revealed)
        if revealed_victims:
            raise ValueError(
                f"Lilis victim(s) were already revealed: {revealed_victims}"
            )

        resolved_events = len(positions) if positions else 1
        pending_events = self.pending_lilis_nights
        if pending_events < resolved_events:
            raise ValueError(
                f"Only {max(0, pending_events)} unresolved Lilis night(s) "
                f"remain, cannot record {resolved_events}"
            )

        # All validation completes before any mutation.
        old_hp = self.hp
        self.night_kills.extend(positions)
        self.night_kill_evil_count += n_evil_among_killed
        self.lilis_nights_resolved += resolved_events
        self.hp = _clamped_post_damage_hp(self.hp, 2 * resolved_events)
        if (
            positions
            and self.has_role_in_deck("Witch")
            and self.current_identity_may_have_moved()
        ):
            self.release_witch_blocks(
                "a hidden victim may have carried current Witch data; "
                "public re-probe required"
            )
        elif n_evil_among_killed > 0:
            self.release_witch_blocks(
                "an evil Lilis victim may have been Witch; public re-probe required"
            )
        if positions and n_evil_among_killed == len(positions):
            for position in positions:
                if position not in self.confirmed_evil:
                    self.confirmed_evil.append(position)

        reset_abilities = self.reset_after_night_abilities(
            completed_nights=resolved_events,
        )
        self.pending_lilis_nights -= resolved_events
        return {
            "positions": positions,
            "n_evil": n_evil_among_killed,
            "resolved_events": resolved_events,
            "old_hp": old_hp,
            "new_hp": self.hp,
            "actor_active": True,
            "reset_abilities": reset_abilities,
        }

    def record_lilis_post_death_night(self) -> dict:
        """Synchronize one persistent Night after the Standard Lilis died.

        Native keeps the NightModeRule and still enters Night every four
        successful reveals. A dead Lilis actor does nothing: no victim and no
        2 HP damage. The Night transition still resets ResetAfterNight
        abilities and must be persisted before reveal automation continues.
        """
        self._require_no_pending_druid_activation(
            "record a post-death Lilis Night"
        )
        if self.has_duplicate_lilis():
            raise ValueError(
                "duplicate Lilis live nights are unsupported: actor liveness "
                "and delayed-victim collisions are not represented"
            )
        if not self.has_lilis_night_rule():
            raise ValueError("no Lilis Night rule exists in this deck")
        if self.has_role_in_deck("Shaman"):
            raise ValueError(
                "Lilis+Shaman actor multiplicity is unsupported; no Night "
                "bookkeeping was changed"
            )
        if self.is_lilis_alive():
            raise ValueError(
                "Lilis is still alive; use night_kill or night_no_kill to "
                "record its 2 HP Night action"
            )
        pending_events = self.pending_lilis_nights
        if pending_events < 1:
            raise ValueError("No unresolved Lilis night remains")

        old_hp = self.hp
        self.lilis_nights_resolved += 1
        reset_abilities = self.reset_after_night_abilities()
        self.pending_lilis_nights -= 1
        return {
            "positions": [],
            "n_evil": 0,
            "resolved_events": 1,
            "old_hp": old_hp,
            "new_hp": self.hp,
            "actor_active": False,
            "reset_abilities": reset_abilities,
        }

    def add_slayer_result(self, slayer_pos: int, target_pos: int, killed: bool,
                          revealed_role: Optional[str] = None,
                          was_corrupted: Optional[bool] = None,
                          was_evil: Optional[bool] = None):
        """Record the public result of Slayer's native kill-and-reveal path.

        Slayer tests registered alignment, which can differ from both the
        revealed role's authored alignment and the physical card's runtime
        alignment. A normal Wretch is the common Good/runtime-Good exception.
        Current-data movement can put an authored-Good role on a runtime-Evil
        body. A positive public HP decrease proves a runtime-Good victim, but
        no decrease is ambiguous between runtime Evil and Good carrying a
        preserved NoDamage status. Omit ``was_evil`` when public evidence does
        not resolve that ambiguity; the kill and current role remain usable
        without inventing alignment, origin, or HP. Authored Evil current data
        does not override a runtime-Good body's registered alignment; Slayer
        cannot kill that body (Wretch is the explicit registered exception).
        """
        from knowledge_base import Alignment, get_card, wrong_exec_cost_for

        if any(sr.get("slayer_pos") == slayer_pos for sr in self.slayer_results):
            raise ValueError(f"Slayer #{slayer_pos} already has a recorded result")
        if not 1 <= slayer_pos <= self.n_cards:
            raise ValueError(
                f"Slayer position must be within 1..={self.n_cards}"
            )
        if not 1 <= target_pos <= self.n_cards:
            raise ValueError(
                f"Slayer target must be within 1..={self.n_cards}"
            )
        actor = next(
            (card for card in self.cards if card.position == slayer_pos),
            None,
        )
        if actor is None or _normalize_role_name(actor.apparent_role) != "Slayer":
            shown = actor.apparent_role if actor is not None else "unrevealed"
            raise ValueError(
                f"Position #{slayer_pos} is {shown}, not an apparent Slayer"
            )

        canonical_role = None
        role_def = None
        if killed:
            if not revealed_role:
                raise ValueError("Slayer kill requires the revealed role")
            public_role = revealed_role.strip()
            role_def = get_card(public_role)
            if role_def is None and public_role not in ("Saint", "SaintVillager"):
                raise ValueError(f"Unknown Slayer revealed role: {revealed_role}")
            canonical_role = (
                role_def.name.replace(" ", "_")
                if role_def is not None
                else public_role
            )
        elif revealed_role:
            raise ValueError("A failed Slayer attempt does not reveal a role")
        elif was_corrupted is not None:
            raise ValueError("A failed Slayer attempt does not reveal target status")
        elif was_evil is not None:
            raise ValueError("A failed Slayer attempt does not reveal target alignment")

        target_was_evil = None
        if role_def is not None:
            if role_def.alignment == Alignment.EVIL:
                if was_evil is False:
                    raise ValueError(
                        f"A runtime-Good body carrying current {role_def.name} "
                        "keeps physical Good registered alignment, so Slayer "
                        "cannot produce a kill/reveal result"
                    )
                if was_evil is not None:
                    target_was_evil = was_evil
                elif not self.current_identity_may_have_moved():
                    target_was_evil = True
            elif was_evil is not None:
                target_was_evil = was_evil
            # A Good-class revealed role can still live on a preserved
            # runtime-Evil Shaman destination. Without the public HP outcome,
            # keep that alignment unresolved instead of asking hidden memory.
        elif was_evil is not None:
            target_was_evil = was_evil

        if target_was_evil is not False and was_corrupted is not None:
            raise ValueError(
                "Target status can only be persisted after the public HP "
                "outcome identifies a runtime-Good Slayer victim"
            )

        result = {
            "slayer_pos": slayer_pos,
            "target_pos": target_pos,
            "killed": killed,
        }
        if canonical_role:
            result["revealed_role"] = canonical_role
        if killed and target_was_evil is not None:
            result["was_evil"] = target_was_evil
        self.slayer_results.append(result)
        self.mark_ability_used(slayer_pos)

        if killed:
            if target_was_evil is True:
                # A transformed runtime-Evil card can reveal a copied Good role.
                # In that case the public current role is carried by
                # slayer_results, while the original Evil identity remains a
                # solver fact rather than being mislabeled as (for example)
                # an Evil Knight.
                self.mark_executed(
                    target_pos,
                    was_evil=True,
                    evil_role=(
                        canonical_role
                        if (
                            not self.current_identity_may_have_moved()
                            and role_def is not None
                            and role_def.alignment == Alignment.EVIL
                        )
                        else None
                    ),
                    true_role=canonical_role,
                    record_current_role=False,
                )
            elif target_was_evil is False:
                self.mark_executed(
                    target_pos,
                    was_evil=False,
                    was_corrupted=was_corrupted,
                    true_role=canonical_role,
                    record_current_role=False,
                )
                # KillAndReveal publishes Character.Kill and therefore base
                # wrong-kill damage, but never runs OnExecuted. In particular,
                # a Slayer-killed corrupted Good Knight costs 5, not 5+4.
                # Native resource handling precedes Bombardier's delayed
                # terminal callback. An explicit Good outcome represents a
                # visible positive HP delta; an omitted alignment remains
                # unresolved because preserved NoDamage can suppress it.
                damage = wrong_exec_cost_for(
                    canonical_role, default=self.wrong_exec_cost,
                )
                self.hp = _clamped_post_damage_hp(self.hp, damage)
            else:
                # Kill and revealed current role are public facts. Runtime
                # alignment, confirmation maps, corruption, and HP remain
                # unresolved unless separate public evidence resolves them.
                if target_pos not in self.executed:
                    self.executed.append(target_pos)
                self._apply_current_death_hooks(target_pos, canonical_role)

    # -- Solver --

    def to_game_state(self) -> GameState:
        deck = DeckComposition(
            villagers=list(self.villagers),
            outcasts=list(self.outcasts),
            minions=list(self.minions),
            demons=list(self.demons),
        )
        return GameState(
            n_cards=self.n_cards,
            deck=deck,
            cards=list(self.cards),
            n_evil=self.n_evil,
            executed=list(self.executed),
            confirmed_evil=list(self.confirmed_evil),
            confirmed_good=list(self.confirmed_good),
            pd_corruption_target=self.pd_corruption_target,
            executed_evil_roles=dict(self.executed_evil_roles),
            slayer_results=list(self.slayer_results),
            pd_ability_results=list(self.pd_ability_results),
            blocked_positions=list(self.blocked_positions),
            night_kills=list(self.night_kills),
            night_kill_evil_count=self.night_kill_evil_count,
            hp=self.hp,
            wrong_exec_cost=self.wrong_exec_cost,
            board_villager_count=self.board_villager_count,
            board_outcast_count=self.board_outcast_count,
            board_count_provenance=self.board_count_provenance,
            rambler_rule_version=self.rambler_rule_version,
            rambler_shut_up_observations=[
                dict(observation)
                for observation in self.rambler_shut_up_observations
            ],
            baker_rule_version=self.baker_rule_version,
            doppel_drunk_rule_version=self.doppel_drunk_rule_version,
            fortune_teller_rule_version=self.fortune_teller_rule_version,
            terminal_loss_role=self.terminal_loss_role,
            executed_current_roles=dict(self.executed_current_roles),
            revealed_night_current_roles=dict(
                self.revealed_night_current_roles
            ),
            reveal_order=list(self.reveal_order),
            executed_good_corrupted=dict(self.executed_good_corrupted),
            executed_good_roles=dict(self.executed_good_roles),
        )

    @classmethod
    def from_game_state(cls, state: GameState,
                        used_abilities: Optional[list[int]] = None,
                        lilis_batch_index: int = 0,
                        lilis_nights_resolved: Optional[int] = None,
                        pending_lilis_nights: int = 0,
                        druid_reset_generations: Optional[dict] = None,
                        druid_pending_activations: Optional[dict] = None,
                        jester_reset_generations: Optional[dict] = None,
                        jester_pending_activations: Optional[dict] = None,
                        ) -> "GameSession":
        session = cls(state.n_cards, state.n_evil)
        session.villagers = list(state.deck.villagers)
        session.outcasts = list(state.deck.outcasts)
        session.minions = list(state.deck.minions)
        session.demons = list(state.deck.demons)
        session.cards = list(state.cards)
        session.executed = list(state.executed)
        session.confirmed_evil = list(state.confirmed_evil)
        session.confirmed_good = list(state.confirmed_good)
        session.pd_corruption_target = state.pd_corruption_target
        session.executed_evil_roles = dict(state.executed_evil_roles)
        session.slayer_results = list(state.slayer_results)
        session.pd_ability_results = list(state.pd_ability_results)
        session.blocked_positions = list(state.blocked_positions)
        session.night_kills = list(state.night_kills)
        session.night_kill_evil_count = state.night_kill_evil_count
        session.hp = state.hp
        session.wrong_exec_cost = state.wrong_exec_cost
        session.board_villager_count = state.board_villager_count
        session.board_outcast_count = state.board_outcast_count
        session.board_count_provenance = state.board_count_provenance
        session.rambler_rule_version = state.rambler_rule_version
        session.rambler_shut_up_observations = [
            dict(observation)
            for observation in state.rambler_shut_up_observations
        ]
        session.baker_rule_version = state.baker_rule_version
        session.doppel_drunk_rule_version = state.doppel_drunk_rule_version
        session.fortune_teller_rule_version = state.fortune_teller_rule_version
        session.terminal_loss_role = _canonical_terminal_loss_role(
            getattr(state, 'terminal_loss_role', None)
        )
        session.executed_current_roles = dict(
            getattr(state, 'executed_current_roles', {})
        )
        session.revealed_night_current_roles = dict(
            getattr(state, 'revealed_night_current_roles', {})
        )
        session.reveal_order = list(state.reveal_order)
        session.executed_good_corrupted = dict(getattr(state, 'executed_good_corrupted', {}))
        session.executed_good_roles = dict(getattr(state, 'executed_good_roles', {}))
        session.used_abilities = list(used_abilities or [])
        for raw_position, generation in (
            druid_reset_generations or {}
        ).items():
            try:
                position = int(raw_position)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "Druid reset-generation position must be an integer"
                ) from exc
            if (
                type(generation) is not int
                or generation < 0
                or not 1 <= position <= state.n_cards
            ):
                raise ValueError("Persisted Druid reset generation is malformed")
            session.druid_reset_generations[position] = generation
        for raw_position, token in (
            druid_pending_activations or {}
        ).items():
            try:
                position = int(raw_position)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "Pending Druid activation position must be an integer"
                ) from exc
            if not 1 <= position <= state.n_cards or not isinstance(token, dict):
                raise ValueError("Persisted pending Druid activation is malformed")
            session.druid_pending_activations[position] = copy.deepcopy(token)
        for raw_position, generation in (
            jester_reset_generations or {}
        ).items():
            try:
                position = int(raw_position)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "Jester reset-generation position must be an integer"
                ) from exc
            if (
                type(generation) is not int
                or generation < 0
                or not 1 <= position <= state.n_cards
            ):
                raise ValueError("Persisted Jester reset generation is malformed")
            session.jester_reset_generations[position] = generation
        for raw_position, token in (
            jester_pending_activations or {}
        ).items():
            try:
                position = int(raw_position)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "Pending Jester activation position must be an integer"
                ) from exc
            if not 1 <= position <= state.n_cards or not isinstance(token, dict):
                raise ValueError("Persisted pending Jester activation is malformed")
            session.jester_pending_activations[position] = copy.deepcopy(token)
        if lilis_nights_resolved is None:
            # Legacy saves retain successful victims but omit no-kill history.
            # Infer only provable successful resolutions; never invent old
            # no-kill evidence from final reveal order or HP.
            session.lilis_nights_resolved = len(session.night_kills)
        else:
            session.lilis_nights_resolved = max(0, int(lilis_nights_resolved))
        session.lilis_batch_index = max(
            int(lilis_batch_index),
            session.lilis_nights_resolved,
        )
        # A missing value means a legacy save. Never infer unresolved native
        # work from historical counters because old no-kill timing is absent.
        session.pending_lilis_nights = max(0, int(pending_lilis_nights))
        return session

    def _solve(self, state: GameState) -> SolverResult:
        """Run the Rust solver."""
        if self.twin_live_solver_unsafe():
            print(
                "\n  !! LIVE SAFETY STOP: every Twin Minion game requires "
                "the ordered current-data trace. Rust was not called and no "
                "solver action was produced. !!\n"
            )
            return SolverResult(
                definite_evil=[],
                definite_good=[],
                bombardier_positions=[],
                n_scenarios=0,
                n_surviving=0,
                surviving_scenarios=[],
                reasoning=[
                    "LIVE SAFETY STOP: Twin Minion solving is paused until "
                    "the exact ordered data permutation is modeled"
                ],
            )
        result = rust_solve_to_objects(state)
        if result is None:
            print("\n  !! RUST SOLVER UNAVAILABLE — run `cargo build --release` !!")
            print("  Returning empty result.\n")
            return SolverResult(
                definite_evil=[], definite_good=[], bombardier_positions=[],
                n_scenarios=0, n_surviving=0, surviving_scenarios=[],
                reasoning=["ERROR: Rust solver binary not found"],
            )
        return result

    def solve(self) -> SolverResult:
        from strategy import _has_terminal_role_loss

        state = self.to_game_state()
        result = self._solve(state)
        print(f"\n=== SOLVER RESULT ===")
        for line in result.reasoning:
            print(f"  {line}")
        terminal_loss = _has_terminal_role_loss(state, result)
        ordinary_bombardiers = ordinary_execution_bombardier_positions(
            state, result,
        )
        safe_definite_evil = [
            pos
            for pos in result.definite_evil
            if pos not in ordinary_bombardiers
        ]
        if terminal_loss:
            print(
                "\n  >> TERMINAL LOSS: a current-role Bombardier died "
                "outside Night. No further execution is legal."
            )
        elif safe_definite_evil:
            print(f"\n  >> EXECUTE: {['#'+str(p) for p in safe_definite_evil]}")
        if ordinary_bombardiers and not terminal_loss:
            print(f"  >> DO NOT EXECUTE (Bombardier): {['#'+str(p) for p in sorted(ordinary_bombardiers)]}")
        if result.n_surviving == 0 and not terminal_loss:
            if self.twin_live_solver_unsafe():
                print(
                    "\n  !! LIVE TWIN SAFETY STOP — no solve or action is "
                    "available until the ordered data trace is modeled. !!"
                )
            else:
                print(f"\n  !! NO VALID SCENARIOS — check your input data !!")
        elif not terminal_loss and not safe_definite_evil:
            print(f"\n  >> No safe definite evil yet. Reveal more cards.")
            # Show per-position evil probability
            if result.n_surviving > 0:
                state = self.to_game_state()
                probs = evil_probabilities(state, result)
                for pos in sorted(probs):
                    pct = probs[pos] * 100
                    if 0 < pct < 100:
                        evil_count = int(round(probs[pos] * result.n_surviving))
                        print(f"     #{pos}: {pct:.0f}% chance evil ({evil_count}/{result.n_surviving})")
        print(f"  ({result.n_surviving} surviving scenarios out of {result.n_scenarios})\n")
        return result

    def next_action(self):
        """Run solver + strategy, print full recommendation."""
        # Validate: warn about positions with no card entry
        entered = {c.position for c in self.cards}
        dead = set(self.executed) | set(self.night_kills)
        blocked = set(self.blocked_positions)
        all_pos = set(range(1, self.n_cards + 1))
        missing = all_pos - entered - dead - blocked
        if missing:
            print(f"  WARNING: No card entry for positions {sorted(missing)}. "
                  f"Did you forget to enter info for flipped cards?")
        # Validate: blocked positions without Witch in deck = likely click failure
        if blocked:
            if not self.has_role_in_deck("Witch"):
                print(f"  !! BLOCKED positions {sorted(blocked)} but NO WITCH in deck!")
                print(f"  !! This is likely a click failure. Re-flip these cards!")
                print(f"  !! Run: python game_loop.py flip")
        # Validate: check HP consistency — warn if wrong execs exist but HP unchanged
        wrong_execs = [p for p in self.executed if p not in self.confirmed_evil]
        if wrong_execs and self.hp == 10:
            print(f"  WARNING: {len(wrong_execs)} wrong execution(s) recorded but HP is still 10. "
                  f"Did you forget to run set_hp?")
        state = self.to_game_state()
        result = self._solve(state)
        for line in result.reasoning:
            print(f"  {line}")
        DecisionLog.log_solver_output(result, state)
        if self.twin_live_solver_unsafe():
            print(
                "  [next] LIVE TWIN SAFETY STOP: no recommendation or "
                "action was produced."
            )
            return None
        action = print_recommendation(state, result, self.used_abilities)
        DecisionLog.log_recommendation(action)
        return action

    def auto_execute(self, pos: int, result, monitor=None, forced_safe: bool = False) -> dict:
        """Perform in-game execution click sequence, verify via memory_reader, record result.

        Uses MemoryMonitor.wait_for() when available for faster, smarter waits.
        Falls back to fixed sleeps if no monitor provided.

        Args:
            forced_safe: Marks an ordinary execution line proven survivable by
                lookahead. It never bypasses a current-role Bombardier guard.

        Returns: {"success": bool, "blocked": bool, "was_evil": bool|None,
                  "evil_role": str|None, "error": str|None}
        """
        import template_match as _tm
        import mouse as _mouse
        from game_utils import all_game_card_coords

        # Include unrepresented natural Good Bombardiers on hidden cards as
        # well as the solver's represented current-data candidates. No stale
        # forced_safe recommendation may bypass the terminal branch.
        state = self.to_game_state()
        ordinary_bombardiers = ordinary_execution_bombardier_positions(
            state, result,
        )
        if pos in ordinary_bombardiers:
            return {"success": False, "was_evil": None, "evil_role": None,
                    "error": f"Bombardier protection: refusing to execute #{pos}"}

        coords = all_game_card_coords(self.n_cards)
        if pos not in coords:
            return {"success": False, "was_evil": None, "evil_role": None,
                    "error": f"Position {pos} not valid for {self.n_cards}-card game"}

        # Step 1: Dismiss mark menu
        print(f"  [auto_exec] Dismissing mark menu...")
        _mouse.click(1280, 690)
        time.sleep(0.3)

        # Step 2: Click execute button
        print(f"  [auto_exec] Clicking execute button...")
        try:
            _tm.safe_click_at(2265, 1235, "btn_execute_sword")
        except Exception as e:
            return {"success": False, "was_evil": None, "evil_role": None,
                    "error": f"Execute button click failed: {e}"}
        time.sleep(0.3)

        # Step 2.5: Check for active ability on target (clicking would activate it)
        if pos not in self.used_abilities:
            from knowledge_base import get_card
            target_card_entry = next((c for c in self.cards if c.position == pos), None)
            if target_card_entry:
                kb_card = get_card(target_card_entry.apparent_role)
                if kb_card and kb_card.activated_ability:
                    return {"success": False, "was_evil": None, "evil_role": None,
                            "error": f"#{pos} ({target_card_entry.apparent_role}) has unused active ability — clicking would activate it, not execute. Use ability_used {pos} first or execute manually."}

        # Step 3: Click target card
        x, y = coords[pos]
        print(f"  [auto_exec] Clicking #{pos} at ({x}, {y})...")
        _tm.safe_click_at(x, y, f"exec_card{pos}")

        # Step 4: Wait for execution animation + verify via memory reader
        # Use monitor.wait_for() if available (smart wait), else fixed sleep + poll
        print(f"  [auto_exec] Waiting for execution result...")
        target_card = None

        if monitor and monitor.is_healthy():
            # Smart wait: poll memory for state change with 1s minimum delay
            # Predicate: card is Dead (executed) OR still Alive after delay (Knight immunity)
            def _exec_resolved(board):
                if not board:
                    return False
                card = next((c for c in board if c['position'] == pos), None)
                return card and card['state'] in ('Dead', 'Revealed')

            resolved = monitor.wait_for(_exec_resolved, timeout=5, min_delay=1.0)
            if resolved:
                board = monitor.get_board()
                target_card = next((c for c in board if c['position'] == pos), None) if board else None
            else:
                # Timeout — check for Knight immunity or click failure
                board = monitor.get_board()
                if board:
                    target_card = next((c for c in board if c['position'] == pos), None)
        else:
            # Fallback: fixed sleep + poll (original behavior)
            time.sleep(3)
            from memory_reader import MemoryReader
            reader = MemoryReader()
            if not reader.open():
                return {"success": False, "was_evil": None, "evil_role": None,
                        "error": "Cannot open game process for verification"}
            for attempt in range(3):
                cards = reader.read_board()
                if cards:
                    target_card = next((c for c in cards if c['position'] == pos), None)
                    if target_card and target_card['state'] == 'Dead':
                        break
                if attempt < 2:
                    time.sleep(1)
            reader.close()

        if not target_card:
            return {"success": False, "was_evil": None, "evil_role": None,
                    "error": f"Position #{pos} not found in memory reader"}

        if target_card['state'] != 'Dead':
            target_entry = next((c for c in self.cards if c.position == pos), None)
            fallback_role = target_entry.apparent_role if target_entry else None
            identity_may_have_moved = self.current_identity_may_have_moved()
            if (
                identity_may_have_moved
                and target_card.get('state') in ('Alive', 'Revealed')
            ):
                return {
                    "success": False,
                    "blocked": False,
                    "was_evil": None,
                    "evil_role": None,
                    "error": (
                        f"#{pos} survived, but current CharacterData may have "
                        "moved; survival does not establish physical alignment. "
                        "No execution bookkeeping was recorded"
                    ),
                }
            if _observed_knight_immunity(
                target_card,
                fallback_role,
                current_identity_may_have_moved=identity_may_have_moved,
            ):
                # The memory-only true identity validates the public blocked
                # outcome but must not enter session state or decision logs.
                self.record_execution_blocked(pos)
                print(f"  [auto_exec] BLOCKED: #{pos} survived with confirmed Knight immunity")
                print(f"  [auto_exec] #{pos} confirmed GOOD. HP remains {self.hp}")
                return {"success": True, "blocked": True, "was_evil": False,
                        "evil_role": None, "error": None}
            # Hidden = click likely missed (game unfocused?)
            if target_card['state'] == 'Hidden':
                return {"success": False, "was_evil": None, "evil_role": None,
                        "error": f"Card still Hidden — click didn't register (game focused?)"}
            if target_card['state'] in ('Alive', 'Revealed'):
                apparent_role = _execution_apparent_role(target_card, fallback_role) or "unknown"
                true_role = _observed_current_role(target_card) or "unknown"
                return {"success": False, "was_evil": None, "evil_role": None,
                        "error": (f"Card survived, but post-action identity/status does not "
                                  f"confirm immunity ({true_role} showing as {apparent_role}); "
                                  "the click may have missed")}
            return {"success": False, "was_evil": None, "evil_role": None,
                    "error": f"Card state is {target_card['state']}, expected Dead"}

        # Step 5: Determine result
        was_evil = target_card['is_evil']
        current_role = _observed_current_role(target_card)
        if self.current_identity_may_have_moved() and not _is_known_role(current_role):
            return {
                "success": False,
                "blocked": False,
                "was_evil": was_evil,
                "evil_role": None,
                "error": (
                    "Execution resolved, but exact public current death role "
                    "was unavailable; session bookkeeping was left unchanged"
                ),
            }
        evil_role = (
            _consensus_original_evil_role(pos, result, current_role)
            if was_evil
            else None
        )
        origin_unresolved = bool(
            was_evil
            and evil_role is None
        )

        # Step 6: Record into session
        was_corrupted = None
        observed_statuses = _observed_status_keys(target_card)
        if not was_evil:
            was_corrupted = "corrupted" in observed_statuses

        self.mark_executed(
            pos,
            was_evil,
            evil_role,
            was_corrupted,
            current_role,
        )

        # Step 7: HP update
        hp_before = self.hp
        damage = 0
        no_damage = "nodamage" in observed_statuses
        if not was_evil:
            from knowledge_base import execution_cost_for, wrong_exec_cost_for
            true_role = current_role
            target_entry = next((c for c in self.cards if c.position == pos), None)
            fallback_role = target_entry.apparent_role if target_entry else None
            apparent_role = _execution_apparent_role(target_card, fallback_role)
            if no_damage:
                damage = 0
            elif self.terminal_loss_role:
                # Current Bombardier data owns the death callback, so no
                # Knight/Drunk OnExecuted modifier applies. ManageResources
                # still charges the ordinary wrong-kill cost first.
                damage = wrong_exec_cost_for(
                    true_role, default=self.wrong_exec_cost,
                )
            else:
                damage = execution_cost_for(
                    true_role,
                    apparent_role=apparent_role,
                    was_evil=False,
                    was_corrupted=bool(was_corrupted),
                    was_killable=True,
                    default=self.wrong_exec_cost,
                )
            self.hp = _clamped_post_damage_hp(self.hp, damage)
            suffix = ""
            if no_damage:
                suffix = " (NoDamage suppressed the wrong-kill cost)"
            elif damage != self.wrong_exec_cost:
                shown = f", showing as {apparent_role}" if apparent_role else ""
                suffix = f" ({true_role or 'unknown'}{shown}: -{damage})"
            print(
                f"  [auto_exec] WRONG EXECUTION! HP {hp_before} -> "
                f"{self.hp}{suffix}"
            )
        else:
            print(f"  [auto_exec] Correct execution. HP remains {self.hp}")
            if origin_unresolved:
                print(
                    "  [auto_exec] ORIGIN UNRESOLVED: current CharacterData "
                    "was not used as the stable Evil role. Recover solver "
                    "worlds or re-run manual bookkeeping with an explicit "
                    "origin."
                )

        if self.terminal_loss_role:
            print(
                "  [auto_exec] TERMINAL LOSS: a current-role Bombardier "
                "died after native resource handling "
                f"(HP {hp_before} -> {self.hp})."
            )

        self.save()
        DecisionLog.log_execution(
            pos,
            was_evil,
            evil_role,
        )

        return {"success": True, "blocked": False,
                "was_evil": was_evil,
                "evil_role": evil_role,
                "origin_unresolved": origin_unresolved,
                "terminal_loss_role": self.terminal_loss_role,
                "error": None}

    def auto_use_ability(self, action, monitor=None) -> dict:
        """Perform in-game active-ability activation + target clicks + auto-parse.

        Template: auto_execute, but for active abilities like Jester/Dreamer/FT/Judge.
        Slayer still uses its dedicated `slayer_result` command. Plague Doctor
        is parsed from its exact public speech text and recorded through the
        same state path as `pd_check`.

        Flow:
          1. Snapshot the actor's append-only native event prefix
          2. Click active card → game enters target-selection mode
          3. Click each target in order
          4. Require a new coherent acted-info suffix and parse it
          5. Record the result, then mark the session ability used

        Returns: {"success": bool, "info_parsed": dict|None, "error": str|None}
        """
        import template_match as _tm
        from game_utils import all_game_card_coords

        if action.action_type != "use_ability":
            return {"success": False, "info_parsed": None,
                    "error": f"Expected use_ability action, got {action.action_type}"}

        pos = action.position
        targets = list(action.targets or [])
        ability_name = (action.ability_name or "").lower().replace(" ", "_")
        actor = next((card for card in self.cards if card.position == pos), None)

        # Slayer's kill/death result still needs its dedicated execution path.
        if ability_name == "slayer":
            return {"success": False, "info_parsed": None,
                    "error": f"{action.ability_name} requires manual handling (use slayer_result)"}
        supported_abilities = {
            "dreamer",
            "druid",
            "fortune_teller",
            "jester",
            "judge",
            "plague_doctor",
        }
        if ability_name not in supported_abilities:
            return {
                "success": False,
                "info_parsed": None,
                "error": (
                    f"{action.ability_name or 'Unknown ability'} has no "
                    "authenticated autonomous result path"
                ),
            }

        if (
            ability_name == "dreamer"
            and (
                len(targets) != 2
                or any(type(target) is not int for target in targets)
                or len(set(targets)) != 2
            )
        ):
            return {"success": False, "info_parsed": None,
                    "error": f"Dreamer requires exactly 2 distinct integer targets, got {targets}"}
        if (
            ability_name == "jester"
            and (
                len(targets) != 3
                or any(type(target) is not int for target in targets)
                or len(set(targets)) != 3
            )
        ):
            return {"success": False, "info_parsed": None,
                    "error": f"Jester requires exactly 3 distinct integer targets, got {targets}"}
        if (
            ability_name == "fortune_teller"
            and (
                len(targets) != 2
                or any(type(target) is not int for target in targets)
                or targets[0] == targets[1]
            )
        ):
            return {"success": False, "info_parsed": None,
                    "error": f"Fortune Teller requires exactly 2 distinct integer targets, got {targets}"}
        if (
            ability_name == "druid"
            and (
                len(targets) != 3
                or any(type(target) is not int for target in targets)
                or len(set(targets)) != 3
            )
        ):
            return {"success": False, "info_parsed": None,
                    "error": f"Druid requires exactly 3 distinct integer targets, got {targets}"}
        if ability_name == "judge" and len(targets) != 1:
            return {"success": False, "info_parsed": None,
                    "error": f"Judge requires exactly 1 target, got {targets}"}
        if ability_name == "plague_doctor" and len(targets) != 1:
            return {"success": False, "info_parsed": None,
                    "error": f"Plague Doctor requires exactly 1 target, got {targets}"}
        if ability_name in supported_abilities:
            actor_role = (
                actor.apparent_role.lower().replace(" ", "_")
                if actor is not None else None
            )
            if actor_role != ability_name:
                shown = actor.apparent_role if actor is not None else "unrevealed"
                display_name = (
                    "Plague Doctor" if ability_name == "plague_doctor"
                    else "Fortune Teller" if ability_name == "fortune_teller"
                    else "Dreamer" if ability_name == "dreamer"
                    else "Jester" if ability_name == "jester"
                    else "Druid" if ability_name == "druid"
                    else "Judge"
                )
                return {
                    "success": False,
                    "info_parsed": None,
                    "error": (
                        f"Position #{pos} is {shown}, not an apparent "
                        f"{display_name}"
                    ),
                }

        druid_pre_events: list[dict] = []
        druid_session_generation = None
        if ability_name == "druid":
            if (
                self.baker_rule_version != BAKER_RULE_VERSION
                or not self.reveal_order
                or len(self.reveal_order) > self.n_cards
                or len(set(self.reveal_order)) != len(self.reveal_order)
                or any(
                    type(position) is not int
                    or not 1 <= position <= self.n_cards
                    for position in self.reveal_order
                )
                or pos not in self.reveal_order
            ):
                return {
                    "success": False,
                    "info_parsed": None,
                    "error": (
                        f"Cannot safely activate Druid #{pos}: verified "
                        "reveal_order/Baker provenance including the actor is "
                        "required"
                    ),
                }

            if (
                isinstance(actor.info_parsed, dict)
                and actor.info_parsed.get("druid_variant")
                == _PUBLIC_CURRENT_VARIANT
            ):
                if (
                    actor.info_parsed.get("callback_ledger_variant")
                    != _ORDERED_CALLBACK_LEDGER_VARIANT
                ):
                    return {
                        "success": False,
                        "info_parsed": None,
                        "error": (
                            f"Cannot safely activate Druid #{pos}: scalar-only "
                            "current evidence cannot be resumed; restart the "
                            "village from verified reveal history"
                        ),
                    }
                try:
                    druid_pre_events = _druid_callback_ledger(
                        actor.info_parsed,
                        actor_position=pos,
                        n_cards=self.n_cards,
                        reveal_order=self.reveal_order,
                        baker_rule_version=self.baker_rule_version,
                    )
                except ValueError as exc:
                    return {
                        "success": False,
                        "info_parsed": None,
                        "error": (
                            f"Cannot safely activate Druid #{pos}: malformed "
                            f"persisted callback ledger ({exc})"
                        ),
                    }
                if pos not in self.druid_reset_generations:
                    return {
                        "success": False,
                        "info_parsed": None,
                        "error": (
                            f"Cannot safely activate Druid #{pos}: persisted "
                            "callback history has no session reset-generation "
                            "provenance; restart the village"
                        ),
                    }
            elif actor.info_parsed or actor.info_text:
                return {
                    "success": False,
                    "info_parsed": None,
                    "error": (
                        f"Cannot safely activate Druid #{pos}: unversioned "
                        "active evidence cannot be joined to an ordered "
                        "current callback ledger; restart the village"
                    ),
                }
            druid_session_generation = self.druid_reset_generations.get(
                pos,
                self.lilis_nights_resolved,
            )
            if pos in self.druid_pending_activations:
                return {
                    "success": False,
                    "info_parsed": None,
                    "error": (
                        f"Cannot safely activate Druid #{pos}: a persisted "
                        "auto-use click is awaiting callback recovery; run "
                        "auto_card before clicking again"
                    ),
                }
            if druid_pre_events and (
                druid_session_generation
                <= druid_pre_events[-1]["reset_generation"]
            ):
                return {
                    "success": False,
                    "info_parsed": None,
                    "error": (
                        f"Cannot safely activate Druid #{pos}: no unconsumed "
                        "reset generation proves another activation"
                    ),
                }

        jester_pre_events: list[dict] = []
        jester_session_generation = None
        if ability_name == "jester":
            if (
                self.baker_rule_version != BAKER_RULE_VERSION
                or not self.reveal_order
                or len(self.reveal_order) > self.n_cards
                or len(set(self.reveal_order)) != len(self.reveal_order)
                or any(
                    type(position) is not int
                    or not 1 <= position <= self.n_cards
                    for position in self.reveal_order
                )
                or pos not in self.reveal_order
            ):
                return {
                    "success": False,
                    "info_parsed": None,
                    "error": (
                        f"Cannot safely activate Jester #{pos}: verified "
                        "reveal_order/Baker provenance including the actor is "
                        "required"
                    ),
                }
            info = actor.info_parsed if isinstance(actor.info_parsed, dict) else {}
            if info.get("jester_variant") == _PUBLIC_CURRENT_VARIANT:
                if info.get(
                    "callback_ledger_variant"
                ) == _ORDERED_CALLBACK_LEDGER_VARIANT:
                    try:
                        jester_pre_events = _jester_callback_ledger(
                            info,
                            actor_position=pos,
                            n_cards=self.n_cards,
                            reveal_order=self.reveal_order,
                            baker_rule_version=self.baker_rule_version,
                        )
                    except ValueError as exc:
                        return {
                            "success": False,
                            "info_parsed": None,
                            "error": (
                                f"Cannot safely activate Jester #{pos}: malformed "
                                f"persisted callback ledger ({exc})"
                            ),
                        }
                    if pos not in self.jester_reset_generations:
                        return {
                            "success": False,
                            "info_parsed": None,
                            "error": (
                                f"Cannot safely activate Jester #{pos}: persisted "
                                "callback history has no session reset-generation "
                                "provenance; restart the village"
                            ),
                        }
                elif set(info) != {"jester_variant"}:
                    return {
                        "success": False,
                        "info_parsed": None,
                        "error": (
                            f"Cannot safely activate Jester #{pos}: scalar-only "
                            "current evidence cannot be resumed; restart the "
                            "village from verified reveal history"
                        ),
                    }
            elif info or actor.info_text:
                return {
                    "success": False,
                    "info_parsed": None,
                    "error": (
                        f"Cannot safely activate Jester #{pos}: unversioned "
                        "active evidence cannot be joined to an ordered current "
                        "callback ledger; restart the village"
                    ),
                }
            jester_session_generation = self.jester_reset_generations.get(
                pos,
                self.lilis_nights_resolved,
            )
            if pos in self.jester_pending_activations:
                return {
                    "success": False,
                    "info_parsed": None,
                    "error": (
                        f"Cannot safely activate Jester #{pos}: a pending "
                        "persisted auto-use click is awaiting callback recovery; run "
                        "auto_card before clicking again"
                    ),
                }
            if jester_pre_events and (
                jester_session_generation
                <= jester_pre_events[-1]["reset_generation"]
            ):
                return {
                    "success": False,
                    "info_parsed": None,
                    "error": (
                        f"Cannot safely activate Jester #{pos}: no unconsumed "
                        "reset generation proves another activation"
                    ),
                }

        if pos in self.used_abilities:
            return {"success": False, "info_parsed": None,
                    "error": f"#{pos} ability already marked used"}

        coords = all_game_card_coords(self.n_cards)
        if pos not in coords:
            return {"success": False, "info_parsed": None,
                    "error": f"Position {pos} not valid for {self.n_cards}-card game"}
        for t in targets:
            if t not in coords:
                return {"success": False, "info_parsed": None,
                        "error": f"Target {t} not valid for {self.n_cards}-card game"}

        from knowledge_base import get_card
        for t in targets:
            # Druid, Judge, and Fortune Teller use picker-first OnClick routing, so
            # every board card is selectable, including self and a target
            # with its own unused active ability.
            if ability_name in {
                "dreamer",
                "druid",
                "fortune_teller",
                "jester",
                "judge",
                "plague_doctor",
            }:
                continue
            if t in self.used_abilities:
                continue
            target_card_entry = next((c for c in self.cards if c.position == t), None)
            if not target_card_entry:
                continue
            kb_card = get_card(target_card_entry.apparent_role)
            if kb_card and kb_card.activated_ability:
                return {"success": False, "info_parsed": None,
                        "error": f"#{t} ({target_card_entry.apparent_role}) has unused active ability; clicking it would activate the card instead of selecting it. Use ability_used {t} first or handle this ability manually."}

        # Every successful active result appends to Character.actedInfos before
        # decrementing the shared physical Day-callback budget. Snapshot that
        # full prefix before any UI input; neither constructor-true ``act`` nor
        # a retained history entry can prove this click completed.
        event_history_ability = ability_name in supported_abilities
        repeatable_event_ability = ability_name in {
            "druid", "fortune_teller", "jester", "judge",
        }
        pre_event = None
        pre_history_snapshot = None
        unowned_repeatable_prefix_count = 0
        if event_history_ability:
            display_name = {
                "dreamer": "Dreamer",
                "druid": "Druid",
                "fortune_teller": "Fortune Teller",
                "jester": "Jester",
                "judge": "Judge",
                "plague_doctor": "Plague Doctor",
            }[ability_name]
            before_board = None
            if monitor and monitor.is_healthy():
                before_board = monitor.get_board()
            else:
                from memory_reader import MemoryReader
                before_reader = MemoryReader()
                if not before_reader.open():
                    return {
                        "success": False,
                        "info_parsed": None,
                        "error": (
                            f"Cannot open memory reader for pre-click {display_name} "
                            "event snapshot"
                        ),
                    }
                try:
                    before_board = before_reader.read_board()
                finally:
                    before_reader.close()
            if not before_board:
                return {
                    "success": False,
                    "info_parsed": None,
                    "error": f"Cannot read board for pre-click {display_name} event snapshot",
                }
            before_card = next(
                (card for card in before_board if card.get('position') == pos),
                None,
            )
            if before_card is None:
                return {
                    "success": False,
                    "info_parsed": None,
                    "error": f"{display_name} #{pos} missing from pre-click memory snapshot",
                }
            observed_role = _observed_active_role_key(before_card)
            if observed_role != ability_name:
                return {
                    "success": False,
                    "info_parsed": None,
                    "error": (
                        f"Cannot safely activate {display_name} #{pos}: "
                        f"pre-click memory shows {observed_role or 'no role'}"
                    ),
                }
            pre_history_snapshot = _acted_history_snapshot(before_card)
            if pre_history_snapshot is None:
                return {
                    "success": False,
                    "info_parsed": None,
                    "error": (
                        f"Cannot safely activate {display_name} #{pos}: "
                        "pre-click acted-info history is unreadable"
                    ),
                }
            remaining = _pickable_uses_remaining(before_card)
            if remaining is None:
                return {
                    "success": False,
                    "info_parsed": None,
                    "error": (
                        f"Cannot safely activate {display_name} #{pos}: native "
                        "pickable-use budget is unreadable"
                    ),
                }
            if remaining <= 0:
                return {
                    "success": False,
                    "info_parsed": None,
                    "error": (
                        f"Cannot safely activate {display_name} #{pos}: native "
                        f"pickable-use budget is {remaining}, so the card is "
                        "not currently available"
                    ),
                }
            pre_event = _latest_acted_event_fingerprint(before_card)
            if repeatable_event_ability:
                session_has_prior_event = (
                    actor is not None
                    and (
                        _has_active_clue_result(actor)
                        or bool(actor.info_text)
                    )
                )
                if ability_name == "druid":
                    parsed_before, druid_before_error = (
                        _parse_druid_result_from_memory(
                            before_card,
                            n_cards=self.n_cards,
                        )
                    )
                    if druid_before_error is not None:
                        return {
                            "success": False,
                            "info_parsed": None,
                            "error": (
                                f"Cannot safely activate Druid #{pos}: "
                                + druid_before_error
                            ),
                        }
                    if parsed_before is None:
                        return {
                            "success": False,
                            "info_parsed": None,
                            "error": (
                                f"Cannot safely activate Druid #{pos}: pre-click "
                                "history ends in an incomplete opaque real callback"
                            ),
                        }
                    raw_before = getattr(parsed_before, "_druid_raw_callbacks", [])
                    if [
                        _druid_callback_signature(event)
                        for event in raw_before
                    ] != [
                        _druid_callback_signature(event)
                        for event in druid_pre_events
                    ]:
                        return {
                            "success": False,
                            "info_parsed": None,
                            "error": (
                                f"Cannot safely activate Druid #{pos}: pre-click "
                                "callback history disagrees with the persisted "
                                "ordered ledger"
                            ),
                        }
                    try:
                        _validate_druid_rambler_sync(
                            druid_pre_events,
                            speaker_position=pos,
                            rambler_observations=(
                                self.rambler_shut_up_observations
                            ),
                        )
                    except ValueError as exc:
                        return {
                            "success": False,
                            "info_parsed": None,
                            "error": (
                                f"Cannot safely activate Druid #{pos}: {exc}"
                            ),
                        }
                if ability_name == "jester":
                    parsed_before, jester_before_error = (
                        _parse_jester_result_from_memory(
                            before_card,
                            n_cards=self.n_cards,
                        )
                    )
                    if jester_before_error is not None:
                        return {
                            "success": False,
                            "info_parsed": None,
                            "error": (
                                f"Cannot safely activate Jester #{pos}: "
                                + jester_before_error
                            ),
                        }
                    if parsed_before is None:
                        return {
                            "success": False,
                            "info_parsed": None,
                            "error": (
                                f"Cannot safely activate Jester #{pos}: pre-click "
                                "history ends in an incomplete opaque real callback"
                            ),
                        }
                    raw_before = getattr(parsed_before, "_jester_raw_callbacks", [])
                    if [
                        _jester_callback_signature(event)
                        for event in raw_before
                    ] != [
                        _jester_callback_signature(event)
                        for event in jester_pre_events
                    ]:
                        return {
                            "success": False,
                            "info_parsed": None,
                            "error": (
                                f"Cannot safely activate Jester #{pos}: pre-click "
                                "callback history disagrees with the persisted "
                                "ordered ledger"
                            ),
                        }
                    try:
                        _validate_jester_rambler_sync(
                            jester_pre_events,
                            speaker_position=pos,
                            rambler_observations=self.rambler_shut_up_observations,
                        )
                    except ValueError as exc:
                        return {
                            "success": False,
                            "info_parsed": None,
                            "error": f"Cannot safely activate Jester #{pos}: {exc}",
                        }
                try:
                    local_expectation = (
                        (
                            pre_event[0],
                            pre_event[1],
                            None,
                            None,
                        )
                        if (
                            ability_name in {"druid", "jester"}
                            and pre_event is not None
                        )
                        else _local_repeatable_event_expectation(
                            actor,
                            n_cards=self.n_cards,
                            rambler_observations=(
                                self.rambler_shut_up_observations
                            ),
                            fortune_teller_rule_version=(
                                self.fortune_teller_rule_version
                            ),
                        )
                    )
                except ValueError as exc:
                    return {
                        "success": False,
                        "info_parsed": None,
                        "error": (
                            f"Cannot safely activate {display_name} #{pos}: "
                            f"{exc}"
                        ),
                    }
                if session_has_prior_event and local_expectation is None:
                    return {
                        "success": False,
                        "info_parsed": None,
                        "error": (
                            f"Cannot safely activate {display_name} #{pos}: "
                            "the session has prior active evidence that cannot be "
                            "reconciled with repeatable event history"
                        ),
                    }
                if pre_event is None and local_expectation is not None:
                    return {
                        "success": False,
                        "info_parsed": None,
                        "error": (
                            f"Cannot safely activate {display_name} #{pos}: the "
                            "session has prior active evidence, but the pre-click "
                            "memory snapshot has no readable newest acted-info event"
                        ),
                    }
                if (
                    ability_name not in {"druid", "jester"}
                    and pre_event is not None
                    and local_expectation is None
                ):
                    # A recovered/reloaded no-info shell may face retained
                    # native history it never persisted. The deliberate click
                    # still provides a strict append boundary; parse only that
                    # authenticated suffix so unowned old events never enter
                    # the current solver state.
                    unowned_repeatable_prefix_count = len(
                        pre_history_snapshot
                    )
                if local_expectation is not None:
                    (
                        minimum_count,
                        expected_latest,
                        expected_normal_history,
                        expected_interruptions,
                    ) = local_expectation
                    if pre_event[0] < minimum_count:
                        return {
                            "success": False,
                            "info_parsed": None,
                            "error": (
                                f"Cannot safely activate {display_name} #{pos}: "
                                "the pre-click acted-info history is shorter than "
                                f"the local minimum ({pre_event[0]} < "
                                f"{minimum_count})"
                            ),
                        }
                    if pre_event[1] != expected_latest:
                        return {
                            "success": False,
                            "info_parsed": None,
                            "error": (
                                f"Cannot safely activate {display_name} #{pos}: "
                                "the pre-click newest acted-info event disagrees "
                                "with locally stored repeatable evidence"
                            ),
                        }
                    if ability_name not in {"druid", "jester"}:
                        try:
                            (
                                memory_normal_history,
                                memory_interruptions,
                            ) = _repeatable_memory_history_projection(
                                before_card,
                                role_key=ability_name,
                                n_cards=self.n_cards,
                            )
                        except ValueError as exc:
                            return {
                                "success": False,
                                "info_parsed": None,
                                "error": (
                                    f"Cannot safely activate {display_name} "
                                    f"#{pos}: {exc}"
                                ),
                            }
                        if (
                            memory_normal_history != expected_normal_history
                            or memory_interruptions != expected_interruptions
                        ):
                            return {
                                "success": False,
                                "info_parsed": None,
                                "error": (
                                    f"Cannot safely activate {display_name} "
                                    f"#{pos}: pre-click native history "
                                    "disagrees with the full local ordered "
                                    "normal/interruption ledgers"
                                ),
                            }

        # Persist the Druid click intent before touching the UI. Native
        # remaining-pickableUses is not cumulative and cannot prove callback
        # grouping; this token plus the verified raw prefix is the strongest
        # available activation provenance.
        if ability_name == "druid":
            generation_was_present = pos in self.druid_reset_generations
            previous_generation = self.druid_reset_generations.get(pos)
            self.druid_reset_generations[pos] = druid_session_generation
            self.druid_pending_activations[pos] = {
                "activation_id": (
                    druid_pre_events[-1]["activation_id"] + 1
                    if druid_pre_events else 1
                ),
                "expected_targets": list(targets),
                "prior_callback_count": len(druid_pre_events),
                "reset_generation": druid_session_generation,
                "settled_reveal_count": len(self.reveal_order),
            }
            try:
                self.save()
            except Exception as exc:
                self.druid_pending_activations.pop(pos, None)
                if generation_was_present:
                    self.druid_reset_generations[pos] = previous_generation
                else:
                    self.druid_reset_generations.pop(pos, None)
                return {
                    "success": False,
                    "info_parsed": None,
                    "error": (
                        f"Cannot persist Druid #{pos} auto-use provenance: {exc}"
                    ),
                }

        if ability_name == "jester":
            generation_was_present = pos in self.jester_reset_generations
            previous_generation = self.jester_reset_generations.get(pos)
            self.jester_reset_generations[pos] = jester_session_generation
            self.jester_pending_activations[pos] = {
                "activation_id": (
                    jester_pre_events[-1]["activation_id"] + 1
                    if jester_pre_events else 1
                ),
                "expected_targets": list(targets),
                "prior_callback_count": len(jester_pre_events),
                "reset_generation": jester_session_generation,
                "settled_reveal_count": len(self.reveal_order),
            }
            try:
                self.save()
            except Exception as exc:
                self.jester_pending_activations.pop(pos, None)
                if generation_was_present:
                    self.jester_reset_generations[pos] = previous_generation
                else:
                    self.jester_reset_generations.pop(pos, None)
                return {
                    "success": False,
                    "info_parsed": None,
                    "error": (
                        f"Cannot persist Jester #{pos} auto-use provenance: {exc}"
                    ),
                }

        # Step 1: Click active card to enter target-selection mode
        x, y = coords[pos]
        print(f"  [auto_ability] Activating {action.ability_name} at #{pos} ({x},{y})...")
        try:
            _tm.safe_click_at(x, y, f"activate_card{pos}")
        except Exception as e:
            recovery = (
                "; persisted click provenance was retained, so inspect/cancel "
                "the picker and recover with auto_card before retrying"
                if ability_name in {"druid", "jester"}
                else ""
            )
            return {"success": False, "info_parsed": None,
                    "error": f"Failed to click active card: {e}{recovery}"}
        time.sleep(0.4)  # Let target-selection mode engage

        # Step 2: Click each target in order
        for t in targets:
            tx, ty = coords[t]
            print(f"  [auto_ability] Target #{t} at ({tx},{ty})...")
            try:
                _tm.safe_click_at(tx, ty, f"ability_target{t}")
            except Exception as e:
                return {"success": False, "info_parsed": None,
                        "error": f"Failed to click target #{t}: {e}"}
            time.sleep(0.25)  # pause between target clicks

        # Step 3: wait for a strict append to the pre-click native event list.
        # Counter decrement is ordered after append and ``act`` never means
        # "used", so neither is a completion signal.
        print(f"  [auto_ability] Waiting for ability result...")
        target_card_data = None

        def _ability_resolved(board):
            if not board:
                return False
            card = next((c for c in board if c['position'] == pos), None)
            if not card:
                return False
            if _observed_active_role_key(card) != ability_name:
                return False
            if ability_name == "druid":
                parsed_druid, parse_error = (
                    _parse_druid_result_from_memory(
                        card,
                        n_cards=self.n_cards,
                        expected_targets=targets,
                    )
                )
                raw_callbacks = (
                    getattr(parsed_druid, "_druid_raw_callbacks", None)
                    if parsed_druid is not None else None
                )
                return (
                    parse_error is None
                    and raw_callbacks is not None
                    and len(raw_callbacks) > len(druid_pre_events)
                    and _has_new_coherent_acted_suffix(
                        card,
                        pre_history_snapshot,
                    )
                )
            if ability_name == "jester":
                parsed_jester, parse_error = (
                    _parse_jester_result_from_memory(
                        card,
                        n_cards=self.n_cards,
                    )
                )
                raw_callbacks = (
                    getattr(parsed_jester, "_jester_raw_callbacks", None)
                    if parsed_jester is not None else None
                )
                return (
                    parse_error is None
                    and raw_callbacks is not None
                    and len(raw_callbacks) > len(jester_pre_events)
                    and _has_new_coherent_acted_suffix(
                        card,
                        pre_history_snapshot,
                    )
                )
            return _has_new_coherent_acted_suffix(
                card,
                pre_history_snapshot,
            )

        # Native RoleAct dispatches real/raw synchronously and both callbacks
        # enter ShowActedDelayed(0.0, ...); see
        # reverse_engineering/notes/systems/gameplay_execution_resolution.md.
        # Thus two stable 0.15s reads are a quiescence check, not a guessed
        # animation bound. A one-record result can still occur when the memory
        # reader races those zero-delay coroutines, so the persisted click
        # provenance emits an ``either`` group; a later raw-history read then
        # atomically upgrades the preserved pair to same_activation_extension.
        if monitor and monitor.is_healthy():
            resolved = monitor.wait_for(_ability_resolved, timeout=6, min_delay=0.8)
            board = monitor.get_board()
            target_card_data = next(
                (c for c in board if c['position'] == pos),
                None,
            ) if board else None
            if (
                resolved
                and ability_name in {"druid", "jester"}
                and target_card_data
            ):
                last_history = _acted_history_fingerprint(target_card_data)
                stable_reads = 0
                for _ in range(4):
                    time.sleep(0.15)
                    board = monitor.get_board()
                    newer = next(
                        (c for c in board if c['position'] == pos),
                        None,
                    ) if board else None
                    if newer is None:
                        continue
                    target_card_data = newer
                    fingerprint = _acted_history_fingerprint(newer)
                    if fingerprint == last_history:
                        stable_reads += 1
                    else:
                        last_history = fingerprint
                        stable_reads = 0
                    if stable_reads >= 2 and _ability_resolved(board):
                        break
        else:
            time.sleep(1.5)  # initial animation delay
            from memory_reader import MemoryReader
            reader = MemoryReader()
            if not reader.open():
                return {"success": False, "info_parsed": None,
                        "error": "Cannot open memory reader for ability verification"}
            try:
                last_history = None
                stable_reads = 0
                for attempt in range(6):
                    cards = reader.read_board()
                    if cards:
                        target_card_data = next((c for c in cards if c['position'] == pos), None)
                        if target_card_data and _ability_resolved(cards):
                            if ability_name not in {"druid", "jester"}:
                                break
                            fingerprint = _acted_history_fingerprint(
                                target_card_data
                            )
                            if fingerprint == last_history:
                                stable_reads += 1
                            else:
                                last_history = fingerprint
                                stable_reads = 0
                            if stable_reads >= 2:
                                break
                    if attempt < 5:
                        time.sleep(0.7)
            finally:
                reader.close()

        if not target_card_data:
            return {"success": False, "info_parsed": None,
                    "error": f"Position #{pos} not found in memory reader after activation"}
        final_observed_role = _observed_active_role_key(target_card_data)
        if final_observed_role != ability_name:
            return {
                "success": False,
                "info_parsed": None,
                "error": (
                    f"{display_name} actor identity changed before final "
                    f"result capture (memory shows "
                    f"{final_observed_role or 'no role'})"
                ),
            }
        if ability_name == "druid":
            parsed_druid, parse_error = _parse_druid_result_from_memory(
                target_card_data,
                n_cards=self.n_cards,
                expected_targets=targets,
            )
            raw_callbacks = (
                getattr(parsed_druid, "_druid_raw_callbacks", None)
                if parsed_druid is not None else None
            )
            has_recorded_result = (
                parse_error is None
                and raw_callbacks is not None
                and len(raw_callbacks) > len(druid_pre_events)
                and _has_new_coherent_acted_suffix(
                    target_card_data,
                    pre_history_snapshot,
                )
            )
        elif ability_name == "jester":
            parsed_jester, parse_error = _parse_jester_result_from_memory(
                target_card_data,
                n_cards=self.n_cards,
            )
            raw_callbacks = (
                getattr(parsed_jester, "_jester_raw_callbacks", None)
                if parsed_jester is not None else None
            )
            has_recorded_result = (
                parse_error is None
                and raw_callbacks is not None
                and len(raw_callbacks) > len(jester_pre_events)
                and _has_new_coherent_acted_suffix(
                    target_card_data,
                    pre_history_snapshot,
                )
            )
        else:
            has_recorded_result = _has_new_coherent_acted_suffix(
                target_card_data,
                pre_history_snapshot,
            )
        if not has_recorded_result:
            return {
                "success": False,
                "info_parsed": None,
                "error": (
                    f"{display_name} result did not produce a coherent strict "
                    "acted-info suffix — click may have missed"
                ),
            }
        parse_card_data = target_card_data
        if (
            unowned_repeatable_prefix_count
            and ability_name in {"fortune_teller", "judge"}
        ):
            appended_infos = target_card_data.get("acted_infos")
            if not isinstance(appended_infos, list):
                return {
                    "success": False,
                    "info_parsed": None,
                    "error": f"{display_name} appended history became unreadable",
                }
            parse_card_data = copy.deepcopy(target_card_data)
            parse_card_data["acted_infos"] = appended_infos[
                unowned_repeatable_prefix_count:
            ]
        if (
            ability_name == "dreamer"
            and not _active_result_refs_match_clicks(
                target_card_data,
                targets,
                n_cards=self.n_cards,
            )
        ):
            return {
                "success": False,
                "info_parsed": None,
                "error": (
                    f"{display_name} newest native references do not match "
                    "the clicked targets"
                ),
            }

        # Ordered-ledger roles own their complete reset history, including a
        # newest Rambler replacement. Other roles retain the shared newest-
        # interruption path.
        if ability_name == "druid":
            parsed, druid_parse_error = _parse_druid_result_from_memory(
                target_card_data,
                expected_targets=targets,
                n_cards=self.n_cards,
            )
            if druid_parse_error is not None:
                return {
                    "success": False,
                    "info_parsed": None,
                    "error": druid_parse_error,
                }
        elif ability_name == "jester":
            parsed, jester_parse_error = _parse_jester_result_from_memory(
                target_card_data,
                expected_targets=targets,
                n_cards=self.n_cards,
            )
            if jester_parse_error is not None:
                return {
                    "success": False,
                    "info_parsed": None,
                    "error": jester_parse_error,
                }
        else:
            parsed, interruption_error = _card_from_rambler_interruption(
                parse_card_data,
                n_cards=self.n_cards,
            )
            if interruption_error is not None:
                return {
                    "success": False,
                    "info_parsed": None,
                    "error": interruption_error,
                }

        # Step 4a: PD has a distinct result object unless Rambler replaced it.
        if parsed is None and ability_name == "plague_doctor":
            pd_result, parse_error = _parse_pd_ability_result_from_memory(
                parse_card_data,
                ability_pos=pos,
                expected_target=targets[0],
                n_cards=self.n_cards,
            )
            if parse_error:
                return {"success": False, "info_parsed": None,
                        "error": parse_error}
            try:
                self.add_pd_ability_result(
                    pos,
                    pd_result["target"],
                    pd_result["is_corrupted"],
                    pd_result["evil_revealed"],
                )
            except ValueError as exc:
                return {
                    "success": False,
                    "info_parsed": None,
                    "error": str(exc),
                }
            self.save()
            DecisionLog.log_ability_used(pos)
            DecisionLog.log_custom(
                "Plague Doctor Result",
                f"#{pos} -> #{pd_result['target']}: "
                + (f"Corrupted, #{pd_result['evil_revealed']} is Evil"
                   if pd_result["is_corrupted"] else "Not Corrupted"),
            )
            print(
                f"  [auto_ability] Plague Doctor #{pos} -> {targets}: "
                f"{pd_result}"
            )
            return {"success": True, "info_parsed": pd_result, "error": None}

        # Step 4b: current Fortune Teller has an exact two-reference history.
        if (
            parsed is None
            and ability_name == "fortune_teller"
            and self.fortune_teller_rule_version
            == FORTUNE_TELLER_RULE_VERSION
        ):
            parsed, parse_error = _parse_fortune_teller_result_from_memory(
                parse_card_data,
                expected_targets=targets,
                n_cards=self.n_cards,
            )
            if parse_error:
                return {"success": False, "info_parsed": None,
                        "error": parse_error}

        # Step 4c: Judge has a strict one-target public result boundary.
        if parsed is None and ability_name == "judge":
            parsed, parse_error = _parse_judge_result_from_memory(
                parse_card_data,
                expected_target=targets[0],
                n_cards=self.n_cards,
            )
            if parse_error:
                return {"success": False, "info_parsed": None,
                        "error": parse_error}
        elif parsed is None:
            # Parse ordinary clue-producing abilities via auto_card.
            parsed = _parse_clue_from_memory(
                parse_card_data,
                n_cards=self.n_cards,
                baker_rule_version=self.baker_rule_version,
                fortune_teller_rule_version=self.fortune_teller_rule_version,
            )
        if parsed is None:
            return {"success": False, "info_parsed": None,
                    "error": f"Could not parse ability result from memory data"}
        parsed_role_key = (
            parsed.apparent_role.casefold().replace(" ", "_")
        )
        if parsed.position != pos or parsed_role_key != ability_name:
            return {
                "success": False,
                "info_parsed": None,
                "error": (
                    f"Authenticated {display_name} result parsed as a "
                    "different actor"
                ),
            }
        if not parsed.info_parsed:
            return {"success": False, "info_parsed": None,
                    "error": f"Parser returned empty info_parsed for {action.ability_name}"}

        # Step 5: Update session
        try:
            self.add_card(parsed, mark_active_result=False)
        except ValueError as exc:
            return {
                "success": False,
                "info_parsed": None,
                "error": (
                    f"Cannot safely record {action.ability_name or ability_name} "
                    f"#{pos}: {exc}"
                ),
            }
        self.mark_ability_used(pos)
        recorded = next(
            card for card in self.cards if card.position == parsed.position
        )
        self.save()
        DecisionLog.log_card(recorded)
        DecisionLog.log_ability_used(pos)

        print(
            f"  [auto_ability] {action.ability_name} #{pos} -> {targets}: "
            f"{recorded.info_parsed}"
        )
        return {
            "success": True,
            "info_parsed": recorded.info_parsed,
            "error": None,
        }

    def auto_next(self):
        """Solve + auto-execute for definite-evil OR lookahead-forced-safe picks.

        Gate: (pos in definite_evil) OR (action.forced_safe AND confidence >= 0.20).
        The forced_safe flag is set by strategy._find_forced_execution when a DFS
        over all surviving scenarios proves a winning line across all branches at
        current HP. That IS the safety proof — confidence alone is misleading.

        Returns (action, result, exec_result).
        """
        state = self.to_game_state()
        result = self._solve(state)

        for line in result.reasoning:
            print(f"  {line}")
        DecisionLog.log_solver_output(result, state)
        if self.twin_live_solver_unsafe():
            print(
                "\n  [auto_next] LIVE TWIN SAFETY STOP: no recommendation, "
                "ability, reveal, or execution was produced."
            )
            return None, result, None
        action = print_recommendation(state, result, self.used_abilities)
        DecisionLog.log_recommendation(action)
        ordinary_bombardiers = ordinary_execution_bombardier_positions(
            state, result,
        )

        # Route USE_ABILITY to auto_use_ability. Slayer still uses its
        # dedicated kill-result command.
        if action.action_type == "use_ability":
            ability_name_lower = (action.ability_name or "").lower().replace(" ", "_")
            if ability_name_lower == "slayer":
                risky_targets = sorted(
                    set(action.targets or []).intersection(
                        result.bombardier_positions
                    )
                )
                if risky_targets:
                    print(
                        "\n  [auto_next] REFUSING Slayer recommendation into "
                        "possible moved Bombardier data at "
                        f"{['#' + str(pos) for pos in risky_targets]}."
                    )
                    return action, result, None
                print(f"\n  [auto_next] {action.ability_name} requires manual handling — use ability_used to skip, or fire the ability in-game and record with slayer_result.")
                return action, result, None
            print(f"\n  === AUTO-ABILITY #{action.position} ({action.ability_name}) -> targets {action.targets} ===")
            exec_result = self.auto_use_ability(action)
            if exec_result["success"]:
                print(f"  AUTO-ABILITY SUCCESS: {action.ability_name} #{action.position} result recorded")
            else:
                print(f"  AUTO-ABILITY FAILED: {exec_result['error']}")
                if ability_name_lower == "plague_doctor":
                    print(
                        "  [RECOVERY] Read the public speech bubble and enter "
                        f"it with `pd_check {action.position} <target> ...`; use "
                        f"`ability_used {action.position}` only if no result exists"
                    )
                else:
                    print(f"  [RECOVERY] Re-run 'next --plan' to see state; enter manually via 'card {ability_name_lower} {action.position} ...' or `ability_used {action.position}`")
            return action, result, exec_result

        # Safety checks for auto-execution
        if action.action_type != "execute":
            print(f"\n  [auto_next] Not an execute recommendation — manual action needed.")
            return action, result, None

        pos = action.position
        is_forced_safe = getattr(action, 'forced_safe', False)
        is_definite = pos in result.definite_evil
        # Belt-and-suspenders: even forced-safe picks need a minimum confidence
        # floor in case a future strategy bug sets forced_safe=True incorrectly.
        FORCED_SAFE_FLOOR = 0.20
        allow_auto = is_definite or (is_forced_safe and action.confidence >= FORCED_SAFE_FLOOR)
        if not allow_auto:
            print(f"\n  [auto_next] #{pos} is not auto-executable "
                  f"(confidence={action.confidence:.0%}, forced_safe={is_forced_safe}) — "
                  f"manual decision needed.")
            return action, result, None

        if pos in ordinary_bombardiers:
            print(f"\n  [auto_next] #{pos} is a possible current-role Bombardier — refusing execution.")
            return action, result, None

        # HP budget guard: skip for forced_safe picks (lookahead budgeted HP)
        # and definite evils (a correct execution cannot reduce HP). This guard
        # is only for future non-definite auto paths.
        if not is_forced_safe and not is_definite:
            if self.hp <= self.wrong_exec_cost and result.n_surviving > 1:
                print(f"\n  [auto_next] HP={self.hp} too low for auto-exec (cost={self.wrong_exec_cost}). Manual decision needed.")
                return action, result, None

        # Re-verify board state from memory before clicking
        from memory_reader import MemoryReader
        reader = MemoryReader()
        board_ok = False
        if reader.open():
            cards = reader.read_board()
            reader.close()
            if cards:
                target = next((c for c in cards if c['position'] == pos), None)
                if target and target['state'] in ('Alive', 'Hidden'):
                    board_ok = True
                else:
                    print(f"\n  [auto_next] #{pos} state is {target['state'] if target else 'missing'} — aborting auto-exec.")
        if not board_ok:
            print(f"\n  [auto_next] Board verification failed — manual execution needed.")
            return action, result, None

        # All checks passed — auto-execute!
        if is_definite:
            print(f"\n  === AUTO-EXECUTING #{pos} (definite evil in all {result.n_surviving} scenarios) ===")
        else:
            print(f"\n  === AUTO-EXECUTING #{pos} (FORCED-SAFE, confidence={action.confidence:.0%}, lookahead proved survival across {result.n_surviving} scenarios) ===")
        exec_result = self.auto_execute(pos, result, forced_safe=is_forced_safe)

        if exec_result["success"]:
            if exec_result.get("blocked"):
                print(f"  AUTO-EXEC BLOCKED: #{pos} survived with Knight immunity (confirmed good)")
            elif exec_result["was_evil"]:
                print(f"  AUTO-EXEC SUCCESS: #{pos} was {exec_result['evil_role']}")
            else:
                print(f"  AUTO-EXEC: #{pos} was GOOD (wrong execution)")
        else:
            print(f"  AUTO-EXEC FAILED: {exec_result['error']}")
            print(f"  [RECOVERY] Re-run 'next --plan' to see state. Use 'execute {pos}' for manual bookkeeping if the click actually landed.")

        print(f"  ({result.n_surviving} surviving scenarios)")
        return action, result, exec_result

    # -- Status --

    def status(self):
        print(f"\n=== GAME SESSION ===")
        print(f"  Cards: {self.n_cards}, Evil: {self.n_evil}")
        if self.villagers:
            print(f"  Deck V: {', '.join(self.villagers)}")
            print(f"       O: {', '.join(self.outcasts)}")
            print(f"       M: {', '.join(self.minions)}")
            print(f"       D: {', '.join(self.demons)}")
        if self.cards:
            print(f"  Revealed cards:")
            for c in self.cards:
                extra = ""
                if c.position in self.executed:
                    extra = " [EXECUTED]"
                if c.position in self.confirmed_evil:
                    extra += " [EVIL]"
                if c.position in self.confirmed_good:
                    extra += " [GOOD]"
                print(f"    #{c.position} {c.apparent_role}: {c.info_parsed}{extra}")
        if self.executed:
            print(f"  Executed: {['#'+str(p) for p in self.executed]}")
        if self.pd_corruption_target:
            print(
                "  PD corruption target (fixture/post-mortem only): "
                f"#{self.pd_corruption_target}"
            )
        print()

    # -- Game actions (wraps game_utils) --

    def screenshot(self, name: Optional[str] = None) -> str:
        import game_utils
        return game_utils.take_game_screenshot(name)

    def reveal(self, pos: int):
        """Click card at position to reveal it. Requires card detection."""
        import game_utils
        path = game_utils.take_game_screenshot("_card_detect")
        positions = game_utils.detect_card_positions(path)
        if pos < 1 or pos > len(positions):
            print(f"[reveal] Position #{pos} out of range (detected {len(positions)} cards)")
            return
        x, y = positions[pos - 1]
        game_utils.reveal_card((x, y))
        print(f"[reveal] Revealed card #{pos} at ({x}, {y})")

    def execute(self, pos: int):
        """Execute card at position."""
        import game_utils
        import card_vision
        path = game_utils.take_game_screenshot("_card_detect")
        if pos < 1 or pos > self.n_cards:
            print(f"[execute] Position #{pos} out of range (board has {self.n_cards} cards)")
            return
        x, y = card_vision.resolved_board_seat_center(path, pos, self.n_cards)
        game_utils.execute_card((x, y))
        print(f"[execute] Executed card #{pos} at ({x}, {y})")

    def deck_view(self) -> str:
        """Hold Tab, screenshot, return path."""
        import game_utils
        return game_utils.hold_tab_screenshot()

    # -- Persistence --

    def save(self, path: str = SESSION_FILE):
        _acquire_session_lock(path)
        try:
            data = self.to_game_state().to_dict()
            data["used_abilities"] = list(self.used_abilities)
            data["lilis_batch_index"] = self.lilis_batch_index
            data["lilis_nights_resolved"] = self.lilis_nights_resolved
            data["pending_lilis_nights"] = self.pending_lilis_nights
            data["druid_reset_generations"] = {
                str(position): generation
                for position, generation in self.druid_reset_generations.items()
            }
            data["druid_pending_activations"] = {
                str(position): copy.deepcopy(token)
                for position, token in self.druid_pending_activations.items()
            }
            data["jester_reset_generations"] = {
                str(position): generation
                for position, generation in self.jester_reset_generations.items()
            }
            data["jester_pending_activations"] = {
                str(position): copy.deepcopy(token)
                for position, token in self.jester_pending_activations.items()
            }

            tmp_path = f"{path}.tmp.{os.getpid()}"
            with open(tmp_path, "w") as f:
                json.dump(data, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, path)
            print(f"[save] Session saved to {path}")
        finally:
            _release_session_lock()

    @classmethod
    def load(cls, path: str = SESSION_FILE) -> "GameSession":
        _acquire_session_lock(path)
        try:
            with open(path) as f:
                data = json.load(f)
            state = GameState.from_dict(data)
            session = cls.from_game_state(
                state,
                used_abilities=data.get("used_abilities", []),
                lilis_batch_index=data.get("lilis_batch_index", 0),
                lilis_nights_resolved=data.get("lilis_nights_resolved"),
                pending_lilis_nights=data.get("pending_lilis_nights", 0),
                druid_reset_generations=data.get(
                    "druid_reset_generations",
                    {},
                ),
                druid_pending_activations=data.get(
                    "druid_pending_activations",
                    {},
                ),
                jester_reset_generations=data.get(
                    "jester_reset_generations",
                    {},
                ),
                jester_pending_activations=data.get(
                    "jester_pending_activations",
                    {},
                ),
            )
            print(f"[load] Session loaded from {path}")
            return session
        finally:
            _release_session_lock()


# ============================================================
# Flip Verification
# ============================================================

def _read_board_once_for_flip() -> Optional[list[dict]]:
    """Read the live board for click verification without owning long-lived state."""
    try:
        from memory_reader import get_monitor as _get_monitor
        mon = _get_monitor()
        if mon.is_healthy():
            return mon.get_board()
    except Exception:
        pass

    try:
        from memory_reader import MemoryReader as _MR
        reader = _MR()
        if reader.open():
            try:
                return reader.read_board()
            finally:
                reader.close()
    except Exception:
        pass
    return None


def _position_flipped_in_board(board: Optional[list[dict]], pos: int) -> bool:
    if not board:
        return False
    card = next((c for c in board if c.get("position") == pos), None)
    if not card:
        return False
    return card.get("state") != "Hidden" or bool(card.get("killed_hidden"))


def _wait_position_flipped(pos: int, timeout: float = 1.5) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _position_flipped_in_board(_read_board_once_for_flip(), pos):
            return True
        time.sleep(0.15)
    return False


def _click_flip_card(pos: int, coords: dict[int, tuple[int, int]], label: str,
                     verified: bool = False) -> bool:
    """Click one card during reveal; verified mode retries immediately on a miss."""
    import template_match as _tm

    x, y = coords[pos]
    print(f"  #{pos} at ({x},{y})")
    if verified:
        _tm.safe_click_at(x, y, label)
        if _wait_position_flipped(pos):
            return True
        print(f"  [flip] #{pos} still hidden after first click; retrying before continuing.")
        _tm.safe_click_at(x, y, f"{label}_retry")
        return _wait_position_flipped(pos)

    _tm.fast_click_at(x, y, label)
    return True


def _verify_flips(cards_or_output, expected_positions: list[int], session) -> dict:
    """Check that all targeted cards actually flipped.

    Accepts either:
    - list[dict]: card dicts from memory_reader.read_board()
    - str: legacy stdout output from subprocess (backward compat)

    Returns:
        {
            "flipped": [positions that successfully flipped],
            "blocked": [positions likely blocked by the global Witch quota],
            "failed": [positions that failed to flip (click didn't register)],
            "dead": [positions resolved dead/hidden, never counted as reveals],
            "success": bool (True only when every expected card was verified revealed),
        }
    """
    import re
    expected = list(dict.fromkeys(expected_positions))
    still_hidden = []
    missing = []
    all_hidden = []
    dead = []

    if isinstance(cards_or_output, list):
        # New path: card dicts from read_board()
        cards_by_position = {
            card.get('position'): card
            for card in cards_or_output
            if card.get('position') is not None
        }
        all_hidden = [
            position
            for position, card in cards_by_position.items()
            if card.get('state') == 'Hidden' and not card.get('killed_hidden')
        ]
        missing = [position for position in expected if position not in cards_by_position]
        dead = [
            position
            for position in expected
            if position in cards_by_position
            and (
                cards_by_position[position].get('killed_hidden')
                or cards_by_position[position].get('state') == 'Dead'
            )
        ]
        for card in cards_or_output:
            pos = card.get('position')
            if pos in expected and card.get('state') == 'Hidden' and not card.get('killed_hidden'):
                still_hidden.append(pos)
    else:
        # Legacy path: parse stdout text
        observed = set()
        for line in cards_or_output.splitlines():
            m = re.match(r'^\s*#\s*(\d+)', line)
            if not m:
                continue
            pos = int(m.group(1))
            observed.add(pos)
            if 'Dead' in line:
                if pos in expected:
                    dead.append(pos)
            elif 'Hidden' in line:
                all_hidden.append(pos)
                if pos in expected:
                    still_hidden.append(pos)
        missing = [position for position in expected if position not in observed]

    failed = list(missing)
    flipped = [
        position
        for position in expected
        if position not in still_hidden
        and position not in failed
        and position not in dead
    ]
    blocked = []

    if still_hidden:
        has_witch = session.has_role_in_deck("Witch")
        witch_known_dead = (
            session.is_witch_known_dead()
            if hasattr(session, "is_witch_known_dead")
            else any(
                _normalize_role_name(role) == "Witch"
                for role in getattr(session, "executed_evil_roles", {}).values()
            )
        )
        # Cipher is a global quota. Ordinary duplicate Witch cards contribute
        # only one Start increment, and either real Witch death releases that
        # quota. Until such a death, any sole hidden seat can be blocked,
        # regardless of its position or identity.
        if (
            has_witch
            and not witch_known_dead
            and len(still_hidden) == 1
            and len(all_hidden) == 1
        ):
            blocked = list(still_hidden)
        else:
            failed.extend(still_hidden)

        print()
        print("!" * 60)
        print("  FLIP VERIFICATION FAILED")
        print(f"  Positions still face-down: {still_hidden}")
        print(f"  Click likely didn't register (game unfocused?).")
        if not has_witch:
            print("  No Witch in deck -- this is NOT a Witch block.")
            print("  DO NOT mark as blocked. Re-run: python game_loop.py flip")
        else:
            if blocked:
                print(
                    f"  Witch IS in deck -- #{blocked[0]} is the sole hidden card "
                    "and is likely blocked by the global Witch quota."
                )
            elif witch_known_dead:
                print("  Witch is already known dead -- this is a click failure, not a block.")
            else:
                print("  Witch IS in deck but multiple cards hidden. Likely click failures.")
                print("  Re-run: python game_loop.py flip")
        print("!" * 60)

    if missing:
        print(f"  Memory verification did not return positions {missing}; treating them as failed.")

    return {
        "flipped": flipped,
        "blocked": blocked,
        "failed": list(dict.fromkeys(failed)),
        "dead": list(dict.fromkeys(dead)),
        "success": len(failed) == 0 and len(blocked) == 0 and len(dead) == 0,
    }


def _apply_flip_verification(
    session,
    expected_positions: list[int],
    verify: dict,
    *,
    persist: bool = True,
) -> bool:
    """Atomically project one verified click batch into session reveal state.

    Only memory-confirmed flips enter Baker reveal order. Both click failures
    and Witch-blocked attempts are removed. A confirmed successful retry drops
    that seat's transient block marker; a newly observed block persists it.
    """
    expected = list(dict.fromkeys(expected_positions))
    flipped = set(verify.get("flipped", []))
    blocked = set(verify.get("blocked", []))
    failed = set(verify.get("failed", []))
    dead = set(verify.get("dead", []))
    before_order = list(session.reveal_order)
    before_blocked = list(session.blocked_positions)

    for position in expected:
        if position in flipped:
            if position not in session.reveal_order:
                session.reveal_order.append(position)
            while position in session.blocked_positions:
                session.blocked_positions.remove(position)
            continue

        if position in blocked or position in failed or position in dead:
            while position in session.reveal_order:
                session.reveal_order.remove(position)
        if position in blocked and position not in session.blocked_positions:
            session.blocked_positions.append(position)

    changed = (
        before_order != session.reveal_order
        or before_blocked != session.blocked_positions
    )
    if changed and persist:
        session.save()
    return changed


# ============================================================
# CLI
# ============================================================

def _parse_role_list(spec: str) -> list[str]:
    """Parse 'knitter,scout,enlightened' into list of canonical role names.

    Case-insensitive and accepts underscores or spaces. Unknown tokens
    pass through as Title Case so downstream warnings still fire.
    """
    if not spec or spec.lower() == "none":
        return []
    from knowledge_base import CARDS_BY_NAME
    canonical_by_key = {
        name.lower().replace(" ", "_"): name for name in CARDS_BY_NAME
    }
    out = []
    for raw in spec.split(","):
        token = raw.strip()
        if not token:
            continue
        key = token.lower().replace(" ", "_")
        out.append(canonical_by_key.get(key, token.replace("_", " ").title()))
    return out


def _parse_pd_ability_result_from_memory(
    card: dict,
    *,
    ability_pos: int,
    expected_target: int,
    n_cards: int,
) -> tuple[Optional[dict], Optional[str]]:
    """Parse PD's exact public result while honoring the UI/memory boundary.

    The newest coherent acted-info reference must begin with the character the
    automation clicked. The visible speech text supplies the status and any
    revealed position. A native self-check can retain a hidden second reference
    even though it displays clean; this parser deliberately ignores that
    reference.
    """
    import re

    if not 1 <= ability_pos <= n_cards:
        return None, (
            f"Plague Doctor position #{ability_pos} is outside 1..{n_cards}"
        )
    if not 1 <= expected_target <= n_cards:
        return None, (
            f"Plague Doctor target #{expected_target} is outside 1..{n_cards}"
        )

    clue_raw = card.get('clue_text')
    clue = clue_raw.strip() if isinstance(clue_raw, str) else ''
    infos = card.get('acted_infos')
    latest = (
        infos[-1]
        if isinstance(infos, list) and infos and isinstance(infos[-1], dict)
        else None
    )
    if latest is None or latest.get('desc') != clue_raw:
        return None, (
            "Plague Doctor result has no newest coherent acted-info event"
        )
    targets = latest.get('targets', [])
    if not isinstance(targets, list):
        return None, "Plague Doctor result references are malformed"
    if not targets:
        return None, "Plague Doctor result has no recorded picked target"
    if any(not isinstance(position, int) or not 1 <= position <= n_cards
           for position in targets):
        return None, (
            f"Plague Doctor memory references must all be within 1..{n_cards}: "
            f"{targets}"
        )
    if targets[0] != expected_target:
        return None, (
            f"Plague Doctor picked-target mismatch: clicked #{expected_target}, "
            f"memory recorded #{targets[0]}"
        )

    corrupted = re.fullmatch(
        r'#\s*(\d+)\s+is\s+Evil\s*#\s*(\d+)\s+is\s+Corrupted',
        clue,
        re.IGNORECASE,
    )
    if corrupted:
        evil_revealed = int(corrupted.group(1))
        clue_target = int(corrupted.group(2))
        if expected_target == ability_pos:
            return None, (
                "Native Plague Doctor self-check cannot display a Corrupted result"
            )
        if not 1 <= evil_revealed <= n_cards or not 1 <= clue_target <= n_cards:
            return None, (
                f"Plague Doctor speech positions must be within 1..{n_cards}"
            )
        if clue_target != expected_target:
            return None, (
                f"Plague Doctor clue-target mismatch: clicked #{expected_target}, "
                f"speech named #{clue_target}"
            )
        if len(targets) != 2:
            return None, (
                "Plague Doctor Corrupted result must contain exactly the "
                "picked and revealed character references"
            )
        if targets[1] != evil_revealed:
            return None, (
                f"Plague Doctor revealed-position mismatch: speech named "
                f"#{evil_revealed}, memory recorded #{targets[1]}"
            )
        return {
            "target": expected_target,
            "is_corrupted": True,
            "evil_revealed": evil_revealed,
        }, None

    clean = re.fullmatch(
        r'#\s*(\d+)\s+is\s+Not\s+Corrupted',
        clue,
        re.IGNORECASE,
    )
    if clean:
        clue_target = int(clean.group(1))
        if not 1 <= clue_target <= n_cards:
            return None, (
                f"Plague Doctor speech position must be within 1..{n_cards}"
            )
        if clue_target != expected_target:
            return None, (
                f"Plague Doctor clue-target mismatch: clicked #{expected_target}, "
                f"speech named #{clue_target}"
            )
        # Only self can carry a hidden non-null result reference while the
        # public formatter still displays clean. Ordinary clean callbacks
        # append null, which memory_reader intentionally omits.
        target_is_self = expected_target == ability_pos
        if not target_is_self and len(targets) != 1:
            return None, (
                "Non-self Plague Doctor clean result must contain only the "
                "picked character reference"
            )
        if target_is_self and len(targets) > 2:
            return None, "Plague Doctor result contains too many character references"
        return {
            "target": expected_target,
            "is_corrupted": False,
            "evil_revealed": None,
        }, None

    return None, f"Unrecognized Plague Doctor result text: {clue!r}"


def _parse_druid_result_from_memory(
    card: dict,
    *,
    n_cards: int,
    expected_targets: Optional[list[int]] = None,
) -> tuple[Optional[CardInfo], Optional[str]]:
    """Validate and preserve the complete append-only Librarian callback list.

    The native list can contain one initial passive empty record, public Druid
    results, exact Rambler replacements, and a foreign real-path callback
    immediately before the public raw Druid callback. Raw memory proves public
    callback order, but never activation grouping or reveal boundaries; those
    are certified by ``GameSession.add_card``.
    """
    position = card.get("position")
    if type(position) is not int or not 1 <= position <= n_cards:
        return None, f"Druid position {position!r} is outside 1..{n_cards}"

    raw_clue = card.get("clue_text")
    if raw_clue is None:
        clue = ""
    elif isinstance(raw_clue, str):
        clue = raw_clue
    else:
        return None, "Druid savedAct text must be a string or null"

    raw_infos = card.get("acted_infos")
    if raw_infos is None:
        return None, "Druid acted_infos history is unreadable"
    if not isinstance(raw_infos, list):
        return None, "Druid acted_infos must be an array"
    if not raw_infos:
        if clue:
            return None, "Druid result has no acted-info record"
        return card_no_info(position, "Druid"), None

    callbacks: list[dict] = []
    passive_count = 0
    saw_action_event = False
    for index, event in enumerate(raw_infos):
        if not isinstance(event, dict):
            return None, f"Druid acted_infos[{index}] must be an object"
        desc = event.get("desc")
        refs = event.get("targets")

        if (desc is None or desc == "") and refs is None:
            if saw_action_event or passive_count:
                return None, (
                    "Druid passive empty event must occur at most once and "
                    "before every action result"
                )
            passive_count += 1
            continue
        if not isinstance(desc, str):
            return None, f"Druid acted_infos[{index}].desc must be a string"
        if not desc:
            return None, f"Druid acted_infos[{index}] has an empty action text"
        try:
            public_refs = _validate_callback_references(
                refs,
                n_cards=n_cards,
                label=f"Druid acted_infos[{index}]",
            )
        except ValueError as exc:
            return None, str(exc)

        shut_up_target = _parse_shut_up_target_text(desc, n_cards=n_cards)
        if shut_up_target is not None:
            expected_desc = _druid_shut_up_text(shut_up_target)
            if desc != expected_desc or public_refs != [shut_up_target]:
                return None, (
                    "Druid Rambler event must use the exact public text and "
                    f"single matching reference {expected_desc!r}"
                )
            saw_action_event = True
            callbacks.append({
                "event_kind": "rambler_interruption",
                "text": desc,
                "references": [shut_up_target],
                "shut_up_target": shut_up_target,
            })
            continue
        if _looks_like_shut_up_text(desc):
            return None, (
                "Druid Rambler event must use exact '#R\\nshut up!' text"
            )

        parsed = _parse_druid_native_text(desc)
        if parsed is not None:
            displayed_targets, found_outcast = parsed
            try:
                click_order = _validate_current_druid_targets(
                    public_refs,
                    n_cards=n_cards,
                )
            except ValueError as exc:
                return None, str(exc)
            if sorted(click_order) != displayed_targets:
                return None, (
                    f"Druid history entry {index} target mismatch: speech named "
                    f"{displayed_targets}, references were {click_order}"
                )
            callbacks.append({
                "event_kind": "druid_result",
                "text": desc,
                "references": click_order,
                "targets": click_order,
                "found_outcast": found_outcast,
            })
            saw_action_event = True
            continue

        # A real-path callback can belong to a different apparent role before
        # raw Librarian emits the second record. Preserve it opaquely; text in
        # either Druid/Rambler family is malformed rather than foreign.
        druid_sentence_prefix = re.match(
            r"\s*Among\s+#",
            desc,
            re.IGNORECASE,
        ) is not None
        displayed_ids = re.findall(r"#\s*\d+", desc)
        druid_result_clause = (
            re.search(
                r"\bthere\s+(?:is|was)\s*:",
                desc,
                re.IGNORECASE,
            )
            or re.search(
                r"\bthere\s+(?:are|were)\b[\s\S]*\bOutcasts?\b",
                desc,
                re.IGNORECASE,
            )
        )
        if (
            (
                druid_sentence_prefix
                and len(displayed_ids) == 3
                and druid_result_clause
            )
            or re.match(r"\s*#\s*\d+.*\bshut\b", desc, re.IGNORECASE)
        ):
            return None, f"Unrecognized Druid acted-info text: {desc!r}"
        callbacks.append({
            "event_kind": "opaque_real",
            "text": desc,
            "references": public_refs,
        })
        saw_action_event = True

    latest = raw_infos[-1]
    latest_desc = latest.get("desc") if isinstance(latest, dict) else None
    expected_saved_act = latest_desc or ""
    if clue != expected_saved_act:
        return None, (
            "Druid savedAct does not match the newest acted-info text: "
            f"{clue!r} != {expected_saved_act!r}"
        )

    if not callbacks:
        return card_no_info(position, "Druid"), None

    latest_callback = callbacks[-1]
    if latest_callback["event_kind"] == "opaque_real":
        # Native real dispatch has settled, but the raw callback may append in
        # the next read. There is no public Druid alias to publish yet.
        return None, None
    if latest_callback["event_kind"] == "rambler_interruption":
        result = _card_current_druid_interruption(
            position,
            latest_callback["shut_up_target"],
            n_cards=n_cards,
        )
    elif latest_callback["event_kind"] == "druid_result":
        if expected_targets is not None:
            try:
                clicked = _validate_current_druid_targets(
                    list(expected_targets),
                    n_cards=n_cards,
                )
            except (TypeError, ValueError) as exc:
                return None, str(exc)
            if clicked != latest_callback["targets"]:
                return None, (
                    "Druid clicked/reference mismatch: clicked "
                    f"{clicked}, newest references were "
                    f"{latest_callback['targets']}"
                )
        result = card_druid(
            position,
            latest_callback["targets"],
            latest_callback["found_outcast"],
            info_text=latest_callback["text"],
            druid_variant=_PUBLIC_CURRENT_VARIANT,
        )
    else:
        return None, "Druid history has no coherent current result"

    # Transient provenance for the session join. CardInfo serialization ignores
    # this attribute, and add_card removes it after reconciliation.
    result._druid_raw_callbacks = copy.deepcopy(callbacks)
    return result, None


def _classify_druid_auto_capture(
    existing: Optional[CardInfo],
    parsed: CardInfo,
    *,
    n_cards: int,
    reveal_order: list[int],
    baker_rule_version: Optional[str],
    rambler_observations: list[dict],
) -> tuple[str, Optional[str]]:
    """Classify one validated raw Druid capture as stale, update, or error.

    ResetAfterNight keeps native acted history append-only. ``auto_card`` may
    therefore see the same last result after the session has reset the use bit;
    that is a wait/no-op, not another Druid observation. Reveal counts remain
    trusted only from the persisted session history.
    """
    if (
        existing is None
        or not isinstance(existing.info_parsed, dict)
        or existing.info_parsed.get("druid_variant")
        != _PUBLIC_CURRENT_VARIANT
    ):
        return "update", None

    try:
        persisted = _druid_callback_ledger(
            existing.info_parsed,
            actor_position=existing.position,
            n_cards=n_cards,
            reveal_order=reveal_order,
            baker_rule_version=baker_rule_version,
        )
    except ValueError as exc:
        return "error", f"Persisted Druid history is malformed: {exc}"
    try:
        _validate_druid_rambler_sync(
            persisted,
            speaker_position=existing.position,
            rambler_observations=rambler_observations,
        )
    except ValueError as exc:
        return "error", str(exc)

    raw_callbacks = getattr(parsed, "_druid_raw_callbacks", None)
    if raw_callbacks is None:
        return "error", "Druid capture has no authenticated raw history"
    raw_signatures = [_druid_callback_signature(event) for event in raw_callbacks]
    persisted_signatures = [
        _druid_callback_signature(event) for event in persisted
    ]
    if raw_signatures == persisted_signatures:
        return "stale", None
    if (
        len(raw_signatures) < len(persisted_signatures)
        or raw_signatures[:len(persisted_signatures)] != persisted_signatures
    ):
        return (
            "error",
            "Raw Druid callback history does not preserve the persisted prefix",
        )
    return "update", None


def _parse_jester_result_from_memory(
    card: dict,
    *,
    n_cards: int,
    expected_targets: Optional[list[int]] = None,
) -> tuple[Optional[CardInfo], Optional[str]]:
    """Validate and preserve Juggler's complete append-only callback list."""
    position = card.get("position")
    if type(position) is not int or not 1 <= position <= n_cards:
        return None, f"Jester position {position!r} is outside 1..{n_cards}"

    raw_clue = card.get("clue_text")
    if raw_clue is None:
        clue = ""
    elif isinstance(raw_clue, str):
        clue = raw_clue
    else:
        return None, "Jester savedAct text must be a string or null"

    raw_infos = card.get("acted_infos")
    if raw_infos is None:
        return None, "Jester acted_infos history is unreadable"
    if not isinstance(raw_infos, list):
        return None, "Jester acted_infos must be an array"
    if not raw_infos:
        if clue:
            return None, "Jester result has no acted-info record"
        result = _card_current_jester_no_info(position)
        result._jester_raw_callbacks = []
        return result, None

    callbacks: list[dict] = []
    passive_count = 0
    saw_action_event = False
    for index, event in enumerate(raw_infos):
        if not isinstance(event, dict):
            return None, f"Jester acted_infos[{index}] must be an object"
        desc = event.get("desc")
        refs = event.get("targets")
        if (desc is None or desc == "") and refs is None:
            if saw_action_event or passive_count:
                return None, (
                    "Jester passive empty event must occur at most once and "
                    "before every action result"
                )
            passive_count += 1
            continue
        if not isinstance(desc, str):
            return None, f"Jester acted_infos[{index}].desc must be a string"
        if not desc:
            return None, f"Jester acted_infos[{index}] has an empty action text"
        try:
            public_refs = _validate_callback_references(
                refs,
                n_cards=n_cards,
                label=f"Jester acted_infos[{index}]",
            )
        except ValueError as exc:
            return None, str(exc)

        shut_up_target = _parse_shut_up_target_text(desc, n_cards=n_cards)
        if shut_up_target is not None:
            expected_desc = _jester_shut_up_text(shut_up_target)
            if desc != expected_desc or public_refs != [shut_up_target]:
                return None, (
                    "Jester Rambler event must use the exact public text and "
                    f"single matching reference {expected_desc!r}"
                )
            callbacks.append({
                "event_kind": "rambler_interruption",
                "text": desc,
                "references": [shut_up_target],
                "shut_up_target": shut_up_target,
            })
            saw_action_event = True
            continue
        if _looks_like_shut_up_text(desc):
            return None, "Jester Rambler event must use exact '#R\\nshut up!' text"

        parsed_text = _parse_jester_native_text(desc)
        if parsed_text is not None:
            displayed_targets, evil_count = parsed_text
            try:
                reference_ids = _validate_jester_reference_ids(
                    public_refs,
                    n_cards=n_cards,
                )
            except ValueError as exc:
                return None, str(exc)
            if sorted(reference_ids) != displayed_targets:
                return None, (
                    f"Jester history entry {index} target mismatch: speech named "
                    f"{displayed_targets}, references were {reference_ids}"
                )
            callbacks.append({
                "event_kind": "jester_result",
                "text": desc,
                "references": reference_ids,
                "evil_count": evil_count,
            })
            saw_action_event = True
            continue

        # A different real role can emit before raw Jester. Preserve that
        # callback opaquely, but never use opaque storage to launder a malformed
        # near-Jester or near-Rambler sentence.
        near_jester = (
            re.match(r"\s*Among\s*:", desc, re.IGNORECASE) is not None
            and re.search(r"\bEvils?\b", desc, re.IGNORECASE) is not None
        )
        if near_jester:
            return None, f"Unrecognized Jester acted-info text: {desc!r}"
        callbacks.append({
            "event_kind": "opaque_real",
            "text": desc,
            "references": public_refs,
        })
        saw_action_event = True

    latest = raw_infos[-1]
    latest_desc = latest.get("desc") if isinstance(latest, dict) else None
    expected_saved_act = latest_desc or ""
    if clue != expected_saved_act:
        return None, (
            "Jester savedAct does not match the newest acted-info text: "
            f"{clue!r} != {expected_saved_act!r}"
        )
    if not callbacks:
        result = _card_current_jester_no_info(position)
        result._jester_raw_callbacks = []
        return result, None

    latest_callback = callbacks[-1]
    if latest_callback["event_kind"] == "opaque_real":
        return None, None
    if latest_callback["event_kind"] == "rambler_interruption":
        result = _card_current_jester_interruption(
            position,
            latest_callback["shut_up_target"],
            n_cards=n_cards,
        )
    elif latest_callback["event_kind"] == "jester_result":
        physical_targets = None
        if expected_targets is not None:
            try:
                physical_targets = _validate_current_jester_targets(
                    list(expected_targets),
                    n_cards=n_cards,
                )
            except (TypeError, ValueError) as exc:
                return None, str(exc)
        else:
            try:
                physical_targets = _validate_current_jester_targets(
                    latest_callback["references"],
                    n_cards=n_cards,
                )
            except ValueError:
                # Native IDs alone cannot identify distinct physical objects
                # when two selected Characters display the same ID. Keep the
                # raw capture for a pending click token to bind in add_card.
                physical_targets = None
        result = (
            CardInfo(
                position,
                "Jester",
                info_text=latest_callback["text"],
                info_parsed={
                    "targets": physical_targets,
                    "evil_count": latest_callback["evil_count"],
                    "jester_variant": _PUBLIC_CURRENT_VARIANT,
                },
            )
            if physical_targets is not None
            else _card_current_jester_no_info(position)
        )
    else:
        return None, "Jester history has no coherent current result"

    result._jester_raw_callbacks = copy.deepcopy(callbacks)
    return result, None


def _classify_jester_auto_capture(
    existing: Optional[CardInfo],
    parsed: CardInfo,
    *,
    n_cards: int,
    reveal_order: list[int],
    baker_rule_version: Optional[str],
    rambler_observations: list[dict],
) -> tuple[str, Optional[str]]:
    """Classify a raw Jester capture as stale, appended, or conflicting."""
    raw_callbacks = getattr(parsed, "_jester_raw_callbacks", None)
    if (
        existing is None
        or not isinstance(existing.info_parsed, dict)
        or existing.info_parsed.get("jester_variant") != _PUBLIC_CURRENT_VARIANT
    ):
        return "update", None
    if raw_callbacks is None:
        if set(parsed.info_parsed or {}) == {"jester_variant"}:
            return "stale", None
        return "error", "Jester capture has no authenticated raw history"
    if set(existing.info_parsed) == {"jester_variant"}:
        return "update", None
    try:
        persisted = _jester_callback_ledger(
            existing.info_parsed,
            actor_position=existing.position,
            n_cards=n_cards,
            reveal_order=reveal_order,
            baker_rule_version=baker_rule_version,
        )
        _validate_jester_rambler_sync(
            persisted,
            speaker_position=existing.position,
            rambler_observations=rambler_observations,
        )
    except ValueError as exc:
        return "error", f"Persisted Jester history is malformed: {exc}"
    raw_signatures = [_jester_callback_signature(event) for event in raw_callbacks]
    persisted_signatures = [
        _jester_callback_signature(event) for event in persisted
    ]
    if raw_signatures == persisted_signatures:
        return "stale", None
    if (
        len(raw_signatures) < len(persisted_signatures)
        or raw_signatures[:len(persisted_signatures)] != persisted_signatures
    ):
        return (
            "error",
            "Raw Jester callback history does not preserve the persisted prefix",
        )
    return "update", None


def _parse_fortune_teller_result_from_memory(
    card: dict,
    *,
    n_cards: int,
    expected_targets: Optional[list[int]] = None,
) -> tuple[Optional[CardInfo], Optional[str]]:
    """Parse every current-build Fortune Teller result coherently.

    Native ``actedInfos`` is append-only and chronological. Each normal event
    stores two ascending references and the exact matching sentence; savedAct
    is the newest event. Rambler replacement events are not Fortune Teller
    observations and are skipped exactly as they are for Judge history.
    """
    position = card.get("position")
    if type(position) is not int or not 1 <= position <= n_cards:
        return None, (
            f"Fortune Teller position {position!r} is outside 1..{n_cards}"
        )

    clue = card.get("clue_text")
    if not isinstance(clue, str) or not clue:
        return None, "Fortune Teller result has no exact savedAct text"

    raw_infos = card.get("acted_infos")
    if not isinstance(raw_infos, list):
        return None, "Fortune Teller acted_infos must be an array"
    if not raw_infos:
        return None, "Fortune Teller result has no acted-info record"

    observations: list[dict] = []
    latest_was_interruption = False
    for index, info in enumerate(raw_infos):
        if not isinstance(info, dict):
            return None, f"Fortune Teller acted_infos[{index}] must be an object"
        desc = info.get("desc")
        if not isinstance(desc, str):
            return None, (
                f"Fortune Teller acted_infos[{index}].desc must be a string"
            )
        shut_up_target = _parse_shut_up_target_text(desc, n_cards=n_cards)
        if shut_up_target is not None:
            latest_was_interruption = index == len(raw_infos) - 1
            continue

        try:
            targets = _fortune_teller_targets(
                info.get("targets"),
                label=f"Fortune Teller acted_infos[{index}]",
                n_cards=n_cards,
                require_ascending=True,
            )
        except ValueError as exc:
            return None, str(exc)

        match = re.fullmatch(
            r"Is #(\d+) or #(\d+) Evil\?: (False|True)",
            desc,
        )
        if match is None:
            return None, f"Unrecognized Fortune Teller acted-info text: {desc!r}"
        speech_targets = [int(match.group(1)), int(match.group(2))]
        if speech_targets != targets:
            return None, (
                f"Fortune Teller history entry {index} target mismatch: "
                f"speech named {speech_targets}, references were {targets}"
            )
        has_evil = match.group(3) == "True"
        observations.append(
            {"targets": targets, "has_evil": has_evil, "text": desc}
        )

    latest = raw_infos[-1]
    latest_desc = latest.get("desc") if isinstance(latest, dict) else None
    if clue != latest_desc:
        return None, (
            "Fortune Teller savedAct does not match the newest acted-info "
            f"text: {clue!r} != {latest_desc!r}"
        )
    if latest_was_interruption:
        return None, (
            "Newest Fortune Teller event was replaced by Rambler; parse the "
            "interruption surface instead"
        )
    if not observations:
        return None, "Fortune Teller history has no normal result"

    latest_observation = observations[-1]
    if expected_targets is not None:
        try:
            clicked_targets = _fortune_teller_targets(
                list(expected_targets),
                label="Fortune Teller clicked pair",
                n_cards=n_cards,
                require_ascending=False,
            )
        except (TypeError, ValueError) as exc:
            return None, str(exc)
        clicked_targets.sort()
        if latest_observation["targets"] != clicked_targets:
            return None, (
                "Fortune Teller clicked/reference mismatch: clicked "
                f"{clicked_targets}, newest references were "
                f"{latest_observation['targets']}"
            )

    return card_fortune_teller(
        position,
        latest_observation["targets"],
        latest_observation["has_evil"],
        info_text=latest_observation["text"],
        observations=observations,
    ), None


def _parse_judge_result_from_memory(
    card: dict,
    *,
    expected_target: int,
    n_cards: int,
) -> tuple[Optional[CardInfo], Optional[str]]:
    """Parse the shipped Judge2 result and cross-check its public reference.

    Judge2 emits exactly one ``ActedInfo`` reference: the picked character.
    Its two public strings are ``#X is\nsaying Truth`` and
    ``#X is\nLying``.  Treat anything else as recovery-worthy instead of
    silently turning an unfamiliar clue into a "Truth" observation.
    """
    import re

    if not 1 <= expected_target <= n_cards:
        return None, f"Judge target #{expected_target} is outside 1..{n_cards}"

    clue = (card.get('clue_text') or '').strip()
    raw_infos = card.get('acted_infos')
    if raw_infos is not None and not isinstance(raw_infos, list):
        return None, "Judge acted_infos must be an array"
    infos = raw_infos or []
    if not infos:
        return None, "Judge result has no acted-info record"

    observations = []
    for index, info in enumerate(infos):
        if not isinstance(info, dict):
            return None, f"Judge acted_infos[{index}] must be an object"
        raw_desc = info.get('desc')
        if raw_desc is not None and not isinstance(raw_desc, str):
            return None, f"Judge acted_infos[{index}].desc must be a string"
        desc = (raw_desc or '').strip()
        if _parse_shut_up_target_text(desc, n_cards=n_cards) is not None:
            # Rambler2 replaces both the description and reference list.  This
            # history entry contains no Judge target/result to validate.
            continue
        targets = info.get('targets')
        if not isinstance(targets, list):
            return None, (
                f"Judge acted_infos[{index}].targets must be an array"
            )
        if len(targets) != 1:
            return None, (
                "Each Judge result must contain exactly one picked-character "
                f"reference; history entry {index} has {targets}"
            )
        recorded_target = targets[0]
        if (
            type(recorded_target) is not int
            or not 1 <= recorded_target <= n_cards
        ):
            return None, (
                f"Judge memory reference must be within 1..{n_cards}: "
                f"{recorded_target!r}"
            )

        match = re.fullmatch(
            r'#\s*(\d+)\s+is\s+(saying\s+Truth|Lying)',
            desc,
            re.IGNORECASE,
        )
        if not match:
            return None, f"Unrecognized Judge acted-info text: {desc!r}"
        clue_target = int(match.group(1))
        if not 1 <= clue_target <= n_cards:
            return None, f"Judge speech position must be within 1..{n_cards}"
        if clue_target != recorded_target:
            return None, (
                f"Judge history entry {index} target mismatch: speech named "
                f"#{clue_target}, memory recorded #{recorded_target}"
            )
        observations.append({
            "target": recorded_target,
            "is_lying": match.group(2).lower() == 'lying',
        })

    if not observations:
        return None, "Judge result contains only Rambler shut-up interference"

    newest = observations[-1]
    recorded_target = newest["target"]
    if recorded_target != expected_target:
        return None, (
            f"Judge picked-target mismatch: clicked #{expected_target}, "
            f"latest memory record is #{recorded_target}"
        )

    latest_desc = (infos[-1].get('desc') or '').strip()
    if clue != latest_desc:
        return None, (
            "Judge saved speech does not match the latest acted-info text: "
            f"{clue!r} != {latest_desc!r}"
        )

    info_parsed = dict(newest)
    if len(observations) > 1:
        info_parsed["observations"] = observations
    return CardInfo(
        card['position'],
        "Judge",
        info_text=clue,
        info_parsed=info_parsed,
    ), None


def _parse_shut_up_target_text(
    text,
    *,
    n_cards: Optional[int] = None,
) -> Optional[int]:
    """Parse the exact public Rambler2 replacement sentence."""
    import re

    if not isinstance(text, str):
        return None
    match = re.fullmatch(
        r'\s*#\s*(\d+)\s+shut\s+up\s*!?\s*',
        text,
        re.IGNORECASE,
    )
    if not match:
        return None
    target = int(match.group(1))
    if target <= 0 or (n_cards is not None and target > n_cards):
        return None
    return target


def _looks_like_shut_up_text(text) -> bool:
    """Whether text belongs to the public shut-up sentence family."""
    import re

    return isinstance(text, str) and re.search(
        r'\bshut\s+up\b',
        text,
        re.IGNORECASE,
    ) is not None


def _rambler_interruption_from_memory(
    card: dict,
    *,
    n_cards: Optional[int] = None,
) -> tuple[Optional[tuple[int, str]], Optional[str]]:
    """Read one current Rambler2 replacement from the newest native event.

    ``savedAct`` and the latest ``ActedInfo.desc`` are two views of the same
    public output.  Treat either missing history or disagreement as pending
    recovery; older entries are history, never the current clue surface.
    """
    raw_clue = card.get('clue_text')
    clue = raw_clue.strip() if isinstance(raw_clue, str) else ''
    raw_infos = card.get('acted_infos')
    infos = raw_infos if isinstance(raw_infos, list) else []
    latest = infos[-1] if infos else None
    latest_desc = (
        latest.get('desc').strip()
        if isinstance(latest, dict)
        and isinstance(latest.get('desc'), str)
        else ''
    )

    if not (
        _looks_like_shut_up_text(clue)
        or _looks_like_shut_up_text(latest_desc)
    ):
        return None, None
    if not clue:
        return None, (
            "Rambler shut-up observation has no nonempty savedAct text to "
            "agree with the latest acted-info record"
        )
    if not isinstance(raw_infos, list) or not raw_infos:
        return None, (
            "Rambler shut-up observation has no current acted-info history; "
            "wait for memory to settle or enter it manually"
        )
    if not isinstance(latest, dict):
        return None, "Latest acted-info record is malformed"
    if not latest_desc:
        return None, "Latest acted-info record has no description"
    if clue != latest_desc:
        return None, (
            "Rambler savedAct does not match the newest acted-info text: "
            f"{clue!r} != {latest_desc!r}"
        )

    target = _parse_shut_up_target_text(clue, n_cards=n_cards)
    if target is None:
        return None, (
            "Malformed or out-of-range Rambler shut-up observation: "
            f"{clue!r}"
        )
    refs = latest.get('targets')
    if (
        not isinstance(refs, list)
        or len(refs) != 1
        or type(refs[0]) is not int
        or refs[0] != target
    ):
        return None, (
            "Rambler shut-up acted-info must reference exactly its displayed "
            f"target #{target}; got {refs!r}"
        )
    return (target, clue), None


def _card_from_rambler_interruption(
    card: dict,
    *,
    n_cards: Optional[int] = None,
) -> tuple[Optional[CardInfo], Optional[str]]:
    interruption, error = _rambler_interruption_from_memory(
        card,
        n_cards=n_cards,
    )
    if error is not None:
        return None, error
    if interruption is None:
        return None, None
    shut_up_target, interruption_text = interruption
    position = card['position']
    role = card.get('disguise') or _observed_current_role(card) or ''
    role_key = role.lower().replace(' ', '_')
    if role_key in {'archivist', 'gambler'}:
        role = 'Gemcrafter'
        role_key = 'gemcrafter'
    elif role_key in {'librarian', 'rangedempath'}:
        role = 'Druid'
        role_key = 'druid'
    elif role_key in {'acrobat2', 'acrobat', 'athlete'}:
        role = 'Bard'
        role_key = 'bard'
    if role_key == 'jester':
        return card_jester_silenced(
            position,
            shut_up_target=shut_up_target,
            info_text=interruption_text,
        ), None
    return (
        card_shut_up(
            position,
            role,
            shut_up_target,
            info_text=interruption_text,
        ),
        None,
    )


def _rambler_quote_targets(position: int, n_cards: int) -> list[int]:
    """Native Rambler2 Day quote refs: predecessor, then successor."""
    if type(position) is not int or not 1 <= position <= n_cards:
        raise ValueError(f"Rambler position must be within 1..{n_cards}")
    predecessor = n_cards if position == 1 else position - 1
    successor = 1 if position == n_cards else position + 1
    return [predecessor, successor]


def _card_from_rambler_quote(
    card: dict,
    *,
    n_cards: int,
) -> tuple[Optional[CardInfo], Optional[str]]:
    """Capture a current Rambler2 Day quote from one consistent event."""
    position = card.get('position')
    if type(position) is not int or not 1 <= position <= n_cards:
        return None, f"Rambler position {position!r} is outside 1..{n_cards}"

    raw_clue = card.get('clue_text')
    clue = raw_clue.strip() if isinstance(raw_clue, str) else ''
    if not clue:
        return None, (
            "Rambler quote has no nonempty savedAct text; wait for memory to "
            "settle or enter it manually"
        )
    infos = card.get('acted_infos')
    if not isinstance(infos, list) or not infos:
        return None, (
            "Rambler quote has no current acted-info history; wait for memory "
            "to settle or enter it manually"
        )
    latest = infos[-1]
    if not isinstance(latest, dict):
        return None, "Latest Rambler acted-info record is malformed"
    desc = latest.get('desc')
    latest_desc = desc.strip() if isinstance(desc, str) else ''
    if not latest_desc:
        return None, "Latest Rambler acted-info record has no description"
    if clue != latest_desc:
        return None, (
            "Rambler savedAct does not match the newest acted-info text: "
            f"{clue!r} != {latest_desc!r}"
        )

    expected_refs = _rambler_quote_targets(position, n_cards)
    refs = latest.get('targets')
    if (
        not isinstance(refs, list)
        or any(type(ref) is not int for ref in refs)
        or refs != expected_refs
    ):
        return None, (
            "Rambler quote acted-info refs must be circular predecessor then "
            f"successor {expected_refs}; got {refs!r}"
        )
    return card_rambler_quote(position, clue), None


def _card_from_rambler_surface(
    card: dict,
    *,
    n_cards: Optional[int],
) -> tuple[Optional[CardInfo], Optional[str]]:
    """Parse the strict current Rambler surface, if this card has one."""
    interrupted, error = _card_from_rambler_interruption(
        card,
        n_cards=n_cards,
    )
    if interrupted is not None or error is not None:
        return interrupted, error
    role = card.get('disguise') or _observed_current_role(card) or ''
    if role.lower().replace(' ', '_') == 'rambler':
        if n_cards is None:
            return None, (
                "Rambler quote capture requires the board size to validate "
                "its circular neighbor references"
            )
        return _card_from_rambler_quote(card, n_cards=n_cards)
    return None, None


def _baker_claim_from_public_text(
    clue: object,
    *,
    strict: bool,
) -> tuple[Optional[str], Optional[str]]:
    """Decode one Baker Day sentence without conflating Baker/original.

    Current capture accepts only the exact English localization templates
    shipped by the audited asset. Legacy capture remains case-insensitive and
    trims outer whitespace, but still requires the complete sentence and a
    canonical role name.
    """
    if not isinstance(clue, str) or not clue:
        return None, "Baker clue has no nonempty savedAct text"

    text = clue if strict else clue.strip()
    if (text == "I am the original Baker" if strict
            else text.casefold() == "i am the original baker"):
        return "original", None

    flags = 0 if strict else re.IGNORECASE
    match = re.fullmatch(r"I was (a|an) (.+)", text, flags)
    if match is None:
        return None, (
            "Baker clue must be exactly 'I am the original Baker' or "
            "'I was a/an <canonical role>'"
        )
    article = match.group(1)
    claimed_text = match.group(2)
    role_def = get_card(claimed_text)
    if role_def is None or (strict and claimed_text != role_def.name):
        return None, f"Baker clue names unknown/noncanonical role {claimed_text!r}"
    expected_article = "an" if role_def.name[:1] in "AEIOU" else "a"
    if strict and article != expected_article:
        return None, (
            f"Baker clue uses article {article!r} for {role_def.name!r}; "
            f"native output uses {expected_article!r}"
        )
    return role_def.name, None


def _card_from_baker_surface(
    card: dict,
    *,
    baker_rule_version: Optional[str],
) -> tuple[Optional[CardInfo], Optional[str]]:
    """Capture Baker's public Day event, with a legacy-only runtime fallback."""
    role = card.get('disguise') or _observed_current_role(card) or ''
    if role.lower().replace(' ', '_') != 'baker':
        return None, None

    pos = card.get('position')
    raw_clue = card.get('clue_text')
    current = baker_rule_version == BAKER_RULE_VERSION

    if current:
        claim, error = _baker_claim_from_public_text(raw_clue, strict=True)
        if error is not None:
            return None, error

        infos = card.get('acted_infos')
        if not isinstance(infos, list) or not infos:
            return None, (
                "Baker clue has no current acted-info history; wait for memory "
                "to settle or enter it manually"
            )
        latest = infos[-1]
        if not isinstance(latest, dict):
            return None, "Latest Baker acted-info record is malformed"
        latest_desc = latest.get('desc')
        if latest_desc != raw_clue:
            return None, (
                "Baker savedAct does not match the newest acted-info text: "
                f"{raw_clue!r} != {latest_desc!r}"
            )
        refs = latest.get('targets')
        if refs != []:
            return None, (
                "Normal Baker acted-info refs must be exactly empty; "
                f"got {refs!r}"
            )
        return card_baker(pos, claim, info_text=raw_clue), None

    # Archived sessions predate coherent-event provenance. Prefer any complete
    # public sentence, then retain the old runtime-data recovery path. A
    # runtime literal Baker is a real role claim, never the original sentinel.
    if isinstance(raw_clue, str) and raw_clue.strip():
        claim, error = _baker_claim_from_public_text(raw_clue, strict=False)
        if error is None:
            return card_baker(pos, claim, info_text=raw_clue), None

    runtime_data = card.get('runtime_data')
    if runtime_data and runtime_data.get('type') == 'baker':
        original = runtime_data.get('original_role')
        if not original or original == '?':
            return card_baker(pos, 'original'), None
        try:
            return card_baker(pos, original), None
        except ValueError:
            return None, f"Legacy Baker runtime role is unknown: {original!r}"

    if not raw_clue:
        return card_baker(pos, 'original'), None
    return None, None


def _parse_clue_from_memory(
    card: dict,
    *,
    n_cards: Optional[int] = None,
    baker_rule_version: Optional[str] = None,
    fortune_teller_rule_version: Optional[str] = None,
) -> Optional[CardInfo]:
    """Parse memory reader card data into a CardInfo, or None if unparseable.

    Handles passive clues read from savedAct/actedInfos/runtimeData. Active
    results require a newest acted-info event that owns the visible savedAct;
    the native ``act`` field is only a publication gate and the remaining
    pickable-use counter is lifecycle metadata, not result provenance.
    """
    pos = card['position']
    role = card.get('disguise') or _observed_current_role(card) or ''
    clue = card.get('clue_text') or ''
    infos = card.get('acted_infos', [])
    rd = card.get('runtime_data')
    first_info = (
        infos[0]
        if isinstance(infos, list) and infos and isinstance(infos[0], dict)
        else {}
    )
    targets = first_info.get('targets', [])
    role_lower = role.lower().replace(' ', '_')
    if role_lower in {'archivist', 'gambler'}:
        # Current public Gemcrafter binds managed Archivist. Gambler remains a
        # compatibility name from older memory-reader builds. Apply this only
        # after choosing disguise > current_role > true_role above.
        role = 'Gemcrafter'
        role_lower = 'gemcrafter'
    elif role_lower in {'librarian', 'rangedempath'}:
        # Current public Druid binds managed Librarian. RangedEmpath is the
        # older unbound Druid-like implementation, not a Bard implementation.
        role = 'Druid'
        role_lower = 'druid'
    elif role_lower == 'juggler':
        role = 'Jester'
        role_lower = 'jester'
    elif role_lower in {'acrobat2', 'acrobat', 'athlete'}:
        # Current public Bard binds managed Acrobat2. Acrobat and Athlete remain
        # historical reader aliases; RangedEmpath belongs to Druid instead.
        role = 'Bard'
        role_lower = 'bard'
    def current_event_refs() -> Optional[list[int]]:
        """Return refs from the newest coherent native event, if any."""
        latest = (
            infos[-1]
            if isinstance(infos, list) and infos and isinstance(infos[-1], dict)
            else None
        )
        if latest is None or latest.get('desc') != clue:
            return None
        refs = latest.get('targets')
        if not isinstance(refs, list) or any(type(ref) is not int for ref in refs):
            return None
        if any(
            ref <= 0 or (n_cards is not None and ref > n_cards)
            for ref in refs
        ):
            return None
        return refs

    def current_event_has_null_refs() -> bool:
        """Whether the newest coherent event owns a native null ref list."""
        latest = (
            infos[-1]
            if isinstance(infos, list) and infos and isinstance(infos[-1], dict)
            else None
        )
        return (
            latest is not None
            and latest.get('desc') == clue
            and 'targets' in latest
            and latest.get('targets') is None
        )

    # Active-only roles authenticate a public result from the newest coherent
    # event, not from ``act`` (a constructor-default publication gate) or the
    # remaining callback budget. Druid owns a stricter full-ledger parser below.
    ACTIVE_ONLY_ROLES = {
        'dreamer', 'druid', 'fortune_teller', 'jester', 'judge',
        'slayer', 'plague_doctor',
    }
    active_event_is_coherent = current_event_refs() is not None
    if (
        role_lower in ACTIVE_ONLY_ROLES
        and role_lower not in {'druid', 'jester'}
        and not active_event_is_coherent
    ):
        if not isinstance(infos, list):
            return None
        if clue or (isinstance(infos, list) and infos):
            return None
        return card_no_info(pos, role)

    # --- Druid/Librarian: validate its complete append-only reset history. ---
    # Librarian can publish its exact callback before the remaining budget is
    # decremented, so its coherent event is authoritative. The dedicated
    # parser also owns Rambler precedence because an interruption replaces the
    # newest Druid refs.
    if role_lower == 'druid':
        if type(n_cards) is not int or n_cards <= 0:
            return None
        druid_surface, druid_error = _parse_druid_result_from_memory(
            card,
            n_cards=n_cards,
        )
        return druid_surface if druid_error is None else None

    # Current memory_reader snapshots expose the native remaining-use field.
    # That field is a build/schema discriminator only; its value never proves
    # a result. Older synthetic/archive parser callers omit it and retain the
    # unmarked compatibility grammar below.
    current_jester_memory = (
        role_lower == 'jester'
        and type(n_cards) is int
        and n_cards > 0
        and (
            any(
                field in card
                for field in (
                    'pickable_uses_remaining',
                    'act_output_enabled',
                    'pickable_available',
                )
            )
            or _execution_role_key(card.get('true_role')) == 'juggler'
            or _execution_role_key(card.get('current_role')) == 'juggler'
        )
    )
    if current_jester_memory:
        jester_surface, jester_error = _parse_jester_result_from_memory(
            card,
            n_cards=n_cards,
        )
        return jester_surface if jester_error is None else None

    # Rambler replacement text owns the public event. The counter may still be
    # transiently unchanged because append precedes decrement in the callback.
    rambler_surface, rambler_error = _card_from_rambler_surface(
        card,
        n_cards=n_cards,
    )
    if rambler_error is not None:
        return None
    if rambler_surface is not None:
        # The emitted refs were rewritten to [shut_up_target]; they are not
        # the interrupted role's original selections.
        return rambler_surface

    baker_surface, baker_error = _card_from_baker_surface(
        card,
        baker_rule_version=baker_rule_version,
    )
    if baker_error is not None:
        return None
    if baker_surface is not None:
        return baker_surface

    # --- Gemcrafter/Archivist: exact text + identical newest one-card ref. ---
    # Archivist writes no RuntimeData, so unrelated data preserved by an
    # identity move is deliberately ignored here.
    if role_lower == 'gemcrafter':
        good_position = _parse_gemcrafter_native_text(clue)
        if (
            good_position is None
            or type(n_cards) is not int
            or n_cards <= 0
            or type(pos) is not int
            or not 1 <= pos <= n_cards
            or good_position > n_cards
            or current_event_refs() != [good_position]
        ):
            return None
        return card_gemcrafter(
            pos,
            good_position,
            info_text=clue,
            gemcrafter_variant=_PUBLIC_CURRENT_VARIANT,
        )

    if (
        role_lower == 'fortune_teller'
        and fortune_teller_rule_version == FORTUNE_TELLER_RULE_VERSION
    ):
        if n_cards is None:
            return None
        fortune_surface, fortune_error = (
            _parse_fortune_teller_result_from_memory(
                card,
                n_cards=n_cards,
            )
        )
        if fortune_error is not None:
            return None
        return fortune_surface

    # --- Enlightened/Shugenja: exact text, zero refs, matching RuntimeData. ---
    if role_lower in {'enlightened', 'shugenja'}:
        direction = _parse_enlightened_native_text(clue)
        if (
            direction is not None
            and type(n_cards) is int
            and n_cards > 0
            and type(pos) is int
            and 1 <= pos <= n_cards
            and current_event_refs() == []
            and _enlightened_runtime_matches(rd, direction)
        ):
            return card_enlightened(
                pos,
                direction,
                info_text=clue,
                enlightened_variant=_PUBLIC_CURRENT_VARIANT,
            )
        return None

    # --- Alchemist: prefer clue_text (works for Drunk-as-Alchemist too) ---
    # Post-patch clue is "# Corruption/Corrupted around me [Range 2] at
    # start of Round (before Cure)".
    # Alchemist is now immune to Corruption — they never lie themselves, but a
    # Drunk-disguised-as-Alchemist still lies intrinsically. Use displayed
    # value from clue_text since that's what we validate against.
    if role_lower == 'alchemist':
        if re.search(r'\b(?:no|none|zero)\s+(?:one\s+)?(?:was\s+|were\s+)?corrupt(?:ed|ion)', clue, re.IGNORECASE):
            return card_alchemist(pos, 0)
        m = re.search(r'(\d+)\s+corrupt(?:ed|ion)', clue, re.IGNORECASE)
        if not m:
            m = re.search(r'corrupt(?:ed|ion)\s+(?:character|villager)?s?\s*[:=]?\s*(\d+)', clue, re.IGNORECASE)
        if not m:
            # Legacy fallback for old "cured N" wording
            m = re.search(r'cured\s+(\d+)', clue, re.IGNORECASE)
        if m:
            return card_alchemist(pos, int(m.group(1)))
        if rd and rd.get('type') in ('corrupted_around', 'cures'):
            val = rd.get('corrupted_around') if rd.get('type') == 'corrupted_around' else rd.get('cures')
            return card_alchemist(pos, val or 0)

    # --- Knitter: exact current sentence and newest zero-reference event. ---
    if role_lower == 'knitter':
        evil_pairs = _parse_knitter_native_text(clue)
        if (
            evil_pairs is not None
            and type(n_cards) is int
            and n_cards > 0
            and type(pos) is int
            and 1 <= pos <= n_cards
            and evil_pairs <= n_cards
            and current_event_refs() == []
        ):
            return card_knitter(
                pos,
                evil_pairs,
                info_text=clue,
                knitter_variant=_PUBLIC_CURRENT_VARIANT,
            )
        return None

    # --- Confessor: exact public sentence + native null reference list. ---
    # Confessor writes no RuntimeData and does not inspect its actor ID or the
    # board. Hidden alignment/status fields must never manufacture this public
    # evidence; only the newest exact ActedInfo event is authenticated here.
    if role_lower == 'confessor':
        if not clue and not infos:
            # A pre-callback read remains an unmarked placeholder that a later
            # coherent current event may safely replace in auto_card.
            return card_no_info(pos, role)
        dizzy = _parse_confessor_native_text(clue)
        if (
            dizzy is None
            or type(n_cards) is not int
            or n_cards <= 0
            or type(pos) is not int
            or not 1 <= pos <= n_cards
            or not current_event_has_null_refs()
        ):
            return None
        return card_confessor(
            pos,
            dizzy,
            info_text=clue,
            confessor_variant=_PUBLIC_CURRENT_VARIANT,
        )

    # --- Bard/Acrobat2: exact text + native range-reference geometry. ---
    # Acrobat2 writes no RuntimeData, so unrelated data preserved through an
    # identity move is deliberately ignored here.
    if role_lower == 'bard':
        corruption_distance = _parse_bard_native_text(clue)
        if (
            corruption_distance is None
            or type(n_cards) is not int
            or n_cards <= 0
            or type(pos) is not int
            or not 1 <= pos <= n_cards
            or not _valid_current_bard_distance(corruption_distance, n_cards)
            or current_event_refs()
            != _current_bard_refs(pos, corruption_distance, n_cards)
        ):
            return None
        return card_bard(
            pos,
            corruption_distance,
            info_text=clue,
            bard_variant=_PUBLIC_CURRENT_VARIANT,
        )

    # --- Lover: exact Empath text + previous/next Character references. ---
    if role_lower == 'lover':
        if n_cards is None or not 1 <= pos <= n_cards:
            return None
        evil_adjacent = _parse_lover_native_text(clue)
        if (
            evil_adjacent is not None
            and current_event_refs() == _current_lover_refs(pos, n_cards)
        ):
            return card_lover(
                pos,
                evil_adjacent,
                info_text=clue,
                lover_variant=_PUBLIC_CURRENT_VARIANT,
            )
        return None

    # --- Hunter: exact current native sentence + circular range refs. ---
    if role_lower == 'hunter':
        if n_cards is None or not 1 <= pos <= n_cards:
            return None
        m = re.fullmatch(
            r'\s*I\s+am\s+(?:(1)\s+card|((?:0|[2-9]\d*))\s+cards)\s+'
            r'away\s+from\s+closest\s+Evil\s*',
            clue,
            re.IGNORECASE | re.DOTALL,
        )
        if m:
            distance = int(m.group(1) or m.group(2))
            expected_refs = _current_hunter_refs(pos, distance, n_cards)
            if (
                _valid_current_hunter_distance(distance, n_cards)
                and current_event_refs() == expected_refs
            ):
                return card_hunter(
                    pos,
                    distance,
                    info_text=clue,
                    hunter_variant=_PUBLIC_CURRENT_VARIANT,
                )
        return None

    # --- Architect: "Left"/"Right"/"Equal" ---
    if role_lower == 'architect':
        cl = clue.lower()
        if 'left' in cl:
            return card_architect(pos, 'Left')
        if 'right' in cl:
            return card_architect(pos, 'Right')
        if 'equal' in cl:
            return card_architect(pos, 'Equal')

    # --- Empress/Noble: exact text + identical newest ordered refs. ---
    if role_lower in {'empress', 'noble'}:
        empress_targets = _parse_empress_native_text(clue)
        if (
            empress_targets is None
            or type(n_cards) is not int
            or n_cards <= 0
            or type(pos) is not int
            or not 1 <= pos <= n_cards
            or any(target > n_cards for target in empress_targets)
            or current_event_refs() != empress_targets
        ):
            return None
        return card_empress(
            pos,
            empress_targets,
            info_text=clue,
            empress_variant=_PUBLIC_CURRENT_VARIANT,
        )

    # --- Witness: one marked/unmarked target, or exact native NO result ---
    if role_lower == 'witness':
        if not isinstance(targets, list):
            return None
        # ActedInfo.desc is the authoritative fallback when savedAct could not
        # be read. This matters for the native NO branch because it has no refs.
        witness_clue = clue or first_info.get('desc') or ''
        if not isinstance(witness_clue, str):
            return None
        positive = re.fullmatch(
            r'\s*#\s*(\d+)\s+was\s+affected\s+by\s+an\s+Evil\s*',
            witness_clue,
            re.IGNORECASE,
        )
        nobody = re.fullmatch(
            r'\s*NO\s+character\s+was\s+affected\s+by\s+an\s+Evil\s*',
            witness_clue,
            re.IGNORECASE,
        )

        # Native positive ActedInfo always references exactly the displayed
        # card.  Preserve the historical target-only fallback when savedAct is
        # unavailable, but reject contradictory or malformed evidence.
        if len(targets) == 1:
            target = targets[0]
            if not isinstance(target, int) or isinstance(target, bool):
                return None
            if target <= 0 or (n_cards is not None and target > n_cards):
                return None
            if nobody is not None:
                return None
            if positive is not None and int(positive.group(1)) != target:
                return None
            if positive is None and witness_clue.strip():
                return None
            return card_witness(pos, target)

        if targets:
            return None
        if nobody is not None:
            return card_witness(pos, 0)
        # A positive string without its required native reference is unsafe to
        # auto-enter; leave it for manual recovery.
        return None

    # --- Fortune Teller: "Is #X or #Y Evil?: True/False" ---
    if role_lower == 'fortune_teller' and targets:
        has_evil = 'true' in clue.lower()
        return card_fortune_teller(pos, targets, has_evil)

    # --- Jester: newest coherent targets + evil count from clue ---
    if (
        role_lower == 'jester'
        and (jester_targets := current_event_refs())
        and len(jester_targets) == 3
        and len(set(jester_targets)) == 3
    ):
        m = re.search(r'(\d+)\s+(?:of them |are |is )?\s*evil', clue, re.IGNORECASE)
        if m:
            evil_count = int(m.group(1))
            if 0 <= evil_count <= 3:
                return card_jester(pos, jester_targets, evil_count)
            return None
        # "none of them are evil"
        if 'none' in clue.lower() or 'no' in clue.lower():
            return card_jester(pos, jester_targets, 0)

    # --- Bishop: exact current public text + shuffled newest refs. ---
    if role_lower == 'bishop':
        bishop_result = _parse_bishop_native_text(clue)
        if (
            bishop_result is None
            or type(n_cards) is not int
            or n_cards <= 0
            or type(pos) is not int
            or not 1 <= pos <= n_cards
        ):
            return None
        bishop_targets, bishop_types = bishop_result
        if (
            any(target > n_cards for target in bishop_targets)
            or not _bishop_refs_match(
                bishop_targets,
                current_event_refs(),
                n_cards=n_cards,
            )
        ):
            return None
        return card_bishop(
            pos,
            bishop_targets,
            bishop_types,
            info_text=clue,
            bishop_variant=_PUBLIC_CURRENT_VARIANT,
        )

    # --- Judge2: exact public sentence + exactly one picked reference ---
    if role_lower == 'judge' and infos:
        latest_targets = infos[-1].get('targets') or []
        if (
            len(latest_targets) != 1
            or not isinstance(latest_targets[0], int)
            or latest_targets[0] < 1
        ):
            return None
        parsed, _ = _parse_judge_result_from_memory(
            card,
            expected_target=latest_targets[0],
            # The general clue parser does not own board state.  The exact
            # automation path below supplies the real upper bound; this still
            # rejects non-positive references and malformed shapes/text.
            n_cards=n_cards or max(
                target
                for info in infos
                for target in (info.get('targets') or [1])
                if isinstance(target, int)
            ),
        )
        return parsed

    # --- Dreamer: public two-target role pair/Cabbage, then legacy one-target ---
    if role_lower == 'dreamer':
        dreamer_refs = current_event_refs()
        # Public shipped Dreamer stores both selections in click order while
        # formatting their IDs in ascending order. Require both surfaces to
        # name the same two physical positions.
        ambiguous = _parse_ambiguous_among(clue)
        if ambiguous:
            amb_targets, options = ambiguous
            if (
                dreamer_refs is None
                or len(dreamer_refs) != 2
                or len(set(dreamer_refs)) != 2
                or sorted(dreamer_refs) != sorted(amb_targets)
            ):
                return None
            return card_dreamer_ambiguous(
                pos,
                amb_targets,
                options,
                info_text=clue,
            )
        cabbage_targets = _parse_cabbage_between(clue)
        if cabbage_targets:
            if (
                dreamer_refs is None
                or len(dreamer_refs) != 2
                or len(set(dreamer_refs)) != 2
                or sorted(dreamer_refs) != sorted(cabbage_targets)
            ):
                return None
            return card_dreamer_cabbage(
                pos,
                cabbage_targets,
                info_text=clue,
            )

        # Old Dreamer1 form. Anchor the complete clue and capture its own ID;
        # otherwise the unbound Dreamer2 "None of them is <type>" sentence can be
        # mistaken for a one-target role clue.
        m = re.fullmatch(
            r"\s*#\s*(\d+)\s+(?:could\s+be|is)\s*:?\s*"
            r"([A-Za-z][A-Za-z _'-]*?)\s*[.!]?\s*",
            clue,
            re.IGNORECASE | re.DOTALL,
        )
        if m:
            target = int(m.group(1))
            if dreamer_refs != [target]:
                return None
            return card_dreamer(
                pos,
                target,
                m.group(2).strip(),
                info_text=clue,
            )

    # --- Oracle: exact current Investigator positive/sentinel surfaces. ---
    if role_lower == 'oracle':
        if n_cards is None or not 1 <= pos <= n_cards:
            return None
        refs = current_event_refs()
        if clue == "There are no minions":
            if refs == []:
                return _card_oracle_no_minions(
                    pos,
                    info_text=clue,
                    oracle_variant=_PUBLIC_CURRENT_VARIANT,
                )
            return None
        m = re.fullmatch(
            r'#(\d+) or #(\d+) is a ([A-Za-z][A-Za-z _\'-]*)',
            clue,
        )
        if m:
            oracle_targets = [int(m.group(1)), int(m.group(2))]
            minion = get_card(m.group(3))
            if (
                oracle_targets == sorted(oracle_targets)
                and all(1 <= target <= n_cards for target in oracle_targets)
                and refs == oracle_targets
                and minion is not None
                and minion.role.value == "Minion"
                and clue == _oracle_native_text(oracle_targets, minion.name)
            ):
                return card_oracle(
                    pos,
                    oracle_targets,
                    minion.name,
                    info_text=clue,
                    oracle_variant=_PUBLIC_CURRENT_VARIANT,
                )
        return None

    # --- Scout: exact current native numeric/sentinel forms, always no refs. ---
    if role_lower == 'scout':
        if n_cards is None or not 1 <= pos <= n_cards:
            return None
        refs = current_event_refs()
        if refs != []:
            return None
        if re.fullmatch(
            r'\s*There\s+is\s+only\s+1\s+Evil\s*',
            clue,
            re.IGNORECASE | re.DOTALL,
        ):
            return _card_scout_one_evil(
                pos,
                info_text=clue,
                scout_variant=_PUBLIC_CURRENT_VARIANT,
            )
        m = re.fullmatch(
            r'\s*([A-Za-z][A-Za-z _\'-]*?)\s+is\s+'
            r'(?:(1)\s+card|([2-9]\d*)\s+cards)\s+'
            r'away\s+from\s+closest\s+Evil\s*',
            clue,
            re.IGNORECASE | re.DOTALL,
        )
        if m:
            named_role = get_card(m.group(1).strip())
            distance = int(m.group(2) or m.group(3))
            if (
                named_role is not None
                and _valid_current_scout_distance(distance, n_cards)
            ):
                return card_scout(
                    pos,
                    named_role.name,
                    distance,
                    info_text=clue,
                    scout_variant=_PUBLIC_CURRENT_VARIANT,
                )
        return None

    # --- Medium: exact current Lookout result and newest one-target event. ---
    if role_lower == 'medium':
        medium_result = _parse_medium_native_text(clue)
        medium_refs = current_event_refs()
        if (
            medium_result is not None
            and type(n_cards) is int
            and n_cards > 0
            and type(pos) is int
            and 1 <= pos <= n_cards
        ):
            good_position, good_role = medium_result
            if (
                1 <= good_position <= n_cards
                and medium_refs == [good_position]
            ):
                return card_medium(
                    pos,
                    good_position,
                    good_role,
                    info_text=clue,
                    medium_variant=_PUBLIC_CURRENT_VARIANT,
                )
        return None

    # --- Poet: exact current Gossip provider list. ---
    if role_lower == 'poet' and clue:
        # Gossip appends one ActedInfo for each result.  Structured providers
        # below must agree with the newest event, never a stale first event.
        latest_poet_info = (
            infos[-1]
            if isinstance(infos, list) and infos and isinstance(infos[-1], dict)
            else None
        )
        latest_poet_desc = (
            latest_poet_info.get('desc')
            if latest_poet_info is not None
            else None
        )
        latest_poet_targets = (
            latest_poet_info.get('targets')
            if latest_poet_info is not None
            else None
        )

        def poet_refs_match(
            displayed: list[int],
            *,
            order_sensitive: bool = True,
        ) -> bool:
            """Require one exact newest native event and its public refs."""
            if latest_poet_desc != clue or not isinstance(latest_poet_targets, list):
                return False
            if any(type(target) is not int for target in latest_poet_targets):
                return False
            if any(
                target <= 0 or (n_cards is not None and target > n_cards)
                for target in latest_poet_targets
            ):
                return False
            if order_sensitive:
                return latest_poet_targets == displayed
            return (
                len(latest_poet_targets) == len(displayed)
                and len(set(latest_poet_targets)) == len(latest_poet_targets)
                and set(latest_poet_targets) == set(displayed)
            )

        def valid_displayed_targets(
            displayed: list[int],
            *,
            require_sorted: bool = True,
            require_distinct: bool = True,
        ) -> bool:
            return (
                bool(displayed)
                and (
                    not require_distinct
                    or len(set(displayed)) == len(displayed)
                )
                and (not require_sorted or displayed == sorted(displayed))
                and all(
                    target > 0 and (n_cards is None or target <= n_cards)
                    for target in displayed
                )
            )

        # Bishop's native logic ignores the Character forwarded by Poet.  The
        # live bridge still requires the Poet actor itself to be on this board,
        # and authenticates the newest exact event.  Native refs are a shuffled
        # set; the claimed faction types are an independent multiset.
        bishop_result = _parse_bishop_native_text(clue)
        if bishop_result is not None:
            bishop_targets, bishop_types = bishop_result
            if (
                type(n_cards) is int
                and n_cards > 0
                and type(pos) is int
                and 1 <= pos <= n_cards
                and all(target <= n_cards for target in bishop_targets)
                and latest_poet_desc == clue
                and _bishop_refs_match(
                    bishop_targets,
                    latest_poet_targets,
                    n_cards=n_cards,
                )
            ):
                return _card_current_poet(
                    pos,
                    "Bishop",
                    {"targets": bishop_targets, "types": bishop_types},
                    info_text=clue,
                )
            return None

        # Empress has no RuntimeData. Authenticate its exact text and refs
        # before the Shugenja-only stale-runtime guard below.
        empress_targets = _parse_empress_native_text(clue)
        if empress_targets is not None:
            if (
                type(n_cards) is int
                and n_cards > 0
                and type(pos) is int
                and 1 <= pos <= n_cards
                and all(target <= n_cards for target in empress_targets)
                and poet_refs_match(empress_targets)
            ):
                return _card_current_poet(
                    pos,
                    "Empress",
                    {"targets": empress_targets},
                    info_text=clue,
                )
            return None

        # Archivist has no RuntimeData. Authenticate its exact text and newest
        # identical one-card ref before the Shugenja-only runtime guard below.
        gemcrafter_target = _parse_gemcrafter_native_text(clue)
        if gemcrafter_target is not None:
            if (
                type(n_cards) is int
                and n_cards > 0
                and type(pos) is int
                and 1 <= pos <= n_cards
                and gemcrafter_target <= n_cards
                and poet_refs_match([gemcrafter_target])
            ):
                return _card_current_poet(
                    pos,
                    "Gemcrafter",
                    {"good_position": gemcrafter_target},
                    info_text=clue,
                )
            return None

        # Acrobat2 has no RuntimeData. Authenticate Bard before the
        # Shugenja-only stale-runtime guard so an identity move cannot make a
        # coherent Bard event look like Shugenja.
        bard_distance = _parse_bard_native_text(clue)
        if bard_distance is not None:
            if (
                type(n_cards) is int
                and n_cards > 0
                and type(pos) is int
                and 1 <= pos <= n_cards
                and _valid_current_bard_distance(bard_distance, n_cards)
                and poet_refs_match(
                    _current_bard_refs(pos, bard_distance, n_cards)
                )
            ):
                return _card_current_poet(
                    pos,
                    "Bard",
                    {"corruption_distance": bard_distance},
                    info_text=clue,
                )
            return None

        # Shugenja stores its claimed enum on the Poet and emits no refs.
        # RuntimeData is strong corroboration when readable; the exact public
        # event remains sufficient when the runtime object is unavailable.
        enlightened_direction = _parse_enlightened_native_text(clue)
        if enlightened_direction is not None:
            if (
                type(n_cards) is int
                and n_cards > 0
                and type(pos) is int
                and 1 <= pos <= n_cards
                and poet_refs_match([])
                and _enlightened_runtime_matches(rd, enlightened_direction)
            ):
                return _card_current_poet(
                    pos,
                    "Enlightened",
                    {"direction": enlightened_direction},
                    info_text=clue,
                )
            return None
        if isinstance(rd, dict) and rd.get('type') == 'direction':
            # A readable Shugenja runtime object cannot authenticate a stale,
            # malformed, or non-Shugenja newest public event.
            return None

        # Bounty Hunter (retained Poet provider, distinct from Hunter).
        m = re.fullmatch(r'#([1-9]\d*)\nis Evil', clue)
        if m:
            evil_position = int(m.group(1))
            # Native Bounty Hunter text carries no Character references.  The
            # latest ActedInfo still owns the exact sentence, so requiring its
            # empty ref list distinguishes it from stale/ambiguous events.
            if (
                type(n_cards) is int
                and n_cards > 0
                and type(pos) is int
                and 1 <= pos <= n_cards
                and valid_displayed_targets([evil_position])
                and clue == _bounty_hunter_native_text(evil_position)
                and poet_refs_match([])
            ):
                return card_bounty_hunter(
                    pos,
                    evil_position,
                    info_text=clue,
                )

        # Oracle: exact truthful no-Minions sentinel.
        if clue == "There are no minions" and poet_refs_match([]):
            return _card_current_poet(
                pos,
                "Oracle",
                {"no_minions": True},
                info_text=clue,
            )

        # Oracle: two public references and one canonical Minion role. Truth
        # selects its registered-Minion and registered-Good pools
        # independently, so a moved Twin identity can repeat one physical ID.
        m = re.fullmatch(
            r'#(\d+) or #(\d+) is a ([A-Za-z][A-Za-z _\'-]*)',
            clue,
        )
        if m:
            oracle_targets = [int(m.group(1)), int(m.group(2))]
            minion = get_card(m.group(3))
            if (
                valid_displayed_targets(
                    oracle_targets,
                    require_distinct=False,
                )
                and poet_refs_match(oracle_targets)
                and minion is not None
                and minion.role.value == "Minion"
                and clue == _oracle_native_text(oracle_targets, minion.name)
            ):
                return _card_current_poet(
                    pos,
                    "Oracle",
                    {
                        "targets": oracle_targets,
                        "minion_role": minion.name,
                    },
                    info_text=clue,
                )

        # Knitter emits one exact sentence and no references.  Its provider
        # ignores the Character forwarded by Poet.
        evil_pairs = _parse_knitter_native_text(clue)
        if (
            evil_pairs is not None
            and type(n_cards) is int
            and n_cards > 0
            and type(pos) is int
            and 1 <= pos <= n_cards
            and evil_pairs <= n_cards
            and poet_refs_match([])
        ):
            return _card_current_poet(
                pos,
                "Knitter",
                {"evil_pairs": evil_pairs},
                info_text=clue,
            )
        # Lover/Empath stores the copied Poet's physical previous/next
        # neighbors, including duplicate Character references on tiny boards.
        evil_adjacent = _parse_lover_native_text(clue)
        if (
            evil_adjacent is not None
            and n_cards is not None
            and 1 <= pos <= n_cards
            and poet_refs_match(_current_lover_refs(pos, n_cards))
        ):
            return _card_current_poet(
                pos,
                "Lover",
                {"evil_adjacent": evil_adjacent},
                info_text=clue,
            )
        # Scout exact native singular/plural sentence and one-Evil sentinel.
        if re.fullmatch(
            r'\s*There\s+is\s+only\s+1\s+Evil\s*',
            clue,
            re.IGNORECASE | re.DOTALL,
        ) and (
            n_cards is not None
            and 1 <= pos <= n_cards
            and poet_refs_match([])
        ):
            return _card_current_poet(
                pos,
                "Scout",
                {"one_evil": True},
                info_text=clue,
            )
        m_scout = re.fullmatch(
            r'\s*([A-Za-z][A-Za-z _\'-]*?)\s+is\s+'
            r'(?:(1)\s+card|([2-9]\d*)\s+cards)\s+'
            r'away\s+from\s+closest\s+Evil\s*',
            clue,
            re.IGNORECASE | re.DOTALL,
        )
        if (
            m_scout
            and n_cards is not None
            and 1 <= pos <= n_cards
            and poet_refs_match([])
        ):
            candidate = get_card(m_scout.group(1).strip())
            distance = int(m_scout.group(2) or m_scout.group(3))
            if (
                candidate is not None
                and _valid_current_scout_distance(distance, n_cards)
            ):
                return _card_current_poet(
                    pos,
                    "Scout",
                    {
                        "evil_role": candidate.name,
                        "distance": distance,
                    },
                    info_text=clue,
                )

        # Hunter exact native singular/plural sentence.
        m = re.fullmatch(
            r'\s*I\s+am\s+(?:(1)\s+card|((?:0|[2-9]\d*))\s+cards)\s+'
            r'away\s+from\s+'
            r'closest\s+Evil\s*',
            clue,
            re.IGNORECASE | re.DOTALL,
        )
        if m and n_cards is not None and 1 <= pos <= n_cards:
            distance = int(m.group(1) or m.group(2))
            hunter_refs = _current_hunter_refs(pos, distance, n_cards)
            if (
                _valid_current_hunter_distance(distance, n_cards)
                and poet_refs_match(hunter_refs)
            ):
                return _card_current_poet(
                    pos,
                    "Hunter",
                    {"distance": distance},
                    info_text=clue,
                )

        # Medium exact normal and Drunk-reveal forms.  Both carry exactly one
        # matching reference and require a live in-board Poet actor.
        medium_result = _parse_medium_native_text(clue)
        if (
            medium_result is not None
            and type(n_cards) is int
            and n_cards > 0
            and type(pos) is int
            and 1 <= pos <= n_cards
        ):
            good_position, good_role = medium_result
            if (
                valid_displayed_targets([good_position])
                and poet_refs_match([good_position])
            ):
                return _card_current_poet(
                    pos,
                    "Medium",
                    {
                        "good_position": good_position,
                        "good_role": good_role,
                    },
                    info_text=clue,
                )

    # --- No-info roles: these roles NEVER have passive speech bubbles ---
    # Any clue_text is evil fabrication or stale data — ignore it.
    NO_INFO_ROLES = {'wretch', 'bombardier', 'knight', 'doppelganger'}
    if role_lower in NO_INFO_ROLES:
        return card_no_info(pos, role)

    # --- Fallback: no clue and no acted_infos = generic no_info ---
    if not clue and not infos:
        return card_no_info(pos, role)

    return None  # Couldn't parse — needs manual entry


def _parse_card_cli(args: list[str], session=None) -> CardInfo:
    """Parse CLI args for a card builder call.

    Format: <role> <pos> [args...]
    Role aliases: fortune_teller, plague_doctor, dreamer_old, no_info
    """
    role = args[0].lower()
    if role == "knitter" and len(args) != 3:
        raise ValueError("Knitter entry requires exactly one pair count")
    if role == "gemcrafter" and len(args) != 3:
        raise ValueError("Gemcrafter entry requires exactly one Good target")
    if role == "enlightened" and len(args) != 3:
        raise ValueError("Enlightened entry requires exactly one direction")
    if role == "empress" and len(args) != 3:
        raise ValueError("Empress entry requires exactly three targets")
    if role == "bishop" and len(args) != 4:
        raise ValueError("Bishop entry requires targets and matching types")
    if role == "bard" and len(args) != 3:
        raise ValueError("Bard entry requires exactly one corruption distance")
    if role == "confessor" and len(args) != 3:
        raise ValueError("Confessor entry requires exactly one Good/dizzy result")
    if role == "druid" and len(args) != 4:
        raise ValueError(
            "Druid entry requires three targets and one normal result"
        )
    if role in {"shut_up", "shutup"} and len(args) != 4:
        raise ValueError(
            "shut_up entry requires an apparent role and one Rambler target"
        )
    pos = int(args[1])

    if role == "enlightened":
        if session is None:
            raise ValueError("Current Enlightened entry requires session board size")
        if not 1 <= pos <= session.n_cards:
            raise ValueError("Enlightened position is outside the current board")
        direction = _canonical_enlightened_direction(args[2])
        return card_enlightened(
            pos,
            direction,
            enlightened_variant=_PUBLIC_CURRENT_VARIANT,
        )
    elif role == "knitter":
        if session is None:
            raise ValueError("Current Knitter entry requires session board size")
        if not 1 <= pos <= session.n_cards:
            raise ValueError("Knitter position is outside the current board")
        try:
            evil_pairs = int(args[2])
        except (TypeError, ValueError) as exc:
            raise ValueError("Knitter pair count must be an integer") from exc
        if not 0 <= evil_pairs <= session.n_cards:
            raise ValueError("Knitter pair count is outside the current board")
        return card_knitter(
            pos,
            evil_pairs,
            knitter_variant=_PUBLIC_CURRENT_VARIANT,
        )
    elif role == "confessor":
        if session is None:
            raise ValueError("Current Confessor entry requires session board size")
        if not 1 <= pos <= session.n_cards:
            raise ValueError("Confessor position is outside the current board")
        dizzy = _canonical_confessor_claim(args[2])
        return card_confessor(
            pos,
            dizzy,
            confessor_variant=_PUBLIC_CURRENT_VARIANT,
        )
    elif role == "gemcrafter":
        if session is None:
            raise ValueError("Current Gemcrafter entry requires session board size")
        if not 1 <= pos <= session.n_cards:
            raise ValueError("Gemcrafter position is outside the current board")
        try:
            good_position = int(args[2])
        except (TypeError, ValueError) as exc:
            raise ValueError("Gemcrafter Good target must be an integer") from exc
        if not 1 <= good_position <= session.n_cards:
            raise ValueError("Gemcrafter Good target is outside the current board")
        return card_gemcrafter(
            pos,
            good_position,
            gemcrafter_variant=_PUBLIC_CURRENT_VARIANT,
        )
    elif role == "lover":
        if session is None:
            raise ValueError("Current Lover entry requires session board size")
        if not 1 <= pos <= session.n_cards:
            raise ValueError("Lover position is outside the current board")
        if len(args) != 3:
            raise ValueError("Lover entry requires exactly one evil count")
        evil_adjacent = int(args[2])
        if evil_adjacent not in {0, 1, 2}:
            raise ValueError("Lover evil count must be 0, 1, or 2")
        return card_lover(
            pos,
            evil_adjacent,
            info_text=_lover_native_text(evil_adjacent),
            lover_variant=_PUBLIC_CURRENT_VARIANT,
        )
    elif role == "scout":
        if session is None:
            raise ValueError("Current Scout entry requires session board size")
        if not 1 <= pos <= session.n_cards:
            raise ValueError("Scout position is outside the current board")
        sentinel_key = re.sub(
            r"[^a-z0-9]",
            "",
            " ".join(args[2:]).casefold(),
        )
        if sentinel_key in {"oneevil", "thereisonly1evil"}:
            return _card_scout_one_evil(
                pos,
                scout_variant=_PUBLIC_CURRENT_VARIANT,
            )
        if len(args) != 4:
            raise ValueError("Scout entry requires a named role and distance")
        named_role = get_card(args[2])
        if named_role is None:
            raise ValueError("Scout named role must be canonical")
        distance = int(args[3])
        if not _valid_current_scout_distance(distance, session.n_cards):
            raise ValueError("Scout distance is outside the native range")
        return card_scout(
            pos,
            named_role.name,
            distance,
            scout_variant=_PUBLIC_CURRENT_VARIANT,
        )
    elif role == "bard":
        if session is None:
            raise ValueError("Current Bard entry requires session board size")
        if not 1 <= pos <= session.n_cards:
            raise ValueError("Bard position is outside the current board")
        try:
            corruption_distance = int(args[2])
        except (TypeError, ValueError) as exc:
            raise ValueError("Bard corruption distance must be an integer") from exc
        if corruption_distance == 0:
            corruption_distance = -1
        if not _valid_current_bard_distance(
            corruption_distance,
            session.n_cards,
        ):
            raise ValueError("Bard corruption distance is outside the native range")
        return card_bard(
            pos,
            corruption_distance,
            bard_variant=_PUBLIC_CURRENT_VARIANT,
        )
    elif role in ("fortune_teller", "ft"):
        targets = [int(x) for x in args[2].split(",")]
        has_evil = args[3].lower() in ("yes", "true", "1")
        return card_fortune_teller(pos, targets, has_evil)
    elif role == "oracle":
        if session is None:
            raise ValueError("Current Oracle entry requires session board size")
        if not 1 <= pos <= session.n_cards:
            raise ValueError("Oracle position is outside the current board")
        sentinel_key = re.sub(
            r"[^a-z0-9]",
            "",
            " ".join(args[2:]).casefold(),
        )
        if sentinel_key in {"nominions", "therearenominions"}:
            return _card_oracle_no_minions(
                pos,
                oracle_variant=_PUBLIC_CURRENT_VARIANT,
            )
        if len(args) != 4:
            raise ValueError("Oracle entry requires two targets and a Minion role")
        try:
            oracle_targets = [int(value.strip()) for value in args[2].split(",")]
        except ValueError as exc:
            raise ValueError("Oracle targets must be comma-separated integers") from exc
        if (
            len(oracle_targets) != 2
            or oracle_targets != sorted(oracle_targets)
            or any(not 1 <= target <= session.n_cards for target in oracle_targets)
        ):
            raise ValueError(
                "Oracle requires two current-board targets in ascending ID order"
            )
        minion = get_card(args[3])
        if minion is None or minion.role.value != "Minion":
            raise ValueError("Oracle named role must be a canonical Minion")
        return card_oracle(
            pos,
            oracle_targets,
            minion.name,
            oracle_variant=_PUBLIC_CURRENT_VARIANT,
        )
    elif role == "medium":
        if session is None:
            raise ValueError("Current Medium entry requires session board size")
        if not 1 <= pos <= session.n_cards:
            raise ValueError("Medium position is outside the current board")
        if len(args) != 4:
            raise ValueError("Medium entry requires exactly one target and role")
        target_pos = int(args[2])
        if not 1 <= target_pos <= session.n_cards:
            raise ValueError("Medium target is outside the current board")
        claimed_role = args[3]
        # "real" means target IS their displayed role — resolve to actual role name
        if claimed_role.lower() == "real":
            target_card = next(
                (c for c in session.cards if c.position == target_pos),
                None,
            )
            if target_card is None:
                raise ValueError(
                    f"Medium 'real' target #{target_pos} has no current card entry"
                )
            claimed_role = target_card.apparent_role
        canonical_role = get_card(claimed_role)
        if canonical_role is None:
            raise ValueError("Medium named role must be canonical")
        return card_medium(
            pos,
            target_pos,
            canonical_role.name,
            medium_variant=_PUBLIC_CURRENT_VARIANT,
        )
    elif role == "hunter":
        if session is None:
            raise ValueError("Current Hunter entry requires session board size")
        if not 1 <= pos <= session.n_cards:
            raise ValueError("Hunter position is outside the current board")
        if len(args) != 3:
            raise ValueError("Hunter entry requires exactly one distance")
        distance = int(args[2])
        if not _valid_current_hunter_distance(distance, session.n_cards):
            raise ValueError("Hunter distance is outside the native range")
        return card_hunter(
            pos,
            distance,
            hunter_variant=_PUBLIC_CURRENT_VARIANT,
        )
    elif role == "architect":
        return card_architect(pos, args[2])  # Left/Right/Equal
    elif role == "empress":
        if session is None:
            raise ValueError("Current Empress entry requires session board size")
        if not 1 <= pos <= session.n_cards:
            raise ValueError("Empress position is outside the current board")
        try:
            targets = sorted(int(value.strip()) for value in args[2].split(","))
        except ValueError as exc:
            raise ValueError(
                "Empress targets must be comma-separated integers"
            ) from exc
        try:
            targets = _validate_current_empress_targets(targets)
        except ValueError as exc:
            raise ValueError(
                "Empress requires three distinct current-board targets"
            ) from exc
        if any(target > session.n_cards for target in targets):
            raise ValueError("Empress target is outside the current board")
        return card_empress(
            pos,
            targets,
            empress_variant=_PUBLIC_CURRENT_VARIANT,
        )
    elif role == "witness":
        return card_witness(pos, int(args[2]))
    elif role == "jester":
        targets = [int(x) for x in args[2].split(",")]
        return card_jester(pos, targets, int(args[3]))
    elif role == "rambler":
        # Bare current Rambler is passive no-info. Explicit quote/talking is a
        # current visible Day quote; silenced tokens remain available only for
        # archived pre-Rambler2 reconstruction:
        #   card rambler 2 silenced            -> silenced, picker unknown
        #   card rambler 2 silenced 6          -> silenced, picker was #6
        #   card rambler 2 talking             -> quote shown
        token = args[2].lower() if len(args) > 2 else ""
        if not token or token in ("current", "no_info", "none"):
            return card_no_info(pos, "Rambler")
        if token in ("quote", "talking", "spoke"):
            quote = " ".join(args[3:]).strip() or "<observed Rambler quote>"
            return card_rambler_quote(pos, quote)
        if token in ("silenced", "quiet", "silent", "true", "yes", "1"):
            silenced_by = (
                int(args[3])
                if len(args) > 3 and args[3].isdigit()
                else None
            )
            return card_rambler(pos, True, silenced_by)
        if token in ("unsilenced", "false", "no", "0"):
            return card_rambler(pos, False)
        raise ValueError(
            f"Unknown Rambler observation token {token!r}; use quote/talking, "
            "current/no_info, or an explicit archived silenced/unsilenced token"
        )
    elif role in ("shut_up", "shutup"):
        # card shut_up <pos> <apparent_role> <target>
        target = int(args[3])
        apparent_role = _normalize_role_name(args[2])
        if apparent_role == "Druid":
            raise ValueError(
                "Current Druid interruptions require authenticated raw "
                "callback history and cannot be entered manually"
            )
        if session is not None and target > session.n_cards:
            raise ValueError(
                f"Rambler shut-up target #{target} is outside "
                f"1..{session.n_cards}"
            )
        return card_shut_up(pos, apparent_role, target)
    elif role in ("dreamer", "dreamer2", "dreamer_ambiguous"):
        targets = [int(x) for x in args[2].split(",")]
        roles = [x.strip().replace("_", " ") for x in args[3].split(",")]
        return card_dreamer_ambiguous(pos, targets, roles)
    elif role in ("dreamer_old", "dreamer1"):
        return card_dreamer(pos, int(args[2]), args[3].replace("_", " "))
    elif role == "dreamer_cabbage":
        targets = [int(x) for x in args[2].split(",")]
        return card_dreamer_cabbage(pos, targets)
    elif role == "judge":
        is_lying = args[3].lower() in ("lying", "true", "1", "yes")
        return card_judge(pos, int(args[2]), is_lying)
    elif role == "alchemist":
        return card_alchemist(pos, int(args[2]))
    elif role == "druid":
        if session is None:
            raise ValueError("Current Druid entry requires session board size")
        if not 1 <= pos <= session.n_cards:
            raise ValueError("Druid position is outside the current board")
        if args[2].strip().casefold() in {"shut_up", "shutup"}:
            raise ValueError(
                "Current Druid interruptions require authenticated raw "
                "callback history and cannot be entered manually"
            )
        try:
            targets = [int(value.strip()) for value in args[2].split(",")]
        except ValueError as exc:
            raise ValueError(
                "Druid targets must be comma-separated integers"
            ) from exc
        targets = _validate_current_druid_targets(
            targets,
            n_cards=session.n_cards,
        )
        found = (
            None
            if args[3].strip().casefold() == "none"
            else _canonical_druid_outcast(args[3])
        )
        return card_druid(
            pos,
            targets,
            found,
            druid_variant=_PUBLIC_CURRENT_VARIANT,
        )
    elif role == "bishop":
        if session is None:
            raise ValueError("Current Bishop entry requires session board size")
        if not 1 <= pos <= session.n_cards:
            raise ValueError("Bishop position is outside the current board")
        try:
            targets = [int(value.strip()) for value in args[2].split(",")]
        except ValueError as exc:
            raise ValueError(
                "Bishop targets must be comma-separated integers"
            ) from exc
        if (
            not 1 <= len(targets) <= 3
            or len(set(targets)) != len(targets)
            or any(not 1 <= target <= session.n_cards for target in targets)
        ):
            raise ValueError(
                "Bishop requires one to three distinct current-board targets"
            )
        targets.sort()
        type_names = {
            "villager": "Villager",
            "outcast": "Outcast",
            "minion": "Minion",
            "demon": "Demon",
        }
        type_values = [value.strip().casefold() for value in args[3].split(",")]
        if len(type_values) != len(targets) or any(
            value not in type_names for value in type_values
        ):
            raise ValueError(
                "Bishop types must match its targets and be Villager, "
                "Outcast, Minion, or Demon"
            )
        types = [type_names[value] for value in type_values]
        return card_bishop(
            pos,
            targets,
            types,
            bishop_variant=_PUBLIC_CURRENT_VARIANT,
        )
    elif role == "baker":
        if len(args) > 2:
            # Only the explicit sentinel means "I am the original Baker".
            # Literal Baker means the distinct "I was a Baker" observation.
            return card_baker(pos, args[2])
        else:
            return card_baker(pos, "original")  # no arg = original Baker
    elif role == "poet":
        if len(args) > 2:
            # Poet clue variant: poet <pos> <clue_type> <args...>
            return card_poet_with_info(
                pos,
                args[2],
                args[3:],
                n_cards=session.n_cards if session is not None else None,
            )
        else:
            return card_no_info(pos, "Poet")  # No info identified
    elif role in ("bounty_hunter", "bountyhunter"):
        return card_poet_with_info(
            pos,
            "Bounty Hunter",
            [args[2]],
            n_cards=session.n_cards if session is not None else None,
        )
    elif role == "no_info":
        return card_no_info(pos, args[2])  # actual role name
    else:
        # Treat unknown as no_info with the role name capitalized
        return card_no_info(pos, role.replace("_", " ").title())


def _parse_true_evils(raw: str) -> dict[int, str]:
    """Parse '3=Shaman,7=Baa' format into {3: 'Shaman', 7: 'Baa'}."""
    result = {}
    for pair in raw.split(","):
        pos_str, role = pair.split("=")
        result[int(pos_str)] = role
    return result


def _resolve_runtime_evil_origins(
    runtime_evil_positions: set[int],
    session,
    result=None,
) -> tuple[dict[int, str], list[str]]:
    """Resolve stable origins from worlds matching the runtime-Evil seat set."""
    resolved: dict[int, str] = {}
    errors: list[str] = []

    if result is None:
        result = session._solve(session.to_game_state())
    scenarios = list(getattr(result, "surviving_scenarios", []) or [])
    matching_worlds = []
    for scenario in scenarios:
        seats = set(scenario.evil_positions)
        if scenario.puppet_position is not None:
            seats.add(scenario.puppet_position)
        if seats == runtime_evil_positions:
            matching_worlds.append(scenario)

    if not matching_worlds:
        return {}, [
            "no surviving scenario has exactly runtime-Evil seat set "
            f"{sorted(runtime_evil_positions)}"
        ]

    for scenario in matching_worlds:
        puppet_position = scenario.puppet_position
        if puppet_position is None:
            continue
        stable_at_puppet = scenario.evil_positions.get(puppet_position)
        if (
            stable_at_puppet is not None
            and _execution_role_key(stable_at_puppet) != "puppet"
        ):
            return {}, [
                f"#{puppet_position} simultaneously carries stable Evil "
                f"origin {stable_at_puppet!r} and generated Puppet state; "
                "one-role game_over recovery cannot represent this overlap"
            ]

    for position in sorted(runtime_evil_positions):
        world_roles: set[str] = set()
        for scenario in matching_worlds:
            role = scenario.evil_positions.get(position)
            if role is None and scenario.puppet_position == position:
                role = "Puppet"
            if role is None:
                world_roles.clear()
                break
            world_roles.add(_normalize_role_name(role))
        if len(world_roles) != 1:
            errors.append(
                f"#{position} has non-unanimous stable origin across "
                f"{len(matching_worlds)} matching scenario(s)"
            )
            continue
        world_origin = next(iter(world_roles))
        if _execution_role_key(world_origin) == "unknown":
            errors.append(
                f"#{position} stable origin is Unknown after execution; "
                "supply the original role from the public execution record"
            )
            continue
        recorded_origin = session.executed_evil_roles.get(position)
        if (
            _is_known_role(recorded_origin)
            and _execution_role_key(recorded_origin)
            != _execution_role_key(world_origin)
        ):
            errors.append(
                f"#{position} recorded origin {recorded_origin!r} conflicts "
                f"with unanimous scenario origin {world_origin!r}"
            )
            continue
        resolved[position] = (
            _normalize_role_name(recorded_origin)
            if _is_known_role(recorded_origin)
            else world_origin
        )

    if errors:
        return {}, errors
    return resolved, []


def _validate_true_evils_against_session(
    true_evils: dict,
    session,
    *,
    expected_runtime_evil_positions: Optional[set[int]] = None,
) -> tuple:
    """Validate that the evils-dict passed to game_over is consistent with session state.

    Rules:
      1. The map is complete when the game supplied an evil count.
      2. Positions are in range and agree with public alignment evidence.
      3. Stable origins come from the authored Evil pool, plus generated Puppet.
      4. Known execution origins and the aggregate Lilis evil-victim count agree.

    Returns (cleaned_dict, errors). If errors is non-empty, caller must refuse save.
    """
    errors = []

    def _normalize(r: str) -> str:
        return r.lower().replace("_", " ").replace("-", " ").strip()

    authored_evil_counts = Counter(
        _normalize(role)
        for role in (list(session.minions) + list(session.demons))
    )
    puppet_allowed = session.has_role_in_deck("Puppeteer")
    claimed_counts = Counter(_normalize(role) for role in true_evils.values())

    if session.n_evil > 0 and len(true_evils) != session.n_evil:
        errors.append(
            f"ERROR: true evils map has {len(true_evils)} position(s), "
            f"expected exactly {session.n_evil}"
        )

    keys = set(true_evils)
    if (
        expected_runtime_evil_positions is not None
        and keys != expected_runtime_evil_positions
    ):
        errors.append(
            "ERROR: true evils map positions "
            f"{sorted(keys)} do not match runtime-Evil seats "
            f"{sorted(expected_runtime_evil_positions)}"
        )
    out_of_range = sorted(
        position
        for position in keys
        if type(position) is not int or not 1 <= position <= session.n_cards
    )
    if out_of_range:
        errors.append(
            f"ERROR: true evil positions are outside 1..{session.n_cards}: "
            f"{out_of_range}"
        )

    missing_confirmed = sorted(set(session.confirmed_evil) - keys)
    if missing_confirmed:
        errors.append(
            "ERROR: confirmed Evil position(s) missing from true evils map: "
            f"{missing_confirmed}"
        )
    contaminated_good = sorted(set(session.confirmed_good) & keys)
    if contaminated_good:
        errors.append(
            "ERROR: confirmed Good position(s) appear in true evils map: "
            f"{contaminated_good}"
        )
    missing_known_origins = sorted(
        set(session.executed_evil_roles) - keys
    )
    if missing_known_origins:
        errors.append(
            "ERROR: executed stable Evil origin position(s) missing from "
            f"true evils map: {missing_known_origins}"
        )

    for role, count in sorted(claimed_counts.items()):
        if role == "puppet":
            allowed = 1 if puppet_allowed else 0
        else:
            allowed = authored_evil_counts.get(role, 0)
        if count > allowed:
            errors.append(
                f"ERROR: stable role {role!r} is claimed {count} time(s), "
                f"but the authored/generated Evil pool permits {allowed}"
            )

    for pos, role in true_evils.items():
        known_origin = session.executed_evil_roles.get(pos)
        if (
            _is_known_role(known_origin)
            and _normalize(known_origin) != _normalize(role)
        ):
            errors.append(
                f"ERROR: #{pos} claimed origin {role!r} conflicts with "
                f"recorded execution origin {known_origin!r}"
            )

    night_evil_positions = keys.intersection(session.night_kills)
    if len(night_evil_positions) != session.night_kill_evil_count:
        errors.append(
            "ERROR: true evils map contains "
            f"{len(night_evil_positions)} night-killed Evil position(s), "
            f"but live history recorded {session.night_kill_evil_count}"
        )
    if errors:
        return ({}, errors)
    return (true_evils, [])


_DECK_OUTCAST_ROLES = frozenset({
    "drunk", "wretch", "bombardier", "doppelganger", "plague_doctor", "rambler",
})


def _baa_hides_outcast(
    only_cv: Counter[str],
    only_mr: Counter[str],
    mr_counts: Counter[str],
    cv_unclassified: int,
) -> bool:
    """Baa obscures one existing Outcast as an eye-symbol in the deck view.

    That produces: exactly one outcast role in only_mr, zero only_cv, and at
    least one unclassified CV box. Native Imp.Act selects an exact Outcast
    CharacterData entry; it does not add a role to the gameplay pool.
    """
    if mr_counts["baa"] < 1:
        return False
    if only_cv or sum(only_mr.values()) != 1 or cv_unclassified < 1:
        return False
    return next(iter(only_mr)) in _DECK_OUTCAST_ROLES


def _baa_post_death_deck_refresh(_session) -> None:
    """Report Baa's native OnDied deck-view refresh.

    Managed Imp.Act removes the stored Outcast from
    DeckView.ObscuredCharacters. It does not reveal or mutate a board card, so
    this hook must never infer a newly flipped position from process memory.
    """
    print("  [Baa] Hidden Outcast is now visible in the deck view; no board card was flipped.")


def _print_baa_deck_count_note(demons: list[str]) -> None:
    """Keep Baa's presentation-only effect out of HUD-count bookkeeping."""
    if any(demon.lower() == "baa" for demon in demons):
        print("  NOTE: BAA hides one existing Outcast identity in the deck view. "
              "Use the HUD no= exactly as shown; do not subtract.")


def _cmd_read_deck(screenshot_path: str):
    """Read deck using both card_vision and memory_reader, cross-check results."""
    import subprocess

    # Card vision
    print("\n--- Card Vision ---")
    cv_result = subprocess.run(
        ["python", "card_vision.py", "classify_dirs", screenshot_path,
         "--context", "deck",
         "--library-dir", "templates/compendium/page1",
         "--library-dir", "templates/compendium/page3",
         "--library-dir", "templates/compendium/page4",
         "--library-dir", "templates/compendium/page5"],
        capture_output=True, text=True
    )
    cv_roles = []
    cv_unclassified = 0
    if cv_result.returncode == 0:
        try:
            import json as _json
            cards = _json.loads(cv_result.stdout)
            cv_roles = [c["name"] for c in cards if c.get("accepted")]
            cv_unclassified = sum(1 for c in cards if not c.get("accepted"))
            factions = {}
            for c in cards:
                if c.get("accepted"):
                    f = c.get("faction", "?")
                    factions.setdefault(f, []).append(c["name"])
            for faction in ["Villager", "Outcast", "Minion", "Demon"]:
                roles = factions.get(faction, [])
                if roles:
                    print(f"  {faction}s ({len(roles)}): {', '.join(roles)}")
            if cv_unclassified:
                print(f"  Unclassified boxes: {cv_unclassified}")
        except Exception as e:
            print(f"  ERROR parsing card_vision output: {e}")
            cv_roles = []
            cv_unclassified = 0
    else:
        print(f"  ERROR: card_vision failed: {cv_result.stderr[:200]}")

    # Memory reader
    print("\n--- Memory Reader ---")
    mr_result = subprocess.run(
        ["python", "memory_reader.py", "--deck"],
        capture_output=True, text=True
    )
    mr_roles = []
    if mr_result.returncode == 0:
        print(mr_result.stdout.strip())
        # Parse memory reader output to extract role names
        for line in mr_result.stdout.strip().split("\n"):
            line = line.strip()
            if line.startswith("Villager") or line.startswith("Outcast") or \
               line.startswith("Minion") or line.startswith("Demon"):
                # Format: "Villagers (7): Oracle, Baker, ..."
                colon_idx = line.find(":")
                if colon_idx > 0:
                    roles_str = line[colon_idx + 1:].strip()
                    mr_roles.extend([r.strip().lower().replace(" ", "_") for r in roles_str.split(",") if r.strip()])
    else:
        print(f"  ERROR: memory_reader failed: {mr_result.stderr[:200]}")

    # Cross-check
    cv_counts = Counter(r.lower().replace(" ", "_") for r in cv_roles)
    mr_counts = Counter(mr_roles)

    if cv_counts and mr_counts:
        if cv_counts == mr_counts:
            print(f"\n  MATCH: Both pipelines agree ({sum(cv_counts.values())} roles)")
        else:
            only_cv = cv_counts - mr_counts
            only_mr = mr_counts - cv_counts
            if _baa_hides_outcast(only_cv, only_mr, mr_counts, cv_unclassified):
                role = next(iter(only_mr))
                print(f"\n  MATCH (Baa hides outcast): CV={sum(cv_counts.values())} classified"
                      f" + '{role}' face-down in deck view (Baa effect)")
            else:
                print(f"\n  MISMATCH!")
                if only_cv:
                    print(f"    Only in card_vision: {only_cv}")
                if only_mr:
                    print(f"    Only in memory_reader: {only_mr}")
                print(f"    STOP AND FIX before proceeding!")
    elif not cv_counts and not mr_counts:
        print("\n  WARNING: Both pipelines returned empty results")
    else:
        print("\n  WARNING: Only one pipeline returned results "
              f"(cv={sum(cv_counts.values())}, mr={sum(mr_counts.values())})")


def _save_and_run_test(name: str, true_evils: dict[int, str], notes: str = ""):
    """Save a regression test case. Full regression runs via cargo test afterward."""
    from tests.test_utils import save_test_case
    # Check for collision
    test_path = os.path.join("tests", "cases_v2", f"{name}.json")
    if os.path.exists(test_path):
        # Append suffix to avoid overwriting
        for suffix in "bcdefgh":
            alt_name = f"{name}{suffix}"
            alt_path = os.path.join("tests", "cases_v2", f"{alt_name}.json")
            if not os.path.exists(alt_path):
                print(f"  WARNING: {name}.json exists, saving as {alt_name}.json instead")
                name = alt_name
                break
    save_test_case(SESSION_FILE, name, true_evils, notes)
    print(f"  Test case saved: tests/cases_v2/{name}.json")


def _parse_pd_check_args(
    args: list[str],
    n_cards: int,
    used_abilities: list[int] | set[int] | tuple[int, ...] = (),
    apparent_roles: Optional[dict[int, str]] = None,
) -> tuple[Optional[dict], Optional[str]]:
    """Validate CLI PD evidence before it can mutate the live session."""
    if len(args) < 3:
        return None, (
            "Usage: pd_check <pd_pos> <target> "
            "<clean|corrupted [evil_pos]>"
        )
    try:
        pd_pos = int(args[0])
        target = int(args[1])
    except ValueError:
        return None, "Plague Doctor and target positions must be integers"

    if not 1 <= pd_pos <= n_cards:
        return None, f"Plague Doctor position #{pd_pos} is outside 1..{n_cards}"
    if not 1 <= target <= n_cards:
        return None, f"Plague Doctor target #{target} is outside 1..{n_cards}"
    if pd_pos in used_abilities:
        return None, f"Plague Doctor #{pd_pos} ability is already recorded as used"
    if apparent_roles is not None:
        apparent_role = apparent_roles.get(pd_pos)
        role_key = (
            apparent_role.lower().replace(" ", "_")
            if apparent_role is not None else None
        )
        if role_key != "plague_doctor":
            shown = apparent_role if apparent_role is not None else "unrevealed"
            return None, (
                f"Position #{pd_pos} is {shown}, not an apparent Plague Doctor"
            )

    status = args[2].lower()
    if status == "clean":
        if len(args) != 3:
            return None, "Clean PD result must not include an evil position"
        return {
            "pd_pos": pd_pos,
            "target": target,
            "is_corrupted": False,
            "evil_revealed": None,
        }, None

    if status == "corrupted":
        if len(args) != 4:
            return None, "Corrupted PD result requires exactly one evil position"
        if target == pd_pos:
            return None, "Native Plague Doctor self-check always displays Not Corrupted"
        try:
            evil_revealed = int(args[3])
        except ValueError:
            return None, "Plague Doctor revealed position must be an integer"
        if not 1 <= evil_revealed <= n_cards:
            return None, (
                f"Plague Doctor revealed position #{evil_revealed} is outside "
                f"1..{n_cards}"
            )
        return {
            "pd_pos": pd_pos,
            "target": target,
            "is_corrupted": True,
            "evil_revealed": evil_revealed,
        }, None

    return None, f"Unknown PD check status: {status} (use 'corrupted' or 'clean')"


def main():
    if len(sys.argv) < 2:
        print("Usage: python game_loop.py <command> [args...]")
        print()
        print("Commands:")
        print("  auto [--games=N] [--risk=conservative] Full autonomous play")
        print("  start                                 Start new game (menu nav + deck read)")
        print("  new <n_cards> <n_evil> [hp=N cost=N] Start new game session")
        print("  start_village <n_cards> <n_evil> nv=N no=N  Combined new+deck via memory_reader")
        print("  deck V=... O=... M=... D=...         Set deck composition")
        print("  read_deck <screenshot>                Read deck (card_vision + memory_reader)")
        print("  flip                                  Flip all cards #1->#N in order")
        print("  flip <pos>                            Flip single card (after Witch death)")
        print("  flip --lilis                          Flip 1-4 cards to the next verified Night boundary")
        print("  card <role> <pos> [args...]           Add a revealed card")
        print("  auto_card                             Auto-enter cards from memory reader")
        print("  execute <pos> [evil|good] [role]      Mark position executed (with evil role name)")
        print("  execute <pos> evil <role> current=<role>  Preserve original Evil role after a transformed reveal")
        print("  execute <pos> <RoleName>              Shorthand: mark as evil with role")
        print("  execute <pos> good blocked            Knight immunity (confirms good only without data movers)")
        print("  execute <pos> good <clean|corrupted> [revealed_role]")
        print("                                           Wrong exec with optional UI-observed role")
        print("  pd_check <pd_pos> <target> corrupted <evil_pos>  PD found corruption + evil")
        print("  pd_check <pd_pos> <target> clean                 PD found no corruption")
        print("  pd_clear <pd_pos>                    Remove a mistaken PD result before re-entry")
        print("  set_hp <hp> [wrong_exec_cost]         Update HP and wrong execution cost")
        print("  solve                                 Run solver")
        print("  status                                Print session state")
        print("  confirm_evil <pos>                    Mark position as confirmed evil")
        print("  confirm_good <pos>                    Mark position as confirmed good")
        print("  next [--plan]                         Solve + auto-execute if safe (definite OR forced-safe). --plan for print-only.")
        print("  auto_next                             Alias for `next` (auto-execute path)")
        print("  ability_used <pos>                    Mark ability as activated")
        print("  slayer_result <pos> <target> kill <role> [good|evil] [clean|corrupted]")
        print("                                           good requires a positive public HP delta (or other exact evidence)")
        print("                                           omit alignment on zero HP delta: Evil and Good+NoDamage are ambiguous")
        print("  slayer_result <pos> <target> fail                           Slayer miss")
        print("  block <pos>                           Mark position as blocked (Witch)")
        print("  unblock <pos>                         Unblock position (after Witch dies)")
        print("  night_kill <pos1,pos2,...> <n_evil>    Resolve pending Lilis night(s), one victim each")
        print("  night_no_kill                         Resolve one pending Night with no victim (0HP when Lilis is known dead; no identity inference)")
        print("  log <label> <text>                    Add reasoning to decision log")
        print("  game_over <w/l> <name> <evils> [note] Log result + auto-save regression test")
        print("  save_test <name> [true_evils_json]    Save game as regression test (manual)")
        print()
        print("Card examples:")
        print("  card enlightened 3 CW")
        print("  card confessor 1 <Good|dizzy>")
        print("  card knitter 2 2")
        print("  card fortune_teller 4 1,3 yes")
        print("  card oracle 5 2,6 Shaman          (or: card oracle 5 no_minions)")
        print("  card bishop 7 4,7,9 Outcast,Minion,Villager")
        print("  card jester 7 1,3,5 1")
        print("  card dreamer 5 3,9 Puppeteer,Lover")
        print("  card dreamer_cabbage 5 3,9")
        print("  card dreamer_old 5 3 Pooka")
        print("  card poet 5 knitter 0       (Poet gave Knitter-style clue)")
        print("  card poet 3 lover 2         (Poet gave Lover-style clue)")
        print("  card poet 4 bard 1          (Poet gave Bard-style clue)")
        print("  card poet 2 gemcrafter 6    (Poet gave Gemcrafter-style clue)")
        print("  card poet 1 bounty_hunter 6 (Poet directly named #6 as Evil)")
        print("  card druid 5 1,2,3 none       (Druid found no Outcasts)")
        print("  card druid 5 1,2,3 Bombardier (Druid named an Outcast)")
        print("  Druid shut-up results require authenticated auto_card memory")
        print("  card no_info 2 Slayer")
        return

    cmd = sys.argv[1].lower()
    args = sys.argv[2:]

    if cmd == "repl":
        repl_loop()
        return

    # Commands that don't need an existing session
    if cmd in ("start", "start_village", "read_deck", "new", "auto"):
        session = dispatch(cmd, args)
        return

    # All other commands need a session
    try:
        session = GameSession.load()
    except FileNotFoundError:
        print("ERROR: No active session. Run 'new' first.")
        return

    dispatch(cmd, args, session)


def repl_loop():
    """Persistent REPL: session stays in memory, no process restart between commands."""
    import shlex

    print("REPL_READY")
    sys.stdout.flush()

    session = None
    try:
        session = GameSession.load()
        print(f"[repl] Loaded session: {session.n_cards} cards, {session.n_evil} evil")
    except FileNotFoundError:
        print("[repl] No active session. Use 'new' to start.")

    while True:
        sys.stdout.flush()
        try:
            line = input()
        except EOFError:
            break

        line = line.strip()
        if not line or line.startswith("#"):
            print("CMD_DONE")
            sys.stdout.flush()
            continue

        if line.lower() in ("quit", "exit"):
            print("[repl] Exiting.")
            break

        if line.lower() == "reload":
            try:
                session = GameSession.load()
                print(f"[repl] Reloaded session from disk")
            except FileNotFoundError:
                print("[repl] No session file found")
            print("CMD_DONE")
            sys.stdout.flush()
            continue

        try:
            parts = shlex.split(line)
        except ValueError as e:
            print(f"ERROR: Could not parse: {e}")
            print("CMD_DONE")
            sys.stdout.flush()
            continue

        cmd = parts[0].lower()
        args = parts[1:]

        try:
            if cmd in ("start", "start_village", "read_deck", "new", "auto"):
                result = dispatch(cmd, args, session)
                if result is not None:
                    session = result
            else:
                if session is None:
                    print("ERROR: No active session. Run 'new' first.")
                else:
                    result = dispatch(cmd, args, session)
                    if result is not None:
                        session = result
        except Exception as e:
            print(f"ERROR: {type(e).__name__}: {e}")

        print("CMD_DONE")
        sys.stdout.flush()


def dispatch(cmd: str, args: list[str], session: Optional[GameSession] = None) -> Optional[GameSession]:
    """Dispatch a game loop command. Returns a new session if one was created (e.g., 'new').

    Args:
        cmd: Command name (lowercase)
        args: Remaining arguments (what would have been sys.argv[2:])
        session: Active session (None for start/read_deck/new)
    """

    if cmd == "start":
        import subprocess
        print("=== STARTING NEW GAME ===")
        print("[1/5] Play Demo...")
        subprocess.run(["python", "template_match.py", "safe_click", "menu_play_demo"])
        time.sleep(1)
        print("[2/5] Standard mode...")
        subprocess.run(["python", "template_match.py", "safe_click", "mode_standard"])
        time.sleep(2)
        print("[3/5] Dismiss intro...")
        subprocess.run(["python", "template_match.py", "safe_click", "btn_close_dialog"])
        time.sleep(1)
        print("[4/5] Parking mouse, screenshotting deck...")
        subprocess.run(["python", "mouse.py", "move", "50", "1350"])
        time.sleep(0.5)
        result = subprocess.run(["python", "screenshot.py", "deck_view"],
                                capture_output=True, text=True)
        screenshot_path = result.stdout.strip()
        print(f"  Deck screenshot: {screenshot_path}")
        print("[5/5] Reading deck (card_vision + memory_reader)...")
        _cmd_read_deck(screenshot_path)
        print("\n=== START COMPLETE ===")
        print("Next: verify deck above, then run:")
        print("  python game_loop.py new <n_cards> <n_evil>")
        print("  python game_loop.py deck V=... O=... M=... D=... nv=N no=N")
        print("  python game_loop.py flip")
        return None

    if cmd == "read_deck":
        screenshot_path = args[0] if len(args) > 0 else None
        if not screenshot_path:
            print("Usage: read_deck <screenshot_path>")
            return None
        _cmd_read_deck(screenshot_path)
        return None

    if cmd == "new":
        n_cards = int(args[0])
        n_evil = int(args[1])
        session = GameSession(n_cards, n_evil)
        i = 2
        while i < len(args):
            arg = args[i]
            if arg.startswith("hp="):
                session.hp = int(arg[3:])
            elif arg.startswith("cost="):
                session.wrong_exec_cost = int(arg[5:])
            elif arg == "--villagers" and i + 1 < len(args):
                i += 1
                session.villagers = _parse_role_list(args[i])
            elif arg == "--outcasts" and i + 1 < len(args):
                i += 1
                session.outcasts = _parse_role_list(args[i])
            elif arg == "--minions" and i + 1 < len(args):
                i += 1
                session.minions = _parse_role_list(args[i])
            elif arg == "--demons" and i + 1 < len(args):
                i += 1
                session.demons = _parse_role_list(args[i])
            i += 1
        session.save()
        DecisionLog.start_game(n_cards, n_evil, session.hp, session.wrong_exec_cost)
        print(f"New session: {n_cards} cards, {n_evil} evil, HP={session.hp}, cost={session.wrong_exec_cost}")
        return session

    if cmd == "start_village":
        # Combined command: new + deck in one call. Reads pool roles from
        # memory_reader.py --deck; caller still provides nv/no (header counts
        # are not in memory).
        #   start_village <n_cards> <n_evil> nv=N no=N [hp=10] [cost=5]
        if len(args) < 2:
            print("Usage: start_village <n_cards> <n_evil> nv=N no=N [hp=10] [cost=5]")
            return None
        n_cards = int(args[0])
        n_evil = int(args[1])
        nv = None
        no = None
        hp_arg = None
        cost_arg = None
        for a in args[2:]:
            if a.lower().startswith("nv="):
                nv = int(a[3:])
            elif a.lower().startswith("no="):
                no = int(a[3:])
            elif a.startswith("hp="):
                hp_arg = int(a[3:])
            elif a.startswith("cost="):
                cost_arg = int(a[5:])
            else:
                print(f"  ERROR: Unrecognized arg '{a}'")
                print("  Required: nv=N no=N. Optional: hp=N cost=N")
                return None
        if nv is None or no is None:
            print("  ERROR: nv=N and no=N are required (header counts from screenshot)")
            return None

        # Read pool from memory_reader.py --deck
        import subprocess as _sp
        mr_result = _sp.run(
            ["python", "memory_reader.py", "--deck"],
            capture_output=True, text=True
        )
        if mr_result.returncode != 0:
            print(f"  ERROR: memory_reader --deck failed: {mr_result.stderr[:200]}")
            return None
        pool = {"villagers": [], "outcasts": [], "minions": [], "demons": []}
        for line in mr_result.stdout.strip().split("\n"):
            line = line.strip()
            faction_key = None
            if line.startswith("Villager"):
                faction_key = "villagers"
            elif line.startswith("Outcast"):
                faction_key = "outcasts"
            elif line.startswith("Minion"):
                faction_key = "minions"
            elif line.startswith("Demon"):
                faction_key = "demons"
            if faction_key:
                colon_idx = line.find(":")
                if colon_idx > 0:
                    roles_str = line[colon_idx + 1:].strip()
                    pool[faction_key] = [r.strip().replace(" ", "_") for r in roles_str.split(",") if r.strip()]
        if not (pool["villagers"] or pool["minions"]):
            print("  ERROR: memory_reader returned no roles. Is the game window active?")
            return None

        # Initialize session with pool + board counts
        session = GameSession(n_cards, n_evil)
        if hp_arg is not None:
            session.hp = hp_arg
        if cost_arg is not None:
            session.wrong_exec_cost = cost_arg
        session.set_deck(pool["villagers"], pool["outcasts"], pool["minions"], pool["demons"])
        session.board_villager_count = nv
        session.board_outcast_count = no
        if nv is not None and no is not None:
            session.board_count_provenance = "trusted_pre_start"
        session.save()
        DecisionLog.start_game(n_cards, n_evil, session.hp, session.wrong_exec_cost)
        DecisionLog.log_deck(pool["villagers"], pool["outcasts"], pool["minions"], pool["demons"])
        _print_baa_deck_count_note(pool["demons"])
        print(f"Village started: {n_cards} cards, {n_evil} evil, HP={session.hp}")
        print(f"  V={pool['villagers']}")
        print(f"  O={pool['outcasts']}")
        print(f"  M={pool['minions']}")
        print(f"  D={pool['demons']}")
        print(f"  board: nv={nv} no={no}")
        print("Next: python game_loop.py flip")
        return session

    if cmd == "set_hp":
        session.hp = int(args[0])
        if len(args) > 1:
            session.wrong_exec_cost = int(args[1])
        session.save()
        print(f"HP set to {session.hp}, wrong exec cost = {session.wrong_exec_cost}")
        return None

    if cmd == "deck":
        villagers, outcasts, minions, demons = [], [], [], []
        parsed_nv: Optional[int] = None
        parsed_no: Optional[int] = None
        for arg in args:
            if arg.startswith("V=") or arg.startswith("v="):
                villagers = _parse_role_list(arg[2:])
            elif arg.startswith("O=") or arg.startswith("o="):
                outcasts = _parse_role_list(arg[2:])
            elif arg.startswith("M=") or arg.startswith("m="):
                minions = _parse_role_list(arg[2:])
            elif arg.startswith("D=") or arg.startswith("d="):
                demons = _parse_role_list(arg[2:])
            elif arg.lower().startswith("nv="):
                parsed_nv = int(arg[3:])
            elif arg.lower().startswith("no="):
                parsed_no = int(arg[3:])
            else:
                print(f"  ERROR: Unrecognized arg '{arg}' -- missing prefix?")
                print(f"  Required: V=roles O=roles M=roles D=roles nv=N no=N")
                print(f"  Command aborted. Fix and re-run deck command.")
                return None
        if (parsed_nv is None) != (parsed_no is None):
            print("  ERROR: nv= and no= must be supplied together.")
            print("  Command aborted without changing the deck or board counts.")
            return None
        session.set_deck(villagers, outcasts, minions, demons)
        if parsed_nv is not None and parsed_no is not None:
            session.board_villager_count = parsed_nv
            session.board_outcast_count = parsed_no
            session.board_count_provenance = "trusted_pre_start"
        _print_baa_deck_count_note(demons)
        pool_size = len(villagers) + len(outcasts) + len(minions) + len(demons)
        if pool_size > session.n_cards and session.board_villager_count is None:
            board_good = session.n_cards - session.n_evil
            board_evil = len(minions) + len(demons)
            if board_evil == session.n_evil:
                print(f"  NOTE: Pool has {pool_size} roles for {session.n_cards} board positions.")
                print(f"  Use nv=N no=N to specify actual board counts (e.g., deck ... nv=6 no=1)")
        session.save()
        DecisionLog.log_deck(villagers, outcasts, minions, demons)
        extra_info = ""
        if session.board_villager_count is not None or session.board_outcast_count is not None:
            extra_info = f" [board: nv={session.board_villager_count} no={session.board_outcast_count}]"
        print(f"Deck set: V={villagers} O={outcasts} M={minions} D={demons}{extra_info}")
        return None

    if cmd == "flip":
        lilis = "--lilis" in args
        single_pos = None
        for arg in args:
            if arg.isdigit():
                single_pos = int(arg)

        if (
            session.has_lilis_night_rule()
            and session.has_role_in_deck("Shaman")
        ):
            print(
                "  ERROR: Lilis+Shaman live reveal automation is paused: "
                "Shaman can erase or duplicate the current Lilis actor, so "
                "0/2/4 HP Night behavior is not yet traceable. No card was clicked."
            )
            return None
        if lilis and not session.has_lilis_night_rule():
            print(
                "  ERROR: --lilis requires Lilis in the recorded deck; "
                "no cards were clicked."
            )
            return None
        if session.has_duplicate_lilis():
            print(
                "  ERROR: Duplicate Lilis live automation is unsupported. "
                "Multiple actors can charge HP and collide on one delayed "
                "victim; no cards were clicked."
            )
            return None
        if session.pending_lilis_nights > 0:
            print(
                f"  ERROR: {session.pending_lilis_nights} Lilis Night "
                "transition(s) still need resolution; no cards were clicked. "
                "Use night_kill or night_no_kill first."
            )
            return None

        from game_utils import all_game_card_coords
        import subprocess
        import template_match as _tm
        import mouse as _mouse
        coords = all_game_card_coords(session.n_cards)

        if single_pos:
            if single_pos not in coords:
                print(f"ERROR: Position {single_pos} not valid for {session.n_cards}-card game")
                return None
            was_revealed = single_pos in session.reveal_order
            print(f"Flipping #{single_pos} with memory verification")
            _click_flip_card(single_pos, coords, f"card{single_pos}", verified=True)
            cards = _read_board_once_for_flip()
            if not cards:
                print(
                    "  WARNING: Could not verify the single-card click in memory; "
                    "session reveal/block state was not changed."
                )
                return None
            verify = _verify_flips(cards, [single_pos], session)
            verification_changed = _apply_flip_verification(
                session,
                [single_pos],
                verify,
                persist=False,
            )
            night_total_reveals = None
            if (
                single_pos in verify["flipped"]
                and not was_revealed
                and session.has_lilis_night_rule()
                and len(session.reveal_order) % 4 == 0
            ):
                night_total_reveals = len(session.reveal_order)
                session.schedule_lilis_night()
            if verification_changed or night_total_reveals is not None:
                # Persist the verified reveal and pending native transition in
                # one replace, never as an intermediate fourth-reveal save.
                session.save()
            if single_pos in verify["blocked"]:
                print(f"  #{single_pos} remains hidden under the Witch quota.")
                return None
            if single_pos in verify["failed"]:
                print(f"  #{single_pos} did not reveal; session state was left unrevealed.")
                return None
            if single_pos not in verify["flipped"]:
                print(
                    f"  #{single_pos} resolved dead/hidden; it was not counted "
                    "as a reveal."
                )
                return None
            print(f"  Verified reveal of #{single_pos}")
            if night_total_reveals is not None:
                # NightModeRule survives Lilis death, so the fourth verified
                # reveal still stops even when actor effects are now no-ops.
                print()
                print("!" * 60)
                print(f"  LILIS NIGHT PHASE TRIGGERED (reveal #{night_total_reveals})")
                if session.is_lilis_alive():
                    print(f"  Lilis deals 2 HP. HP: {session.hp} -> {session.hp - 2}")
                else:
                    print(
                        "  Lilis is known dead: the persistent Night rule "
                        f"still runs, but HP stays {session.hp}."
                    )
                print("!" * 60)
                print(f"\n  --- Waiting for Lilis night animation ---")
                try:
                    from memory_reader import get_monitor as _get_mon
                    _mon = _get_mon()
                    if _mon.is_healthy():
                        already_done = (
                            set(session.reveal_order)
                            | set(session.night_kills)
                            | set(session.executed)
                        )
                        def _night_resolved(board):
                            if not board:
                                return False
                            return any(c.get('killed_hidden') for c in board
                                       if c['position'] not in already_done)
                        _mon.wait_for(_night_resolved, timeout=8, min_delay=2.0)
                    else:
                        time.sleep(5)
                except Exception:
                    time.sleep(5)
                print(f"  Night phase complete.")
                if session.is_lilis_alive():
                    print(f"  Run: night_kill <pos> <n_evil>  OR  night_no_kill")
                    print(f"  (HP auto-deducted by night_kill/night_no_kill commands)")
                else:
                    print(
                        "  Run: night_no_kill to persist the zero-damage "
                        "post-death Night transition."
                    )
            return None

        already_done = (
            set(session.reveal_order)
            | set(session.night_kills)
            | set(session.executed)
            | set(session.blocked_positions)
        )
        positions = [p for p in sorted(coords.keys()) if p not in already_done]
        if not positions:
            print("All cards already flipped/dead. Nothing to flip.")
            return None
        if lilis:
            reveals_before_batch = len(session.reveal_order)
            batch_size = 4 - (reveals_before_batch % 4)
            batch = positions[:batch_size]
            expected_positions = batch
            print(
                f"Flipping toward next Lilis Night boundary "
                f"({batch_size} verified reveal(s) needed): "
                f"{['#'+str(p) for p in batch]}"
            )
            for idx, pos in enumerate(batch):
                _click_flip_card(pos, coords, f"card{pos}", verified=(idx == 0))
                time.sleep(0.2)
            print(f"Batch complete: {['#'+str(p) for p in batch]}")
            remaining = positions[batch_size:]
        else:
            expected_positions = positions
            print(f"Flipping all {len(positions)} cards: #1 -> #{positions[-1]}")
            for idx, pos in enumerate(positions):
                _click_flip_card(pos, coords, f"card{pos}", verified=(idx == 0))
                time.sleep(0.2)
            print(f"All {len(positions)} cards flipped in order #1->#{positions[-1]}")
        print("\n--- Parking mouse & reading memory ---")
        _mouse.move(1280, 690)

        # Smart wait: use monitor if available, else fixed sleep
        from memory_reader import MemoryReader as _MR, print_board as _print_board
        try:
            from memory_reader import get_monitor as _get_monitor
            _mon = _get_monitor()
            if _mon.is_healthy():
                def _flips_done(board):
                    if not board:
                        return False
                    for p in expected_positions:
                        card = next((c for c in board if c['position'] == p), None)
                        if card and card['state'] == 'Hidden' and not card.get('killed_hidden'):
                            return False  # still waiting
                    return True
                time.sleep(0.5)
                _mon.wait_for(_flips_done, timeout=3, min_delay=0.5)
                cards = _mon.get_board()
            else:
                raise RuntimeError("monitor not healthy")
        except Exception:
            # Fallback: fixed sleep + manual read
            time.sleep(1.5)
            _reader = _MR()
            cards = None
            if _reader.open():
                cards = _reader.read_board()
                _reader.close()

        print("\n--- Memory Reader (board state) ---")
        if cards:
            _print_board(cards)
            verify = _verify_flips(cards, expected_positions, session)
            verification_changed = _apply_flip_verification(
                session,
                expected_positions,
                verify,
                persist=False,
            )
            if lilis:
                resolved_positions = (
                    set(session.reveal_order)
                    | set(session.night_kills)
                    | set(session.executed)
                    | set(session.blocked_positions)
                )
                remaining = [
                    position for position in sorted(coords)
                    if position not in resolved_positions
                ]
            reveals_after_batch = len(session.reveal_order)
            lilis_night_triggered = (
                lilis
                and reveals_after_batch // 4 > reveals_before_batch // 4
            )
            if lilis_night_triggered:
                session.schedule_lilis_night()
            if verification_changed or lilis_night_triggered:
                # Reveal order and pending Night become durable together.
                session.save()
            if verify["blocked"]:
                print(
                    "  [reveal_order] Witch-blocked attempts were kept out of "
                    f"reveal order: {verify['blocked']}"
                )
            if verify["failed"]:
                print(
                    "  [reveal_order] Failed attempts were kept out of reveal "
                    f"order: {verify['failed']}"
                )
            if lilis_night_triggered:
                print(f"\n  --- Lilis night phase (4 verified reveals; waiting for kill animation) ---")
                if session.is_lilis_alive():
                    print(f"  Lilis deals 2 HP. HP: {session.hp} -> {session.hp - 2}")
                else:
                    print(
                        "  Lilis is known dead: the persistent Night rule "
                        f"still runs, but HP stays {session.hp}."
                    )
                try:
                    from memory_reader import get_monitor as _get_mon
                    _mon = _get_mon()
                    if _mon.is_healthy():
                        _already = (
                            set(session.reveal_order)
                            | set(session.night_kills)
                            | set(session.executed)
                        )
                        def _night_kill_check(board):
                            if not board:
                                return False
                            return any(
                                card.get('killed_hidden')
                                for card in board
                                if card['position'] not in _already
                            )
                        _mon.wait_for(_night_kill_check, timeout=8, min_delay=2.0)
                    else:
                        time.sleep(5)
                except Exception:
                    time.sleep(5)
                print("  Night phase complete. Take screenshot to check for kills before continuing.")
                print("  Run: python screenshot.py night_check && python memory_reader.py")
                if remaining:
                    print(f"  Remaining to flip: {['#'+str(p) for p in remaining]}")
                else:
                    print("  No more cards to flip. Check for night kill/damage.")
                if session.is_lilis_alive():
                    print("  Run: night_kill <pos> <n_evil>  OR  night_no_kill")
                    print("  (HP auto-deducted by night_kill/night_no_kill commands)")
                else:
                    print(
                        "  Run: night_no_kill to persist the zero-damage "
                        "post-death Night transition."
                    )
            elif lilis:
                verified_in_batch = reveals_after_batch - reveals_before_batch
                print(
                    "  Lilis night did not trigger: "
                    f"{verified_in_batch}/{batch_size} required reveal(s) were "
                    "memory-verified. Retry failed clicks or resolve the Witch block."
                )
        else:
            print(
                "  WARNING: memory_reader returned no cards; session reveal/block "
                "state was not changed"
            )
        print("\nNow screenshot and enter card info in order #1->#{}.".format(expected_positions[-1]))
        return None

    if cmd == "auto_card":
        from memory_reader import MemoryReader as _MR, print_board as _print_board
        _reader = _MR()
        if not _reader.open():
            print("ERROR: Could not open game process")
            return None
        cards = _reader.read_board()
        _reader.close()
        if cards:
            _print_board(cards)
        if not cards:
            print("ERROR: No board data from memory reader")
            return None

        entered = {c.position: c for c in session.cards}
        dead = set(session.executed) | set(session.night_kills)
        auto_count = 0
        manual_needed = []

        for mc in cards:
            pos = mc['position']
            if pos in dead:
                continue
            state = mc.get('state', '')
            if state not in ('Alive', 'Revealed'):
                continue  # Hidden/Dead — skip

            memory_role_key = (
                mc.get('disguise') or _observed_current_role(mc) or ''
            ).lower().replace(' ', '_')
            is_druid_memory_role = memory_role_key in {
                "druid",
                "librarian",
                "rangedempath",
            }
            is_jester_memory_role = memory_role_key in {"jester", "juggler"}

            parsed = _parse_clue_from_memory(
                mc,
                n_cards=session.n_cards,
                baker_rule_version=session.baker_rule_version,
                fortune_teller_rule_version=session.fortune_teller_rule_version,
            )
            rambler_capture_error = None
            baker_capture_error = None
            fortune_capture_error = None
            druid_capture_error = None
            jester_capture_error = None
            if parsed is None:
                if is_druid_memory_role:
                    _, druid_capture_error = _parse_druid_result_from_memory(
                        mc,
                        n_cards=session.n_cards,
                    )
                if is_jester_memory_role:
                    _, jester_capture_error = _parse_jester_result_from_memory(
                        mc,
                        n_cards=session.n_cards,
                    )
                _, rambler_capture_error = _card_from_rambler_surface(
                    mc,
                    n_cards=session.n_cards,
                )
                if rambler_capture_error is None:
                    _, baker_capture_error = _card_from_baker_surface(
                        mc,
                        baker_rule_version=session.baker_rule_version,
                    )
                if (
                    rambler_capture_error is None
                    and baker_capture_error is None
                    and memory_role_key == "fortune_teller"
                    and session.fortune_teller_rule_version
                    == FORTUNE_TELLER_RULE_VERSION
                ):
                    _, fortune_capture_error = (
                        _parse_fortune_teller_result_from_memory(
                            mc,
                            n_cards=session.n_cards,
                        )
                    )
            if parsed:
                existing = entered.get(pos)
                parsed_role_key = _execution_role_key(
                    parsed.apparent_role
                ).replace(" ", "_")
                guarded_active_capture = parsed_role_key in {
                    "dreamer",
                    "fortune_teller",
                    "judge",
                    "plague_doctor",
                    "slayer",
                }
                if (
                    guarded_active_capture
                    and _has_active_clue_result(parsed)
                ):
                    remaining = _pickable_uses_remaining(mc)
                    if remaining is None:
                        manual_needed.append(
                            f"  #{pos} {parsed.apparent_role}: [RECOVERY] "
                            "active result has an unreadable native "
                            "pickable-use budget"
                        )
                        continue
                    if remaining > 0:
                        # A coherent event can be retained from a prior Night
                        # or village. Without a pre-click prefix it is not a
                        # current-cycle result. Preserve an existing session
                        # record, or enter only the active no-info shell.
                        if existing is not None:
                            continue
                        parsed = (
                            _card_current_jester_no_info(pos)
                            if (
                                parsed_role_key == "jester"
                                and parsed.info_parsed.get("jester_variant")
                                == _PUBLIC_CURRENT_VARIANT
                            )
                            else card_no_info(pos, parsed.apparent_role)
                        )
                        parsed_role_key = _execution_role_key(
                            parsed.apparent_role
                        ).replace(" ", "_")
                current_druid_capture = (
                    parsed_role_key == "druid"
                    and isinstance(parsed.info_parsed, dict)
                    and parsed.info_parsed.get("druid_variant")
                    == _PUBLIC_CURRENT_VARIANT
                )
                if current_druid_capture:
                    capture_status, capture_status_error = (
                        _classify_druid_auto_capture(
                            existing,
                            parsed,
                            n_cards=session.n_cards,
                            reveal_order=session.reveal_order,
                            baker_rule_version=session.baker_rule_version,
                            rambler_observations=(
                                session.rambler_shut_up_observations
                            ),
                        )
                    )
                    if capture_status_error is not None:
                        role = (
                            mc.get('disguise')
                            or _observed_current_role(mc)
                            or '?'
                        )
                        manual_needed.append(
                            f"  #{pos} {role}: [RECOVERY] "
                            f"{capture_status_error}"
                        )
                        continue
                    if capture_status == "stale":
                        if (
                            _active_cycle_is_spent(mc)
                            and _has_active_clue_result(parsed)
                        ):
                            session.mark_ability_used(parsed.position)
                        continue
                current_jester_capture = (
                    parsed_role_key == "jester"
                    and isinstance(parsed.info_parsed, dict)
                    and parsed.info_parsed.get("jester_variant")
                    == _PUBLIC_CURRENT_VARIANT
                )
                jester_pending_resolution = False
                if current_jester_capture:
                    had_pending_jester_click = (
                        pos in session.jester_pending_activations
                    )
                    raw_jester_callbacks = getattr(
                        parsed,
                        "_jester_raw_callbacks",
                        None,
                    )
                    existing_owns_jester_history = (
                        existing is not None
                        and isinstance(existing.info_parsed, dict)
                        and existing.info_parsed.get("jester_variant")
                        == _PUBLIC_CURRENT_VARIANT
                        and existing.info_parsed.get("callback_ledger_variant")
                        == _ORDERED_CALLBACK_LEDGER_VARIANT
                    )
                    if (
                        raw_jester_callbacks
                        and not had_pending_jester_click
                        and not existing_owns_jester_history
                    ):
                        remaining = _pickable_uses_remaining(mc)
                        if remaining is None:
                            manual_needed.append(
                                f"  #{pos} Jester: [RECOVERY] retained native "
                                "history has no readable current-cycle budget"
                            )
                            continue
                        if remaining > 0:
                            # ResetAfterNight retains actedInfos. Without a
                            # click token or owned prefix, an available actor
                            # proves only that this history is old; keep/enter
                            # the strict no-result shell.
                            if existing is not None:
                                continue
                            parsed = _card_current_jester_no_info(pos)
                            parsed_role_key = "jester"
                    capture_status, capture_status_error = (
                        _classify_jester_auto_capture(
                            existing,
                            parsed,
                            n_cards=session.n_cards,
                            reveal_order=session.reveal_order,
                            baker_rule_version=session.baker_rule_version,
                            rambler_observations=(
                                session.rambler_shut_up_observations
                            ),
                        )
                    )
                    if capture_status_error is not None:
                        role = (
                            mc.get('disguise')
                            or _observed_current_role(mc)
                            or '?'
                        )
                        manual_needed.append(
                            f"  #{pos} {role}: [RECOVERY] "
                            f"{capture_status_error}"
                        )
                        continue
                    if capture_status == "stale":
                        if (
                            _active_cycle_is_spent(mc)
                            and _has_active_clue_result(parsed)
                        ):
                            session.mark_ability_used(parsed.position)
                        continue
                    jester_pending_resolution = (
                        had_pending_jester_click
                        and capture_status == "update"
                    )
                current_repeatable_capture = (
                    existing is not None
                    and memory_role_key in {"fortune_teller", "judge"}
                    and (
                        memory_role_key != "fortune_teller"
                        or session.fortune_teller_rule_version
                        == FORTUNE_TELLER_RULE_VERSION
                    )
                )
                if current_repeatable_capture:
                    capture_status, capture_status_error = (
                        _classify_repeatable_memory_capture(
                            existing,
                            mc,
                            n_cards=session.n_cards,
                            rambler_observations=(
                                session.rambler_shut_up_observations
                            ),
                            fortune_teller_rule_version=(
                                session.fortune_teller_rule_version
                            ),
                        )
                    )
                    if capture_status_error is not None:
                        manual_needed.append(
                            f"  #{pos} {parsed.apparent_role}: [RECOVERY] "
                            f"{capture_status_error}"
                        )
                        continue
                    if capture_status == "stale":
                        if _active_cycle_is_spent(mc):
                            session.mark_ability_used(parsed.position)
                        continue
                if existing:
                    same_role = (
                        _execution_role_key(existing.apparent_role)
                        == _execution_role_key(parsed.apparent_role)
                    )
                    changed = (
                        existing.info_parsed != parsed.info_parsed
                        or existing.info_text != parsed.info_text
                    )
                    active_update = (
                        _active_cycle_is_spent(mc)
                        and same_role
                        and changed
                        and _has_active_clue_result(parsed)
                        and _execution_role_key(parsed.apparent_role)
                        not in {"druid", "jester"}
                    )
                    # Passive reveal callbacks can settle after an initial
                    # memory read. Never let an earlier ordinary/no-info entry
                    # hide a later verified public Rambler replacement.
                    shut_up_update = (
                        same_role
                        and changed
                        and type(parsed.info_parsed.get('shut_up_target')) is int
                    )
                    quote_update = (
                        same_role
                        and changed
                        and parsed.info_parsed.get('quote_observed') is True
                        and type(existing.info_parsed.get('shut_up_target')) is not int
                        and existing.info_parsed.get('quote_observed') is not True
                    )
                    baker_update = (
                        session.baker_rule_version == BAKER_RULE_VERSION
                        and same_role
                        and changed
                        and _execution_role_key(parsed.apparent_role) == "baker"
                        and "original_role" in parsed.info_parsed
                    )
                    # Gossip's Day callback can settle after an initial
                    # no-info snapshot. Replace only that empty placeholder
                    # with a fully provenance-marked current payload; a
                    # nonempty legacy/manual Poet observation remains owned by
                    # the operator and is never silently rewritten.
                    poet_update = (
                        same_role
                        and changed
                        and _execution_role_key(parsed.apparent_role) == "poet"
                        and existing.info_parsed == {}
                        and parsed.info_parsed.get("poet_variant") == POET_VARIANT
                    )
                    gemcrafter_update = (
                        same_role
                        and changed
                        and _execution_role_key(parsed.apparent_role)
                        == "gemcrafter"
                        and existing.info_parsed == {}
                        and parsed.info_parsed.get("gemcrafter_variant")
                        == _PUBLIC_CURRENT_VARIANT
                    )
                    bard_update = (
                        same_role
                        and changed
                        and _execution_role_key(parsed.apparent_role) == "bard"
                        and existing.info_parsed == {}
                        and parsed.info_parsed.get("bard_variant")
                        == _PUBLIC_CURRENT_VARIANT
                    )
                    confessor_update = (
                        same_role
                        and changed
                        and _execution_role_key(parsed.apparent_role)
                        == "confessor"
                        and existing.info_parsed == {}
                        and parsed.info_parsed.get("confessor_variant")
                        == _PUBLIC_CURRENT_VARIANT
                    )
                    druid_update = (
                        same_role
                        and changed
                        and _execution_role_key(parsed.apparent_role) == "druid"
                        and parsed.info_parsed.get("druid_variant")
                        == _PUBLIC_CURRENT_VARIANT
                        and (
                            existing.info_parsed == {}
                            or existing.info_parsed.get("druid_variant")
                            == _PUBLIC_CURRENT_VARIANT
                        )
                    )
                    jester_update = (
                        same_role
                        and changed
                        and _execution_role_key(parsed.apparent_role) == "jester"
                        and parsed.info_parsed.get("jester_variant")
                        == _PUBLIC_CURRENT_VARIANT
                        and (
                            existing.info_parsed == {}
                            or existing.info_parsed.get("jester_variant")
                            == _PUBLIC_CURRENT_VARIANT
                        )
                    )
                    if not (
                        active_update
                        or shut_up_update
                        or quote_update
                        or baker_update
                        or poet_update
                        or gemcrafter_update
                        or bard_update
                        or confessor_update
                        or druid_update
                        or jester_update
                    ):
                        # A prior append-before-decrement read may already have
                        # stored this exact active result without consuming the
                        # local cycle. Reconcile once native remaining uses
                        # settles, even though the evidence itself is unchanged.
                        if (
                            same_role
                            and not changed
                            and _active_cycle_is_spent(mc)
                            and _has_active_clue_result(parsed)
                        ):
                            session.mark_ability_used(parsed.position)
                        continue
                try:
                    session.add_card(parsed, mark_active_result=False)
                except ValueError as exc:
                    if current_druid_capture or current_jester_capture:
                        role = (
                            mc.get('disguise')
                            or _observed_current_role(mc)
                            or '?'
                        )
                        manual_needed.append(
                            f"  #{pos} {role}: [RECOVERY] {exc}"
                        )
                        continue
                    raise
                recorded = next(
                    card
                    for card in session.cards
                    if card.position == parsed.position
                )
                DecisionLog.log_card(recorded)
                if (
                    jester_pending_resolution
                    and pos not in session.jester_pending_activations
                ) or (
                    _active_cycle_is_spent(mc)
                    and _has_active_clue_result(parsed)
                ):
                    session.mark_ability_used(parsed.position)
                verb = "updated" if pos in entered else "entered"
                print(
                    f"  [auto] {verb} #{recorded.position} "
                    f"{recorded.apparent_role}: {recorded.info_parsed}"
                )
                entered[pos] = recorded
                auto_count += 1
            else:
                capture_error = (
                    druid_capture_error
                    or jester_capture_error
                    or rambler_capture_error
                    or baker_capture_error
                    or fortune_capture_error
                )
                if capture_error is not None:
                    role = mc.get('disguise') or _observed_current_role(mc) or '?'
                    manual_needed.append(
                        f"  #{pos} {role}: [RECOVERY] {capture_error}"
                    )
                    continue
                if pos in entered:
                    continue
                clue = mc.get('clue_text', '')
                role = mc.get('disguise') or _observed_current_role(mc) or '?'
                if clue:
                    manual_needed.append(f"  #{pos} {role}: \"{clue}\"")
                else:
                    manual_needed.append(f"  #{pos} {role}: (no clue — active ability?)")

        if auto_count > 0:
            session.save()
        print(f"\n[auto_card] Entered {auto_count} cards automatically.")
        if manual_needed:
            print(f"[auto_card] {len(manual_needed)} cards need manual entry:")
            for line in manual_needed:
                print(line)
        return None

    if cmd == "card":
        card = _parse_card_cli(args, session=session)
        session.add_card(card)
        recorded = next(
            stored
            for stored in session.cards
            if stored.position == card.position
        )
        session.save()
        DecisionLog.log_card(recorded)
        print(
            f"Added #{recorded.position} {recorded.apparent_role}: "
            f"{recorded.info_parsed}"
        )
        return None

    if cmd == "execute":
        pos = int(args[0])
        was_evil = None
        evil_role = None
        was_corrupted = None
        knight_blocked = False
        corruption_explicit = False
        target_entry = next((c for c in session.cards if c.position == pos), None)
        apparent_role = target_entry.apparent_role if target_entry else None
        observed_target = None
        observed_true_role = None
        if len(args) > 1:
            w = args[1].lower()
            if w.startswith(("current=", "revealed=")):
                observed_true_role = _normalize_role_name(
                    args[1].split("=", 1)[1]
                )
            elif w in ("evil", "true", "1", "yes"):
                was_evil = True
                if len(args) > 2:
                    evil_role = _normalize_role_name(args[2])
                for raw in args[3:]:
                    if raw.casefold().startswith(("current=", "revealed=")):
                        observed_true_role = _normalize_role_name(
                            raw.split("=", 1)[1]
                        )
            elif w in ("good", "false", "0", "no"):
                was_evil = False
                knight_blocked = False
                outcome_args = args[2:]
                for raw in outcome_args:
                    c = raw.lower()
                    if c.startswith(("current=", "revealed=")):
                        observed_true_role = _normalize_role_name(
                            raw.split("=", 1)[1]
                        )
                    elif c in ("blocked", "immune", "knight_block") or (
                            c == "knight" and len(outcome_args) == 1):
                        knight_blocked = True
                    elif c in ("corrupted", "corrupt", "c"):
                        was_corrupted = True
                        corruption_explicit = True
                    elif c in ("clean", "uncorrupted", "u", "not_corrupted"):
                        was_corrupted = False
                        corruption_explicit = True
                    elif observed_true_role is None:
                        observed_true_role = _normalize_role_name(raw)
            else:
                was_evil = True
                evil_role = _normalize_role_name(args[1])
                for raw in args[2:]:
                    if raw.casefold().startswith(("current=", "revealed=")):
                        observed_true_role = _normalize_role_name(
                            raw.split("=", 1)[1]
                        )

        if was_evil is True:
            # Runtime alignment and current CharacterData can diverge after a
            # Shaman overwrite. Read the post-action public reveal separately
            # from the supplied original Evil role, so a current Bombardier
            # can trigger its death hook without being recorded as an Evil
            # named Bombardier. ``current=Role`` is the offline equivalent.
            try:
                from memory_reader import MemoryReader
                reader = MemoryReader()
                if reader.open():
                    try:
                        cards = reader.read_board()
                        if cards:
                            observed_target = next(
                                (c for c in cards if c.get('position') == pos),
                                None,
                            )
                    finally:
                        reader.close()
            except Exception as e:
                print(f"  WARNING: Memory reader error ({e})")

            if observed_target:
                if observed_target.get('state') != 'Dead':
                    print(
                        f"  REFUSING BOOKKEEPING: #{pos} is still "
                        f"{observed_target.get('state')}; the execution may "
                        "not have resolved."
                    )
                    return None
                observed_true_role = _observed_current_role(observed_target)
                apparent_role = _execution_apparent_role(
                    observed_target, apparent_role
                )

        if was_evil is False:
            # This command is run only after the in-game action. Memory validates
            # that just-observed result; it is never consulted to choose a target.
            try:
                from memory_reader import MemoryReader
                reader = MemoryReader()
                if reader.open():
                    try:
                        cards = reader.read_board()
                        if cards:
                            observed_target = next(
                                (c for c in cards if c.get('position') == pos),
                                None,
                            )
                    finally:
                        reader.close()
                else:
                    print("  WARNING: Could not open memory reader for post-execution validation")
            except Exception as e:
                print(f"  WARNING: Memory reader error ({e})")

            if observed_target:
                observed_true_role = _observed_current_role(observed_target)
                apparent_role = _execution_apparent_role(observed_target, apparent_role)
                statuses = observed_target.get('statuses', [])
                if observed_target.get('state') == 'Dead':
                    memory_active_corrupted = 'Corrupted' in statuses
                    if (was_corrupted is not None
                            and was_corrupted != memory_active_corrupted):
                        print("  Post-action validation overrides the supplied corruption "
                              "flag with the active memory status.")
                    was_corrupted = memory_active_corrupted
                    if _execution_role_key(observed_true_role) == "drunk":
                        active_word = (
                            "ACTIVE Corrupted" if was_corrupted
                            else "no active Corrupted"
                        )
                        print(f"  Post-action validation: #{pos} {active_word}; "
                              "Drunk execution reports clean")
                    else:
                        corruption_word = "CORRUPTED" if was_corrupted else "NOT corrupted"
                        print(f"  Post-action validation: #{pos} {corruption_word}")

                identity_may_have_moved = (
                    session.current_identity_may_have_moved()
                )
                if (
                    identity_may_have_moved
                    and observed_target.get('state') in ('Alive', 'Revealed')
                ):
                    print(
                        f"  REFUSING BOOKKEEPING: #{pos} survived in a deck "
                        "where current CharacterData may have moved."
                    )
                    print(
                        "  Survival is alignment-neutral; no blocked, Good, "
                        "execution, HP, or decision-log state was recorded."
                    )
                    return None
                if _observed_knight_immunity(
                    observed_target,
                    apparent_role,
                    current_identity_may_have_moved=identity_may_have_moved,
                ):
                    knight_blocked = True
                    print(f"  Post-action validation: #{pos} survived with Knight immunity")
                elif knight_blocked:
                    true_role = observed_true_role or "unknown"
                    shown = apparent_role or "unknown"
                    print(f"  REFUSING BOOKKEEPING: explicit blocked outcome contradicts "
                          f"live #{pos} state/identity ({observed_target.get('state')}, "
                          f"{true_role} showing as {shown}).")
                    print("  Re-check the UI and memory observation before recording the result.")
                    return None
                elif observed_target.get('state') in ('Alive', 'Revealed', 'Hidden'):
                    true_role = observed_true_role or "unknown"
                    shown = apparent_role or "unknown"
                    print(f"  REFUSING BOOKKEEPING: #{pos} is still {observed_target.get('state')} "
                          f"({true_role} showing as {shown}), but identity/status does not "
                          "confirm Knight immunity.")
                    print("  The click may have missed. Re-check the UI; use 'execute "
                          f"{pos} good blocked' only if the game visibly blocked it.")
                    return None
            elif (not knight_blocked and not corruption_explicit
                  and _execution_role_key(apparent_role) == "knight"):
                # Offline/card-only Knight data cannot distinguish a protected
                # Knight from a killable Drunk-as-Knight. Require observation.
                print(f"  Cannot classify apparent Knight #{pos} without post-action memory.")
                print(f"  Use 'execute {pos} good blocked' if it survived, or "
                      f"'execute {pos} good corrupted'/'clean' if it died.")
                return None

            if not knight_blocked and was_corrupted is None:
                print("  WARNING: No corruption flag available. Use 'execute <pos> good "
                      "corrupted' or 'execute <pos> good clean' when offline.")

        if knight_blocked:
            if session.current_identity_may_have_moved():
                print(
                    f"  REFUSING BOOKKEEPING: #{pos} survived in a deck "
                    "where current CharacterData may have moved."
                )
                print(
                    "  Survival is alignment-neutral; no blocked, Good, "
                    "execution, HP, or decision-log state was recorded."
                )
                return None
            # Knight immunity: card survives, confirmed good, no HP loss
            session.record_execution_blocked(pos)
            print(f"Executed #{pos} -> BLOCKED (Knight immunity)")
            print(f"  #{pos} confirmed GOOD. No HP loss. HP: {session.hp}/10")
        else:
            if (
                session.current_identity_may_have_moved()
                and not _is_known_role(observed_true_role)
            ):
                print(
                    f"  REFUSING BOOKKEEPING: #{pos} died in a deck where "
                    "current CharacterData may have moved, but its exact "
                    "public death role was unavailable."
                )
                print(
                    "  Re-read the post-action board or use "
                    f"execute {pos} ... current=Role; no state was saved."
                )
                return None
            terminal_current_bombardier = bool(
                _canonical_terminal_loss_role(observed_true_role)
            )
            recorded_was_evil = was_evil
            recorded_evil_role = evil_role if was_evil is True else None
            session.mark_executed(
                pos,
                recorded_was_evil,
                recorded_evil_role,
                was_corrupted,
                observed_true_role,
            )

            terminal_resource_result = None
            if terminal_current_bombardier and recorded_was_evil is False:
                # An online post-action observation exposes the exact current
                # role and preserved statuses. Offline, NoDamage remains
                # unknowable, so leave local HP untouched and require set_hp.
                if observed_target is not None and "statuses" in observed_target:
                    from knowledge_base import wrong_exec_cost_for
                    observed_statuses = _observed_status_keys(observed_target)
                    no_damage = "nodamage" in observed_statuses
                    damage = (
                        0
                        if no_damage
                        else wrong_exec_cost_for(
                            observed_true_role,
                            default=session.wrong_exec_cost,
                        )
                    )
                    old_hp = session.hp
                    session.hp = _clamped_post_damage_hp(session.hp, damage)
                    terminal_resource_result = (
                        old_hp, session.hp, damage, no_damage
                    )
            session.save()
            DecisionLog.log_execution(
                pos, recorded_was_evil, recorded_evil_role
            )
            tag = (
                f" (evil: {recorded_evil_role})"
                if recorded_evil_role
                else (
                    f" (was_evil={recorded_was_evil})"
                    if recorded_was_evil is not None
                    else ""
                )
            )
            corr_tag = ""
            if terminal_current_bombardier:
                corr_tag = ""
            elif was_corrupted is True:
                corr_tag = (
                    " <ACTIVE Corrupted; observed clean>"
                    if _execution_role_key(observed_true_role) == "drunk"
                    else " <Corrupted>"
                )
            elif was_corrupted is False and was_evil is False:
                corr_tag = " (clean)"
            print(f"Executed #{pos}{tag}{corr_tag}")
            if session.terminal_loss_role:
                print(
                    "  TERMINAL LOSS: a current-role Bombardier died after "
                    "native resource handling."
                )
                if recorded_was_evil is True:
                    print(
                        f"  Runtime Evil execution: no wrong-kill HP cost. "
                        f"HP: {session.hp}/10"
                    )
                    if recorded_evil_role is None:
                        print(
                            "  ORIGIN UNRESOLVED: supply the stable Evil role "
                            "in post-game/manual truth bookkeeping."
                        )
                elif terminal_resource_result is not None:
                    old_hp, new_hp, damage, no_damage = terminal_resource_result
                    detail = (
                        "NoDamage suppressed the wrong-kill cost"
                        if no_damage
                        else f"base wrong-kill cost -{damage}"
                    )
                    print(
                        f"  Runtime Good resource result ({detail}): "
                        f"HP {old_hp} -> {new_hp}."
                    )
                else:
                    print(
                        "  HP outcome unresolved: offline/public evidence "
                        "does not expose whether preserved NoDamage applied."
                    )
                    print(
                        "  Check the live HP display and run: "
                        "set_hp <current_hp>"
                    )
            elif was_evil:
                print(f"  HP: {session.hp}/10 (correct execution, no HP loss)")
            elif was_evil is False:
                if observed_true_role is None:
                    print("  WARNING: Wrong execution recorded, but exact HP damage cannot "
                          "be inferred without the revealed current role.")
                    print("  Check the live HP display and run: set_hp <current_hp>")
                    return None
                if (observed_target is None
                        and _execution_role_key(observed_true_role) == "drunk"
                        and _execution_role_key(apparent_role) == "knight"):
                    print("  WARNING: Offline Drunk-as-Knight damage is ambiguous: the "
                          "revealed clean observation does not expose whether its active "
                          "Corrupted status fired Knight's +4 effect.")
                    print("  Check the live HP display and run: set_hp <current_hp>")
                    return None
                from knowledge_base import execution_cost_for
                cost = execution_cost_for(
                    observed_true_role,
                    apparent_role=apparent_role,
                    was_evil=False,
                    was_corrupted=bool(was_corrupted),
                    # Reaching this branch records a successful, non-blocked
                    # execution. In offline mode that outcome is user-supplied.
                    was_killable=True,
                    default=session.wrong_exec_cost,
                )
                new_hp = _clamped_post_damage_hp(session.hp, cost)
                suffix = ""
                if cost != session.wrong_exec_cost or _execution_role_key(apparent_role) == "knight":
                    shown = f", showing as {apparent_role}" if apparent_role else ""
                    suffix = f" ({observed_true_role or 'unknown'}{shown}: -{cost})"
                print(f"  WARNING: Wrong execution!{suffix} HP {session.hp} -> {new_hp}. Run: set_hp {new_hp}")
            else:
                print(f"  REMINDER: Update HP with 'set_hp <current_hp>' after checking result")

        return None

    if cmd == "pd_check":
        parsed, error = _parse_pd_check_args(
            args,
            session.n_cards,
            session.used_abilities,
            {
                card.position: card.apparent_role
                for card in session.cards
            },
        )
        if error:
            print(f"  ERROR: {error}")
            return None
        try:
            session.add_pd_ability_result(
                parsed["pd_pos"],
                parsed["target"],
                parsed["is_corrupted"],
                parsed["evil_revealed"],
            )
        except ValueError as exc:
            print(f"  ERROR: {exc}")
            return None
        session.save()
        if parsed["is_corrupted"]:
            print(
                f"PD #{parsed['pd_pos']} checked #{parsed['target']}: "
                f"Corrupted, #{parsed['evil_revealed']} is Evil"
            )
        else:
            print(
                f"PD #{parsed['pd_pos']} checked #{parsed['target']}: "
                "Not Corrupted"
            )
        return None

    if cmd == "pd_clear":
        if len(args) != 1:
            print("  ERROR: Usage: pd_clear <pd_pos>")
            return None
        try:
            pd_pos = int(args[0])
        except ValueError:
            print("  ERROR: Plague Doctor position must be an integer")
            return None
        if not 1 <= pd_pos <= session.n_cards:
            print(
                f"  ERROR: Plague Doctor position #{pd_pos} is outside "
                f"1..{session.n_cards}"
            )
            return None
        removed = session.clear_pd_ability_result(pd_pos)
        if not removed:
            print(f"  ERROR: No Plague Doctor result recorded for #{pd_pos}")
            return None
        session.save()
        print(f"Cleared {removed} Plague Doctor result(s) for #{pd_pos}; re-enter with pd_check")
        return None

    if cmd == "solve":
        session.solve()
        return None

    if cmd == "status":
        session.status()
        return None

    if cmd == "confirm_evil":
        pos = int(args[0])
        if pos not in session.confirmed_evil:
            session.confirmed_evil.append(pos)
        session.save()
        print(f"#{pos} confirmed evil")
        return None

    if cmd == "block":
        pos = int(args[0])
        if not session.has_role_in_deck("Witch"):
            print(f"  !! WARNING: No Witch in deck! Only Witch can block cards.")
            print(f"  !! This is likely a click failure. Try re-flipping instead:")
            print(f"  !! Run: python game_loop.py flip {pos}")
            print(f"  !! If you still want to mark as blocked, run: block_force {pos}")
            return None
        if pos not in session.blocked_positions:
            session.blocked_positions.append(pos)
        # Card wasn't actually revealed — remove from reveal_order if flip added it
        if pos in session.reveal_order:
            session.reveal_order.remove(pos)
            session.baker_rule_version = None
        session.save()
        print(f"#{pos} blocked (Witch)")
        return None

    if cmd == "block_force":
        pos = int(args[0])
        if pos not in session.blocked_positions:
            session.blocked_positions.append(pos)
        if pos in session.reveal_order:
            session.reveal_order.remove(pos)
            session.baker_rule_version = None
        session.save()
        print(f"#{pos} force-blocked (override -- no Witch check)")
        return None

    if cmd == "unblock":
        pos = int(args[0])
        if pos in session.blocked_positions:
            session.blocked_positions.remove(pos)
        session.save()
        print(f"#{pos} unblocked")
        return None

    if cmd == "confirm_good":
        pos = int(args[0])
        if pos not in session.confirmed_good:
            session.confirmed_good.append(pos)
        session.save()
        print(f"#{pos} confirmed good")
        return None

    if cmd == "next":
        # Default: auto-execute where safe (definite evil OR forced-safe forced_safe).
        # Use `next --plan` or `next --dry` for print-only inspection mode.
        dry_run = "--plan" in args or "--dry" in args
        if dry_run:
            session.next_action()
        else:
            session.auto_next()
        return None

    if cmd == "auto_next":
        # Explicit alias for `next` (preserved for muscle memory).
        session.auto_next()
        return None

    if cmd == "auto":
        from state_machine import BatchGameRunner
        n_games = 1
        risk = "conservative"
        for arg in args:
            if arg.startswith("--games="):
                n_games = int(arg.split("=")[1])
            elif arg.startswith("--risk="):
                risk = arg.split("=")[1]
            elif arg.isdigit():
                n_games = int(arg)
        runner = BatchGameRunner(n_games=n_games, risk=risk)
        runner.run()
        return None

    if cmd == "auto_loop":
        from state_machine import GameStateMachine
        from memory_reader import get_monitor
        try:
            monitor = get_monitor()
        except Exception:
            monitor = None
        sm = GameStateMachine(session, monitor=monitor)
        # Store on session for resume access
        session._state_machine = sm
        sm.start()
        return None

    if cmd == "resume":
        sm = getattr(session, '_state_machine', None)
        if sm is None:
            print("No active auto_loop to resume. Run auto_loop first.")
        else:
            sm.resume()
        return None

    if cmd == "ability_used":
        pos = int(args[0])
        session.mark_ability_used(pos)
        session.save()
        DecisionLog.log_ability_used(pos)
        print(f"Ability at #{pos} marked as used")
        return None

    if cmd == "slayer_result":
        if len(args) < 3:
            print("  ERROR: Usage: slayer_result <pos> <target> <kill|fail> [role] ...")
            return None
        try:
            slayer_pos = int(args[0])
            target_pos = int(args[1])
        except ValueError:
            print("  ERROR: Slayer and target positions must be integers.")
            return None
        outcome = args[2].lower()
        kill_outcomes = ("kill", "killed", "true", "1", "yes")
        fail_outcomes = ("fail", "failed", "false", "0", "no")
        if outcome not in kill_outcomes + fail_outcomes:
            print(f"  ERROR: Unknown Slayer outcome: {args[2]}")
            print("  Use 'kill' or 'fail'.")
            return None
        killed = outcome in kill_outcomes
        revealed_role = args[3] if len(args) > 3 else None
        was_corrupted = None
        was_evil = None
        for detail in args[4:]:
            detail_key = detail.lower()
            if detail_key in ("corrupted", "clean"):
                if was_corrupted is not None:
                    print("  ERROR: Slayer target status was supplied more than once.")
                    return None
                was_corrupted = detail_key == "corrupted"
            elif detail_key in ("evil", "good"):
                if was_evil is not None:
                    print("  ERROR: Slayer target alignment was supplied more than once.")
                    return None
                was_evil = detail_key == "evil"
            else:
                print(f"  ERROR: Unknown Slayer result detail: {detail}")
                print("  Use 'good'/'evil' and/or 'clean'/'corrupted'.")
                return None
        if killed and not revealed_role:
            print("  ERROR: Slayer kill requires revealed_role! Game reveals the role on kill.")
            print(
                f"  Usage: slayer_result {slayer_pos} {target_pos} kill "
                "<revealed_role> [good|evil] [clean|corrupted]"
            )
            return None
        if not killed and revealed_role:
            print("  ERROR: Failed Slayer attempts do not reveal a role.")
            print(f"  Usage: slayer_result {slayer_pos} {target_pos} fail")
            return None
        old_hp = session.hp
        try:
            session.add_slayer_result(
                slayer_pos,
                target_pos,
                killed,
                revealed_role=revealed_role,
                was_corrupted=was_corrupted,
                was_evil=was_evil,
            )
        except ValueError as exc:
            print(f"  ERROR: {exc}")
            return None
        session.save()
        result_str = f"killed #{target_pos}" if killed else f"couldn't kill #{target_pos}"
        recorded_role = slayer_revealed_role(session.slayer_results[-1])
        if recorded_role:
            result_str += f" (revealed: {recorded_role})"
        DecisionLog.log_slayer_result(
            slayer_pos,
            target_pos,
            killed,
            recorded_role,
        )
        print(f"Slayer #{slayer_pos} {result_str}")
        if session.terminal_loss_role:
            print(
                "  TERMINAL LOSS: a current-role Bombardier died; "
                "native play cannot continue."
            )
        if session.hp != old_hp:
            print(f"  Wrong Slayer kill: HP {old_hp} -> {session.hp}")
        return None

    if cmd == "night_kill":
        if (
            session.has_lilis_night_rule()
            and session.has_role_in_deck("Shaman")
        ):
            print(
                "  ERROR: Lilis+Shaman Night bookkeeping is paused until "
                "ordered Start data movement can establish exact actor count."
            )
            return None
        if not args or not args[0].strip():
            print("  ERROR: Usage: night_kill <pos1,pos2,...> <n_evil>")
            return None
        try:
            positions = [int(x) for x in args[0].split(",")]
            n_evil_among_killed = int(args[1]) if len(args) > 1 else 0
        except ValueError:
            print("  ERROR: Lilis positions and evil-victim count must be integers")
            return None
        # Second arg = how many of the killed cards were evil (usually 0).
        # NOT the total evil count in the game! Lost asc68_v5 0-scenario bug from this confusion.
        try:
            result = session.record_lilis_night_result(
                positions,
                n_evil_among_killed,
            )
        except ValueError as exc:
            print(f"  ERROR: {exc}")
            return None
        session.save()
        confirmed_msg = ""
        if n_evil_among_killed == len(positions) and n_evil_among_killed > 0:
            confirmed_msg = f" (confirmed evil: {['#'+str(p) for p in positions]})"
        print(
            f"Night kills: {['#'+str(p) for p in positions]}, "
            f"{n_evil_among_killed} evil among them{confirmed_msg}"
        )
        print(
            f"  Resolved {result['resolved_events']} Lilis night(s); "
            f"HP: {result['old_hp']} -> {result['new_hp']}"
        )
        if result["reset_abilities"]:
            print(
                "  ResetAfterNight abilities ready again: "
                f"{['#' + str(position) for position in result['reset_abilities']]}"
            )
        return None

    if cmd == "night_no_kill":
        if (
            session.has_lilis_night_rule()
            and session.has_role_in_deck("Shaman")
        ):
            print(
                "  ERROR: Lilis+Shaman Night bookkeeping is paused until "
                "ordered Start data movement can establish exact actor count."
            )
            return None
        try:
            if (session.has_lilis_night_rule()
                    and not session.is_lilis_alive()):
                result = session.record_lilis_post_death_night()
            else:
                result = session.record_lilis_night_result([], 0)
        except ValueError as exc:
            print(f"  ERROR: {exc}")
            return None
        session.save()
        if result["actor_active"]:
            print("Lilis night dealt 2HP but no victim was recorded.")
            print(
                "  No Lilis position can be inferred: a selected clean Knight or "
                "HealthyBluff Doppelganger-as-Knight can survive without a reroll."
            )
        else:
            print(
                "Persistent Night completed after Lilis death: no actor effect, "
                "no victim, and no HP damage."
            )
        print(f"  HP: {result['old_hp']} -> {result['new_hp']}")
        if result["reset_abilities"]:
            print(
                "  ResetAfterNight abilities ready again: "
                f"{['#' + str(position) for position in result['reset_abilities']]}"
            )
        return None

    if cmd == "log":
        label = args[0] if len(args) > 0 else "Claude Reasoning"
        text = args[1] if len(args) > 1 else ""
        DecisionLog.log_custom(label, text)
        print(f"[log] Logged: {label}")
        return None

    if cmd == "game_over":
        result = args[0] if len(args) > 0 else "unknown"
        test_name = args[1] if len(args) > 1 else None
        true_evils_str = None
        notes = ""
        if len(args) > 2:
            raw_candidate = args[2].strip()
            candidate = raw_candidate.strip('"').strip("'")
            if candidate and "=" in candidate:
                true_evils_str = candidate
                notes = " ".join(args[3:]) if len(args) > 3 else ""
            elif candidate:
                notes = " ".join(args[2:])
            elif len(args) > 3:
                notes = " ".join(args[3:])

        # Auto-read true evils from memory_reader if not provided
        if not true_evils_str and test_name:
            try:
                from memory_reader import MemoryReader
                reader = MemoryReader()
                board = None
                if reader.open():
                    try:
                        board = reader.read_board()
                    finally:
                        reader.close()
                if board:
                    runtime_evil_positions = {
                        card["position"]
                        for card in board
                        if card.get("is_evil") is True
                    }
                    if runtime_evil_positions:
                        auto_evils, origin_errors = _resolve_runtime_evil_origins(
                            runtime_evil_positions,
                            session,
                        )
                        if origin_errors:
                            print("[game_over] Stable Evil origin recovery failed:")
                            for err in origin_errors:
                                print(f"  {err}")
                            print("[game_over] Falling back to manual evils entry.")
                            auto_evils = {}
                        else:
                            _auto_cleaned, _auto_errors = _validate_true_evils_against_session(
                                auto_evils,
                                session,
                                expected_runtime_evil_positions=runtime_evil_positions,
                            )
                            if _auto_errors:
                                print("[game_over] Auto-detected evils failed validation:")
                                for err in _auto_errors:
                                    print(f"  {err}")
                                print("[game_over] Falling back to manual evils entry.")
                            else:
                                true_evils_str = ",".join(
                                    f"{p}={r}"
                                    for p, r in sorted(auto_evils.items())
                                )
                                print(
                                    "[game_over] Resolved stable true evils from "
                                    f"runtime-Evil seats: {true_evils_str}"
                                )
                    else:
                        print("[game_over] Memory reader found no runtime-Evil seats")
                else:
                    print("[game_over] Could not read runtime-Evil seats from memory reader")
            except Exception as e:
                print(f"[game_over] Memory reader auto-read failed: {e}")

        DecisionLog.log_game_over(result, session.hp, notes)
        print(f"[game_over] Logged: {result.upper()}, HP={session.hp}")

        from scorecard import record as scorecard_record
        scorecard_record(result, session.hp, test_name or "", notes)

        if test_name and true_evils_str:
            true_evils = _parse_true_evils(true_evils_str)
            cleaned, errors = _validate_true_evils_against_session(true_evils, session)
            if errors:
                print("\n[game_over] Refusing to save test case — validation failed:")
                for err in errors:
                    print(f"  {err}")
                print(f"\n  Re-run: game_over {result} {test_name} <corrected-evils-dict> [notes]")
                print("  NOTE: scorecard and decision log already recorded; only the test")
                print("  case save was aborted. Re-run game_over with corrected evils to")
                print("  save the test case.")
                print("\n=== POST-GAME CHECKLIST ===")
                print("  [ ] Fix evils dict and re-run game_over")
                return None
            _save_and_run_test(test_name, cleaned, notes)
            print("\n--- Full v2 regression (Rust) ---")
            import subprocess as _sp
            try:
                reg = _sp.run(["cargo", "test", "--release", "--test", "simulation"],
                              capture_output=True, text=True, timeout=120)
                for line in reg.stderr.strip().split("\n"):
                    if "test result:" in line or "FAILED" in line:
                        print(f"  {line.strip()}")
                if reg.returncode != 0:
                    print("  WARNING: Regression failures detected! Fix before next game.")
                    # Surface the last ~20 stderr lines so failure details are
                    # visible without rerunning cargo manually.
                    stderr_tail = (reg.stderr or '').splitlines()[-20:]
                    if stderr_tail:
                        print("  --- cargo stderr tail ---")
                        for line in stderr_tail:
                            print(f"    {line}")
            except _sp.TimeoutExpired:
                print("  WARNING: cargo test timed out (120s). Run manually.")
        elif not test_name:
            print("[game_over] Tip: add test name + true evils to auto-save regression test:")
            print("  game_over win/loss <name> <pos=Role,...> [notes]")

        print("\n=== POST-GAME CHECKLIST ===")
        print("  [ ] git add + commit (test case, scorecard, game_session_state.md, code fixes)")
        print("  [ ] git push")
        if result.lower() in ("loss", "l", "lose"):
            print("  [ ] Analyze loss: spawn agent to check critical decisions")
            print("  [ ] Fix solver bugs BEFORE next game")
        return None

    if cmd == "save_test":
        name = args[0] if len(args) > 0 else "unnamed"
        true_evils = {}
        if len(args) > 1:
            raw = args[1]
            if raw.startswith("{"):
                import ast
                true_evils = {int(k): v for k, v in ast.literal_eval(raw).items()}
            else:
                true_evils = _parse_true_evils(raw)
        cleaned, errors = _validate_true_evils_against_session(true_evils, session)
        if errors:
            print("\n[save_test] Refusing to save — validation failed:")
            for err in errors:
                print(f"  {err}")
            return None
        _save_and_run_test(name, cleaned)
        return None

    if cmd == "screenshot":
        name = args[0] if len(args) > 0 else None
        path = session.screenshot(name)
        print(f"Screenshot: {path}")
        return None

    if cmd == "reveal":
        pos = int(args[0])
        session.reveal(pos)
        return None

    if cmd == "deck_view":
        path = session.deck_view()
        print(f"Deck view: {path}")
        return None

    if cmd == "decisions":
        from decision_analysis import cmd_analyze, cmd_analyze_all
        if args:
            cmd_analyze(args[0])
        else:
            cmd_analyze_all()
        return None

    if cmd == "failure_report":
        from decision_analysis import cmd_failure_report
        cmd_failure_report()
        return None

    print(f"Unknown command: {cmd}")
    print("Run 'python game_loop.py' for usage.")
    return None


if __name__ == "__main__":
    main()
