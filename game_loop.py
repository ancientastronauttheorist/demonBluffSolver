"""Game loop adapter: bridges Claude's vision reads to the constraint solver.

Card builder functions, session tracking, CLI interface.
"""

from __future__ import annotations
import atexit
from collections import Counter
import json
import os
import sys
import time
from datetime import datetime
from typing import Optional

from solver import (
    CardInfo,
    DeckComposition,
    GameState,
    RAMBLER_RULE_VERSION,
    SolverResult,
    slayer_revealed_role,
)
from rust_solver import rust_solve_to_objects
from strategy import recommend_action, print_recommendation, evil_probabilities


# ============================================================
# Card Builder Functions
# ============================================================

def card_enlightened(pos: int, direction: str) -> CardInfo:
    """direction: 'CW', 'CCW', or 'Equidistant'"""
    return CardInfo(pos, "Enlightened", info_parsed={"direction": direction})

def card_knitter(pos: int, evil_pairs: int) -> CardInfo:
    return CardInfo(pos, "Knitter", info_parsed={"evil_pairs": evil_pairs})

def card_confessor(pos: int, dizzy: bool) -> CardInfo:
    return CardInfo(pos, "Confessor", info_parsed={"dizzy": dizzy})

def card_gemcrafter(pos: int, good_position: int) -> CardInfo:
    return CardInfo(pos, "Gemcrafter", info_parsed={"good_position": good_position})

def card_lover(pos: int, evil_adjacent: int) -> CardInfo:
    return CardInfo(pos, "Lover", info_parsed={"evil_adjacent": evil_adjacent})

def card_scout(pos: int, evil_role: str, distance: int) -> CardInfo:
    return CardInfo(pos, "Scout", info_parsed={"evil_role": evil_role, "distance": distance})

def card_bard(pos: int, corruption_distance: int) -> CardInfo:
    # 0 means "no corrupted characters exist" — map to -1 sentinel
    if corruption_distance == 0:
        corruption_distance = -1
    return CardInfo(pos, "Bard", info_parsed={"corruption_distance": corruption_distance})

def card_fortune_teller(pos: int, targets: list[int], has_evil: bool) -> CardInfo:
    return CardInfo(pos, "Fortune Teller", info_parsed={"targets": targets, "has_evil": has_evil})

def card_oracle(pos: int, targets: list[int], minion_role: str) -> CardInfo:
    return CardInfo(pos, "Oracle", info_parsed={"targets": targets, "minion_role": minion_role})

def card_medium(pos: int, good_position: int, good_role: str) -> CardInfo:
    return CardInfo(pos, "Medium", info_parsed={"good_position": good_position, "good_role": good_role})

def card_hunter(pos: int, distance: int) -> CardInfo:
    return CardInfo(pos, "Hunter", info_parsed={"distance": distance})

def card_architect(pos: int, side: str) -> CardInfo:
    """side: 'Left', 'Right', or 'Equal'"""
    return CardInfo(pos, "Architect", info_parsed={"side": side})

def card_empress(pos: int, targets: list[int]) -> CardInfo:
    return CardInfo(pos, "Empress", info_parsed={"targets": targets})

def card_witness(pos: int, affected_position: int) -> CardInfo:
    return CardInfo(pos, "Witness", info_parsed={"affected_position": affected_position})

def card_jester(pos: int, targets: list[int], evil_count: int) -> CardInfo:
    return CardInfo(pos, "Jester", info_parsed={"targets": targets, "evil_count": evil_count})

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
    if role in {"fortune_teller", "jester", "druid", "judge"}:
        return bool(info)
    return False


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

def card_druid(pos: int, targets: list[int], found_outcast: Optional[str] = None) -> CardInfo:
    return CardInfo(pos, "Druid", info_parsed={"targets": targets, "found_outcast": found_outcast})

def card_bishop(pos: int, targets: list[int], types: list[str] = None) -> CardInfo:
    info = {"targets": targets}
    if types:
        info["types"] = types
    return CardInfo(pos, "Bishop", info_parsed=info)

def card_bounty_hunter(pos: int, evil_position: int) -> CardInfo:
    """Pseudo-clue used for Poet's direct evil-call variant."""
    return CardInfo(pos, "Poet", info_parsed={
        "copied_role": "Bounty Hunter",
        "evil_position": evil_position,
    })


def card_poet_with_info(pos: int, copied_role: str, copied_args: list[str]) -> CardInfo:
    """Poet clue parser.

    Usage: card poet <pos> <copied_role> <copied_role_args...>
    Examples:
        card poet 5 knitter 0          (Poet gave Knitter-style clue)
        card poet 3 lover 2            (Poet gave Lover-style clue)
        card poet 7 architect left     (Poet gave Architect-style clue)
        card poet 2 gemcrafter 5       (Poet gave Gemcrafter-style clue)
        card poet 4 bard 1             (Poet gave Bard-style clue)
        card poet 1 bounty_hunter 6    (Poet directly named #6 as Evil)
    """
    copied_key = copied_role.lower().replace(" ", "_")
    if copied_key in ("bounty_hunter", "bountyhunter", "evil"):
        return card_bounty_hunter(pos, int(copied_args[0]))

    # Build the copied role's info_parsed by delegating to _parse_card_cli
    fake_args = [copied_role, str(pos)] + copied_args
    copied_card = _parse_card_cli(fake_args)
    info = copied_card.info_parsed.copy()
    info["copied_role"] = copied_card.apparent_role  # Clue type, not necessarily in play
    return CardInfo(pos, "Poet", info_parsed=info)


def card_baker(pos: int, original_role: str) -> CardInfo:
    """Baker: 'I am the original Baker' or 'I was a <Role>'.

    original_role: 'original' for the first Baker, or the Villager role name
    the Baker claims to have been before conversion.
    """
    return CardInfo(pos, "Baker", info_parsed={"original_role": original_role})


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


def _execution_apparent_role(observed: dict | None,
                             fallback_role: str | None = None) -> str | None:
    """Return the displayed role from a post-action memory observation.

    The live bluff pointer is preferred.  A card entry is UI-derived and is a
    safe fallback when memory has no bluff object; true identity is last.
    """
    if observed:
        return observed.get("disguise") or fallback_role or observed.get("true_role")
    return fallback_role


def _observed_knight_immunity(observed: dict | None,
                              fallback_role: str | None = None) -> bool:
    """Whether a just-attempted execution is natively consistent with immunity.

    This is intentionally a post-action validator, never a pre-click decision
    helper.  A clean good true Knight is protected.  A Doppelganger showing as
    Knight is protected only while HealthyBluff makes it delegate protection to
    the bluff role.  Drunk-as-Knight and other merely apparent Knights remain
    killable and must not be auto-labelled immune from deck/card data alone.
    """
    if not observed or observed.get("state") not in ("Alive", "Revealed"):
        return False
    if observed.get("is_evil") is not False:
        return False
    apparent_role = _execution_apparent_role(observed, fallback_role)
    if _execution_role_key(apparent_role) != "knight":
        return False

    true_role = _execution_role_key(observed.get("true_role"))
    statuses = {
        _execution_role_key(status).replace(" ", "")
        for status in observed.get("statuses", [])
    }
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
        self.executed_good_roles: dict[int, str] = {}  # pos -> revealed true role after a wrong execution
        self.board_villager_count: Optional[int] = None  # Normalized pre-Start header V count
        self.board_outcast_count: Optional[int] = None   # Normalized pre-Start header O count
        self.board_count_provenance: str = "legacy_unknown"
        self.rambler_rule_version: Optional[str] = RAMBLER_RULE_VERSION
        self.rambler_shut_up_observations: list[dict] = []
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
        """Whether the persistent every-four-reveals Night rule is installed."""
        return self.lilis_deck_count() > 0

    def has_duplicate_lilis(self) -> bool:
        """Whether live Night effects exceed the Standard one-actor model."""
        return self.lilis_deck_count() > 1

    def is_lilis_alive(self) -> bool:
        """Whether at least one public Lilis actor is not known dead.

        The NightModeRule persists after every Lilis dies, so callers deciding
        reveal batching must use ``has_lilis_night_rule`` instead. This method
        is only for actor effects such as victim selection and 2 HP damage.
        """
        deck_count = self.lilis_deck_count()
        if deck_count == 0:
            return False
        known_dead = sum(
            _normalize_role_name(role) == "Lilis"
            for role in self.executed_evil_roles.values()
        )
        return known_dead < deck_count

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
            for role in self.executed_evil_roles.values()
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

    def add_card(self, card: CardInfo):
        role_key = card.apparent_role.lower().replace(" ", "_")
        existing = next(
            (previous for previous in self.cards if previous.position == card.position),
            None,
        )
        existing_role_key = (
            existing.apparent_role.lower().replace(" ", "_")
            if existing is not None else None
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

        # Track reveal order (first entry per position)
        if card.position not in self.reveal_order:
            self.reveal_order.append(card.position)
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
        existing_shut_up_target = (
            existing.info_parsed.get("shut_up_target")
            if existing is not None else None
        )

        def merge_judge_history(
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
                    else merge_judge_history(
                        list(older_rounds),
                        current_judge_history,
                    )
                )
            elif reset_judge_event:
                observations = (
                    list(prior_judge_history)
                    if incoming_is_shut_up
                    else merge_judge_history(
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

        # The ledger is chronological public-event state, not an audit log of
        # parser corrections. Editing a non-reset event replaces/removes its
        # current record in place, preserving global event order. A later
        # ResetAfterNight Judge event appends a new record even if identical.
        incoming_is_shut_up = type(incoming_shut_up_target) is int
        existing_is_shut_up = type(existing_shut_up_target) is int
        incoming_is_event = role_key != "judge" or judge_event_observed

        if incoming_is_event:
            new_record = (
                {
                    "speaker_position": card.position,
                    "shut_up_target": incoming_shut_up_target,
                }
                if incoming_is_shut_up else None
            )
            if existing is None or reset_judge_event:
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
        if role_key in active_result_roles and _has_active_clue_result(card):
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

    def mark_executed(self, pos: int, was_evil: Optional[bool] = None,
                      evil_role: Optional[str] = None,
                      was_corrupted: Optional[bool] = None,
                      true_role: Optional[str] = None):
        if pos not in self.executed:
            self.executed.append(pos)
        if was_evil is True and pos not in self.confirmed_evil:
            self.confirmed_evil.append(pos)
        elif was_evil is False and pos not in self.confirmed_good:
            self.confirmed_good.append(pos)
        if evil_role:
            self.executed_evil_roles[pos] = evil_role.replace(' ', '_')
            if _normalize_role_name(evil_role) == "Witch":
                self.release_witch_blocks(f"known Witch death at #{pos}")
        # Execution bookkeeping exposes Drunk as clean even when its active
        # Corrupted status drove role effects such as Knight's +4 damage.
        if was_evil is False and was_corrupted is not None:
            observed_corrupted = (
                False if _execution_role_key(true_role) == "drunk"
                else was_corrupted
            )
            self.executed_good_corrupted[pos] = observed_corrupted
        if (was_evil is False and true_role
                and true_role.strip().lower() not in {"unknown", "?", "none"}):
            self.executed_good_roles[pos] = true_role.replace(' ', '_')

    def record_execution_blocked(self, pos: int,
                                 reason: str = "Knight immunity") -> None:
        """Persist a confirmed-good execution attempt that left the card alive."""
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

    def reset_after_night_abilities(self) -> list[int]:
        """Apply shipped ResetAfterNight usage to the session model.

        The current public roster audit has proven this usage mode for Judge.
        Keep its accumulated clue evidence, but make each apparent Judge
        available to the recommender again after a completed night.
        """
        from knowledge_base import get_card

        resettable = set()
        for card in self.cards:
            card_def = get_card(card.apparent_role)
            if card_def and card_def.ability_resets_after_night:
                resettable.add(card.position)
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
        if self.has_duplicate_lilis():
            raise ValueError(
                "duplicate Lilis live nights are unsupported: multiple actors "
                "can charge HP while colliding on one delayed victim"
            )
        if not self.has_lilis_night_rule():
            raise ValueError("no Lilis Night rule exists in this deck")
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

        if n_evil_among_killed > 0:
            self.release_witch_blocks(
                "an evil Lilis victim may have been Witch; public re-probe required"
            )
        if positions and n_evil_among_killed == len(positions):
            for position in positions:
                if position not in self.confirmed_evil:
                    self.confirmed_evil.append(position)

        reset_abilities = self.reset_after_night_abilities()
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
        if self.has_duplicate_lilis():
            raise ValueError(
                "duplicate Lilis live nights are unsupported: actor liveness "
                "and delayed-victim collisions are not represented"
            )
        if not self.has_lilis_night_rule():
            raise ValueError("no Lilis Night rule exists in this deck")
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
        Shaman/stale-register compositions can also make another Good-class
        role enter the kill branch, so those outcomes must supply ``was_evil``
        explicitly for accurate public HP and confirmation bookkeeping.
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
            role_def = get_card(revealed_role)
            if role_def is None:
                raise ValueError(f"Unknown Slayer revealed role: {revealed_role}")
            canonical_role = role_def.name.replace(" ", "_")
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
                        f"Revealed Evil role {role_def.name} cannot be recorded as runtime Good"
                    )
                target_was_evil = True
            elif was_evil is not None:
                target_was_evil = was_evil
            # A Good-class revealed role can still live on a preserved
            # runtime-Evil Shaman destination. Without the public HP outcome,
            # keep that alignment unresolved instead of asking hidden memory.

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
                        if role_def.alignment == Alignment.EVIL
                        else None
                    ),
                )
            elif target_was_evil is False:
                self.mark_executed(
                    target_pos,
                    was_evil=False,
                    was_corrupted=was_corrupted,
                    true_role=canonical_role,
                )
                # KillAndReveal publishes Character.Kill and therefore base
                # wrong-kill damage, but never runs OnExecuted. In particular,
                # a Slayer-killed corrupted Good Knight costs 5, not 5+4.
                damage = wrong_exec_cost_for(
                    canonical_role, default=self.wrong_exec_cost,
                )
                self.hp = _clamped_post_damage_hp(self.hp, damage)
            else:
                # Kill and revealed current role are public facts. Runtime
                # alignment, confirmation maps, corruption, and HP remain
                # unresolved until the visible HP result is entered.
                if target_pos not in self.executed:
                    self.executed.append(target_pos)

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
        session.reveal_order = list(state.reveal_order)
        session.executed_good_corrupted = dict(getattr(state, 'executed_good_corrupted', {}))
        session.executed_good_roles = dict(getattr(state, 'executed_good_roles', {}))
        session.used_abilities = list(used_abilities or [])
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
        state = self.to_game_state()
        result = self._solve(state)
        print(f"\n=== SOLVER RESULT ===")
        for line in result.reasoning:
            print(f"  {line}")
        if result.definite_evil:
            print(f"\n  >> EXECUTE: {['#'+str(p) for p in result.definite_evil]}")
        if result.bombardier_positions:
            print(f"  >> DO NOT EXECUTE (Bombardier): {['#'+str(p) for p in result.bombardier_positions]}")
        if result.n_surviving == 0:
            print(f"\n  !! NO VALID SCENARIOS — check your input data !!")
        elif not result.definite_evil:
            print(f"\n  >> No definite evil yet. Reveal more cards.")
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
        action = print_recommendation(state, result, self.used_abilities)
        DecisionLog.log_recommendation(action)
        return action

    def auto_execute(self, pos: int, result, monitor=None, forced_safe: bool = False) -> dict:
        """Perform in-game execution click sequence, verify via memory_reader, record result.

        Uses MemoryMonitor.wait_for() when available for faster, smarter waits.
        Falls back to fixed sleeps if no monitor provided.

        Args:
            forced_safe: If True, bypass Bombardier guard (lookahead proved safety).

        Returns: {"success": bool, "blocked": bool, "was_evil": bool|None,
                  "evil_role": str|None, "error": str|None}
        """
        import template_match as _tm
        import mouse as _mouse
        from game_utils import all_game_card_coords

        # Bombardier hard guard (Issue #9: bypass when forced_safe is True)
        if pos in result.bombardier_positions and not forced_safe:
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
            if _observed_knight_immunity(target_card, fallback_role):
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
                true_role = target_card.get('true_role') or "unknown"
                return {"success": False, "was_evil": None, "evil_role": None,
                        "error": (f"Card survived, but post-action identity/status does not "
                                  f"confirm immunity ({true_role} showing as {apparent_role}); "
                                  "the click may have missed")}
            return {"success": False, "was_evil": None, "evil_role": None,
                    "error": f"Card state is {target_card['state']}, expected Dead"}

        # Step 5: Determine result
        was_evil = target_card['is_evil']
        evil_role = target_card['true_role'] if was_evil else None

        # Step 6: Record into session
        was_corrupted = None
        if not was_evil:
            statuses = target_card.get('statuses', [])
            was_corrupted = 'Corrupted' in statuses

        self.mark_executed(
            pos,
            was_evil,
            evil_role,
            was_corrupted,
            target_card.get('true_role') if not was_evil else None,
        )

        # Step 7: HP update
        if not was_evil:
            from knowledge_base import execution_cost_for
            true_role = target_card.get('true_role')
            target_entry = next((c for c in self.cards if c.position == pos), None)
            fallback_role = target_entry.apparent_role if target_entry else None
            apparent_role = _execution_apparent_role(target_card, fallback_role)
            cost = execution_cost_for(
                true_role,
                apparent_role=apparent_role,
                was_evil=False,
                was_corrupted=bool(was_corrupted),
                was_killable=True,
                default=self.wrong_exec_cost,
            )
            old_hp = self.hp
            self.hp = _clamped_post_damage_hp(self.hp, cost)
            suffix = ""
            if cost != self.wrong_exec_cost:
                shown = f", showing as {apparent_role}" if apparent_role else ""
                suffix = f" ({true_role or 'unknown'}{shown}: -{cost})"
            print(f"  [auto_exec] WRONG EXECUTION! HP {old_hp} -> {self.hp}{suffix}")
        else:
            print(f"  [auto_exec] Correct execution. HP remains {self.hp}")

        self.save()
        DecisionLog.log_execution(pos, was_evil, evil_role)

        return {"success": True, "blocked": False, "was_evil": was_evil,
                "evil_role": evil_role, "error": None}

    def auto_use_ability(self, action, monitor=None) -> dict:
        """Perform in-game active-ability activation + target clicks + auto-parse.

        Template: auto_execute, but for active abilities like Jester/Dreamer/FT/Judge.
        Slayer still uses its dedicated `slayer_result` command. Plague Doctor
        is parsed from its exact public speech text and recorded through the
        same state path as `pd_check`.

        Flow:
          1. Click active card → game enters target-selection mode
          2. Click each target in order
          3. Wait for memory to show uses>0 or acted_infos populated
          4. Call _parse_clue_from_memory to extract info_parsed
          5. session.add_card() + session.mark_ability_used()

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

        # Slayer's kill/death result still needs its dedicated execution path.
        if ability_name == "slayer":
            return {"success": False, "info_parsed": None,
                    "error": f"{action.ability_name} requires manual handling (use slayer_result)"}

        if ability_name == "dreamer" and len(targets) != 2:
            return {"success": False, "info_parsed": None,
                    "error": f"Dreamer requires exactly 2 targets, got {targets}"}
        if ability_name == "judge" and len(targets) != 1:
            return {"success": False, "info_parsed": None,
                    "error": f"Judge requires exactly 1 target, got {targets}"}
        if ability_name == "plague_doctor" and len(targets) != 1:
            return {"success": False, "info_parsed": None,
                    "error": f"Plague Doctor requires exactly 1 target, got {targets}"}
        if ability_name in {"judge", "plague_doctor"}:
            actor = next((card for card in self.cards if card.position == pos), None)
            actor_role = (
                actor.apparent_role.lower().replace(" ", "_")
                if actor is not None else None
            )
            if actor_role != ability_name:
                shown = actor.apparent_role if actor is not None else "unrevealed"
                display_name = (
                    "Plague Doctor" if ability_name == "plague_doctor"
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
            # Judge's picker-first OnClick branch accepts every board card,
            # including self and targets with their own unused active ability.
            if ability_name == "judge":
                continue
            # Native PD explicitly supports self-targeting. Once its picker is
            # active, the repeated card click is routed to that picker.
            if ability_name == "plague_doctor" and t == pos:
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

        # Judge is ResetAfterNight, so an old acted-info list is expected on
        # later uses. Snapshot the latest event before any click and accept
        # only a newly appended or mutated current event afterward.
        judge_pre_event = None
        if ability_name == "judge":
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
                            "Cannot open memory reader for pre-click Judge "
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
                    "error": "Cannot read board for pre-click Judge event snapshot",
                }
            before_card = next(
                (card for card in before_board if card.get('position') == pos),
                None,
            )
            if before_card is None:
                return {
                    "success": False,
                    "info_parsed": None,
                    "error": f"Judge #{pos} missing from pre-click memory snapshot",
                }
            judge_pre_event = _latest_acted_event_fingerprint(before_card)

        # Step 1: Click active card to enter target-selection mode
        x, y = coords[pos]
        print(f"  [auto_ability] Activating {action.ability_name} at #{pos} ({x},{y})...")
        try:
            _tm.safe_click_at(x, y, f"activate_card{pos}")
        except Exception as e:
            return {"success": False, "info_parsed": None,
                    "error": f"Failed to click active card: {e}"}
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

        # Step 3: Wait for ability result in memory. Public Dreamer can set the act
        # flag and clue text while leaving uses at 0.
        print(f"  [auto_ability] Waiting for ability result...")
        target_card_data = None

        def _ability_resolved(board):
            if not board:
                return False
            card = next((c for c in board if c['position'] == pos), None)
            if not card:
                return False
            if ability_name == "judge":
                latest = _latest_acted_event_fingerprint(card)
                return latest is not None and latest != judge_pre_event
            return (
                card.get('uses', 0) > 0
                or bool(card.get('acted_infos'))
                or bool(card.get('ability_used') and card.get('clue_text'))
            )

        if monitor and monitor.is_healthy():
            resolved = monitor.wait_for(_ability_resolved, timeout=6, min_delay=0.8)
            if resolved:
                board = monitor.get_board()
                target_card_data = next((c for c in board if c['position'] == pos), None) if board else None
        else:
            time.sleep(1.5)  # initial animation delay
            from memory_reader import MemoryReader
            reader = MemoryReader()
            if not reader.open():
                return {"success": False, "info_parsed": None,
                        "error": "Cannot open memory reader for ability verification"}
            try:
                for attempt in range(5):
                    cards = reader.read_board()
                    if cards:
                        target_card_data = next((c for c in cards if c['position'] == pos), None)
                        if target_card_data and _ability_resolved(cards):
                            break
                    if attempt < 4:
                        time.sleep(0.7)
            finally:
                reader.close()

        if not target_card_data:
            return {"success": False, "info_parsed": None,
                    "error": f"Position #{pos} not found in memory reader after activation"}
        if ability_name == "judge":
            latest_judge_event = _latest_acted_event_fingerprint(target_card_data)
            has_recorded_result = (
                latest_judge_event is not None
                and latest_judge_event != judge_pre_event
            )
        else:
            has_recorded_result = (
                target_card_data.get('uses', 0) > 0
                or bool(target_card_data.get('acted_infos'))
                or bool(target_card_data.get('ability_used') and target_card_data.get('clue_text'))
            )
        if not has_recorded_result:
            if ability_name == "judge":
                return {
                    "success": False,
                    "info_parsed": None,
                    "error": (
                        "Judge result did not produce a new or changed latest "
                        "acted-info event — click may have missed"
                    ),
                }
            return {"success": False, "info_parsed": None,
                    "error": f"Ability result not detected (uses=0, acted_infos empty) — click may have missed"}

        # Rambler2 replaces an adjacent actor's normal result surface.  Handle
        # that before role-specific strict parsers: the emitted references now
        # name the Rambler, not the target(s) the active role originally chose.
        parsed, interruption_error = _card_from_rambler_interruption(
            target_card_data,
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
                target_card_data,
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

        # Step 4b: Judge has a strict one-target public result boundary.
        if parsed is None and ability_name == "judge":
            parsed, parse_error = _parse_judge_result_from_memory(
                target_card_data,
                expected_target=targets[0],
                n_cards=self.n_cards,
            )
            if parse_error:
                return {"success": False, "info_parsed": None,
                        "error": parse_error}
        elif parsed is None:
            # Parse ordinary clue-producing abilities via auto_card.
            parsed = _parse_clue_from_memory(
                target_card_data,
                n_cards=self.n_cards,
            )
        if parsed is None:
            return {"success": False, "info_parsed": None,
                    "error": f"Could not parse ability result from memory data"}
        if not parsed.info_parsed:
            return {"success": False, "info_parsed": None,
                    "error": f"Parser returned empty info_parsed for {action.ability_name}"}

        # Step 5: Update session
        self.add_card(parsed)
        self.mark_ability_used(pos)
        self.save()
        DecisionLog.log_card(parsed)
        DecisionLog.log_ability_used(pos)

        print(f"  [auto_ability] {action.ability_name} #{pos} -> {targets}: {parsed.info_parsed}")
        return {"success": True, "info_parsed": parsed.info_parsed, "error": None}

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
        action = print_recommendation(state, result, self.used_abilities)
        DecisionLog.log_recommendation(action)

        # Route USE_ABILITY to auto_use_ability. Slayer still uses its
        # dedicated kill-result command.
        if action.action_type == "use_ability":
            ability_name_lower = (action.ability_name or "").lower().replace(" ", "_")
            if ability_name_lower == "slayer":
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

        if pos in result.bombardier_positions and not is_forced_safe:
            print(f"\n  [auto_next] #{pos} is potential Bombardier (not forced-safe) — manual decision needed.")
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

    The first acted-info reference must be the character the automation
    clicked. The visible speech text supplies the status and any revealed
    position. A native self-check can retain a hidden second reference even
    though it displays clean; this parser deliberately ignores that reference.
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

    clue = (card.get('clue_text') or '').strip()
    infos = card.get('acted_infos') or []
    targets = infos[0].get('targets', []) if infos else []
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
    role = card.get('disguise') or card.get('true_role', '')
    role_key = role.lower().replace(' ', '_')
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
    role = card.get('disguise') or card.get('true_role', '')
    if role.lower().replace(' ', '_') == 'rambler':
        if n_cards is None:
            return None, (
                "Rambler quote capture requires the board size to validate "
                "its circular neighbor references"
            )
        return _card_from_rambler_quote(card, n_cards=n_cards)
    return None, None


def _parse_clue_from_memory(
    card: dict,
    *,
    n_cards: Optional[int] = None,
) -> Optional[CardInfo]:
    """Parse memory reader card data into a CardInfo, or None if unparseable.

    Handles passive clues read from savedAct/actedInfos/runtimeData.
    Active abilities (FT, Judge, Jester, Druid, Dreamer, Slayer, PD) are
    guarded by the ability_used flag — stale clue data from prior villages
    is ignored unless the ability was actually used this game.
    """
    import re
    pos = card['position']
    role = card.get('disguise') or card.get('true_role', '')
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
    ability_used = card.get('ability_used', False)

    # --- Guard: active-ability-only roles with unused abilities ---
    # These roles have NO passive speech bubble. If ability hasn't been used,
    # any clue_text/acted_infos is stale from a previous village — ignore it.
    # Prefer `uses`, but public Dreamer currently sets ability_used=True while leaving
    # uses at 0 after a real fire. PD is stricter because its game-start setup
    # can set the act flag before the active check ability is used.
    ACTIVE_ONLY_ROLES = {
        'dreamer', 'druid', 'fortune_teller', 'jester', 'judge',
        'slayer', 'plague_doctor',
    }
    uses_count = card.get('uses', 0)
    active_fired = uses_count > 0 or (ability_used and role_lower != 'plague_doctor')
    if role_lower in ACTIVE_ONLY_ROLES and not active_fired:
        return card_no_info(pos, role)

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

    # --- RuntimeData: Enlightened direction (always reliable) ---
    if rd and rd.get('type') == 'direction':
        return card_enlightened(pos, rd['direction'])

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

    # --- Baker: prefer clue_text over runtime_data ---
    # Runtime original_role can mismatch displayed text (e.g., Shaman games,
    # or when Baker chain text differs from internal tracking).
    if rd and rd.get('type') == 'baker':
        # Try clue text first ("I was a <Role>" or "I am the original Baker")
        m = re.search(r'I was (?:a |an )?(.+)', clue, re.IGNORECASE)
        if m:
            claimed = m.group(1).strip()
            # "I was a Baker" means converted FROM Baker (rare, but valid).
            # Do NOT convert to 'original' -- that means "I am the original Baker".
            return card_baker(pos, claimed)
        if 'original' in clue.lower():
            return card_baker(pos, 'original')
        # Fallback to runtime_data
        original = rd.get('original_role')
        if not original or original == '?':
            return card_baker(pos, 'original')
        if original.lower() == 'baker':
            # runtime says original was Baker but clue didn't match "I was a" pattern.
            # If clue says "I am the original Baker" we already returned above.
            # Otherwise this is ambiguous -- treat as original.
            return card_baker(pos, 'original')
        return card_baker(pos, original)

    # --- Knitter: "X evil pair(s)" / "X pairs of Evil" / "Evils are not adjacent" ---
    if role_lower == 'knitter':
        if 'not adjacent' in clue.lower() or 'no evil' in clue.lower():
            return card_knitter(pos, 0)
        m = re.search(r'(\d+)\s+(?:evil\s+)?pair', clue, re.IGNORECASE)
        if m:
            return card_knitter(pos, int(m.group(1)))

    # --- Confessor: "dizzy" or "feeling good" ---
    if role_lower == 'confessor':
        if 'dizzy' in clue.lower() or 'dirty' in clue.lower():
            return card_confessor(pos, True)
        if 'good' in clue.lower() or 'clean' in clue.lower():
            return card_confessor(pos, False)

    # --- Bard: "no Corrupted" or "X card(s) away from Corrupted" ---
    if role_lower == 'bard':
        if 'no corrupted' in clue.lower() or 'are not corrupted' in clue.lower():
            return card_bard(pos, -1)
        m = re.search(r'(\d+)\s+card', clue, re.IGNORECASE)
        if m:
            return card_bard(pos, int(m.group(1)))

    # --- Lover: "X of my neighbors are evil" or "none" ---
    if role_lower == 'lover':
        m = re.search(r'(\d+)', clue)
        if m:
            return card_lover(pos, int(m.group(1)))
        if 'none' in clue.lower() or 'no' in clue.lower():
            return card_lover(pos, 0)

    # --- Hunter: "nearest evil is X away" ---
    if role_lower == 'hunter':
        m = re.search(r'(\d+)', clue)
        if m:
            return card_hunter(pos, int(m.group(1)))

    # --- Architect: "Left"/"Right"/"Equal" ---
    if role_lower == 'architect':
        cl = clue.lower()
        if 'left' in cl:
            return card_architect(pos, 'Left')
        if 'right' in cl:
            return card_architect(pos, 'Right')
        if 'equal' in cl:
            return card_architect(pos, 'Equal')

    # --- Empress: targets from actedInfos ---
    if role_lower == 'empress' and targets:
        return card_empress(pos, targets)

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

    # --- Gemcrafter: single target ---
    if role_lower == 'gemcrafter' and targets:
        return card_gemcrafter(pos, targets[0])

    # --- Fortune Teller: "Is #X or #Y Evil?: True/False" ---
    if role_lower == 'fortune_teller' and targets:
        has_evil = 'true' in clue.lower()
        return card_fortune_teller(pos, targets, has_evil)

    # --- Jester: targets + evil count from clue ---
    if role_lower == 'jester' and targets:
        m = re.search(r'(\d+)\s+(?:of them |are |is )?\s*evil', clue, re.IGNORECASE)
        if m:
            return card_jester(pos, targets, int(m.group(1)))
        # "none of them are evil"
        if 'none' in clue.lower() or 'no' in clue.lower():
            return card_jester(pos, targets, 0)

    # --- Bishop: targets + types from clue ---
    if role_lower == 'bishop' and targets:
        types = []
        for t in ['Villager', 'Outcast', 'Minion', 'Demon']:
            if t.lower() in clue.lower():
                types.append(t)
        if types:
            return card_bishop(pos, targets, types)
        return card_bishop(pos, targets)

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
        # Public shipped Dreamer: IDs serialized in the clue are authoritative.
        ambiguous = _parse_ambiguous_among(clue)
        if ambiguous:
            amb_targets, options = ambiguous
            return card_dreamer_ambiguous(
                pos,
                amb_targets,
                options,
                info_text=clue,
            )
        cabbage_targets = _parse_cabbage_between(clue)
        if cabbage_targets:
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
            return card_dreamer(
                pos,
                int(m.group(1)),
                m.group(2).strip(),
                info_text=clue,
            )

    # --- Druid: "Among #A, #B, #C there is: <Outcast>" or no outcasts ---
    if role_lower == 'druid' and targets:
        cl = clue.lower()
        if 'no outcast' in cl or 'none' in cl:
            return card_druid(pos, targets, None)
        m = re.search(r'there\s+(?:is|was)\s*:?\s*([A-Za-z][A-Za-z ]*)', clue, re.IGNORECASE)
        if m:
            found = m.group(1).strip().rstrip('.!').replace(' ', '_')
            return card_druid(pos, targets, found)

    # --- Oracle: targets + minion role ---
    if role_lower == 'oracle' and targets:
        # Look for a role name in the clue
        m = re.search(r'is\s+(?:a\s+)?(\w[\w\s]*)', clue, re.IGNORECASE)
        if m:
            minion_role = m.group(1).strip().replace(' ', '_')
            return card_oracle(pos, targets, minion_role)

    # --- Baker: "I was a <Role>" or "I am the original Baker" ---
    if role_lower == 'baker':
        m = re.search(r'I was (?:a |an )?(.+)', clue, re.IGNORECASE)
        if m:
            claimed = m.group(1).strip()
            # "I was a Baker" = converted from Baker (keep as 'Baker', not 'original')
            return card_baker(pos, claimed)
        if 'original' in clue.lower() or not clue.strip():
            return card_baker(pos, 'original')

    # --- Scout: "<Role> is N cards away from closest Evil" ---
    if role_lower == 'scout':
        m = re.search(r'(\w[\w\s]*?)\s+is\s+(\d+)\s+card', clue, re.IGNORECASE)
        if m:
            evil_role = m.group(1).strip()
            distance = int(m.group(2))
            return card_scout(pos, evil_role, distance)

    # --- Medium: "#N is a real <Role>" ---
    if role_lower == 'medium' and targets:
        m = re.search(r'is\s+a\s+real\s+(\w[\w\s]*)', clue, re.IGNORECASE)
        if m:
            good_role = m.group(1).strip()
            return card_medium(pos, targets[0], good_role)

    # --- Poet: copies a random villager ability. Try to detect which one. ---
    if role_lower == 'poet' and clue:
        cl = clue.lower()
        # Bard pattern
        if 'corrupted' in cl and ('card' in cl or 'no corrupted' in cl):
            if 'no corrupted' in cl or 'are not corrupted' in cl:
                return CardInfo(pos, "Poet", info_parsed={"corruption_distance": -1, "copied_role": "Bard"})
            m = re.search(r'(\d+)\s+card', clue, re.IGNORECASE)
            if m:
                return CardInfo(pos, "Poet", info_parsed={"corruption_distance": int(m.group(1)), "copied_role": "Bard"})
        # Knitter pattern
        if 'pair' in cl or 'not adjacent' in cl:
            if 'not adjacent' in cl:
                return CardInfo(pos, "Poet", info_parsed={"evil_pairs": 0, "copied_role": "Knitter"})
            m = re.search(r'(\d+)\s+(?:evil\s+)?pair', clue, re.IGNORECASE)
            if m:
                return CardInfo(pos, "Poet", info_parsed={"evil_pairs": int(m.group(1)), "copied_role": "Knitter"})
        # Lover pattern
        if 'neighbor' in cl:
            m = re.search(r'(\d+)', clue)
            if m:
                return CardInfo(pos, "Poet", info_parsed={"evil_adjacent": int(m.group(1)), "copied_role": "Lover"})
            return CardInfo(pos, "Poet", info_parsed={"evil_adjacent": 0, "copied_role": "Lover"})
        # Scout pattern: "<EvilRole> is N cards away from closest Evil"
        # Must come before Hunter ("I am N cards away..."): both contain "closest evil".
        m_scout = re.search(r'^\s*([A-Z][\w\s]*?)\s+is\s+(\d+)\s+card', clue, re.IGNORECASE)
        if m_scout and 'away' in cl and ('nearest evil' in cl or 'closest evil' in cl):
            candidate = m_scout.group(1).strip()
            if candidate.lower() not in ('i', 'i am'):
                return CardInfo(pos, "Poet", info_parsed={
                    "evil_role": candidate,
                    "distance": int(m_scout.group(2)),
                    "copied_role": "Scout",
                })
        # Hunter pattern
        if ('nearest evil' in cl or 'closest evil' in cl) and 'away' in cl:
            m = re.search(r'(\d+)\s+card', clue, re.IGNORECASE)
            if m:
                return CardInfo(pos, "Poet", info_parsed={"distance": int(m.group(1)), "copied_role": "Hunter"})
        # Enlightened pattern
        if 'clockwise' in cl or 'equidistant' in cl:
            if 'counter' in cl:
                return CardInfo(pos, "Poet", info_parsed={"direction": "CCW", "copied_role": "Enlightened"})
            elif 'equidistant' in cl:
                return CardInfo(pos, "Poet", info_parsed={"direction": "Equidistant", "copied_role": "Enlightened"})
            else:
                return CardInfo(pos, "Poet", info_parsed={"direction": "CW", "copied_role": "Enlightened"})
        # Architect pattern
        if cl.strip().startswith('left') or cl.strip().startswith('right') or cl.strip().startswith('equal'):
            if 'left' in cl:
                return CardInfo(pos, "Poet", info_parsed={"side": "Left", "copied_role": "Architect"})
            elif 'right' in cl:
                return CardInfo(pos, "Poet", info_parsed={"side": "Right", "copied_role": "Architect"})
            else:
                return CardInfo(pos, "Poet", info_parsed={"side": "Equal", "copied_role": "Architect"})
        # Confessor pattern
        if 'dizzy' in cl or 'feeling good' in cl:
            dizzy = 'dizzy' in cl
            return CardInfo(pos, "Poet", info_parsed={"dizzy": dizzy, "copied_role": "Confessor"})
        # Medium pattern: "#N is a real <Role>"
        m = re.search(r'is\s+a\s+real\s+(\w[\w\s]*)', clue, re.IGNORECASE)
        if m and targets:
            good_role = m.group(1).strip()
            return CardInfo(pos, "Poet", info_parsed={"good_position": targets[0], "good_role": good_role, "copied_role": "Medium"})
        # Baker pattern
        m = re.search(r'I was (?:a |an )?(.+)', clue, re.IGNORECASE)
        if m:
            claimed = m.group(1).strip()
            # "I was a Baker" = converted from Baker (keep as 'Baker', not 'original')
            return CardInfo(pos, "Poet", info_parsed={"original_role": claimed, "copied_role": "Baker"})

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
    pos = int(args[1])

    if role == "enlightened":
        return card_enlightened(pos, args[2])  # CW/CCW/Equidistant
    elif role == "knitter":
        return card_knitter(pos, int(args[2]))
    elif role == "confessor":
        dizzy = args[2].lower() in ("dizzy", "dirty", "true", "1", "yes")
        return card_confessor(pos, dizzy)
    elif role == "gemcrafter":
        return card_gemcrafter(pos, int(args[2]))
    elif role == "lover":
        return card_lover(pos, int(args[2]))
    elif role == "scout":
        return card_scout(pos, args[2], int(args[3]))
    elif role == "bard":
        return card_bard(pos, int(args[2]))
    elif role in ("fortune_teller", "ft"):
        targets = [int(x) for x in args[2].split(",")]
        has_evil = args[3].lower() in ("yes", "true", "1")
        return card_fortune_teller(pos, targets, has_evil)
    elif role == "oracle":
        targets = [int(x) for x in args[2].split(",")]
        return card_oracle(pos, targets, args[3])
    elif role == "medium":
        target_pos = int(args[2])
        claimed_role = args[3]
        # "real" means target IS their displayed role — resolve to actual role name
        if claimed_role.lower() == "real":
            if session is None:
                print(f"  ERROR: 'real' keyword requires session context — enter role name instead")
                claimed_role = args[3]
            else:
                target_card = next((c for c in session.cards if c.position == target_pos), None)
                if target_card:
                    claimed_role = target_card.apparent_role
                    print(f"  [medium] Resolved 'real' -> '{claimed_role}' (target #{target_pos} apparent role)")
                else:
                    print(f"  WARNING: 'real' used but no card entry for #{target_pos} — enter role name instead")
        return card_medium(pos, target_pos, claimed_role)
    elif role == "hunter":
        return card_hunter(pos, int(args[2]))
    elif role == "architect":
        return card_architect(pos, args[2])  # Left/Right/Equal
    elif role == "empress":
        targets = [int(x) for x in args[2].split(",")]
        return card_empress(pos, targets)
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
        if session is not None and target > session.n_cards:
            raise ValueError(
                f"Rambler shut-up target #{target} is outside "
                f"1..{session.n_cards}"
            )
        return card_shut_up(pos, args[2], target)
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
        targets = [int(x) for x in args[2].split(",")]
        found = args[3] if len(args) > 3 and args[3].lower() != "none" else None
        return card_druid(pos, targets, found)
    elif role == "bishop":
        targets = [int(x) for x in args[2].split(",")]
        types = None
        if len(args) > 3:
            types = [t.strip().capitalize() for t in args[3].split(",")]
        return card_bishop(pos, targets, types)
    elif role == "baker":
        if len(args) > 2:
            claim = args[2]
            # "original" or "Baker" (same role) means the original Baker
            if claim.lower() in ("original", "baker"):
                claim = "original"
            return card_baker(pos, claim)
        else:
            return card_baker(pos, "original")  # no arg = original Baker
    elif role == "poet":
        if len(args) > 2:
            # Poet clue variant: poet <pos> <clue_type> <args...>
            return card_poet_with_info(pos, args[2], args[3:])
        else:
            return card_no_info(pos, "Poet")  # No info identified
    elif role in ("bounty_hunter", "bountyhunter"):
        return card_bounty_hunter(pos, int(args[2]))
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


def _validate_true_evils_against_session(true_evils: dict, session) -> tuple:
    """Validate that the evils-dict passed to game_over is consistent with session state.

    Rules:
      1. Every position must be dead (in executed OR night_kills).
      2. If a position is night-killed, the claimed role must be one of the deck's
         evil roles (minions + demons). Lilis CAN kill evils (asc33_v5.json proves
         this), but a night-killed GOOD card (e.g. a lilis-killed Bard) should never
         appear in the evils dict — that's the asc54_v4 bug.

    Returns (cleaned_dict, errors). If errors is non-empty, caller must refuse save.
    """
    errors = []
    dead = set(session.executed) | set(session.night_kills)
    nk_set = set(session.night_kills)

    def _normalize(r: str) -> str:
        return r.lower().replace("_", " ").replace("-", " ").strip()

    evil_roles_normalized = {
        _normalize(r) for r in (list(session.minions) + list(session.demons))
    }

    for pos, role in true_evils.items():
        if pos not in dead:
            errors.append(
                f"ERROR: #{pos} is not dead (not in executed or night_kills) — "
                f"true_evil_positions must only contain dead positions"
            )
            continue
        if pos in nk_set:
            if _normalize(role) not in evil_roles_normalized:
                errors.append(
                    f"ERROR: #{pos} ({role}) is night-killed but '{role}' is not in this "
                    f"deck's evil list {sorted(evil_roles_normalized)} — either the card "
                    f"was good (omit from evils) or the role name is wrong"
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
        print("  execute <pos> <RoleName>              Shorthand: mark as evil with role")
        print("  execute <pos> good blocked            Knight immunity (no HP loss, confirmed good)")
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
        print("                                           good/evil comes from the visible HP result")
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
        print("  card confessor 1 clean")
        print("  card knitter 2 2")
        print("  card fortune_teller 4 1,3 yes")
        print("  card oracle 5 2,6 Shaman")
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
        print("  card druid 5 1,2,3 none      (Druid checked 1,2,3: no outcasts)")
        print("  card druid 5 1,2,3 Bombardier (Druid found Bombardier among 1,2,3)")
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

            parsed = _parse_clue_from_memory(mc, n_cards=session.n_cards)
            rambler_capture_error = None
            if parsed is None:
                _, rambler_capture_error = _card_from_rambler_surface(
                    mc,
                    n_cards=session.n_cards,
                )
            if parsed:
                existing = entered.get(pos)
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
                        (mc.get('uses', 0) > 0 or mc.get('ability_used', False))
                        and same_role
                        and changed
                        and _has_active_clue_result(parsed)
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
                    if not active_update and not shut_up_update and not quote_update:
                        continue
                session.add_card(parsed)
                DecisionLog.log_card(parsed)
                if (mc.get('uses', 0) > 0 or mc.get('ability_used', False)) and _has_active_clue_result(parsed):
                    session.mark_ability_used(parsed.position)
                verb = "updated" if pos in entered else "entered"
                print(f"  [auto] {verb} #{parsed.position} {parsed.apparent_role}: {parsed.info_parsed}")
                entered[pos] = parsed
                auto_count += 1
            else:
                if rambler_capture_error is not None:
                    role = mc.get('disguise') or mc.get('true_role', '?')
                    manual_needed.append(
                        f"  #{pos} {role}: [RECOVERY] {rambler_capture_error}"
                    )
                    continue
                if pos in entered:
                    continue
                clue = mc.get('clue_text', '')
                role = mc.get('disguise') or mc.get('true_role', '?')
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
        session.save()
        DecisionLog.log_card(card)
        print(f"Added #{card.position} {card.apparent_role}: {card.info_parsed}")
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
            if w in ("evil", "true", "1", "yes"):
                was_evil = True
                if len(args) > 2:
                    evil_role = _normalize_role_name(args[2])
            elif w in ("good", "false", "0", "no"):
                was_evil = False
                knight_blocked = False
                outcome_args = args[2:]
                for raw in outcome_args:
                    c = raw.lower()
                    if c in ("blocked", "immune", "knight_block") or (
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
                observed_true_role = observed_target.get('true_role')
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

                if _observed_knight_immunity(observed_target, apparent_role):
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
            # Knight immunity: card survives, confirmed good, no HP loss
            session.record_execution_blocked(pos)
            print(f"Executed #{pos} -> BLOCKED (Knight immunity)")
            print(f"  #{pos} confirmed GOOD. No HP loss. HP: {session.hp}/10")
        else:
            session.mark_executed(
                pos,
                was_evil,
                evil_role,
                was_corrupted,
                observed_true_role if was_evil is False else None,
            )
            session.save()
            DecisionLog.log_execution(pos, was_evil, evil_role)
            tag = f" (evil: {evil_role})" if evil_role else (f" (was_evil={was_evil})" if was_evil is not None else "")
            corr_tag = ""
            if was_corrupted is True:
                corr_tag = (
                    " <ACTIVE Corrupted; observed clean>"
                    if _execution_role_key(observed_true_role) == "drunk"
                    else " <Corrupted>"
                )
            elif was_corrupted is False and was_evil is False:
                corr_tag = " (clean)"
            print(f"Executed #{pos}{tag}{corr_tag}")
            if was_evil:
                print(f"  HP: {session.hp}/10 (correct execution, no HP loss)")
            elif was_evil is False:
                if observed_true_role is None:
                    print("  WARNING: Wrong execution recorded, but exact HP damage cannot "
                          "be inferred without the revealed true role.")
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

            # Any Baa death refreshes its previously obscured deck-view entry.
            if evil_role and evil_role.lower().replace(' ', '_') == "baa":
                _baa_post_death_deck_refresh(session)
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
        session.save()
        print(f"#{pos} blocked (Witch)")
        return None

    if cmd == "block_force":
        pos = int(args[0])
        if pos not in session.blocked_positions:
            session.blocked_positions.append(pos)
        if pos in session.reveal_order:
            session.reveal_order.remove(pos)
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
        if killed and revealed_role and was_evil is None:
            from knowledge_base import Alignment, get_card
            revealed_def = get_card(revealed_role)
            if (revealed_def is not None
                    and revealed_def.alignment == Alignment.GOOD):
                print(
                    "  ERROR: A Good-class revealed role does not expose its "
                    "preserved runtime alignment."
                )
                print(
                    "  Use the public HP result: pass 'good' if base damage "
                    "occurred, or 'evil' if HP did not change."
                )
                print("  No Slayer state was recorded.")
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
        if session.hp != old_hp:
            print(f"  Wrong Slayer kill: HP {old_hp} -> {session.hp}")
        if recorded_role == "Baa":
            _baa_post_death_deck_refresh(session)
        return None

    if cmd == "night_kill":
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
                import subprocess as _sp
                mr = _sp.run(["python", "memory_reader.py"],
                             capture_output=True, text=True, timeout=10)
                if mr.returncode == 0:
                    # Parse memory_reader output for evil cards
                    auto_evils = {}
                    in_evil_section = False
                    for line in mr.stdout.split("\n"):
                        line = line.strip()
                        import re
                        if line.startswith("Evil cards"):
                            in_evil_section = True
                            continue
                        if in_evil_section:
                            if not line:
                                continue
                            if line.startswith("Clue data"):
                                in_evil_section = False
                                continue
                            m = re.match(r"#\s*(\d+)\s+(.+?)(?:\s+\(|$)", line)
                            if m:
                                auto_evils[int(m.group(1))] = m.group(2).strip().replace(" ", "_")
                                continue
                        # Older format: "#N: RoleName (Evil) ..."
                        m = re.match(r"#(\d+):\s+(.+?)\s+\(Evil\)", line)
                        if m:
                            auto_evils[int(m.group(1))] = m.group(2).strip().replace(" ", "_")
                            continue
                        # Current table format: "# 7* Pooka ... Evil ..."
                        m = re.match(r"#\s*(\d+)\*?\s+(.+?)\s{2,}.+\s{2,}Evil\b", line)
                        if m:
                            auto_evils[int(m.group(1))] = m.group(2).strip().replace(" ", "_")
                    if auto_evils:
                        # Validate auto-detected evils against session state before accepting
                        _auto_cleaned, _auto_errors = _validate_true_evils_against_session(
                            auto_evils, session
                        )
                        if _auto_errors:
                            print("[game_over] Auto-detected evils failed validation:")
                            for err in _auto_errors:
                                print(f"  {err}")
                            print("[game_over] Falling back to manual evils entry.")
                        else:
                            true_evils_str = ",".join(f"{p}={r}" for p, r in sorted(auto_evils.items()))
                            print(f"[game_over] Auto-detected true evils from memory: {true_evils_str}")
                    else:
                        print("[game_over] Could not auto-detect evils from memory reader")
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
