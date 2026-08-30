"""Game loop adapter: bridges Claude's vision reads to the constraint solver.

Card builder functions, session tracking, CLI interface.
"""

from __future__ import annotations
import atexit
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

def card_jester_silenced(pos: int, targets: list[int], shut_up_target: Optional[int] = None) -> CardInfo:
    """Jester whose ability fired but was silenced (e.g. by Rambler).

    Records the targets (for UI / audit) plus silenced:True so the Rust
    validator can treat the clue as vacuous (no constraint) instead of
    accidentally returning true in the targets/evil_count lookups (asc78_v6).
    """
    info = {"targets": list(targets), "silenced": True}
    if shut_up_target is not None:
        info["shut_up_target"] = shut_up_target
    return CardInfo(pos, "Jester", info_parsed=info)

def card_rambler(pos: int, silenced: bool, silenced_by: Optional[int] = None) -> CardInfo:
    info = {"silenced": silenced}
    if silenced_by is not None:
        info["silenced_by"] = silenced_by
    return CardInfo(pos, "Rambler", info_parsed=info)

def card_shut_up(pos: int, role: str, target: int) -> CardInfo:
    """A Rambler-redesign clue: this card said "#target shut up!"."""
    return CardInfo(pos, _normalize_role_name(role), info_parsed={"shut_up_target": target})

def card_dreamer(pos: int, target: int, evil_role: str) -> CardInfo:
    return CardInfo(pos, "Dreamer", info_parsed={"target": target, "evil_role": evil_role})


def card_dreamer_ambiguous(pos: int, targets: list[int], evil_role_options: list[str]) -> CardInfo:
    """Dreamer2 post-patch output: "Among #X, #Y there is: R1 or R2".

    The Rust solver handles this shape in validators/mod.rs (Shape 2 ambiguous
    Dreamer): {targets, evil_role_options}. One of the named roles is at one
    of the listed positions, but the mapping is unknown.

    Also fires for corrupted Dreamer1 (Drunk-as-Dreamer in asc74_v7) which
    the game renders in the same ambiguous form.
    """
    return CardInfo(pos, "Dreamer", info_parsed={
        "targets": list(targets),
        "evil_role_options": list(evil_role_options),
    })


def _has_active_clue_result(card: CardInfo) -> bool:
    """True when an active ability entry contains a real clue result."""
    role = card.apparent_role.lower().replace(" ", "_")
    info = card.info_parsed or {}
    if role == "dreamer":
        return bool(info.get("target") or info.get("targets"))
    if role in {"fortune_teller", "jester", "druid", "judge"}:
        return bool(info)
    return False


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
    m = re.search(
        r'Among\s+((?:#\d+(?:\s*,\s*)?)+)\s+there\s+is\s*:?\s*([\w\s]+?)\s+or\s+([\w\s]+?)\s*\.?\s*$',
        clue, re.IGNORECASE | re.DOTALL
    )
    if not m:
        return None
    targets = [int(x) for x in re.findall(r'#(\d+)', m.group(1))]
    options = [m.group(2).strip(), m.group(3).strip()]
    return (targets, options)

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
        self.reveal_order: list[int] = []  # Order positions were flipped (for Baker)
        self.lilis_batch_index: int = 0  # Explicit Lilis batch counter (don't derive from reveal_order)

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
        self.reveal_order.clear()
        self.lilis_batch_index = 0
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

    def is_lilis_alive(self) -> bool:
        """Check if Lilis is in the deck and has not been executed."""
        if not self.has_role_in_deck("Lilis"):
            return False
        return not any(
            _normalize_role_name(r) == "Lilis"
            for r in self.executed_evil_roles.values()
        )

    def set_deck(self, villagers: list[str], outcasts: list[str],
                 minions: list[str], demons: list[str]):
        self.villagers = villagers
        self.outcasts = outcasts
        self.minions = minions
        self.demons = demons

    # -- Cards --

    def add_card(self, card: CardInfo):
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
        }
        role_key = card.apparent_role.lower().replace(" ", "_")
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

    def set_pd_target(self, pos: int):
        self.pd_corruption_target = pos

    def add_pd_ability_result(self, pd_pos: int, target: int, is_corrupted: bool,
                              evil_revealed: Optional[int] = None):
        self.pd_ability_results.append({
            "pd_pos": pd_pos,
            "target": target,
            "is_corrupted": is_corrupted,
            "evil_revealed": evil_revealed,
        })
        self.mark_ability_used(pd_pos)

    def mark_ability_used(self, pos: int):
        if pos not in self.used_abilities:
            self.used_abilities.append(pos)

    def add_slayer_result(self, slayer_pos: int, target_pos: int, killed: bool,
                          revealed_role: Optional[str] = None,
                          was_corrupted: Optional[bool] = None):
        """Record the public result of Slayer's native kill-and-reveal path.

        Slayer tests registered alignment, so a real Wretch registers Evil and
        dies even though its revealed alignment is Good.  Classify the target
        from the revealed role, never from the fact that Slayer killed it.
        """
        from knowledge_base import Alignment, execution_cost_for, get_card

        if any(sr.get("slayer_pos") == slayer_pos for sr in self.slayer_results):
            raise ValueError(f"Slayer #{slayer_pos} already has a recorded result")

        canonical_role = None
        role_def = None
        if killed:
            if not revealed_role:
                raise ValueError("Slayer kill requires the revealed role")
            role_def = get_card(revealed_role)
            if role_def is None:
                raise ValueError(f"Unknown Slayer revealed role: {revealed_role}")
            canonical_role = role_def.name.replace(" ", "_")
            if role_def.alignment == Alignment.GOOD and role_def.name != "Wretch":
                raise ValueError(
                    "Native Slayer can only kill an Evil character or a Good Wretch"
                )
        elif revealed_role:
            raise ValueError("A failed Slayer attempt does not reveal a role")
        elif was_corrupted is not None:
            raise ValueError("A failed Slayer attempt does not reveal target status")

        if (role_def is not None and role_def.alignment == Alignment.EVIL
                and was_corrupted is not None):
            raise ValueError("Corruption evidence is only recorded for a killed Good Wretch")

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
            if role_def.alignment == Alignment.EVIL:
                self.mark_executed(
                    target_pos,
                    was_evil=True,
                    evil_role=canonical_role,
                )
            else:
                self.mark_executed(
                    target_pos,
                    was_evil=False,
                    was_corrupted=was_corrupted,
                    true_role=canonical_role,
                )
                damage = execution_cost_for(
                    canonical_role,
                    apparent_role=canonical_role,
                    was_killable=True,
                    default=self.wrong_exec_cost,
                )
                self.hp = _clamped_post_damage_hp(self.hp, damage)

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
            reveal_order=list(self.reveal_order),
            executed_good_corrupted=dict(self.executed_good_corrupted),
            executed_good_roles=dict(self.executed_good_roles),
        )

    @classmethod
    def from_game_state(cls, state: GameState,
                        used_abilities: Optional[list[int]] = None,
                        lilis_batch_index: int = 0) -> "GameSession":
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
        session.reveal_order = list(state.reveal_order)
        session.executed_good_corrupted = dict(getattr(state, 'executed_good_corrupted', {}))
        session.executed_good_roles = dict(getattr(state, 'executed_good_roles', {}))
        session.used_abilities = list(used_abilities or [])
        session.lilis_batch_index = lilis_batch_index
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
        Slayer and Plague_Doctor use dedicated commands (slayer_result, pd_check)
        and are explicitly rejected here.

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

        # Slayer and Plague Doctor use dedicated result-entry commands.
        if ability_name in ("slayer", "plague_doctor"):
            return {"success": False, "info_parsed": None,
                    "error": f"{action.ability_name} requires manual handling (use slayer_result/pd_check)"}

        if ability_name == "dreamer" and len(targets) != 2:
            return {"success": False, "info_parsed": None,
                    "error": f"Dreamer2 requires exactly 2 targets, got {targets}"}

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
            if t in self.used_abilities:
                continue
            target_card_entry = next((c for c in self.cards if c.position == t), None)
            if not target_card_entry:
                continue
            kb_card = get_card(target_card_entry.apparent_role)
            if kb_card and kb_card.activated_ability:
                return {"success": False, "info_parsed": None,
                        "error": f"#{t} ({target_card_entry.apparent_role}) has unused active ability; clicking it would activate the card instead of selecting it. Use ability_used {t} first or handle this ability manually."}

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

        # Step 3: Wait for ability result in memory. Dreamer2 can set the act
        # flag and clue text while leaving uses at 0.
        print(f"  [auto_ability] Waiting for ability result...")
        target_card_data = None

        def _ability_resolved(board):
            if not board:
                return False
            card = next((c for c in board if c['position'] == pos), None)
            if not card:
                return False
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
        has_recorded_result = (
            target_card_data.get('uses', 0) > 0
            or bool(target_card_data.get('acted_infos'))
            or bool(target_card_data.get('ability_used') and target_card_data.get('clue_text'))
        )
        if not has_recorded_result:
            return {"success": False, "info_parsed": None,
                    "error": f"Ability result not detected (uses=0, acted_infos empty) — click may have missed"}

        # Step 4: Parse the result via the existing auto_card pipeline
        parsed = _parse_clue_from_memory(target_card_data)
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

        # Route USE_ABILITY to auto_use_ability. Slayer and Plague Doctor still
        # use dedicated result-entry commands.
        if action.action_type == "use_ability":
            ability_name_lower = (action.ability_name or "").lower().replace(" ", "_")
            if ability_name_lower in ("slayer", "plague_doctor"):
                print(f"\n  [auto_next] {action.ability_name} requires manual handling — use ability_used to skip, or fire the ability in-game and record with slayer_result/pd_check.")
                return action, result, None
            print(f"\n  === AUTO-ABILITY #{action.position} ({action.ability_name}) -> targets {action.targets} ===")
            exec_result = self.auto_use_ability(action)
            if exec_result["success"]:
                print(f"  AUTO-ABILITY SUCCESS: {action.ability_name} #{action.position} result recorded")
            else:
                print(f"  AUTO-ABILITY FAILED: {exec_result['error']}")
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
            print(f"  PD corruption target: #{self.pd_corruption_target}")
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
            "blocked": [positions likely Witch-blocked (last card, Witch in deck)],
            "failed": [positions that failed to flip (click didn't register)],
            "success": bool (True if no failures),
        }
    """
    import re
    still_hidden = []

    if isinstance(cards_or_output, list):
        # New path: card dicts from read_board()
        for card in cards_or_output:
            pos = card.get('position')
            if pos in expected_positions and card.get('state') == 'Hidden' and not card.get('killed_hidden'):
                still_hidden.append(pos)
    else:
        # Legacy path: parse stdout text
        for line in cards_or_output.splitlines():
            m = re.match(r'^\s*#\s*(\d+)', line)
            if not m:
                continue
            pos = int(m.group(1))
            if pos in expected_positions and 'Hidden' in line and 'Dead' not in line:
                still_hidden.append(pos)

    flipped = [p for p in expected_positions if p not in still_hidden]
    blocked = []
    failed = []

    if still_hidden:
        has_witch = session.has_role_in_deck("Witch")
        # If Witch in deck and only the last expected position is hidden, likely Witch block
        if has_witch and len(still_hidden) == 1 and still_hidden[0] == max(expected_positions):
            blocked = still_hidden
        else:
            failed = still_hidden

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
                print(f"  Witch IS in deck -- #{blocked[0]} is likely Witch-blocked (last card).")
            else:
                print("  Witch IS in deck but multiple cards hidden. Likely click failures.")
                print("  Re-run: python game_loop.py flip")
        print("!" * 60)

    return {
        "flipped": flipped,
        "blocked": blocked,
        "failed": failed,
        "success": len(failed) == 0 and len(blocked) == 0,
    }


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


def _parse_clue_from_memory(card: dict) -> Optional[CardInfo]:
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
    targets = infos[0]['targets'] if infos else []
    role_lower = role.lower().replace(' ', '_')
    ability_used = card.get('ability_used', False)

    # --- Guard: active-ability-only roles with unused abilities ---
    # These roles have NO passive speech bubble. If ability hasn't been used,
    # any clue_text/acted_infos is stale from a previous village — ignore it.
    # Prefer `uses`, but Dreamer2 currently sets ability_used=True while leaving
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

    shut_up_pat = re.search(r'#\s*(\d+)\s*shut\s*up', clue, re.IGNORECASE)
    if shut_up_pat and role_lower != 'jester':
        # New Rambler behavior replaces an adjacent truthful character's
        # normal clue with "#X shut up!". Preserve the target as a global
        # Rambler constraint instead of parsing the number as role info.
        return card_shut_up(pos, role, int(shut_up_pat.group(1)))

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

    # --- Rambler: silenced <=> no quote text ---
    if role_lower == 'rambler':
        silenced = not clue.strip()
        return card_rambler(pos, silenced)

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

    # --- Witness: single target ---
    if role_lower == 'witness' and targets:
        return card_witness(pos, targets[0])

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
        # Silenced Jester (e.g. by Rambler): clue is flavor like "#X shut up!"
        # or empty. Targets are preserved, but no evil_count is recoverable.
        # Without this branch the Rust validator's targets/evil_count lookups
        # returned true unconditionally, masking the constraint (asc78_v6).
        silenced_pat = re.search(r'#\s*(\d+)\s*shut\s*up', clue, re.IGNORECASE)
        if silenced_pat or (not clue.strip()):
            # Guard: only silenced if no numeric evil count is extractable.
            if not re.search(r'\d+\s+(?:of them |are |is )?\s*evil', clue, re.IGNORECASE):
                shut_target = int(silenced_pat.group(1)) if silenced_pat else None
                return card_jester_silenced(pos, targets, shut_target)
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

    # --- Judge: target + lying from clue ---
    if role_lower == 'judge' and targets:
        is_lying = 'lying' in clue.lower() or 'liar' in clue.lower()
        return card_judge(pos, targets[0], is_lying)

    # --- Dreamer: target + evil role from clue ---
    if role_lower == 'dreamer':
        # Dreamer2 post-patch: "Among #X, #Y there is: R1 or R2" — try ambiguous form first.
        ambiguous = _parse_ambiguous_among(clue)
        if ambiguous:
            amb_targets, options = ambiguous
            return card_dreamer_ambiguous(pos, amb_targets or targets, options)
        # Old Dreamer1 form: "#N could be: <Role>" or "#N is <Role>"
        if targets:
            m = re.search(r'(?:could be|is)\s*:?\s*(\w[\w\s]*)', clue, re.IGNORECASE)
            if m:
                evil_role = m.group(1).strip()
                return card_dreamer(pos, targets[0], evil_role)

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
    Role aliases: fortune_teller, plague_doctor, no_info
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
        # Accepted forms:
        #   card rambler 2 silenced            -> silenced, picker unknown
        #   card rambler 2 silenced 6          -> silenced, picker was #6
        #   card rambler 2 talking             -> quote shown
        token = args[2].lower() if len(args) > 2 else ""
        silenced = token in ("silenced", "quiet", "silent", "true", "yes", "1")
        silenced_by = int(args[3]) if len(args) > 3 and args[3].isdigit() else None
        return card_rambler(pos, silenced, silenced_by)
    elif role in ("shut_up", "shutup"):
        # card shut_up <pos> <apparent_role> <target>
        return card_shut_up(pos, args[2], int(args[3]))
    elif role == "dreamer":
        return card_dreamer(pos, int(args[2]), args[3])
    elif role in ("dreamer2", "dreamer_ambiguous"):
        targets = [int(x) for x in args[2].split(",")]
        roles = [x.strip().replace("_", " ") for x in args[3].split(",") if x.strip()]
        return card_dreamer_ambiguous(pos, targets, roles)
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


def _baa_hides_outcast(only_cv: set, only_mr: set, mr_set: set, cv_unclassified: int) -> bool:
    """Baa in the deck renders one outcast as a face-down eye-symbol in the pool view.

    That produces: exactly one outcast role in only_mr, zero only_cv, and at
    least one unclassified CV box. Confirmed asc78 (Doppelganger hidden).
    """
    if "baa" not in mr_set:
        return False
    if only_cv or len(only_mr) != 1 or cv_unclassified < 1:
        return False
    return next(iter(only_mr)) in _DECK_OUTCAST_ROLES


def _baa_post_execute_reveal(session) -> None:
    """When Baa dies, the game auto-reveals the Outcast it was hiding on the board.
    Detect that position via memory reader and add it to reveal_order so the
    user can immediately enter its info via auto_card / card.
    """
    from memory_reader import MemoryReader, get_monitor
    already = set(session.reveal_order) | set(session.executed) | set(session.night_kills)
    blocked = set(session.blocked_positions)
    cards = None
    try:
        mon = get_monitor()
        if mon and mon.is_healthy():
            def _revealed(board):
                if not board:
                    return False
                for c in board:
                    p = c.get('position')
                    if p in already or p in blocked:
                        continue
                    if c.get('state') in ('Alive', 'Revealed'):
                        return True
                return False
            time.sleep(0.5)
            mon.wait_for(_revealed, timeout=4, min_delay=0.5)
            cards = mon.get_board()
    except Exception:
        cards = None
    if cards is None:
        time.sleep(1.5)
        reader = MemoryReader()
        if reader.open():
            try:
                cards = reader.read_board()
            finally:
                reader.close()
    if not cards:
        print("  [Baa] Could not read memory — flip the newly-revealed card manually.")
        return
    newly = []
    for c in cards:
        p = c.get('position')
        if p in already or p in blocked:
            continue
        if c.get('state') in ('Alive', 'Revealed'):
            newly.append(c)
    if not newly:
        print("  [Baa] No newly-revealed position detected (Witch may have blocked it).")
        return
    for c in newly:
        p = c['position']
        session.reveal_order.append(p)
        role = c.get('disguise') or c.get('true_role') or '?'
        print(f"  [Baa] Revealed #{p} -> {role}. Run: auto_card  (or: card {role.lower()} {p} ...)")
    session.save()


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
    cv_set = set(r.lower().replace(" ", "_") for r in cv_roles)
    mr_set = set(mr_roles)

    if cv_set and mr_set:
        if cv_set == mr_set:
            print(f"\n  MATCH: Both pipelines agree ({len(cv_set)} roles)")
        else:
            only_cv = cv_set - mr_set
            only_mr = mr_set - cv_set
            if _baa_hides_outcast(only_cv, only_mr, mr_set, cv_unclassified):
                role = next(iter(only_mr))
                print(f"\n  MATCH (Baa hides outcast): CV={len(cv_set)} classified"
                      f" + '{role}' face-down in deck view (Baa effect)")
            else:
                print(f"\n  MISMATCH!")
                if only_cv:
                    print(f"    Only in card_vision: {only_cv}")
                if only_mr:
                    print(f"    Only in memory_reader: {only_mr}")
                print(f"    STOP AND FIX before proceeding!")
    elif not cv_set and not mr_set:
        print("\n  WARNING: Both pipelines returned empty results")
    else:
        print(f"\n  WARNING: Only one pipeline returned results (cv={len(cv_set)}, mr={len(mr_set)})")


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
        print("  flip --lilis                          Flip in batches of 4 (Lilis games)")
        print("  card <role> <pos> [args...]           Add a revealed card")
        print("  auto_card                             Auto-enter cards from memory reader")
        print("  execute <pos> [evil|good] [role]      Mark position executed (with evil role name)")
        print("  execute <pos> <RoleName>              Shorthand: mark as evil with role")
        print("  execute <pos> good blocked            Knight immunity (no HP loss, confirmed good)")
        print("  execute <pos> good <clean|corrupted> [revealed_role]")
        print("                                           Wrong exec with optional UI-observed role")
        print("  pd_target <pos>                       Set Plague Doctor corruption target")
        print("  pd_check <pd_pos> <target> corrupted <evil_pos>  PD found corruption + evil")
        print("  pd_check <pd_pos> <target> clean                 PD found no corruption")
        print("  set_hp <hp> [wrong_exec_cost]         Update HP and wrong execution cost")
        print("  solve                                 Run solver")
        print("  status                                Print session state")
        print("  confirm_evil <pos>                    Mark position as confirmed evil")
        print("  confirm_good <pos>                    Mark position as confirmed good")
        print("  next [--plan]                         Solve + auto-execute if safe (definite OR forced-safe). --plan for print-only.")
        print("  auto_next                             Alias for `next` (auto-execute path)")
        print("  ability_used <pos>                    Mark ability as activated")
        print("  slayer_result <pos> <target> kill <role> [clean|corrupted]  Slayer kill")
        print("  slayer_result <pos> <target> fail                           Slayer miss")
        print("  block <pos>                           Mark position as blocked (Witch)")
        print("  unblock <pos>                         Unblock position (after Witch dies)")
        print("  night_kill <pos1,pos2,...> <n_evil>    Lilis night kills (positions + evil count)")
        print("  night_no_kill                         Lilis night dealt 2HP but killed nobody (she's last unrevealed)")
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
        if any(d.lower() == "baa" for d in pool["demons"]):
            print("  WARNING: BAA in deck -- deck view shows +1 fake Outcast. "
                  "Subtract only if no= came from deck view; do not adjust HUD no=.")
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
        if any(d.lower() == "baa" for d in demons):
            print("  WARNING: BAA in deck -- deck view shows +1 fake Outcast. "
                  "Subtract only if no= came from deck view; do not adjust HUD no=.")
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

        from game_utils import all_game_card_coords
        import subprocess
        import template_match as _tm
        import mouse as _mouse
        coords = all_game_card_coords(session.n_cards)

        if single_pos:
            if single_pos not in coords:
                print(f"ERROR: Position {single_pos} not valid for {session.n_cards}-card game")
                return None
            x, y = coords[single_pos]
            print(f"Flipping #{single_pos} at ({x},{y})")
            _tm.safe_click_at(x, y, f"card{single_pos}")
            print(f"Flipped #{single_pos}")
            # Record reveal order
            if single_pos not in session.reveal_order:
                session.reveal_order.append(single_pos)
            # Remove from blocked if it was Witch-blocked
            if single_pos in session.blocked_positions:
                session.blocked_positions.remove(single_pos)
                print(f"  (unblocked #{single_pos})")
            # Lilis night check for single flips
            if session.is_lilis_alive():
                total_reveals = len(session.reveal_order)
                if total_reveals % 4 == 0:
                    print()
                    print("!" * 60)
                    print(f"  LILIS NIGHT PHASE TRIGGERED (reveal #{total_reveals})")
                    print(f"  Lilis deals 2 HP. HP: {session.hp} -> {session.hp - 2}")
                    print("!" * 60)
                    print(f"\n  --- Waiting for Lilis night animation ---")
                    try:
                        from memory_reader import get_monitor as _get_mon
                        _mon = _get_mon()
                        if _mon.is_healthy():
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
                    print(f"  Run: night_kill <pos> <n_evil>  OR  night_no_kill")
                    print(f"  (HP auto-deducted by night_kill/night_no_kill commands)")
            session.save()
            return None

        already_done = set(session.reveal_order) | set(session.night_kills) | set(session.executed)
        positions = [p for p in sorted(coords.keys()) if p not in already_done]
        if not positions:
            print("All cards already flipped/dead. Nothing to flip.")
            return None
        if lilis:
            batch_size = 4
            batch = positions[:batch_size]
            expected_positions = batch
            print(f"Flipping batch: {['#'+str(p) for p in batch]}")
            for idx, pos in enumerate(batch):
                _click_flip_card(pos, coords, f"card{pos}", verified=(idx == 0))
                time.sleep(0.2)
            print(f"Batch complete: {['#'+str(p) for p in batch]}")
            # Record reveal order for this batch
            for p in batch:
                if p not in session.reveal_order:
                    session.reveal_order.append(p)
            session.lilis_batch_index += 1
            remaining = positions[batch_size:]
            # Lilis night triggers every batch (batch_index tracks explicitly)
            if len(batch) == batch_size:
                print(f"\n  --- Lilis night phase (waiting for kill animation) ---")
                print(f"  Lilis deals 2 HP. HP: {session.hp} -> {session.hp - 2}")
                try:
                    from memory_reader import get_monitor as _get_mon
                    _mon = _get_mon()
                    if _mon.is_healthy():
                        _already = set(session.reveal_order) | set(session.night_kills) | set(session.executed)
                        def _night_kill_check(board):
                            if not board:
                                return False
                            return any(c.get('killed_hidden') for c in board
                                       if c['position'] not in _already)
                        _mon.wait_for(_night_kill_check, timeout=8, min_delay=2.0)
                    else:
                        time.sleep(5)
                except Exception:
                    time.sleep(5)
                print(f"  Night phase complete. Take screenshot to check for kills before continuing.")
                print(f"  Run: python screenshot.py night_check && python memory_reader.py")
                if remaining:
                    print(f"  Remaining to flip: {['#'+str(p) for p in remaining]}")
                else:
                    print(f"  No more cards to flip. Check for night kill/damage.")
                print(f"  Run: night_kill <pos> <n_evil>  OR  night_no_kill")
                print(f"  (HP auto-deducted by night_kill/night_no_kill commands)")
            elif remaining:
                print(f"  Remaining to flip: {['#'+str(p) for p in remaining]}")
        else:
            expected_positions = positions
            print(f"Flipping all {len(positions)} cards: #1 -> #{positions[-1]}")
            for idx, pos in enumerate(positions):
                _click_flip_card(pos, coords, f"card{pos}", verified=(idx == 0))
                time.sleep(0.2)
            print(f"All {len(positions)} cards flipped in order #1->#{positions[-1]}")
            # Record reveal order
            for p in positions:
                if p not in session.reveal_order:
                    session.reveal_order.append(p)

        session.save()
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
                    for p in positions:
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
            # Flake-failed clicks never actually revealed — strip them from
            # reveal_order so a subsequent `flip <pos>` retry lands them at
            # the true reveal index. Critical for Baker-chain validation:
            # wrong reveal_order corrupts the chain seed and can collapse
            # the scenario space to 0 after an unrelated wrong exec
            # (asc78_v6 halt, 2026-04-21).
            failed = verify.get("failed", [])
            if failed:
                removed = False
                for p in failed:
                    if p in session.reveal_order:
                        session.reveal_order.remove(p)
                        removed = True
                if removed:
                    print(f"  [reveal_order] Removed failed positions {failed} from reveal_order; retry via `flip {failed[0]}` will re-append at true index.")
                    session.save()
        else:
            print("  WARNING: memory_reader returned no cards")
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

            parsed = _parse_clue_from_memory(mc)
            if parsed:
                existing = entered.get(pos)
                if existing:
                    active_update = (
                        (mc.get('uses', 0) > 0 or mc.get('ability_used', False))
                        and existing.apparent_role == parsed.apparent_role
                        and existing.info_parsed != parsed.info_parsed
                        and _has_active_clue_result(parsed)
                    )
                    if not active_update:
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

            # Baa death reveals the previously-hidden Outcast on the board.
            if evil_role and evil_role.lower().replace(' ', '_') == "baa":
                _baa_post_execute_reveal(session)
        return None

    if cmd == "pd_target":
        pos = int(args[0])
        session.set_pd_target(pos)
        session.save()
        print(f"PD corruption target set to #{pos}")
        return None

    if cmd == "pd_check":
        pd_pos = int(args[0])
        target = int(args[1])
        status = args[2].lower()
        if status == "corrupted":
            evil_revealed = int(args[3])
            session.add_pd_ability_result(pd_pos, target, True, evil_revealed)
            session.save()
            print(f"PD #{pd_pos} checked #{target}: Corrupted, #{evil_revealed} is Evil")
        elif status == "clean":
            session.add_pd_ability_result(pd_pos, target, False)
            session.save()
            print(f"PD #{pd_pos} checked #{target}: Not Corrupted")
        else:
            print(f"Unknown PD check status: {status} (use 'corrupted' or 'clean')")
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
        slayer_pos = int(args[0])
        target_pos = int(args[1])
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
        if len(args) > 4:
            status = args[4].lower()
            if status in ("corrupted", "true", "1", "yes"):
                was_corrupted = True
            elif status in ("clean", "false", "0", "no"):
                was_corrupted = False
            else:
                print(f"  ERROR: Unknown Slayer target status: {args[4]}")
                print("  Use 'clean' or 'corrupted'.")
                return None
        if killed and not revealed_role:
            print("  ERROR: Slayer kill requires revealed_role! Game reveals the role on kill.")
            print(f"  Usage: slayer_result {slayer_pos} {target_pos} kill <revealed_role> [clean|corrupted]")
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
        if (recorded_role == "Wretch" and was_corrupted is None):
            print("  WARNING: Wretch corruption status was not recorded. If visible, use "
                  "'clean' or 'corrupted' in the Slayer command.")
        if recorded_role == "Baa":
            _baa_post_execute_reveal(session)
        return None

    if cmd == "night_kill":
        positions = [int(x) for x in args[0].split(",")]
        # Second arg = how many of the killed cards were evil (usually 0).
        # NOT the total evil count in the game! Lost asc68_v5 0-scenario bug from this confusion.
        n_evil_among_killed = int(args[1]) if len(args) > 1 else 0
        if n_evil_among_killed > len(positions):
            print(f"  ERROR: n_evil_among_killed ({n_evil_among_killed}) > killed positions ({len(positions)}).")
            print(f"  This arg is 'how many killed cards were evil', NOT total game evil count.")
            print(f"  Usually 0 (Lilis kills random Good). Use 0 or 1.")
            return
        # Night kills go in session.night_kills only; executed is for day executions.
        # Readers (lines 645, 1858, 1887, 1977, 2293) union the two sets.
        session.night_kills.extend(positions)
        session.night_kill_evil_count += n_evil_among_killed
        n_evil = n_evil_among_killed  # alias for existing code below
        for p in positions:
            # Issue #8: Remove killed positions from blocked_positions (Witch+Lilis interaction)
            if p in session.blocked_positions:
                session.blocked_positions.remove(p)
                print(f"  (removed #{p} from blocked_positions — killed by Lilis)")
        if n_evil == len(positions) and n_evil > 0:
            for p in positions:
                if p not in session.confirmed_evil:
                    session.confirmed_evil.append(p)
        # Issue #7: Auto-deduct 2 HP for Lilis night
        old_hp = session.hp
        session.hp -= 2
        session.save()
        confirmed_msg = ""
        if n_evil == len(positions) and n_evil > 0:
            confirmed_msg = f" (confirmed evil: {['#'+str(p) for p in positions]})"
        print(f"Night kills: {['#'+str(p) for p in positions]}, {n_evil} evil among them{confirmed_msg}")
        print(f"  Lilis night HP: {old_hp} -> {session.hp}")
        return None

    if cmd == "night_no_kill":
        revealed = {c.position for c in session.cards} | set(session.reveal_order)
        dead = set(session.executed) | set(session.night_kills)
        all_positions = set(range(1, session.n_cards + 1))
        unrevealed = all_positions - revealed - dead
        # Only auto-confirm Lilis when the lone unrevealed card is still
        # blocked (e.g. Witch-blocked). If the user just flipped it and
        # hasn't entered card data yet, the unrevealed=1 check is wrong.
        blocked = set(session.blocked_positions)
        # Issue #7: Auto-deduct 2 HP for Lilis night
        old_hp = session.hp
        session.hp -= 2
        if len(unrevealed) == 1 and unrevealed.issubset(blocked):
            lilis_pos = next(iter(unrevealed))
            if lilis_pos not in session.confirmed_evil:
                session.confirmed_evil.append(lilis_pos)
            session.save()
            print(f"Lilis night dealt 2HP but no kill — only unrevealed card is #{lilis_pos} (blocked)")
            print(f"  => #{lilis_pos} confirmed as Lilis (can't kill herself)")
            print(f"  HP: {old_hp} -> {session.hp}")
        elif len(unrevealed) == 0:
            session.save()
            print("Lilis night dealt 2HP but no kill; all positions are revealed/dead.")
            print("  No Lilis position can be inferred from this no-kill.")
            print(f"  HP: {old_hp} -> {session.hp}")
        else:
            session.save()
            print(f"Lilis night dealt 2HP but no kill; {len(unrevealed)} unrevealed positions remain: {sorted(unrevealed)}")
            print("  No Lilis position can be inferred yet.")
            print("  If a card actually died, use night_kill instead.")
            print(f"  HP: {old_hp} -> {session.hp}")
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
