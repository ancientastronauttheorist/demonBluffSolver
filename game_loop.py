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

from solver import CardInfo, DeckComposition, GameState, SolverResult
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

def card_dreamer(pos: int, target: int, evil_role: str) -> CardInfo:
    return CardInfo(pos, "Dreamer", info_parsed={"target": target, "evil_role": evil_role})

def card_judge(pos: int, target: int, is_lying: bool) -> CardInfo:
    return CardInfo(pos, "Judge", info_parsed={"target": target, "is_lying": is_lying})

def card_alchemist(pos: int, cured_count: int) -> CardInfo:
    return CardInfo(pos, "Alchemist", info_parsed={"cured_count": cured_count})

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


def card_no_info(pos: int, role: str) -> CardInfo:
    """For cards with no deduction info: Slayer, Knight, Bombardier, Wretch, etc."""
    role = _normalize_role_name(role)
    return CardInfo(pos, role, info_parsed={})


SESSION_FILE = os.path.join(os.path.dirname(__file__), "game_session.json")
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
    """Acquire a process-wide lock for the session file.

    Each CLI invocation is a short-lived process. Holding the lock from load
    until process exit serializes read-modify-write commands and prevents
    `game_session.json` corruption under rapid repeated automation.
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
        self.slayer_results: list[dict] = []  # [{slayer_pos, target_pos, killed}]
        self.night_kills: list[int] = []  # Positions killed by Lilis night
        self.night_kill_evil_count: int = 0  # How many night kills were evil
        self.hp: int = 10
        self.wrong_exec_cost: int = 5  # Asc4+ default (Drunk=2, Lilis=2 are exceptions)
        self.pd_ability_results: list[dict] = []  # [{"pd_pos": N, "target": N, "is_corrupted": bool, "evil_revealed": N|None}]
        self.blocked_positions: list[int] = []  # Positions blocked from reveal (e.g. Witch)
        self.executed_good_corrupted: dict[int, bool] = {}  # pos -> was corrupted (from execution observation)
        self.board_villager_count: Optional[int] = None  # Actual villagers on board (pool > board)
        self.board_outcast_count: Optional[int] = None   # Actual outcasts on board (pool > board)
        self.reveal_order: list[int] = []  # Order positions were flipped (for Baker)

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
        # Auto-mark ability used for active abilities entered with results
        # (Judge with target info; PD and Slayer have dedicated commands)
        if card.apparent_role == "Judge" and card.info_parsed.get("target"):
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
                      was_corrupted: Optional[bool] = None):
        if pos not in self.executed:
            self.executed.append(pos)
        if was_evil is True and pos not in self.confirmed_evil:
            self.confirmed_evil.append(pos)
        elif was_evil is False and pos not in self.confirmed_good:
            self.confirmed_good.append(pos)
        if evil_role:
            self.executed_evil_roles[pos] = evil_role
        # Track corruption status for executed good cards
        if was_evil is False and was_corrupted is not None:
            self.executed_good_corrupted[pos] = was_corrupted

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
                          evil_role: Optional[str] = None):
        self.slayer_results.append({
            "slayer_pos": slayer_pos,
            "target_pos": target_pos,
            "killed": killed,
        })
        self.mark_ability_used(slayer_pos)
        # Auto-mark killed target as executed (dead)
        if killed:
            if target_pos not in self.executed:
                self.executed.append(target_pos)
            if target_pos not in self.confirmed_evil:
                self.confirmed_evil.append(target_pos)
            if evil_role:
                self.executed_evil_roles[target_pos] = evil_role

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
            reveal_order=list(self.reveal_order),
            executed_good_corrupted=dict(self.executed_good_corrupted),
        )

    @classmethod
    def from_game_state(cls, state: GameState,
                        used_abilities: Optional[list[int]] = None) -> "GameSession":
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
        session.reveal_order = list(state.reveal_order)
        session.executed_good_corrupted = dict(getattr(state, 'executed_good_corrupted', {}))
        session.used_abilities = list(used_abilities or [])
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

    def auto_execute(self, pos: int, result) -> dict:
        """Perform in-game execution click sequence, verify via memory_reader, record result.

        Returns: {"success": bool, "was_evil": bool|None, "evil_role": str|None, "error": str|None}
        """
        import template_match as _tm
        import mouse as _mouse
        from game_utils import all_game_card_coords

        # Bombardier hard guard
        if pos in result.bombardier_positions:
            return {"success": False, "was_evil": None, "evil_role": None,
                    "error": f"Bombardier protection: refusing to execute #{pos}"}

        coords = all_game_card_coords(self.n_cards)
        if pos not in coords:
            return {"success": False, "was_evil": None, "evil_role": None,
                    "error": f"Position {pos} not valid for {self.n_cards}-card game"}

        # Step 1: Dismiss mark menu
        print(f"  [auto_exec] Dismissing mark menu...")
        _mouse.click(1280, 690)
        time.sleep(0.5)

        # Step 2: Click execute button
        print(f"  [auto_exec] Clicking execute button...")
        try:
            _tm.safe_click_at(2265, 1235, "btn_execute_sword")
        except Exception as e:
            return {"success": False, "was_evil": None, "evil_role": None,
                    "error": f"Execute button click failed: {e}"}
        time.sleep(0.5)

        # Step 3: Click target card
        x, y = coords[pos]
        print(f"  [auto_exec] Clicking #{pos} at ({x}, {y})...")
        _tm.safe_click_at(x, y, f"exec_card{pos}")
        time.sleep(3)  # wait for execution animation

        # Step 4: Verify via memory_reader
        print(f"  [auto_exec] Verifying execution via memory reader...")
        from memory_reader import MemoryReader
        reader = MemoryReader()
        if not reader.open():
            return {"success": False, "was_evil": None, "evil_role": None,
                    "error": "Cannot open game process for verification"}

        # Poll for state change (up to 3 attempts)
        target_card = None
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
            # Could be Knight immunity
            if target_card['state'] == 'Alive':
                print(f"  [auto_exec] #{pos} still Alive — possible Knight immunity")
                return {"success": True, "was_evil": False, "evil_role": None,
                        "error": "Knight immunity — card not killed"}
            return {"success": False, "was_evil": None, "evil_role": None,
                    "error": f"Card state is {target_card['state']}, expected Dead"}

        # Step 5: Determine result
        was_evil = target_card['is_evil']
        evil_role = target_card['true_role'] if was_evil else None

        # Step 6: Record into session
        was_corrupted = None
        if not was_evil:
            # Check if card was corrupted (for session tracking)
            statuses = target_card.get('statuses', [])
            was_corrupted = 'Corrupted' in statuses

        self.mark_executed(pos, was_evil, evil_role, was_corrupted)

        # Step 7: HP update
        if not was_evil:
            old_hp = self.hp
            self.hp -= self.wrong_exec_cost
            print(f"  [auto_exec] WRONG EXECUTION! HP {old_hp} -> {self.hp}")
        else:
            print(f"  [auto_exec] Correct execution. HP remains {self.hp}")

        self.save()
        DecisionLog.log_execution(pos, was_evil, evil_role)

        return {"success": True, "was_evil": was_evil, "evil_role": evil_role, "error": None}

    def auto_next(self):
        """Solve + auto-execute if definite evil. Returns (action, result, exec_result)."""
        state = self.to_game_state()
        result = self._solve(state)

        for line in result.reasoning:
            print(f"  {line}")
        DecisionLog.log_solver_output(result, state)
        action = print_recommendation(state, result, self.used_abilities)
        DecisionLog.log_recommendation(action)

        # Safety checks for auto-execution
        if action.action_type != "execute":
            print(f"\n  [auto_next] Not an execute recommendation — manual action needed.")
            return action, result, None

        pos = action.position
        if pos not in result.definite_evil:
            print(f"\n  [auto_next] #{pos} is NOT definite evil — manual decision needed.")
            return action, result, None

        if pos in result.bombardier_positions:
            print(f"\n  [auto_next] #{pos} is potential Bombardier — manual decision needed.")
            return action, result, None

        # Check HP budget — refuse if we can't afford a wrong exec
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
        print(f"\n  === AUTO-EXECUTING #{pos} (definite evil in all {result.n_surviving} scenarios) ===")
        exec_result = self.auto_execute(pos, result)

        if exec_result["success"]:
            if exec_result["was_evil"]:
                print(f"  AUTO-EXEC SUCCESS: #{pos} was {exec_result['evil_role']}")
            else:
                print(f"  AUTO-EXEC: #{pos} was GOOD (wrong execution)")
        else:
            print(f"  AUTO-EXEC FAILED: {exec_result['error']}")

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
        data = self.to_game_state().to_dict()
        data["used_abilities"] = list(self.used_abilities)

        tmp_path = f"{path}.tmp.{os.getpid()}"
        with open(tmp_path, "w") as f:
            json.dump(data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
        print(f"[save] Session saved to {path}")

    @classmethod
    def load(cls, path: str = SESSION_FILE) -> "GameSession":
        _acquire_session_lock(path)
        with open(path) as f:
            data = json.load(f)
        state = GameState.from_dict(data)
        session = cls.from_game_state(state, used_abilities=data.get("used_abilities", []))
        print(f"[load] Session loaded from {path}")
        return session


# ============================================================
# Flip Verification
# ============================================================

def _verify_flips(mr_output: str, expected_positions: list[int], session):
    """Check memory reader output to verify all targeted cards actually flipped.

    Parses memory reader table for state=Hidden lines and cross-references
    against positions we tried to flip. Flags unflipped cards loudly.
    """
    import re
    still_hidden = []
    for line in mr_output.splitlines():
        # Memory reader format: "# 1  RoleName   ... Hidden   NO  BLOCKED"
        m = re.match(r'^\s*#\s*(\d+)', line)
        if not m:
            continue
        pos = int(m.group(1))
        if pos in expected_positions and 'Hidden' in line and 'Dead' not in line:
            still_hidden.append(pos)

    if still_hidden:
        print()
        print("!" * 60)
        print("  FLIP VERIFICATION FAILED")
        print(f"  Positions still face-down: {still_hidden}")
        print(f"  Click likely didn't register (game unfocused?).")
        # Check if Witch is in the deck -- if not, this is definitely a click failure
        has_witch = session.has_role_in_deck("Witch")
        if not has_witch:
            print("  No Witch in deck -- this is NOT a Witch block.")
            print("  DO NOT mark as blocked. Re-run: python game_loop.py flip")
        else:
            print("  Witch IS in deck -- could be Witch blocking last card.")
            print("  If only last card is hidden, likely Witch. Otherwise re-flip.")
        print("!" * 60)


# ============================================================
# CLI
# ============================================================

def _parse_role_list(spec: str) -> list[str]:
    """Parse 'Knitter,Scout,Enlightened' into list."""
    if not spec or spec.lower() == "none":
        return []
    return [r.strip() for r in spec.split(",") if r.strip()]


def _parse_clue_from_memory(card: dict) -> Optional[CardInfo]:
    """Parse memory reader card data into a CardInfo, or None if unparseable.

    Handles passive clues read from savedAct/actedInfos/runtimeData.
    Active abilities (FT, Judge, Jester, Druid, Dreamer, Slayer) only work
    if the ability has already been used (acted_infos populated).
    """
    import re
    pos = card['position']
    role = card.get('disguise') or card.get('true_role', '')
    clue = card.get('clue_text') or ''
    infos = card.get('acted_infos', [])
    rd = card.get('runtime_data')
    targets = infos[0]['targets'] if infos else []
    role_lower = role.lower().replace(' ', '_')

    # --- RuntimeData takes priority (structured, no text parsing needed) ---
    if rd:
        if rd.get('type') == 'direction':
            return card_enlightened(pos, rd['direction'])
        if rd.get('type') == 'cures':
            return card_alchemist(pos, rd['cures'] or 0)

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
    if role_lower == 'dreamer' and targets:
        # Try to extract role name after target info
        m = re.search(r'is\s+(?:a\s+)?(\w[\w\s]*)', clue, re.IGNORECASE)
        if m:
            evil_role = m.group(1).strip()
            return card_dreamer(pos, targets[0], evil_role)

    # --- Oracle: targets + minion role ---
    if role_lower == 'oracle' and targets:
        # Look for a role name in the clue
        m = re.search(r'is\s+(?:a\s+)?(\w[\w\s]*)', clue, re.IGNORECASE)
        if m:
            minion_role = m.group(1).strip()
            return card_oracle(pos, targets, minion_role)

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

    # --- Wretch: no info ---
    if role_lower == 'wretch':
        return card_no_info(pos, 'Wretch')

    # --- No-info roles (Knight, Bombardier, Slayer pre-ability, etc.) ---
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
    elif role == "dreamer":
        return card_dreamer(pos, int(args[2]), args[3])
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
    if cv_result.returncode == 0:
        try:
            import json as _json
            cards = _json.loads(cv_result.stdout)
            cv_roles = [c["name"] for c in cards if c.get("accepted")]
            factions = {}
            for c in cards:
                if c.get("accepted"):
                    f = c.get("faction", "?")
                    factions.setdefault(f, []).append(c["name"])
            for faction in ["Villager", "Outcast", "Minion", "Demon"]:
                roles = factions.get(faction, [])
                if roles:
                    print(f"  {faction}s ({len(roles)}): {', '.join(roles)}")
        except Exception as e:
            print(f"  ERROR parsing card_vision output: {e}")
            cv_roles = []
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
    from tests.test_regression import save_test_case
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
        print("  start                                 Start new game (menu nav + deck read)")
        print("  new <n_cards> <n_evil> [hp=N cost=N] Start new game session")
        print("  deck V=... O=... M=... D=...         Set deck composition")
        print("  read_deck <screenshot>                Read deck (card_vision + memory_reader)")
        print("  flip                                  Flip all cards #1->#N in order")
        print("  flip <pos>                            Flip single card (after Witch death)")
        print("  flip --lilis                          Flip in batches of 4 (Lilis games)")
        print("  card <role> <pos> [args...]           Add a revealed card")
        print("  auto_card                             Auto-enter cards from memory reader")
        print("  execute <pos> [evil|good] [role]      Mark position executed (with evil role name)")
        print("  execute <pos> <RoleName>              Shorthand: mark as evil with role")
        print("  pd_target <pos>                       Set Plague Doctor corruption target")
        print("  pd_check <pd_pos> <target> corrupted <evil_pos>  PD found corruption + evil")
        print("  pd_check <pd_pos> <target> clean                 PD found no corruption")
        print("  set_hp <hp> [wrong_exec_cost]         Update HP and wrong execution cost")
        print("  solve                                 Run solver")
        print("  status                                Print session state")
        print("  confirm_evil <pos>                    Mark position as confirmed evil")
        print("  confirm_good <pos>                    Mark position as confirmed good")
        print("  next                                  Full strategy recommendation")
        print("  auto_next                             Solve + auto-execute if definite evil")
        print("  ability_used <pos>                    Mark ability as activated")
        print("  slayer_result <pos> <target> kill/fail [evil_role]  Slayer ability result")
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
    if cmd in ("start", "read_deck", "new"):
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
            if cmd in ("start", "read_deck", "new"):
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

    if cmd == "set_hp":
        session.hp = int(args[0])
        if len(args) > 1:
            session.wrong_exec_cost = int(args[1])
        session.save()
        print(f"HP set to {session.hp}, wrong exec cost = {session.wrong_exec_cost}")
        return None

    if cmd == "deck":
        villagers, outcasts, minions, demons = [], [], [], []
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
                session.board_villager_count = int(arg[3:])
            elif arg.lower().startswith("no="):
                session.board_outcast_count = int(arg[3:])
            else:
                print(f"  ERROR: Unrecognized arg '{arg}' -- missing prefix?")
                print(f"  Required: V=roles O=roles M=roles D=roles nv=N no=N")
                print(f"  Command aborted. Fix and re-run deck command.")
                return None
        session.set_deck(villagers, outcasts, minions, demons)
        if any(d.lower() == "baa" for d in demons):
            print("  WARNING: BAA in deck -- deck view shows +1 fake Outcast. "
                  "Subtract 1 from displayed outcast count for no= value.")
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
            return None

        already_done = set(session.reveal_order) | set(session.night_kills) | set(session.executed)
        positions = [p for p in sorted(coords.keys()) if p not in already_done]
        if not positions:
            print("All cards already flipped/dead. Nothing to flip.")
            return None
        if lilis:
            batch_size = 4
            batch = positions[:batch_size]
            print(f"Flipping batch: {['#'+str(p) for p in batch]}")
            for pos in batch:
                x, y = coords[pos]
                print(f"  #{pos} at ({x},{y})")
                _tm.fast_click_at(x, y, f"card{pos}")
                time.sleep(0.2)
            print(f"Batch complete: {['#'+str(p) for p in batch]}")
            remaining = positions[batch_size:]
            if remaining:
                print(f"\n  --- Lilis night phase (wait 5s for kill animation) ---")
                time.sleep(5)
                print(f"  Night phase complete. Take screenshot to check for kills before continuing.")
                print(f"  Run: python screenshot.py night_check && python memory_reader.py")
                print(f"  Remaining to flip: {['#'+str(p) for p in remaining]}")
        else:
            print(f"Flipping all {len(positions)} cards: #1 -> #{positions[-1]}")
            for pos in positions:
                x, y = coords[pos]
                print(f"  #{pos} at ({x},{y})")
                _tm.fast_click_at(x, y, f"card{pos}")
                time.sleep(0.2)
            print(f"All {len(positions)} cards flipped in order #1->#{positions[-1]}")

        print("\n--- Parking mouse & reading memory ---")
        time.sleep(1.5)
        _mouse.move(1280, 690)
        time.sleep(0.5)
        print("\n--- Memory Reader (board state) ---")
        mr = subprocess.run(["python", "memory_reader.py"], capture_output=True, text=True)
        if mr.returncode == 0:
            print(mr.stdout.strip())
            _verify_flips(mr.stdout, positions, session)
        else:
            print(f"  WARNING: memory_reader failed: {mr.stderr[:200]}")
        print("\nNow screenshot and enter card info in order #1->#{}.".format(positions[-1]))
        return None

    if cmd == "auto_card":
        import subprocess as _sp
        mr = _sp.run(["python", "memory_reader.py"], capture_output=True, text=True, timeout=10)
        if mr.returncode != 0:
            print(f"ERROR: memory_reader failed: {mr.stderr[:200]}")
            return None

        from memory_reader import MemoryReader
        reader = MemoryReader()
        if not reader.open():
            print("ERROR: Could not open game process")
            return None
        cards = reader.read_board()
        reader.close()
        if not cards:
            print("ERROR: No board data from memory reader")
            return None

        entered = {c.position for c in session.cards}
        dead = set(session.executed) | set(session.night_kills)
        auto_count = 0
        manual_needed = []

        for mc in cards:
            pos = mc['position']
            if pos in entered or pos in dead:
                continue
            state = mc.get('state', '')
            if state not in ('Alive', 'Revealed'):
                continue  # Hidden/Dead — skip

            parsed = _parse_clue_from_memory(mc)
            if parsed:
                session.add_card(parsed)
                DecisionLog.log_card(parsed)
                print(f"  [auto] #{parsed.position} {parsed.apparent_role}: {parsed.info_parsed}")
                auto_count += 1
            else:
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
        if len(args) > 1:
            w = args[1].lower()
            if w in ("evil", "true", "1", "yes"):
                was_evil = True
                if len(args) > 2:
                    evil_role = _normalize_role_name(args[2])
            elif w in ("good", "false", "0", "no"):
                was_evil = False
                if len(args) > 2:
                    c = args[2].lower()
                    if c in ("corrupted", "corrupt", "c"):
                        was_corrupted = True
                    elif c in ("clean", "uncorrupted", "u", "not_corrupted"):
                        was_corrupted = False
                else:
                    was_corrupted = None
                    print("  WARNING: No corruption flag given. Use 'execute <pos> good corrupted' or 'execute <pos> good clean'.")
            else:
                was_evil = True
                evil_role = _normalize_role_name(args[1])
        session.mark_executed(pos, was_evil, evil_role, was_corrupted)
        session.save()
        DecisionLog.log_execution(pos, was_evil, evil_role)
        tag = f" (evil: {evil_role})" if evil_role else (f" (was_evil={was_evil})" if was_evil is not None else "")
        corr_tag = ""
        if was_corrupted is True:
            corr_tag = " <Corrupted>"
        elif was_corrupted is False and was_evil is False:
            corr_tag = " (clean)"
        print(f"Executed #{pos}{tag}{corr_tag}")
        if was_evil:
            print(f"  HP: {session.hp}/10 (correct execution, no HP loss)")
        elif was_evil is False:
            new_hp = session.hp - session.wrong_exec_cost
            print(f"  WARNING: Wrong execution! HP {session.hp} -> {new_hp}. Run: set_hp {new_hp}")
        else:
            print(f"  REMINDER: Update HP with 'set_hp <current_hp>' after checking result")
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
        session.next_action()
        return None

    if cmd == "auto_next":
        session.auto_next()
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
        killed = args[2].lower() in ("kill", "killed", "true", "1", "yes")
        evil_role = args[3] if len(args) > 3 else None
        if killed and not evil_role:
            print(f"  ERROR: Slayer kill requires evil_role! Game reveals the role on kill.")
            print(f"  Usage: slayer_result {slayer_pos} {target_pos} kill <evil_role>")
            return None
        session.add_slayer_result(slayer_pos, target_pos, killed, evil_role=evil_role)
        session.save()
        result_str = f"killed #{target_pos}" if killed else f"couldn't kill #{target_pos}"
        if evil_role:
            result_str += f" (revealed: {evil_role})"
        print(f"Slayer #{slayer_pos} {result_str}")
        return None

    if cmd == "night_kill":
        positions = [int(x) for x in args[0].split(",")]
        n_evil = int(args[1])
        session.night_kills.extend(positions)
        session.night_kill_evil_count += n_evil
        for p in positions:
            if p not in session.executed:
                session.executed.append(p)
        if n_evil == len(positions) and n_evil > 0:
            for p in positions:
                if p not in session.confirmed_evil:
                    session.confirmed_evil.append(p)
        session.save()
        confirmed_msg = ""
        if n_evil == len(positions) and n_evil > 0:
            confirmed_msg = f" (confirmed evil: {['#'+str(p) for p in positions]})"
        print(f"Night kills: {['#'+str(p) for p in positions]}, {n_evil} evil among them{confirmed_msg}")
        return None

    if cmd == "night_no_kill":
        revealed = {c.position for c in session.cards}
        dead = set(session.executed)
        all_positions = set(range(1, session.n_cards + 1))
        unrevealed = all_positions - revealed - dead
        if len(unrevealed) == 1:
            lilis_pos = unrevealed.pop()
            if lilis_pos not in session.confirmed_evil:
                session.confirmed_evil.append(lilis_pos)
            session.save()
            print(f"Lilis night dealt 2HP but no kill — only unrevealed card is #{lilis_pos}")
            print(f"  => #{lilis_pos} confirmed as Lilis (can't kill herself)")
        elif len(unrevealed) == 0:
            print("WARNING: No unrevealed positions. Night shouldn't have triggered.")
        else:
            print(f"WARNING: {len(unrevealed)} unrevealed positions remain: {sorted(unrevealed)}")
            print("  Cannot auto-deduce Lilis — multiple unrevealed cards exist.")
            print("  Check if a card was actually killed and use night_kill instead.")
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
        true_evils_str = args[2] if len(args) > 2 else None
        notes = args[3] if len(args) > 3 else ""

        # Auto-read true evils from memory_reader if not provided
        if not true_evils_str and test_name:
            try:
                import subprocess as _sp
                mr = _sp.run(["python", "memory_reader.py"],
                             capture_output=True, text=True, timeout=10)
                if mr.returncode == 0:
                    # Parse memory_reader output for evil cards
                    auto_evils = {}
                    for line in mr.stdout.split("\n"):
                        line = line.strip()
                        # Format: "#N: RoleName (Evil) ..."
                        if "(Evil)" in line and line.startswith("#"):
                            import re
                            m = re.match(r"#(\d+):\s+(\S+)\s+\(Evil\)", line)
                            if m:
                                auto_evils[int(m.group(1))] = m.group(2)
                    if auto_evils:
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
            _save_and_run_test(test_name, true_evils, notes)
            print("\n--- Full v2 regression (Rust) ---")
            import subprocess as _sp
            try:
                reg = _sp.run(["cargo", "test", "--release", "--test", "replay"],
                              capture_output=True, text=True, timeout=120)
                for line in reg.stderr.strip().split("\n"):
                    if "test result:" in line or "FAILED" in line:
                        print(f"  {line.strip()}")
                if reg.returncode != 0:
                    print("  WARNING: Regression failures detected! Fix before next game.")
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
        _save_and_run_test(name, true_evils)
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

    print(f"Unknown command: {cmd}")
    print("Run 'python game_loop.py' for usage.")
    return None


if __name__ == "__main__":
    main()
