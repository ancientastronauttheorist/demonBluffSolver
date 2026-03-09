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

from solver import CardInfo, DeckComposition, GameState, SolverResult, solve
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


def card_no_info(pos: int, role: str) -> CardInfo:
    """For cards with no deduction info: Slayer, Knight, Bombardier, Wretch, etc."""
    return CardInfo(pos, role, info_parsed={})


# Builder dispatch table for CLI
CARD_BUILDERS = {
    "enlightened":     card_enlightened,
    "knitter":         card_knitter,
    "confessor":       card_confessor,
    "gemcrafter":      card_gemcrafter,
    "lover":           card_lover,
    "scout":           card_scout,
    "bard":            card_bard,
    "fortune_teller":  card_fortune_teller,
    "oracle":          card_oracle,
    "medium":          card_medium,
    "hunter":          card_hunter,
    "architect":       card_architect,
    "empress":         card_empress,
    "witness":         card_witness,
    "jester":          card_jester,
    "dreamer":         card_dreamer,
    "judge":           card_judge,
    "alchemist":       card_alchemist,
    "druid":           card_druid,
    "bishop":          card_bishop,
    "no_info":         card_no_info,
}

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
        self.wrong_exec_cost: int = 2  # Ascension 4 default
        self.pd_ability_results: list[dict] = []  # [{"pd_pos": N, "target": N, "is_corrupted": bool, "evil_revealed": N|None}]
        self.blocked_positions: list[int] = []  # Positions blocked from reveal (e.g. Witch)
        self.board_villager_count: Optional[int] = None  # Actual villagers on board (pool > board)
        self.board_outcast_count: Optional[int] = None   # Actual outcasts on board (pool > board)
        self.reveal_order: list[int] = []  # Order positions were flipped (for Baker)

    # -- Deck --

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
        # Replace if same position already exists (re-read)
        self.cards = [c for c in self.cards if c.position != card.position]
        self.cards.append(card)
        self.cards.sort(key=lambda c: c.position)
        # Auto-mark ability used for active abilities entered with results
        # (Judge with target info; PD and Slayer have dedicated commands)
        if card.apparent_role == "Judge" and card.info_parsed.get("target"):
            self.mark_ability_used(card.position)

    def mark_executed(self, pos: int, was_evil: Optional[bool] = None,
                      evil_role: Optional[str] = None):
        if pos not in self.executed:
            self.executed.append(pos)
        if was_evil is True and pos not in self.confirmed_evil:
            self.confirmed_evil.append(pos)
        elif was_evil is False and pos not in self.confirmed_good:
            self.confirmed_good.append(pos)
        if evil_role:
            self.executed_evil_roles[pos] = evil_role

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
        session.used_abilities = list(used_abilities or [])
        return session

    def solve(self) -> SolverResult:
        state = self.to_game_state()
        result = solve(state)
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
                unrevealed = set(range(1, self.n_cards + 1)) - set(self.executed) - set(result.definite_good)
                for pos in sorted(unrevealed):
                    evil_count = sum(1 for s in result.surviving_scenarios
                                     if pos in s.evil_positions or pos == s.puppet_position)
                    pct = evil_count / result.n_surviving * 100
                    if 0 < pct < 100:
                        print(f"     #{pos}: {pct:.0f}% chance evil ({evil_count}/{result.n_surviving})")
        print(f"  ({result.n_surviving} surviving scenarios out of {result.n_scenarios})\n")
        return result

    def recommend(self) -> str:
        result = self.solve()
        safe_to_execute = [p for p in result.definite_evil
                           if p not in self.executed and p not in result.bombardier_positions]
        if safe_to_execute:
            return f"Execute #{safe_to_execute[0]}"
        return "Reveal more cards"

    def next_action(self):
        """Run solver + strategy, print full recommendation."""
        state = self.to_game_state()
        result = solve(state)
        for line in result.reasoning:
            print(f"  {line}")
        DecisionLog.log_solver_output(result, state)
        action = print_recommendation(state, result, self.used_abilities)
        DecisionLog.log_recommendation(action)
        return action

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
# CLI
# ============================================================

def _parse_role_list(spec: str) -> list[str]:
    """Parse 'Knitter,Scout,Enlightened' into list."""
    if not spec or spec.lower() == "none":
        return []
    return [r.strip() for r in spec.split(",") if r.strip()]


def _parse_card_cli(args: list[str]) -> CardInfo:
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
    elif role == "fortune_teller":
        targets = [int(x) for x in args[2].split(",")]
        has_evil = args[3].lower() in ("yes", "true", "1")
        return card_fortune_teller(pos, targets, has_evil)
    elif role == "oracle":
        targets = [int(x) for x in args[2].split(",")]
        return card_oracle(pos, targets, args[3])
    elif role == "medium":
        return card_medium(pos, int(args[2]), args[3])
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
            # Normalize: "Baker" or "original" both mean the original Baker
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


def main():
    if len(sys.argv) < 2:
        print("Usage: python game_loop.py <command> [args...]")
        print()
        print("Commands:")
        print("  new <n_cards> <n_evil> [hp=N cost=N] Start new game session")
        print("  deck V=... O=... M=... D=...         Set deck composition")
        print("  card <role> <pos> [args...]           Add a revealed card")
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
        print("  ability_used <pos>                    Mark ability as activated")
        print("  slayer_result <pos> <target> kill/fail [evil_role]  Slayer ability result")
        print("  block <pos>                           Mark position as blocked (Witch)")
        print("  unblock <pos>                         Unblock position (after Witch dies)")
        print("  night_kill <pos1,pos2,...> <n_evil>    Lilis night kills (positions + evil count)")
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

    if cmd == "new":
        n_cards = int(sys.argv[2])
        n_evil = int(sys.argv[3])
        session = GameSession(n_cards, n_evil)
        # Optional: hp, wrong_exec_cost, and deck via --flags
        i = 4
        while i < len(sys.argv):
            arg = sys.argv[i]
            if arg.startswith("hp="):
                session.hp = int(arg[3:])
            elif arg.startswith("cost="):
                session.wrong_exec_cost = int(arg[5:])
            elif arg == "--villagers" and i + 1 < len(sys.argv):
                i += 1
                session.villagers = _parse_role_list(sys.argv[i])
            elif arg == "--outcasts" and i + 1 < len(sys.argv):
                i += 1
                session.outcasts = _parse_role_list(sys.argv[i])
            elif arg == "--minions" and i + 1 < len(sys.argv):
                i += 1
                session.minions = _parse_role_list(sys.argv[i])
            elif arg == "--demons" and i + 1 < len(sys.argv):
                i += 1
                session.demons = _parse_role_list(sys.argv[i])
            i += 1
        session.save()
        DecisionLog.start_game(n_cards, n_evil, session.hp, session.wrong_exec_cost)
        print(f"New session: {n_cards} cards, {n_evil} evil, HP={session.hp}, cost={session.wrong_exec_cost}")
        return

    if cmd == "set_hp":
        session = GameSession.load()
        session.hp = int(sys.argv[2])
        if len(sys.argv) > 3:
            session.wrong_exec_cost = int(sys.argv[3])
        session.save()
        print(f"HP set to {session.hp}, wrong exec cost = {session.wrong_exec_cost}")
        return

    if cmd == "deck":
        session = GameSession.load()
        villagers, outcasts, minions, demons = [], [], [], []
        for arg in sys.argv[2:]:
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
        session.set_deck(villagers, outcasts, minions, demons)
        # Auto-detect extra roles: if pool > board, infer board counts
        pool_size = len(villagers) + len(outcasts) + len(minions) + len(demons)
        if pool_size > session.n_cards and session.board_villager_count is None:
            # Evil roles are always all on board; extra roles are among good
            board_good = session.n_cards - session.n_evil
            board_evil = len(minions) + len(demons)
            if board_evil == session.n_evil:
                # Derive: board_outcasts + board_villagers = board_good
                # Pool has extra villagers and/or outcasts
                extra = pool_size - session.n_cards
                extra_v = len(villagers) - (board_good - len(outcasts))
                extra_o = len(outcasts) - (board_good - len(villagers))
                # Simpler: we know total good = n_cards - n_evil
                # board_v + board_o = board_good
                # pool_v + pool_o = board_good + extra
                # We can't determine split without header info, so prompt user
                print(f"  NOTE: Pool has {pool_size} roles for {session.n_cards} board positions.")
                print(f"  Use nv=N no=N to specify actual board counts (e.g., deck ... nv=6 no=1)")
        session.save()
        DecisionLog.log_deck(villagers, outcasts, minions, demons)
        extra_info = ""
        if session.board_villager_count is not None or session.board_outcast_count is not None:
            extra_info = f" [board: nv={session.board_villager_count} no={session.board_outcast_count}]"
        print(f"Deck set: V={villagers} O={outcasts} M={minions} D={demons}{extra_info}")
        return

    if cmd == "card":
        session = GameSession.load()
        card = _parse_card_cli(sys.argv[2:])
        session.add_card(card)
        session.save()
        DecisionLog.log_card(card)
        print(f"Added #{card.position} {card.apparent_role}: {card.info_parsed}")
        return

    if cmd == "execute":
        session = GameSession.load()
        pos = int(sys.argv[2])
        was_evil = None
        evil_role = None
        if len(sys.argv) > 3:
            w = sys.argv[3].lower()
            if w in ("evil", "true", "1", "yes"):
                was_evil = True
                # Optional 4th arg: evil role name (e.g., "Chancellor")
                if len(sys.argv) > 4:
                    evil_role = sys.argv[4]
            elif w in ("good", "false", "0", "no"):
                was_evil = False
            else:
                # Treat as evil role name directly: execute 2 Chancellor
                was_evil = True
                evil_role = sys.argv[3]
        session.mark_executed(pos, was_evil, evil_role)
        session.save()
        DecisionLog.log_execution(pos, was_evil, evil_role)
        tag = f" (evil: {evil_role})" if evil_role else (f" (was_evil={was_evil})" if was_evil is not None else "")
        print(f"Executed #{pos}{tag}")
        return

    if cmd == "pd_target":
        session = GameSession.load()
        pos = int(sys.argv[2])
        session.set_pd_target(pos)
        session.save()
        print(f"PD corruption target set to #{pos}")
        return

    if cmd == "pd_check":
        session = GameSession.load()
        pd_pos = int(sys.argv[2])
        target = int(sys.argv[3])
        status = sys.argv[4].lower()
        if status == "corrupted":
            evil_revealed = int(sys.argv[5])
            session.add_pd_ability_result(pd_pos, target, True, evil_revealed)
            session.save()
            print(f"PD #{pd_pos} checked #{target}: Corrupted, #{evil_revealed} is Evil")
        elif status == "clean":
            session.add_pd_ability_result(pd_pos, target, False)
            session.save()
            print(f"PD #{pd_pos} checked #{target}: Not Corrupted")
        else:
            print(f"Unknown PD check status: {status} (use 'corrupted' or 'clean')")
        return

    if cmd == "solve":
        session = GameSession.load()
        session.solve()
        return

    if cmd == "status":
        session = GameSession.load()
        session.status()
        return

    if cmd == "confirm_evil":
        session = GameSession.load()
        pos = int(sys.argv[2])
        if pos not in session.confirmed_evil:
            session.confirmed_evil.append(pos)
        session.save()
        print(f"#{pos} confirmed evil")
        return

    if cmd == "block":
        session = GameSession.load()
        pos = int(sys.argv[2])
        if pos not in session.blocked_positions:
            session.blocked_positions.append(pos)
        session.save()
        print(f"#{pos} blocked (Witch)")
        return

    if cmd == "unblock":
        session = GameSession.load()
        pos = int(sys.argv[2])
        if pos in session.blocked_positions:
            session.blocked_positions.remove(pos)
        session.save()
        print(f"#{pos} unblocked")
        return

    if cmd == "confirm_good":
        session = GameSession.load()
        pos = int(sys.argv[2])
        if pos not in session.confirmed_good:
            session.confirmed_good.append(pos)
        session.save()
        print(f"#{pos} confirmed good")
        return

    if cmd == "next":
        session = GameSession.load()
        session.next_action()
        return

    if cmd == "ability_used":
        session = GameSession.load()
        pos = int(sys.argv[2])
        session.mark_ability_used(pos)
        session.save()
        DecisionLog.log_ability_used(pos)
        print(f"Ability at #{pos} marked as used")
        return

    if cmd == "slayer_result":
        session = GameSession.load()
        slayer_pos = int(sys.argv[2])
        target_pos = int(sys.argv[3])
        killed = sys.argv[4].lower() in ("kill", "killed", "true", "1", "yes")
        evil_role = sys.argv[5] if len(sys.argv) > 5 else None
        session.add_slayer_result(slayer_pos, target_pos, killed, evil_role=evil_role)
        session.save()
        result_str = f"killed #{target_pos}" if killed else f"couldn't kill #{target_pos}"
        if evil_role:
            result_str += f" (revealed: {evil_role})"
        print(f"Slayer #{slayer_pos} {result_str}")
        return

    if cmd == "night_kill":
        session = GameSession.load()
        positions = [int(x) for x in sys.argv[2].split(",")]
        n_evil = int(sys.argv[3])
        session.night_kills = positions
        session.night_kill_evil_count = n_evil
        # Also mark as executed (dead)
        for p in positions:
            if p not in session.executed:
                session.executed.append(p)
        session.save()
        print(f"Night kills: {['#'+str(p) for p in positions]}, {n_evil} evil among them")
        return

    if cmd == "log":
        # Log Claude's reasoning: python game_loop.py log "label" "text"
        label = sys.argv[2] if len(sys.argv) > 2 else "Claude Reasoning"
        text = sys.argv[3] if len(sys.argv) > 3 else ""
        DecisionLog.log_custom(label, text)
        print(f"[log] Logged: {label}")
        return

    if cmd == "game_over":
        # Log game result + auto-save regression test
        # Usage: python game_loop.py game_over win/loss <test_name> <true_evils> [notes]
        # Example: python game_loop.py game_over loss s34_v1_asc6 "3=Shaman,7=Baa" "trusted fake PD"
        session = GameSession.load()
        result = sys.argv[2] if len(sys.argv) > 2 else "unknown"
        test_name = sys.argv[3] if len(sys.argv) > 3 else None
        true_evils_str = sys.argv[4] if len(sys.argv) > 4 else None
        notes = sys.argv[5] if len(sys.argv) > 5 else ""
        DecisionLog.log_game_over(result, session.hp, notes)
        print(f"[game_over] Logged: {result.upper()}, HP={session.hp}")

        # Record to scorecard
        from scorecard import record as scorecard_record
        scorecard_record(result, session.hp, test_name or "", notes)

        # Auto-save regression test if true evils provided
        if test_name and true_evils_str:
            true_evils = {}
            for pair in true_evils_str.split(","):
                pos_str, role = pair.split("=")
                true_evils[int(pos_str)] = role
            from tests.test_regression import save_test_case, load_test_case, run_test
            save_test_case(SESSION_FILE, test_name, true_evils, notes)
            case = load_test_case(os.path.join("tests", "cases", f"{test_name}.json"))
            passed, messages = run_test(case)
            status = "PASS" if passed else "FAIL"
            print(f"\n[{status}] {test_name}")
            for msg in messages:
                print(f"  {msg}")
        elif not test_name:
            print("[game_over] Tip: add test name + true evils to auto-save regression test:")
            print("  game_over win/loss <name> <pos=Role,...> [notes]")
        return

    if cmd == "save_test":
        session = GameSession.load()
        name = sys.argv[2] if len(sys.argv) > 2 else "unnamed"
        # Parse true evil positions: "2=Chancellor,7=Baa" or JSON
        true_evils = {}
        if len(sys.argv) > 3:
            raw = sys.argv[3]
            if raw.startswith("{"):
                import ast
                true_evils = {int(k): v for k, v in ast.literal_eval(raw).items()}
            else:
                for pair in raw.split(","):
                    pos_str, role = pair.split("=")
                    true_evils[int(pos_str)] = role
        from tests.test_regression import save_test_case
        save_test_case(SESSION_FILE, name, true_evils)
        # Also run the test immediately
        from tests.test_regression import load_test_case, run_test
        case = load_test_case(os.path.join("tests", "cases", f"{name}.json"))
        passed, messages = run_test(case)
        status = "PASS" if passed else "FAIL"
        print(f"\n[{status}] {name}")
        for msg in messages:
            print(f"  {msg}")
        return

    # Game action commands (require game running)
    if cmd == "screenshot":
        session = GameSession.load()
        name = sys.argv[2] if len(sys.argv) > 2 else None
        path = session.screenshot(name)
        print(f"Screenshot: {path}")
        return

    if cmd == "reveal":
        session = GameSession.load()
        pos = int(sys.argv[2])
        session.reveal(pos)
        return

    if cmd == "deck_view":
        session = GameSession.load()
        path = session.deck_view()
        print(f"Deck view: {path}")
        return

    print(f"Unknown command: {cmd}")
    print("Run 'python game_loop.py' for usage.")


if __name__ == "__main__":
    main()
