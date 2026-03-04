"""Game loop adapter: bridges Claude's vision reads to the constraint solver.

Card builder functions, session tracking, CLI interface.
"""

from __future__ import annotations
import json
import os
import sys
from dataclasses import dataclass, field, asdict
from typing import Optional

from solver import CardInfo, DeckComposition, GameState, SolverResult, solve


# ============================================================
# Card Builder Functions
# ============================================================

def card_enlightened(pos: int, direction: str) -> CardInfo:
    """direction: 'CW', 'CCW', or 'Equidistant'"""
    return CardInfo(pos, "Enlightened", info_parsed={"direction": direction})

def card_knitter(pos: int, evil_pairs: int) -> CardInfo:
    return CardInfo(pos, "Knitter", info_parsed={"evil_pairs": evil_pairs})

def card_confessor(pos: int, dirty: bool) -> CardInfo:
    return CardInfo(pos, "Confessor", info_parsed={"dirty": dirty})

def card_gemcrafter(pos: int, good_position: int) -> CardInfo:
    return CardInfo(pos, "Gemcrafter", info_parsed={"good_position": good_position})

def card_lover(pos: int, evil_adjacent: int) -> CardInfo:
    return CardInfo(pos, "Lover", info_parsed={"evil_adjacent": evil_adjacent})

def card_scout(pos: int, evil_role: str, distance: int) -> CardInfo:
    return CardInfo(pos, "Scout", info_parsed={"evil_role": evil_role, "distance": distance})

def card_bard(pos: int, corruption_distance: int) -> CardInfo:
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

def card_bishop(pos: int, targets: list[int]) -> CardInfo:
    return CardInfo(pos, "Bishop", info_parsed={"targets": targets})

def card_no_info(pos: int, role: str) -> CardInfo:
    """For cards with no deduction info: Slayer, Knight, Bombardier, Wretch, Baker, etc."""
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

    # -- Deck --

    def set_deck(self, villagers: list[str], outcasts: list[str],
                 minions: list[str], demons: list[str]):
        self.villagers = villagers
        self.outcasts = outcasts
        self.minions = minions
        self.demons = demons

    # -- Cards --

    def add_card(self, card: CardInfo):
        # Replace if same position already exists (re-read)
        self.cards = [c for c in self.cards if c.position != card.position]
        self.cards.append(card)
        self.cards.sort(key=lambda c: c.position)

    def mark_executed(self, pos: int, was_evil: Optional[bool] = None):
        if pos not in self.executed:
            self.executed.append(pos)
        if was_evil is True and pos not in self.confirmed_evil:
            self.confirmed_evil.append(pos)
        elif was_evil is False and pos not in self.confirmed_good:
            self.confirmed_good.append(pos)

    def set_pd_target(self, pos: int):
        self.pd_corruption_target = pos

    # -- Solver --

    def _build_game_state(self) -> GameState:
        deck = DeckComposition(
            villagers=self.villagers,
            outcasts=self.outcasts,
            minions=self.minions,
            demons=self.demons,
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
        )

    def solve(self) -> SolverResult:
        state = self._build_game_state()
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
        path = game_utils.take_game_screenshot("_card_detect")
        positions = game_utils.detect_card_positions(path)
        if pos < 1 or pos > len(positions):
            print(f"[execute] Position #{pos} out of range (detected {len(positions)} cards)")
            return
        x, y = positions[pos - 1]
        game_utils.execute_card((x, y))
        print(f"[execute] Executed card #{pos} at ({x}, {y})")

    def deck_view(self) -> str:
        """Hold Tab, screenshot, return path."""
        import game_utils
        return game_utils.hold_tab_screenshot()

    # -- Persistence --

    def save(self, path: str = SESSION_FILE):
        data = {
            "n_cards": self.n_cards,
            "n_evil": self.n_evil,
            "villagers": self.villagers,
            "outcasts": self.outcasts,
            "minions": self.minions,
            "demons": self.demons,
            "cards": [
                {"position": c.position, "apparent_role": c.apparent_role,
                 "info_text": c.info_text, "info_parsed": c.info_parsed}
                for c in self.cards
            ],
            "executed": self.executed,
            "confirmed_evil": self.confirmed_evil,
            "confirmed_good": self.confirmed_good,
            "pd_corruption_target": self.pd_corruption_target,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[save] Session saved to {path}")

    @classmethod
    def load(cls, path: str = SESSION_FILE) -> "GameSession":
        with open(path) as f:
            data = json.load(f)
        session = cls(data["n_cards"], data["n_evil"])
        session.villagers = data["villagers"]
        session.outcasts = data["outcasts"]
        session.minions = data["minions"]
        session.demons = data["demons"]
        session.cards = [
            CardInfo(c["position"], c["apparent_role"], c.get("info_text", ""), c["info_parsed"])
            for c in data["cards"]
        ]
        session.executed = data.get("executed", [])
        session.confirmed_evil = data.get("confirmed_evil", [])
        session.confirmed_good = data.get("confirmed_good", [])
        session.pd_corruption_target = data.get("pd_corruption_target")
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
        dirty = args[2].lower() in ("dirty", "true", "1", "yes")
        return card_confessor(pos, dirty)
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
        return card_bishop(pos, targets)
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
        print("  new <n_cards> <n_evil>              Start new game session")
        print("  deck V=... O=... M=... D=...        Set deck composition")
        print("  card <role> <pos> [args...]          Add a revealed card")
        print("  execute <pos> [evil|good]            Mark position executed")
        print("  pd_target <pos>                      Set Plague Doctor corruption target")
        print("  solve                                Run solver")
        print("  status                               Print session state")
        print("  confirm_evil <pos>                   Mark position as confirmed evil")
        print("  confirm_good <pos>                   Mark position as confirmed good")
        print()
        print("Card examples:")
        print("  card enlightened 3 CW")
        print("  card confessor 1 clean")
        print("  card knitter 2 2")
        print("  card fortune_teller 4 1,3 yes")
        print("  card oracle 5 2,6 Shaman")
        print("  card jester 7 1,3,5 1")
        print("  card no_info 2 Slayer")
        return

    cmd = sys.argv[1].lower()

    if cmd == "new":
        n_cards = int(sys.argv[2])
        n_evil = int(sys.argv[3])
        session = GameSession(n_cards, n_evil)
        session.save()
        print(f"New session: {n_cards} cards, {n_evil} evil")
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
        session.set_deck(villagers, outcasts, minions, demons)
        session.save()
        print(f"Deck set: V={villagers} O={outcasts} M={minions} D={demons}")
        return

    if cmd == "card":
        session = GameSession.load()
        card = _parse_card_cli(sys.argv[2:])
        session.add_card(card)
        session.save()
        print(f"Added #{card.position} {card.apparent_role}: {card.info_parsed}")
        return

    if cmd == "execute":
        session = GameSession.load()
        pos = int(sys.argv[2])
        was_evil = None
        if len(sys.argv) > 3:
            w = sys.argv[3].lower()
            if w in ("evil", "true", "1", "yes"):
                was_evil = True
            elif w in ("good", "false", "0", "no"):
                was_evil = False
        session.mark_executed(pos, was_evil)
        session.save()
        tag = f" (was_evil={was_evil})" if was_evil is not None else ""
        print(f"Executed #{pos}{tag}")
        return

    if cmd == "pd_target":
        session = GameSession.load()
        pos = int(sys.argv[2])
        session.set_pd_target(pos)
        session.save()
        print(f"PD corruption target set to #{pos}")
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

    if cmd == "confirm_good":
        session = GameSession.load()
        pos = int(sys.argv[2])
        if pos not in session.confirmed_good:
            session.confirmed_good.append(pos)
        session.save()
        print(f"#{pos} confirmed good")
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
