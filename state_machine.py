"""Autonomous game state machine for Demon Bluff Solver.

Drives the game loop through phases using MemoryMonitor as the primary
data source. Pauses for human input when automation can't proceed safely.

Usage in REPL:
    auto_loop        # start autonomous play
    resume           # resume after NEEDS_HUMAN
"""

from enum import Enum
from typing import Optional


class GamePhase(Enum):
    IDLE = "idle"
    FLIPPING = "flipping"
    ENTERING_CLUES = "entering_clues"
    SOLVING = "solving"
    EXECUTING = "executing"
    LILIS_NIGHT = "night"
    GAME_OVER = "game_over"
    NEEDS_HUMAN = "needs_human"


class GameStateMachine:
    """Autonomous game loop driven by memory reader.

    Safety gates (always NEEDS_HUMAN):
    - Bombardier on board and target not in definite_evil
    - Active ability recommended (Slayer, Judge, Jester, Druid)
    - Probabilistic execution (not definite_evil)
    - HP <= wrong_exec_cost and multiple scenarios remain
    - 0 surviving scenarios (solver bug)
    - Lilis night kill entry (need visual confirmation)
    """

    def __init__(self, session, monitor=None):
        self.session = session
        self.monitor = monitor
        self.phase = GamePhase.IDLE
        self._needs_human_reason = ""
        self._last_card_count = len(session.cards)
        self._last_exec_count = len(session.executed)
        self._last_ability_count = len(session.used_abilities)

    @property
    def is_active(self):
        return self.phase not in (GamePhase.IDLE, GamePhase.GAME_OVER, GamePhase.NEEDS_HUMAN)

    def start(self):
        """Begin autonomous play from current session state."""
        print(f"\n=== AUTO LOOP STARTED ===")
        # Determine starting phase from session state
        entered = {c.position for c in self.session.cards}
        dead = set(self.session.executed) | set(self.session.night_kills)
        all_positions = set(range(1, self.session.n_cards + 1))
        flipped = set(self.session.reveal_order)
        blocked = set(self.session.blocked_positions)
        unrevealed = all_positions - flipped - dead - blocked

        if unrevealed and not flipped:
            self.phase = GamePhase.FLIPPING
        elif len(entered) < len(flipped - dead):
            self.phase = GamePhase.ENTERING_CLUES
        else:
            self.phase = GamePhase.SOLVING

        self._run_loop()

    def resume(self):
        """Resume after NEEDS_HUMAN. Re-evaluates state and picks phase."""
        if self.phase != GamePhase.NEEDS_HUMAN:
            print(f"  [auto] Not in NEEDS_HUMAN state (currently {self.phase.value})")
            return

        # Detect what changed since we paused
        new_cards = len(self.session.cards) > self._last_card_count
        new_exec = len(self.session.executed) > self._last_exec_count
        new_ability = len(self.session.used_abilities) > self._last_ability_count

        if new_cards or new_exec or new_ability:
            self._snapshot_counts()
            self.phase = GamePhase.SOLVING
        else:
            # Check if all clues are entered now
            entered = {c.position for c in self.session.cards}
            flipped = set(self.session.reveal_order)
            dead = set(self.session.executed) | set(self.session.night_kills)
            needs_entry = flipped - dead - entered - set(self.session.blocked_positions)
            if not needs_entry:
                self.phase = GamePhase.SOLVING
            else:
                self.phase = GamePhase.ENTERING_CLUES

        print(f"  [auto] Resuming at phase: {self.phase.value}")
        self._run_loop()

    def _run_loop(self):
        """Main loop: tick until GAME_OVER or NEEDS_HUMAN."""
        while self.phase not in (GamePhase.IDLE, GamePhase.GAME_OVER, GamePhase.NEEDS_HUMAN):
            try:
                self.tick()
            except Exception as e:
                print(f"\n  [auto] ERROR in {self.phase.value}: {e}")
                self._pause(f"Error: {e}")
                return

    def tick(self):
        """Advance one step based on current phase."""
        if self.phase == GamePhase.FLIPPING:
            self._do_flipping()
        elif self.phase == GamePhase.ENTERING_CLUES:
            self._do_entering_clues()
        elif self.phase == GamePhase.SOLVING:
            self._do_solving()
        elif self.phase == GamePhase.EXECUTING:
            self._do_executing()
        elif self.phase == GamePhase.LILIS_NIGHT:
            self._do_lilis_night()
        elif self.phase == GamePhase.GAME_OVER:
            pass
        elif self.phase == GamePhase.NEEDS_HUMAN:
            pass

    def _do_flipping(self):
        """Flip all unrevealed cards."""
        from game_loop import dispatch
        print(f"\n  [auto] Phase: FLIPPING")

        if self.session.is_lilis_alive():
            dispatch("flip", ["--lilis"], self.session)
            # Lilis night handled by flip command; check if night kill needed
            total_reveals = len(self.session.reveal_order)
            if total_reveals % 4 == 0:
                self._pause("Lilis night — verify kills and enter: night_kill <pos> <n_evil> OR night_no_kill")
                return
            # More batches needed
            remaining = self._unrevealed_positions()
            if remaining:
                return  # stay in FLIPPING for next batch
        else:
            dispatch("flip", [], self.session)

        self.phase = GamePhase.ENTERING_CLUES

    def _do_entering_clues(self):
        """Auto-enter clues from memory reader."""
        from game_loop import dispatch
        print(f"\n  [auto] Phase: ENTERING CLUES")

        dispatch("auto_card", [], self.session)

        # Check what still needs manual entry
        entered = {c.position for c in self.session.cards}
        flipped = set(self.session.reveal_order)
        dead = set(self.session.executed) | set(self.session.night_kills)
        blocked = set(self.session.blocked_positions)
        needs_entry = sorted(flipped - dead - entered - blocked)

        if needs_entry:
            roles = []
            if self.monitor and self.monitor.is_healthy():
                board = self.monitor.get_board()
                if board:
                    for pos in needs_entry:
                        card = next((c for c in board if c['position'] == pos), None)
                        if card:
                            role = card.get('disguise') or '?'
                            roles.append(f"#{pos} ({role})")
                        else:
                            roles.append(f"#{pos}")
                else:
                    roles = [f"#{p}" for p in needs_entry]
            else:
                roles = [f"#{p}" for p in needs_entry]
            self._pause(f"Manual card entry needed: {', '.join(roles)}")
        else:
            self.phase = GamePhase.SOLVING

    def _do_solving(self):
        """Run solver and decide next action."""
        from strategy import recommend_action, print_recommendation, evil_probabilities, _compute_confidence
        print(f"\n  [auto] Phase: SOLVING")

        state = self.session.to_game_state()
        result = self.session._solve(state)

        for line in result.reasoning:
            print(f"  {line}")

        probs = evil_probabilities(state, result)
        action = recommend_action(state, result, self.session.used_abilities)
        action.confidence = _compute_confidence(action, state, result, probs)

        print(f"\n  [auto] Recommendation: {action.action_type.upper()}", end="")
        if action.position:
            print(f" #{action.position}", end="")
        print(f" (confidence: {action.confidence:.0%}, {result.n_surviving} scenarios)")
        print(f"  Reason: {action.reasoning}")
        for w in action.warnings:
            print(f"  WARNING: {w}")

        # Decision tree
        if action.action_type == "win":
            self.phase = GamePhase.GAME_OVER
            print(f"\n  [auto] ALL EVIL EXECUTED - GAME WON!")

        elif action.action_type == "error":
            self._pause(f"Solver error: {action.reasoning}")

        elif action.action_type == "execute":
            if action.position in result.definite_evil and action.position not in result.bombardier_positions:
                # Safe auto-execute
                self._pending_exec = (action.position, result)
                self.phase = GamePhase.EXECUTING
            else:
                # Not definite evil or Bombardier risk — human decides
                self._pause(f"Execute #{action.position}? ({action.confidence:.0%} confident) — not definite evil, needs manual decision")

        elif action.action_type == "use_ability":
            self._pause(f"Use {action.ability_name} on #{action.position} -> targets {action.targets}")

        elif action.action_type == "reveal":
            self._pause(f"Reveal #{action.position} — unrevealed cards remain")

        else:
            self._pause(f"Unknown action: {action.action_type}")

    def _do_executing(self):
        """Auto-execute a definite evil target."""
        pos, result = self._pending_exec
        print(f"\n  [auto] Phase: EXECUTING #{pos}")

        exec_result = self.session.auto_execute(pos, result, monitor=self.monitor)

        if exec_result["success"]:
            if exec_result["was_evil"]:
                print(f"  [auto] Executed #{pos}: {exec_result['evil_role']} (EVIL)")
            elif exec_result.get("error") and "Knight" in exec_result["error"]:
                print(f"  [auto] Knight immunity on #{pos}")
                self._pause(f"Knight immunity on #{pos} — manual handling needed")
                return
            else:
                print(f"  [auto] WRONG EXECUTION on #{pos}! HP now {self.session.hp}")
                if self.session.hp <= 0:
                    self.phase = GamePhase.GAME_OVER
                    print(f"\n  [auto] HP DEPLETED - GAME LOST!")
                    return
                self._pause(f"Wrong exec on definite evil #{pos} — possible solver bug!")
                return
        else:
            self._pause(f"Execution failed: {exec_result['error']}")
            return

        self._snapshot_counts()
        # Re-solve to check for more definite evils or win
        self.phase = GamePhase.SOLVING

    def _do_lilis_night(self):
        """Handle Lilis night phase."""
        self._pause("Lilis night — verify kills and enter: night_kill <pos> <n_evil> OR night_no_kill")

    def _pause(self, reason: str):
        """Transition to NEEDS_HUMAN with a reason."""
        self.phase = GamePhase.NEEDS_HUMAN
        self._needs_human_reason = reason
        self._snapshot_counts()
        print(f"\n  [auto] PAUSED: {reason}")
        print(f"  [auto] Run commands manually, then type 'resume' to continue.")

    def _snapshot_counts(self):
        """Snapshot current counts for change detection in resume()."""
        self._last_card_count = len(self.session.cards)
        self._last_exec_count = len(self.session.executed)
        self._last_ability_count = len(self.session.used_abilities)

    def _unrevealed_positions(self):
        """Get positions that still need flipping."""
        all_positions = set(range(1, self.session.n_cards + 1))
        done = set(self.session.reveal_order) | set(self.session.night_kills) | set(self.session.executed)
        return sorted(all_positions - done)
