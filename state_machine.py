"""Autonomous game state machine for Demon Bluff Solver.

Drives the game loop through phases using MemoryMonitor as the primary
data source. Pauses for human input when automation can't proceed safely.

Usage in REPL:
    auto_loop        # start autonomous play
    resume           # resume after NEEDS_HUMAN
"""

import copy
from enum import Enum
from typing import Optional


class GamePhase(Enum):
    IDLE = "idle"
    MENU_NAV = "menu_nav"
    DECK_READ = "deck_read"
    SESSION_INIT = "session_init"
    FLIPPING = "flipping"
    ENTERING_CLUES = "entering_clues"
    SOLVING = "solving"
    EXECUTING = "executing"
    REVEALING = "revealing"
    ABILITY_USE = "ability_use"
    LILIS_NIGHT = "night"
    NIGHT_RESOLVE = "night_resolve"
    GAME_OVER = "game_over"
    POST_GAME = "post_game"
    NEEDS_HUMAN = "needs_human"


def _acted_history_baseline(card):
    """Copy one readable native acted-info history for freshness checks."""
    if not isinstance(card, dict):
        return None
    history = card.get("acted_infos")
    if not isinstance(history, list):
        return None
    return copy.deepcopy(history)


def _coherent_appended_acted_event(card, baseline):
    """Return the newest event only for an exact strict history extension."""
    if not isinstance(card, dict) or not isinstance(baseline, list):
        return None
    history = card.get("acted_infos")
    if (
        not isinstance(history, list)
        or len(history) <= len(baseline)
        or history[:len(baseline)] != baseline
    ):
        return None
    newest = history[-1]
    if not isinstance(newest, dict):
        return None
    desc = newest.get("desc")
    if not isinstance(desc, str) or not desc:
        return None
    if card.get("clue_text") != desc:
        return None
    return copy.deepcopy(newest)


class GameStateMachine:
    """Autonomous game loop driven by memory reader.

    Safety gates (always NEEDS_HUMAN):
    - Probabilistic execution (not definite_evil and not forced_safe)
    - HP <= wrong_exec_cost and multiple scenarios remain
    - 0 surviving scenarios (solver bug)
    - Active ability with complex kill-result interaction (Slayer)
    - Target card has unused active ability (would trigger wrong ability)
    """

    def __init__(self, session=None, monitor=None, risk="conservative"):
        self.session = session
        self.monitor = monitor
        self.risk = risk  # "conservative", "moderate", "aggressive"
        self.phase = GamePhase.IDLE
        self._needs_human_reason = ""
        self._last_card_count = len(session.cards) if session else 0
        self._last_exec_count = len(session.executed) if session else 0
        self._last_ability_count = len(session.used_abilities) if session else 0
        # Pending actions (set by SOLVING, consumed by phase handlers)
        self._pending_exec = None       # (pos, result, forced_safe)
        self._pending_reveal = None     # (pos,)
        self._pending_ability = None    # (pos, targets, ability_name, result)
        # Game result (set by GAME_OVER, consumed by POST_GAME)
        self._game_result = None        # "win" or "loss"

    @property
    def is_active(self):
        return self.phase not in (GamePhase.IDLE, GamePhase.GAME_OVER, GamePhase.POST_GAME, GamePhase.NEEDS_HUMAN)

    def start_full_game(self):
        """Start a full game from menu navigation (for batch runner)."""
        print(f"\n=== AUTO FULL GAME STARTED ===")
        self.phase = GamePhase.MENU_NAV
        self._run_loop()

    def start(self):
        """Begin autonomous play from current session state."""
        print(f"\n=== AUTO LOOP STARTED ===")
        # Determine starting phase from session state
        entered = {c.position for c in self.session.cards}
        dead = set(self.session.executed) | set(self.session.night_kills)
        all_positions = set(range(1, self.session.n_cards + 1))
        flipped = set(self.session.reveal_order)
        blocked = self._active_blocked_positions()
        unrevealed = all_positions - flipped - dead - blocked

        if self.session.pending_lilis_nights > 0:
            self.phase = GamePhase.LILIS_NIGHT
        elif unrevealed and not flipped:
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

        if self.session.pending_lilis_nights > 0:
            self.phase = GamePhase.LILIS_NIGHT
        elif new_cards or new_exec or new_ability:
            self._snapshot_counts()
            self.phase = GamePhase.SOLVING
        else:
            # Check if all clues are entered now
            entered = {c.position for c in self.session.cards}
            flipped = set(self.session.reveal_order)
            dead = set(self.session.executed) | set(self.session.night_kills)
            needs_entry = flipped - dead - entered - self._active_blocked_positions()
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
                import traceback
                traceback.print_exc()
                self._pause(f"Error: {e}")
                return

    def tick(self):
        """Advance one step based on current phase."""
        if self.phase == GamePhase.MENU_NAV:
            self._do_menu_nav()
        elif self.phase == GamePhase.DECK_READ:
            self._do_deck_read()
        elif self.phase == GamePhase.SESSION_INIT:
            self._do_session_init()
        elif self.phase == GamePhase.FLIPPING:
            self._do_flipping()
        elif self.phase == GamePhase.ENTERING_CLUES:
            self._do_entering_clues()
        elif self.phase == GamePhase.SOLVING:
            self._do_solving()
        elif self.phase == GamePhase.EXECUTING:
            self._do_executing()
        elif self.phase == GamePhase.REVEALING:
            self._do_revealing()
        elif self.phase == GamePhase.ABILITY_USE:
            self._do_ability_use()
        elif self.phase == GamePhase.LILIS_NIGHT:
            self._do_lilis_night()
        elif self.phase == GamePhase.NIGHT_RESOLVE:
            self._do_night_resolve()
        elif self.phase == GamePhase.GAME_OVER:
            self._do_game_over()
        elif self.phase == GamePhase.POST_GAME:
            pass  # handled by BatchGameRunner
        elif self.phase == GamePhase.NEEDS_HUMAN:
            pass

    # ================================================================
    # Phase: MENU_NAV (click through menus to start a game)
    # ================================================================

    def _do_menu_nav(self):
        """Navigate game menus: Play Demo -> Standard -> dismiss intro."""
        import time
        import template_match as _tm

        print(f"\n  [auto] Phase: MENU_NAV")

        steps = [
            ("menu_play_demo", 1.5),
            ("mode_standard", 2.5),
            ("btn_close_dialog", 1.0),
        ]

        for template, wait in steps:
            for attempt in range(3):
                match = _tm.safe_click(template)
                if match:
                    time.sleep(wait)
                    break
                else:
                    if attempt < 2:
                        print(f"  [auto] '{template}' not found, retry {attempt + 2}/3...")
                        time.sleep(0.5)
                    else:
                        self._pause(f"Menu navigation failed: '{template}' not found after 3 tries")
                        return

        # Wait for game to load, then read deck
        print(f"  [auto] Menu navigation complete, waiting for game to load...")
        time.sleep(1.0)
        self.phase = GamePhase.DECK_READ

    # ================================================================
    # Phase: DECK_READ (read deck from memory)
    # ================================================================

    def _do_deck_read(self):
        """Read deck composition from memory reader."""
        import time
        from memory_reader import MemoryReader, restart_monitor

        print(f"\n  [auto] Phase: DECK_READ")

        # Ensure fresh monitor for new game
        self.monitor = restart_monitor()

        # Wait for game_connected (board becomes readable)
        if self.monitor and self.monitor.is_healthy():
            def _board_ready(board):
                return board is not None and len(board) > 0
            ready = self.monitor.wait_for(_board_ready, timeout=10, min_delay=1.0)
            if not ready:
                self._pause("Deck read: board not ready after 10s — game loaded?")
                return

        # Read deck from memory (100% accurate for role names)
        reader = MemoryReader()
        if not reader.open():
            self._pause("Deck read: cannot open game process")
            return

        deck = reader.read_deck()
        board = reader.read_board()
        reader.close()

        if not deck:
            self._pause("Deck read: memory reader returned no deck data")
            return

        if not board:
            self._pause("Deck read: memory reader returned no board data")
            return

        # Store deck data for SESSION_INIT
        self._deck_data = deck
        self._board_data = board
        self.phase = GamePhase.SESSION_INIT

    # ================================================================
    # Phase: SESSION_INIT (create session from memory data)
    # ================================================================

    def _do_session_init(self):
        """Create a fresh GameSession from memory-read deck and board data."""
        import time
        import template_match as _tm
        from game_loop import GameSession, DecisionLog

        print(f"\n  [auto] Phase: SESSION_INIT")

        deck = self._deck_data
        board = self._board_data

        villagers = deck.get('Villager', [])
        outcasts = deck.get('Outcast', [])
        minions = deck.get('Minion', [])
        demons = deck.get('Demon', [])

        n_cards = len(board)
        n_evil = len(minions) + len(demons)
        n_good = n_cards - n_evil

        print(f"  [auto] Board: {n_cards} cards, {n_evil} evil")
        print(f"  [auto] Deck: V={villagers} O={outcasts} M={minions} D={demons}")

        # Create fresh session
        session = GameSession(n_cards, n_evil)
        session.set_deck(villagers, outcasts, minions, demons)
        session.hp = 10
        session.wrong_exec_cost = 5  # Asc3+ default

        # Derive board counts
        pool_good = len(villagers) + len(outcasts)
        if pool_good == n_good:
            # Pool matches board — exact counts
            session.board_villager_count = len(villagers)
            session.board_outcast_count = len(outcasts)
            print(f"  [auto] Board counts exact: nv={session.board_villager_count} no={session.board_outcast_count}")
        elif pool_good > n_good:
            # Pool > board (Asc10+) — need header counts
            # For now, NEEDS_HUMAN to enter nv/no from screenshot
            # Baa obscures one existing Outcast only in the deck-pool view.
            has_baa = any(d == "Baa" for d in demons)
            print(f"  [auto] Pool ({pool_good}) > board good ({n_good}) — need header counts")
            if has_baa:
                print("  [auto] NOTE: Baa hides one existing Outcast in deck view; HUD no= is unchanged")
            self._pause(
                f"Enter board counts: set_hp 10 5, then deck ... nv=N no=N. "
                f"Pool has {len(villagers)}V + {len(outcasts)}O for {n_good} good slots."
            )
            # Still save partial session so user can complete it
            session.save()
            self.session = session
            DecisionLog.start_game(n_cards, n_evil, 10, 5)
            DecisionLog.log_deck(villagers, outcasts, minions, demons)
            return

        session.save()
        self.session = session
        self._last_card_count = 0
        self._last_exec_count = 0
        self._last_ability_count = 0

        DecisionLog.start_game(n_cards, n_evil, 10, 5)
        DecisionLog.log_deck(villagers, outcasts, minions, demons)

        # Close deck panel
        _tm.safe_click("icon_deck_purple")
        time.sleep(0.5)

        print(f"  [auto] Session initialized: {n_cards} cards, {n_evil} evil, HP=10")
        self.phase = GamePhase.FLIPPING

    # ================================================================
    # Phase: GAME_OVER (detect win/loss, record result)
    # ================================================================

    def _do_game_over(self):
        """Handle game over — record result and transition to POST_GAME."""
        print(f"\n  [auto] Phase: GAME_OVER")

        terminal_loss_role = getattr(self.session, "terminal_loss_role", None)
        if terminal_loss_role == "Bombardier":
            self._game_result = "loss"
            print(
                "  [auto] GAME LOST — "
                f"{terminal_loss_role} died outside Night "
                f"(HP after native resource handling: {self.session.hp})"
            )
        elif self.session.hp <= 0:
            self._game_result = "loss"
            print(f"  [auto] GAME LOST — HP depleted ({self.session.hp})")
        elif self._game_result == "loss":
            # SOLVING can establish a terminal loss from exact scenario/public
            # death evidence before a legacy session has a persisted marker.
            # Never let the final phase overwrite that known loss with a win.
            print(
                "  [auto] GAME LOST — terminal loss was established "
                f"during planning (recorded HP: {self.session.hp})"
            )
        else:
            self._game_result = "win"
            remaining_evil = self.session.n_evil - len([
                p for p in self.session.confirmed_evil if p in self.session.executed
            ])
            print(f"  [auto] GAME WON — HP={self.session.hp}, remaining evil={remaining_evil}")

        self.session.save()
        self.phase = GamePhase.POST_GAME

    # ================================================================
    # Phase: FLIPPING (initial bulk flip)
    # ================================================================

    def _do_flipping(self):
        """Flip all unrevealed cards."""
        from game_loop import dispatch
        print(f"\n  [auto] Phase: FLIPPING")

        if (
            self.session.has_lilis_night_rule()
            and self.session.has_role_in_deck("Shaman")
        ):
            self._pause(
                "Lilis+Shaman reveal automation is unsafe before any click: "
                "current Lilis actor count can be 0, 1, or 2"
            )
            return
        if self.session.has_duplicate_lilis():
            self._pause(
                "Duplicate Lilis live automation is unsupported: multiple "
                "actors can charge HP and collide on one delayed victim"
            )
            return

        if self.session.pending_lilis_nights > 0:
            self.phase = GamePhase.LILIS_NIGHT
            return

        if self.session.has_lilis_night_rule():
            before_pending_nights = self.session.pending_lilis_nights
            dispatch("flip", ["--lilis"], self.session)
            # Only this call's four verified reveals can open a fresh night.
            if self.session.pending_lilis_nights > before_pending_nights:
                self.phase = GamePhase.LILIS_NIGHT
                return
            remaining = self._unrevealed_positions()
            if remaining:
                return  # stay in FLIPPING for next batch
        else:
            dispatch("flip", [], self.session)

        self.phase = GamePhase.ENTERING_CLUES

    # ================================================================
    # Phase: ENTERING_CLUES
    # ================================================================

    def _do_entering_clues(self):
        """Auto-enter clues from memory reader."""
        from game_loop import dispatch
        print(f"\n  [auto] Phase: ENTERING CLUES")

        dispatch("auto_card", [], self.session)

        # Check what still needs manual entry
        entered = {c.position for c in self.session.cards}
        flipped = set(self.session.reveal_order)
        dead = set(self.session.executed) | set(self.session.night_kills)
        blocked = self._active_blocked_positions()
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

    # ================================================================
    # Phase: SOLVING
    # ================================================================

    def _do_solving(self):
        """Run solver and decide next action."""
        from strategy import (
            _compute_confidence,
            _has_terminal_role_loss,
            evil_probabilities,
            ordinary_execution_bombardier_positions,
            recommend_action,
        )
        print(f"\n  [auto] Phase: SOLVING")

        state = self.session.to_game_state()
        result = self.session._solve(state)

        for line in result.reasoning:
            print(f"  {line}")

        if self.session.twin_live_solver_unsafe():
            self._pause(
                "Live Twin Minion solving is paused until the ordered "
                "current-data trace is modeled; no action was produced"
            )
            return

        probs = evil_probabilities(state, result)
        action = recommend_action(state, result, self.session.used_abilities)
        action.confidence = _compute_confidence(action, state, result, probs)
        ordinary_bombardiers = ordinary_execution_bombardier_positions(
            state, result,
        )

        print(f"\n  [auto] Recommendation: {action.action_type.upper()}", end="")
        if action.position:
            print(f" #{action.position}", end="")
        if action.ability_name:
            print(f" ({action.ability_name})", end="")
        if action.targets:
            print(f" -> {['#'+str(t) for t in action.targets]}", end="")
        print(f" (confidence: {action.confidence:.0%}, {result.n_surviving} scenarios)")
        print(f"  Reason: {action.reasoning}")
        for w in action.warnings:
            print(f"  WARNING: {w}")

        # Decision tree
        if action.action_type == "loss":
            # A legacy/unmarked state can still prove the native Bombardier
            # terminal from public death evidence or every surviving exact
            # world. Persist that fact before GAME_OVER so a later reload and
            # final-result classification cannot reinterpret it as a win.
            if _has_terminal_role_loss(state, result):
                self.session.terminal_loss_role = "Bombardier"
            self._game_result = "loss"
            self.phase = GamePhase.GAME_OVER
            self.session.save()
            print(f"\n  [auto] TERMINAL LOSS - {action.reasoning}")

        elif action.action_type == "win":
            self.phase = GamePhase.GAME_OVER
            print(f"\n  [auto] ALL EVIL EXECUTED - GAME WON!")

        elif action.action_type == "error":
            self._pause(f"Solver error: {action.reasoning}")

        elif action.action_type == "execute":
            is_definite = action.position in result.definite_evil
            is_bombardier = action.position in ordinary_bombardiers
            is_forced_safe = getattr(action, 'forced_safe', False)

            if is_bombardier:
                self._pause(
                    f"Refusing execution of possible current-role "
                    f"Bombardier #{action.position}"
                )
            elif is_definite or is_forced_safe:
                self._pending_exec = (action.position, result, is_forced_safe)
                self.phase = GamePhase.EXECUTING
            else:
                self._pause(f"Execute #{action.position}? ({action.confidence:.0%} confident) — not definite evil, needs manual decision")

        elif action.action_type == "use_ability":
            ability_key = (action.ability_name or "").lower().replace(" ", "_")
            risky_slayer_targets = sorted(
                set(action.targets or []).intersection(
                    result.bombardier_positions
                )
            )
            if ability_key == "slayer" and risky_slayer_targets:
                self._pause(
                    "Refusing Slayer into possible moved Bombardier data at "
                    f"{['#' + str(pos) for pos in risky_slayer_targets]}"
                )
                return
            self._pending_ability = (action.position, action.targets, action.ability_name, result)
            self.phase = GamePhase.ABILITY_USE

        elif action.action_type == "reveal":
            self._pending_reveal = (action.position,)
            self.phase = GamePhase.REVEALING

        else:
            self._pause(f"Unknown action: {action.action_type}")

    # ================================================================
    # Phase: EXECUTING
    # ================================================================

    def _do_executing(self):
        """Auto-execute a definite evil target."""
        pos, result, forced_safe = self._pending_exec
        print(f"\n  [auto] Phase: EXECUTING #{pos}")

        exec_result = self.session.auto_execute(pos, result, monitor=self.monitor, forced_safe=forced_safe)

        if exec_result["success"]:
            terminal_loss_role = getattr(
                self.session, "terminal_loss_role", None
            )
            if terminal_loss_role == "Bombardier":
                self._game_result = "loss"
                self.phase = GamePhase.GAME_OVER
                self.session.save()
                print(
                    f"\n  [auto] GAME LOST - {terminal_loss_role} "
                    "died outside Night "
                    f"(HP after native resource handling: {self.session.hp})"
                )
                return
            if exec_result.get("blocked"):
                print(f"  [auto] Execution blocked on #{pos}: confirmed Knight immunity")
            elif exec_result["was_evil"]:
                print(f"  [auto] Executed #{pos}: {exec_result['evil_role']} (EVIL)")
            else:
                print(f"  [auto] WRONG EXECUTION on #{pos}! HP now {self.session.hp}")
                if self.session.hp <= 0:
                    self.phase = GamePhase.GAME_OVER
                    print(f"\n  [auto] HP DEPLETED - GAME LOST!")
                    return
                if forced_safe:
                    # Lookahead planned for this — wrong exec is expected, continue
                    print(f"  [auto] Forced-safe execution: wrong exec was anticipated by lookahead, continuing")
                else:
                    self._pause(f"Wrong exec on definite evil #{pos} — possible solver bug!")
                    return
        else:
            self._pause(f"Execution failed: {exec_result['error']}")
            return

        self._snapshot_counts()
        self.phase = GamePhase.SOLVING

    # ================================================================
    # Phase: REVEALING (new — auto-reveal a single card)
    # ================================================================

    def _do_revealing(self):
        """Auto-reveal a single unrevealed card."""
        import time
        from game_utils import all_game_card_coords
        import template_match as _tm
        from game_loop import (
            DecisionLog,
            _apply_flip_verification,
            _parse_clue_from_memory,
            _verify_flips,
        )

        pos = self._pending_reveal[0]
        print(f"\n  [auto] Phase: REVEALING #{pos}")

        if (
            self.session.has_lilis_night_rule()
            and self.session.has_role_in_deck("Shaman")
        ):
            self._pause(
                "Lilis+Shaman reveal automation is unsafe before any click: "
                "current Lilis actor count can be 0, 1, or 2"
            )
            return
        if self.session.has_duplicate_lilis():
            self._pause(
                "Duplicate Lilis live automation is unsupported: refusing a "
                "reveal that may start an unmodeled multi-actor Night"
            )
            return
        if self.session.pending_lilis_nights > 0:
            self.phase = GamePhase.LILIS_NIGHT
            return

        # Click safety protocol
        entered = {c.position for c in self.session.cards}
        was_revealed = pos in self.session.reveal_order
        dead = set(self.session.executed) | set(self.session.night_kills)
        blocked = set(self.session.blocked_positions)

        if pos in entered:
            print(f"  [auto] #{pos} already has card entry — skipping reveal")
            self.phase = GamePhase.SOLVING
            return

        if pos in dead:
            print(f"  [auto] #{pos} is dead — skipping reveal")
            self.phase = GamePhase.SOLVING
            return

        if pos in blocked:
            print(
                f"  [auto] Re-probing previously blocked #{pos}; memory "
                "verification will either clear or restore the marker."
            )

        # Check for unused active ability at this position
        if self._has_active_ability(pos):
            self._pause(f"#{pos} has unused active ability — clicking would activate it, not reveal")
            return

        coords = all_game_card_coords(self.session.n_cards)
        if pos not in coords:
            self._pause(f"Position #{pos} not valid for {self.session.n_cards}-card game")
            return

        x, y = coords[pos]

        # Click the card (use safe_click_at for first card, fast_click_at otherwise)
        if pos == 1:
            _tm.safe_click_at(x, y, f"reveal_card{pos}")
        else:
            _tm.fast_click_at(x, y, f"reveal_card{pos}")

        # Wait for flip via memory reader
        if self.monitor and self.monitor.is_healthy():
            def _card_flipped(board):
                if not board:
                    return False
                card = next((c for c in board if c['position'] == pos), None)
                return card and card['state'] in ('Alive', 'Revealed')
            flipped = self.monitor.wait_for(_card_flipped, timeout=3, min_delay=0.3)
            if not flipped:
                # Retry once
                print(f"  [auto] Card #{pos} didn't flip — retrying...")
                _tm.safe_click_at(x, y, f"reveal_card{pos}_retry")
                flipped = self.monitor.wait_for(_card_flipped, timeout=3, min_delay=0.3)
            board = self.monitor.get_board()
            if not board:
                self._pause(
                    f"Card #{pos} click could not be verified from memory; "
                    "session state was not changed"
                )
                return
            verify = _verify_flips(board, [pos], self.session)
            verification_changed = _apply_flip_verification(
                self.session,
                [pos],
                verify,
                persist=False,
            )
            night_triggered = (
                pos in verify["flipped"]
                and not was_revealed
                and self.session.has_lilis_night_rule()
                and len(self.session.reveal_order) % 4 == 0
            )
            if night_triggered:
                self.session.schedule_lilis_night()
            if verification_changed or night_triggered:
                # Persist the fourth reveal and pending native transition as
                # one state, before parsing or later automation can fail.
                self.session.save()
            if pos in verify["blocked"]:
                print(f"  [auto] #{pos} is blocked by the active Witch quota")
                self._snapshot_counts()
                self.phase = GamePhase.SOLVING
                return
            if pos in verify["failed"] or pos not in verify["flipped"]:
                self._pause(
                    f"Card #{pos} failed verified reveal after retry — game focused?"
                )
                return
        else:
            self._pause(
                f"No healthy memory monitor to verify reveal #{pos}; "
                "session state was not changed"
            )
            return

        # NightModeRule survives Lilis death. Stop at every fourth verified
        # reveal even when the dead actor will contribute no victim or damage.
        if night_triggered:
            total_reveals = len(self.session.reveal_order)
            print(f"  [auto] Lilis night triggered (reveal #{total_reveals})")
            # Let the delayed native kill (or protected/no-victim timeout)
            # settle before reading the result surface.
            # Parse clue first before night phase
            if not self._auto_enter_single_card(pos):
                self._pause(
                    f"Public clue for revealed #{pos} did not settle in memory; "
                    "enter it manually before resolving the pending Night"
                )
                return
            self.phase = GamePhase.LILIS_NIGHT
            self.session.save()
            return

        # Parse clue from memory
        if not self._auto_enter_single_card(pos):
            self._pause(
                f"Public clue for revealed #{pos} did not settle in memory; "
                "enter it manually, then resume"
            )
            return
        self.session.save()

        self._snapshot_counts()
        self.phase = GamePhase.SOLVING

    def _auto_enter_single_card(self, pos: int):
        """Wait for one coherent public clue surface, then enter it."""
        from game_loop import (
            DecisionLog,
            _active_cycle_is_spent,
            _card_current_jester_no_info,
            _has_active_clue_result,
            _parse_clue_from_memory,
            _pickable_uses_remaining,
            card_no_info,
        )

        if not self.monitor or not self.monitor.is_healthy():
            print(f"  [auto] No monitor — manual entry needed for #{pos}")
            return False

        captured = {"parsed": None, "card": None}

        def _parse_from_board(board):
            if not board:
                return None
            mc = next(
                (card for card in board if card.get('position') == pos),
                None,
            )
            if not mc:
                return None
            captured["card"] = mc
            return _parse_clue_from_memory(
                mc,
                n_cards=self.session.n_cards,
                baker_rule_version=self.session.baker_rule_version,
                fortune_teller_rule_version=self.session.fortune_teller_rule_version,
            )

        parsed = _parse_from_board(self.monitor.get_board())
        if parsed is None:
            def _clue_settled(board):
                candidate = _parse_from_board(board)
                if candidate is None:
                    return False
                captured["parsed"] = candidate
                return True

            settled = self.monitor.wait_for(
                _clue_settled,
                timeout=2.5,
                min_delay=0.15,
            )
            if settled:
                parsed = captured["parsed"]
                if parsed is None:
                    # Some lightweight monitor adapters report readiness
                    # without invoking the predicate; verify their latest
                    # snapshot once before treating it as a timeout.
                    parsed = _parse_from_board(self.monitor.get_board())

        if parsed:
            if _has_active_clue_result(parsed):
                remaining = _pickable_uses_remaining(captured["card"])
                if remaining is None:
                    return False
                if remaining > 0:
                    existing = next(
                        (
                            card for card in self.session.cards
                            if card.position == parsed.position
                        ),
                        None,
                    )
                    if existing is not None:
                        return True
                    parsed = (
                        _card_current_jester_no_info(parsed.position)
                        if (
                            parsed.apparent_role.casefold() == "jester"
                            and parsed.info_parsed.get("jester_variant")
                            == "public_current"
                        )
                        else card_no_info(
                            parsed.position,
                            parsed.apparent_role,
                        )
                    )
            self.session.add_card(parsed, mark_active_result=False)
            if (
                _active_cycle_is_spent(captured["card"])
                and _has_active_clue_result(parsed)
            ):
                self.session.mark_ability_used(parsed.position)
            DecisionLog.log_card(parsed)
            print(f"  [auto] #{parsed.position} {parsed.apparent_role}: {parsed.info_parsed}")
            return True
        else:
            mc = captured["card"] or {}
            role = (
                mc.get('disguise')
                or mc.get('current_role')
                or mc.get('true_role', '?')
            )
            clue = mc.get('clue_text', '')
            print(f"  [auto] #{pos} {role}: couldn't parse clue \"{clue}\" — needs manual entry")
            return False

    # ================================================================
    # Phase: ABILITY_USE (new — auto-use active abilities)
    # ================================================================

    def _do_ability_use(self):
        """Auto-use an active ability: click card, click targets, read result."""
        import time
        from game_utils import all_game_card_coords
        import template_match as _tm
        from game_loop import (
            DecisionLog,
            _active_result_refs_match_clicks,
            _observed_active_role_key,
            _parse_clue_from_memory,
        )

        pos, targets, ability_name, result = self._pending_ability
        print(f"\n  [auto] Phase: ABILITY_USE #{pos} ({ability_name}) -> targets {targets}")

        # Slayer's kill/death result still needs its dedicated manual path.
        if ability_name == "Slayer":
            self._pause(
                f"Use {ability_name} on #{pos} -> targets {targets}. "
                f"Complex ability — enter result manually, then 'resume'."
            )
            return

        # Reuse the strict session path for Plague Doctor and the resettable
        # Druid/Judge/Fortune Teller abilities so autonomous play gets the same
        # native target checks, event-freshness boundary, and recovery
        # semantics as `next`/`auto_next`.
        if ability_name in (
            "Plague Doctor",
            "Plague_Doctor",
            "Druid",
            "Jester",
            "Judge",
            "Fortune Teller",
        ):
            from strategy import Action

            strict_action = Action(
                action_type="use_ability",
                position=pos,
                targets=list(targets or []),
                ability_name=ability_name.replace("_", " "),
            )
            exec_result = self.session.auto_use_ability(
                strict_action,
                monitor=self.monitor,
            )
            if not exec_result["success"]:
                display_name = ability_name.replace("_", " ")
                if display_name == "Plague Doctor":
                    recovery = (
                        "Read the public speech bubble and enter it with "
                        "pd_check, then 'resume'."
                    )
                elif display_name == "Fortune Teller":
                    recovery = (
                        "Read the public speech bubble and enter it with "
                        "card fortune_teller, then 'resume'."
                    )
                elif display_name == "Druid":
                    recovery = (
                        "For a normal result use card druid <actor> "
                        "<a,b,c> <Outcast|none> only to preserve the visible "
                        "scalar surface; it cannot resume ResetAfterNight "
                        "history. A Druid interruption cannot be entered "
                        "manually—recover authenticated acted-info memory or "
                        "restart."
                    )
                elif display_name == "Jester":
                    recovery = (
                        "Recover the authenticated acted-info history with "
                        "auto_card. A scalar manual result cannot safely resume "
                        "Jester's ResetAfterNight callback ledger."
                    )
                else:
                    recovery = (
                        "Read the public speech bubble, enter it manually, "
                        "then 'resume'."
                    )
                self._pause(
                    f"{display_name} ability on #{pos} could not be recorded: "
                    f"{exec_result['error']}. {recovery}"
                )
                return

            self._snapshot_counts()
            self.phase = GamePhase.SOLVING
            return

        if (
            ability_name == "Dreamer"
            and (
                len(targets or []) != 2
                or any(type(target) is not int for target in (targets or []))
                or len(set(targets or [])) != 2
            )
        ):
            self._pause(
                f"Dreamer requires exactly 2 distinct integer targets; "
                f"solver returned {targets}. "
                f"Handle manually, then 'resume'."
            )
            return
        if (
            ability_name == "Jester"
            and (
                len(targets or []) != 3
                or any(type(target) is not int for target in (targets or []))
                or len(set(targets or [])) != 3
            )
        ):
            self._pause(
                f"Jester requires exactly 3 distinct integer targets; "
                f"solver returned {targets}. Handle manually, then 'resume'."
            )
            return
        if ability_name not in ("Dreamer", "Jester"):
            self._pause(
                f"Ability {ability_name} has no authenticated autonomous "
                "result path; handle it manually, then 'resume'."
            )
            return

        coords = all_game_card_coords(self.session.n_cards)
        if pos not in coords:
            self._pause(f"Position #{pos} not valid for {self.session.n_cards}-card game")
            return
        invalid_targets = [target for target in (targets or []) if target not in coords]
        if invalid_targets:
            self._pause(
                f"Targets {invalid_targets} are not valid for "
                f"{self.session.n_cards}-card game"
            )
            return

        # Dreamer and Jester share this generic transition. Snapshot the full
        # native callback history before the actor click; persistent savedAct,
        # act flags, and prior actedInfos are otherwise indistinguishable from
        # a result produced by this activation.
        baseline_history = None
        baseline_remaining = None
        if not self.monitor or not self.monitor.is_healthy():
            self._pause(
                f"Ability {ability_name} on #{pos} cannot be activated "
                "safely without a readable pre-click acted-info history."
            )
            return
        baseline_board = self.monitor.get_board()
        baseline_card = next(
            (
                card for card in (baseline_board or [])
                if card.get("position") == pos
            ),
            None,
        )
        baseline_history = _acted_history_baseline(baseline_card)
        if baseline_card is None or baseline_history is None:
            self._pause(
                f"Ability {ability_name} on #{pos} has no readable "
                "pre-click acted-info history."
            )
            return
        expected_role_key = ability_name.casefold().replace(" ", "_")
        observed_role_key = _observed_active_role_key(baseline_card)
        if observed_role_key != expected_role_key:
            self._pause(
                f"Ability {ability_name} on #{pos} cannot be activated: "
                f"pre-click memory shows {observed_role_key or 'no role'}."
            )
            return
        baseline_remaining = baseline_card.get("pickable_uses_remaining")
        if type(baseline_remaining) is not int:
            self._pause(
                f"Ability {ability_name} on #{pos} has an unreadable native "
                "pickable-use budget."
            )
            return
        if baseline_remaining <= 0:
            self._pause(
                f"Ability {ability_name} on #{pos} is not currently "
                f"available (remaining budget {baseline_remaining})."
            )
            return

        # Step 1: Click the ability card to activate
        ax, ay = coords[pos]
        _tm.safe_click_at(ax, ay, f"ability_{ability_name}_{pos}")
        time.sleep(0.5)

        # Step 2: Click each target
        for t in (targets or []):
            tx, ty = coords[t]
            _tm.fast_click_at(tx, ty, f"ability_target_{t}")
            time.sleep(0.3)

        # Step 3: Wait for a genuinely appended native result event.
        resolved_card = {"card": None}
        counter_decrease = {"seen": False}
        if self.monitor and self.monitor.is_healthy():
            def _ability_resolved(board):
                if not board:
                    return False
                card = next((c for c in board if c['position'] == pos), None)
                if not card:
                    return False
                if _observed_active_role_key(card) != expected_role_key:
                    return False
                remaining = card.get("pickable_uses_remaining")
                if (
                    type(remaining) is int
                    and remaining < baseline_remaining
                ):
                    # Native exposes a remaining-use counter. Its decrease
                    # corroborates a click, but never substitutes for a new
                    # coherent callback event.
                    counter_decrease["seen"] = True
                if _coherent_appended_acted_event(
                    card,
                    baseline_history,
                ) is None:
                    return False
                if not _active_result_refs_match_clicks(
                    card,
                    list(targets or []),
                    n_cards=self.session.n_cards,
                ):
                    return False
                resolved_card["card"] = copy.deepcopy(card)
                return True

            resolved = self.monitor.wait_for(_ability_resolved, timeout=5, min_delay=0.5)
            if resolved and resolved_card["card"] is None:
                # Lightweight monitor adapters may report readiness without
                # invoking the predicate. Authenticate their latest snapshot.
                resolved = _ability_resolved(self.monitor.get_board())
            if not resolved:
                detail = (
                    " The remaining-use counter decreased, but no coherent "
                    "new acted-info event was appended."
                    if counter_decrease["seen"] else ""
                )
                self._pause(
                    f"Ability {ability_name} on #{pos} didn't resolve — "
                    f"check if clicks landed correctly.{detail}"
                )
                return
        else:
            self._pause(
                f"Ability {ability_name} on #{pos} lost its readable "
                "memory monitor before a fresh acted-info event was verified."
            )
            return

        # Step 4: Parse result from the exact snapshot authenticated above.
        mc = resolved_card["card"]
        if mc:
            parsed = _parse_clue_from_memory(
                mc,
                n_cards=self.session.n_cards,
                baker_rule_version=self.session.baker_rule_version,
                fortune_teller_rule_version=self.session.fortune_teller_rule_version,
            )
            if parsed:
                parsed_role_key = (
                    parsed.apparent_role.casefold().replace(" ", "_")
                )
                if parsed.position != pos or parsed_role_key != expected_role_key:
                    self._pause(
                        f"Ability {ability_name} on #{pos}: authenticated "
                        "memory parsed as a different actor."
                    )
                    return
                try:
                    self.session.add_card(parsed, mark_active_result=False)
                except ValueError as exc:
                    self._pause(
                        f"Ability {ability_name} on #{pos}: validated result "
                        f"could not be stored: {exc}"
                    )
                    return
                self.session.mark_ability_used(pos)
                DecisionLog.log_card(parsed)
                print(f"  [auto] Ability result: #{parsed.position} {parsed.apparent_role}: {parsed.info_parsed}")
            else:
                clue = mc.get('clue_text', '')
                self._pause(
                    f"Ability {ability_name} on #{pos}: couldn't parse result "
                    f"\"{clue}\" — enter manually, then 'resume'."
                )
                return
        else:
            self._pause(
                f"Ability {ability_name} on #{pos}: verified result "
                "snapshot disappeared before parsing."
            )
            return

        self.session.save()
        self._snapshot_counts()
        self.phase = GamePhase.SOLVING

    # ================================================================
    # Phase: LILIS_NIGHT (triggers transition to NIGHT_RESOLVE)
    # ================================================================

    def _do_lilis_night(self):
        """Handle Lilis night phase — wait for animation, transition to resolve."""
        print(f"\n  [auto] Phase: LILIS_NIGHT")

        if self.session.has_role_in_deck("Shaman"):
            self._pause(
                "Lilis+Shaman Night automation is unsupported: exact current "
                "actor multiplicity and 0/2/4 HP outcome require manual recovery"
            )
            return
        if self.session.has_duplicate_lilis():
            self._pause(
                "Duplicate Lilis Night cannot be resolved safely: multiple "
                "actors and delayed-victim collisions are not represented"
            )
            return
        if self.session.pending_lilis_nights <= 0:
            self._pause(
                "No explicitly persisted Lilis Night is pending; legacy "
                "batch/resolution counters are not safe to infer"
            )
            return

        if self.monitor and self.monitor.is_healthy():
            already_dead = set(self.session.night_kills) | set(self.session.executed)

            def _night_kill_detected(board):
                if not board:
                    return False
                return any(
                    c.get('killed_hidden') for c in board
                    if c['position'] not in already_dead
                )

            resolved = self.monitor.wait_for(_night_kill_detected, timeout=8, min_delay=2.0)
            if resolved:
                self.phase = GamePhase.NIGHT_RESOLVE
                return
            else:
                if self.session.is_lilis_alive():
                    # No victim and a protected Knight victim share this surface.
                    print("  [auto] No kill detected (timeout) — resolving zero/protected victim")
                else:
                    print(
                        "  [auto] Post-death Night animation settled — "
                        "resolving the zero-damage rule transition"
                    )
                self.phase = GamePhase.NIGHT_RESOLVE
                return
        else:
            self._pause("Lilis night — no monitor, verify kills manually: night_kill <pos> <n_evil> OR night_no_kill")

    # ================================================================
    # Phase: NIGHT_RESOLVE (new — auto-detect and record night kills)
    # ================================================================

    def _do_night_resolve(self):
        """Auto-detect killed positions from memory, deduct HP, record kills."""
        print(f"\n  [auto] Phase: NIGHT_RESOLVE")

        if self.session.has_role_in_deck("Shaman"):
            self._pause(
                "Lilis+Shaman Night resolution is unsupported: refusing to "
                "infer actor multiplicity or HP from hidden deaths"
            )
            return
        if self.session.has_duplicate_lilis():
            self._pause(
                "Duplicate Lilis Night cannot be resolved safely: multiple "
                "actors and delayed-victim collisions are not represented"
            )
            return

        if not self.monitor or not self.monitor.is_healthy():
            self._pause("Night resolve: no monitor — enter manually: night_kill <pos> <n_evil> OR night_no_kill")
            return

        board = self.monitor.get_board()
        if not board:
            self._pause("Night resolve: no board data — enter manually")
            return

        # Find newly killed positions (killed_hidden flag set, not already in our records)
        already_dead = set(self.session.night_kills) | set(self.session.executed)
        killed = []
        for c in board:
            pos = c['position']
            if pos in already_dead:
                continue
            if c.get('killed_hidden'):
                killed.append(pos)

        actor_active = self.session.is_lilis_alive()
        if killed and not actor_active:
            self._pause(
                "A new hidden death appeared after Lilis was known dead; "
                "refusing to classify it as a Lilis result"
            )
            return
        if killed:
            positions = ",".join(str(position) for position in killed)
            self._pause(
                "Night victim position(s) are public, but hidden memory "
                "alignment is validation-only. Enter the public aggregate "
                f"with: night_kill {positions} <n_evil>"
            )
            return
        elif actor_active:
            print("  [auto] No kills detected — zero-victim/protected-victim Lilis night")
            print(
                "  [auto] No identity inference: a clean Knight or HealthyBluff "
                "Doppelganger-as-Knight can survive without a reroll"
            )
        else:
            print(
                "  [auto] Lilis is known dead — synchronizing the persistent "
                "Night rule with zero actor damage"
            )

        try:
            if actor_active:
                result = self.session.record_lilis_night_result([], 0)
            else:
                result = self.session.record_lilis_post_death_night()
        except ValueError as exc:
            self._pause(f"Lilis night result rejected without mutation: {exc}")
            return

        print(
            f"  [auto] Resolved {result['resolved_events']} Lilis night(s); "
            f"HP: {result['old_hp']} -> {result['new_hp']}"
        )

        if self.session.hp <= 0:
            self.phase = GamePhase.GAME_OVER
            print(f"\n  [auto] HP DEPLETED - GAME LOST!")
            self.session.save()
            return

        reset_abilities = result["reset_abilities"]
        if reset_abilities:
            print(
                "  [auto] ResetAfterNight abilities ready again: "
                f"{['#' + str(position) for position in reset_abilities]}"
            )

        self.session.save()
        self._snapshot_counts()

        # Check if there are more cards to flip (Lilis batches)
        remaining = self._unrevealed_positions()
        if remaining and self.session.has_lilis_night_rule():
            self.phase = GamePhase.FLIPPING
        else:
            self.phase = GamePhase.ENTERING_CLUES

    # ================================================================
    # Helpers
    # ================================================================

    def _has_active_ability(self, pos: int) -> bool:
        """Check if a position has an unused active ability."""
        if pos in self.session.used_abilities:
            return False
        from knowledge_base import get_card
        target_card = next((c for c in self.session.cards if c.position == pos), None)
        if target_card:
            kb_card = get_card(target_card.apparent_role)
            if kb_card and kb_card.activated_ability:
                return True
        return False

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

    def _active_blocked_positions(self):
        """Current markers, ignoring the ordinary quota after any Witch death."""
        if (
            hasattr(self.session, "is_witch_known_dead")
            and self.session.is_witch_known_dead()
        ):
            return set()
        return set(self.session.blocked_positions)

    def _unrevealed_positions(self):
        """Get positions that still need flipping."""
        all_positions = set(range(1, self.session.n_cards + 1))
        done = (
            set(self.session.reveal_order)
            | set(self.session.night_kills)
            | set(self.session.executed)
            | self._active_blocked_positions()
        )
        return sorted(all_positions - done)


# ====================================================================
# BatchGameRunner — plays N games with per-game isolation
# ====================================================================

class BatchGameRunner:
    """Run multiple games autonomously with per-game isolation.

    Usage:
        runner = BatchGameRunner(n_games=10, risk="conservative")
        runner.run()

    Per-game isolation protocol:
        1. session.full_reset()
        2. Kill Rust solver daemon
        3. Restart memory monitor
        4. Cleanup old screenshots
        5. Wait for game_connected event
    """

    def __init__(self, n_games: int = 1, risk: str = "conservative"):
        self.n_games = n_games
        self.risk = risk
        self.results: list[dict] = []
        self.consecutive_failures = 0
        self._MAX_CONSECUTIVE_FAILURES = 3

    def run(self):
        """Run the batch. Returns list of game result dicts."""
        import time
        import os

        print(f"\n{'='*60}")
        print(f"  BATCH RUN: {self.n_games} game(s), risk={self.risk}")
        print(f"{'='*60}")

        for i in range(self.n_games):
            print(f"\n{'='*60}")
            print(f"  GAME {i + 1}/{self.n_games}")
            print(f"{'='*60}")

            # Disk space check
            try:
                stat = os.statvfs(".") if hasattr(os, "statvfs") else None
                if stat and stat.f_bavail * stat.f_frsize < 100 * 1024 * 1024:
                    print("  ABORT: Less than 100MB free disk space")
                    break
            except (OSError, AttributeError):
                pass  # Windows doesn't have statvfs; skip check

            game_result = self._run_single_game(i + 1)
            self.results.append(game_result)

            if game_result["status"] == "error":
                self.consecutive_failures += 1
                print(f"  Game {i + 1} FAILED: {game_result['error']}")
                if self.consecutive_failures >= self._MAX_CONSECUTIVE_FAILURES:
                    print(f"\n  ABORT: {self.consecutive_failures} consecutive failures")
                    break
            else:
                self.consecutive_failures = 0
                print(f"  Game {i + 1}: {game_result['status'].upper()}, HP={game_result.get('hp', '?')}")

            # Cleanup between games
            from game_loop import cleanup_screenshots
            cleanup_screenshots(keep=20)

        self._print_summary()
        return self.results

    def _run_single_game(self, game_num: int) -> dict:
        """Play one game. Returns result dict."""
        import time
        from game_loop import GameSession

        t0 = time.time()
        result = {"game_num": game_num, "status": "error", "error": None, "hp": None}

        try:
            # Create state machine (session created by SESSION_INIT phase)
            sm = GameStateMachine(session=None, monitor=None, risk=self.risk)

            # Start from menu navigation
            sm.start_full_game()

            # Check outcome
            if sm.phase == GamePhase.POST_GAME:
                result["status"] = sm._game_result or "unknown"
                result["hp"] = sm.session.hp if sm.session else None
                result["n_cards"] = sm.session.n_cards if sm.session else None
            elif sm.phase == GamePhase.NEEDS_HUMAN:
                result["status"] = "needs_human"
                result["error"] = sm._needs_human_reason
            elif sm.phase == GamePhase.GAME_OVER:
                result["status"] = sm._game_result or "unknown"
                result["hp"] = sm.session.hp if sm.session else None
            else:
                result["status"] = "incomplete"
                result["error"] = f"Stopped in phase {sm.phase.value}"

        except Exception as e:
            result["error"] = str(e)
            import traceback
            traceback.print_exc()

        result["elapsed_s"] = round(time.time() - t0, 1)
        return result

    def _print_summary(self):
        """Print batch results summary."""
        wins = sum(1 for r in self.results if r["status"] == "win")
        losses = sum(1 for r in self.results if r["status"] == "loss")
        errors = sum(1 for r in self.results if r["status"] == "error")
        human = sum(1 for r in self.results if r["status"] == "needs_human")
        total = len(self.results)

        print(f"\n{'='*60}")
        print(f"  BATCH COMPLETE: {total} game(s)")
        print(f"  Wins: {wins}, Losses: {losses}, Errors: {errors}, Human: {human}")
        if wins + losses > 0:
            print(f"  Win rate: {wins/(wins+losses):.0%}")
        avg_time = sum(r.get("elapsed_s", 0) for r in self.results) / max(total, 1)
        print(f"  Avg time per game: {avg_time:.1f}s")
        print(f"{'='*60}")
