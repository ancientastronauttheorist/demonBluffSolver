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


class GameStateMachine:
    """Autonomous game loop driven by memory reader.

    Safety gates (always NEEDS_HUMAN):
    - Probabilistic execution (not definite_evil and not forced_safe)
    - HP <= wrong_exec_cost and multiple scenarios remain
    - 0 surviving scenarios (solver bug)
    - Active ability with complex multi-step interaction (Slayer, PD)
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
            # Check if Baa is in deck (adds fake outcast to header)
            has_baa = any(d == "Baa" for d in demons)
            print(f"  [auto] Pool ({pool_good}) > board good ({n_good}) — need header counts")
            if has_baa:
                print(f"  [auto] NOTE: Baa in deck — displayed outcast count includes 1 fake")
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

        if self.session.hp <= 0:
            self._game_result = "loss"
            print(f"  [auto] GAME LOST — HP depleted ({self.session.hp})")
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

        if self.session.is_lilis_alive():
            dispatch("flip", ["--lilis"], self.session)
            # Check if Lilis night triggered (batch was full 4 cards)
            if self.session.lilis_batch_index > 0:
                remaining = self._unrevealed_positions()
                if remaining:
                    # Night phase needed before next batch
                    self.phase = GamePhase.LILIS_NIGHT
                    return
                # No more to flip — might still need night resolve
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

    # ================================================================
    # Phase: SOLVING
    # ================================================================

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
        if action.ability_name:
            print(f" ({action.ability_name})", end="")
        if action.targets:
            print(f" -> {['#'+str(t) for t in action.targets]}", end="")
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
            is_definite = action.position in result.definite_evil
            is_bombardier = action.position in result.bombardier_positions
            is_forced_safe = getattr(action, 'forced_safe', False)

            if (is_definite and not is_bombardier) or is_forced_safe:
                self._pending_exec = (action.position, result, is_forced_safe)
                self.phase = GamePhase.EXECUTING
            else:
                self._pause(f"Execute #{action.position}? ({action.confidence:.0%} confident) — not definite evil, needs manual decision")

        elif action.action_type == "use_ability":
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
            if exec_result.get("blocked"):
                print(f"  [auto] Execution blocked on #{pos}: confirmed Knight immunity")
            elif exec_result["was_evil"]:
                print(f"  [auto] Executed #{pos}: {exec_result['evil_role']} (EVIL)")
                # Check if we just killed the Witch — unblock any blocked position
                if exec_result['evil_role'] and 'Witch' in exec_result['evil_role']:
                    if self.session.blocked_positions:
                        blocked_pos = self.session.blocked_positions[0]
                        print(f"  [auto] Witch killed! Unblocking #{blocked_pos}")
                        # Will be handled by next solve -> reveal recommendation
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
        from game_loop import _parse_clue_from_memory, DecisionLog

        pos = self._pending_reveal[0]
        print(f"\n  [auto] Phase: REVEALING #{pos}")

        # Click safety protocol
        entered = {c.position for c in self.session.cards}
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
            self._pause(f"#{pos} is Witch-blocked — execute Witch first")
            return

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

        # Record reveal order
        if pos not in self.session.reveal_order:
            self.session.reveal_order.append(pos)

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
                if not flipped:
                    self._pause(f"Card #{pos} failed to flip after retry — game focused?")
                    return
        else:
            time.sleep(1.5)

        # Check Lilis night trigger (every 4th reveal)
        if self.session.is_lilis_alive():
            total_reveals = len(self.session.reveal_order)
            if total_reveals % 4 == 0:
                self.session.lilis_batch_index += 1
                print(f"  [auto] Lilis night triggered (reveal #{total_reveals})")
                self.phase = GamePhase.NIGHT_RESOLVE
                # Parse clue first before night phase
                self._auto_enter_single_card(pos)
                self.session.save()
                return

        # Parse clue from memory
        self._auto_enter_single_card(pos)
        self.session.save()

        self._snapshot_counts()
        self.phase = GamePhase.SOLVING

    def _auto_enter_single_card(self, pos: int):
        """Parse and enter a single card's clue from memory reader."""
        from game_loop import _parse_clue_from_memory, DecisionLog

        if not self.monitor or not self.monitor.is_healthy():
            print(f"  [auto] No monitor — manual entry needed for #{pos}")
            return False

        board = self.monitor.get_board()
        if not board:
            return False

        mc = next((c for c in board if c['position'] == pos), None)
        if not mc:
            return False

        parsed = _parse_clue_from_memory(mc)
        if parsed:
            self.session.add_card(parsed)
            DecisionLog.log_card(parsed)
            print(f"  [auto] #{parsed.position} {parsed.apparent_role}: {parsed.info_parsed}")
            return True
        else:
            role = mc.get('disguise') or mc.get('true_role', '?')
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
        from game_loop import _parse_clue_from_memory, DecisionLog

        pos, targets, ability_name, result = self._pending_ability
        print(f"\n  [auto] Phase: ABILITY_USE #{pos} ({ability_name}) -> targets {targets}")

        # Slayer and PD are complex — pause for human
        if ability_name in ("Slayer", "Plague Doctor", "Plague_Doctor"):
            self._pause(
                f"Use {ability_name} on #{pos} -> targets {targets}. "
                f"Complex ability — enter result manually, then 'resume'."
            )
            return

        if ability_name == "Dreamer" and len(targets or []) != 2:
            self._pause(
                f"Dreamer requires exactly 2 targets; solver returned {targets}. "
                f"Handle manually, then 'resume'."
            )
            return

        # Safety: check that targets don't have unused active abilities
        for t in (targets or []):
            if self._has_active_ability(t):
                self._pause(
                    f"Target #{t} has unused active ability — "
                    f"clicking would activate it instead. Manual handling needed."
                )
                return

        coords = all_game_card_coords(self.session.n_cards)
        if pos not in coords:
            self._pause(f"Position #{pos} not valid for {self.session.n_cards}-card game")
            return

        # Step 1: Click the ability card to activate
        ax, ay = coords[pos]
        _tm.safe_click_at(ax, ay, f"ability_{ability_name}_{pos}")
        time.sleep(0.5)

        # Step 2: Click each target
        for t in (targets or []):
            if t not in coords:
                self._pause(f"Target #{t} not valid for {self.session.n_cards}-card game")
                return
            tx, ty = coords[t]
            _tm.fast_click_at(tx, ty, f"ability_target_{t}")
            time.sleep(0.3)

        # Step 3: Wait for result via memory reader
        # Monitor for uses count change or clue_text change
        old_uses = None
        if self.monitor and self.monitor.is_healthy():
            board = self.monitor.get_board()
            if board:
                mc = next((c for c in board if c['position'] == pos), None)
                if mc:
                    old_uses = mc.get('uses', 0)

            def _ability_resolved(board):
                if not board:
                    return False
                card = next((c for c in board if c['position'] == pos), None)
                if not card:
                    return False
                new_uses = card.get('uses', 0)
                return (
                    new_uses > (old_uses or 0)
                    or bool(card.get('acted_infos'))
                    or bool(card.get('ability_used') and card.get('clue_text'))
                )

            resolved = self.monitor.wait_for(_ability_resolved, timeout=5, min_delay=0.5)
            if not resolved:
                self._pause(
                    f"Ability {ability_name} on #{pos} didn't resolve — "
                    f"check if clicks landed correctly."
                )
                return
        else:
            time.sleep(2)

        # Step 4: Parse result from memory and enter
        self.session.mark_ability_used(pos)

        if self.monitor and self.monitor.is_healthy():
            board = self.monitor.get_board()
            if board:
                mc = next((c for c in board if c['position'] == pos), None)
                if mc:
                    parsed = _parse_clue_from_memory(mc)
                    if parsed:
                        self.session.add_card(parsed)
                        DecisionLog.log_card(parsed)
                        print(f"  [auto] Ability result: #{parsed.position} {parsed.apparent_role}: {parsed.info_parsed}")
                    else:
                        role = mc.get('disguise') or mc.get('true_role', '?')
                        clue = mc.get('clue_text', '')
                        self._pause(
                            f"Ability {ability_name} on #{pos}: couldn't parse result "
                            f"\"{clue}\" — enter manually, then 'resume'."
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
                # Timeout — might be a 0-kill case (Lilis is only unrevealed)
                print(f"  [auto] No kill detected (timeout) — checking 0-kill case")
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
        n_evil = 0
        for c in board:
            pos = c['position']
            if pos in already_dead:
                continue
            if c.get('killed_hidden'):
                killed.append(pos)
                # Honor rule: we use is_evil for VALIDATION only, not deduction
                # But we do need the evil count for the solver's night_kill_evil_count
                if c.get('is_evil'):
                    n_evil += 1

        if killed:
            print(f"  [auto] Night kills detected: {['#'+str(p) for p in killed]}, {n_evil} evil")
            # Record kills
            self.session.night_kills.extend(killed)
            self.session.night_kill_evil_count += n_evil
            for p in killed:
                if p not in self.session.executed:
                    self.session.executed.append(p)
                # Remove from blocked if Witch-blocked (Issue #8)
                if p in self.session.blocked_positions:
                    self.session.blocked_positions.remove(p)
                    print(f"  [auto] Removed #{p} from blocked_positions (killed by Lilis)")
            if n_evil == len(killed) and n_evil > 0:
                for p in killed:
                    if p not in self.session.confirmed_evil:
                        self.session.confirmed_evil.append(p)
        else:
            print(f"  [auto] No kills detected — 0-kill Lilis night")
            # Check if Lilis is only unrevealed (confirms her position)
            revealed = {c.position for c in self.session.cards}
            dead = set(self.session.executed) | set(self.session.night_kills)
            all_pos = set(range(1, self.session.n_cards + 1))
            unrevealed = all_pos - revealed - dead
            if len(unrevealed) == 1:
                lilis_pos = unrevealed.pop()
                if lilis_pos not in self.session.confirmed_evil:
                    self.session.confirmed_evil.append(lilis_pos)
                print(f"  [auto] Only unrevealed card is #{lilis_pos} — confirmed as Lilis")

        # Auto-deduct 2 HP
        old_hp = self.session.hp
        self.session.hp -= 2
        print(f"  [auto] Lilis night HP: {old_hp} -> {self.session.hp}")

        if self.session.hp <= 0:
            self.phase = GamePhase.GAME_OVER
            print(f"\n  [auto] HP DEPLETED - GAME LOST!")
            self.session.save()
            return

        self.session.save()
        self._snapshot_counts()

        # Check if there are more cards to flip (Lilis batches)
        remaining = self._unrevealed_positions()
        if remaining and self.session.is_lilis_alive():
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

    def _unrevealed_positions(self):
        """Get positions that still need flipping."""
        all_positions = set(range(1, self.session.n_cards + 1))
        done = set(self.session.reveal_order) | set(self.session.night_kills) | set(self.session.executed)
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
