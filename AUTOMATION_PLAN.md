# Autonomous Self-Improving Game Loop: Master Plan

## Goal
Run games as fast as possible to collect data and improve the solver. Replace Claude-as-orchestrator with a fully autonomous loop.

| Metric | Current | Target |
|--------|---------|--------|
| Time per game | 3-5 min | 30-60 sec |
| Games per hour | 12-20 | 60-120 |
| Human interventions | 5-10/game | 0-1/game |
| Data pipeline | Manual analysis | Auto-detect solver weaknesses |

---

## Three Pillars

### Pillar 1: Autonomous Loop
Single command (`python game_loop.py auto --games N`) plays N games start-to-finish with zero human intervention.

### Pillar 2: Speed Optimization
Eliminate sleep delays, subprocess overhead, and screenshot latency. Game animations become the only bottleneck.

### Pillar 3: Self-Improvement Pipeline
After every game, auto-detect solver weaknesses, categorize failures, and prioritize fixes by impact.

---

## Critical Issues (Must Fix Before Building)

11 issues found by adversarial review that would cause data corruption, stuck loops, or game losses:

| # | Issue | Severity | Fix |
|---|-------|----------|-----|
| 1 | Session state leaks between games | CRITICAL | `full_reset()` clears ALL mutable state + explicit `new` call per game |
| 2 | Rust daemon accumulates state across games | CRITICAL | Kill daemon between games via `_kill_daemon()` |
| 3 | MemoryMonitor doesn't detect new game board | CRITICAL | Shutdown monitor between games, wait for `game_connected` event |
| 4 | Stale deck read from previous village | CRITICAL | Wait for `game_connected` event before `read_deck()` |
| 5 | Clicking revealed card triggers active ability | CRITICAL | Check `used_abilities` + `session.cards` before every target click |
| 6 | `_verify_flips` prints warnings instead of returning status | CRITICAL | Return structured `{flipped: [], blocked: [], failed: []}` |
| 7 | Night phase HP not auto-deducted (prints "run set_hp") | CRITICAL | `session.hp -= 2` in NIGHT_RESOLVE phase automatically |
| 8 | Witch-blocked card killed by Lilis stays in blocked_positions | CRITICAL | Remove from `blocked_positions` in `night_kill` handler |
| 9 | Bombardier guard blocks provably-safe forced execution | CRITICAL | Trust forced-exec when it explicitly confirms Bombardier safe |
| 10 | No game version detection for memory offsets | CRITICAL | Add GameAssembly.dll version fingerprint before trusting memory |
| 11 | Screenshot disk fills up over 100 games (~4MB/game) | CRITICAL | Auto-cleanup between games, keep only last 20 screenshots |

---

## Per-Game Isolation Protocol

Between every game, the batch runner MUST execute:

```
1. session.full_reset()      -- clear ALL mutable state (cards, executed, confirmed, abilities)
2. _kill_daemon()            -- fresh Rust solver process
3. _shutdown_monitor()       -- fresh memory reader connection
4. cleanup_screenshots(20)   -- keep only last 20 files
5. Wait for game_connected   -- confirms new village loaded in memory
```

This prevents state leakage (issues #1-4) and disk exhaustion (#11).

---

## State Machine Design

Extends existing `state_machine.py` with new phases:

```
MENU_NAV --> DECK_READ --> SESSION_INIT --> FLIPPING
    ^                                         |
    |                                         v
    |                                  ENTERING_CLUES
    |                                         |
    |                                         v
    |          WITCH_UNBLOCK --+          SOLVING
    |               ^         |              |
    |               |         v    +---------+---------+
    |          SOLVING <------+    |         |         |
    |               ^              v         v         v
    |               |         EXECUTING   REVEAL   LILIS_NIGHT
    |               |              |         |         |
    |               |              v         v         v
    |               +---------SOLVING    FLIPPING  NIGHT_RESOLVE
    |                                   (single)       |
    |                                      |           v
    |                                      v        SOLVING
    |                               ENTERING_CLUES
    |                                  (single)
    |
    +--- POST_GAME <--- GAME_OVER
```

### Phase Details

#### MENU_NAV (replaces `start` subprocess calls)
- Click `menu_play_demo` -> `mode_standard` -> `btn_close_dialog`
- Verify each click landed via template re-match or memory `game_connected`
- Handle unexpected dialogs: ascension-complete, tutorial, "Continue?" prompt
- Fallback: if template not found after 3 retries, NEEDS_HUMAN

#### DECK_READ
- `memory_reader.read_deck()` in-process (100% accurate for role names, confirmed Asc44+)
- nv/no header counts: screenshot parse until memory offset found
- Cross-check memory vs card_vision; mismatch = NEEDS_HUMAN
- Guard: wait for `game_connected` before reading (prevents stale deck from previous village)
- Guard: recognize Baa's one obscured existing Outcast in deck view; never adjust HUD `no=`

#### SESSION_INIT
- Derive n_cards from `len(board)` via memory reader
- Derive n_evil from deck composition
- Create fresh `GameSession(n_cards, n_evil)`, set deck, set HP (10, cost=5 for high asc)
- Close deck panel: `safe_click icon_deck_purple`

#### FLIPPING (extends existing)
- Standard: iterate 1..N, `fast_click_at` each, 0.2s between
- Lilis: batches of 4, then LILIS_NIGHT phase
- Verification: `monitor.wait_for()` until all flipped positions show state != Hidden
- Witch detection: if last card stays Hidden and Witch in deck, mark blocked
- Card #1 guard: use `safe_click_at` for first card (focus-loss prone)
- MUST return structured result from `_verify_flips`

#### ENTERING_CLUES (extends existing)
- Run `_parse_clue_from_memory()` for each flipped, un-entered card
- ~90% of roles parse correctly from memory (confirmed Asc44: 6/6)
- Unparseable cards: enter as `no_info` (solver works, loses one constraint)
- Fallback: NEEDS_HUMAN with raw clue text displayed
- Process in strict position order (preserves reveal_order for Baker)

#### SOLVING (extends existing)
- `session._solve(state)` -> `recommend_action(state, result, used_abilities)`
- Decision routing:

| Action Type | Behavior |
|-------------|----------|
| `execute` + definite_evil + not Bombardier | Auto-execute |
| `execute` + probabilistic + confidence >= threshold + HP budget allows | Auto-execute (--risk moderate) |
| `execute` + probabilistic + below threshold | NEEDS_HUMAN |
| `reveal` | Auto-reveal: click card, wait for flip, enter clue |
| `use_ability` | Auto-use: click card, click targets, read result from memory |
| `win` | Transition to GAME_OVER -> POST_GAME |
| `error` (0 scenarios) | NEEDS_HUMAN -- solver bug |

#### REVEAL (new phase)
- Strategy recommends which card to flip for max information gain
- Click safety: verify position is NOT already flipped, does NOT have unused active ability
- `fast_click_at` target position
- `monitor.wait_for()` until state != Hidden
- Parse clue from memory, add to session
- Transition to SOLVING

#### ABILITY_USE (new phase)
- Click ability card to activate
- Click target(s) -- multi-step for Jester (3), Druid (3), FT (2)
- Guard: if target has unused active ability, NEEDS_HUMAN (would trigger wrong ability)
- `monitor.wait_for()` until `clue_text` or `uses_count` changes
- Parse result from memory (`savedAct`, `actedInfos`, `runtimeData`)
- Record with `session.add_card()` + `session.mark_ability_used()`
- Special: Slayer kill -- check if target is Wretch (wrong exec penalty, NOT confirmed_evil)
- Special: PD check -- two-step result (corrupted/clean, then evil reveal if corrupted)

#### EXECUTING (extends existing)
- Uses existing `auto_execute()`: dismiss mark menu, click execute button, click target, verify
- Structured return: `{killed: bool, knight_immunity: bool, was_evil: bool, role: str}`
- Knight immunity on definite-evil target = solver bug, flag explicitly
- After Witch execution: check for blocked position to unblock (transition to WITCH_UNBLOCK)
- After execution: re-solve

#### LILIS_NIGHT (extends existing)
- Triggered after every 4th reveal when Lilis alive
- `monitor.wait_for(killed_hidden, min_delay=2.0, timeout=8)`
- Transition to NIGHT_RESOLVE

#### NIGHT_RESOLVE (new phase)
- Auto-detect killed positions from `killed_hidden` flag in memory
- Count evil kills using memory `is_evil` field (honor rule: validation only, not deduction)
- Auto-deduct 2 HP (`session.hp -= 2`)
- Remove killed positions from `blocked_positions` (Witch+Lilis interaction)
- Handle 0-kill case: if Lilis is only unrevealed, auto-execute `night_no_kill`
- Track Lilis batch index explicitly (don't derive from `len(reveal_order)`)

#### GAME_OVER
- Detect: all evil Dead in memory, or HP <= 0
- Read true evils from EXISTING monitor's cached board BEFORE clicking Next
- Screenshot end screen for records

#### POST_GAME (new phase)
- Generate test name from ascension/village counters
- `game_over` dispatch: save test case, record scorecard
- Skip per-game `cargo test` compilation in batch mode (compile once at end)
- Click "Next" at ~(1280, 865), "Continue" at ~(1280, 950)
- Handle ascension-complete dialogs
- Run per-game isolation protocol (reset session, kill daemon, shutdown monitor)
- Transition to MENU_NAV for next game

---

## Click Safety Protocol

Before ANY card click in the autonomous loop:

```
1. Is position already flipped? (check session.cards + reveal_order) --> skip
2. Does position have unused active ability? (check knowledge_base + used_abilities) --> NEEDS_HUMAN
3. Is game window focused? (ensure_game_focused()) --> refocus if not
4. After click: verify expected state change via memory reader
5. If state didn't change: retry once at safe speed, then NEEDS_HUMAN
```

---

## Speed Optimizations

### Safe (implement immediately)

| Optimization | Time Saved | Notes |
|---|---|---|
| In-process calls instead of subprocess | 3-7s/game | Fix session lock to per-command acquire/release. Fix `pyautogui.PAUSE` import-time reset. |
| Poll interval 0.5s -> 0.2s | 1-3s/game | CPU impact negligible (0.25% single core) |
| Lilis `monitor.wait_for()` (remove fixed sleep fallback) | 2-3s/game | Already implemented, ensure fallback path uses it |
| Post-click delay 0.3s -> 0.15s | 0.5s/game | Test with 5-10 games first |
| Focus delay 0.2s -> 0.05s (when already focused) | 0.3s/game | Only when game is already in foreground |

### Do NOT Reduce

| Setting | Current | Why |
|---|---|---|
| `pyautogui.PAUSE` | 0.3s | Below 0.2s causes double-clicks, missed clicks, focus races |
| Inter-card flip delay | 0.2s | Unknown if game queues or drops fast clicks. Empirical testing needed. |
| Hover delay (safe_click) | 0.3s | Verification screenshot is debugging gold. Non-negotiable. |
| Batch-end delay (Lilis) | 1.5s | Risks reading memory mid-animation. Use `monitor.wait_for()` instead. |

---

## Memory Reader: Current Status

### Reliable (trust as-is)
- True role, disguise, alignment, card state, flip detection
- Multi-village (native name fix confirmed)
- Thread safety (GIL-enforced, copy-on-write)
- killed_hidden timing (flag set immediately, 2s min_delay covers animation)
- Execution detection (uses card state, not Score.killedEvils)

### Mostly Reliable (use with cross-checks)
- Clue parsing: ~90% of roles parse correctly
- Connection lifecycle: smooth between villages, handles crashes
- Process attachment: first-match PID (fails silently with multiple instances)

### Fragile (fix before trusting as primary)
- **No version detection**: offsets hardcoded, game update silently breaks everything
- **No offset validation**: garbage reads produce wrong data without errors
- **No sanity checks**: positions could be negative, alignments could be garbage

### Required Before Primary Use
1. Add GameAssembly.dll version fingerprint (PE timestamp or build hash)
2. Add sanity checks: if positions are garbage or all alignments None, error out
3. Add multiple-instance guard: error if >1 Demon Bluff process found
4. Keep screenshot cross-checks until version detection is proven

### Missing Offsets (block full autonomy)
- **HP value**: blocks damage tracking. Workaround: infer from Score.killedEvils delta.
- **Game phase enum**: blocks game-over detection. Workaround: all evil Dead = win, HP <= 0 = loss.
- **nv/no board counts**: blocks eliminating deck screenshot. Workaround: screenshot header parse.

---

## Self-Improvement Pipeline

### Build Order (by feasibility and value)

#### Phase 1: Post-Game Decision Analysis (READY, ~200 LOC)
Replay each completed game with ground truth, annotate each decision:
- Re-run solver at each step via existing `replay_game()` infrastructure
- Compare solver recommendation against action actually taken
- Classify each wrong decision: missing_constraint, bad_heuristic, probabilistic_loss, data_entry_error
- Output: `decision_analysis/{case_name}.json`

Data available: 169 v2 test cases, full replay in 72 seconds.

#### Phase 2: Bug Fix Impact Measurement (READY, ~100 LOC)
- Run full v2 test suite, record baseline (72 seconds)
- Apply solver fix
- Re-run test suite, measure delta
- Track: which cases changed behavior, did win rate improve?
- Output: `impact_{fix_name}.json`

#### Phase 3: Cross-Game Failure Pattern Detection (READY, ~150 LOC)
- Group decisions by {action_type, role_combo, hp_remaining}
- Find recurring failure combos (e.g., "Poet+Lilis breaks validator X")
- Flag after 2+ occurrences
- Output: `failure_report` command showing top 20 patterns

#### Phase 4: Confidence Calibration (NEEDS_WORK, ~250 LOC)
- Track solver confidence vs actual outcomes across games
- Build calibration curves: when solver says 80% confident, is it right 80% of the time?
- Identify over/under-confident ranges for strategy threshold tuning
- Minimum viable: ~1,700 data points from 169 cases x ~10 decisions each

#### SKIP: Impact-Based Prioritization
Requires causal inference (counterfactual game tree branching) that's impractical.
Simpler alternative: rank by `frequency x HP_cost` from failure patterns.

---

## Error Handling & Recovery

### Tier 1: Auto-recoverable (no pause)
- Template not found: retry 3x with 0.5s delay, fresh screenshot each
- Click didn't register (card still Hidden): re-click position
- Memory reader returns None board: reconnect via MemoryMonitor
- Solver cache miss: solve fresh

### Tier 2: Degraded but continuing (log warning)
- Single clue fails to parse: enter as `no_info`, solver loses one constraint
- Template confidence < 0.8: log warning, proceed
- Wrong execution on definite evil: log loudly, continue if HP permits

### Tier 3: NEEDS_HUMAN (pause the loop)
- 0 surviving scenarios: solver bug, must fix
- HP depleted: game lost, transition to POST_GAME
- All remaining positions are Bombardier candidates: too risky
- Knight immunity on definite-evil target: solver bug
- Game process not found: game crashed

### Tier 4: Fatal (abort the run)
- Memory reader can't open process after 5 retries
- Rust solver binary not found
- Regression test failures after game_over
- 3 consecutive game failures (likely systemic issue)

### Multi-Game Error Strategy
- Each game wrapped in try/except
- On unrecoverable error: save session state, log error, attempt next game
- Track consecutive failures: abort after 3
- Disk space check before each game: abort if < 100MB free

---

## Implementation Roadmap

### Week 1: Safe Speed + Critical Fixes
- [x] Switch to in-process calls (fix session lock to per-command, fix pyautogui.PAUSE import)
- [x] Reduce safe delays (post-click 0.3->0.15, focus 0.2->0.05)
- [x] Poll interval 0.5->0.2
- [x] Fix all 11 critical issues listed above
- [x] Add `full_reset()` to GameSession
- [x] Make `_verify_flips` return structured result
- [x] Auto-deduct HP in night phase
- [x] Add GameAssembly.dll version fingerprint to memory reader

### Week 2: State Machine Extensions
- [x] Add REVEAL phase (with click-safety protocol)
- [x] Add ABILITY_USE phase (with multi-step support for Jester/Druid/FT)
- [x] Add NIGHT_RESOLVE phase (auto-detect kills, auto-deduct HP)
- [ ] Make all dispatch commands return structured results (not just print)
- [x] Handle Witch+Lilis interaction (remove killed from blocked_positions) — done in Week 1
- [x] Track Lilis batch index explicitly

### Week 3: Full Autonomous Loop
- [x] Add MENU_NAV, DECK_READ, SESSION_INIT, POST_GAME phases
- [x] Add `python game_loop.py auto [--games N] [--risk conservative|moderate|aggressive]`
- [x] BatchGameRunner with per-game isolation protocol
- [x] Screenshot cleanup, disk space checks — cleanup done in Week 1, disk check in BatchGameRunner
- [x] Error categorization (recoverable vs fatal) — 3 consecutive failures = abort
- [ ] Skip per-game cargo test in batch mode

### Week 4: Self-Improvement Pipeline
- [x] Post-game decision analyzer (extend replay infrastructure)
- [ ] Bug fix impact measurement (before/after test suite diffs)
- [x] Cross-game failure pattern detection
- [x] `decisions <game>`, `failure_report` commands

### Ongoing: Memory Reader Gaps
- [ ] Find HP offset (IL2CPP GameData or Gameplay class)
- [ ] Find game phase enum offset
- [ ] Find nv/no board count offsets (eliminate last deck screenshot)
- [ ] Improve clue parsing success rate (target 95%+)

---

## Performance Estimates (Revised)

| Phase | Time/Game | Games/Hour | Human Interventions | Bottleneck |
|-------|-----------|------------|---------------------|------------|
| Current | 3-5 min | 12-20 | 5-10/game | Claude thinking + typing |
| After Week 1 | 2-3 min | 20-30 | 3-5/game | Sleep delays + subprocess |
| After Week 2 | 1-2 min | 30-60 | 1-2/game | Unparseable clues + risk decisions |
| After Week 3 | 30-60 sec | 60-120 | 0-1/game | Game animations only |
| After Week 4 | 30-60 sec | 60-120 | 0/game (auto-diagnose) | Game animations only |

The 5-10x speedup comes primarily from eliminating Claude-as-orchestrator (Week 3), not from reducing sleep delays (Week 1). Week 1 gets a safe 1.5-2x; Week 3 gets the full multiplier.

---

## Risk Tolerance Configuration

```
--risk conservative   Only auto-execute definite evil (default for data collection)
--risk moderate       Auto-execute if confidence >= 65% AND HP >= 2x wrong_exec_cost
--risk aggressive     Auto-execute if confidence >= 50% (maximum data collection speed)
```

Default to `conservative` for batch data collection. Switch to `moderate` when optimizing win rate.

---

## What Always Stays Manual (irreducible)
- 0-scenario diagnosis and solver bug fixing (requires human judgment)
- Loss analysis (understanding why we lost)
- Solver code changes (fixing validators, adding constraints)
- CLAUDE.md updates (documenting new learnings)
- Commit and push decisions
