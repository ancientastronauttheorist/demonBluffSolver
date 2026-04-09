# Automation Plan: Play Faster, Collect More Data, Fix Solver Faster

## Goal
Reduce game time from ~8 min to ~4-5 min by automating mechanical steps while keeping Claude for judgment calls.

## Phase 0: Foundation (8-10h)

### REPL Mode (~6-8h)
Extract `dispatch(cmd, args, session)` from `main()` (28 command branches, 58 sys.argv refs).
Add `python game_loop.py repl` — persistent process, session in memory, `REPL_READY`/`CMD_DONE` sentinels.

**Critical fixes required:**
- Replace `sys.exit(1)` in `slayer_result` (line 1361) with raised exception
- Wrap all `subprocess.run(check=True)` in try-except (flip, start commands)
- Change session lock to per-command acquire/release (not held for REPL lifetime)
- Add `sys.stdout.flush()` before every `CMD_DONE` sentinel
- Isolate subprocess stdout from REPL sentinels (prefix/delimit)
- Catch all exceptions in dispatch loop — print error, continue REPL
- Use `shlex.split()` for argument parsing with error handling
- Handle `new` command returning fresh session to REPL loop

**Files:** game_loop.py

### Fast Flip (~2h)
Add `fast_click_at()` to template_match.py — direct function call (not subprocess), skip verification screenshot.
Rewrite flip loop to use direct imports instead of subprocess spawning.

**Critical fixes required:**
- MUST keep `ensure_game_focused()` per card — focus loss cascade is the #1 risk
- Cannot lower `pyautogui.PAUSE` via subprocess (each subprocess resets it) — only works with direct calls
- Start with 0.2s inter-card delay (not 0.15s) — test empirically, lower if reliable
- Batch all clicks, verify once at end via memory_reader
- Auto-retry failed flips at safe (normal) speed
- Single-card flip (post-Witch) should NOT use fast_click — use safe_click_at

**Files:** template_match.py, game_loop.py, mouse.py

## Phase 1: Automation Core (8-10h)

### Persistent Rust Subprocess (~3h)
Add `--daemon` flag to `crates/solver-cli/src/main.rs`: read JSON lines via `BufReader::lines()` (NOT `read_to_string`), write compact JSON line responses, flush after each.

**Critical fixes required:**
- Use `BufReader::lines()` NOT `read_to_string()` (blocks until EOF)
- `catch_unwind` around each solve to prevent panics killing daemon
- Always write exactly one response line per request (never pretty-print)
- Explicit `stdout.flush()` after each response (Windows pipe buffering)
- Handle stderr: don't let it accumulate in pipe buffer (redirect or drain)
- Python side: `Popen` with pipe management, `_daemon_lock` for thread safety
- Fallback to one-shot mode if daemon fails
- `atexit.register(shutdown_daemon)` for cleanup (but document: won't run on crash/kill)
- Per-request `"__summary": true` field (not CLI flag, which is process-lifetime)

**Files:** crates/solver-cli/src/main.rs, rust_solver.py

### Player.log Parser (~3h)
New `log_parser.py` — parse Player.log events.

**Known limitations (from investigation):**
- INIT entries have NO position numbers — reverse-order mapping is assumed but unvalidated
- ACT/OnExecuted events have NO position data — log is a *witness*, not *source of truth*
- Only Confessor logs clue content (`NANI: DIZZY`) — most roles log nothing useful
- File gets overwritten on game restart (old -> Player-prev.log)
- File has mixed CRLF/CR encoding — use universal newline mode

**Useful for:**
- Ground truth verification of true roles (cross-check with memory_reader)
- Event sequence confirmation (reveals, night kills, executions)
- Game-over auto-detection (Player info reset = village boundary)

**NOT useful for:** Auto card entry (no clue content for most roles), position tracking.

**Files:** log_parser.py (new), game_loop.py (integration commands)

### Game-Over Streamlining (~2h)
Auto-read true evils from memory_reader, auto-detect win/loss, skip redundant single replay.

**Critical fixes required:**
- Do NOT detect win/loss from HP alone — Bombardier instant-loss has HP > 0. Check game state from memory.
- Read memory BEFORE user clicks "Next" (game auto-advances, stale data after)
- Add test name collision detection (check if file exists before save)
- Implement test name generation from scorecard history (doesn't exist yet)
- Add `timeout=120` to cargo test subprocess (prevent indefinite hang)
- Keep manual override: `game_over win/loss <name> "<evils>" "[notes]"` still works

**Files:** game_loop.py, memory_reader.py

## Phase 2: Auto-Read Clues (14-18h) — HIGHEST RISK

### Memory Reader Expansion (~10-12h)
Read clue/ability data from Character objects in process memory.

**Target fields (from dump.cs):**
- Character+0x128: `actedInfos` (List<ActedInfo>) — each has `desc` (string, 0x10) + `characters` (List<Character>, 0x18)
- Character+0x158: `savedAct` (string — cached clue text)
- Character+0x68: `runtimeData` (polymorphic: EnlightenedRuntimeData.direction, AlchemistRuntimeData.cures, BakerRuntimeData.charName)
- Character+0xBC: `uses` (int — ability use count)
- Character+0x161: `act` (bool — ability activated)

**Critical fixes required:**
- MUST dispatch runtimeData reads by role name (polymorphic — can't read blindly)
- Active ability cards (Jester, FT, Judge, Slayer, Dreamer) have EMPTY clues until activated — skip these
- Baker and Poet require manual entry regardless (dynamic role changes, copied abilities)
- Validate offsets against live game before trusting (read known field first, verify)
- Add version fingerprinting (PE timestamp) to detect game updates
- Localization breaks text parsing — English-only for now, detect and warn
- Null pointer checks at every level of pointer chain traversal
- Cap list reads at 20 elements to prevent runaway on corruption

**Roles that CAN be auto-read (passive, simple clues):**
Enlightened (direction enum), Knitter (evil_pairs int), Confessor (dizzy bool), Lover (evil_adjacent int), Hunter (distance int), Architect (side enum), Bard (corruption_distance int), Empress (targets list), Witness (target), Alchemist (cures int)

**Roles that CANNOT be auto-read (active abilities, complex formats):**
Jester, Fortune Teller, Judge, Slayer, Dreamer, Druid, Poet, Baker, Medium, Oracle, Scout

### Auto Card Entry (~4-6h)
New `auto_card` command: reads memory → parses clues → generates card commands.

**Design:**
- Only auto-enter roles from the "CAN be auto-read" list above
- Print proposed entries for user confirmation before committing
- Skip positions with active abilities (uses=0, act=false)
- Cross-validate against screenshot (CLAUDE.md rule: screenshot is ground truth)
- Fall back to manual entry for any card where parsing fails

**Files:** memory_reader.py, game_loop.py

## Phase 3: Autonomous Execution (10-12h) — HIGH RISK

### Auto-Execute (~8h)
New `auto_next` command: solve → if definite_evil → click sequence → verify → record.

**Critical fixes required:**
- ONLY auto-execute `definite_evil` (100% across ALL scenarios, NOT bombardier, NOT Knight free-check)
- Triple bombardier guard: strategy filters it, auto_next checks it, auto_execute asserts it
- Re-read board via memory_reader IMMEDIATELY before clicking (race condition: night kills can change state)
- Use FRESH screenshot for card coordinates (positions shift after deaths)
- Dismiss mark menu → screenshot → verify menu gone → find execute button → click → click target → wait → verify
- Poll memory_reader for state=Dead (don't use fixed 2.5s wait — retry with backoff)
- Handle Wretch: memory shows alignment=Good, true_role=Wretch → mark as wrong execution, decrement HP
- Handle Knight immunity: card stays Alive after execution → not a failure, it's immunity
- Handle Doppelganger-as-Knight: use true_role from memory, not apparent role
- Detect game-over: if HP <= 0 after wrong exec, stop immediately
- Detect victory: if last evil killed, expect victory screen
- Abort gates: HP <= wrong_exec_cost (no budget), Bombardier in play, multiple definite_evil (ambiguous)

### Full Loop Orchestration (~2-4h)
Chain: flip → auto_card → next → auto_next if certain → repeat.
Claude intervenes for: probabilistic decisions, ability activations, 0-scenario diagnosis.

**Files:** game_loop.py, strategy.py, template_match.py, memory_reader.py

## Realistic Targets

| Milestone | Time/Game | Games/Hour | Key Bottleneck |
|-----------|-----------|------------|----------------|
| Current | ~8 min | ~7-8 | Claude types everything manually |
| Phase 0 | ~6-7 min | ~8-9 | Reduced sleep times + no subprocess overhead |
| Phase 1 | ~5-6 min | ~10-11 | Faster solver + streamlined game_over |
| Phase 2 | ~3-4 min | ~15-18 | Auto-read clues for simple roles |
| Phase 3 | ~3 min | ~18-20 | Auto-execute definite evil |

**Floor:** ~3 min/game. Manual decisions, complex roles, ability activations create an irreducible minimum.

## What Always Stays Manual
- Speech bubble reading for complex roles (Poet, Baker, Oracle, Medium, Scout)
- Ability activation + result reading (Jester, FT, Judge, Slayer, Dreamer, Druid)
- Probabilistic execution decisions (< 100% confidence)
- 0-scenario diagnosis and solver bug fixing
- Loss analysis
- Commit and push
- Board count verification (V, O, M, D icons)
