# Demon Bluff Solver

## Goal
**Track A — Solver hardening (primary).** Win consistently at high ascensions. Fix rule gaps, bad heuristics, edge cases.

**Track B — Memory reader (secondary).** Build `memory_reader.py` to replace the visual pipeline. Runs alongside every screenshot — any mismatch = stop and fix.

## Core Rules
1. **Always follow the solver.** Execute the solver's top pick, even probabilistic. A wrong answer is a bug to fix between games, not override. See `memory/feedback_never_override_solver.md` (asc50_v1, asc52_v6).
2. **HONOR RULE: memory reader is VALIDATION ONLY.** Memory reader shows true game state including evils. Allowed: cross-check screenshots, verify bugs after the game, auto-fill metadata via `auto_card`. Forbidden: using evil positions to decide execution, re-entering data until the solver lands on the memory-confirmed position. If the solver is degraded, fix the data entry — do not reverse-engineer the truth (asc52_v6).
3. **0 scenarios = STOP.** See [Recovery Protocol](#recovery-protocol).
4. **Fix bugs before the next game.** Research the wiki (https://demonbluff.wiki.gg) first. Fix code, run `cargo test --release --test simulation`, verify. Solver validates against v2 tests only (`tests/cases_v2/`).
5. **After every loss, analyze.** Spawn an agent to check critical decisions. Fix or confirm unavoidable before proceeding.
6. **Commit and push after every game.** Do not batch.
7. **Mouse only.** No keyboard shortcuts during live runs.
8. **Memory reader with every screenshot.** Screenshot is ground truth. Mismatch = stop and fix. (Rule 2 governs what "use" means.)
9. **Serialize state-mutating commands.** Do not issue `game_loop.py` commands (`new`, `deck`, `card`, `execute`, `ability_used`, etc.) in parallel.
10. **Self-improving CLAUDE.md.** When a process error occurs: STOP → identify root cause → update this file → resume. **Prefer editing or tightening an existing rule over appending a new one** — if two rules say the same thing, collapse them. Put loss anecdotes in `memory/losses_postmortem.md`; keep the CLAUDE.md citation to the ascension tag only.

## Recovery Protocol
Triggered by 0 scenarios. Do NOT guess; do NOT reset entries to make the solver happy.

**(a) Identify the most recent data entry** — which card or ability result was added last. That's almost always the source.

**(b) Re-screenshot and verify.** Trust the screenshot, not your memory. Check whether `auto_card`/`auto_ability` misparsed the speech bubble — e.g., a Rambler silencing message like "#X shut up!" does NOT imply FT `has_evil=False`; it's flavor text on a silencing event. Correct with manual `card ...` if needed.

**(c) If the entry is correct and 0 scenarios persist, it's a solver bug.** Save the test case as a loss. Do not "fix" mid-game by resetting other entries — resetting cascades errors and corrupts state (asc52_v6).

**(d) Forbidden recovery:** Do NOT cycle through different values hoping the solver "comes back." Do NOT reset unrelated cards to free up scenarios. Do NOT use memory reader to figure out what value would make the solver happy. All three hide the bug and corrupt state.

**(e) "Accept the loss" ≠ "abandon in-app."** `game_over loss` is a tracking command; it does not require quitting the in-app game. Before giving up on the village:
- **Exhaust ALL unused active abilities first.** Slayer/Druid/PD/etc. can fire in 0-scenario state — a successful kill or check often breaks the deadlock by adding a hard-confirmed data point. Check the last non-zero `next` output for `Available abilities`.
- **Re-check every auto-entered card** for memory-reader parsing errors (see step b).
- Only after exhausting abilities AND verifying all entries should you conclude you're stuck.

**(f) NEVER abandon in-app via pause → "Go to menu" while HP > 0, unused abilities remain, unflipped cards remain, or memory reader is readable.** Abandoning kills the memory reader stream — you lose post-mortem ground truth (asc74_v1). If HP drops to 0 from in-game damage, the game ends itself and final memory state is still capturable. Rage-quit to menu resets the process and loses truth. When in doubt: leave the game on its turn, run `game_over loss` for tracking, diagnose live memory before doing anything else in-app.

## Screen & Coordinates
- **Resolution**: 2560x1440
- **Mouse parking** (before screenshots): `(1280, 690)` — board center void. Avoid cards, panels, buttons, deck icon.
- If a screenshot has a hover tooltip, park and retake.
- `safe_click` auto-focuses the game window. **Prefer `safe_click` over manual move+click.**
- For card clicks, prefer detected card-box centers from the current screenshot. Circle formula (`game_utils.game_card_coords`) is a fallback only — never hardcode card positions (asc45_v5).
- **Execute button**: RED SWORD at ~(2265, 1235). Dismiss mark menu first by clicking `(1280, 690)`, then `safe_click btn_execute_sword`. The mark menu overlaps the execute button (template confidence ~0.45 < 0.70 threshold).
- **Deck icon**: `safe_click icon_deck_purple` at ~(2485, 100). NEVER click near center-top (~(1230, 62)) — that hits card #7 in a 7-card game.
- **Buttons highlight red on hover** — no highlight means the game is unfocused.
- **Escape opens pause menu** — verify cursor before clicking.
- **Dialog Close/Next**: ~y=860. End-screen "Next": ~(1280, 865). Score "Continue": ~(1280, 950).
- **Pause menu**: Go to menu ~y=625, Abandon ~y=770, Settings ~y=840, Back ~y=900.

## Game Loop

### Start
1. **`python game_loop.py start`** — automates Play Demo → Standard → dismiss intro → park mouse → screenshot deck → run card_vision + memory_reader cross-check. `read_deck <screenshot>` re-reads an existing screenshot.
2. Verify the deck output; read board counts (V, O, M, D icons top-right) from screenshot.
3. `python game_loop.py new <n_cards> <n_evil>` — **n_evil = the game's displayed "Find and Execute N Evil Characters" count, NOT just minions+demons.** Puppet counts as an extra evil. Read the intro dialog directly (asc68_v4).
4. `python game_loop.py deck V=... O=... M=... D=... nv=<villager_count> no=<outcast_count>` — prefixes REQUIRED; command errors on missing prefix. When unsure about a role's faction, check `knowledge_base.py` — it's the source of truth (asc14_v1 Knight, asc26_v3 Druid, asc36_v6 PD).
5. Close deck: `safe_click icon_deck_purple`.

### Reveal & Enter
6. **`python game_loop.py flip`** — flips all cards #1→#N in strict order, then auto-runs memory_reader to verify.
   ```
   python game_loop.py flip              # Standard: all cards #1->#N
   python game_loop.py flip --lilis      # Lilis: batches of 4, stops for night phase
   python game_loop.py flip <pos>        # Single card (after Witch death)
   ```
   - **NEVER manually construct click chains.** The `flip` command handles ordering.
   - **Why order matters**: (a) Witch blocks the LAST card attempted — consistent order makes blocked card predictable (#N). (b) `reveal_order` must match flip order for Baker. (c) Card info must be entered in same order.
   - **After Witch death**: `flip <blocked_pos>` — only that one card.
   - **Night-killed (Lilis)**: skull overlay, skip them.
7. **Verify ALL cards flipped.** If flip verification fails (positions still Hidden), re-run `flip`. Card #1 is especially prone to click-not-registering. **NEVER mark a position as blocked without Witch in the deck** — `block` rejects this (asc37_v5). See `memory/feedback_flip_verification.md`.
8. **`auto_card`** reads clues from memory and auto-enters parseable cards. Run after flipping; shows which cards need manual entry.
8b. **Enter card info in order #1→#N** (preserves `reveal_order`). Built-in validation warns on out-of-order entry. `next` warns on missing entries. Active abilities (lightning bolt icon): `card no_info <pos> <Role>`.
   - **Poet "#X is Evil"**: `card poet <pos> bounty_hunter <target>` (NOT medium) (asc37_v3).
9. `set_hp <hp> <wrong_exec_cost>` at game start (defaults to cost=5).
   - **Shaman games (multiple Bakers)**: Shaman swaps villagers to Baker at game start. Swapped Bakers say "I was a [original role]" (self-ID). Original Baker says "I am the original Baker."
   - **Druid "Wretch" result**: enter as `card druid <pos> <targets> Wretch` (not `none`) — evil Druid lying by claiming an outcast.
   - **PD ability**: `pd_check <pd_pos> <target> corrupted <evil_pos>` or `pd_check <pd_pos> <target> clean`. Use `pd_check`, NOT `card pd_check` (asc46_v1).

### Solve & Act
10. `python game_loop.py next` — **do what it says**. Warns on missing entries or HP inconsistency. Auto-executes definite-evil or lookahead forced-safe picks (confidence ≥ 20%). Use `next --plan` or `next --dry` for print-only inspection.
11. **Abilities**: click card → click targets → enter result → `ability_used <pos>` → `next`. WARNING: clicking a card with an unused active ability activates THAT ability instead of selecting it as a target.
12. **Execute**: click center board `(1280, 690)` to dismiss mark menu → `safe_click btn_execute_sword` → click target → screenshot → `execute <pos> <evil_role|good>`. Auto-prints HP reminder.
13. Repeat `next` until game ends.

### End
14. Screenshot end screen. Read true evils + check `<Corrupted>` tags. Game auto-advances after "Next" click — screenshot BEFORE clicking Next if you need end-screen details.
15. `python game_loop.py game_over win/loss <name> "<pos=Role,...>" "[notes]"` — auto-saves test, runs single replay, runs full v2 regression, prints commit checklist. `game_over` auto-reads true evils from memory_reader when not provided. **Only EVIL positions in the dict.** Do NOT include night-killed Good cards — every key is treated as an executed evil (asc54_v4).
16. Follow the printed checklist: commit, push, analyze loss if applicable.

## Village Timing (Asc 45+)
Track time per village from deck read to game_over. Goal: measure automation speedup.
- **Start timer** when deck is read (after clicking Next on previous victory).
- **Stop timer** when `game_over` completes.
- Log format in commit: `Xm:Ys` (e.g., `3m:45s`).

## Deck Read: Memory vs Screenshot Accuracy
Memory reader `--deck` reads the role POOL (all possible roles). Screenshot shows pool + HEADER (board counts: V=N, O=N).
- Memory reader deck is 100% accurate for role names (confirmed across 7 Asc44 villages).
- Memory reader does NOT read header counts (`nv=`, `no=`). These must come from screenshot or manual entry.
- **Goal**: read nv/no from memory to eliminate the deck screenshot. Needs `GameData` IL2CPP offset work.

## Memory Reader — Continuous Validation
Reads game state from process memory (`memory_reader.py`). Goal: replace the visual pipeline entirely.

**Clue reading (Phase 2, confirmed Asc44):**
- `savedAct` (0x158): speech bubble text — works for passive AND active ability results.
- `actedInfos` (0x128): List of {desc, targets} including referenced position numbers.
- `runtimeData` (0x68): Enlightened direction, Alchemist cures, Baker original role.
- `auto_card` uses these to auto-enter cards.

Every screenshot, memory reader reads state and compares against what the screenshot shows. Screenshot is ground truth. **Any mismatch = stop, diagnose, fix, verify, resume.**

**Notes**:
- Multi-village: fixed. Uses Unity native object name at `m_CachedPtr(0x10)+0x48` with `characterId` fallback.
- Player.log: `%LOCALAPPDATA%Low/UmiArt/Demon Bluff/Player.log` — INIT entries have true roles in reverse position order.
- Name mappings: Gambler→Gemcrafter, Imp→Baa, Baron→Chancellor, plus the
  asset-proven internal mappings Skinwalker→Mutant, Marionette→Twin Minion,
  Mezepheles→Puppeteer, and Puzzlemaster→Plague Doctor (see `DISPLAY_NAMES`).

## Setup
- Screen: 2560x1440, Python 3.13, Rust 2021 edition.
- Python: `mss`, `pyautogui`, `Pillow`.
- Cargo workspace at repo root; Rust crates in `crates/`.
- **Test directories**: `tests/cases_v2/` (card_vision pipeline, default for `game_over`), `tests/cases/` (legacy, kept for reference, not run).
- **REPL mode**: `python game_loop.py repl` — persistent process, no import overhead per command. Uses `REPL_READY`/`CMD_DONE` sentinels.

## Rust Solver (`crates/solver-core`)
- **Rust solver is PRIMARY** — `game_loop.py` calls `rust_solve_to_objects()` exclusively. Fix solver bugs in Rust, not Python.
- **Persistent daemon**: `--daemon` mode keeps solver binary alive across calls; falls back to one-shot.
- Build: `cargo build --release` | Test: `cargo test --release --test simulation`.
- Python bridge: `rust_solver.py` wraps the CLI binary via subprocess.
- Game loop, strategy, screenshots, memory reader, card vision remain Python.

## Gotchas (working correctly but surprising)
See `memory/project_open_solver_issues.md` for the full list. Highlights:
- **Doppelganger counts in nv, not no** (header). Solver handles via `n_disguisers` allowance. Enter `no=` as displayed.
- **Two Bakers from a single-Baker pool** (no Shaman) is valid — Baker conversion chain activates at game start.
- **Drunk counts as Villager in header** — `board_outcast_count` may undercount. Handled.
- **`next` auto-executes by default** (≥20% confidence). Use `next --plan` or `--dry` for print-only.
- **Baa's eye-symbol mismatch is deck-VIEW only, not HUD.** Native `Imp.Act` obscures one existing Outcast record and reveals that deck-strip identity on Baa's death; it does not add a gameplay role or flip a board card. Top-right HUD counts Baa as Demon (1D), so do NOT adjust `no=` (asc75_v3).
- **Public Dreamer:** the shipped asset binds managed `Dreamer`, not the unbound `Dreamer2` alternate. It picks exactly 2 characters and immediately returns `Among #X, #Y there is: R1 or R2`, with no role picker; a selected Wretch instead produces the truthful Cabbage clue. The current native fallback can truthfully report both selected roles, and a lying result can match one selected real role through the other target's bluff, so validation and recommendation must use native output support rather than a simple one-match/zero-match rule. **Firing Dreamer manually is PREFERRED** when the solver recommends it. Flow: click Dreamer card → pick 2 characters → read the immediate result. Only use `ability_used <pos>` if you've confirmed Dreamer truly cannot narrow. While target-selection mode is active, do not re-click Dreamer; that can cancel/reactivate and cost HP.

## Game Overview
Puzzle/deduction game. Circle of face-down cards — reveal for role info, deduce Evil, execute. Evil disguises as Villagers and lies. Good can become corrupted (unreliable info). Win by executing all Evil before HP runs out.
