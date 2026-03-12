# Demon Bluff Solver

## Goal
**Track A — Solver hardening (primary).** Win consistently at high ascensions. Fix rule gaps, bad heuristics, edge cases.

**Track B — Memory reader (secondary).** Build `memory_reader.py` to replace the visual pipeline. Runs alongside every screenshot — any mismatch = stop and fix.

## Core Rules
1. **Always follow the solver.** No second-guessing, no manual overrides.
2. **0 scenarios = STOP.** Fix the solver immediately. Do not guess.
3. **Fix bugs before the next game.** Research the wiki (https://demonbluff.wiki.gg) first. Fix code, run replay tests (`python -m tests.test_replay --v2-only`), verify. Same urgency for solver and memory reader bugs. **Solver work validates against v2 tests only** (card_vision pipeline, high-accuracy data). Legacy tests (`tests/cases/`) are kept for broad regression but may have manual data entry errors.
4. **After every loss, analyze.** Spawn an agent to check critical decisions. Fix or confirm unavoidable before proceeding.
5. **Commit and push after every game.** Do not batch.
6. **Mouse only.** No keyboard shortcuts during live runs.
7. **Memory reader with every screenshot.** Compare against screenshot (ground truth). Mismatch = stop and fix.
8. **Serialize state-mutating commands.** Do not issue `game_loop.py` commands (`new`, `deck`, `card`, `execute`, `ability_used`, etc.) in parallel.
9. **Self-improving CLAUDE.md.** When any process error occurs during the game loop (wrong flip order, missed step, data entry mistake, wrong coordinates, etc.): STOP immediately -> identify root cause -> update THIS FILE with a guard/fix to prevent recurrence -> think about related edge cases -> resume. Every mistake should make the process permanently better.

## Screen & Coordinates
- **Resolution**: 2560x1440
- **Mouse parking** (before screenshots): `(1280, 690)` -- board center void. Avoid cards, panels, buttons, deck icon.
- If a screenshot has a hover tooltip, park and retake.
- `safe_click` auto-focuses the game window. **Prefer `safe_click` over manual move+click.**
- For card clicks, prefer detected card-box centers from the current screenshot. Circle formula is a fallback only.

## Game Loop

### Start
1. **`python game_loop.py start`** -- automates: Play Demo -> Standard -> dismiss intro -> park mouse -> screenshot deck -> run card_vision + memory_reader with cross-check. Or use `read_deck <screenshot>` to just re-read an existing screenshot.
2. Verify the deck output, read board counts (V, O, M, D icons top-right) from screenshot.
3. `python game_loop.py new <n_cards> <n_evil>`
4. `python game_loop.py deck V=... O=... M=... D=... nv=<villager_count> no=<outcast_count>` -- prefixes REQUIRED (command errors on missing prefix instead of silently ignoring). **CRITICAL: Plague Doctor = OUTCAST (Good), NOT a Minion! Enter as O=...,Plague_Doctor. Lost asc36_v6 replay test to this. Check knowledge_base.py when unsure about a role's faction.**
5. Close deck: `safe_click icon_deck_purple` (at ~(2485, 100)). **NEVER click near the top-center area (e.g., ~(1230, 62)) thinking it's the deck icon — that hits card #7 in a 7-card game.**

### Reveal & Enter
6. **`python game_loop.py flip`** -- flips all cards #1->#N in strict order, then auto-runs memory_reader to show board state.
   ```
   python game_loop.py flip              # Standard: all cards #1->#N
   python game_loop.py flip --lilis      # Lilis: batches of 4, stops for night phase
   python game_loop.py flip <pos>        # Single card (after Witch death)
   ```
   - **NEVER manually construct click chains.** The `flip` command handles ordering.
   - **Why order matters**: (a) Witch blocks the LAST card attempted -- consistent order makes blocked card predictable (#N). (b) `reveal_order` must match flip order for Baker. (c) Card info must be entered in same order.
   - **After Witch death**: `flip <blocked_pos>` -- only flips that one card.
   - **Night-killed cards (Lilis)**: Show skull overlay, skip them.
7. Screenshot and verify memory_reader output (auto-printed by `flip`). **HONOR RULE: memory reader shows true evil roles -- DO NOT use for solving. Validation only.**
8. **Enter card info in order #1->#N** (preserves `reveal_order`). Built-in validation warns on out-of-order entry. `next` warns if positions are missing entries. Active abilities (lightning bolt icon): `card no_info <pos> <Role>`.
9. `set_hp <hp> <wrong_exec_cost>` at game start (defaults to cost=5).

### Solve & Act
10. `python game_loop.py next` -- **do what it says**. Warns if card entries missing or HP inconsistent.
11. **Abilities**: click card -> click targets -> enter result -> `ability_used <pos>` -> `next`
    - WARNING: clicking a card with an unused active ability activates THAT ability instead of selecting as target
12. **Execute**: Dismiss mark menu first by clicking center board `(1280, 690)`, THEN `safe_click btn_execute_sword` -> click target -> screenshot -> `execute <pos> <evil_role|good>`. Execute command auto-prints HP reminder. The mark menu (bottom-right) can overlap `btn_execute_sword` causing template match failure.
13. Repeat `next` until game ends

### End
14. Screenshot end screen. Read true evils + check `<Corrupted>` tags. **Note: game auto-advances to next village after "Next" click — screenshot BEFORE clicking Next if you need end-screen details.** End-screen dialogs: "Next" at ~(1280, 865), "Continue" (score summary) at ~(1280, 950). Ascension-complete dialogs may need the same y range.
15. `python game_loop.py game_over win/loss <name> "<pos=Role,...>" "[notes]"` -- auto-saves test, runs single replay, runs full v2 regression, prints commit checklist.
16. Follow the printed checklist: commit, push, analyze loss if applicable.

## Memory Reader -- Continuous Validation
Reads game state from process memory (`memory_reader.py`). Goal: replace the visual pipeline entirely.

Every screenshot, memory reader reads state and compares against what the screenshot shows. Screenshot is ground truth. **Any mismatch = stop the game, diagnose, fix, verify, resume.**

**Notes**:
- Multi-village: FIXED. Uses Unity native object name at `m_CachedPtr(0x10)+0x48` (always correct) with `characterId` fallback.
- Player.log: `%LOCALAPPDATA%Low/UmiArt/Demon Bluff/Player.log` -- INIT entries have true roles in reverse position order.
- Name mappings: Gambler->Gemcrafter, Imp->Chancellor, etc. (see DISPLAY_NAMES dict).

## Setup
- Screen: 2560x1440, Python 3.13
- Dependencies: `mss`, `pyautogui`, `Pillow`
- `game_loop.py` (CLI/session), `solver.py` (constraints), `strategy.py` (action selection), `knowledge_base.py` (roles), `screenshot.py`, `mouse.py`, `card_vision.py`
- **Test directories**: `tests/cases_v2/` (new, card_vision pipeline), `tests/cases/` (legacy, manual entry). Solver work validates against v2 only.

## Game Overview
Puzzle/deduction game. Circle of face-down cards -- reveal for role info, deduce Evil, execute them. Evil disguises as Villagers and lies. Good can become corrupted (unreliable info). Win by executing all Evil before HP runs out.
