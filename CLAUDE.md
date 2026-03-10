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

## Screen & Coordinates
- **Resolution**: 2560x1440
- **Mouse parking** (before screenshots): `(1280, 690)` — board center void. Avoid cards, panels, buttons, deck icon.
- If a screenshot has a hover tooltip, park and retake.
- `safe_click` auto-focuses the game window. **Prefer `safe_click` over manual move+click.**
- For card clicks, prefer detected card-box centers from the current screenshot. Circle formula is a fallback only.

## Game Loop

### Start
1. `safe_click menu_play_demo` → `safe_click mode_standard` → dismiss intro with `safe_click btn_close_dialog`
2. Deck auto-opens. **Park mouse to bottom-left corner** `(50, 1350)` before screenshotting to avoid hover tooltips. Take screenshot, then read the deck with **both** card_vision and memory reader:
   ```
   python card_vision.py classify_dirs <screenshot> --context deck --library-dir templates/compendium/page1 --library-dir templates/compendium/page3 --library-dir templates/compendium/page4 --library-dir templates/compendium/page5
   python memory_reader.py --deck
   ```
   **Cross-check both outputs — they must match.** Any mismatch = stop and fix. Use card_vision + memory reader as ground truth for the `deck` command. Also read board counts (V, O, M, D icons top-right) from screenshot.
3. `python game_loop.py new <n_cards> <n_evil>`
4. `python game_loop.py deck V=... O=... M=... D=... nv=<villager_count> no=<outcast_count>` — include ALL pool roles, prefixes REQUIRED
5. Close deck by clicking neutral area

### Reveal & Enter
6. Click all cards #1→#N (use `detect_card_positions` or `all_game_card_coords(n)`). Verify: `python template_match.py find_all card_facedown`
7. After all cards flipped, screenshot and run `python memory_reader.py` to cross-check roles match what the screenshot shows. **Mismatch = stop and fix.** **HONOR RULE: memory reader shows true evil roles — DO NOT use this to cheat. We are honorable like Finn. The solver must solve it from the card info alone. Memory reader is for validation only (verifying data entry accuracy, post-game analysis).**
8. Enter card info. Active abilities (lightning bolt icon): `card no_info <pos> <Role>` until used. Passive info: enter immediately.
9. `set_hp <hp> <wrong_exec_cost>` at game start (5 at Asc4+)

### Solve & Act
10. `python game_loop.py next` — **do what it says**
11. **Abilities**: click card → click targets → enter result → `ability_used <pos>` → `next`
    - WARNING: clicking a card with an unused active ability activates THAT ability instead of selecting as target
12. **Execute**: `safe_click btn_execute_sword` FIRST → click target → screenshot → `execute <pos> <evil_role|good>` → `set_hp`
13. Repeat `next` until game ends

### End
14. Screenshot end screen. Read true evils + check `<Corrupted>` tags
15. `python game_loop.py game_over win/loss <name> "<pos=Role,...>" "[notes]"` — saves to `tests/cases_v2/`, runs step-by-step replay test automatically
16. `python -m tests.test_replay --v2-only` — full replay regression on all v2 tests
17. Commit and push

## Memory Reader — Continuous Validation
Reads game state from process memory (`memory_reader.py`). Goal: replace the visual pipeline entirely.

Every screenshot, memory reader reads state and compares against what the screenshot shows. Screenshot is ground truth. **Any mismatch = stop the game, diagnose, fix, verify, resume.**

**Known issues**:
- Multi-village: `dataRef` (0x50) not updated. `chName.m_text` IS updated. Fix: read chName, parse Player.log, or find current CharacterData pointer.
- Player.log: `%LOCALAPPDATA%Low/UmiArt/Demon Bluff/Player.log` — INIT entries have true roles in reverse position order.
- Name mappings: Gambler→Gemcrafter, Imp→Chancellor, etc. (see DISPLAY_NAMES dict).

## Setup
- Screen: 2560x1440, Python 3.13
- Dependencies: `mss`, `pyautogui`, `Pillow`
- `game_loop.py` (CLI/session), `solver.py` (constraints), `strategy.py` (action selection), `knowledge_base.py` (roles), `screenshot.py`, `mouse.py`, `card_vision.py`
- **Test directories**: `tests/cases_v2/` (new, card_vision pipeline), `tests/cases/` (legacy, manual entry). Solver work validates against v2 only.

## Game Overview
Puzzle/deduction game. Circle of face-down cards — reveal for role info, deduce Evil, execute them. Evil disguises as Villagers and lies. Good can become corrupted (unreliable info). Win by executing all Evil before HP runs out.
