# Demon Bluff Solver

## Current Goal
The project already has a working solver. The current goal is to make the full loop reliable enough that the game can be solved from live screenshots with minimal manual help.

Priority order:
1. Improve vision and automation so deck state, flipped cards, dead seats, and ability results are captured accurately.
2. Use live runs to find rule gaps, bad heuristics, and incorrect assumptions in the solver.
3. Convert each real-game discovery into code changes, regression cases, and better tooling.
4. Push toward a standalone Python workflow that can read the board, reason correctly, and act without a model in the loop.

## Autonomous Workflow
Codex should operate in this cycle:
1. Capture the current puzzle state from screenshots, hover text, compendium pages, and other in-game ground truth.
2. Enter or infer that state into the solver, then choose the best next action.
3. Act in game, observe the result, and update the state.
4. After each puzzle, log what happened and save regression coverage for anything learned.
5. Refine solver logic, strategy, and vision tooling whenever the live game exposes a weakness.
6. At the end of each completed loop, commit the resulting code, test, and regression updates locally.

## Interaction Rules
- Use the mouse only for in-game interaction. Do not use keyboard shortcuts during live runs.
- To open the current deck, click the purple card icon in the top-right corner instead of using `Tab`.
- Prefer hover-and-screenshot verification before committing to a click when a UI target is ambiguous.
- Do not issue state-mutating `game_loop.py` commands in parallel. The session lock prevents file corruption, but it does not preserve command ordering. For `new`, `deck`, `card`, `execute`, `ability_used`, `slayer_result`, and similar updates, use one serialized command stream or a single in-process script.

## Mouse Parking
- Before taking screenshots, park the mouse in a neutral area so hover tooltips do not block card text or UI.
- Preferred parking spots on a 2560x1440 screen:
  - Board-center void: around `(1280, 690)` when no modal is open. This keeps the cursor off cards and side UI.
  - Lower-right dark margin: around `(2400, 1300)` when no settings tooltip is open.
- Avoid parking on:
  - the left status panel
  - the purple deck icon / top-right markers
  - the eye button on the right
  - the execute/settings cluster in the lower-right
  - any revealed or facedown card
- If a screenshot includes a hover panel, move to a parking spot and immediately retake it instead of trying to read through the obstruction.

## Setup
- Screen: 2560x1440
- Python 3.13
- Dependencies: `mss`, `pyautogui`, `Pillow`
- [screenshot.py](/C:/Users/BMO/Documents/code/Codex/DBclone/screenshot.py): capture screenshots
- [mouse.py](/C:/Users/BMO/Documents/code/Codex/DBclone/mouse.py): mouse control
- [card_vision.py](/C:/Users/BMO/Documents/code/Codex/DBclone/card_vision.py): card detection and template-based recognition
- [game_loop.py](/C:/Users/BMO/Documents/code/Codex/DBclone/game_loop.py): CLI/session bridge for solver state
- [solver.py](/C:/Users/BMO/Documents/code/Codex/DBclone/solver.py): constraint solver
- [strategy.py](/C:/Users/BMO/Documents/code/Codex/DBclone/strategy.py): action selection
- [knowledge_base.py](/C:/Users/BMO/Documents/code/Codex/DBclone/knowledge_base.py): card role database

## Game Overview
- Demon Bluff is a puzzle and deduction game with a circle of face-down character cards.
- Reveal cards to get role info, deduce which characters are Evil, and execute them.
- Evil characters disguise themselves as Villagers and can lie.
- Good characters can also become corrupted, which makes their info unreliable without changing their apparent role.
- Win by executing all Evil characters before running out of health.
