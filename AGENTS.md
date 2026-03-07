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
