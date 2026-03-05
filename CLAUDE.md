# Demon Bluff Solver

## Project Goal
Build a fully automated solver for Demon Bluff (Steam). Two-phase approach:

1. **Learn phase**: Claude autonomously plays the game via screenshot + mouse control, learning mechanics, card interactions, and deduction patterns through hands-on experience
2. **Build phase**: Use everything learned to write a standalone Python solver script that can solve Demon Bluff puzzles algorithmically — no Claude in the loop

## Autonomous Workflow
Claude runs continuously in this cycle:
1. **Play** — screenshot → analyze board → reason about deductions → click to act
2. **Learn** — after each puzzle, record new insights in memory files
3. **Build/Refine** — periodically update solver code with deduction logic learned from playing
4. **Repeat** — start next puzzle, keep going

## Setup
- Screen: 2560x1440
- Python 3.13, deps: mss, pyautogui, Pillow
- `screenshot.py` — capture screenshots
- `mouse.py` — mouse control (click, move)
- `knowledge_base.py` — card role database

## Game Overview
- Puzzle/deduction game: circle of face-down character cards
- Reveal cards to get role info, deduce which are Evil, execute them
- Evil characters Disguise (appear as Villagers) and Lie (false info)
- Win by executing all Evil characters before running out of health
