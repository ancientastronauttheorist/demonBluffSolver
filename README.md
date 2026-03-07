# Demon Bluff Solver

A constraint-satisfaction solver for [Demon Bluff](https://store.steampowered.com/app/2568800/Demon_Bluff/) (Steam), a puzzle/deduction game where you reveal character cards, interpret their info, and deduce which characters are Evil.

## How It Works

The solver enumerates all possible evil placements across the board and filters them against observed information (role claims, ability results, corruption status, etc.). It then uses Shannon entropy to recommend the highest-value action — which card to reveal, which ability to activate, or which character to execute.

### Key Components

| File | Purpose |
|------|---------|
| `solver.py` | Constraint-satisfaction engine — generates and filters evil placement scenarios |
| `strategy.py` | Shannon entropy-based action recommender |
| `game_loop.py` | CLI interface for entering game state and running the solver |
| `knowledge_base.py` | Card role database (abilities, types, factions) |
| `game_utils.py` | Board geometry, coordinate helpers |
| `scorecard.py` | Win/loss tracking and stats |

### Automation Tools

| File | Purpose |
|------|---------|
| `screenshot.py` | Screen capture (mss) |
| `mouse.py` | Mouse control (pyautogui) |
| `template_match.py` | Template matching for UI element detection + safe clicking |
| `ui.py` | UI interaction helpers |

## Game Mechanics

- A circle of face-down character cards, each with a hidden role
- Reveal cards to learn their role and get info from their speech bubble
- Evil characters **Disguise** (appear as a Villager role) and **Lie** (give false info)
- Some cards get **Corrupted** by evil, making good characters give false info too
- Win by executing all Evil characters before running out of HP
- Wrong executions cost HP (5 at higher ascensions)

## Solver Features

- Full constraint-satisfaction over all possible evil placements
- Handles disguises, lying, corruption, and role-specific validation
- Role-specific ability modeling: Slayer, Judge, PD, Dreamer, Baker, Druid, Architect, Bard, and more
- Bombardier protection (instant loss if wrongly executed)
- Ascension 10+ pool-vs-board role count validation
- Lilis night-kill tracking

## Requirements

- Python 3.13+
- `mss`, `pyautogui`, `Pillow`

## Usage

```bash
# Start a new game session
python game_loop.py

# Commands (inside game_loop):
#   new <n_cards> <n_evil>    — start a new puzzle
#   deck V=... O=... M=... D=...  — set the role pool
#   card <role> <pos> <info>  — enter a revealed card
#   execute <pos>             — execute a character
#   solve                    — run the solver
#   next                     — get recommended action
#   game_over win/loss ...   — record result

# View stats
python scorecard.py
```

## Current Stats

**16 games played** — 81% win rate (13W / 3L), 5 perfect games (10 HP)

Tested through Ascension 10 with increasingly complex boards (corruption, extra role pools, Lilis night kills).
