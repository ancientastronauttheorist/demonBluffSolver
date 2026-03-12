# Demon Bluff Solver

A constraint-satisfaction solver for [Demon Bluff](https://store.steampowered.com/app/2568800/Demon_Bluff/) (Steam), a puzzle/deduction game where you reveal character cards, interpret their info, and deduce which characters are Evil.

Built to be played entirely by an AI agent (Claude) — from clicking cards and reading the screen to entering data, running the solver, and executing decisions. No human reasoning during gameplay; the solver handles all deduction.

## Current Stats

**191 games played** — 92% win rate (175W / 16L), 83 perfect games (10 HP), 24-game win streak

Tested through **Ascension 33** with 10-card boards, 4 evils, corruption, extra role pools, Lilis night kills, and Witch card-blocking.

## How It Works

The solver enumerates all possible evil placements across the board and filters them against observed information (role claims, ability results, corruption status, etc.). It then uses Shannon entropy to recommend the highest-value action — which card to reveal, which ability to activate, or which character to execute.

The full pipeline:

1. **Screen capture** — `screenshot.py` grabs the game at 2560x1440
2. **Card vision** — `card_vision.py` classifies card roles from screenshots using OpenCV template matching against a compendium library
3. **Memory validation** — `memory_reader.py` reads live game state via IL2CPP (GameAssembly.dll) and cross-checks against what the screenshot shows
4. **Data entry** — `game_loop.py` CLI feeds card info into the solver
5. **Constraint solving** — `solver.py` filters evil placement scenarios against all observed info
6. **Action selection** — `strategy.py` picks the highest-entropy action
7. **Execution** — `template_match.py` safe-clicks UI elements to carry out the action

## Key Components

### Solver & Strategy

| File | Purpose |
|------|---------|
| `solver.py` | Constraint-satisfaction engine — generates and filters evil placement scenarios |
| `strategy.py` | Shannon entropy-based action recommender |
| `game_loop.py` | CLI interface for entering game state and running the solver |
| `knowledge_base.py` | Card role database — 30+ roles with abilities, types, factions |
| `game_utils.py` | Board geometry, coordinate helpers |
| `scorecard.py` | Win/loss tracking and stats |

### Vision & Validation

| File | Purpose |
|------|---------|
| `card_vision.py` | OpenCV card classification from screenshots (compendium template matching) |
| `memory_reader.py` | IL2CPP memory reader — reads live game state from process memory |
| `replay_analysis.py` | Post-game scenario-narrowing analysis for test cases |

### Automation

| File | Purpose |
|------|---------|
| `screenshot.py` | Screen capture via mss |
| `mouse.py` | Mouse control via pyautogui |
| `template_match.py` | Template matching for UI elements + safe clicking |
| `ui.py` | UI interaction helpers |

### Testing

| Directory | Purpose |
|-----------|---------|
| `tests/cases_v2/` | 53 test cases — card_vision pipeline, high accuracy |
| `tests/cases/` | 138 legacy test cases — manual data entry |
| `tests/test_replay.py` | Step-by-step replay validation against saved games |
| `tests/test_regression.py` | Full regression suite |

## Game Mechanics

- A circle of face-down character cards, each with a hidden role
- Reveal cards to learn their role and get info from their speech bubble
- Evil characters **disguise** (appear as a Villager role) and **lie** (give false info)
- Some cards get **corrupted** by evil, making good characters give false info too
- Win by executing all Evil characters before running out of HP
- Wrong executions cost HP (5 at Ascension 4+)
- At Ascension 10+, the role pool is larger than the board — not all roles are in play

## Solver Features

- Full constraint-satisfaction over all possible evil placements
- Handles disguises, lying, corruption, and role-specific validation
- 30+ role abilities modeled: Slayer, Judge, PD, Dreamer, Baker, Druid, Architect, Bard, Confessor, Poet, Knight, Bombardier, and more
- Bombardier protection (instant loss if wrongly executed)
- Ascension 10+ pool-vs-board role count validation
- Lilis night-kill tracking
- Witch card-blocking mechanics
- Forced-execution lookahead with HP-aware decision making
- Drunk execution cost modeling (2 HP vs 5 HP)

## Requirements

- Python 3.13+
- `mss`, `pyautogui`, `Pillow`, `opencv-python`

## Usage

```bash
# Start a new game session
python game_loop.py

# Commands:
#   new <n_cards> <n_evil>           — start a new puzzle
#   deck V=... O=... M=... D=...    — set the role pool
#   card <role> <pos> <info>         — enter a revealed card
#   execute <pos> <role|good>        — execute a character
#   ability_used <pos>               — mark ability as used
#   set_hp <hp> <wrong_exec_cost>   — update HP
#   solve                            — run the solver
#   next                             — get recommended action
#   game_over win/loss ...           — record result + save test case

# Card vision (classify roles from screenshot)
python card_vision.py classify_dirs <screenshot> --context deck \
  --library-dir templates/compendium/page1 \
  --library-dir templates/compendium/page3

# Memory reader (validate game state)
python memory_reader.py         # read board state
python memory_reader.py --deck  # read deck pool

# Run tests
python -m tests.test_replay --v2-only  # v2 test cases only
python -m tests.test_replay            # all test cases

# View stats
python scorecard.py
```
