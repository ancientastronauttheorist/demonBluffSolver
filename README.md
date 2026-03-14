# Demon Bluff Solver

A constraint-satisfaction solver for [Demon Bluff](https://store.steampowered.com/app/2568800/Demon_Bluff/) (Steam), a puzzle/deduction game where you reveal character cards, interpret their info, and deduce which characters are Evil.

Built to be played entirely by an AI agent (Claude) — from clicking cards and reading the screen to entering data, running the solver, and executing decisions. No human reasoning during gameplay; the solver handles all deduction.

## Current Stats

**240 games played** — 92% win rate (222W / 18L), 104 perfect games (10 HP)

Tested through **Ascension 40** with 10-card boards, 4 evils, corruption, extra role pools, Lilis night kills, and Witch card-blocking.

## How It Works

The solver enumerates all possible evil placements across the board and filters them against observed information (role claims, ability results, corruption status, etc.). It then uses Shannon entropy to recommend the highest-value action — which card to reveal, which ability to activate, or which character to execute.

The full pipeline:

1. **Screen capture** — `screenshot.py` grabs the game at 2560×1440
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
| `strategy.py` | Shannon entropy-based action recommender with execution lookahead |
| `game_loop.py` | CLI interface for game sessions, data entry, and solver interaction |
| `knowledge_base.py` | Card role database — 30+ roles with abilities, types, factions |
| `game_utils.py` | Board geometry, coordinate helpers |
| `scorecard.py` | Win/loss tracking and stats |

### Vision & Validation

| File | Purpose |
|------|---------|
| `card_vision.py` | OpenCV card classification from screenshots (compendium template matching) |
| `memory_reader.py` | IL2CPP memory reader — reads live game state from process memory for cross-validation |
| `replay_analysis.py` | Post-game scenario-narrowing analysis for test cases |

### Automation

| File | Purpose |
|------|---------|
| `screenshot.py` | Screen capture via mss |
| `mouse.py` | Mouse control via pyautogui |
| `template_match.py` | Template matching for UI elements + safe clicking with focus verification |

### Testing

| Directory | Purpose |
|-----------|---------|
| `tests/cases_v2/` | 102 test cases — card vision pipeline, high accuracy |
| `tests/cases/` | 138 legacy test cases — manual data entry |
| `tests/test_replay.py` | Step-by-step replay validation (reveals → abilities → executions) |
| `tests/test_regression.py` | Full regression suite |

## Solver Features

- Full constraint-satisfaction over all possible evil placements
- Handles disguises, lying, corruption, and role-specific validation
- 30+ role abilities modeled: Slayer, Judge, Plague Doctor, Dreamer, Baker, Druid, Architect, Bard, Confessor, Poet, Knight, Bombardier, Doppelganger, and more
- Bombardier protection (instant loss if wrongly executed)
- Execution lookahead with HP-aware decision making
- Ascension 10+ pool-vs-board role count validation
- Lilis night-kill tracking and Witch card-blocking mechanics
- Drunk execution cost modeling (2 HP vs 5 HP)
- Baker conversion chain validation with reveal-order tracking
- Shaman role duplication handling
- Flip verification via memory reader — detects click failures before they become misdiagnosed blocks

## Game Mechanics

- A circle of face-down character cards, each with a hidden role
- Reveal cards to learn their role and get info from their speech bubble
- Evil characters **disguise** (appear as a Villager role) and **lie** (give false info)
- Some cards get **corrupted** by evil, making good characters give false info too
- Win by executing all Evil characters before running out of HP
- Wrong executions cost HP (5 at Ascension 4+)
- At Ascension 10+, the role pool is larger than the board — not all roles are in play

## Requirements

- Python 3.13+
- `mss`, `pyautogui`, `Pillow`, `opencv-python`

## Usage

```bash
# Start a new game session
python game_loop.py start

# Core commands:
#   new <n_cards> <n_evil>           — start a new puzzle
#   deck V=... O=... M=... D=...    — set the role pool
#   flip                             — flip all cards (with auto-verification)
#   card <role> <pos> <info>         — enter a revealed card
#   next                             — run solver + get recommended action
#   execute <pos> <role|good>        — execute a character
#   set_hp <hp> <wrong_exec_cost>    — update HP
#   game_over win/loss ...           — record result + save test case

# Card vision (classify roles from screenshot)
python card_vision.py classify_dirs <screenshot> --context deck \
  --library-dir templates/compendium/page1 \
  --library-dir templates/compendium/page3

# Memory reader (validate game state)
python memory_reader.py         # read board state
python memory_reader.py --deck  # read deck pool

# Run tests
python -m tests.test_replay --v2-only  # 102 v2 replay tests
python -m tests.test_replay            # all test cases

# View stats
python scorecard.py
```
