# Demon Bluff Solver

A constraint-satisfaction solver for [Demon Bluff](https://store.steampowered.com/app/2568800/Demon_Bluff/) (Steam), a puzzle/deduction game where you reveal character cards, interpret their info, and deduce which characters are Evil.

Built to be played entirely by an AI agent (Claude) — from clicking cards and reading the screen to entering data, running the solver, and executing decisions. No human reasoning during gameplay; the solver handles all deduction.

## Current Stats

**461 games played** — 92% win rate (426W / 35L), 218 perfect games (10 HP)

Tested through **Ascension 68** with 9-card boards, up to 4 evils, corruption, extra role pools, Lilis night kills, Witch card-blocking, Puppeteer/Puppet mechanics, and Shaman Baker-conversion chains.

## How It Works

The solver enumerates all possible evil placements across the board and filters them against observed information (role claims, ability results, corruption status, etc.). It then uses Shannon entropy to recommend the highest-value action — which card to reveal, which ability to activate, or which character to execute.

The full pipeline:

1. **Screen capture** — `screenshot.py` grabs the game at 2560×1440
2. **Card vision** — `card_vision.py` classifies card roles from screenshots using OpenCV template matching against a compendium library
3. **Memory validation** — `memory_reader.py` reads live game state via IL2CPP (GameAssembly.dll) and cross-checks against what the screenshot shows
4. **Data entry** — `game_loop.py` CLI feeds card info into the solver; `auto_card` auto-enters parseable cards from memory
5. **Constraint solving** — Rust solver (`crates/solver-core`) filters evil placement scenarios against all observed info; Python `solver.py` is kept as a cross-check reference
6. **Action selection** — `strategy.py` picks the highest-entropy action; `next` auto-executes definite-evil and high-confidence lookahead picks
7. **Execution** — `template_match.py` safe-clicks UI elements to carry out the action

## Key Components

### Solver & Strategy

| File | Purpose |
|------|---------|
| `crates/solver-core/` | **Primary solver** — Rust constraint-satisfaction engine (Rayon-parallelized, ~2.5× Python speed) |
| `crates/solver-cli/` | Rust CLI binary (`demon-bluff-solver.exe`) — reads GameState JSON from stdin, writes SolverResult JSON to stdout |
| `rust_solver.py` | Python bridge to the Rust solver via subprocess |
| `solver.py` | Python solver — stripped to types/helpers only (~300 lines); all solve logic in Rust |
| `strategy.py` | Shannon entropy-based action recommender with execution lookahead and auto-execute |
| `game_loop.py` | CLI interface for game sessions, data entry, and solver interaction (REPL mode: `repl`) |
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
| `tests/cases_v2/` | 307 test cases — card vision pipeline, high accuracy |
| `tests/cases/` | 137 legacy test cases — manual data entry |
| `tests/simulation.rs` | Rust simulation test — constraint validation + strategy-driven execution on all v2 cases |
| `tests/test_replay.py` | Step-by-step replay validation (reveals → abilities → executions) |

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
- Puppeteer/Puppet mechanics — Puppet is evil but truthful, auto-generated from adjacent Villager
- Flip verification via memory reader — detects click failures before they become misdiagnosed blocks
- Persistent daemon mode for the Rust solver — keeps binary alive across calls for faster response

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
- Rust 2021 edition (`cargo build --release` to build the solver)

## Usage

```bash
# Build the Rust solver
cargo build --release

# Start a new game session
python game_loop.py start

# REPL mode (persistent process, no import overhead)
python game_loop.py repl

# Core commands:
#   new <n_cards> <n_evil>           — start a new puzzle
#   deck V=... O=... M=... D=...    — set the role pool (prefixes required)
#   flip                             — flip all cards in order (with auto-verification)
#   auto_card                        — auto-enter parseable cards from memory reader
#   card <role> <pos> <info>         — enter a revealed card
#   pd_check <pd_pos> <target> ...   — enter Plague Doctor ability result
#   next                             — run solver + auto-execute recommended action
#   next --plan                      — print recommendation without executing
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
cargo test --release --test simulation  # Rust simulation tests (307 v2 cases)
python -m tests.test_replay --v2-only  # Python v2 replay tests
python -m tests.test_replay            # all test cases

# View stats
python scorecard.py
```
