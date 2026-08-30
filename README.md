# Demon Bluff Solver

A constraint-satisfaction solver for [Demon Bluff](https://store.steampowered.com/app/2568800/Demon_Bluff/) (Steam), a puzzle/deduction game where you reveal character cards, interpret their info, and deduce which characters are Evil.

Built to be played by an AI coding/automation agent from clicking cards and reading the screen to entering data, running the solver, and executing decisions. No human reasoning during gameplay; the solver handles all deduction.

## Current Stats

**592 games played** - 92% win rate (545W / 47L), 272 perfect games (10 HP)

Tested through **Ascension 84** with up to 10-card boards, 4+ evils, corruption, extra role pools, Lilis night kills, Witch card-blocking, Puppeteer/Puppet mechanics, Shaman Baker-conversion chains, Doppelganger disguises, Baa hidden-deck reveal handling, Rambler shut-up constraints, Alchemist corrupted-count clues, and public Dreamer role-pair output.

## How It Works

The solver enumerates all possible evil placements across the board and filters them against observed information (role claims, ability results, corruption status, etc.). It then uses Shannon entropy to recommend the highest-value action: which card to reveal, which ability to activate, or which character to execute.

The full pipeline:

1. **Screen capture** - `screenshot.py` grabs the game at 2560x1440
2. **Card vision** - `card_vision.py` classifies card roles from screenshots using OpenCV template matching against a compendium library
3. **Memory validation** - `memory_reader.py` reads live game state via IL2CPP (`GameAssembly.dll`) and cross-checks against what the screenshot shows
4. **Data entry** - `game_loop.py` CLI feeds card info into the solver; `auto_card` auto-enters parseable cards from memory
5. **Constraint solving** - Rust solver (`crates/solver-core`) filters evil placement scenarios against all observed info; Python `solver.py` is kept as a cross-check reference
6. **Action selection** - `strategy.py` picks the highest-entropy action; `next` auto-uses supported active abilities and auto-executes definite-evil or high-confidence lookahead picks
7. **Execution** - `template_match.py` safe-clicks UI elements to carry out the action

## Key Components

### Solver & Strategy

| File | Purpose |
|------|---------|
| `crates/solver-core/` | **Primary solver** - Rust constraint-satisfaction engine |
| `crates/solver-cli/` | Rust CLI binary (`demon-bluff-solver.exe`) - reads GameState JSON from stdin, writes SolverResult JSON to stdout |
| `rust_solver.py` | Python bridge to the Rust solver via subprocess or persistent daemon |
| `solver.py` | Python types/helpers and legacy reference logic; active solve path is Rust |
| `strategy.py` | Shannon entropy-based action recommender with execution lookahead and auto-execute |
| `game_loop.py` | CLI interface for game sessions, data entry, solver interaction, and live automation |
| `state_machine.py` | Higher-level live automation state machine |
| `knowledge_base.py` | Card role database with abilities, types, factions, and special flags |
| `game_utils.py` | Board geometry and coordinate helpers |
| `scorecard.py` | Win/loss tracking and stats |

### Vision & Validation

| File | Purpose |
|------|---------|
| `card_vision.py` | OpenCV card classification from screenshots |
| `memory_reader.py` | IL2CPP memory reader for live deck, board, score, clue, and ability-state validation |
| `replay_analysis.py` | Post-game scenario-narrowing analysis for saved test cases |

### Automation

| File | Purpose |
|------|---------|
| `screenshot.py` | Screen capture via `mss` |
| `mouse.py` | Mouse control via `pyautogui` |
| `template_match.py` | Template matching for UI elements plus safe clicking with focus verification |

### Testing

| Directory | Purpose |
|-----------|---------|
| `tests/cases_v2/` | 426 active live-run test cases |
| `tests/cases/` | 137 legacy reference cases |
| `crates/solver-core/tests/simulation.rs` | Rust simulation test: constraint validation plus strategy-driven execution on all v2 cases |
| `tests/test_*.py` | Python unit tests for card vision, parsers, UI helpers, state I/O, and live-loop behavior |

## Solver Features

- Full constraint-satisfaction over all possible evil placements
- Handles disguises, lying, corruption, and role-specific validation
- 30+ role abilities modeled: Slayer, Judge, Plague Doctor, Dreamer, Baker, Druid, Architect, Bard, Confessor, Poet, Knight, Bombardier, Doppelganger, and more
- Current patch support: Rambler adjacent-truthful shut-up behavior, Alchemist corruption-count clues with immunity, Baa hidden-deck reveal after death, and public Dreamer two-target automation
- Bombardier protection (instant loss if wrongly executed)
- Execution lookahead with HP-aware decision making
- Ascension 10+ pool-vs-board role count validation
- Lilis night-kill tracking and Witch card-blocking mechanics
- Drunk execution cost modeling and current Drunk corruption-status nuance
- Baker conversion chain validation
- Shaman role duplication handling
- Puppeteer/Puppet mechanics: Puppet is evil but truthful, auto-generated from adjacent Villager
- Flip verification via memory reader, including first-click recovery for multi-card flips
- Persistent daemon mode for the Rust solver, keeping the binary alive across calls for faster response

## Current Patch Notes

The live build as of 2026-05-05 changed several solver-relevant rules:

- **Rambler:** adjacent truthful characters say `#X shut up!` instead of giving normal info. The old "picked by a liar silences Rambler" mechanic is obsolete.
- **Alchemist:** cannot be corrupted and now reports how many corrupted characters were around them at the start of the round, before cure. Zero wording such as `NO one was Corrupted around me` is parsed as `corrupted_count: 0`.
- **Baa:** hides an Outcast in deck view; when Baa dies, the hidden deck card should reveal and `_baa_post_execute_reveal()` checks for it.
- **Dreamer:** the public asset picks exactly two targets and immediately returns a role pair like `Among #X, #Y there is: RoleA or RoleB`, or a Wretch/Cabbage result. `next` can auto-fire it when the solver recommends two targets, while still refusing targets with unused active abilities. The separate managed `Dreamer2` class is not bound by current gameplay assets.
- **Deckbuilding mode:** experimental and not the primary supported live loop yet; current automation is focused on Standard ascension play.

## Game Mechanics

- A circle of face-down character cards, each with a hidden role
- Reveal cards to learn their role and get info from their speech bubble
- Evil characters **disguise** as non-evil roles and **lie** when giving info
- Some cards get **corrupted** by evil, making good characters give false info too
- Win by executing all Evil characters before running out of HP
- Wrong executions cost HP, usually 5 at high ascension
- At Ascension 10+, the role pool is larger than the board, so not all roles are in play

## Requirements

- Python 3.13+
- `mss`, `pyautogui`, `Pillow`, `opencv-python`
- Rust 2021 edition (`cargo build --release` to build the solver)

## Usage

```bash
# Build the Rust solver
cargo build --release

# Start a new Standard game session
python game_loop.py start

# REPL mode (persistent process, no import overhead)
python game_loop.py repl

# Core commands:
#   new <n_cards> <n_evil>           - start a new puzzle
#   deck V=... O=... M=... D=...    - set the role pool (prefixes required)
#   flip                             - flip all cards in order (with auto-verification)
#   auto_card                        - auto-enter parseable cards from memory reader
#   card <role> <pos> <info>         - enter a revealed card
#   ability_used <pos>               - mark an active ability as spent
#   pd_check <pd_pos> <target> ...   - enter Plague Doctor ability result
#   slayer_result <pos> <target> ... - enter Slayer ability result
#   next                             - run solver + auto-use/auto-execute recommended action
#   next --plan                      - print recommendation without executing
#   execute <pos> <role|good>        - execute a character
#   set_hp <hp> <wrong_exec_cost>    - update HP
#   game_over win/loss ...           - record result + save test case

# Card vision (classify roles from screenshot)
python card_vision.py classify_dirs <screenshot> --context deck \
  --library-dir templates/compendium/page1 \
  --library-dir templates/compendium/page3

# Memory reader (validate game state)
python memory_reader.py         # read board state
python memory_reader.py --deck  # read deck pool
python memory_reader.py --score # read ascension/run score

# Run tests
cargo test --release -p solver-core        # Rust solver + simulation tests
cargo test --release --test simulation     # Rust simulation tests (426 v2 cases)
python -m unittest discover tests          # Python unit tests

# View stats
python scorecard.py
```
