# Demon Bluff Solver

A constraint-satisfaction solver for [Demon Bluff](https://store.steampowered.com/app/2568800/Demon_Bluff/) (Steam), a puzzle/deduction game where you reveal character cards, interpret their info, and deduce which characters are Evil.

Built to be played by an AI coding/automation agent from clicking cards and reading the screen to entering data, running the solver, and executing decisions. No human reasoning during gameplay; the solver handles all deduction.

## Current Stats

**592 games played** - 92% win rate (545W / 47L), 272 perfect games (10 HP)

Tested through **Ascension 84** with up to 10-card boards, 4+ evils, corruption, extra role pools, Lilis night kills, Witch card-blocking, Puppeteer/Puppet mechanics, native-traced Shaman role overwrites, Doppelganger disguises, native-verified Baa deck-view hiding, Rambler shut-up constraints, Alchemist corrupted-count clues, and public Dreamer role-pair output.

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
- Current patch support: Rambler adjacent-truthful shut-up behavior, Alchemist corruption-count clues with immunity, native-verified Baa deck-view reveal on death, ordered Shaman/Twin identity traces, public Dreamer two-target automation, the native-current Poet selector, and exact Confessor, Bard, Gemcrafter, Empress, Bishop, Enlightened, Medium, Knitter, Bounty Hunter, Lover, Scout, Oracle, and Hunter clue contracts
- Bombardier terminal protection (any qualifying non-Lilis death, including Slayer and current-role overwrites)
- Execution lookahead with HP-aware decision making
- Ascension 10+ pool-vs-board role count validation
- Lilis night-kill tracking and Witch card-blocking mechanics
- Drunk execution cost modeling and current Drunk corruption-status nuance
- Baker conversion chain validation
- Shaman one-way role overwrite modeling, including both resistance-aware Witness marker attempts and copied-Alchemist Start timing
- Puppeteer/Puppet mechanics: Puppet is evil but truthful, auto-generated from adjacent Villager
- Flip verification via memory reader, including first-click recovery for multi-card flips
- Persistent daemon mode for the Rust solver, keeping the binary alive across calls for faster response

## Current Patch Notes

The live build as of 2026-05-05 changed several solver-relevant rules:

- **Rambler:** adjacent truthful characters say `#X shut up!` instead of giving normal info. The old "picked by a liar silences Rambler" mechanic is obsolete.
- **Alchemist:** cannot be corrupted and now reports how many corrupted characters were around them at the start of the round, before cure. Zero wording such as `NO one was Corrupted around me` is parsed as `corrupted_count: 0`.
- **Baa:** managed `Imp` hides one existing Outcast identity in deck view at Start. Any Baa death reveals that deck-strip identity; no board card is flipped and the HUD counts are unchanged.
- **Shaman:** managed `Illuzionist` selects an ordered apparent-Villager pair after Plague Doctor and before Alchemist. The source stays put; the destination is overwritten and immediately fires the copied role's Start action with its preserved status/runtime state. Shaman independently attempts `MessedUpByEvil` on both endpoints, so resistance can suppress either marker. The solver models the ordered identity trace and copied-Alchemist timing; copied Baker/runtime-data composition remains deliberately opaque. Managed `Cipher` is Witch, not Shaman.
- **Plague Doctor:** managed `Puzzlemaster` checks raw active Corrupted status. Its active picker includes self and dead cards; self always displays clean. Corrupted truth results uniformly reveal registered/runtime Evil (including Wretch), while Bluff inverts the status and uniformly names Good as Evil. The strategy scores this stochastic output instead of inventing a lowest-position reveal.
- **Dreamer:** the public asset picks exactly two targets and immediately returns a role pair like `Among #X, #Y there is: RoleA or RoleB`, or a Wretch/Cabbage result. `next` can auto-fire it when the solver recommends two targets, while still refusing targets with unused active abilities. The separate managed `Dreamer2` class is not bound by current gameplay assets.
- **Poet:** the public asset binds managed `Gossip`, whose exact provider pool is Lover, Scout, Oracle, Bounty Hunter, Medium, Knitter, Hunter, Enlightened, Empress, Bishop, Gemcrafter, and Bard. Every truthful or bluff result makes a fresh provider draw. Current parsing stamps a strict provider-specific provenance payload, while unmarked archived cases keep their legacy interpretation.
- **Scout / Hunter:** Scout selects an actual runtime-Evil occurrence, reports its nearest other registered Evil, uses an explicit one-Evil sentinel, and can lie only with distance 1–3 while retaining a selected identity. Public Hunter binds managed `Tracker`; truth is the nearest other registered Evil or exactly `N-1`, and lies come from the remaining half-circle distances. Hunter ingestion verifies its ordered two-reference event, including duplicate opposite seats.
- **Oracle:** public Oracle binds managed `Investigator`. Truth independently samples one current registered Minion and one current registered-Good character, so a moved Twin Minion can produce a duplicate same-card reference; an empty Minion pool emits exactly `There are no minions`. Bluff samples two distinct registered-Good characters and labels them with a Minion from the script pool, falling back to the all-ascension pool. Direct and Poet observations share the same exact current schema.
- **Lover:** public Lover binds managed `Empath`. Truth counts the previous and next physical occurrences whose registered alignment is Evil and stores those exact references, including duplicate self/other entries on one- and two-card boards. Bluff removes the actual count from the ascending `0..=min(2, authored Minions + Demons)` pool; Puppet, Wretch, and runtime-added Evils can change truth without enlarging that authored domain. Direct and Poet/Lover observations share exact text, reference, and provenance checks.
- **Bounty Hunter:** the shipped public asset binds managed `BountyHunter`, but direct Bounty Hunter is absent from the current Standard/Ascension candidate rosters, so its Start mutation remains deliberately unmodeled in live scenarios. The retained Poet provider is active: truth samples registered Evil, bluff samples registered Good, and both emit exact `#ID\nis Evil` text with no acted references. Current observations share one joint anonymous-Wretch assignment; unmarked archived fixtures retain their legacy predicate.
- **Medium:** public Medium binds managed `Lookout`. Truth samples the complete registered-Good board, excluding itself only when another eligible character exists, and reports that character's register-as-first live identity. Bluff samples non-self characters with a persisted raw bluff, falling back to self only when no other holder exists, and reports the raw bluff identity. Direct and Poet observations enforce the exact two-line `real`/Drunk `actually` wording and one-reference result while unmarked archived fixtures retain their legacy behavior. Raw bluff acquisition order and arbitrary register-as identity remain explicit conservative solver boundaries.
- **Knitter:** public Knitter binds managed `Knitter`. Truth counts circular adjacent pairs whose two physical occurrences register as Evil, without filtering hidden, dead, executed, or corrupted cards; one-card boards retain the self-edge and two-card boards retain both directional edges. Bluff removes the true count from `[0, max(authored Demons + Minions, 2))` and draws one remaining value. Direct and Poet observations enforce exact text, zero references, and current provenance while unmarked fixtures retain their legacy path.
- **Enlightened:** public Enlightened binds managed `Shugenja`. Truth scans the complete physical circle for the nearest registered Evil: increasing public IDs are Counter-clockwise, decreasing IDs are Clockwise, and ties or no Evil are Equidistant. Bluff always chooses one of the two false directions. Direct and Poet observations enforce exact text, zero references, runtime-data agreement, shared anonymous-Wretch worlds, and Baker-to-Spy registration chronology while unmarked fixtures retain their legacy path.
- **Bishop:** public Bishop binds managed `Bishop`. Truth separately samples one live registered Outcast and Villager when those pools exist, then a Minion if present or otherwise a Demon. Bluff samples two live registered Villagers, plus a third when the authored Outcast count is nonzero, while its separately ordered type multiset comes from the authored town/outcast/minion counts. Direct and Poet observations enforce exact one-to-three-reference text and share disguise, anonymous-Wretch, identity-mover, and Baker-to-Spy worlds; unmarked fixtures retain their legacy path.
- **Empress:** public Empress binds managed `Noble`. Truth removes the actor only from the registered-Good pool, samples two distinct registered-Good occurrences plus one registered-Evil occurrence, and therefore permits an Evil-registering actor to name itself. Bluff samples three distinct registered-Good occurrences after actor removal. Both paths sort the three references by public ID and emit exact `One is Evil:` text with matching acted references. Direct and Poet observations share the strict current schema and one anonymous-Wretch/Baker-to-Spy world; unmarked fixtures retain their legacy path, including one archived two-target Poet record that is invalid under current provenance.
- **Gemcrafter:** public Gemcrafter binds managed `Archivist`; the memory reader now recognizes that current managed name while retaining the older `Gambler` alias. Truth samples one live registered-Good occurrence and bluff samples one live registered-Evil occurrence. Each path removes the actor only when its original candidate pool has more than one member, so a sole eligible actor may name itself. Direct and Poet observations enforce exact `#X is Good` text and its single matching reference while sharing anonymous-Wretch and Baker-to-Spy worlds; unmarked archive and Rambler-interruption records retain their legacy paths.
- **Bard:** public Bard binds managed `Acrobat2`, distinct from `Acrobat`; the reader retains the older `RangedEmpath` and `Athlete` aliases. Truth reports the nearest other directly Corrupted physical card and ignores the actor's own Corruption, while bluff draws from fixed native domain `{0,1,2,3}` after removing the truth when present. Direct and Poet observations enforce exact text plus forward/reverse range references, including duplicate opposite seats and empty oversized ranges. The solver also preserves real-role-before-bluff callback order and joins raw-bluff identity with its existing hidden-state search; unmarked archive clues retain their legacy predicate.
- **Confessor:** public Confessor binds managed `Confessor`. Truth and bluff identically say `I am dizzy` when the physical actor is directly Corrupted or registers Evil, except current Spy data always says `I am Good`. The native result has a null reference list, no runtime data, and no RNG; dizzy alone triggers animated art. Direct observations enforce exact text and null-reference provenance, current Poet remains unsupported, and the solver joins current data, raw bluff/register-as identity, callback ordering, anonymous Wretch, identity movers, and Baker-to-Spy chronology while preserving unmarked archive clues.
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
#   pd_check <pd_pos> <target> ...   - recover/manual-enter the public PD result
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
