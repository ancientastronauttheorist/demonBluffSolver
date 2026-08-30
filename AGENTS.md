# Demon Bluff Solver Agent Guide

This file is the operating guide for Codex and other coding agents working in
this repository. It is adapted from `CLAUDE.md`, with Claude-specific process
language translated into agent-neutral rules.

## Goal

Primary: harden the Rust solver so it wins consistently at high ascensions.
Fix rule gaps, bad heuristics, edge cases, and incorrect strategy assumptions.

Secondary: keep the live automation loop reliable. `memory_reader.py`,
screenshots, card vision, and `game_loop.py` should agree. Any mismatch means
stop, diagnose, fix, verify, then resume.

## Core Rules

1. Always follow the solver during live play. Execute the solver's top pick,
   even if probabilistic. A wrong answer is a bug to fix between games, not a
   reason to override mid-game.
2. Honor rule: memory reader is validation only. It can cross-check screenshots,
   verify bugs after the game, and help `auto_card` fill metadata. Do not use
   true evil positions from memory to decide executions or patch state until the
   solver lands on the position itself.
3. 0 scenarios means stop. Use the recovery protocol below. Do not guess and do
   not reset unrelated entries to make scenarios appear.
4. Fix bugs before the next game. Check known game rules or the wiki first,
   patch code, run focused tests, then verify with the v2 simulation suite when
   relevant.
5. After every loss, analyze the critical decisions and either fix the bug or
   document why the loss was unavoidable.
6. Commit and push after every completed game or discrete live-run fix. Do not
   batch unrelated discoveries.
7. Mouse only in live runs. No keyboard shortcuts.
8. Pair screenshots with memory-reader checks. Screenshot is UI ground truth;
   memory is validation and post-mortem truth.
9. Serialize state-mutating `game_loop.py` commands. Do not run `new`, `deck`,
   `card`, `execute`, `ability_used`, `pd_check`, `slayer_result`, `game_over`,
   or similar commands in parallel.
10. When a process error happens, improve this file. Prefer tightening an
    existing rule over appending duplicate guidance.
11. Serialize Ghidra headless commands that open the same saved project.
    Ghidra takes a project lock even for read-only exports, so parallel target
    exports against one baseline or typed project will race and one will fail.

## Recovery Protocol

Triggered by 0 scenarios.

1. Identify the most recent data entry: card, ability result, execution result,
   blocked card, night kill, or HP update.
2. Re-screenshot and verify that entry. Trust the screenshot and the live UI, not
   memory or prior assumptions.
3. Check whether `auto_card` or `auto_ability` misparsed a speech bubble. For
   example, a `#X shut up!` line is a silencing result, not a normal role clue.
4. If the entry is wrong, correct that entry manually. If it is correct and 0
   scenarios persist, save the case as a solver bug.
5. Do not cycle through values hoping the solver recovers. Do not reset unrelated
   cards. Do not use memory-reader truth to find the value that would make the
   solver happy.
6. Before accepting a loss, exhaust all unused active abilities, re-check every
   auto-entered card, and verify all entries.
7. Do not abandon in-app through pause menu while HP remains, unused abilities
   remain, unflipped cards remain, or memory reader is still readable. Leaving
   the game view destroys useful post-mortem ground truth.

## Screen And Mouse

- Resolution: 2560x1440.
- Park mouse at `(1280, 690)` before screenshots when no modal is open. Avoid
  cards, deck icon, side panels, and lower-right buttons.
- If a screenshot has a hover tooltip, park the mouse and retake it.
- Prefer `safe_click` over manual move/click; it focuses the game window first.
- For card clicks, prefer detected card-box centers from the current screenshot.
  `game_utils.game_card_coords` is a fallback only.
- Execute button is the red sword near `(2265, 1235)`. Dismiss the mark menu by
  clicking `(1280, 690)`, then use `safe_click btn_execute_sword`.
- Deck icon: use `safe_click icon_deck_purple` near `(2485, 100)`.
- Never click near center-top around `(1230, 62)` to open deck; that can hit a
  card in small games.
- Buttons highlight red on hover. No highlight usually means the game is
  unfocused.
- Escape opens pause menu, so avoid keyboard shortcuts.

## Live Game Loop

### Start

1. `python game_loop.py start` automates Play Demo -> Standard, dismisses intro,
   parks mouse, screenshots deck, and cross-checks card vision plus memory.
2. Verify deck output and read board header counts from the screenshot.
3. `python game_loop.py new <n_cards> <n_evil>`.
   `n_evil` is the displayed "Find and Execute N Evil Characters" count, not
   just minions plus demons. Puppet counts as an extra evil.
4. `python game_loop.py deck V=... O=... M=... D=... nv=<count> no=<count>`.
   Prefixes are required. Use `knowledge_base.py` as the source of truth for
   role factions.
5. Close deck with `safe_click icon_deck_purple`.

### Reveal And Enter

1. `python game_loop.py flip` flips all cards in strict #1-to-#N order.
   Use `flip --lilis` for Lilis batches and `flip <pos>` for a single card after
   Witch death.
2. Never manually construct click chains. `flip` preserves reveal order for
   Baker, makes Witch blocks predictable, and verifies the board afterward.
3. The first click of any multi-card `flip` can be swallowed by focus or board
   readiness. Use the verified first-click path in `game_loop.py`; if #1 still
   remains hidden, recover with `flip 1` before `auto_card`.
4. If verification reports positions still hidden, rerun `flip`. Do not mark a
   position blocked unless Witch is in the deck.
5. Run `auto_card` after flipping. It reads clues from memory and enters
   parseable cards.
6. Enter manual card info in reveal order. Active-only cards can be recorded as
   `card no_info <pos> <Role>` until their ability is used.
7. At game start, set HP if needed: `set_hp <hp> <wrong_exec_cost>`. Default
   high-ascension wrong execution cost is 5.

Important entry reminders:

- Poet `#X is Evil`: enter as `card poet <pos> bounty_hunter <target>`.
- Druid claiming Wretch: enter `card druid <pos> <targets> Wretch`, not `none`.
- Plague Doctor active ability: use `pd_check <pd_pos> <target> corrupted
  <evil_pos>` or `pd_check <pd_pos> <target> clean`.
- Shaman can overwrite any eligible Villager with another Villager's role at
  game start. When the copied role is Baker, later Baker clue text remains the
  safest identity surface: chain Bakers say "I was a <role>" and an original
  Baker says "I am the original Baker."

### Solve And Act

1. Run `python game_loop.py next` and do what it says. Use `next --plan` or
   `next --dry` only when you need print-only inspection.
2. For abilities: click the ability card, click targets, enter the result, run
   `ability_used <pos>`, then run `next`.
3. Warning: clicking a card with an unused active ability activates that ability
   instead of selecting it as a target.
4. For executions: dismiss mark menu, click sword, click target, screenshot, then
   run `execute <pos> <evil_role|good>`.
5. Repeat until the game ends.

### End

1. Screenshot the end screen before clicking Next; the game can auto-advance.
2. Run `python game_loop.py game_over win/loss <name> "<pos=Role,...>" "[notes]"`.
   `game_over` can read true evils from memory when available.
3. The true-evil dictionary contains only evil positions. Do not include
   night-killed or executed good cards.
4. Run the printed replay/regression checklist, then commit and push.

## Memory Reader

`memory_reader.py` reads live IL2CPP process state. It is used for validation,
deck cross-checks, clue extraction, and post-mortem truth.

Current build fingerprint:

- `GameAssembly.dll` size: `44834304`
- PE timestamp: `1777936964`

If the fingerprint changes, expect offsets to be stale. Re-run Il2CppDumper on:

- `B:\SteamLibrary\steamapps\common\Demon Bluff Playtest\GameAssembly.dll`
- `B:\SteamLibrary\steamapps\common\Demon Bluff Playtest\Demon Bluff_Data\il2cpp_data\Metadata\global-metadata.dat`

Then update the offsets in `memory_reader.py` and verify:

```
python -m py_compile memory_reader.py
python memory_reader.py --deck
python memory_reader.py --score
python memory_reader.py
```

Memory reader notes:

- Deck reading gives the role pool, not header counts. `nv=` and `no=` still
  come from screenshot/manual reading.
- Native Unity object names are preferred for multi-village correctness.
- `savedAct` is speech bubble text. `actedInfos` stores referenced targets.
- `runtimeData` stores role-specific data such as Enlightened direction,
  Alchemist count, and Baker original role.

## Rust Solver

- Rust solver is primary. Fix solver bugs in `crates/solver-core`, not in the
  legacy Python solver.
- `game_loop.py` calls `rust_solve_to_objects()`.
- Build: `cargo build --release`.
- Main regression suite: `cargo test --release --test simulation`.
- Tests in `tests/cases_v2/` are the active live-run corpus. `tests/cases/` are
  legacy reference cases.
- Python bridge: `rust_solver.py` wraps the CLI binary and persistent daemon.

## Current Patch Notes To Respect

- Alchemist cannot be corrupted. Their clue now reports how many corrupted
  characters were around them in range 2 at the start of the round, before the
  cure. This is represented as `corrupted_count`; legacy `cured_count` exists
  only for historical cases. Live wording may be `There was N Corruption around
  me`, not only `N Corrupted around me`.
- Baa is managed internally as `Imp`. At Start it selects one existing Outcast
  and adds that exact record to `DeckView.ObscuredCharacters`; current assets
  make the selection uniform because every `usuallyDisguised` flag is false.
  On any Baa death it removes that record and refreshes the deck view. This
  reveals only the hidden deck-strip identity, not a board card.
- Shaman is managed internally as `Illuzionist`; Witch is `Cipher`. After
  Plague Doctor and before Alchemist, Shaman selects an ordered pair of
  apparent Villagers, attempts `MessedUpByEvil` on the source, overwrites the
  destination with the source's bluff-or-real identity, immediately fires the
  copied Start action, then attempts the marker on the destination. The source
  is unchanged. `InitWithNoReset` preserves destination statuses, resistance,
  and runtime data. The solver's `ShamanTrace` keeps ordered endpoints, copied
  role, and a viable erased-role candidate class; copied Baker/runtime-data
  composition remains opaque pending its own native audit.
- The public Dreamer asset binds managed `Dreamer`, not the unbound alternate
  `Dreamer2`. It picks exactly two characters and immediately produces either
  `Among #X, #Y there is: RoleA or RoleB` or the truthful Wretch/Cabbage clue;
  there is no role picker. Native current-build fallback can truthfully name
  both selected roles, and a lying clue can collide with one selected real role
  through the other target's bluff. Validate exact native output support rather
  than enforcing authored one-match/zero-match counts. Solver recommendations
  must include two targets; if only one is printed, stop and fix the strategy.
  New observations carry `dreamer_variant: public_current`; unversioned role
  pairs are archived pre-audit fixtures and intentionally use their conservative
  legacy predicate. Do not infer that a role such as Gravedigger was removed
  from a few outputs.
- Rambler was redesigned. Old solver code modeled "picked by a liar silences
  Rambler"; that rule is obsolete. New rule: adjacent truthful characters tell
  Rambler to shut up instead of sharing their own info. `auto_card` should record
  non-Jester `#X shut up!` as `shut_up_target`, not no-info and not the role's
  normal numeric clue. Live asc83_v7 data: truthful #2 Puppet and #9 Baker told
  real Rambler #1 to shut up; lying #3 Puppeteer and #5 Baa pointed shut-up at
  fake Rambler #4.

## Known Gotchas

- Doppelganger counts in `nv`, not `no`.
- Drunk can count as Villager in the header.
- In the 2026-05-05 live build, Drunk still lies and wrong-exec costs 2 HP.
  Plague Doctor reads the active `Corrupted` status directly, including on an
  ordinary Drunk. The clean Drunk in asc84_v2 was Chancellor-generated from an
  Alchemist: inherited resistance blocked Drunk's Start status. Execution
  bookkeeping still projects Drunk as clean; do not apply that projection to
  Plague Doctor.
- Baa's eye-symbol mismatch applies only to deck view, not HUD. HUD counts Baa
  as a Demon. If reading `no=` from HUD, do not subtract 1.
- `next` can auto-execute by default. Use `next --plan` or `--dry` when you need
  inspection only.
- Wrong-executing Drunk has special HP behavior. Keep HP in sync with `set_hp`.
- Knight immunity and corrupted Knight damage need careful confirmation in live
  games.
- Current serialized role/display mappings include internal `Marionette` ->
  Twin Minion, `Mezepheles` -> Puppeteer, and `Puzzlemaster` -> Plague Doctor.
  Do not infer a public role name from its managed class name.
- Plague Doctor can target any board character, including self and dead cards.
  A self-check always displays `Not Corrupted`. A truthful corrupted check
  uniformly names a registered/runtime Evil character (including Wretch or a
  dead Evil); a lying clean check uniformly names Good and falsely calls it
  Evil. `next`/the autonomous loop parses and cross-checks the exact public
  speech; on failure, recover with `pd_check`. Never inject the hidden Start
  target from memory into live solver state.

## Setup

- Screen: 2560x1440.
- Python 3.13.
- Rust 2021 workspace at repo root.
- Python dependencies include `mss`, `pyautogui`, and `Pillow`.
- REPL mode: `python game_loop.py repl` keeps a persistent process and uses
  `REPL_READY` / `CMD_DONE` sentinels.

## Game Overview

Demon Bluff is a deduction puzzle game with a circle of face-down cards. Reveal
cards for role info, deduce which characters are evil, and execute all evils
before HP runs out. Evil characters disguise and lie. Good characters can become
corrupted, making their info unreliable without changing their apparent role.
