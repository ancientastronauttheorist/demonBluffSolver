# Demon Bluff Solver Agent Guide

This file is the operating guide for Codex and other coding agents working in
this repository. It is adapted from `CLAUDE.md`, with Claude-specific process
language translated into agent-neutral rules.

When subagents are requested or authorized, inherit the parent agent's model.

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
   batch unrelated discoveries. After pushing, use a new follow-up commit for
   corrections rather than amending published history or requiring a force push.
7. Mouse only in live runs. No keyboard shortcuts.
8. Pair screenshots with memory-reader checks. Screenshot is UI ground truth;
   memory is validation and post-mortem truth.
9. Serialize state-mutating `game_loop.py` commands. Do not run `new`, `deck`,
   `card`, `execute`, `ability_used`, `pd_check`, `slayer_result`, `game_over`,
   or similar commands in parallel.
10. When a process error happens, improve this file. Prefer tightening an
    existing rule over appending duplicate guidance.
    In PowerShell, pass ripgrep a directory and `-g '*.json'` (or the relevant
    glob). For a filename family use `rg <pattern> <directory> -g 'prefix*.md'`.
    Restrict source/note content searches to their relevant extensions; use
    `rg -l` for report discovery and parse allowlisted report metadata rather
    than printing matching minified corpus lines.
    Do not pass wildcard paths that the shell leaves unexpanded, including
    a trailing `/*` or partial-name wildcard on a directory argument; this also
    applies to documentation filename prefixes in multi-command batches.
    Resolve all uncertain filenames, including audit scripts and status/summary
    documents, versioned coverage inventories, role/knowledge-base files and Rust module roots, with
    `rg --files` before reading or using them as content-search operands. A
    verified file in a multi-file search does not verify its guessed siblings. Do not
    infer an audit filename from a role name or descriptive note title, expand a
    Rust import alias or `mod` declaration into a guessed basename, or assume a
    module uses `mod.rs`. A confirmed child directory does not establish its
    module root; resolve the sibling `.rs` file or `mod.rs` before opening either.
    Follow an audit's exact linked script path after verifying it exists.
    Retain the returned directory when opening a resolved basename; do not
    reconstruct a sibling path after discovery. A script name mentioned in
    notes is not necessarily relative to the repository root.
    Resolve extractor outputs separately from tool installation directories;
    a Dumper version identifies neither its build-specific output directory nor
    a verified copy of script.json, dump.cs or il2cpp.h.
    A dependency directory does not establish a virtual environment or an
    interpreter path; resolve the interpreter separately before invoking it.
    Resolve commit identities with `git rev-parse` before writing provenance;
    never expand a short revision into an unverified full hash.
    Before creating an audit artifact, verify its assigned full output directory
    and preserve it in the write path. Before freeze, compare each resolved
    script, note and report path with its assigned path.
    Normalize repository-relative paths with as_posix() when comparing against
    Git's forward-slash path output on Windows.
    Inspect filename-discovery results before issuing dependent reads; do not
    batch discovery with reads against guessed paths, even for a previously
    discussed family whose exact note basename is not recorded. A worker's source-ready
    notice does not establish that its planned note or report already exists.
    Resolve private diagnostic outputs by filename and format too; a verified
    JSON decode does not establish a separate per-method text export.
    Verify a documented directory exists before searching it, including optional
    tool configuration directories such as `.cargo`, since directory maps can
    describe intended layout. Start with repository-root `rg --files`
    when even the containing directory is unconfirmed.
    Evidence sources can name corpus directories as well as files; validate the
    referenced path's existence without assuming every source is a single file.
    Resolve and validate every coverage target before appending evidence; a new
    partial method audit can require an explicit unresolved classification.
    Do not guess filenames or repeat an unexpanded wildcard search. Inspect the
    current read result before preparing an exact-match patch to any changed file,
    including earlier inserted prose; use
    its returned lines rather than remembered fragments, and
    keep patch hunks in file order and omit empty placeholder hunks. Do not
    include no-op context-only hunks; after a failed patch, rebuild its hunks
    from the fresh read instead of resubmitting the failed patch. Check every
    hunk contains an addition or deletion, even when its file already has a diff.
    Validate each changed file's current context before combining additions
    and edits in one patch.
    Verify that every file declared changed by a patch has an actual diff.
    For literal source fragments containing regex metacharacters, use `rg -F`
    with separate `-e` arguments; do not prepare a dependent patch after a failed search.
    Re-read relevant lines after formatting before preparing an exact-match patch;
    formatting can change line wrapping even when the code's behavior is unchanged.
    Use explicit UTF-8 for repository text reads and writes in Python on Windows.
    Preserve physical newline translation when reproducing text reports for
    byte comparisons; Windows text-mode writes can emit CRLF from a final LF.
    For Unicode diagnostics, use ASCII-safe JSON or explicitly configure UTF-8
    console output; the default Windows console encoding may reject valid text.
    Prefer apply_patch or PowerShell here-strings for multiline Python edits
    and diagnostic scripts; save nontrivial multiline diagnostics instead of
    embedding exec strings in python -c. Nested shell quoting can fail before
    the code executes.
    Build report input snapshots from explicit serializable fields; `locals()`
    can also capture closure functions and fail only at final JSON serialization.
    Match diagnostic availability when comparing normal and stopped reports:
    preceding snapshots may be omitted on failure runs. Compare common prefix
    fields and require the selected stop snapshot and final state to match exactly.
    Check imported report/helper and target-manifest schemas before indexing
    their fields; a build constant is not necessarily repeated inside a returned
    layout dictionary, and manifest function rows need not use a `targets` key.
    Expand snapshot pooling only when the report declares `snapshot_encoding`;
    an imported report corpus can contain both pooled and ordinary reports.
    Inspect exact returned report keys rather than substituting a similar
    semantic description for a helper's field name; metadata arrays such as
    `script.json`'s `Addresses` can contain integers rather than row dictionaries.
    Distinguish absent/null row fields before indexing reference pairs; verify
    their container type and required length. Set PowerShell diagnostic scans
    to stop on errors so partial counters cannot be mistaken for valid totals.
    Read exact report counters and filtered corpus sizes before authoring
    checkpoint or native-fixture test assertions; do not hand-count operand
    pins or substitute a nearby summary count. Read a family's actual caller
    sentinel and native base before adapting another family's Rust fixtures.
    Before interpreting a world-set disagreement, verify the reference
    projection preserves the admitted public role multiset; one role name
    does not represent repeated roster occurrences.
    Calculate capacity boundaries
    from complete storage and future snapshot costs before asserting admission.
    Allowlist explicit metadata fields for diagnostics; excluding guessed corpus
    keys can accidentally print an entire retained or stopped report corpus.
    Locate report JSON with filename discovery or `rg -l`; do not print matching
    lines across report corpora, since one line can contain the entire corpus.
    Inspect sequence container shapes before iterating calls; retained sequences
    can be wrapper records with a `calls` field rather than lists of call rows.
    Inspect inherited emulator initialization before using its attributes;
    a dependency imported locally by a base class need not be an instance field.
    Inspect inherited identity normalization before using it for service arguments;
    stack addresses, immediates and metadata tokens need not be managed objects.
    Reset dynamically added identity/string labels with allocator and object
    registries before each independent case; reused addresses must not retain
    labels from a previous case.
    Check inherited harness preconditions before composing retained native calls;
    derive branch expectations from retained storage rather than fresh-fixture defaults.
    Emulation can return at a time/instruction budget without completing a call.
    Verify RIP and the declared stop/completion before applying return ABI checks;
    a bounded continuation must retain the live CPU, stack and service chronology.
    Qualify callback plans by the invocation that can reach them; do not carry
    a later call's mutation plan onto an earlier setup-only call.
    Bind indexed literal-array expressions before using them in serde_json::json!
    values, or parenthesize the complete expression for the macro parser.
    Parenthesize Rust cast expressions before comparison operators, especially
    `as T` followed by `<`, to avoid parsing the comparison as generic arguments.
    Match borrowed serde_json::Value operands in fixture assertions; a borrowed
    input needs a borrowed expected Value rather than an owned macro result.
    Coordinate shared Rust builds after agents confirm all declared module and
    test files and pending review fixes are complete, including filtered test
    runs that compile other integration targets. Reconfirm freeze after a
    failed comparison triggers edits; an earlier freeze no longer applies.
    Keep compiled sources and embedded fixtures frozen until that Cargo process
    exits. Queue review edits during a build and verify the final frozen revision.
    Create declared child test files before running rustfmt; it resolves child
    modules even when the new parent has not entered a shared Cargo build.
    Format edited Rust files explicitly instead of running workspace-wide
    `cargo fmt --all`; it can rewrite unrelated legacy source and test files.
    Inspect the diff afterward and restore unrelated formatting within an
    edited file, especially large legacy tables touched only by a comment edit.
    A whitespace-only comparison can still differ on formatter-added trailing
    commas; inspect those punctuation changes before treating it as a source edit.
    If spawning hits the agent thread limit, reuse available workers only when
    their model matches the current user preference; otherwise continue in the
    primary agent. Completed tasks may still retain their thread slots, and
    existing workers do not change models when this guide changes.
    Resolve each completed module's actual path before adding its declaration;
    a module name alone does not identify whether its parent is `lib.rs` or a
    nested module such as `bluff.rs`.
    Gate dependent shell steps on successful exit codes. In PowerShell a failed
    native command does not stop later lines; keep validation and commit in a
    checked subprocess sequence or explicitly exit on failure. This also applies
    to source-edit or integration helper commands before their dependent steps.
    Wait for report-producing processes to finish successfully before opening
    their output paths; a yielded session does not establish that a report exists.
    Have the originating agent resume its yielded shell session. A session ID
    returned by another agent need not be accessible here; an unknown-session
    response is not evidence that its producer stopped or should be restarted.
    Compile edited Python audit syntax before launching report producers; bind
    or parenthesize Boolean expressions following equality comparisons.
11. Serialize Ghidra headless commands that open the same saved project.
    Ghidra takes a project lock even for read-only exports, so parallel target
    exports against one baseline or typed project will race and one will fail.
    Keep complete proprietary native instruction bytes and disassembly in the
    private artifact workspace. Tracked audits use authored selected operand
    assertions and per-body fingerprints instead of complete native exports.
    Resolve export filenames from the target manifest or directory listing;
    public-role names and hexadecimal filename widths can differ from assumptions.
    Inspect shared method bodies without printing their potentially enormous
    alias-header line.
    When present, strip the complete comment header; some export formats have
    no header. Fixed line-count skipping is unreliable.
    Unity type trees can omit custom MonoBehaviour fields; check consumed size
    and treat partial reads as headers, not complete serialized objects.
    Serialized Boolean fields may align individually; do not apply contiguous
    IL2CPP runtime offsets to asset bytes. Validate following reference IDs.
    For native PE inspection, distinguish zero-filled virtual data from file-
    backed bytes; get_offset_from_rva alone does not establish raw backing.
    Check the section's raw extent before reading data slots, require a full
    file read before unpacking and supply runtime
    globals and Windows TIB/TLS state explicitly in emulation; do not bypass
    conversion bodies to hide missing thread-local initialization. With fast-load PE readers, explicitly parse
    the required data directories before using their tables. Verify an unwind entry contains a queried RVA
    before treating it as that instruction's chunk. Pointer-backed leaf entries
    can lack unwind records or saved Ghidra definitions; check lookup presence
    before selecting an entry, and verify their decoded
    wrapper before requesting a containing-function export.
    A method can span adjacent unwind chunks; the first chunk's end is not
    necessarily the method's end. Resolve the next verified managed entry.
    Decode from a verified entry/instruction boundary before selecting a later
    output range; arbitrary byte windows can silently misdecode native code.
    Derive helper entries from decoded concrete call sites before adding them
    to diagnostic target lists; a remembered constructor RVA is not evidence.
    Include the entire final instruction when sizing a decode range, exclude
    trailing alignment padding, account for embedded jump tables as data, and
    verify all return paths rather than stopping
    at the first `ret`. Assert requested addresses decoded before indexing them.
    Derive exact instruction assertions from that decode, including operands
    on folded return stubs and expected call-site counts; do not infer encoding
    from decompiled C or count sites manually. Retain instruction mnemonics in
    target inventories; a tail jump must not become a call assertion.
    A pinned instruction is not evidence that a fixture executed its branch;
    verify trigger predicates and retained state before claiming a write occurred.
    Read numeric constants before assigning units or expected magnitudes.
    Derive native wait output fields from the reviewed producer and live record,
    including signed frame increments; do not copy input timing into assertions.
    Resolve the pinned class's exact field declarations before naming offsets
    or asserting publication scope; adjacent saved and current roster fields
    are distinct state.
    Derive RIP-relative literal slots from decoded operands and resolve their
    exact strings before writing executable probes; do not leave unresolved
    numeric placeholders. Property names do not establish serialized preference keys.
    Use the decoded displacement location when an instruction has an immediate
    operand; the final four instruction bytes need not be its RIP displacement.
    Give overloaded target signatures distinct `prototype_name` values while
    preserving their original metadata signatures and exact RVAs.
    For a shared RVA, reuse its established canonical `applied_prototype_name`
    before invoking exports when parameter counts agree; preserve each declaration's
    metadata signature. Incompatible folded declarations need separate exact
    evidence, not a false prototype or a target that fails union validation.
    Native fixture metadata names must match Dumper's exact namespace syntax;
    assert every required slot was found before executing warmed fixtures.
    Validate ABI arguments and returns at the decoded operand width; byte register writes
    preserve upper bits. Check call-site register setup before trusting inferred
    decompiler parameters or constructor return values. A write-barrier
    notification after a struct copy can pass a null second argument; qualify
    its caller and verify the preceding native stores instead of treating that
    argument as the stored reference. Initialize recorded unused
    volatile entry registers explicitly so preceding fixtures cannot supply
    accidental diagnostic bits. Capture raw service arguments and caller sites
    at entry, including stopped services, rather than only after completion.
    Derive release assertions from exact writes and lifetime effects; logical
    free does not imply zeroed storage. A supplied allocator can retain cached
    pointer bytes after handles are cleared and owner links are removed; those
    bytes are inspection evidence, not a live reference.
    Author failed API output
    effects explicitly; a failure status alone does not specify changes to
    input capacity/type fields that a retry may consume. Gate callbacks and snapshots by
    phase, since base constructors can invoke overrides before derived state exists.
    Qualify callback mutations at shared gateways by the verified native caller
    or decoded return site so a parent call cannot trigger a callee-only effect.
    Separate per-invocation service ordinals from retained chronological logs;
    a prior call's count must not suppress the next call's authored callback.
    Pause/reentry retries must consume inherited diagnostic counters and labels
    only once, as well as consuming service effects and ledger entries once.
    Bind snapshot decoders to native storage flags; short inline strings need
    not share the pointer representation of parsed or longer strings.
    Resolve exact type declarations, including enums, before extracting dump blocks; prefix matches
    can select another class and modifiers can differ from an assumed declaration.
    Exact simple names can also collide across namespaces; qualify the namespace
    or a verified TypeDefIndex before selecting that declaration.
    Bound each block by that declaration's closing brace, not an optional
    Properties marker that can skip into the next type.
    For Dumper RVA/VA/Offset comments, preserve the observed spelling or parse
    and compare numeric values; do not regenerate exact comment strings from
    normalized lowercase addresses.
    Discover compiler-generated iterator suffixes from metadata or exact dump
    declarations; a remembered ordinal is not a verified generated class name.
    Preserve exact floating-point values when loading native timing fixtures;
    check parser rounding before weakening a failed exact comparison.
    Check each requested export's result before reading its file: a successful
    headless process can still report missing functions or partial exports.
    An export batch may abort at its first missing target; require a per-target
    completion record before reading any subsequent requested output. Inspect
    those records before issuing a separate file read, even when exit status is zero.
    Do not batch a process wait with a dependent output read; inspect the wait
    result first and read only targets explicitly reported complete.
    Pass explicit `0x`-prefixed RVAs to `DumpContaining`; bare numeric strings
    can select decimal addresses and bare hexadecimal letters fail parsing.
    For omitted initializer or virtual functions, resolve entries from their native pointer
    table before defining them in a read-only export session.
    Internal-call requests can include parameter signatures that registrations
    omit. Inspect the exact request and audit its fallback lookup before
    asserting an exact request-to-registration string match.

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
3. Warning: outside an active native picker, clicking a card with an unused
   active ability activates that ability. Judge's picker routes target clicks
   first, so an unused-active target is legal once Judge is already activated.
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
  and adds that exact record to `DeckView.ObscuredCharacters`. It first draws
  from all Outcasts, then replaces that draw from the usually-disguised priority
  pool when nonempty. Current Drunk and Doppelganger assets have that flag set.
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
- The public Judge asset binds managed `Judge2`, not the unbound `Arbiter`.
  Judge reports `CheckLyingAppearance(target)` when its actor is truthful and
  deterministically inverts that value when the actor is lying, including a
  corrupted Good Judge. Its picker accepts any board card: self, dead, hidden,
  or a target with an unused active ability. The exact results are `#X is\nLying`
  and `#X is\nsaying Truth`, with one target reference per chronological
  `actedInfos` entry. Judge resets after every Night without clearing prior
  results; preserve and validate the full history.
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
- Native Knight precedence is HealthyBluff protection, then Corrupted/runtime-
  Evil killability. Ordinary Corrupted Good Knight execution costs base 5 plus
  4; Drunk showing Knight costs 2 plus 4. Lilis and Slayer deaths omit the
  OnExecuted 4-HP hook.
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
