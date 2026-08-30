# Full Reconstruction Roadmap

## Definition of complete

For this project, “fully decompiled” means:

1. Every type in the game-owned `Assembly-CSharp.dll` range is inventoried.
2. Every nontrivial native method is classified as reconstructed, understood
   boilerplate/generated code, unreachable, or explicitly unresolved.
3. Gameplay-critical methods have readable authored pseudocode or clean-room
   implementations, call relationships, field layouts, and validation evidence.
4. Deck construction, board lifecycle, statuses, corruption, clues, active
   abilities, execution/damage, night resolution, scoring, ascension rules, and
   every role have differential tests against observed behavior.
5. A new game build can be fingerprinted, dumped, diffed, and triaged by the
   checked-in scripts without relying on undocumented local steps.

This does not claim recovery of original variable names, comments, project
layout, or byte-for-byte C# source. Those do not survive IL2CPP compilation.

## Milestones

- [x] Create and push a dedicated branch.
- [x] Fingerprint the current game and metadata.
- [x] Produce a current-build Il2CppDumper extraction.
- [x] Commit the reproducible foundation and build manifest.
- [x] Generate the first complete `Assembly-CSharp` type inventory.
- [x] Establish the complete 4,207-method coverage and evidence ledger.
- [x] Produce Cpp2IL managed-IL recovery and an explicit quality baseline.
- [x] Import `GameAssembly.dll` into Ghidra and apply IL2CPP method, metadata,
  and string symbols.
- [x] Import IL2CPP headers, selected prototypes, and reachable field layouts
  into an isolated typed Ghidra project; complete full auto-analysis and
  read-only post-save signature/ABI validation.
- [x] Export and confirm the first gameplay-core native target set.
- [x] Recover and native-audit the first roster-selection helper boundary.
- [x] Map and baseline-export the 28-method gameplay-lifecycle boundary, then
  native-audit its first 11-method setup, board, reveal, and click/kill slice.
- [x] Native-audit the remaining 17 initialization, reveal/kill-helper,
  bookkeeping, and Night-flow methods; close the lifecycle boundary.
- [x] Expand the isolated typed project to all 28 lifecycle methods and pass
  post-save ABI validation plus baseline-versus-typed quality checks.
- [x] Map and baseline-export the 30-method execution-resolution boundary.
- [x] Native-audit the first 16-method execution, damage, protection, and
  terminal-result slice.
- [x] Native-audit the remaining 14 status-insertion, Night-rule, Striga,
  Demon-selection, and collection-helper methods; close the boundary.
- [x] Expand the isolated typed project to all 77 methods across the four
  reviewed target sets; pass post-save ABI validation and body-free quality
  checks for every set.
- [x] Map and baseline-export the 40-method status, corruption, truth/lie, and
  bluff-orchestration boundary, including explicit C prototype aliases for
  overloaded managed methods.
- [x] Native-audit the first 16 status-storage, cure-gating, selection, and
  truth/appearance methods in that boundary.
- [x] Native-audit the next 11 Pooka, Poisoner, Puzzlemaster/Plague Doctor,
  Drunk, and Alchemist status-lifecycle methods in that boundary.
- [x] Native-audit the final 13 bluff storage, Puppet/Puppeteer,
  Doppelganger, Confessor, Reveal, and shared orchestration methods; close the
  40-method status/corruption/truth boundary.
- [x] Map and native-audit the 20-method bluff-acquisition boundary, including
  common assignment, Demon/Minion/Spy/Mutant selectors, pool mutations,
  shared-body identity, and stale-role lifecycle reachability.
- [x] Add bluff acquisition to deterministic checked-target discovery and the
  six-set typed-header/refresh union with exact overload aliases.
- [x] Refresh the preserved typed project and publish the bluff-acquisition
  baseline-versus-typed quality report.
- [x] Map, baseline-export, and native-audit the complete Slayer and Wretch
  role implementations; join registered alignment to kill-and-reveal behavior
  and fix the live Wretch bookkeeping regression.
- [x] Expand the deterministic typed union to eight target sets and 154 target
  memberships; support folded per-role native bodies in apply/validation and
  publish both role quality reports.
- [ ] Add a versioned offset registry and migrate `memory_reader.py` to it.
- [ ] Live-validate HP, gameplay-state, and board-count pointer chains.
- [x] Recover the gameplay lifecycle and its call graph.
- [ ] Recover deck/board construction and ascension rules.
- [x] Recover status, corruption, truth/lie, and bluff-acquisition pipelines.
- [ ] Recover clue-generation pipelines.
- [x] Recover execution, damage, protection, and night-resolution pipelines.
- [ ] Recover and validate every role implementation.
- [ ] Extract an authored clean-room behavioral core with differential tests.
- [ ] Run and publish the final method-classification coverage audit.

## Method classification

The coverage ledger will use these terminal states:

- `reconstructed`: readable behavior and tests exist.
- `understood`: behavior is documented; standalone reconstruction is unnecessary.
- `generated`: compiler/Unity boilerplate with its origin identified.
- `unreachable`: not reachable in the shipped Standard/Ascension game surface.
- `unresolved`: work remains, with the exact blocker recorded.

No nontrivial method may disappear from the denominator.

## Validation gates

- Verify both `GameAssembly.dll` and `global-metadata.dat` SHA-256 values before
  extraction or memory-layout use.
- Reject RVAs outside valid PE sections and offsets not tied to a declaring type.
- Pair live memory observations with screenshots; the screen remains UI truth.
- Record controlled one-event before/after traces for state transitions.
- Keep CI independent of proprietary inputs through manifests and synthetic
  fixtures.
- Run Python tests, `cargo build --release`, and
  `cargo test --release --test simulation` for gameplay-facing changes.
- Commit and push each discrete subsystem or role milestone.
