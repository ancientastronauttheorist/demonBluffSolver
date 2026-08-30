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
- [x] Map, baseline-export, type, and native-audit all 12 methods in the
  internal `Dreamer2` boundary, including its randomized type-exclusion clue
  and the complete `GetDreamerClue` provider set.
- [x] Asset-bind the public Dreamer card to managed `Dreamer`; prove that
  `Dreamer2` and `DreamerOld` are unbound in the current gameplay assets.
- [x] Map, baseline-export, type, and native-audit the complete public
  `Dreamer` boundary: all 11 role methods plus five compiler-generated helpers,
  including its Cabbage branch and exact current-build role-pair weighting.
- [x] Implement and regression-test the public Dreamer parser, native-support
  validator, and weighted role-pair recommendation model.
- [x] Asset-bind public Baa to managed `Imp`; native-audit its complete role
  class plus deck-view add/remove helpers, and remove the false board-reveal
  inference from the live wrapper.
- [x] Asset-bind public Shaman to managed `Illuzionist`; native-audit its four
  role methods plus seven selection, status, and lifecycle helpers, expand the
  typed union to twelve target sets and 198 memberships, and publish its
  baseline-versus-typed quality report.
- [x] Asset-bind public Plague Doctor to managed `Puzzlemaster`; native-audit
  all 11 role methods plus 12 dispatch, click, picker, status, and filter
  helpers, close truthful/bluff Day output and Drunk status handling, expand
  the typed union to thirteen target sets and 221 memberships, and publish its
  baseline-versus-typed quality report.
- [x] Asset-bind public Judge to managed `Judge2`; native-audit all ten role
  methods plus eight dispatch, truth-appearance, click, and picker helpers,
  close unrestricted target legality, deterministic corrupted-actor inversion,
  exact one-reference output, and ResetAfterNight history, expand the typed
  union to fourteen target sets and 239 memberships, and publish its
  baseline-versus-typed quality report.
- [x] Asset-bind public Witch to managed `Cipher`; native-audit all five role
  methods plus 14 ordered-Start, inherited-dispatch, global-value, hidden-count,
  click, reset, and ordinary/night-death helpers, close the exact last-card
  predicate, lack of blocked identity, killed-hidden membership, self-block,
  stacking/duplicate behavior, and death cleanup, expand the typed union to 15
  target sets and 258 memberships, and publish its baseline-versus-typed
  quality report.
- [x] Asset-bind public Chancellor to managed `Baron`; native-audit all five
  role methods, all eight Witness methods, and 18 ordered-Start, selection,
  status, identity-mutation, and death helpers; close anywhere-Villager
  eligibility, exact anchor/neighbour order, `c/v/o/f/a` identity equations,
  duplicate and resistance behavior, current-status Witness truth/bluff
  semantics, and death persistence; expand the typed union to 16 target sets
  and 289 memberships and publish its baseline-versus-typed quality report.
- [x] Asset-bind public Lilis and Knight to managed `Striga` and `Immortal`;
  native-audit all 13 role methods plus 41 ordered-Start, Night-rule,
  selection, delayed-kill, protection, ordinary-execution, Slayer, HP, status,
  and reset helpers; close hard registered-Good priority, protected no-kill and
  duplicate-Night behavior, exact Knight killability precedence, and the
  additional-four/total-nine corrupted-Good execution result; expand the typed
  union to 17 target sets and 343 memberships and publish the combined quality
  report.
- [x] Asset-bind public Rambler to managed `Rambler2`; native-audit all 14 role
  methods, both compiler-generated closure methods, and 20 setup, dispatch,
  adjacency, interference, acted-history, and reveal helpers; close pre-flip
  AfterRoundStart installation, actual-source versus apparent-target truth,
  hidden callback persistence and last-writer behavior, duplicate/small-board
  adjacency, immediate versus pre-append history, and constraint-free Day
  quotes with exact references; expand the typed union to 18 target sets and
  379 memberships and publish its baseline-versus-typed quality report.
- [x] Asset-bind public Baker to managed `Baker`; native-audit all 11 role
  methods, Baker runtime history, all three achievement-helper methods, and 21
  click, reveal, dispatch, filter, replacement, lookup, and acted-history
  helpers; close synchronous Day-only chain timing, exact real/lying role-name
  pools, runtime-cast and status composition, registered candidate eligibility,
  physical order/duplicates, small boards, and achievement ordering; expand
  the typed union to 19 target sets and 415 memberships and publish its
  baseline-versus-typed quality report.
- [x] Asset-bind public Doppelganger and Drunk to managed `Doppleganger` and
  `Drunk`; native-audit all 17 role methods plus 22 setup, delayed-reveal,
  source-filter, unique-pool, registration, status, and execution helpers;
  close ordered Start/Puppeteer conversion before disguise selection,
  erased-Villager exclusion, clean/corrupted source pools, state and duplicate
  weighting, Drunk's two-draw must-include priority and bounded not-in-play
  guarantee, failure mutations, and register-as/HUD separation; expand the
  typed union to 20 target sets and 454 memberships and publish its combined
  baseline-versus-typed quality report.
- [x] Asset-bind public Fortune Teller to managed `FortuneTeller`;
  native-audit all 11 role methods, all six compiler-generated helpers, and
  eight dispatch, registered-alignment, click, picker, and acted-record
  helpers; close unrestricted two-target legality, exact-reference toggle and
  `OnPicked` ordering, truthful registered-Evil OR and deterministic lying
  complement, discarded RNG consumption, ascending-ID speech/reference shape,
  cancellation, ResetAfterNight history, and the both-Evil achievement; expand
  the typed union to 21 target sets and 479 memberships and publish its
  baseline-versus-typed quality report.
- [x] Asset-bind public Bombardier to exact managed `Saint`; native-audit all
  five role methods plus 18 dispatch, ordinary/forced/Demon death,
  bookkeeping, and terminal helpers; close the broader non-Demon-death rule,
  current-`dataRef` managed-type identity, Shaman/Chancellor replacement
  composition, ordinary-bluff and Drunk/Doppel non-composition,
  `SaintVillager` exclusion, and automatic-loss precedence; expand the typed
  union to 22 target sets and 502 memberships and publish its
  baseline-versus-typed quality report.
- [x] Asset-bind public Pooka to exact managed `Pooka`; native-audit all five
  declared role methods plus four status, real-type, adjacency, and rotation
  helpers; close Start-only Evil dispatch, deterministic two-neighbour current-
  real-Villager eligibility, independent Corrupted/MessedUpByEvil attempts,
  ordinary duplicate and small-board behavior, and the native-xref proof that
  private random-one-neighbour `PoisonClosestNeighbours` is unreachable in the
  shipped flow; expand the typed union to 23 target sets and 511 memberships
  and publish its baseline-versus-typed quality report.
- [x] Asset-bind public Poisoner to exact managed `Poisoner`; native-audit all
  four declared role methods plus 13 ordered-Start, dispatch, output,
  adjacency, real-type, status, resistance, and integer-RNG helpers; close
  previous-then-next eligibility, live Corrupted exclusion, independent marker
  resistance, all-match high-ID-first duplicates, dead and small-board
  behavior, and the stale managed-description/nonexistent-dormant-helper
  distinction; expand the typed union to 24 target sets and 528 memberships and
  publish its baseline-versus-typed quality report.
- [x] Replace the solver's generic Shaman duplicate allowance with an ordered
  source/target/copied trace plus a viable overwritten-identity class,
  native-timed status effects, and copied-Alchemist Start regressions.
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
