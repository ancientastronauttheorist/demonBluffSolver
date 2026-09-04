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
- [x] Asset-bind public Twin Minion to exact managed `Marionette`; native-audit
  all five declared role methods plus 15 ordered-Start, dispatch, Demon-filter,
  alive-adjacency, current-data replacement, delayed-reveal, bluff, and integer-
  RNG helpers; close the two-draw current-`CharacterData` swap, physical-state
  preservation, duplicate/small-board behavior, pending-coroutine multiplicity,
  and dormant-helper reachability; expand the typed union to 25 target sets and
  548 memberships and publish its baseline-versus-typed quality report.
- [x] Asset-bind public Poet to exact managed `Gossip`; native-audit all six
  declared role methods, the twelve exact provider constructors, and generic
  Character real/bluff dispatch; close the ordered provider pool, fresh
  per-invocation real/bluff draw, Day-only callback routing, and strict current
  provenance schema while preserving unmarked legacy fixtures; expand the typed
  union to 26 target sets and 568 memberships and publish its
  baseline-versus-typed quality report.
- [x] Asset-bind public Scout to managed `Scout` and public Hunter to managed
  `Tracker`; native-audit all 17 declared role methods plus nine exact runtime-
  alignment, registration, circular-distance, range-reference, calculator,
  and RNG helpers; close Scout's occurrence-weighted target identity,
  one-Evil sentinel, strict 1-through-3 bluff domain, Hunter's exact `N - 1`
  exhaustion value and half-circle bluff domain, and ordered duplicate-preserving
  acted references; expand the typed union to 27 target sets and 594
  memberships and publish its baseline-versus-typed quality report.
- [x] Asset-bind public Oracle to managed `Investigator`; native-audit all seven
  declared role methods, all six generated comparer methods, and six exact
  Character, script-pool, all-ascension-pool, and registration helpers; close
  the independent truthful Minion/Good draws, moved-Twin duplicate reference,
  exact no-Minions sentinel, distinct-Good bluff pair, and fallback Minion
  label pool for both direct and Poet observations; expand the typed union to
  28 target sets and 613 memberships and publish its baseline-versus-typed
  quality report.
- [x] Asset-bind public Lover to managed `Empath`; native-audit all nine role
  methods, exact circular-adjacency and registered-alignment helpers, and all
  four achievement-helper methods; close registered-Evil occurrence counting,
  duplicate small-board references, exact truth text, the authored
  Minion-plus-Demon bluff domain, and truth-only achievement subscriptions for
  both direct and Poet observations; expand the typed union to 29 target sets
  and 628 memberships and publish its baseline-versus-typed quality report.
- [x] Asset-bind public Bounty Hunter to managed `BountyHunter`; native-audit
  all eight declared methods plus registered-alignment, board-filter, acted-
  record, and integer-RNG helpers; close its dormant direct Start mutation,
  active Poet truth/bluff pools, exact zero-reference clue, and joint anonymous-
  Wretch constraints; expand the typed union to 30 target sets and 640
  memberships and publish its baseline-versus-typed quality report.
- [x] Asset-bind public Medium to managed `Lookout`; native-audit all eight
  declared methods plus registered-alignment, live-identity, raw-status,
  acted-record, and integer-RNG helpers; close its actor-sensitive truthful
  pool, raw-bluff-holder fallback, exact one-reference two-line clue, and
  conditional execution achievement; expand the typed union to 31 target sets
  and 654 memberships and publish its baseline-versus-typed quality report.
- [x] Asset-bind public Knitter to managed `Knitter`; native-audit all eight
  declared methods plus registered-alignment, acted-record, count-removal, and
  integer-RNG helpers; close circular registered-Evil pair counting, exact
  small-board occurrence geometry, truth text, the authored false-count domain,
  direct/Poet parity, and Baker-to-Spy registration chronology; expand the
  typed union to 32 target sets, 666 memberships, 426 selected managed methods,
  and 365 unique native RVAs, then publish its typed-quality report.
- [x] Asset-bind public Enlightened to managed `Shugenja`; native-audit all
  nine declared methods plus registered-alignment, acted-record, physical-list,
  runtime-data, and float-RNG helpers; close exact direction text, public circle
  orientation, no-Evil and small-board ties, always-false bluff support,
  direct/Poet parity, joint anonymous-Wretch worlds, and Baker-to-Spy
  registration chronology; expand the typed union to 33 target sets, 680
  memberships, 435 selected managed methods, and 372 unique native RVAs, then
  publish its typed-quality report.
- [x] Asset-bind public Bishop to managed `Bishop`; native-audit all 17 declared
  role/compiler-generated methods plus registered-data, type-filter, acted-
  record, list-shuffle, and RNG helpers; close exact truth category precedence,
  authored-count bluff construction, live register-as-first references,
  separate ID/type/reference ordering, direct/Poet parity, joint anonymous-
  Wretch worlds, identity movers, and Baker-to-Spy chronology; expand the typed
  union to 34 target sets, 705 memberships, 455 selected managed methods, and
  382 unique native RVAs, then publish its typed-quality report.
- [ ] Preserve removed executed-evil role-to-position assignments during
  scenario construction, branch them before Start, and replay their complete
  ordered mutation histories so `Unknown` seats can be resolved without
  validator-local faction guesses.
  - [x] Stable-origin checkpoint: branch each untyped dead Evil over the exact
    authored multiset before construction; retain role-to-seat identity in the
    scenario; enforce trusted Minion/Demon quotas, exact trusted HUD Evil totals
    with a provenance-gated archival Puppet-count ambiguity, and identity-aware
    native Puppeteer/Puppet conversion in stable worlds, including an explicit
    stable-Twin/current-Puppet body overlay and conservative projection of its
    real Villager source through Start, while retaining conservative mixed-
    writer branches; and re-enable card and historical validators only for the
    resulting exact supported worlds.
  - [ ] Complete the remaining general ordered replay beyond the exact gated
    Twin/Puppeteer, Puppeteer/Shaman, and Twin/Shaman slices: broader mixed
    writer pools, duplicate mutators, split Twin presentation/action provenance,
    and probability-exact occurrence weighting. Strict current observations
    remain fail-closed for inferred incomplete writers.
    - [x] Implement the pure post-Twin Puppeteer boundary: select the first
      current actor, preserve physical previous/next occurrences, filter exact
      real Villagers, remove only the first Saint occurrence, make nonempty
      conversion mandatory, and retain the erased Villager role in an exact
      serializable replay trace.
    - [x] Integrate an atomic exact Twin-to-Puppeteer scenario slice for trusted
      no-Outcast boards with exactly the selected Twin/Puppeteer Minions and
      supported identity-stable writers: enumerate the complete pre-Twin
      Villager occurrence map, replay current-data relocation before selecting
      the Puppeteer actor and target, preserve erased-role provenance, validate
      exact current/public evidence, and fall back wholesale on unsupported or
      capped inputs without resurrecting exact contradictions.
    - [x] Integrate an atomic exact Puppeteer-to-Shaman scenario slice for
      trusted no-Outcast boards with exactly those two Minions, fully dealt
      Lilis Demons, and one deterministic non-Saint Villager neighbour for
      Puppeteer: enumerate the complete initial Villager occurrence map, replace
      the selected identity with Puppet before constructing Shaman's ordered
      Villager pairs, preserve both writer traces and all three marker attempts,
      prevent the erased Puppet identity from re-entering Shaman provenance,
      validate exact final current/public evidence, and fall back wholesale on
      ambiguity, preserved-state hazards, or caps.
    - [x] Cross the existing ordered Shaman trace with every exact Twin outcome
      when all possible Twin endpoints are proven structural non-Villagers, so
      the live Shaman candidate pool is invariant even if Shaman data relocates
      onto the Twin body. Preserve both trace identities and the complete
      Cartesian product, admit the no-Demon path, reject copied Bounty Hunter,
      and fall back atomically if any Twin branch touches a Villager or unknown
      endpoint.
    - [x] Integrate the first candidate-changing Twin-to-Shaman role-flow slice
      for trusted no-Outcast Scout/Witness boards with exactly Twin, Shaman,
      and fully dealt Lilis Demons: enumerate the complete Villager occurrence
      map, replay every Twin occurrence before rebuilding Shaman's live ordered
      pair pool, preserve both trace identities and duplicate-role RNG weight,
      validate the complete baseline and both native traces independently, and
      distinguish exact contradiction from cap/incomplete fallback. Distinct
      swaps with any captured reveal/action history fall back wholesale until
      runtime alignment, dispatched-role truth, and delayed Minion bluff
      presentation have separate provenance.
    - [x] Replay every selected current Plague Doctor at the global Start slot
      in descending displayed-ID order, rebuilding the live eligible pool for
      each actor and retaining ordered target/no-op history through Alchemist
      convergence. Preserve exact uniform target mass only for the singular,
      one-actor Start kernel; grouped Chancellor/Shaman/Poisoner/Twin/
      Puppeteer roots keep equal logical-world semantics.
    - [x] Implement the pure latent Shaman-copied Plague Doctor callback for a
      caller-proven ordinary runtime-Good/no-stale-bluff destination: rebuild
      its live apparent-Villager pool after global PD, preserve separate copied
      target/no-op provenance through Alchemist convergence, and prove that the
      overwritten destination drops when pre-Reveal `registerAs` is null.
      Keep it outside normal scenario generation because shipped initial Start
      has no live source bluff and therefore cannot copy Outcast PD data.
    - [x] Derive the settled physical `AppearTruthfull` status for both exact
      Shaman-copied Confessor endpoints from the existing ordered trace, retain
      it through later Baker no-reset presentation changes, and project it into
      Judge and shipped Rambler appearance checks without changing actual truth
      dispatch. Keep grouped erased-prior Confessor candidates fail-closed.
- [x] Asset-bind public Empress to managed `Noble`; native-audit all 14 declared
  role/compiler-generated methods plus registered-alignment, acted-record,
  pool-filter, and RNG helpers; close its direct/Poet three-reference schema,
  exact truth/bluff registered-alignment pools, lifecycle eligibility, actor-
  self parity, identity-mover behavior, text/reference ordering, anonymous-
  Wretch and Baker-to-Spy worlds, and RNG chronology; expand the typed union to
  35 target sets, 724 memberships, 468 selected managed definitions, and 390
  unique native RVAs, then publish its typed-quality report.
- [x] Asset-bind public Gemcrafter to managed `Archivist`; native-audit all
  seven declared methods plus registered-alignment, acted-record, pool-filter,
  and integer-RNG helpers; close exact direct/Poet text and reference parity,
  registered-Good truth and registered-Evil bluff pools, conditional actor
  removal and sole-pool self support, full lifecycle eligibility, managed-name
  ingestion, identity movers, anonymous-Wretch and Baker-to-Spy worlds, and
  RNG chronology; expand the typed union to 36 target sets, 735 memberships,
  474 selected managed definitions, and 394 unique native RVAs, then publish
  its typed-quality report.
- [x] Asset-bind public Bard to managed `Acrobat2`; native-audit all nine
  declared methods plus acted-record, circular-order, range-reference, false-
  number, and integer-RNG helpers; close exact direct/Poet text and ordered
  reference geometry, actor-self exclusion, full-lifecycle Corruption scans,
  fixed non-board-clamped bluff domain, managed-name ingestion, native real/bluff
  callback ordering, raw-bluff identity, identity movers, Baker-to-Spy
  chronology, and archive compatibility; expand the typed union to 37 target
  sets, 750 memberships, 482 selected managed definitions, and 401 unique
  native RVAs, then publish its typed-quality report.
- [x] Asset-bind public Confessor to managed `Confessor`; native-audit all nine
  declared methods plus acted-record, status, registered-alignment, animated-
  art, and exact-membership helpers; close exact direct text and native-null
  reference provenance, truth-identical Corrupted/registered-Evil behavior,
  current-Spy override, Poet absence, real/raw callback ordering, raw-bluff and
  register-as identity, identity movers, Baker-to-Spy chronology, and archive
  compatibility; expand the typed union to 38 target sets, 764 memberships,
  492 selected managed definitions, and 410 unique native RVAs, then publish
  its typed-quality report.
- [x] Asset-bind public Druid to managed `Librarian`; native-audit all ten
  declared role methods, all six compiler-generated ordering helpers, and 20
  picker, acted-record, registered-data, pool-filter, lifecycle, and RNG
  helpers; close exact three-target selection, click-order references versus
  sorted display IDs, registered-Outcast truth, the complementary authored
  false-role ladder, full lifecycle eligibility, ResetAfterNight history,
  managed-name ingestion, Poet absence, identity movers, anonymous Outcast and
  Wretch worlds, raw callback ordering, Baker-to-Spy chronology, and archive
  compatibility; expand the typed union to 39 target sets, 800 memberships,
  512 selected managed definitions, and 424 unique native RVAs, then publish
  its typed-quality report.
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
  - [x] Add the offline Demon/ordinary-Minion/Drunk selector ledger with
    occurrence-preserving pool mutations, exact rational path probabilities,
    script registration, Drunk corruption-attempt effects, and conditional
    equivalence tests against the one-Lilis prefix. Keep unsupported Reveal
    hooks, dispatch, scheduler order, and intervening writers outside this API.
  - [x] Compose the selector ledger with the bounded Lilis/Twin/Drunk Reveal
    callback projection: constant-null register-as, live-bluff guard,
    GiveBluff, repeated continuations, separate real/copied Init/AfterRoundStart
    dispatch, and Scout/Witness/Confessor callbacks with exact status targets.
    Require explicit resume/acquisition provenance and exclude subscriptions,
    HealthyBluff re-entry, view epilogues, and intervening writers.
  - [x] Add the versioned Spy register-as override with explicit data-role cache
    identity, shared/distinct-object provenance, script-occurrence weighting,
    cache reuse after script growth, and register-as updates despite live bluff.
    Preserve the v1 schema and exclude unsupported callback identities.
  - [x] Add versioned HealthyBluff Start latch provenance, status-only Drunk/Lilis
    callbacks, frozen per-trigger dispatch, resistance and repeated-Reveal
    regressions. Preserve v1/v2 serialized shapes and reject reached Twin/Spy
    Start callbacks atomically.
  - [x] Add an isolated Twin Start writer kernel with occurrence-weighted Demon
    and alive-neighbor selection, ordered InitWithNoReset effects, immediate
    action-role clones, preserved stale register-as/copied-role storage and
    explicit new continuation counts. Test self-swaps and a moved Drunk resume.
  - [x] Compose the Twin writer with one explicit Character.Act(Start), its
    one-shot guard, frozen truth decision, current copied-role reread, and
    optional second Twin swap. Preserve reset latches and unconditional mass.
  - [x] Join Character.Act(Start) composition to Reveal acquisition and later
    Init/AfterRoundStart under explicit resume/acquisition provenance. Transport
    writer-created continuation counts and validate repeated new-data resumes.
  - [ ] Establish scheduler provenance and enumerate justified interleavings,
    including branch-dependent acquisition events and omitted view epilogues.
  - [ ] Extend callback replay to remaining register-as overrides, additional roles,
    Twin/Spy Start, subscriptions, and writer-created continuations; establish
    scheduler order provenance or justified interleaving support before
    scenario integration.
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
