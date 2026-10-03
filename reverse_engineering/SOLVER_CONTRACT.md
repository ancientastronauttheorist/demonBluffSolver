# Solver contract and reachable blockers: S0, version 1

Date: 2026-10-03. Authority: [solver-first plan](SOLVER_FIRST_PLAN.md), introduced
at `92363e756f4faf6fdefc0f59a7a8cba527c5936a`. This is the frozen evaluation
contract, not an exact supported-gameplay certificate. Solver baseline: that
commit, before solver implementation changes. Existing native audits retain
their individual domains; new composition cases are development evidence.
No live control or numeric performance/win-rate target is authorized here.

## Build and evidence identity

Steam app `3749680`, build `23084916`, Windows x86-64, Unity `2022.3.10f1`,
metadata 29, build identity `f530404b0f3f_807de4a83df4`. The
[build manifest](manifests/builds/f530404b0f3f_807de4a83df4.json),
[extraction manifest](manifests/extractions/f530404b0f3f_807de4a83df4_il2cppdumper-v6.7.46.json)
and [toolchain lock](toolchain/toolchain.lock.json) pin binaries/assets,
Dumper outputs and tools. The installed app manifest agrees with the checked-in
app/build identifiers; the original plan's app ID was a transcription error.
New public reports retain normalized identifiers, not account fields or
identifying machine paths.

Every evaluation input names those manifests, exact solver commit, parser and
corpus versions, information mode, admitted deck/role/phase domain and supplied
services. Missing identity prevents promotion; changed hashes require triage.
The pool harness's synthetic CharacterData rows are authored field fixtures,
not serialized role assets or observations of actual generation.

## Player-observation history v1

Canonical input is append-only history. The envelope identifies
`schema_version`, `build_id`, `solver_commit`, `parser_version`, `corpus_version`,
`information_mode=player` and `domain_id`. Each event has strictly increasing
`ordinal`, `phase`, `action_ordinal` (null before any action), `kind`, `payload`
and `evidence_id`. Capture timestamps are optional; missing timestamps never
license inventing native scheduling from wall-clock time. Native schedule and
UI observation order remain separate provenance.

| Kind | Permitted payload and admission |
| --- | --- |
| `deck_observed` | Public role pool and displayed header counts; preserve multiplicity, obscuration and count provenance. Admit strip identities only once actually exposed. |
| `card_revealed` | Physical position, visible apparent role, exact speech, ordered public target references and parser/rule version. Record verified reveal, not click attempt. |
| `action_requested` | Selected reveal/ability/execution, actor, ordered targets and known public costs. Request and completion are distinct. |
| `ability_observed` | Actor, exact public result, target references and chronological use/reset event. Preserve all Judge history; missing result is unknown. |
| `execution_observed` | Public target/outcome/exposed role, HP/objective changes and publicly justified alignment/corruption feedback. Unexposed identity stays absent. |
| `status_observed` | Visible blocked/silenced/night-killed/protected feedback and affected positions; an unknown cause cannot become a hidden-role fact. |
| `phase_observed` | Public phase/HP/objectives and established ability resets; no pending iterator state or future RNG. |
| `terminal_observed` | Public win/loss/scoring/progression. Hidden truth used to grade belongs to the oracle lane. |

Each evidence ID resolves to a capture/provenance record proving availability
at that prefix. Memory transcription requires a prior visibility gate and UI
cross-check. An ID or readable field alone is insufficient. Retain noisy or
missing data explicitly; evidence-backed corrections reference the original
event, rather than rewriting history to match oracle truth.

Privileged roles, evil positions, true corruption/truth flags, Start targets,
unrevealed choices, queue state and game RNG belong to a separate oracle
artifact. In particular existing `GameState.pd_corruption_target`,
`twin_recipient_bluff_context` and `twin_recipient_bluff_prefix_context` are
offline inputs, not player-history fields. The new native actor/pool snapshots
are validation artifacts and cannot feed a fair planner as observations.

The [legacy Rust input type](../crates/solver-core/src/types.rs) mixes public and
offline fields. The new [strict history boundary](notes/systems/player_history_boundary.md)
rejects unknown/privileged JSON fields, binds exact events and prefixes to a
separate trusted UI-review registry, and preserves chronological results. Its
mechanical legacy projection admits the named single-Day Judge domain and
conditional initial-Day Hunter/Baa domain; it rejects unmodeled transitions
rather than flattening them. The registry's
caller must actually review captures; the module does not inspect pixels.
CLI/live capture integration and a certified rules-domain adapter remain open.
Paired hidden worlds with identical permitted histories must yield identical
planner inputs and policy distributions under the same planner seed. Development
tests establish input/projection equality only; policy distributions remain
unevaluated. This contract does not certify the existing legacy entry point.

## Results and provisional objective

Track model support separately from search completeness. Within a supported
model, complete search yields contradiction for zero worlds, unique for one,
ambiguous for multiple. Incomplete search, unsupported rules or uncertain
extraction cannot prove emptiness, uniqueness, optimality or impossibility.
Attach evidence-linked contradictory subsets when feasible. Probability
requires reviewed generation/path and clue likelihoods; unknown distributions
stay unknown, and uniform worlds are not an implicit confidence model. Retain
failure/rejection mass in unconditional comparisons.

Small-case certificates compare complete canonical worlds at every prefix;
larger cases report explored nodes/worlds, budgets and sound bounds when
available. Uncertified policy is `best_found`. Initial offline policy objective
is provisionally one-village win probability only where path weights are
justified. Else compare legality/deductions and expose bounds. Full-ascension
return, HP priority, risk preference, mode/ascension scope and numeric budgets
remain open user choices. Contract/frontier work does not depend on resolving
those choices or imply live autonomy.

## Reachable subsystem matrix

Accounted means named, not recovered; reconstructed means an authored replay
exists; validated identifies the exact comparison and supplied services;
integrated means the relevant solver/player-history path consumes it.

| Subsystem | Accounted / reconstructed / validated domain | Integration and remaining gate |
| --- | --- | --- |
| Observation/extraction | [Strict history boundary](notes/systems/player_history_boundary.md): typed history, exact-prefix trusted review, chronological result preservation and paired-oracle development input checks. | Exported Rust module with narrow Judge and conditional Hunter/Baa snapshot projections; archived pixel review records admission gaps. Trusted capture/CLI/live integration and held-out evaluation remain absent. |
| Deck/pool generation | [Actual pool composition](notes/systems/manage_pool_composition.md): caller/builders/getters/filters/predicate with supplied collection/RNG/layout. [Original N5 factors](notes/systems/first_village_bluff_generation.md) preserve native roster/pools and separately invoked Minion catalogue support. | [Ledger bridge](notes/systems/manage_pool_ledger_bridge.md) is offline with later-setup provenance required. N5 factors are uncombined and unweighted; actual acquisition caller, generation-to-observation domain and held-out results remain open. |
| Initialization/Reveal scheduling | [Retained Init join](notes/systems/manage_initialization_join.md) and [publication/action join](notes/systems/manage_publication_action_join.md): actual pool/Init/Hidden/first yield/publication/generic Act routing. [Original N5 publication/Init](notes/systems/first_village_publication_init.md) retains actual generation, five constructors/Init returns/first yields, publication and five concrete Act(Init) calls, stopping before ordered Start. [Engine first-step evidence](notes/systems/unity_coroutine_bridge.md) is independent. [Conditional Hunter acquisition](notes/systems/hunter_acquisition_publication.md) retains one actor through native delayed acquisition, release, click and publication. | [Rust initialization batch](notes/systems/setup_initialization_batch.md) compares five original completed Actor/Continuation prefixes. [Setup action V2](notes/systems/character_action_setup.md) compares the original N5 semantic Init checkpoint and rejects acquisition resumes; physical publication/CPU/clone-delegate assertions remain native-only. [Scheduled Rust publication](notes/systems/scheduled_role_publication.md) matches conditional post-click checkpoints. Original ordered Start, five-owner queue admission, generated acquisition-to-observation and rendered admission remain open. |
| Role clues/truth/corruption | [Truth/status audit](notes/systems/gameplay_status_corruption_truth.md), version-bearing predicates and [independent conditional Hunter/Baa world reference](notes/systems/conditional_hunter_baa_world_reference.md). [Concrete original N5 Init/Start dependencies](notes/systems/first_village_role_setup.md) verify Confessor status 25 separately from actual lying; the retained publication/Init caller now verifies the original N5 Init effect. | Production public-history projection preserves complete worlds at all declared four/five-seat development prefixes; finished setup and availability supplied. Original N5 classes are admitted only by the offline setup-only V2/V4 bridge, with native semantic checkpoint and status mutation sensitivity. No new acquisition/Day, generation or cross-role/held-out certificate. |
| Legal abilities/actions | [Setup action audit](notes/systems/character_action_setup.md), individual role/picker evidence and guarded writer bridge. | Recommendations/automation exist; target/use/reset/history and policy certificate remain open. ActivatePick drafts stay unvalidated unless needed. |
| Execution/death/protection | [Execution evidence](notes/systems/gameplay_execution_resolution.md) supplies bounded damage/protection/terminal behavior. | Existing bookkeeping/constraints; cross-role chronology needs original before/action/after comparisons. |
| Night/phase | Bounded lifecycle, coroutine and role transitions with explicit timing contracts. | Snapshot histories and offline scheduled/clocked replays; complete village chronology/reset certificate absent. |
| Scoring/ascension | [Score lifecycle](notes/systems/score_lifecycle.md), [Standard progression](notes/systems/standard_mode_progression.md) and bounded replays. | Full-ascension belief/policy integration and frozen baseline/candidate outcomes remain open. |

All rows have **no independently frozen held-out end-to-end result** recorded
by this tranche. Existing regressions and native fixture counts are development
evidence, not substitutes. Freeze fresh families by role interaction/deck/phase,
provenance, metrics and budget before tuning. Promoted failures become
regressions and require fresh held-out families; thresholds stay unselected.

## Tranche exit and next milestone

Uncertain decision: legal clue interpretation can depend on which shared actor
identity and continuation survives setup. Separate pool/initializer fixtures
cannot establish that input. Scenario: repeated physical actor occurrences in
ManageCharacters, board-field replacement and stopped second Init. This is an
adversarial native setup-domain fixture, not proof of live board generation.

Exit test: execute actual pool builders and Init occurrences in one retained
fixture; preserve iterator bytes, current actor identity, caller-generated IDs
and failure prefixes; stop before publication. The new join closes this
necessary dependency, not S1 or a certified solver-ready gameplay domain.

The distinct publication/action audit joins generic Act Init/Start over
retained actors/pools; concrete virtual role bodies remain supplied inert
services in that audit. The Hunter checkpoint below adds one conditional concrete
producer join. Continue S1 by establishing
queue/Reveal admission through native engine contracts. Produce chronological
legal observations for one named role/deck domain and feed the strict adapter.
The independent Hunter/Baa reference compares conditional complete worlds; it
does not establish this generation-to-observation path. S2 promotion across
original-derived domains and S3 policy comparison remain required.

## Initial retained-Init checkpoint

Historical checkpoint at `05fe37b`; the current boundary and publication/action
work supersede its next-step and unimplemented-adapter statements. The later
folded-return correction intentionally removes supplied disposal events while
preserving actor/continuation/pool projections, as documented in the Init note.

```text
Build/assets / solver commit / corpus version / information mode / objective:
  Pinned manifests above; baseline 92363e756f4faf6fdefc0f59a7a8cba527c5936a;
  manage_initialization_join_v1; offline native development validation;
  setup-to-observation dependency closure, no policy evaluation.
Named decision blocker and reachable deck/phase/role scenario:
  Setup actor/clone/continuation identity used by later clue interpretation;
  authored repeated-body Manage Init scenario, live generation not established.
Before -> after supported behavior:
  Separate pool and Init fixtures -> retained native pool-to-Init transaction
  before publication, including board replacement and partial callback stop.
Original evidence; supplied runtime/scheduler boundaries:
  Actual pinned native bodies; collections/RNG/UI/clone/coroutine services
  supplied as declared in the join note; no engine queue composition yet.
Differential attempted / admitted / passed / failed / excluded by subsystem:
  Native join: 8 / 8 / 8 / 0 / 0 fixtures; 16 attempted Init, 12 completed.
  Rust first-yield projection: 8 / 2 / 2 / 0 / 6 fixture families/stages;
  admitted two successful cases compare four complete Init calls.
  7 initializer and 8 batch Rust tests passed; Python syntax checks passed.
  Both new join and existing 760-case pool reports reproduced byte-identically.
World-set soundness/completeness; ambiguity/contradiction/unsupported results:
  Not evaluated. Partial native calls and state-zero scheduling remain outside
  the Rust primitive domain; strict player-history adapter is unimplemented.
Policy certificate or best-found budget; prior/likelihood assumptions:
  No policy claim, selected budget or assumed world prior.
Held-out outcomes and latency/memory versus frozen baseline:
  Not evaluated; new cases are development evidence. No win-rate claim.
New uncertainty, regressions and remaining exclusions:
  Native setup scheduling/publication/action/Reveal join still incomplete;
  app-ID transcription corrected against authoritative manifests.
Next end-to-end milestone; continue / stop / reprioritize rationale:
  Continue S1 publication and supported actions to legal observations;
  the named retained-state dependency closed, justifying the next tranche.
```

## Boundary, publication and deduction checkpoint at 810f655

2026-10-03. Candidate source/reference commit
`810f655f6fe9bccc13afb041b76b84fca37c7918`; strict history implementation
commit `6e68102a4e63456aae3e07748f5490bc4c0708b9`. The original planning
baseline remains `92363e756f4faf6fdefc0f59a7a8cba527c5936a`.

```text
Build/assets / solver commit / corpus version / information mode / objective:
  Pinned build/assets above; candidate and baseline commits above.
  manage_publication_action_join_v1, player_history_v1 and conditional
  Hunter/Baa development v1; native/oracle validation separated from
  reviewed player inputs. Dependency closure and exact conditional deduction;
  no village-policy evaluation.
Named decision blocker and reachable deck/phase/role scenario:
  Retained setup identity through publication/generic action routing;
  illegal hidden-data admission or temporal flattening in input;
  complete possible worlds for supplied four/five-seat Hunter/Baa setup.
  Native repeated-body fixture and conditional roster are development domains,
  not a claim of actual live generation or complete phase legality.
Before -> after supported behavior:
  Init-only native join -> retained publication, board rereads and generic
  Init/Start routing with exact stopped prefixes and retained continuations.
  Legacy snapshot only -> exported strict reviewed-history Rust boundary and
  deliberately narrow single-Day Judge compatibility projection.
  Local clue tests -> independent complete canonical-world comparison at
  every prefix of the declared Hunter/Baa finite family.
Original evidence; supplied runtime/scheduler boundaries:
  Actual native pool/Init/publication/Act/RoleAct/CheckLying bodies; supplied
  collections, clone metadata, UI, ordinary inert virtual roles and scheduling.
  Hunter/Baa reference rules derive from reviewed native-static evidence;
  finished setup, bluff acquisition, public multiplicity and board order supplied.
  History fixtures have trusted authored UI-review registrations; actual image
  review/capture authenticity remains the caller's responsibility.
Differential attempted / admitted / passed / failed / excluded by subsystem:
  Native publication/action: 161 / 161 / 161 / 0 / 0 fixtures;
  322 Init attempts, 321 complete; 266 Act attempts, 154 complete.
  44 selected instruction assertions, 1,284 native addresses, 103 snapshots.
  New native report independently reproduced byte-identically; original
  760-case pool report reproduced byte-identically by the native worker.
  Strict input: 14 tests passed; initializer: 7 tests passed.
  Corrected conditional reference: 7 tests passed; all 1,392 histories and
  8,160 generated prefixes compared, plus four coherent contradiction fixtures.
  Legacy simulation: all 34 tests passed; release build passed.
World-set soundness/completeness; ambiguity/contradiction/unsupported results:
  Exact set equality at every generated conditional prefix: 4,376 ambiguous,
  3,784 unique, no unexplained disagreement. Four authored coherent impossible
  histories yielded complete empty sets. Malformed/richer histories explicitly
  unsupported by the reference gate; missing/unmodeled inputs distinguished
  by the history projection. These are bounded development checks, not a
  certificate for the legacy CLI or an original generation-to-observation path.
Policy certificate or best-found budget; prior/likelihood assumptions:
  No policy certificate, assumed uniform-world prior or measured objective gain.
  Reference-only linear-distance mutation changes 1,104 complete histories;
  production source was not mutated. Paired-world checks establish input
  equality, not unimplemented policy-distribution equality.
Held-out outcomes and latency/memory versus frozen baseline:
  No independently frozen held-out outcome or fixed-budget baseline comparison.
  Legacy regressions retain documented exclusions; passing them is not a
  fair-information win-rate or calibration result.
New uncertainty, regressions and remaining exclusions:
  Review caught folded-return interception and cross-history memory gates;
  fixes preserve native state and bind each transcription to its exact prefix.
  First reference run failed two tests because its projection omitted public
  Hunter multiplicity; corrected projection preserves the original family
  and expected worlds. Production deduction rules were unchanged.
  Concrete native clue generation, mixed engine queue readiness, resumed Reveal,
  trusted capture/CLI/live integration, temporal rules and policy remain open.
Next end-to-end milestone; continue / stop / reprioritize rationale:
  Continue S1 actual Character.Act(Day) -> Tracker real/bluff generation ->
  captured callback and result/speech publication over retained distinct actors.
  Reuse existing callback/publication audits, with supplied schedule explicit.
  Gate player admission on pinned reviewed captures establishing full public
  deck multiplicity, board IDs, reveal order and exact visible Hunter speech.
  Preserve native-only references separately from public geometric derivations.
  Necessary dependencies closed; goal remains active, S1/S2 promotion and
  S3/S4 gates remain incomplete. No live game control was used.
```

The next native consumer dependency is already bounded by the
[callback audit](notes/systems/character_role_callback.md) and
[publication audit](notes/systems/character_role_publication.md). The remaining
mixed DelayReveal/result/speech engine admission differs from the existing
[wait-queue projection](notes/systems/unity_wait_queue_projection.md).
An archived after-flip capture exposes one Hunter clue and board geometry, but
does not establish this conditional domain, full public pool, pinned capture
identity or complete reveal chronology. It cannot promote those missing facts.

## Current Hunter producer, public adapter and capture checkpoint

2026-10-03. Public-adapter commit
`6240e85eacffcf29f0e84897332bbab8cc659b49`; native producer/reference candidate is
`bb65844c0d6bc3594bc0cdde54edce1c6246fae1`. The original planning baseline
remains `92363e756f4faf6fdefc0f59a7a8cba527c5936a`.

```text
Build/assets / solver commit / corpus version / information mode / objective:
  Pinned manifests and candidate commits above; Hunter role/publication native
  development v1, conditional Hunter/Baa development v2, archived pixel-review v1.
  Oracle/native validation separate from supplied reviewed public inputs.
  Exact conditional deduction and dependency closure; no policy objective score.
Named decision blocker and reachable deck/phase/role scenario:
  Concrete Hunter Day output through captured delegate/history/speech;
  public sentence interpretation without native-only references;
  four/five-seat conditional Hunter/Baa deduction, four-seat native producer.
Before -> after supported behavior:
  Supplied clue result -> concrete native real/bluff producer and consumer join.
  Legacy fixture projection -> strict production public-history projection
  preserving complete worlds, repeated public roles and unknown HUD provenance.
  Archived clue mention -> hash-bound pixel review with admission gaps explicit.
Original evidence; supplied runtime/scheduler boundaries:
  Actual Character.Act(Day), CheckLying, RoleAct, Imp Day return, Tracker real/bluff,
  distance/register alignment/range/text selection, ActedInfo and result/speech.
  Finished setup/bluff acquisition, legal Day request, runtime/collections/UI,
  uniform-index draw contract and explicit coroutine resume schedule supplied.
  Native prior_info is retention stress, not an initial-Day public history;
  supplied runtime uses zero wraps to FFFFFFFF; Init is not executed here.
  Asset abilityUsage zero does not prove the post-initialization runtime count.
Differential attempted / admitted / passed / failed / excluded by subsystem:
  Conditional native producer: 20 / 20 / 20 / 0 / 0 completions, 12 truth/8 bluff;
  2168 exact stopped-service prefixes, 54 selected pins, 13 body fingerprints,
  1248 executed instruction/service addresses, 20 captured callbacks, 8 draws,
  8 duplicate-opposite reference completions. Two independent reruns byte-identical.
  Strict input: all 20 unit tests passed. Independent reference: all 10 tests
  passed, including native sentences in separate synthetic availability histories.
  Release build passed. Prior full 34-test legacy simulation result retained;
  unchanged legacy solver rules did not justify repeating that long suite.
  Archived captures: 2 reviewed, 0 admitted; review report byte-identical rerun.
World-set soundness/completeness; ambiguity/contradiction/unsupported results:
  Exact complete worlds at all 8160 synthetic family prefixes through production
  adapter: 4376 ambiguous, 3784 unique; four coherent contradictions empty.
  All 20 native sentence cases match independently enumerated complete worlds
  and include actual Baa in the isolated grading lane. Native refs never enter
  public targets; malformed refs cannot alter unchanged public input/worlds.
  Missing fields remain incomplete, unmodeled combinations unsupported.
Policy certificate or best-found budget; prior/likelihood assumptions:
  No policy certificate or uniform-world prior. Conditional bluff draw law
  does not establish seat-generation priors, path/failure mass or independence.
  Reference-only linear-distance mutation still differs on 1104 complete histories.
Held-out outcomes and latency/memory versus frozen baseline:
  No independent held-out family, fixed-budget outcome comparison or calibration.
  Local test durations are development checks, not performance certificates.
New uncertainty, regressions and remaining exclusions:
  Review corrected premature flag fabrication, static-write reachability prose,
  allocator label reset, CLR fixture metadata and passive-use provenance.
  Day30 preserves the start-acted flag; its pinned write is gated by trigger5.
  Actual engine immediate starts, mixed queue readiness, legal Day invocation,
  resumed acquisition, public capture/build binding and generation remain open.
  Ten-card archived mixed-role capture cannot establish this conditional domain,
  full public deck or reveal chronology. No live control used.
Next end-to-end milestone; continue / stop / reprioritize rationale:
  Continue S1: compose actual immediate result/speech coroutine starts and mixed
  queue draining with retained acquisition/actor state and explicit owner/clock/
  full-frame/generation evidence. DelayReveal acquires; Day must be a separately
  established request. Speech text before its wait is not rendered availability.
  Gate capture admission on pinned complete public deck and real reveal prefixes.
  Conditional action-policy comparison separately needs retained execution/HP/
  death-list/terminal transitions and legal UI entry; static anchors alone cannot
  establish action availability or an optimal reveal-first policy.
  Necessary concrete producer/input dependencies closed; continue broad goal.
  S1 end-to-end promotion, held-out S2 and all S3/S4 gates remain incomplete.
```

## Fresh Baker runtime, generation and scheduled Hunter checkpoint

2026-10-03. Generation evidence commit
`acf7f5fc03543237de2b13a8f45f76a559a3c7ab`; Baker correction commit
`7374437e7c2910d7b3889b18b003f3d7046996c9`; scheduled-publication and
candidate Rust commit `2c84588a0b1f190736602fee696bdc612db0cab0`.
The original full goal remains the plan at
`92363e756f4faf6fdefc0f59a7a8cba527c5936a`.

Sources: [first-village generation](notes/systems/first_village_profile_generation.md),
[scheduled Hunter publication](notes/systems/hunter_scheduled_publication.md),
[Baker](notes/roles/gameplay_role_baker.md) and
[Shaman](notes/roles/gameplay_role_shaman.md).

```text
Build/assets / solver commit / corpus version / information mode / objective:
  Same pinned build/assets; source and candidate commits above.
  first_village_profile_generation and hunter_scheduled_publication development
  corpora; native/oracle checks separate from public history. Close generation,
  chronology and clue-model dependencies; no village-policy evaluation.
Named decision blocker and reachable deck/phase/role scenario:
  Can a fresh Shaman-copied Baker on Alchemist speak original before its source?
  Profile 21689 (Standard group 2/village 1) supports the authored N8 roster:
  five Villagers, Plague Doctor, Shaman and Baa, with forced Shaman. This is
  asset/count support, not a composed generation-to-capture certificate.
  Does Hunter speech startup precede result return, and do native waits retain
  the actor/history through owner, clock, full-frame and generation gates?
Before -> after supported behavior:
  Erased Alchemist/Enlightened identity incorrectly implied nonnull runtime ->
  fresh Shaman Start preserves actual null runtime. Init clears runtime;
  Alchemist Init adds resistance, its bluff Start follows Shaman, and
  Enlightened writes runtime at Day. The N8 regression failed before the fix.
  Reveal chronology corrected: onClick invokes Reveal/Day while Hidden, then
  OnReveal records order; OnClick changes actor to Alive afterward.
  Authored resume order -> actual immediate result/speech starts and native
  queue drains under explicit engine/runtime services. Internal Rust stepper
  preserves synchronous speech startup before picker hide/result return;
  the original explicit replay contract remains supported.
Original evidence; supplied runtime/scheduler boundaries:
  Actual profile selection/materialization/reset/Standard roster bodies;
  CopyData engine class conversion and collection/random services supplied.
  Actual Hunter producers, click/Reveal/Day/result/speech bodies, managed bridge,
  native dispatcher, wait insertion/drain, callback and reference release.
  Finished acquisition, record creation, CLR/virtual dispatch services,
  owner responses, engine clock/frame/phase availability and UI services supplied.
  Initial raw UI state zero is supplied, not native UI Init. Retained snapshots
  now include initReveal, raw reveal state and global reveal count independently.
Differential attempted / admitted / passed / failed / excluded by subsystem:
  Generation: 6 retained cases, 2 copy/lifetime sensitivity cases, 960 ordered
  rosters (120 original five-role pool; 840 failed-accumulation seven-role pool),
  61 stopped prefixes; 16 body fingerprints, 1077 executed instructions.
  Scheduled publication: 34 compositions, 39 Day requests, 31 completed
  publication compositions, 173 drains and 254 exact managed stopped prefixes;
  20 managed/21 engine fingerprints, 1621 managed/690 engine addresses,
  54 inherited/16 selected/60 bridge checks and 1493 lossless pooled snapshots.
  Both final native reports independently rerun byte-identically; scheduled
  report reproduced by both root and reviewer after final snapshot repair.
  Rust: all 957 library tests, 10 small-world reference tests and 34 simulation
  tests passed; release build passed. Snapshot codec: all 4 tests passed.
World-set soundness/completeness; ambiguity/contradiction/unsupported results:
  Earlier exact 8160 conditional public-history prefix comparisons retained.
  Baker regression checks the shared history predicate, not full N8 world-set
  completeness or other-role clues. No new generation-to-public-history
  admission: repeated-Hunter N4 fixtures remain conditional supplied setup.
Policy certificate or best-found budget; prior/likelihood assumptions:
  No policy certificate, selected budget or justified world prior. Ordered
  roster enumeration under a declared draw contract is not path probability.
Held-out outcomes and latency/memory versus frozen baseline:
  No frozen independent held-out outcome, calibration or fixed-budget baseline
  comparison. Test runtimes and report compression are development checks.
New uncertainty, regressions and remaining exclusions:
  Original first Standard profile 21674 selects four of five starting
  Villagers plus ordinary Minion. The seven-role pool follows a supplied
  all-profile accumulation that fails natively; it is not normal startup.
  Minion bluff acquisition may use the wider profile Villager catalogue;
  initial five-role physical pool does not bound its public clue identities.
  Later arbitrary no-reset/non-null Baker runtime histories are unmodeled.
  Actual acquisition completion before click, full SetupDelay, native UI Init,
  rendered availability and complete public capture remain open. Same-actor
  direct double-Day cases are retention stress, not legal player chronology.
Next end-to-end milestone; continue / stop / reprioritize rationale:
  Continue S1 with one original truthful Hunter's actual DelayReveal first
  step and 0.3-second resume on the same native owner/queue. Admit its first
  click only after acquisition callback completion/release, then retain the
  existing result/speech path. This closes a mixed acquisition/publication
  dependency while other actors' acquisition stays explicitly supplied.
  For an original generated domain, compose profile 21674 GetRandomCharacters
  once through its actual Standard call, retained Manage pool prefix and both
  Minion acquisition branches. Do not silently rerun roster generation or
  restrict bluff candidates to the five starting Villagers.
  Necessary dependencies and a solver contradiction resolved; continue the
  active full goal. S1 promotion, held-out S2 and all S3/S4 gates remain open.
```

## Native checkpoint projection into scheduled Rust publication

2026-10-03. Candidate and projection commit
`abe6035b8f17ba6b1f7e6ee9502b278db93d89d9`; planning baseline remains
`92363e756f4faf6fdefc0f59a7a8cba527c5936a`.
See [scheduled role publication](notes/systems/scheduled_role_publication.md).

```text
Build/assets / solver commit / corpus version / information mode / objective:
  Same pinned build/assets; candidate above; scheduled_role_publication_v1.
  Opaque captured publication storage and original native checkpoints remain
  offline validation inputs. No native hidden identity enters player history.
  Objective: preserve synchronous speech storage and native queue readiness
  through retained Rust transitions, not optimize a village policy yet.
Named decision blocker and reachable deck/phase/role scenario:
  The old explicit replay can reach the same final state while flattening
  immediate speech startup and later Show. Conditional post-click one-result
  real/bluff Hunter cases, signed frames/generations and rejected owners.
Before -> after supported behavior:
  Manual result-then-speech order -> complete result/speech-only queue with
  explicit drains, exact pending bindings, immediate nested speech start,
  owner suppression and per-drain retained managed/queue state.
  Legacy explicit replay remains guarded by its original ordering contract;
  scheduled construction requires empty legacy orders and a false order flag.
Original evidence; supplied runtime/scheduler boundaries:
  Projection uses the expanded native scheduled Hunter report, SHA-256
  e8182a5a51353941b4d6e07d6564dc2c78f7c9e30c7e8de7366eb37aca62efcf.
  Expected states derive from its native checkpoints, with no Rust import.
  Initial typed/UI storage, first result yield, owner responses, producer
  clocks and normal lifetime/inert services remain explicit supplied contracts.
  Actual first-yield producer/acquisition/Day legality and pixels are not
  reconstructed by this Rust adapter. Initial uses zero is the old stress
  context, not a claimed native Init value or abilityUsage projection.
Differential attempted / admitted / passed / failed / excluded by subsystem:
  All 29 projected native cases and 161 drain checkpoints passed exact queue,
  callback/visit order, history/use/saved-text/Show/iterator-state comparisons.
  Excludes the five double-Day stress compositions and 254 failed-service
  prefixes; failed services are rejected, not emulated in this Rust version.
  Six adapter tests passed, including synthetic two-result generation
  suppression, owner discard, inconsistent late evidence and the 16/17-drain
  boundary. All 963 Rust library tests and the release build passed.
  Fixture values and physical bytes independently reproduced by review;
  fixture SHA-256 ade513d51eaef007c64714894c26daac9d8c7e43dd88f63e9c02d3ec1f747498.
  Privacy, relative links, schema and projection reproducibility passed.
World-set soundness/completeness; ambiguity/contradiction/unsupported results:
  No new world-set promotion. Earlier 8160 conditional public-history prefixes
  and complete reference sets retained; this transition adapter has no live,
  CLI or PlayerHistory admission caller. Owner-discarded waits remain explicitly
  distinguished from completed managed iterators. Invalid later evidence
  rejects the entire bounded batch without a valid-looking explored prefix.
Policy certificate or best-found budget; prior/likelihood assumptions:
  No policy certificate, schedule probability or assumed uniform-world prior.
  Schedules are explicit conditional inputs, not chosen or weighted by policy.
Held-out outcomes and latency/memory versus frozen baseline:
  No independent held-out outcome or fixed-budget comparison. The recent
  34-test simulation pass remains applicable to unchanged deduction/action
  rules; this offline adapter did not justify repeating the long suite.
New uncertainty, regressions and remaining exclusions:
  Original Init-derived use count, acquisition-to-first-click admission and
  original first-village bluff catalogue are being composed separately; their
  unfinished sources are not published or promoted by this checkpoint.
  Full SetupDelay/startup, native UI Init, rendered availability, complete
  public capture, correlated generation/path weights and policy remain open.
Next end-to-end milestone; continue / stop / reprioritize rationale:
  Complete and independently verify the same-owner Hunter acquisition/click
  join and bounded original N5 catalogue support, then connect the acquired
  first-result state to this queue adapter. Native constructor/Init use
  chronology must replace the old stress count in that new domain.
  Necessary solver transition dependency closed; continue the original goal.
  S1 end-to-end promotion, held-out S2 and all S3/S4 gates remain incomplete.
```

## Acquired publication and original N5 catalogue checkpoint

2026-10-03. Acquisition evidence commit
`30ba3766597de55eaac6bbc3d60b0ac00f5f3389`; original N5 evidence commit
`f04068dd8332a226afed8eba7a7002603596cd5e`; acquired Rust checkpoint commit
`4320af73f0b363baf9bf657bf9b53fb3112ec1ae`. Planning baseline remains
`92363e756f4faf6fdefc0f59a7a8cba527c5936a`.

```text
Build/assets / solver commit / corpus version / information mode / objective:
  Same pinned build/assets; commits above. hunter_acquisition_publication_v1,
  first_village_bluff_generation schema1, scheduled_role_publication_acquired_v1.
  Native identities/statuses/queues remain offline validation artifacts.
  Objective: close acquired initial-use/history provenance and identify the
  original Minion bluff catalogue before composing generation-to-observation.
Named decision blocker and reachable deck/phase/role scenario:
  Does delayed acquisition complete on the same actor/clone before first click,
  and does native Init leave use/history state matching scheduled publication?
  Conditional four-actor Hunter/Baa reset stress domain; other actors supplied.
  Separately, original Standard profile21674 N5, four Villagers/one Minion:
  are Minion bluff candidates confined to its five starting Villagers?
Before -> after supported behavior:
  Supplied standalone post-click uses0/prior history -> six acquired native
  first-result inputs with uses1/empty history, preserved record ID/generation.
  Actual Init/acquisition/result chronology is 1 -> 1 -> 0; speech storage
  precedes final picker hide/result return and later timed Show.
  N5 actual generation/pool factors witness 24 Villager selector outputs,
  including identities outside the five starting Villagers.
Original evidence; supplied runtime/scheduler boundaries:
  Actual Init/Hidden refresh/DelayReveal first step and resume/Reveal/click/
  Hunter Day/result/speech execute with one retained engine queue and owner.
  CLR, distinct clone implementation, native record creation, UI, clocks and
  other actors remain supplied; no constructor-fresh history or N4 generation.
  Actual original N5 GetRandomCharacters/Standard call/Manage pool prefix and
  separately invoked Minion selector execute. CurrentScript publication,
  Random.value, stable deferred sort and collection adapters are supplied.
Differential attempted / admitted / passed / failed / excluded by subsystem:
  Acquisition: 9 compositions, 60 drains, 192 exact stopped managed prefixes;
  6 publications and 3 rejected owners. Independent final native reproduction
  matched physical bytes SHA3603e404ea1a55ff481fe075bfef36bfe5d339b6c9738bba4632bfc7c4f5c9d8.
  N5 separate factors: 2880 generation-index histories, 120 ordered rosters,
  5 subsets, 1125 pool rows, 8960 selector rows and 573 stopped prefixes.
  24 complete catalogue witnesses; 48 fallback occurrences, each asset twice.
  Independent native rerun matched physical bytes SHAd4aa00d6b99df42b8140beb7b725305a5520dd32dddc639754315cc4ca6c0c47.
  Rust acquired projection: all6cases/24drain checkpoints match. Old29/161
  fixture remains byte-identical. All8 adapter tests and all965 library tests
  passed; no production Rust change, prior release build retained.
  Both fixture projections independently reproduced values/physical bytes;
  privacy, source/prior hashes and 44 relative links passed across 12 files.
  Rust excludes 3 rejected acquisition owners (no first click) and 192 failed
  services. N5 factors exclude their uncombined Cartesian histories.
World-set soundness/completeness; ambiguity/contradiction/unsupported results:
  No new world-set or PlayerHistory admission. Acquired Rust context starts
  after the native result first yield; it does not replay acquisition itself.
  Rejected-owner release stays distinct from managed completion. Existing
  conditional complete-world reference guarantees remain unchanged.
Policy certificate or best-found budget; prior/likelihood assumptions:
  No policy certificate, recovered PRNG law, joint weights or assumed uniform
  prior. Separate catalogue support factors are not weighted full histories.
Held-out outcomes and latency/memory versus frozen baseline:
  No independent held-out outcome or fixed-budget comparison. Existing34
  simulation results retained for unchanged deduction/action rules; new965
  library pass and native reproduction are development validation only.
New uncertainty, regressions and remaining exclusions:
  N5 selector invocation stops outside actual Init/Reveal caller composition;
  all24 copied-role behaviors remain future mixed-role obligations.
  Full SetupDelay, actor construction, concrete ordered writers, original UI
  Init, deck-list/HUD publication, rendered availability and reviewed public
  chronology are not established by these joins. All eight global gates remain.
Next end-to-end milestone; continue / stop / reprioritize rationale:
  Retain generation row0 -> pool row0 with current Villagers
  [Confessor21614,Lover21626,Hunter21621,Enlightened21618], Minion21596 and
  sorted return[21596,21614,21626,21621,21618]. Duplicate die1/index2 selects
  Hunter21621 without script registration. This is an exact existing witness.
  Execute actual five Manage/Init occurrences and their first delayed yields
  without rebuilding pools or replacing actors; compare retained asset/clone,
  status/history/use/ID and continuation state at pre-publication handoff.
  Continue the live Manage CPU/heap, stopping at36D01E; B's saved pool views
  retain heap state but not CPU context. Restoring those views and issuing
  standalone Init calls would not prove this caller continuation.
  Character.Init is distinct from Act(Init3); concrete role OnInit dispatch
  occurs after this stop and is a separate next boundary.
  Then compose concrete Init/Start effects, including Confessor OnInit, actual
  acquisition and public deck/clue publication before world-set admission.
  Two dependencies closed with explicit boundaries; continue the original
  active goal. S1 promotion, independent held-out S2 and S3/S4 remain open.
```

## Original N5 retained initialization checkpoint

2026-10-03. Native initializer commit
`64ad44867f5158ebe0b5c7ed0ecaa838a5e6e8b5`; concrete role dependency commit
`218d58d020dea3f9a2b216a7e484b40adec2b792`; Rust comparison commit
`16d67d2897f4b15d26a32f347d9a924a5d5ab82b`. Original planning baseline remains
`92363e756f4faf6fdefc0f59a7a8cba527c5936a`.

```text
Build/assets / solver commit / corpus version / information mode / objective:
  Same pinned build/assets; commits above. first_village_initialization_v1,
  first_village_role_setup_v1, five-prefix offline Rust fixture version1.
  Native actor/source/status/continuation identities are oracle validation
  evidence, never legitimate player observations or planner input.
Named decision blocker and reachable deck/phase/role scenario:
  Can original profile21674 generation row0/poolrow0 reach five actual Init
  returns in the same Manage frame before publication, preserving earlier
  actors, clones and waits? Sorted N5 roster is Minion, Confessor, Lover,
  Hunter, Enlightened. Does later concrete Confessor Init change actual truth?
Before -> after supported behavior:
  Pre-first-Init catalogue witness -> native constructors and five retained
  caller Init returns, with descending IDs5..1 and distinct first-yield waits.
  Primitive post-Init effects -> five source-derived Actor/Continuation prefix
  comparisons in Rust. Separate supplied-state concrete role audit establishes
  Confessor AppearTruthfull25 without repairing Corrupted/Evil actual lying.
Original evidence; supplied runtime/scheduler boundaries:
  Actual generation/Standard, pool builders, constructor, Init, Hidden refresh,
  RefreshView and DelayReveal state0 execute. One live CPU/heap continues to
  36D01E before publication; integer/XMM nonvolatiles, caller SP/root sentinel,
  earlier actors/lists/backings/iterators/waits/clones and source/pool graphs
  are asserted. CLR/base constructor, scene/status/UI hydration, raw scene
  state20, role clone and synchronous first-step entry are supplied services.
  Concrete role audit independently supplies post-Init actors and virtual
  slots; it is not a retained continuation of the generated initializer audit.
Differential attempted / admitted / passed / failed / excluded by subsystem:
  Native original initialization:5ctors/5Init/5firstyields,583Manage services,
  115new-service exact stopped prefixes,2071instruction addresses/13pins.
  Concrete role audit:38completed calls/320exact stops,17memberships at13RVAs;
  2143visited addresses include generation/pool provenance, not full branches.
  Independent full native producers exited0 and matched frozen report bytes:
  SHA855f8976081da9541161bcdf15e0a130cbd6f0115f2351c027708b6c3fb08eb1;
  SHAc1985696bd79303bc71293e90441bc8dc4ba6108e22570dc71450876a176e1b7.
  Rust:25whole Actor and15Continuation checkpoints across5successful prefixes;
  all9batch tests and all966library tests passed. No production Rust change.
  Fixture independently reproduced bytes/values, SHA0efc7e616901c64415101721266e9c0e471c946aa5bd10676806b1d82ad2d223.
  Privacy, source/prior/target hashes and relative links passed. Failed native
  service partial mutations, physical list/CPU/wait-duration/pool equivalence
  are not compared by this Rust adapter; native assertions retain their scope.
World-set soundness/completeness; ambiguity/contradiction/unsupported results:
  No world-set or PlayerHistory promotion. Synthetic Init/Publish failures only
  delimit successful prefixes; they do not claim native exceptions or actual
  publication. Slot labels are supplied positions, not certified UI ordering.
  Existing setup action bridge rejects original N5 classes outside its enum;
  no accepted-world solver counterexample found in this concrete dependency.
Policy certificate or best-found budget; prior/likelihood assumptions:
  No policy, recovered PRNG law, joint generation/path weights or uniform prior.
Held-out outcomes and latency/memory versus frozen baseline:
  No independently frozen held-out outcome or budgeted baseline comparison.
  Library validation finished20.75s; prior release build and34simulation pass
  retained for unchanged production deduction/action rules. No new mutation
  campaign or optimality/perfection claim is justified by these comparisons.
New uncertainty, regressions and remaining exclusions:
  No full SetupDelay/engine acquisition drain, native UI Init/deck/HUD/pixels,
  all24copied-role behavior, legal Day request/public capture or S1 admission.
  Separate concrete Start probes do not establish original ordered-Start calls.
Next end-to-end milestone; continue / stop / reprioritize rationale:
  Continue the same five-Init caller through actual publication and concrete
  Confessor ActInit/status writer, retaining original source/runtime roles,
  status backing and first waits. Then connect each correct acquisition owner
  through actual resume and mixed queue chronology; never substitute prepared
  post-Init actors while claiming original generated acquisition.
  Necessary dependency and exact Rust prefix comparison closed. Continue the
  full active goal; all eight gates, S1 promotion, held-out S2 and S3/S4 remain.
```

## Original N5 publication and concrete Init checkpoint

2026-10-03. Native retained caller commit
`8100947e4afe076c30b2990ab85f63dea1254dc3`; Rust semantic integration commit
`2a4ead4a5b1dbe795a371a83e41c58b994add53d`. Original planning baseline remains
`92363e756f4faf6fdefc0f59a7a8cba527c5936a`.

```text
Build/assets / solver commit / corpus version / information mode / objective:
  Same pinned build/assets; commits above. first_village_publication_init_v1,
  independent semantic fixture v1, setup_action_bridge_native_v2 and embedded
  bounded_setup_callbacks_native_v4. Offline oracle validation only; privileged
  roles/statuses/native identities do not become player history or policy input.
Named decision blocker and reachable deck/phase/role scenario:
  Does the original generated N5 caller publish the same five actors and run
  concrete Init without losing the first waits or conflating Confessor's
  appearance with actual truth? Minion, Confessor, Lover, Hunter, Enlightened.
Before -> after supported behavior:
  Stop36D01E before publication -> actual Gameplay.UpdateCharacters, distinct
  shallow-copy CurrentCharacters list and five actual Character.Act(Init3)
  returns, stopping36D0F9 before ordered Start. Confessor status[25]/version2;
  other active lists empty/version1. Uses1, empty histories, saved fields,
  five distinct callbacks/first waits and original sources/pools remain asserted.
  Previously rejected N5 classes -> explicit setup-only Rust semantic state,
  compared with native actors/bodies/Init traces/data/order/pools/pending owners.
Original evidence; supplied runtime/scheduler boundaries:
  Same continuous original generation/pools/constructors/Init/first yields and
  live Manage CPU/heap. Actual publication store/barrier, CheckLying, RoleAct,
  concrete class Init and status bodies. Source vtable/MethodInfo hydration is
  a named pre-entry provider; clone, CLR collection, delegate/allocation/log,
  scene/UI services and synchronous first-step scheduling remain supplied.
  Native invariants protect earlier actors/list backings/source+clone fields,
  pools/rosters/iterators/waits, integer/XMM nonvolatiles and caller stack.
  Rust carries semantic effects; it does not reconstruct physical publication
  storage, callback/closure identity, native CPU or status-list versions. A
  separate version-delta assertion derives insertion from initializer version.
Differential attempted / admitted / passed / failed / excluded by subsystem:
  Native5ctors/5Init/5firstyields/5ActInit;652Manage services, complete69-service
  postpublication suffix and69exact stopped prefixes at ordinals584..652.
  2393visited instruction addresses,25selected postpins,59body memberships,
  351pooled snapshots and11source hashes. No unexplained admitted mismatch.
  Independent native producer exited0 and reproduced exact report bytes,
  SHA31929f003a87c6e0ae4242ea9dc3eb090fba00bb641eedad0526c1afc651b48e.
  Independent no-Rust projection/reviewer reproduced the one checkpoint bytes,
  SHAd6c639806e6d2afda5e22853ca6906eeb067dca7a82e414fe6c9df844b2f0326.
  All10focused bridge and970library tests passed; release build passed.
  Confessor insertion mutation25->26 failed the corresponding native semantic
  checkpoint test with exit101; exact original source restored before final
  library run. Legacy types, copied-callback bypass, absent latches, invalid
  pools and all acquisition resumes reject. Concrete Twin Start remainsV3-only.
  Privacy, source/prior hashes, links, reproduction and projection checks pass.
World-set soundness/completeness; ambiguity/contradiction/unsupported results:
  No new world-set, PlayerHistory or S1 admission. UI booleans and slot labels
  are declared adapter inputs. replay_init_prefix ends before ordered Start;
  separate Start/latch probes do not establish original Start scheduling.
Policy certificate or best-found budget; prior/likelihood assumptions:
  No policy certificate, original PRNG law, joint path mass or uniform prior.
Held-out outcomes and latency/memory versus frozen baseline:
  No frozen held-out outcome or budgeted baseline comparison. Final library
  suite16.44s; prior34simulation results retained for unchanged production
  deduction/action rules. Compile time and fixture totals are not policy claims.
New uncertainty, regressions and remaining exclusions:
  Actual ordered Start, runtime onSetup subscription, Shuffle, native engine
  five-owner queue admission/drains/resumes, acquisition, Day/public capture,
  pixels/HUD and complete generation-to-legal-history remain open. All eight
  global gates remain open; no optimality or perfection claim.
Next end-to-end milestone; continue / stop / reprioritize rationale:
  Extract the actual serialized Characters ordered array with full object
  consumption and exact asset bindings. Continue original caller through that
  scan; independently verify row0's five assets produce zero Start calls.
  Register each original first yield through the reviewed engine bridge before
  its Init returns, preserving five distinct physical owners in one queue.
  Smallest next exit36D2DB before unknown onSetup read; do not substitute an
  empty Start array, restart state1 iterators or replace generated actors.
  Necessary setup dependency and tested Rust integration closed. Continue the
  full active goal through legal observations, independent S2 and S3/S4.
```
