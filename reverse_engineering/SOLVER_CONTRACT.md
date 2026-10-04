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
The [conditional deduction API](notes/systems/conditional_player_deduction.md)
reports complete finite four/five-seat Hunter/Baa worlds, explicit supplied
assumptions, unique/ambiguous/contradictory conclusions and incomplete/unsupported
inputs. It does not turn a mechanical Judge adapter or the proposed mixed N5
family into a certified complete model.
CLI/live capture integration and a certified rules-domain adapter remain open.
Paired hidden worlds with identical permitted histories must yield identical
planner inputs and policy distributions under the same planner seed. Development
tests establish input/projection and conditional deduction equality; policy distributions remain
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
| Deck/pool generation | [Actual pool composition](notes/systems/manage_pool_composition.md): caller/builders/getters/filters/predicate with supplied collection/RNG/layout. [Original N5 factors](notes/systems/first_village_bluff_generation.md) preserve native roster/pools and separately invoked Minion catalogue support. | [Ledger bridge](notes/systems/manage_pool_ledger_bridge.md) is offline with later-setup provenance required. The retained acquisition witness below joins one original Minion caller and recorded source choice; N5 joint weights, generation-to-observation domain and held-out results remain open. |
| Initialization/Reveal scheduling | [Retained Init join](notes/systems/manage_initialization_join.md) and [publication/action join](notes/systems/manage_publication_action_join.md): actual pool/Init/Hidden/first yield/publication/generic Act routing. [Original N5 publication/Init](notes/systems/first_village_publication_init.md) retains generation through five concrete Init calls. [Original Start/queue](notes/systems/first_village_start_queue.md) adds the 15-entry scan, 75 comparisons, zero Start calls and five admissions. [Conditional Shuffle admission](notes/systems/first_village_shuffle_admission.md) adds normal return and a sixth wait under supplied onSetup=null. [Original subscriber binding](notes/systems/first_village_on_setup_binding.md) supplies static/asset evidence. [Installed subscriber admission](notes/systems/first_village_subscriber_admission.md) executes OnEnable, Manage callback and nested first steps through normal return with eight records/seven owners, under named lifecycle, delegate-list, audio-null and visual providers. [Retained N5 acquisition](notes/systems/first_village_retained_acquisition.md) executes five animation resumes and five card acquisitions on the same queue, retaining Audio/Shuffle waits. [Conditional Hunter acquisition](notes/systems/hunter_acquisition_publication.md) joins delayed acquisition to publication. | [Rust initialization batch](notes/systems/setup_initialization_batch.md) compares five completed prefixes. [Setup action V2](notes/systems/character_action_setup.md) compares N5 semantics, all ordered comparisons, a separate registered callback request/return and eight first-wait timings; the older scheduled V1/V4 boundary rejects mixed-queue acquisition. [Guarded non-Day setup Reveal](notes/systems/non_day_setup_reveal.md) adds V5/V2 semantic support with complete auxiliary-queue guards; [independent retained-native comparison](notes/systems/first_village_retained_acquisition_projection.md) matches setup, eight preceding queue drains and the recorded Confessor acquisition path while retaining all six supported Rust outcomes. Subscriber bodies/effects, physical publication/CPU/delegates/engine ownership remain native-only. [Scheduled Rust publication](notes/systems/scheduled_role_publication.md) matches conditional post-click checkpoints. Animation/acquisition native dependency is now retained and reproduced. Shuffle state1/events, Day readiness, generated legal history and rendered admission remain open. |
| Role clues/truth/corruption | [Truth/status audit](notes/systems/gameplay_status_corruption_truth.md), version-bearing predicates and [independent conditional Hunter/Baa world reference](notes/systems/conditional_hunter_baa_world_reference.md). [Concrete original N5 Init/Start dependencies](notes/systems/first_village_role_setup.md) verify Confessor status 25 separately from actual lying; the retained publication/Init caller now verifies the original N5 Init effect. | Production public-history projection preserves complete worlds at all declared four/five-seat development prefixes; finished setup and availability supplied. Original N5 setup is admitted by offline V2/V4; V5 adds bounded non-Day acquisition callbacks and V2 scheduling guards, with copied-Alchemist mutation sensitivity. The separate native witness executes the recorded copied-Confessor path. No Day, legal-observation, world-set or cross-role/held-out certificate. |
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

## Original N5 ordered Start and first-wait admission checkpoint

2026-10-03. Native retained caller commit
`5f64e1db3f834fdb5a2a23cfb04dafc0c5043b2e`; Rust semantic integration commit
`6c9be5bc74828229525a1246c6a8cdc3e10532e7`. Planning baseline remains
`92363e756f4faf6fdefc0f59a7a8cba527c5936a`.

```text
Build/assets / solver commit / corpus version / information mode / objective:
  Same pinned build/assets; commits above. first_village_start_queue_v1;
  setup_action_bridge_native_v2 and setup-only bounded callbacks V4 unchanged.
  Offline oracle validation only. Native identities, true roles and statuses
  remain outside legal player history and planner input.
Named decision blocker and reachable deck/phase/role scenario:
  Can the original row0/poolrow0 N5 caller execute its genuine ordered Start
  array while admitting each original first wait into the same native queue?
  Minion, Confessor, Lover, Hunter and Enlightened; no actor replacement.
Before -> after supported behavior:
  Original generation/publication/Init -> actual 15-entry serialized Start scan,
  75 pointer comparisons and zero Start calls. Each original state-zero iterator
  is stepped/admitted before its Init returns, preserving five distinct supplied
  native owners in one queue. Stop before onSetup read36D2DB.
  Rust matches represented actors/bodies/Init calls/data/order/pools/pending,
  all75comparisons and five derived admission timings. Physical status versions
  remain native storage facts; a separate insertion-delta check covers them.
Original evidence; supplied runtime/scheduler boundaries:
  Full332-byte serialized Characters object consumed, exact script/asset links,
  original15Start assets and20source bindings. Same retained original Manage,
  constructors/Init/publication/ActInit; actual engine synchronous first step,
  managed MoveNext, wait producer, insertion tree and reference-release bodies.
  Runtime source/class hierarchy hydration, owner/record creation, GC handles,
  runtime invocation, clocks and yield mirrors are named supplied services.
  Native payload/iterator/GC/owner/tree identities, reciprocal links, queue keys,
  actors/status backing/source+clone/pools and outer CPU/stack remain asserted.
  Fresh normal versus fresh paused execution has exact ledger/graph/CPU parity.
Differential attempted / admitted / passed / failed / excluded by subsystem:
  Five constructors/Init/first yields/ActInit/owners/queue records;15order entries,
  75comparisons,0Start calls. 913paused/reentered service prefixes comprise
  908Unicorn and5Python-direct entries;913reentry tokens consumed exactly once.
  These are distinct prefixes, not913independent failure/exception scenarios.
  Thirteen fresh restart-abort representatives cover new families/phases.
  2479managed+394engine addresses,62body memberships,9selected local pins,
  382pooled snapshots,15source hashes. No unexplained admitted mismatch.
  Independent native producer reproduced complete report bytes, SHA
  7439fd1370909aa79bf2575b101ff8e943941fa148bb75e15254f822a14b3a60.
  Independent no-Rust projection and reviewer reproduced fixture bytes, SHA
  8c30de632c902477f14d28d84640903143ff0ee3a1f69d8015309d8c977324e9.
  All11focused bridge tests and final971library tests passed (51.66s).
  Frame increment1->0 failed the new checkpoint with frame7versus8, exit101;
  exact original source bytes restored before final library validation.
  Source/prior hashes, privacy, links, prefix and projection checks passed.
World-set soundness/completeness; ambiguity/contradiction/unsupported results:
  No new world-set, PlayerHistory or S1 admission. V4 rejects eligible acquisition
  even with every callback boundary supplied. Synthetic predeadline retention
  is adapter compatibility, not original native drain chronology. UI/slot labels
  remain supplied adapter facts. The unused15classes gain no concrete Start claim.
Policy certificate or best-found budget; prior/likelihood assumptions:
  No policy certificate, recovered PRNG law, joint path mass or uniform prior.
Held-out outcomes and latency/memory versus frozen baseline:
  No frozen held-out or budgeted outcome comparison. Library elapsed time above
  is validation only. Prior release build and34simulation results retained for
  unchanged production deduction/action rules; no optimality/perfection claim.
New uncertainty, regressions and remaining exclusions:
  Runtime onSetup installation/absence, Shuffle first step/admission and later
  events, actual drains/resumes/acquisition, Day/public capture/pixels/HUD and
  complete generation-to-legal-history remain open. All eight global gates open.
Next end-to-end milestone; continue / stop / reprioritize rationale:
  Continue same retained caller conditionally with explicitly supplied runtime
  onSetup=null; no reviewed installer/clearer establishes original absence.
  Execute actual Shuffle state-zero0.5f first wait, sixth distinct manager owner,
  heterogeneous queue admission and normal Manage return. Preserve earlier five
  actors/waits and full ABI; do not restart or substitute prepared iterators.
  Later Shuffle state1/events and acquisition/public history remain separate.
  This tranche closed a necessary original setup dependency with tested Rust
  integration. Continue full active goal through legal observations, S2 and S3/S4.
```

## Conditional Shuffle return and original subscriber checkpoint

2026-10-03. Native conditional return commit
`c999e60ce0c09226209325a672776fef58a80b92`; original subscriber static/asset
evidence commit `0491eeaf170af07c5847ae20984f291617a521bb`; Rust checkpoint
commit `4ff4dd84851285e256f52c0d09343e868ee45c48`. Planning baseline unchanged.

```text
Build/assets / solver commit / corpus version / information mode / objective:
  Same pinned build/assets; commits above. first_village_shuffle_admission_v1,
  first_village_on_setup_binding_v1 and independent conditional semantic fixture.
  Offline oracle validation; no privileged state becomes planner/player input.
Named decision blocker and reachable deck/phase/role scenario:
  Can same original row0/poolrow0 N5 caller return without losing five earlier
  waits, and is runtime onSetup really absent? Minion, Confessor, Lover, Hunter,
  Enlightened; original scene manager137026 and animation component137027.
Before -> after supported behavior:
  Pre-onSetup stop -> actual normal Manage return with explicitly supplied null
  callback, actual Shuffle state-zero first step and sixth heterogeneous wait.
  All five earlier actors/clones/statuses/publication/waits remain unchanged.
  Unknown installer -> exact original enabled animation component/ancestor chain
  and static OnEnable/OnDisable Combine/Remove binding to Animates. Its first-state
  trace starts nested audio .4f and outer animation .05f before manager Shuffle.
  This changes the next action; six records do not certify configured scene queue.
Original evidence; supplied runtime/scheduler boundaries:
  Same retained generation/pools/constructors/Init/publication/ordered scan.
  Actual Shuffle376B00 state0 -> .5f/state1 via immediate managed bridge and
  actual engine type/wait/tree/release bodies. Sixth manager owner supplied,
  distinct from five card owners. Exact native payload/GC/node/owner links and
  heterogeneous live timing retained; stack/nonvolatiles/XMM6..15 checked at
  normal Manage return36D356/root sentinel. Current-yield mirror remains supplied.
  Original84-byte animation object consumed fully with exact script/manager
  references and five active authored ancestor GameObjects. Seven raw-backed
  native intervals,41selected operands,9metadata joins,4exact float literals.
  Binding audit executes zero native bodies. Lifecycle/invocation-list/delegate,
  physical pivots/tween/audio effects remain providers, not recovered execution.
Differential attempted / admitted / passed / failed / excluded by subsystem:
  Conditional native5ctors/Init/ActInit/card first yields plus1Shuffle firstyield;
  6owners/records,15ordered entries/75comparisons/0Start/0onSetup calls.
  935services =935paused/reentered prefixes/tokens:929Unicorn+6direct providers.
  22tail services;15fresh representative tail-family aborts. These are distinct
  prefix/reentry checks, not935independent failure or managed-exception cases.
  2532managed+394engine addresses,63body memberships,12local pins,16source hashes,
  393pooled snapshots. No unexplained admitted mismatch. Prior corpus unchanged.
  Independent native producer and reviewer reproduced exact report bytes, SHA
  512c221a8362d0fcb1f43cfb0f553d2dd3f63f86a9b7ae8105180b50aeac1565.
  Independent no-Rust projection/reviewer reproduced conditional fixture bytes,
  SHAedad333793ab1c70d1bba35856445b2e4cd0c582413114e3032287f7bf5c3bf2.
  Original binding report independently reproduced byte-exact, SHA
  d7fbd2a7160f2d1f8a8c5a89bd2f35e84952bd94dca60593c8c258b9b0b69dd0.
  All12focused bridge tests and final972library tests passed20.56s. Hashes,
  raw prefixes, metadata/component joins, privacy and links passed. Production
  deduction/action kernels unchanged; previous status/frame mutation evidence
  remains scoped to its existing represented rules, no new campaign claimed.
World-set soundness/completeness; ambiguity/contradiction/unsupported results:
  No world-set, PlayerHistory or S1 promotion. Full action replay ends at modeled
  Start; generic caller return under null is tested separately. Native Shuffle
  state1 is separate from generic registration state0. Mixed queue cannot enter
  five-acquisition scheduled Reveal; aligned-cursor rejection isolates extra wait.
  Predeadline queue compatibility is synthetic, not original native drain history.
Policy certificate or best-found budget; prior/likelihood assumptions:
  No policy certificate, PRNG-law recovery, joint path weights or uniform prior.
Held-out outcomes and latency/memory versus frozen baseline:
  No independently frozen held-out or budgeted outcome comparison. Library time
  is validation, not policy performance. Prior release build/34simulation results
  retained for unchanged production rules. No optimality or perfection claim.
New uncertainty, regressions and remaining exclusions:
  Enabled authored component does not prove runtime lifecycle/invocation list.
  Actual subscriber dispatch/first-step admission, audio event200 effects,
  transform/tween mappings, later animation/Shuffle callbacks, acquisition,
  Day readiness and original-to-legal-history remain open. All eight gates open.
Next end-to-end milestone; continue / stop / reprioritize rationale:
  Invoke original installer under named lifecycle/delegate identity contracts;
  join actual Manage subscriber and nested audio/animation first steps, then
  manager Shuffle admission, retaining all five prior card waits and semantic
  state. Two records share animation owner; derive total/order from native run.
  With same producer clock .05animation can precede .3card waits: next drain must
  honor actual native selection/animation continuation before Minion acquisition.
  Reprioritize from absent-subscriber assumption to concrete configured handler.
  Necessary conditional dependency and original binding uncertainty closed;
  continue full goal through legal observations, independent S2 and S3/S4.
```

## Installed subscriber and retained eight-wait checkpoint

2026-10-03. Native witness commit
`9c52a5a4789237c163c6de3ff6bd548631ad3419`; independent projection and Rust
checkpoint commit `9f865d95de5246e754837ae71ebcc4530d34bccb`. The full planning
baseline and all eight acceptance gates remain authoritative.

```text
Build/assets / solver commit / corpus version / information mode / objective:
  Same pinned build/assets; commits above. first_village_subscriber_admission_v1
  and one independent scoped semantic/timing fixture. Offline development oracle
  validation; no native hidden fields become planner/player inputs.
Named decision blocker and reachable deck/phase/role scenario:
  Does original scene animation add earlier waits to the generated N5 transaction
  before acquisition? Minion/Confessor/Lover/Hunter/Enlightened, row0/poolrow0,
  original manager137026 and animation137027, retaining all five card waits.
Before -> after supported behavior:
  Conditional onSetup=null/six records -> actual OnEnable installation, bound
  Action invocation and nested audio/Animate first steps, then manager Shuffle
  and normal Manage return. Eight records share seven native owners, with two
  ordered payloads on animation owner107. All original semantic actors/clones,
  statuses/source fields/publication/first waits remain retained.
  Registration order:five cards,Animate,audio,Shuffle. Wait admission:five cards,
  audio,Animate,Shuffle. Native tree order:Animate,five cards,audio,Shuffle.
Original evidence; supplied runtime/scheduler boundaries:
  Actual installer363500,Animates363060,Animate375130state0,PlayAudioDelay375DB0
  state0,Manage36CE30,Shuffle376B00state0 and original engine/managed bridge bodies.
  Exact original scene binding/assets consumed; lifecycle invocation supplied.
  Initial onSetup/three later GameplayEvents lists null, closed Action constructor
  and null-existing Combine layouts supplied. AudioEvents=null conditional.
  GetComponent/pivot mappings, warm Vector3 storage, tween requests/unconsumed
  return and Current-yield field/mirror getter supplied; generic Current body
  and full Vector3 cctor not executed. Native runtime owner identities supplied.
  Nested engine/managed CPU contexts and disjoint parent stack bytes preserved;
  original five owner/payload bytes immutable. Only shared-animation link fields
  may change, with exact reciprocal registration-order membership.
  Exact3753C8 barrier is notification after a24-byte struct copy; RDX0 does not
  overwrite its nonnull original-board reference. Full nonvolatile/XMM return
  and installed delegate/event-field invariants checked.
Differential attempted / admitted / passed / failed / excluded by subsystem:
  5constructors/Init/ActInit/card first yields,2subscriber first yields,1Shuffle;
  4Actions/1onSetup callback,8registrations/records,7owners,15array entries,
  75ordered comparisons/0Start. 16installation+1002Manage services=1018paused
  prefixes/reentry tokens:1010Unicorn+8direct record-creation providers.
  24fresh representative installer/subscriber-family abort prefixes match;
  these are not1018independent failure/exception cases. Prior corpus unchanged.
  69fingerprinted memberships/49selected pins,2901managed+401engine addresses,
  18source/10consumed-report hashes,442pooled snapshots. Fingerprints do not
  establish every body/path executed. No unexplained admitted disagreement.
  Independent native reproduction byte-exact report SHA
  70ade0e711d803735a8076d26e0030a9f61fc4458834a28884e375b6423b3f06.
  Independent no-Rust projector/reviewer byte-exact fixture SHA
  89ffa07667d5b16b05b8a3e180aa2a95b90c348941c5f9c068d8dcdc64f0964c.
  All13focused bridge tests and973library tests pass18.09s. Exact source/prior
  hashes, raw prefixes/storage/CPU, privacy and links pass. Deduction/action
  production behavior unchanged; no new mutation or outcome campaign claimed.
World-set soundness/completeness; ambiguity/contradiction/unsupported results:
  No world-set, legal-history or S1 promotion. Rust actor replay ends at modeled
  Start; registered callback records a separate generic request/normal return,
  not subscriber effects. Inert flags apply to modeled role/UI/service projection.
  Five pending labels unchanged; fresh auxiliary labels use registration order,
  then queue order follows live native tree. All8timings bind raw payload/owner/
  node/GC fields. Synthetic predeadline probe preserves8; aligned-cursor mixed
  queue rejection protects the five-acquisition/V4 unsupported resume boundary.
Policy certificate or best-found budget; prior/likelihood assumptions:
  No policy certificate, seed-law recovery, path likelihoods or uniform-world prior.
Held-out outcomes and latency/memory versus frozen baseline:
  No held-out partition/outcome/budgeted policy evaluation. Library runtime is
  verification, not policy performance. Earlier unchanged production release
  build/34simulation results retained. No perfection/optimality certificate.
New uncertainty, regressions and remaining exclusions:
  Actual lifecycle/multicast/disable/audio services, animation/Shuffle resumes,
  acquisition/native consumption, original Day readiness and rendered/public
  admission remain open. OriginalN5 physical slots have display IDs5..1; fair
  numbered-seat/orientation mapping must be reviewed, not inferred from slots.
  Existing PlayerHistory supports Judge/conditionalHunterBaa, not this mixedN5.
Next end-to-end milestone; continue / stop / reprioritize rationale:
  Actual consumer-selected Animate resumes first on same retained queue; then
  all5Hidden acquisitions with pending audio/manager records preserved. Recorded
  Minion roll1/duplicate index0 selects original Confessor21614 source role, not
  the true actor's clone. Join GiveBluff/concrete Init/AfterRoundStart; appearance
  25 does not change actual lying, real Confessor's repeated25 is not insertion.
  Native chronology/source/status/UI providers remain explicit. Then original
  Day transition, legal click, visible speech and its own fair public adapter.
  Necessary subscriber/ordering dependency closed; continue full active goal
  through S1 legal observations, independent S2, decision S3 and broader S4.
```

## Guarded non-Day setup Reveal checkpoint

2026-10-03. Rust implementation commit
`62472eb746ab44c543c03f4a84788f6320e749b9`. This supersedes the earlier
setup-only resume rejection only within the new explicitly versioned boundary.
Historical certificates above retain their original domains.

```text
Build/assets / solver commit / corpus version / information mode / objective:
  Same pinned build/assets; Rust commit above. Thirteen authored development
  regressions, setup Reveal V5 and scheduled setup Reveal V2. Offline oracle
  transition support; no public-history or planner input expansion.
Named decision blocker and reachable deck/phase/role scenario:
  Can all five Hidden N5 actors acquire their original-source bluffs while
  Audio/Shuffle remain pending, without confusing appearance with actual truth?
Before -> after supported behavior:
  V4 setup-only rejection remains unchanged. V5 admits original Minion and four
  folded-null source selectors with concrete non-Day Init/AfterRoundStart
  callbacks. All six positive Minion pool outcomes survive; copied Confessor
  changes appearance25, while actual Evil remains lying. Copied Alchemist does
  not inherit the real Alchemist Init hook. Scheduled V2 preserves the complete
  card-plus-Audio/Shuffle queue, rejects either auxiliary callback atomically
  when eligible and requires Hidden bodies and exact callback completion.
Original evidence; supplied runtime/scheduler boundaries:
  Evidence links and exact domain in non_day_setup_reveal.md. Callback support
  derives from reviewed original role/selector bodies. Clock, phase, live owner,
  release and callback return contracts supplied. Authored generation5/cursor8
  fixture does not stand in for the separate native generation9/cursor12 trace.
  Non-Day names role triggers3/7, not a global Gameplay-phase certificate.
  Hidden body5 and native queue phase2 are separate from the outer caller phase.
Differential attempted / admitted / passed / failed / excluded by subsystem:
  All422focused bluff tests,986library tests14.79s,release build and34simulation
  tests1392.17s passed. Thirteen new regressions. Deliberate copied-Alchemist
  status-hook mutation failed its focused test; source restored byte-exact and
  fresh compilation confirmed. Stale Windows timestamp run excluded from
  successful verification. Scope formatting, links, privacy and prior artifact
  hashes passed. Native differential fixture integration remains pending.
World-set soundness/completeness; ambiguity/contradiction/unsupported results:
  No new canonical worlds or legal-history certificate. Version/identity/role/
  phase/complete-queue guards reject unsupported inputs without partial output.
Policy certificate or best-found budget; prior/likelihood assumptions:
  Conditional uniform selector-provider fractions only: duplicate outcomes1/10
  each and unique outcomes3/10 each. No original PRNG law, full-generation/path
  prior, policy comparison, optimality or sampled-performance promotion.
Held-out outcomes and latency/memory versus frozen baseline:
  No independent held-out partition or budgeted baseline/candidate evaluation.
  Test durations describe verification, not planner performance.
New uncertainty, regressions and remaining exclusions:
  Exact native retained comparison, Audio/Shuffle callbacks, original Day/input
  readiness, visible speech and original N5 public-history adapter remain open.
  All eight full-plan acceptance gates remain open.
Next end-to-end milestone; continue / stop / reprioritize rationale:
  Compare the separately reproduced native N5 acquisition checkpoint through an
  independent no-Rust projection, then join the smallest original Day/readiness
  dependency required for legal observations. Necessary guarded semantic support
  closed; continue the full active goal through S1, independent S2, S3 and S4.
```

## Retained native animation and original-source acquisition checkpoint

2026-10-03. Native witness commit
`0dac5c99ca9443bc2a96bd2c1fd46f46b8a49d0e`; scoped Rust support remains at
`62472eb746ab44c543c03f4a84788f6320e749b9`. Independent Rust/native checkpoint
integration is a separate pending tranche.

```text
Build/assets / solver commit / corpus version / information mode / objective:
  Same pinned build/assets; commits above. first_village_retained_acquisition_v1.
  Offline development oracle; no fair planner or public observation promotion.
Named decision blocker and reachable deck/phase/role scenario:
  Does the installed animation finish on the same original N5 graph before five
  acquisitions, and does Minion copy the original Confessor source rather than
  its true actor's runtime clone? Generation row0/poolrow0, five retained cards.
Before -> after supported behavior:
  Eight admitted waits -> three rejecting gate drains, five actual Animate
  resumes with four re-yields, then all five DelayReveal state1 callbacks. Final
  generation9/cursor12/tree[5,7] retains Audio/Shuffle. All actors remain Hidden5
  with uses1 and empty histories; one Minion source-copy receives appearance25
  while actual alignment remains Evil. True Confessor's existing25 is unchanged.
Original evidence; supplied runtime/scheduler boundaries:
  Actual native queue/gates/callback/managed role/release bodies execute on the
  retained graph. Original source virtual register/bluff slots independently
  pinned; folded null selectors for four Good roles, actual Minion draw1 then0.
  Source clone provider copies0x48 bytes without Init or actor binding. RoleAct
  writes onActed, not charRef/dataRef. Lifecycle, initial delegate-list/null audio,
  runtime identities, scene UI/Sprite resolution, tween and named rendering/API
  providers remain supplied. Native CPU/stack/storage invariant masks checked.
  Queue phase2 and actor Hidden5 do not certify outer Gameplay phase/readiness.
Differential attempted / admitted / passed / failed / excluded by subsystem:
  Ten iterator resume calls,9drains,12insertions,8registrations,7owners,6releases,
  2pending,2draws,1clone,458new drain services. All1476pause/reentry tokens agree
  with complete normal ledgers/entry CPUs/snapshots;7fresh representative restart
  stops agree. They are not1476independent API failures. 3476managed/752engine
  addresses,81fingerprints,11managed+6engine pins,18source/11prior hashes,
  673pooled snapshots. Independent complete producer byte-exact report SHA
  f74f10f6f9eccd5ae1af512871dc01e5c173e590290a414b59a11b74122ce9f3.
  Source syntax, unchanged hashes, raw owner/payload linkage, privacy and links
  pass independent review. No unexplained admitted disagreement.
World-set soundness/completeness; ambiguity/contradiction/unsupported results:
  No world-set or legal-history certificate. The recorded copied-Confessor path
  is native dependency evidence; other positive six-role Rust branches are not
  native retained executions in this report. Rust comparison fixture pending.
Policy certificate or best-found budget; prior/likelihood assumptions:
  Recorded draws explain this oracle trace only. No PRNG law, path prior, policy
  likelihood, uniform-world assumption or optimality promotion.
Held-out outcomes and latency/memory versus frozen baseline:
  No independently frozen held-out or fixed-budget outcome comparison.
New uncertainty, regressions and remaining exclusions:
  Pending Audio/Shuffle state1, full outer setup/Day/input readiness and visible
  speech remain unresolved. Presentation calls do not establish pixels. Freed
  storage retained for audit is not safe real post-free ownership. Original N5
  public-history adapter remains absent; all eight acceptance gates remain open.
Next end-to-end milestone; continue / stop / reprioritize rationale:
  Complete independent Rust/native semantic and queue comparison, then establish
  the smallest outer setup/input boundary required for legal N5 observations.
  Animation/acquisition retained dependency closed; continue S1 rather than
  expanding unrelated rendering work or treating native counters as completion.
```

## Independent retained N5 acquisition differential checkpoint

2026-10-03. Projector, fixture and Rust comparison commit
`dda6b682844df8ee1bb0dbec240f7e42dc2e6d3a`. Evidence and exclusions are in the
[comparison note](notes/systems/first_village_retained_acquisition_projection.md).

```text
Build/assets / solver commit / corpus version / information mode / objective:
  Same pinned build/assets; commit above, first_village_retained_acquisition_v1
  synthetic projection. Offline oracle differential, not planner observations.
Named decision blocker and reachable deck/phase/role scenario:
  Does guarded Rust consume the original five-card retained setup/animation
  chronology and acquire Minion's original-source Confessor without dropping
  other positive outcomes or resetting pending waits?
Before -> after supported behavior:
  Separate native/V5-V2 support -> independent no-Rust projection plus original
  native checkpoint comparison. Setup semantics feed actual iterator-label
  mapping0..4; three gated and five animation queue drains match gen0→8/cursor12,
  then five acquisitions match gen9/cursor12 and only Audio5/Shuffle7 pending.
  Native displayIDs5..1 remain distinct from offline modeled positions1..5.
Original evidence; supplied runtime/scheduler boundaries:
  Frozen native report/source, original assets, concrete callback entries,
  status versions and native owner/payload/queue ledgers. Rust animation replay
  consumes explicit native-derived wait mutations/result1/release responses;
  complete Animate/tween execution stays native-only. Supplied UI booleans do
  not certify native name/art/color routing or pixels. Global Gameplay phase,
  scene lifecycle/input readiness and pending deck callbacks stay outside scope.
Differential attempted / admitted / passed / failed / excluded by subsystem:
  Five setup actors/full semantics,8preceding queue drains,5acquisition callbacks
  match; all6positive selector outcomes retained before grading recordedConf.
  Real/copied role, slot, trigger, dispatch, target and status insertion/version
  deltas compared. NoStart/replacements; four null selectors consume no RNG.
  Independent projector/reviewer/Root reproductions byte-exact128002B fixture
  SHA e35322e0e415469613910ab464551b9ab5de7a3eec7b07476bf333d271c60e40.
  Projector SHA da5c819004588eb2fce396602eca152e3703cf4d7201e8cd49f13465a32131eb.
  Focused new comparison passed afterfresh3m11compile;987library tests passed
  29.54s. Syntax, scopedformat, links, privacy and unchangedhashchecks passed.
  No unexplained admitted mismatch; no new mutation or outcome campaign.
World-set soundness/completeness; ambiguity/contradiction/unsupported results:
  No world-set or mixed-N5 public-history promotion. Test complete semantic
  states and typed unsupported boundaries; oracle data remain isolated.
Policy certificate or best-found budget; prior/likelihood assumptions:
  Six conditional supplied-uniform selector fractions only. Recorded native
  draws are validation evidence, not planner seed/prior or generation weights.
Held-out outcomes and latency/memory versus frozen baseline:
  No held-out or fixed-budget policy comparison. Prior release build/34simulation
  passes retained for unchanged production behavior; currentchange adds tests.
New uncertainty, regressions and remaining exclusions:
  Read-only investigation identifies a PinnedDeckView null-first-yield and
  aliased-root dependency. Original installer/event/child-lifetime native
  composition, lifecycle invocation and null-yield scheduling remain pending;
  no inert-handler or empty-readiness-queue certificate is established.
  Pending Audio/Shuffle, legal click/speech and public adapter remain open.
  All eight full-plan acceptance gates remain open.
Next end-to-end milestone; continue / stop / reprioritize rationale:
  Resolve bounded deck/readiness subscriber and null-yield continuation effects
  on retained N5 before one legal original Hunter reveal. Reprioritize from an
  assumed empty readiness queue to the identified coroutine dependency. Continue S1
  legal history, then independent S2, legal decision S3 and broader S4.
```

## Conditional reviewed-history deduction checkpoint

2026-10-03. Implementation and exclusions are in the
[conditional API note](notes/systems/conditional_player_deduction.md).

```text
Build/assets / solver commit / corpus version / information mode / objective:
  Same pinned build/assets; parent ccef829b6b69bbba7347aff9bf6655b4ca54e763.
  conditional_hunter_baa_development_v2, fair public history under explicitly
  supplied setup/availability; complete conditional deduction, no action policy.
Named decision blocker and reachable deck/phase/role scenario:
  Mechanical snapshots lacked typed complete/ambiguous/contradictory results.
  Four/five distinct cyclic seats, one Baa and otherwise identical Hunters,
  finished clean initial Day and acquired Hunter bluff are supplied assumptions.
Before -> after supported behavior:
  Snapshot-only adapter -> typed core API requiring AdmittedPlayerHistory.
  Complete finite worlds sorted by Baa seat, conditional definites and full
  supporting public ordinals; missing/unsupported inputs remain explicit.
Original evidence; supplied runtime/scheduler boundaries:
  Existing native Hunter/Baa rules and twenty native-generated sentences.
  Synthetic reviewed public setup/reveal availability; no original pixels,
  full native generation/queue-to-click chronology or acquired-bluff certificate.
Differential attempted / admitted / passed / failed / excluded by subsystem:
  Six backend fault-injection tests, twelve independent integration tests and
  all993 library tests passed; release build passed. Final integration12.80s,
  full library21.03s, not per-decision latency. Scoped format/links/privacy pass.
  Core SHA32e51af0f4609579baecaa23d843e3c8642b129259a8c117f410c269029d453d.
  Reference SHA571b80000184ece50e094c1a644e62b6684c8f0277eb88e0e8928639ac40314e.
World-set soundness/completeness; ambiguity/contradiction/unsupported results:
  All8160 development prefixes agree with independent complete worlds:
  4376 ambiguous/3784 unique; four coherent contradictions completely empty.
  Candidate set must cover allN worlds once; residual latent state, duplicates,
  bad counts/definites fail incomplete, never false complete emptiness.
  Paired hidden worlds, native-reference mutation and capture/corpus metadata
  changes preserve output. Judge and mixed originalN5 remain unsupported.
Policy certificate or best-found budget; prior/likelihood assumptions:
  No probability, prior, recommendation or RNG. All five model/availability
  assumptions explicitly marked assumed_conditional; no exact live promotion.
Held-out outcomes and latency/memory versus frozen baseline:
  No independent held-out or fixed-budget policy/outcome comparison.
  Legacy behavior unchanged; prior34simulation passes retained, not rerun.
New uncertainty, regressions and remaining exclusions:
  Original legal observations, full setup provenance, broader worlds, weights,
  legal policy and ascension continuations remain open. Guard tests inject
  backend faults; no new production-rule mutation campaign is claimed.
  All eight full-plan acceptance gates remain open.
Next end-to-end milestone; continue / stop / reprioritize rationale:
  Continue retained deck/null-timed queue composition toward one original
  legal Hunter reveal. Keep mixedN5 public-deck/latent-world proposal separate
  from the development Hunter/Baa model; then independent held-out S2/S3/S4.
```

## Guarded real Gemcrafter setup checkpoint

2026-10-04. Evidence and exclusions are in the
[Gemcrafter note](notes/systems/non_day_gemcrafter_setup.md).

```text
Build/assets / solver commit / corpus version / information mode / objective:
  Same pinned build/assets; parent69b1802800b75a319cda98851caeffd20ea483e0.
  RevealV6/setup bridgeV3/scheduledV3, authored offline input-conditioned model.
Named decision blocker and reachable deck/phase/role scenario:
  Four of the five original Standard subsets contain real Gemcrafter; older
  domains admit only its copied bluff and reject the real data/class binding.
Before -> after supported behavior:
  Opt-in real Archivist source/clone binding, inherited null selectors, no
  selector draws and state-preserving Init/After callbacks with truth routing.
  Complete card/Audio/Shuffle queue guards and no-Start boundary retained.
Original evidence; supplied runtime/scheduler boundaries:
  Exact managed inheritance, slots20/21 folded null body, shared Day-only
  Act/BluffAct native bodies and original Start-order absence. Authored roster,
  source/clone, empty Start order and queue inputs; no new retained Gem run.
Differential attempted / admitted / passed / failed / excluded by subsystem:
  Seven new authored regressions; all12focused and1000library tests pass,
  full library20.97s; release build passes. Initial two fixture failures fixed
  by explicit order,
  without weakening production missing-order rejection. Scoped format/diff
  and23note-link checks pass. No native full-history comparison claimed.
World-set soundness/completeness; ambiguity/contradiction/unsupported results:
  No public-domain or complete world-set promotion. Older versions still
  reject real Gemcrafter; malformed bindings and deferred callbacks reject
  atomically. Real Alchemist and18other fallback callbacks remain unsupported.
Policy certificate or best-found budget; prior/likelihood assumptions:
  No policy, native generation/path weights or probability promotion.
Held-out outcomes and latency/memory versus frozen baseline:
  No held-out outcome/budget comparison. Prior34simulation passes retained
  for unchanged live behavior; this opt-in model has no live bridge consumer.
New uncertainty, regressions and remaining exclusions:
  Original retained Gem histories, full fallback, deck publication, global
  Day and legal PlayerHistory remain open. Native readiness/Hunter prototypes
  are private and unvalidated; a service-pause mismatch prevents certification.
  All eight full-plan acceptance gates remain open.
Next end-to-end milestone; continue / stop / reprioritize rationale:
  Continue exact retained continuation/pause closure and one original Hunter
  publication. Remove only the supported class gap from the mixedN5 proposal;
  preserve the public-deck/latent-world/held-out and legal-policy prerequisites.
```

## Private retained continuation and conditional Hunter checkpoint

2026-10-04. This supersedes the unresolved pause comparison in the preceding
checkpoint. Native witness sources and full reports remain unpublished drafts;
the results below are bounded dependency evidence. The
[mixed N5 frontier](notes/systems/mixed_n5_observation_frontier.md) still separates
conditional setup support from a legally admitted observation history.

```text
Build/assets / solver commit / corpus version / information mode / objective:
  Same pinned build/assets; parent 03bb7cd56a88d7f10c464194cb1662b06d7efe3f.
  Private retained-readiness verification v0 and conditional Hunter prototype v0.
  Native/oracle validation only; no new planner inputs or live control.
Named decision blocker and reachable deck/phase/role scenario:
  Original Standard row 0, five retained actors, recorded Confessor acquisition,
  installed Pinned subscriber and pending Audio/Shuffle. Determine whether
  nested continuations preserve the queue and permit a conditional Hunter result.
Before -> after supported behavior:
  Exact normal/all-entry pause parity closes the transient actor-scope mismatch.
  Native Hunter click, Day clue, result and speech complete on the retained graph;
  nine interleaved drains preserve the two-child Pinned continuation.
Original evidence; supplied runtime/scheduler boundaries:
  Original generation, initialization, acquisition, native queue consumers,
  subscriber, Hunter/Reveal and publication bodies. Lifecycle, Unity providers,
  equal-key whole-list Shuffle and null modal DeckView subscriber are supplied.
  Hunter global Day/input fields and Current mirror are supplied; tween
  completion and rendered capture are excluded. No original RNG weights.
Differential attempted / admitted / passed / failed / excluded by subsystem:
  Root independently reproduces the readiness normal/all-entry pause report
  byte-exact: 1,876 entries, including 122 within Shuffle, and 1,876 one-shot
  reentries/prefixes.
  Private report SHA bcc856e3534893a91589682cd44391513dc5c6cd8708eb98c48c783c0efee245.
  All 59 selected fresh stops pass complete prefixes and exact suspended states.
  Full private report: 61,704,585 bytes; SHA
  67058e2ce911e8d6dec2a6729eabc50c8f4498dc14576363bf9506e6aefbd641.
  Its normal trace and all entry-state digests match Root's reproduced baseline.
  Selection groups by service name/machine/prior-or-Shuffle scope. The physical
  inventory finds zero missing entry variants within those selected families,
  but excludes 126 unselected caller variants. Argument/type/callback-plan
  variants and arbitrary services are not certified. The root_dispose label
  executes Current provider c10 only; configured Dispose provider c20 is
  unexecuted. Actual interface Dispose uses the separately selected supplied
  0x40F0 shim.
  Root's independent fresh stop at new-strip Action construction, ordinal 1,822,
  passes complete prefix equality against that reproduced baseline and exact
  selected-versus-actual suspended state equality. Private report SHA
  8b2136252cb9a7b5c2e93a56ebfdccbbf03c2d9bc9d4b5c62e1ab72f80546e1f.
  Root independently reproduces the conditional Hunter report byte-exact:
  158,766,191 bytes; SHA 4aa8ceb5af7dd0b7ceaf57e2bef11563c83623ef7c0e131a4b3c9c8dde6cfd24.
  Twenty protected native writes, two iterator guards, nine drains. Its 524-entry
  verifier passed source review, syntax and import checks; no Hunter pause/stop run.
World-set soundness/completeness; ambiguity/contradiction/unsupported results:
  No new complete world-set comparison or public-domain promotion. Hunter speech
  is native-produced but target references remain oracle-only. No PlayerHistory
  admission or reuse of the narrower Hunter/Baa model for mixed N5.
Policy certificate or best-found budget; prior/likelihood assumptions:
  No policy, probabilities, generation prior or fixed-budget search claim.
Held-out outcomes and latency/memory versus frozen baseline:
  No new held-out or policy/outcome evaluation. Existing 1,000 library/release
  checks for the published Gemcrafter change remain the current solver checks.
New uncertainty, regressions and remaining exclusions:
  Exact two-child iterator loops can repeat state; final-root false/-1 branch
  was not reached by the new Hunter continuation. One UI record remains pending.
  Original global Day chronology, complete public deck, capture/HP provenance
  and legal history remain open. Readiness writers reject the exact reports
  directory; descendant containment needs hardening before source publication.
  Full reports, CPU/stack/guarded bytes and corpora remain private.
  All eight full-plan acceptance gates remain open.
Next end-to-end milestone; continue / stop / reprioritize rationale:
  Bounded readiness selection and physical inventory are complete; retain the
  declared caller exclusions and private source-publication obligations.
  Verify the conditional Hunter continuation and join modal DeckView
  publication on the retained row-zero graph, preserving original Day and
  trusted capture as separate gates. Independently compare six setup/acquisition
  outcomes for one explicitly supplied row-zero input before any mixed S2
  history admission. Preserve legal S3 and broader S4 gates.
```

## Independent fixed-row setup/acquisition support checkpoint

2026-10-04. The [comparison note](notes/systems/first_village_retained_acquisition_projection.md#independent-six-outcome-conditional-support-reference)
records the new reference and its narrow input domain.

```text
Build/assets / solver commit / corpus version / information mode / objective:
  Same pinned build/assets; parent a2c803c3816ee390d99bebb769a3d5613f81e69a.
  Unchanged retained-acquisition fixture v1, setupV2/acquisitionV5/scheduledV2.
  Offline input-conditioned development reference; no new planner input.
Named decision blocker and reachable deck/phase/role scenario:
  Fixed original row-zero roster Minion/Confessor/Lover/Hunter/Enlightened,
  four duplicate and two unique choices. Check all six modeled alternatives
  without trusting production transitions or fixture expected-output fields.
Before -> after supported behavior:
  Independent full modeled initialization, setup and six acquisition outcomes;
  complete callbacks, queue chronology, deferred identities and support keys.
  Exact15 nonmatching Start entries retained; no Start calls or new continuations.
Original evidence; supplied runtime/scheduler boundaries:
  Existing retained Confessor path is the sole native full-history anchor.
  Other five outcomes are authored rule support. Physical order, fresh statuses,
  resistance/target, inert services and acquisition clock/queue are supplied.
Differential attempted / admitted / passed / failed / excluded by subsystem:
  Six modeled outcome/state/chronology comparisons pass in two new tests.
  Initial compile failure corrected to exact optional roster container schema.
  Focused release2/2 and full release library1002/1002 pass; full run18.93s.
  Child SHA5a7b12adab73f96445abb68b333e026a62b0def0ff5a63e3052a91572cb56430.
  Fixture SHAe35322e0e415469613910ab464551b9ab5de7a3eec7b07476bf333d271c60e40.
World-set soundness/completeness; ambiguity/contradiction/unsupported results:
  Six complete modeled support keys agree for this fixed conditional input.
  This is not a generative world set or public belief domain. Bad bindings,
  inconsistent late queue/callback fields, unsupported selectors and eligible
  deferred Audio/Shuffle reject atomically without input mutation.
Policy certificate or best-found budget; prior/likelihood assumptions:
  Conditional probability fields excluded; no generation prior or policy claim.
Held-out outcomes and latency/memory versus frozen baseline:
  No new held-out, fixed-budget decision or outcome comparison. Production
  behavior is unchanged; prior34simulation passes retained without rerun.
New uncertainty, regressions and remaining exclusions:
  Status versions are projected from modeled insertions, not stored RevealActor
  fields. Other placements, full fallback, native all-six histories, legal Day,
  complete public deck, trusted capture and PlayerHistory admission remain open.
  All eight full-plan acceptance gates remain open.
Next end-to-end milestone; continue / stop / reprioritize rationale:
  Conditional support dependency closed. Continue the retained modal deck
  publication and original Day/capture join before admitting mixedN5 history;
  preserve independent S2 held-out worlds, legal S3 policy and broader S4 gates.
```

## Private conditional Hunter pause and first-stop checkpoint

2026-10-04. This supersedes the pending Hunter verification in the earlier
continuation checkpoint. Native sources and full reports remain private drafts.

```text
Build/assets / solver commit / corpus version / information mode / objective:
  Same pinned build/assets; parent7766e25ebd8b16a485b6c4e569c0a6e027ef921a.
  Conditional Hunter verification draftv0, corrected wrapper source SHA
  0749cc2003994d9cbff973f671d442f613281cfb67083b803a6dec79b4f32c51.
  Native/oracle dependency validation; no new player inputs or live control.
Named decision blocker and reachable deck/phase/role scenario:
  Retained row-zero Confessor-acquisition graph, conditional supplied Day,
  one real Hunter click, native result/speech and interleaved Pinned waits.
  Verify physical pause/reentry and a genuine pre-publication stopped prefix.
Before -> after supported behavior:
  All524 one-shot service boundaries match complete normal state after reentry.
  First physical UI gateway fresh stop preserves exact pre-entry state/prefix;
  Hunter remains Hidden with use1/history0 and no new publication or iterator.
Original evidence; supplied runtime/scheduler boundaries:
  Frozen Hunter/readiness bodies and supplied Day/input/Current-mirror profile
  unchanged. Corrected diagnostic window clamps within each actual mapped
  stack and records the entire mapping; no native map or state is changed.
Differential attempted / admitted / passed / failed / excluded by subsystem:
  Initial diagnostic failed on an unmapped centered stack read before pause.
  Corrected normal/all524 pause report passes; SHA
  dfd3bba283c716111f561aa32c065c38e99966d09b6d5d356eabe244229c4284.
  One selected fresh stop passes, ordinal1/UI entry1C7DC50/caller386B42;
  report SHA d611364a1cc1352ab107e16c0c3140b0bb3cd792c4f52d81cb89347653e241c1.
  Root independently compares all524 ledger digests, normal exit and41 input
  hashes across pause/stop reports; expanded selected/actual states and prefix
  digest agree. Both complete0x2000 windows match their full mapped stacks.
  Measured engine RSP0x30000F010/window[0x30000E000,0x300010000);
  its raw RSP was only inferred for the earlier failed diagnostic.
  Other295 physical candidate fresh stops are unrun; arbitrary plans excluded.
World-set soundness/completeness; ambiguity/contradiction/unsupported results:
  No new world-set or public-domain promotion. Native hidden state, target
  references and stack/CPU bytes remain validation-only private evidence.
Policy certificate or best-found budget; prior/likelihood assumptions:
  No policy, likelihood, generation prior or probability certificate.
Held-out outcomes and latency/memory versus frozen baseline:
  No new held-out/outcome comparison. Solver checks remain1002release library
  passes from the preceding fixed-input reference checkpoint.
New uncertainty, regressions and remaining exclusions:
  Pending UI record29/generation18/cursor30 survives normal/pause completion;
  stopped prefix retains record18/generation9/cursor19. New two-child Pinned
  final exhaustion, complete modal roster, original Day and capture remain open.
  Static modal review identifies DeckState-dependent faction suppression and
  actual Shuffle UI channel delivery; retained modal composition is unexecuted.
  Full CPU/storage reports and unpublished source-containment work stay private.
  All eight full-plan acceptance gates remain open.
Next end-to-end milestone; continue / stop / reprioritize rationale:
  Bounded Hunter verification closes this dependency. Next compose actual modal
  OnEnable after acquisition, then retained Audio/Shuffle delivery with explicit
  DeckState and child-generation boundaries; preserve original Day/capture as
  separate gates before mixedN5 admission, independent S2, legal S3 and broader S4.
```

## Private retained modal data publication checkpoint

2026-10-04. This supersedes the unexecuted modal dependency in the preceding
checkpoint. The [mixed N5 frontier](notes/systems/mixed_n5_observation_frontier.md)
keeps the original Day and legal-public-history obligations separate. Full
native reports and the new producer remain private drafts.

```text
Build/assets / solver commit / corpus version / information mode / objective:
  Same pinned build/assets; parent887f133bd2a3598f751531b527aeb748caef85bf.
  Modal draftv0, source a619b369a696832a0898f3c21ce7466748bd3737aed4a3e8ea4365b5e0e78181.
  Offline native/oracle dependency validation; no new player inputs or live control.
Named decision blocker and reachable deck/phase/role scenario:
  Original row-zero Confessor-acquisition graph: the complete current deck must
  distinguish four duplicate choices from two unique additions before a public
  Hunter reference could filter beliefs. Pinned ShowAll omits Villagers.
Before -> after supported behavior:
  Actual modal OnEnable executes after acquisition, then retained Audio/Shuffle
  delivers UI+48 through native call376C28 into native UpdateDeckView/RemoveAll.
  Five native Init/GetData publications equal current roster occurrences in
  faction order [4,0,1,0]. Original board actors remain unchanged.
Original evidence; supplied runtime/scheduler boundaries:
  Five full fingerprinted bodies: OnEnable, RemoveAll, UpdateDeckView, Init,
  GetData. Native constructor-result typing and 16+8-byte stack cursor copies
  now follow their actual tested registers and native stores.
  Empty runtime modal roots, None0, null prior event/hover channels, opaque
  instantiated buffers and whole inert Character.InitReward are supplied.
  Whole ShuffleList uses fresh occurrence-preserving equal-key lists; roster
  enumeration/Dispose, Unity/runtime services and prior scheduler boundaries
  remain explicit providers. No original RNG or rendered-preview claim.
Differential attempted / admitted / passed / failed / excluded by subsystem:
  One pre-emulator dependency launch failed; four native normal attempts:
  three wrapper failures preserved, then one complete normal exit passed.
  Fixes qualify Action allocation, distinguish Combine result from old channel,
  and bind copied live cursors instead of their overwritten hidden-return buffers.
  The v2 failure independently preserves actual managed/engine stacks, parent
  window and outside-child bytes; its diagnostic is not a selected-stop certificate.
  Normal report64,369,790 bytes; SHA
  82a1fdf49fc7b028d823decae1a803626e00657e39f4d1924c75650d87508589.
  Thirty-eight consumed input hashes pass. Root independently expands declared
  snapshot pooling and checks five body fingerprints, complete eight-invocation
  chronology,249 services with counts[8,17,224,0,0,0,0,0], four exact24-byte
  copy records, five data-pointer occurrences and six direct native frames.
  Nineteen reached native stores comprise two Obscured writes, two modal channel
  publications and five hover/hover-exit/data triples; five InitReward calls are
  the declared whole inert preview service.
  Each frame matches invocation receiver/method, completion RIP/RSP, nonvolatile
  registers, parent-window bytes and equal outside-child entry/exit hashes.
  Each getter's returned RAX equals its published pointer; it uses zero services.
  Fresh failure output is absent on success. No modal pause/reentry or fresh-stop
  selection is run; other rosters and deck-state profiles remain excluded.
World-set soundness/completeness; ambiguity/contradiction/unsupported results:
  No new complete world set, legal-history adapter or public-domain promotion.
  Native pointer publication does not establish rendered public visibility.
Policy certificate or best-found budget; prior/likelihood assumptions:
  No probabilities, likelihoods, generation prior or policy certificate.
Held-out outcomes and latency/memory versus frozen baseline:
  No new held-out, fixed-budget or outcome evaluation. Production Rust and its
  prior1002 release library/34simulation checks are unchanged and not rerun.
New uncertainty, regressions and remaining exclusions:
  One Pinned record18/generation9/next19 remains at deadline1.5/frame17/mask10;
  two Pinned instances remain, with four deferred destruction requests rather
  than a destruction commit. Five board actors acquired, nine payloads released.
  No native frames, borrowed payloads or live modal cursor remain at normal exit.
  Raw service phase labels persist as modal_installation; the eight invocation
  records and physical callers establish scope, not that diagnostic phase string.
  Pinned final exhaustion, preview/tween completion, original lifecycle/Day,
  trusted capture/HP/resource provenance and PlayerHistory admission remain open.
  Full proprietary bytes/CPU/storage and unpublished sources remain private.
  All eight full-plan acceptance gates remain open.
Next end-to-end milestone; continue / stop / reprioritize rationale:
  Conditional modal data dependency closed; continue original SameHandOut into
  fresh SetupDelay state0, its actual first wait/resume, native Intro5->Day10/Prev5
  and Characters.Init/Manage. Do not rerun Init on the acquired checkpoint.
  Keep exact supplied mode, scene/pool, event/rule/relic/resource boundaries and
  pending child waits explicit. Then close trusted public capture before mixedN5
  admission, independent S2 worlds, legal S3 policy and broader S4 continuations.
```

## Original setup dependency triage checkpoint

2026-10-04. The [original Day frontier](notes/systems/first_village_original_day_frontier.md#fresh-pool-and-rule-bindings-for-the-next-composition)
now resolves fresh pool creation, inherited empty-rule getters and the
script-selection draw. This is static dependency evidence; the new original
setup draft has not executed a native fixture.

```text
Build/assets / solver commit / corpus version / information mode / objective:
  Same pinned build/assets; parent100306e08d70689fd79a5dfb1df0bff827d5ce55.
  Original setup static triage; offline validation lane, no planner input.
Named decision blocker and reachable deck/phase/role scenario:
  Row-zero N5 Confessor/Lover/Hunter/Enlightened/Minion setup: the original Day
  writer and fresh physical board must precede any admitted Hunter history.
Before -> after supported behavior:
  Replacing supplied generation/Day requires actual SameHandOut/SetupDelay and
  actual Characters.Init factory selection, not a prefilled acquired board.
  Exact pool/prefab/placeholder and inherited GetRules bindings are resolved.
Original evidence; supplied runtime/scheduler boundaries:
  Six full body fingerprints and selected original caller operands are pinned.
  Selected base GetRules produces fresh empty lists; original UpdateRules checks
  null then Count and skips aggregation. Original creator uses two passes.
  Scene/runtime hydration, mode/day entry, null events, empty relic inventory,
  allocation/cloning and engine admission remain declared provider obligations.
Differential attempted / admitted / passed / failed / excluded by subsystem:
  No new original setup execution. Independent static readers check four outer
  bodies/twelve direct calls and five role declarations/ten getter-branch pins.
  A mistaken static memory-call filter is corrected from actual register loads;
  complete JSON parsing and native receiver/branch checks are tightened in guide.
  Fresh N5 actor-array and five empty-getter counts are proposed exit assertions,
  not measured native totals. No fresh stopped/pause/reentry comparison exists.
World-set soundness/completeness; ambiguity/contradiction/unsupported results:
  No world-set comparison, public history or supported-domain promotion.
Policy certificate or best-found budget; prior/likelihood assumptions:
  No new policy, probability or prior. Recorded draw indices remain oracle inputs.
Held-out outcomes and latency/memory versus frozen baseline:
  No held-out/outcome/performance run; production Rust unchanged, not rerun.
New uncertainty, regressions and remaining exclusions:
  The source draft remains unpublished and unexecuted. Original first-start
  lifecycle, currentDay provenance, complete event/resource effects, scheduler
  parent/child retention and rendered public capture remain open.
  All eight full-plan acceptance gates remain open.
Next end-to-end milestone; continue / stop / reprioritize rationale:
  Bind the exact finite draw schedule and fresh scene pool, then freeze/run the
  original SameHandOut->SetupDelay normal composition. Require actual .1f
  wait/resume, Intro5->Day10/Prev5, fresh Init/Manage returns and terminal false,
  preserving all pending child waits. Keep Pinned/modal outside this first witness.
```
