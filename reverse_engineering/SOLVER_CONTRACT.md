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
mechanical legacy projection admits only the named single-Day Judge domain;
it rejects unmodeled transitions rather than flattening them. The registry's
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
| Observation/extraction | [Strict history boundary](notes/systems/player_history_boundary.md): typed history, exact-prefix trusted review, chronological result preservation and paired-oracle development input checks. | Exported Rust module with narrow Judge snapshot projection; trusted capture/CLI/live integration and held-out evaluation remain absent. |
| Deck/pool generation | [Actual pool composition](notes/systems/manage_pool_composition.md): caller/builders/getters/filters/predicate with supplied collection/RNG/layout. | [Ledger bridge](notes/systems/manage_pool_ledger_bridge.md) is offline with later-setup provenance required. Generation-to-observation domain and held-out results remain open. |
| Initialization/Reveal scheduling | [Retained Init join](notes/systems/manage_initialization_join.md) and [publication/action join](notes/systems/manage_publication_action_join.md): actual pool/Init/Hidden/first yield/publication/generic Act routing. [Engine first-step evidence](notes/systems/unity_coroutine_bridge.md) is independent. | Offline kernels; concrete role bodies, engine queue/resumed Reveal and full Manage-to-observation transaction remain open. |
| Role clues/truth/corruption | [Truth/status audit](notes/systems/gameplay_status_corruption_truth.md), version-bearing predicates and [independent conditional Hunter/Baa world reference](notes/systems/conditional_hunter_baa_world_reference.md). | Development complete-world comparison of supplied four/five-seat finished setup; no generation or cross-role/held-out certificate. |
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

The distinct publication/action audit now joins generic Act Init/Start over
retained actors/pools; concrete virtual role bodies remain supplied inert
services. Continue S1 by binding one concrete supported clue role and establishing
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

## Current boundary, publication and deduction checkpoint

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
