# Player-history boundary: development implementation v1

Build: `f530404b0f3f_807de4a83df4`, Steam build `23084916`. Authority:
[S0 contract v1](../../SOLVER_CONTRACT.md). Source:
[player_history.rs](../../../crates/solver-core/src/player_history.rs), tests:
[player_history_tests.rs](../../../crates/solver-core/src/player_history_tests.rs).

This is a strict player-information input boundary and a bounded compatibility
projection. It is not a certified gameplay domain, complete possible-world
model, native-generation trace or policy evaluation. Its fixtures are authored
development cases, not independently held-out original observations.

## Exact boundary and trust responsibility

`PlayerHistory` deserialization requires the versioned identity envelope and
typed events with sibling `kind` / `payload`. It rejects unknown fields at the
envelope, event and payload levels, including offline `pd_corruption_target`,
Twin context fields, true role/corruption/lying flags, queue state and original
RNG. No open parsed-clue map is accepted. An obscured deck slot has no identity
field. Exact speech, missing speech, ordered references and duplicate exposed
deck identities remain distinct.

`validate_history_shape(&PlayerHistory)` checks build/schema identities,
nonempty parser/corpus/domain versions, a syntactically complete solver commit,
increasing ordinals, explicit phase changes, board-position bounds and matching
request/result actor and ordered targets. An `action_requested` references its
own ordinal; later events reference the latest request, or null before any
request. Completion cannot be duplicated. Serial single-card reveals are
supported; a new action before completion and multi-card reveal requests are
rejected because cancellation/batch completion semantics are not implemented.
This is structural chronology, not a certificate that every requested action,
picker target, reset or phase transition is legal under native rules.

Raw history contains no evidence registry and cannot certify availability by
including an evidence ID, `visible=true`, or an arbitrary rule-version label.
The separate `ReviewedEvidenceRegistry` is neither deserializable nor
serializable. Its public registration API is an explicit trusted-review
capability; callers must not expose that capability as automatic approval of
raw stdin/JSON or memory rows. The implementation does not inspect screenshot
pixels, authenticate capture references, or perform the external human/tool
review. That review remains the trusted caller's responsibility.

The concrete API is:

```text
record_trusted_ui_review(&PlayerHistory, ordinal, capture_reference)
bind_reviewed_memory_transcription(evidence_id, prior_reveal_evidence_id)
admit_history(&PlayerHistory, &ReviewedEvidenceRegistry)
AdmittedPlayerHistory::planner_history()
AdmittedPlayerHistory::project_legacy_snapshot()
```

UI review binds an exact event and the complete history prefix through it,
including identity, chronology and payload. Registry IDs cannot be rebound.
Admission compares every current prefix against its reviewed prefix. Review of
one prefix does not require future events; truncation preserves prior admission.
Memory transcription first needs that exact UI cross-check and a separately
reviewed prior reveal of its actor. The selected gate must match the exact
event and prefix inside the transcription's reviewed prefix. A matching actor
or ordinal from another history is insufficient. Registration is not proof
that a native memory read was already visible to the player.

`planner_history()` retains legal observation order, action links and
build/parser/domain versions, but excludes capture IDs/references, timestamps,
solver-commit and corpus metadata. All chronological result/reset events
survive, including repeated Judge results. No oracle object is accepted by
admission, projection or planner-input APIs. Equal admitted public histories
produce equal planner inputs; no policy distributions are evaluated here.

## Bounded Judge legacy projection

`public_judge_single_day_v1` names the mechanical adapter domain. It permits a
fully exposed public deck, observed HP and wrong-execution cost, serial Day
reveals of apparent Judge with observed empty passive speech and no references,
and an exact current public Judge result for a requested one-target ability.
The speech is exactly `#N is\nLying` or `#N is\nsaying Truth`; the claimed result
is derived from the text, not accepted as a truth flag. Public empty Judge
passive speech and these active result strings are supported by the
[native Judge audit](../roles/gameplay_role_judge.md).

One initial Setup checkpoint may precede the first Day checkpoint, provided
observed HP/cost do not change. Repeated phase checkpoints, return to Setup,
changed HP or execution cost, and nonzero action-cost effects fail closed.
The adapter does not establish transitions for any of those budget changes.
Visible HUD counts do not establish pre-Start provenance; the legacy projection
keeps `BoardCountProvenance::LegacyUnknown`. Default legacy HP/cost never
substitute for missing public capture.

Night/reset chronology, repeated Judge uses, obscured deck identities, deck
refreshes, correction/re-reveal history, other role clues, execution/death,
status feedback and terminal/scoring/progression projection return
`ProjectionError::Unsupported`. Missing deck/HP/cost, missing speech, an
unrevealed ability actor, or a request without observed completion return
`ProjectionError::Incomplete`. Admission can preserve supported typed temporal
events that the projection rejects. The existing Judge validator compares
results against one static truth-appearance state; flattening results across
Night status changes is therefore not justified.

Neither error asserts a world-set contradiction. Zero-world contradiction,
unique/ambiguous results and complete versus limited search remain separate
solver-model work. Probability priors, observation likelihoods, legal action
recommendations and policy optimality are not supplied by this adapter.

## Conditional initial-Day Hunter/Baa projection

`conditional_initial_day_hunter_baa_v1` is an additional mechanical adapter
selected by `project_legacy_snapshot()`. It requires four/five numbered seats,
one initial publicly exposed pool with exactly `N-1` Hunter Villager occurrences
and one Baa Demon, `n_evil=1`, complete current HUD counts `[N-1,0,0,1]`, observed
HP/cost and an initial Day. Each revealed seat is distinct and apparent Hunter;
distance derives only from exact native singular/plural speech, and support is
exactly `1..=floor(N/2)` or the `N-1` sentinel. Five-seat distance 3 is an
unsupported native output, not a world-set contradiction. Missing speech
or HUD counts return `Incomplete`; other text, roles/pools/counts, duplicate
reveals, statuses, writers, budget changes, phase reentry, Night, executions and
abilities return `Unsupported`. Only optional serial cost-free single-seat
reveal requests are projected. Multiplicity and current HUD provenance survive;
the adapter never claims pre-Start counts.

Hunter speech does not name target IDs. Public `targets` must be empty in this
domain. Even geometrically coherent native `acted_infos` reference order,
including an opposite-seat duplicate, is rejected rather than relabeled as
publicly captured references. No schema-v1 redesign or target provenance
shortcut is introduced. See the exact
[conditional capture obligations](hunter_capture_admission.md).

The externally supplied conditional setup is four/five distinct native actor
references in cyclic physical order `[1,...,N]`, finished setup with Baa plus
Hunter current data elsewhere, Hunter bluff acquisition, and no corruption,
registration/identity/status writers or intervening gameplay transitions. The
schema does not encode or prove those conditions; declaring a domain ID and
reviewing pixels does not establish native generation or runtime alias freedom.
They must remain explicit conditional fixture scope and separate validation
provenance. They are not hidden data passed to the planner.

## Development checks and remaining gates

The first fourteen unit tests cover the strict boundary: strict
roundtrip, duplicate role preservation, nested unknown/privileged rejection,
raw evidence rejection, exact payload and prior-prefix binding, UI-reviewed
memory transcription and cross-history same-ordinal gate rejection, paired
different hidden worlds with identical public input/projection, capture/time
independence, chronology/action mismatches, bounds/parser identity, chronological
repeated Judge results, explicit incomplete/unsupported outcomes, phase/budget
flattening rejection and obscured-identity/truth-flag smuggling. Six additional
authored Hunter/Baa adapter tests cover exact text/multiplicity, rejection of
native-only references, missing versus malformed speech/counts, unsupported
pools/sizes/repeated seats, serial reveal completion versus excluded actions,
and paired-oracle input equality with retained reveal order. Root-coordinated
execution after source freeze passed all twenty tests.

The paired-oracle test isolates authored hidden worlds from every production
API; it proves input independence for these fixtures, not compatibility with
native generation or equality of unimplemented policy distributions. Focused
Cargo validation was coordinated by the root agent after source freeze:
`cargo test --release -p solver-core player_history::tests -- --nocapture`
passed all twenty tests. The cross-history provenance defect found in review
is covered by an explicit regression.

The independent [small-world integration](../../../crates/solver-core/tests/small_world_reference.rs)
now routes authored public histories through the production admission and
Hunter/Baa projection. Root-coordinated execution passed all nine integration
tests, with complete canonical world equality at 8,160 prefixes (4,376 ambiguous
and 3,784 unique) and four coherent contradiction histories empty. Its trusted
availability review is a supplied synthetic development service, not an
original UI capture, native-generation proof or independently held-out family.

Legacy/offline `GameState` callers and CLI behavior are preserved. No trusted
capture pipeline, CLI player-history solving mode, live automation admission,
original-to-history native composition, independent held-out evaluation,
cross-role complete world-set model, legal belief-state planner or policy
certificate is integrated by this module. Automatic review of arbitrary incoming evidence
IDs would invalidate this boundary's trust assumption. No live control was used.
