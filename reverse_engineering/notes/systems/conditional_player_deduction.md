# Conditional player-history deduction API

Build `f530404b0f3f_807de4a83df4`. This is a finite conditional deduction
interface, not native generation, legal capture readiness, likelihood or action
policy certification. The full [solver-first plan](../../SOLVER_FIRST_PLAN.md)
remains active and its eight acceptance gates remain open.

## Decision blocker and scope

The [strict history boundary](player_history_boundary.md) already admits exact
externally reviewed public prefixes. Its mechanical legacy snapshot alone does
not distinguish complete conditional search from unsupported or incomplete
input. The [independent reference](conditional_hunter_baa_world_reference.md)
already compares the complete four/five-seat initial-Day Hunter/Baa family;
the new [core API](../../../crates/solver-core/src/player_deduction.rs) exposes
that narrow result without extending the admitted domain.

`deduce_conditional_history` accepts only `AdmittedPlayerHistory`. Raw history
JSON cannot provide a trusted review registry or construct that capability.
External UI review still has to occur: admission checks exact event/prefix
bindings, not pixels or the external procedure. No CLI self-approval path is
added. The legacy CLI and live loop retain their existing interfaces.

The named domain is `conditional_initial_day_hunter_baa_v1`. A complete result
marks its assumptions `assumed_conditional`: distinct fixed cyclic seats;
one Baa and otherwise identical Hunters; finished setup without identity
writers, statuses or deaths; acquired Hunter bluff; and supplied legal
observation availability. Public deck/HUD/reveal text and a domain string do
not prove these hidden setup conditions. The boundary requires four/five seats,
the exact public occurrence multiset, initial Day HP/cost, exact current Hunter
speech and empty public target lists. Missing observations are incomplete;
unsupported domains, richer transitions and unsupported syntax are unsupported.
Judge has a mechanical snapshot adapter but no complete model in this API.
The [mixed N5 proposal](mixed_n5_observation_frontier.md) remains unsupported.

## Complete result and fail-closed checks

The canonical conditional world is the Baa seat: all other identities and
statuses are fixed by the stated assumptions. Before reporting completeness,
the API checks that candidate enumeration contains every seat exactly once.
Each candidate and survivor must equal the complete default `Scenario` plus
that one Baa assignment, comparing every serialized field. It rejects residual
corruption, identity traces, role alternatives, duplicates and inconsistent
counts instead of silently erasing latent distinctions.

Survivors are ordered by seat. The result reports
`complete_finite_conditional_worlds` and `unique`, `ambiguous` or `contradiction`,
with conditional definite Good/Evil positions and the public event ordinals
used. Empty worlds have both definite lists empty. Unexpected backend state
returns `incomplete` with `backend_invariant`; missing observations use
`missing_observation`. These failures cannot become a complete empty set.

Output excludes capture IDs/times, corpus/solver metadata, native targets,
oracle truth, probabilities and recommendations. Identical public histories
therefore have identical deductions within the same versioned conditional
model. No original or planner RNG is consumed. Ordinals identify the full
supporting public prefix, not a minimized contradictory subset.

## Validation and exclusions

The [independent reference test](../../../crates/solver-core/tests/small_world_reference.rs)
now exercises this production API on all 8,160 generated observation prefixes,
four coherent contradictions, twenty original-native Hunter sentences under
separately supplied public setup, and paired equivalent hidden worlds. The
independent enumerator does not call production geometry, validators or
scenario generation. Expected family counters remain 4,376 ambiguous and
3,784 unique prefixes. Native-only target mutation and changes to capture,
corpus and solver metadata must preserve the production deduction.

The [backend guard tests](../../../crates/solver-core/src/player_deduction_tests.rs)
inject missing/duplicate worlds behind matching counts, residual state, wrong
roles/bounds, bad counts/definites and extraneous role-action output. These
must fail closed. Other tests exercise missing HP/deck, unsupported Judge/mixed
domains, six seats, N5 distance-three speech and absent review capability.

Commands after shared Rust source freeze:

```text
cargo test --release -p solver-core --lib player_deduction::tests -- --nocapture
cargo test --release -p solver-core --test small_world_reference -- --nocapture
cargo test --release -p solver-core --lib
cargo build --release
```

The six backend guard tests passed. All twelve independent integration tests
passed in 12.80s after the final assumption-identity assertion, preserving the
full counters above and all four empty contradictions. The full library suite
passed all 993 tests in 21.03s; the release build passed. Source was frozen for
each Cargo run. Scoped formatting, relative evidence links and whitespace
checks passed. These suite durations are not per-decision latency benchmarks.

Development comparisons are not an independent held-out split or fixed-budget
baseline evaluation. Original
generation-to-legal-observation chronology, setup/acquisition provenance,
probability weights, legal actions, policy, latency/memory and broader ascension
continuations remain open. Complete search here is conditional on the finite
model and is not an exact-domain promotion of the live game.
