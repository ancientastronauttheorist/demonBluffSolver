# Offline round unique-pool replay

`bluff::round_bluffs` provides the versioned `round_bluffs_native_v1` replay for the audited PickRoundBluffs caller and its separately executed capture predicate. It is an opt-in offline API; live solver behavior is unchanged.

`replay_trace` retains the full attempted event sequence, all logical lists/versions, captured-script publication, singleton presence, RNG requests, first failure and predicate AL result. It follows the native ordering: closure allocation/construction, class initialization, all-character getter, singleton reread and script getter, captured-reference publication/barrier, unique-pool clear, predicate allocation/construction, then RemoveAll. A clear service failure occurs after the count/version writes; Add failure occurs before append; Remove failure occurs after append.

The initial sampler checks empty Villager candidates before drawing, samples up to four occurrences with first-equal removal, then at most one Outcast. If the resulting pool has zero or one item, it performs exactly one fallback through GetAllAscensionCharacters, Bluffable, Good alignment and real Villager providers. The fallback has no pre-draw emptiness guard, no script exclusion, and no post-draw removal. Empty fallback therefore retains the prior pool and reports bounds after an attempted zero-width RNG request; successful fallback from an empty pool produces one entry.

The standalone predicate loads its captured script, fails on a null capture, and forwards candidate identity to an explicit Contains service. Its declared return value is preserved as AL without normalizing high return-register bits. This predicate replay is separate from the enclosing RemoveAll service; it does not assert that native List.RemoveAll executes the modeled predicate or rolls back partial mutations.

## Weighted support and failure mass

`replay_weighted` reruns the pure trace to its next unresolved RNG request, branches uniformly over occurrence indices, and retains each resulting trace. This preserves duplicate-reference occurrence mass without merging equal outputs. A failing RNG service or zero-width request does not split probability. Earlier failures retain unit mass; failures after earlier draws retain their incoming probability. Successful paths are never renormalized after excluding failed paths.

A recorded index zero on an empty/failing Range attempt is the declared planned service response, not a successful selected occurrence or proof of PRNG advancement. Only the weighted API supplies the uniform probability contract.

## Explicit contract and bounds

The context requires initialized metadata, stable services, distinct lists with sufficient capacity, reference-membership RemoveAll and first-reference Remove, and uniform occurrence support. All/script/fallback inputs and asset predicate fields are explicit. Filter services commit their selected logical lists before optionally returning null; RemoveAll commits after its supplied membership outcomes. Controlled singleton disappearance after a named successful getter is supported, with native caller rereads preserved. Arbitrary callbacks/reentrancy, native List internals, comparer behavior, managed unwinding, allocation layout and Unity PRNG state remain outside the contract.

At most 32 combined input occurrences and 32 asset records are accepted. Weighted support is capped at 1,024 paths and 1,048,576 retained logical event/list units, accounting for pending request prefixes and completed traces together. Rejected contexts/capacity leave the input immutable and publish no partial distribution.

## Validation

Six focused tests compare all 108 native cases, including exact events and interned snapshots, final lists, draw requests, error and standalone predicate AL. The 18 initial paths and nine fallback paths are matched individually to native probability and final-state records. Additional checks cover all pre-draw failure mass, sampling failure order, version wrap, unsupported-context atomicity and branch capacity. All six unique-pool tests passed in the 747-test release library run.

Evidence: `scripts/audit_round_bluffs.py`, `reports/f530404b0f3f_807de4a83df4_round_bluffs.json`, and `notes/systems/round_bluffs.md`. Rust: `crates/solver-core/src/bluff/round_bluffs.rs` and adjacent `round_bluffs_tests.rs`.

Two explicit null-source regressions preserve the shared filter gateway labels
before the null failure. A null source does not carry a fallback-list identity;
trace comparisons retain the native fixture's supplied service naming contract.

The trace field `project_present` names the modeled Gameplay singleton presence;
it does not represent a ProjectContext lookup or an editor project state.
