# Offline round candidate composition

`bluff::round_candidate_composition` implements the pinned `round_candidate_native_v1` caller composition. It prepares distinct script, Bluffable, real-type Villager, real-type Outcast and discarded Good lists before using the existing `round_duplicates` occurrence support kernel. This opt-in API does not alter the live solver.

`replay_trace` accepts supplied integer RNG indices and returns exact attempted service events, complete logical-list snapshots, method-entry order, RNG requests, final lists/versions, and the first error. The API preserves failures before selection: concatenation failure leaves the old duplicate pool intact, while a later filter failure retains the already-cleared pool and every successfully allocated/appended intermediate list. The discarded Good filter remains part of this ordering.

`replay_weighted` first executes that same preparation. A preparation failure returns one failed path with probability one. Otherwise it feeds the actual filtered occurrence lists to the existing duplicate kernel, preserving reference-equal removal and the four-Villager/one-Outcast cap. Every sampled support path is replayed through the exact trace kernel. Final pool contents/version, remaining candidate lists and error must agree with the existing kernel. Failed RNG calls keep incoming mass; failures after selection retain the selected occurrence's probability and native partial writes. Outputs are not merged by role identity or renormalized after failure.

## Contract and limits

Inputs require the exact version, initialized metadata, stable services, reference-based Remove, uniform occurrence support, and distinct intermediate lists with sufficient capacity. The four input roster list identities and duplicate pool identity are structurally distinct. Asset identities may repeat within and across those lists. Asset predicates are the native real-type i32, alignment i32 and nonzero bluffable byte; starting-Evil Villagers/Outcasts remain eligible. Null roster collections and null data occurrences are represented separately.

The bounded API accepts at most 32 total roster/pool occurrences and 32 asset records, at most five supplied indices, and at most 1,024 weighted paths. It also caps retained logical snapshot/event units at 1,048,576, counting actual per-event snapshot lengths and accounting for preparation, sampler support and accumulated result traces together. The existing sampler applies its own bounds before this composition materializes traces. Exceeding a bound rejects the entire invocation; input context is immutable and no partial result is published.

The native ample-capacity fixture has no collection growth. Backing-slot retention, actual List implementations, managed exception unwinding, provider reentrancy and Unity PRNG state remain outside the contract. The supplied-index trace is diagnostic execution, not a probability assertion for an out-of-range supplied index. Only `replay_weighted` supplies the declared uniform occurrence probabilities. Trace draw indices describe the declared attempted service response; an index 0 recorded for a failed/empty RNG request is not a successfully selected occurrence and does not imply PRNG advancement.

## Validation

The authored tests compare all 124 pinned native composition cases against exact entries, events, interned snapshots, draws, final lists and failures. The 18 weighted paths are matched individually to their native draws and 1/18 probabilities. Additional tests cover preparation-failure unit mass, first RNG failure, inline append versus Outcast Add failure, invalid-context atomicity, capacity rejection and u32 version wrap. All six candidate-composition tests passed in the 741-test release library run.

Native evidence remains `reports/f530404b0f3f_807de4a83df4_round_candidate_composition.json` and `scripts/audit_round_candidate_composition.py`. Implementation: `crates/solver-core/src/bluff/round_candidate_composition.rs`; tests: adjacent `round_candidate_composition_tests.rs`.
