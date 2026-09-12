# Round duplicates and shuffle factory boundary

Build `f530404b0f3f_807de4a83df4`. `scripts/audit_round_duplicates.py` and `reports/f530404b0f3f_807de4a83df4_round_duplicates.json` add **52 native fixtures**, including 18 weighted occurrence paths. The three exact declarations have completed private baseline exports. GameAssembly and Dumper script hashes are pinned; no typed targets or live game state are changed.

## Coverage selection

Characters.PickRoundDuplicates (36D720) already has detailed static semantics in `gameplay_bluff_acquisition.md`; this audit adds executable caller evidence. Characters.ShuffleDeck.MoveNext (376B00) and Gameplay.ShuffleDeck (380CE0) already have coverage through the Rambler setup boundary and are not recounted here. The uncovered Characters.ShuffleDeck factory (36E490) and generated Reset (376C40) are included. Existing scheduler evidence continues to govern acquisition and resumption; this audit introduces no new yield timing claim.

## Pool construction and draws

PickRoundDuplicates initializes Gameplay if needed, reads its singleton and obtains the combined script list before clearing DuplicatesPool (+48). Clear increments the pool version and sets count zero before calling Array.Clear for an originally nonempty list. An early script/class failure preserves the old pool; Array.Clear failure retains the changed version and zero logical count.

Next, the native caller requests bluffable candidates, real-type10 Villagers, real-type20 Outcasts, and a Good alignment filter. The last filter's returned list is discarded, but the call and its possible failure remain observable before any draw. These candidate builders are stable explicit gateways here; their individual selection behavior remains documented by the existing acquisition audits.

The Villager draw loop is unguarded on entry. It requests a uniform integer occurrence index, indexes the local candidate list, appends that reference to DuplicatesPool, then removes the first equal occurrence from the local candidate list. It continues until the local list empties or four Villager draws have completed. It then takes at most one Outcast occurrence, again appending before removal. Results preserve order and repeated references.

An empty Villager list issues `Range(0,0)` and calls indexed access with zero; the declared get_Item service rejects that access. The native control flow has not reached its final fallback branch. Under successful stable append/removal services, at least one Villager append makes the fallback count test false. The harness treats reaching the fallback as a failed contract assertion rather than pretending it recovers an empty Villager pool.

## Weighted and partial results

All 18 paths for Villagers `[A,A,B]` and Outcasts `[C,C,D]` execute with widths `3,2,1,3`. Each consumed occurrence path has unconditional probability 1/18; their total is one. Equal identities are not merged. An independent occurrence/removal model verifies the exact output and remaining pools for each path, including removal of the first equal occurrence when a later duplicate index was drawn. Additional runs verify the four-Villager cap with six initial candidates.

The inline Villager Add mutates count, version and the backing slot before its write barrier. A barrier failure therefore retains the appended occurrence and does not perform its removal. A later Remove failure likewise preserves the append. Outcast Add is a before-effect gateway; its injected failure leaves the earlier Villager prefix. Outcast Remove failure retains both additions. Growth failure is tested from a valid initially empty zero-capacity pool: the clear and attempted native Add have already incremented the version, while count remains zero.

These are local candidate/removal and pool-write fixtures. RNG, get_Item bounds, first-equal Remove, candidate filters, growth and Outcast Add remain explicit services. Their implementation or exception-internal rollback is not reconstructed. Normal caller returns verify the stack and all eight Windows nonvolatile integer registers.

## Shuffle scaffolding

Characters.ShuffleDeck allocates its exact generated iterator, executes the native empty base stub, writes state0 and returns. It ignores the supplied Characters receiver, including null, and captures no receiver reference. Current and other object fields remain zero under the managed-zeroed allocation fixture. The factory does not start or resume the iterator.

Generated Reset unconditionally allocates and constructs NotSupportedException, resolves its own exact MethodInfo and calls the runtime throw helper. State and Current remain unchanged. It does not restart the already covered shuffle coroutine. Allocation, exception construction and throwing are gateways; no scheduler or managed unwind semantics are inferred.

Reproduce with `python scripts/audit_round_duplicates.py GAME_ROOT DUMPER_ROOT --output REPORT`, using private Unicorn 2.1.4 through PYTHONPATH.

## Offline weighted replay

`bluff/round_duplicates.rs` exposes versioned `round_duplicates_native_v1` replay. Its inputs provide distinct list identities, stable filter results, reference-equality removal, uniform occurrence support, initial logical pool/version/capacity and explicit successful growth capacity. Aliased lists and unestablished service contracts are rejected. Null filter results are represented separately from empty returned lists. The discarded Good filter remains an observable failing gateway.

Each occurrence path retains its unconditional rational probability, ordered pool contents, remaining candidates, version, capacity and attempted gateway history. Failed paths are retained without renormalizing survivors. An empty Villager list records a zero-width range attempt, no selected occurrence, mass one, and the subsequent indexed-access bounds failure under the audited empty-range-return-zero service. A failing range gateway also retains incoming mass before any successful selection. Neither case claims a native PRNG advance. The native harness's authored index-zero response is deliberately absent from the replay's selected-occurrence field for these attempts.

Inline clear and Villager Add retain their native mutation-before-failure prefixes. Outcast Add, growth and first-equal Remove use the explicit fixture service effects. The replay models logical contents, not stale backing slots or managed unwind internals. Successful growth reserves the caller-provided additional slots; it does not infer a runtime growth policy. Input entries, retained entries and occurrence support are bounded; exceeding those limits rejects the replay as unsupported capacity instead of inventing a native failure. No shared game state, acquisition scheduling or live RNG is consumed.

The accompanying tests compare all 47 duplicate-selection fixtures (the five shuffle scaffolding fixtures remain native-only), all 18 weighted paths, exact event operands, unchanged inputs, empty/failing range mass and provenance/capacity/version guards.

Validation: all four Rust duplicate-selector test groups passed, including the
47 native duplicate fixtures and all 18 occurrence paths. Independent review
found no mismatch in rational mass, retained budgets or service failure effects.
