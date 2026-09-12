# Characters filter tail

Build `f530404b0f3f_807de4a83df4`. The supplemental `scripts/audit_characters_filter_tail.py` and `reports/f530404b0f3f_807de4a83df4_characters_filter_tail.json` cover three Characters callers plus the exact captured predicate, with **83 passing native cases**. The caller baseline exports were prepared separately; the predicate is decoded directly from the pinned PE. GameAssembly, Dumper script and dump hashes are checked. No typed target or live state is changed.

## Status and visibility predicates

FilterCharacterContainsStatus (36A6D0) allocates a result list before checking its input. In input occurrence order it follows Character.statuses (+F0), then CharacterStatuses.statuses (+10), and asks the enum-list Contains gateway about the supplied int32 status. Matching characters are appended unchanged. It does not inspect the resistance list or infer status from alignment or appearance. Null input, character, wrapper or inner status list fails at the relevant boundary; earlier output remains local and unreturned.

FilterRevealedCharacters (36BCD0) also allocates before the input check. It keeps every nonnull Character whose state (+E4) is **not exactly 5**. This includes tested unknown and negative state values, zero, and other positive states. The name does not establish a Boolean revealed flag or a whitelist of revealed states. Duplicate input occurrences remain duplicated; a null character fails after any prior appends.

Both callers ignore their Characters receiver in these fixtures. Allocation, list construction, enumerator operations, enum Contains and Add remain gateways with checked generic metadata. The authored enumerator records its source version and rejects a changed version on the next MoveNext. A version-changing callback case retains the completed output prefix. This is the explicit collection-service contract, not an execution of native enumerator internals or managed exception unwinding.

## Unique not-in-play filter

FilterNotInPlayCharactersUnique (36B3F0) first shallow-copies its supplied CharacterData list into a new list. It initializes Gameplay if needed and iterates the global current Character list. For each nonnull board Character, it passes dataRef (+50), including null, to the copied list's managed Contains gateway. Only a true response constructs a closure capturing that dataRef and invokes RemoveAll.

The captured predicate at **377510** is not string equality or a local reference comparison. It loads the captured data reference, initializes UnityEngine.Object if needed, and tailcalls **UnityEngine.Object.op_Equality (1C822C0)** with candidate and captured references. The audit executes these predicate instructions for each occurrence visited by its RemoveAll gateway. It verifies the exact closure MethodInfo at delegate construction, keeping it separate from managed Contains's generic context.

The fixtures deliberately keep managed Contains and Unity-object equality as distinct gateways. A supplied Unity null-equivalence example removes both literal null and an equivalently classified object only after the earlier managed Contains gate succeeds. When that gate is false, no predicate is called. These controlled responses demonstrate the caller dependency; they do not claim recovered Equals/comparer or destroyed-object behavior inside either service.

Within the ordinary reference-equality fixtures, a matching board entry removes every matching occurrence from the copy. Repeated board entries then see the already filtered copy. Absent matches preserve duplicates. Null board lists/characters stop iteration; original input occurrences remain unchanged. The caller is not a general input deduplicator.

## Failure scope

Every RemoveAll invocation in this harness runs the actual native captured predicate over a snapshot and commits its authored removals only after all predicates succeed. A predicate failure therefore preserves that invocation's pre-removal list in **this service contract**. Native List.RemoveAll can have its own internal partial writes, which this audit neither executes nor promises to roll back. Earlier completed RemoveAll invocations still remain observable.

The tests cover source nulls, all status-chain nulls, signed state/status inputs, duplicate/null occurrences, ordered board removal, separate equality gates, allocation/copy/Contains/RemoveAll/predicate/class-init failures and version failure after an append. Normal caller returns verify the stack and all eight Windows nonvolatile integer registers. The fixture records original input preservation and local output prefixes; failed callers do not return those prefixes successfully.

Reproduce with `python scripts/audit_characters_filter_tail.py GAME_ROOT DUMPER_ROOT --output REPORT`, with private Unicorn 2.1.4 available through PYTHONPATH. Collection internals, Unity equality, runtime class initialization, allocation, callback mutation beyond the declared version fixture and exception unwinding remain explicit boundaries.

## Offline Rust replay

`solver_core::character_filters` exposes a deterministic, versioned
`character_filter_native_v1` caller replay. The strict context requires stable
input contents and the snapshot/commit RemoveAll service contract above.
Managed Contains and Unity equality are supplied as separate ordered response
sequences; the model does not choose their equality policy. Missing response
provenance rejects the invocation, and unused suffix responses are allowed.
Preflight requires complete potential-path responses even when a requested
service failure would stop earlier.

The kernel returns local output prefixes, service order, consumed response
counts, initialized-class flags and the resulting input version. It distinguishes
native null/version/service failures from unsupported contexts and capacity
rejection. An explicit Add callback option advances input version without
changing input contents, reproducing the native audit's next-MoveNext version
failure. Input/board/character/status sizes are capped at 4,096 and equality
response sequences at 65,536. No weighted branch inference is required.

The companion `character_filter_tests.rs` normalizes all 83 native cases and
checks independent equality responses, strict fields and unsupported service
provenance. The API is offline and does not affect live solver behavior.

The Rust result also records every attempted managed Contains target and every
Unity `(candidate, captured)` pair, including failing calls. All 83 comparisons
check those operands against native events; a focused test changes the captured
target while keeping supplied equality responses independent.
