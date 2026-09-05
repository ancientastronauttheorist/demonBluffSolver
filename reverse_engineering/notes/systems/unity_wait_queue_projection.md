# Finite one-shot wait queue projection

`bluff::wait_queue` implements `unity_wait_queue_native_v1` from the
[tree](unity_wait_tree.md), [consumer](unity_wait_consumer.md) and
[finite timing](unity_wait_boundary.md) audits. It reconstructs one bounded
drain with explicit owner outcomes and callback effects. It does not infer
hidden readiness, native object identity or callback effects from postmortems.

## Input and output contract

`WaitQueueState` contains the complete in-order one-shot queue, its current
32-bit generation and a monotonic logical-ID cursor. Entries contain finite
timing records and an explicit release-slot-presence flag. Equal-deadline input
order is retained; unsorted input, duplicate/reused IDs and nonfinite keys are
rejected. Logical IDs are history-local labels, not native pointers or handles.

`WaitQueueContext` adds the separately versioned drain-entry clock/frame/phase
snapshot and responses for initial records. The generation-before value must
match the queue. A response becomes required only when a visited record passes
all timing gates. It says either that owner resolution failed, or supplies the
resolved-owner callback's return value and ordered mutations. Unvisited/skipped
records need no response, and response keys outside the initial queue fail.

Supported callback mutations are newly produced WaitForSeconds records and
cancellation of an explicitly identified, still-linked record. Insertions use
the audited float-to-double arithmetic, producer clock, producer frame plus one,
mask `0xA` and current drain generation. Fresh logical labels are allocated in
mutation order. Cancellation never reuses a label, and canceling an already
consumed or canceled label fails instead of assuming a no-op.

The output contains the surviving queue and a chronological trace of visits,
erasures, callbacks, insertions, callback results and releases. Callback results
are recorded after their mutations; release follows only for result exactly
`1` with a present release slot. Missing owners and explicit cancellations erase
and release without invoking a dispatch callback.

## Native ordering and bounds

Future deadlines stop traversal. Other failed gates preserve the record and
advance to its precomputed successor. A callback insertion preserves that saved
cursor, while canceling the saved successor advances it before removal. A newly
inserted record may therefore be unvisited or visited and generation-skipped.
The clock/frame snapshot remains fixed through the drain.

The implementation accepts at most 4,096 initial/current records and 4,096
declared callback mutations. A further traversal bound, checked ID allocation
and whole-call failure prevent partial exploration from being returned as a
valid result. The caller's context is never mutated on either success or error.

This contract assumes one-shot records, non-reentrant callbacks and release
bodies that do not mutate the queue. Native owner lookup, callback lifetime,
actual engine phase observations and producer snapshots remain explicit
provenance. There is no new live-solver, scenario or logical-continuation-registry
caller. In particular, the sealed-ready explorer does not gain authority to
invent a ready set from this API.

## Independent native differential fixtures

The consumer audit can export
[`unity_wait_consumer_v1.json`](../../fixtures/synthetic/unity_wait_consumer_v1.json),
which contains authored synthetic inputs and results obtained from isolated
native execution. The Rust regression compares all 23 cases for complete queue
state, visited IDs, erasure order, callback order and release order. The fixture
contains no game data or native bytes and runs without native dependencies.

The projection omits callback writes to engine clock storage because the native
drain retains its entry-time samples. Subsequent fixture drains provide their
own sampled values. Zero-duration producer snapshots reproduce the controlled
native insertion records used by this corpus; arbitrary producer arithmetic is
separately covered by the finite timing tests.

Eleven additional Rust regressions cover signed-zero occurrence order, ordered
stop/skip gates, owner/release behavior, successor cancellation, insertion on
either side of the cursor, same-frame versus later-frame drains, generation
rollover, full-width signed counters, atomic failures, capacity/allocation
limits and required versioned serialization.

Rebuild the fixture from the pinned local engine:

```powershell
python reverse_engineering/scripts/audit_unityplayer_wait_consumer.py `
  'B:\SteamLibrary\steamapps\common\Demon Bluff Playtest\UnityPlayer.dll' `
  --output reverse_engineering/reports/f530404b0f3f_807de4a83df4_unity_wait_consumer.json `
  --projection-fixture reverse_engineering/fixtures/synthetic/unity_wait_consumer_v1.json
cargo test --release -p solver-core --lib bluff::wait_queue
```

Validation passed 625 Rust library tests, the release build, all 34 simulation
tests across the 426-fixture v2 corpus, 778 Python tests, 32 reverse-engineering
tests, native fixture regeneration, formatting and diff checks.
