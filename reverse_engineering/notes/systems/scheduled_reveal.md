# Queue-driven weighted delayed Reveal

`bluff::scheduled_reveal` implements `scheduled_delay_reveal_native_v1`. It
connects the [native-tested one-shot queue](unity_wait_queue_projection.md) to
the existing exact Reveal writer and logical continuation registry. Evidence
is compositional: queue behavior has an independent native differential corpus;
Reveal/Start/writer behavior has its own pinned audits and Rust regressions.
The combined adapter has synthetic Rust integration tests, not a claim of a
captured live end-to-end execution.

## Explicit input boundary

The input contains a complete DelayReveal-only queue and complete continuation
registry, with identical pending labels and allocation cursors. Labels remain
history-local simulation identities, never native pointers. Every queue entry
must have the audited WaitForSeconds phase mask `0xA` and a present release slot.
The queue's finite deadline order, unique labels, generation and dispatch
snapshot are validated by the shared queue kernel.

For each initial record that actually passes timing gates, the caller supplies
confirmation that lookup resolves the same live owner as the coroutine payload,
the native callback return value, and the producer's retained frame clock and
full signed frame counter. Missing evidence for a skipped record is acceptable;
missing evidence at an actual callback fails the entire exploration. Unknown
record labels and false matching-owner confirmations are rejected.

The producer snapshot is explicitly stable through that synchronous callback
and its immediate writer-created coroutine starts. It is not inferred from the
consumer's public selected time. The native callback return value includes
lifetime accounting and is not inferred from managed MoveNext completion.
Only a return value of exactly one generates the queue's release event.

This version requires non-reentrant dispatch and queue-inert release bodies.
It excludes cancellation, failed/stale owners, unrelated waits, nested/other
yield types, callback subscriptions outside the Reveal contract, and missing
view-state provenance. It is an offline API with no scenario or live caller.

## Branching and new wait production

The queue kernel pauses immediately after erasing an eligible one-shot entry.
The adapter advances exactly that logical continuation through a singleton
registry batch and forks the paused kernel for each exact RNG branch. Queue
order is not permuted or assigned probability. Branch probabilities multiply
the existing reduced rational weights across callbacks.

Each Twin replacement trace creates a logical continuation in actual write
order. The adapter produces one corresponding `0.3f` WaitForSeconds record,
using the [immediate coroutine-start binding](unity_coroutine_bridge.md), the
explicit producer snapshot and current drain generation. Queue and registry
allocate the same fresh labels in the same order. Distinct self-swap writes
therefore stay distinct even when both target one body.

The shared saved-successor kernel determines whether an inserted record is
visited later in this drain. If reached, it cannot pass the current-generation
gate. Newly inserted waits are never admitted to a second Reveal in the same
drain, even when their supplied deadline and signed frame threshold are past.
A later drain has its own clock/frame/phase snapshot and owner evidence.

One registry batch is recorded per actual callback, so its local resume and
acquisition ordinal starts at zero. The outer callback vector and logical ID
provide within-drain chronology. Drains with no callbacks leave the registry's
batch ordinal unchanged. Returned state roundtrips for the next explicit drain;
the returned weight is conditional on this drain, not a prior-history weight.

## Failure and validation

The adapter bounds a drain to 16 Reveal callbacks, 65,536 branches and 1,048,576
retained logical entries, including saved callback states and detailed traces.
The shared queue and registry enforce their own record, mutation and checked-ID
limits. Any unsupported path, missing later callback evidence or overflow
returns an error without returning a valid-looking explored prefix or mutating
the caller's context.

Ten integration regressions cover exact weighted equivalence with a separately
selected sealed schedule, equal-deadline occurrence order, overdue insertion
visits and generation skips, distinct producer clocks, subsequent-frame
handoffs, exact release-result behavior, malformed joins and owner evidence,
16/17-callback and allocation limits, wrapping generation, finite values and
required versioned serialization. The queue's 23 native differential cases
continue to exercise the same implementation used by this adapter.

```powershell
cargo test --release -p solver-core --lib scheduled_
cargo test --release -p solver-core --lib bluff::wait_queue
```

Validation passed 635 Rust library tests, all 34 release simulations across 426
fixtures, the release build, 778 Python tests, 32 reverse-engineering tests,
formatting and diff checks.
