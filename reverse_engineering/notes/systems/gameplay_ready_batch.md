# Sealed ready-batch exploration

`bluff::ready_batch::replay_ready_batch` is a bounded offline interleaving
explorer layered over the ordered Reveal v2 model. It is not an implementation
or observation of Unity's scheduler. The native boundary remains the engine
handoff described in the bluff-acquisition lifecycle audit: Reveal is atomic
game code, while coroutine eligibility and ordering belong to UnityPlayer.

## Required interpretation

The `sealed_ready_batch_native_v1` input identifies up to six distinct, already
ready coroutine instances by caller-provided IDs and physical body positions.
Each body's ready count must fit its explicit pending-continuation count. IDs
must be unique even when several instances reference the same body. The input
also supplies the full v2 board/UI state with an empty resume list.

The caller must prove that no other coroutine becomes eligible before this
batch drains and that no unmodeled events interleave. This condition is not
derived from pending counts, board order, registration order, equal delays,
wall-clock time, or the existence of the six-item cap. Arbitrary partial-ready
sets without that guarantee are not valid uses of this contract. UI/asset and
Unity-lifecycle assumptions from ordered Reveal v2 still apply.

## Enumeration and probability

All permutations of the supplied IDs are explored, including distinct orders
of siblings that reference one body. The input order only determines output
enumeration order. Each result records its ID order and its own collection of
game-RNG paths. There is deliberately no probability on a schedule. For each
fixed schedule, native selector/writer probabilities multiply normally; a
probability sum across different schedules has no interpretation.

Before each resume on each RNG branch, the explorer reads that branch's raw
bluff liveness to decide whether acquisition occurs. It calls the ordered v2
kernel for exactly one event, then carries the updated board, UI and accumulated
trace into the next event. The trace uses zero-based **simulation-local** resume
ordinals and the same local index for acquisitions when present. These numbers
are not captured native event provenance and must never be promoted into a
live solver observation. Existing ordered-replay contracts still enforce their
original explicit acquisition assertions.

This branch-local decision handles a formerly unsupported fixed-flag case:
after several Twin self-versus-distinct swaps, one branch can retain a live
Drunk bluff while another has cleared its bluff and needs Minion acquisition.
Both remain in the fixed schedule's distribution. No branch is dropped or
renormalized to satisfy a shared acquisition flag.

## Continuations and limits

Only IDs in the initial sealed batch are resumed. Twin-created continuations
stay pending and receive no invented IDs or readiness. Each final path reports
new continuation counts by physical body, separately from initial pending
instances outside the batch. The count follows from the audited absence of
cancellation: final pending minus initial pending not included in this batch.
An empty batch yields one identity schedule and path.

The explorer rejects malformed readiness/identity input, any unsupported or
failing positive-mass game branch in any order, checked probability overflow,
more than 65,536 retained result paths, or more than 1,048,576 retained entries.
It does not return a successful subset of schedules after a failure. UI state,
historical callback/view traces and deferred-count maps count toward the bound.
The maximum six-item set bounds upfront permutation generation to 720 orders.

Six regressions cover separate schedule weights, deferred writer registrations,
branch-local reacquisition with each schedule retaining unit RNG mass,
equivalence to explicit one-event replay, strict identity/count/schema input,
empty identity and atomic game-branch failure. The implementation adds no native
method classifications: 532 classified methods and 276 evidence records remain.

Native readiness capture, constraints spanning several readiness batches,
engine ordering behavior and unrestricted scene callbacks remain open. This
explorer is offline only and has no live solver or scenario-generation caller.

## Logical continuation handoff

`bluff::continuation_registry::advance_ready_batch` adds the separate
`logical_continuation_registry_native_v1` contract. The input carries complete
v2 board/UI state, a map of all pending logical IDs to physical bodies, a
monotonic next-ID cursor and a batch ordinal. Per-body registry counts must
exactly equal remaining_continuations, including instances not ready in this
batch. Every existing ID must be below the allocation cursor. Registries are
limited to 4,096 entries and individual ready batches to six distinct IDs.

The caller supplies ready IDs from that registry under the same sealed-batch
guarantee. The adapter maps them to the existing bounded explorer, then maps
each schedule back to the logical IDs. Every completed path removes exactly
the requested ready instances and preserves all unready initial instances.
It walks the actual Start callback/replacement traces in execution order and
allocates one new label for each InitWithNoReset-created continuation. The
trace includes resume ordinal, callback index and replacement index so two
same-body self-swap instances remain distinguishable. Allocated counts must
agree with the explorer's independent new-continuation count calculation.

The returned state can be passed directly to another explicitly ready batch.
No new label is automatically ready. An already consumed or unknown ID is
rejected. Allocation and batch-counter overflow reject the whole call; IDs
are never wrapped or reused along a history. An empty batch preserves board,
UI and pending state while advancing only the logical batch ordinal.

Labels are **history-scoped simulation identities**, not native handles. The
same numeric label can identify different hypothetical instances on different
RNG/schedule branches. A caller must carry the complete selected history/state,
not join IDs from alternative paths. Native registration addresses, readiness,
frame boundaries and elapsed time are not inferred from these labels. As with
all contexts, a serialized handoff is trusted provenance, not a tamper-proof
record of an earlier execution.

Each returned probability is conditional on the current batch's order and
input state. It is not the probability of the prior history; a caller composing
histories must retain and multiply prior RNG weights separately, without
inventing probabilities for schedule choices. The registry adapter preserves
the explorer's fail-whole-result behavior and applies an additional retained
entry budget including registry, creation provenance, UI and callback traces.

Six regressions cover ordered allocation and preserved unready instances,
later-batch consumption exactly once, self-swap identity separation, complete
registry/ready-set validation, counter overflow, and empty-batch serialization.
This extends logical handoff only; native readiness evidence remains open.

Registry checkpoint validation passed 606 Rust library tests, 778 Python tests,
13 reverse-engineering tests, release build, formatting and coverage integrity.
Coverage remains 532 classifications and 276 evidence records. The long
simulation suite was not rerun for this offline-only addition.

Checkpoint validation passed 600 Rust library tests, 778 Python tests, 13
reverse-engineering tests, release build, formatting and coverage integrity.
The long simulation suite was not rerun for this isolated offline addition.
