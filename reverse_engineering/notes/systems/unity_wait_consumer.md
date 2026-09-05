# One-shot native wait traversal under callback mutation

Evidence: `native-static` and `native-emulated` against UnityPlayer SHA-256
`B5D48235E7CC02FF9496FB33A07D5921ADFC4B40DED1BC64C96A7A7C10B4DFB2`.
This extends the [tree audit](unity_wait_tree.md) and
[consumer boundary](unity_wait_boundary.md). It supplies a bounded mutation
contract, while owner lookup and actual coroutine lifetime remain separate.

## Recovered traversal contract

Consumer `0x43bd90` samples the deadline clock and full signed frame counter
once, increments the wrapping 32-bit generation, and starts at the minimum
node. A future deadline stops traversal before the phase gate. Otherwise the
consumer computes and saves that node's current in-order successor before
checking phase, generation, frame and owner eligibility.

Skipping a phase, generation or frame gate preserves the record and proceeds
to the saved successor. An eligible one-shot record is erased before its
callback runs. A missing owner table, missing owner entry or null owner pointer
instead erases it and invokes its release callback if present, without a
dispatch callback. After a dispatched callback, release occurs only when its
return value is exactly `1` and the release slot is nonnull.

The cursor-aware erase helpers `0x43bc60` and `0x43bb00` check whether the
removed node is the saved successor. If so, they advance that cursor before
unlinking the node. The latter helper additionally invokes the stored release
callback after erasure. Canceling several successive nodes can advance the
cursor all the way to the end sentinel.

Insertion leaves the saved successor unchanged. A new node before that cursor
is not visited during this drain. A node after the cursor can become a later
node's successor and be visited. Insertion when the saved cursor is already
the end sentinel does not restart traversal. The generation check excludes a
newly produced wait even if the drain does visit it.

The [phase audit](unity_wait_phases.md) identifies two mask-2 callbacks in the
default loop. A later drain can therefore have a different generation in the
same frame; the independent frame threshold still excludes a newly created
WaitForSeconds record until its threshold passes. Changing engine time or
frame storage during a callback does not replace the current drain's sampled
values. The following drain samples anew.

## Isolated native execution

[`audit_unityplayer_wait_consumer.py`](../../scripts/audit_unityplayer_wait_consumer.py)
executes the native consumer, both cursor-aware erase helpers and native tree
operations in Unicorn. Native code can execute only within explicitly audited
ranges. Allocation, owner-hash lookup, profiling, callback bodies and release
bodies are synthetic boundaries. In particular, a successful synthetic owner
lookup does not establish that a real coroutine owner exists.

At a dispatch callback the emulator pauses. The harness saves its CPU context,
runs requested native insertion/cancellation operations on a separate emulated
stack region, restores the paused context and supplies the callback result.
This avoids recursive emulator execution and retains mutations to the shared
emulated queue. Synthetic release bodies only record the event and return.

The [23-case report](../../reports/f530404b0f3f_807de4a83df4_unity_wait_consumer.json)
checks callback, visit, release and surviving-record orders for:

- Equal deadlines, future-deadline stop and all three skip gates.
- Wrapping generations, signed counters and a threshold differing only above
  the public frameCount getter's low 32 bits.
- Missing tables/entries, null owners, null release slots and callback results
  `0`, `1` and `2`.
- Cancellation of the saved successor, later nodes and the entire remainder.
- Insertions before/after the cursor and after the saved end sentinel.
- Same-frame versus later-frame rechecks and retained entry-time clock samples.

Every callback boundary and completed drain checks surviving tree payloads,
links, red/black invariants, extrema, count and stable deadline order. Three
additional CI tests verify the authored one-shot record fixture's field widths,
null release slot and repeat-flag exclusion without native dependencies.

## Reproduce and limits

```powershell
python -m pip install unicorn==2.1.4 pefile
python reverse_engineering/scripts/audit_unityplayer_wait_consumer.py `
  'B:\SteamLibrary\steamapps\common\Demon Bluff Playtest\UnityPlayer.dll' `
  --output reverse_engineering/reports/f530404b0f3f_807de4a83df4_unity_wait_consumer.json
python -m unittest discover -s reverse_engineering/tests
```

The audit does not cover repeating waits, recursive drain calls, mutation from
release bodies, native owner-table behavior or full coroutine cancellation and
lifetime. It does not infer game callback actions, runtime clock snapshots or
readiness from a completed postmortem. Those remain explicit provenance needed
before automatic Reveal scheduling or scenario integration.

Validation passed all 23 native consumer cases, a repeat of the 26,496-operation
tree corpus after extending its harness, 778 Python tests, 32 reverse-engineering
tests, Python compilation and diff checks. The tree report was unchanged.

The subsequent [Rust queue projection](unity_wait_queue_projection.md) replays
these 23 native cases from redistributable synthetic fixtures. The audit's
optional `--projection-fixture` argument regenerates their complete queue
states, visit order and callback/erase/release events from native execution.
