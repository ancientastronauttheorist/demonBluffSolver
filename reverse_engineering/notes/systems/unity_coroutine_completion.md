# Native coroutine completion and queue ownership

Evidence: `native-static` and `native-emulated` against the pinned UnityPlayer,
GameAssembly and IL2CPP metadata. This composes the existing
[managed-step bridge](unity_coroutine_bridge.md),
[reference release](unity_coroutine_release.md) and
[one-shot consumer](unity_wait_consumer.md) in an isolated native harness.

## Managed invocation and survival guard

Dispatcher `0x778d90` temporarily increments the payload's reference count,
constructs an invocation frame through native helper `0x75d3a0`, and invokes
the cached `SetupCoroutine.InvokeMoveNext`. The cached-enumerator mode reads
payload `+0x20`; the other supported mode obtains the enumerator through the
GC handle at `+0x10`. The native frame passes that managed object and an address
at which the managed bridge writes its boolean result.

The new audit independently binds the parameter-count and runtime-invoke
export slots and checks the actual argument layout in the synthetic invocation
boundary. Native frame construction executes; it is not replaced by manually
initialized stack offsets.

After invocation and optional exception reporting, the dispatcher records the
current reference count and releases its temporary reference. If that recorded
count was exactly one, cleanup has consumed the last reference: the dispatcher
returns zero immediately. It does not write the optional error-out byte or
attempt further owner/yield work after this guard.

With a surviving reference, it writes the error flag when an output pointer
was supplied. An exception prevents both completion-link handling and yielded-
current processing. A successful continuing result reaches current-yield
processing only while the payload still has an owner. A successful completed
result takes another temporary reference around completion-link handling;
the no-linked-child cases return one with their original surviving count.

## What the queue owns

For a completed coroutine with one initial reference and no intervening release,
the native one-shot queue path performs these count transitions:

| Boundary | Reference count |
| --- | --- |
| Dispatcher enters managed invocation | `1 → 2` |
| Invocation temporary reference released | `2 → 1` |
| Completion-link temporary reference | `1 → 2 → 1` |
| Callback returns one; queue releases its reference | `1 → 0` |

The final release unlinks the payload and clears its GC handles. With a
secondary managed-wrapper handle present, allocation release remains deferred
to the finalizer path established in the separate release audit.

The harness also executes controlled native releases while managed invocation
is suspended. If those consume every external reference, the dispatcher's
temporary release cleans up, the callback returns zero, and the queue performs
no second release. With other references remaining, the queue releases exactly
its one reference after a return of one. This is why a managed completion flag
alone is insufficient to infer the native callback result.

## StopCoroutine and mismatched owners

The harness can call the actual registered `StopCoroutineManaged` body while
the synthetic managed invocation is suspended. In the covered graph, the
current one-shot wait has already been erased. Stop unlinks the payload and
clears its owner without consuming its reference count. After a continuing
managed result, the dispatcher observes the cleared owner and suppresses
current-yield processing. Native release still follows the reference rules.

Callback `0x778b30` compares the resolved owner against payload `+0x58` before
entering the dispatcher. A mismatch calls a diagnostic sink and returns one
without invoking MoveNext. The queue then releases its reference. The audit
covers both direct callback invocation and actual queue entry for this case.

Four additional cases compose actual StopAll with two independent coroutines.
The current wait has already been erased when its invocation calls StopAll;
the sibling is the saved successor. StopAll removes and releases that sibling's
wait, advances the saved queue cursor, and then detaches surviving list nodes.
A sibling with one reference is destroyed and unlinked during queue removal;
one with two references remains for the subsequent owner-list detach. The
queued sibling never invokes MoveNext. Both native and managed-wrapper-retained
allocation cases are checked.

## Reproduction and limits

```powershell
python reverse_engineering/scripts/audit_unityplayer_completion.py `
  'B:\SteamLibrary\steamapps\common\Demon Bluff Playtest\UnityPlayer.dll' `
  --output reverse_engineering/reports/f530404b0f3f_807de4a83df4_unity_completion.json
```

The audit verifies 42 new native relationships, rechecks the 60 bridge and 95
release relationships and 14 release cases, then runs 848 completion cases.
The matrix varies reference counts one through four, cached/GC enumerators,
return and exception states, actual StopCoroutine calls, secondary handles,
zero through all external reference releases, and direct/queued entry. The
report retains representative traces plus a deterministic digest of the whole
verified corpus. Native bytes and decompiler bodies remain private.

Managed execution and exception construction, GC APIs, owner-context lookup,
diagnostic and allocation/free sinks are synthetic boundaries. Current-yield
processing is only a recorded gateway in direct continuing tests; queued cases
that would require it are excluded. Native calls made during invocation run on
a separate emulated stack with the suspended CPU context restored afterward.
Free sinks leave memory mapped for inspection; snapshots do not authorize real
reads after free.

Arbitrary linked/nested coroutine graphs, complete current-yield production,
reentrant queue drains and the full lifetime graph remain unresolved. The
[scheduled Reveal adapter](scheduled_reveal.md) consequently continues to
require explicit matching-owner and callback-result provenance.

Validation passed all 848 native cases, the pinned bridge/release rechecks,
32 reverse-engineering tests, Python compilation and diff checks. The preceding
queue-driven Reveal checkpoint passed the full Rust, Python and simulation gates.
