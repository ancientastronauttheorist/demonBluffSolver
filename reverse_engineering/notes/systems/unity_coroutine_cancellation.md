# Coroutine cancellation entry points and queue matching

Evidence: `native-static` and `native-emulated` against UnityPlayer SHA-256
`B5D48235E7CC02FF9496FB33A07D5921ADFC4B40DED1BC64C96A7A7C10B4DFB2`.
This extends the [coroutine bridge](unity_coroutine_bridge.md) and
[cursor-aware wait removal](unity_wait_consumer.md). It covers valid managed
wrappers and controlled owner lists with inert release bodies; full native
reference-count destruction and nested lifetime remain open.

## Registered native entry points

The audit rechecks the 3,447-pair registration mechanism and independently
verifies these exact registered names, indexes and targets:

| Registered internal call | Table index | RVA | Next native boundary |
| --- | ---: | --- | --- |
| `UnityEngine.MonoBehaviour::StopCoroutineManaged` | 2275 | `0x100f80` | Queue scan, then `0x77d120` |
| `UnityEngine.MonoBehaviour::StopCoroutineFromEnumeratorManaged` | 2276 | `0x101190` | `0x77d340` |
| `UnityEngine.MonoBehaviour::StopAllCoroutines` | 2264 | `0xff000` | `0x43b8e0`, then owner-list `0x77d120` calls |

The string-named `StopCoroutine` overload is outside this checkpoint.

## Handle cancellation

For valid owner and Coroutine wrappers, the handle overload reads the native
coroutine pointer from the Coroutine object's `+0x10` field. A null native
handle returns without scanning or unlinking. A nonnull handle scans the wait
tree and removes every record satisfying all of:

- Its owner key matches the supplied owner's instance ID.
- Its dispatch callback is the coroutine callback `0x778b30`.
- Its payload is the requested native coroutine, or the payload's `+0x80`
  marker is nonzero and its nonnull `+0x68` link equals the requested coroutine.

Matching waits use `0x43bb00`, preserving the drain's saved successor and
invoking a present release callback after erasure. Owner-key or callback
mismatches retain the wait even when the payload pointer matches. A link alone
without the marker does not qualify.

After the queue scan, `0x77d120` unlinks the requested coroutine from its owner
list, zeroes its list links and clears owner field `+0x58`. The tested simple
coroutines have null `+0x70` and `+0x78` lifetime links. Additional branches
through those fields remain outside the emulated lifetime contract.

## StopAll behavior

StopAll first checks whether the owner's coroutine list at `owner+0x70` is empty.
If it is empty, it returns before queue removal. The synthetic case with an
empty owner list and a matching queued record verifies this guard; it is not
evidence that such an inconsistent combination occurs in normal game state.

With a nonempty list, helper `0x43b8e0` removes waits whose owner key matches and
whose dispatch callback is either `0x778b30` or null. It does not require a
particular payload pointer. Other-owner and other-callback waits remain.
Removal updates the saved cursor and invokes only nonnull release slots.

The wrapper then repeatedly stops the first remaining coroutine in the owner
list until the list is empty. A two-node list confirms that both surviving
nodes are detached under the inert-release test boundary.

## IEnumerator behavior

The IEnumerator overload passes the managed enumerator identity to `0x77d340`.
A null enumerator is a no-op. Its queue scan requires the same owner key and
coroutine callback, plus a nonzero payload handle field at `+0x10`.

Mode `2` at payload `+0x18` compares the cached managed identity at `+0x20`.
Other modes resolve the handle through `il2cpp_gchandle_get_target`. If the
payload's identity differs, a nonzero `+0x80` marker allows the same identity
check through a nonnull `+0x68` link. Unmarked links do not qualify.

The queue scan removes all matching waits. It then searches the surviving
owner list and stops the first matching coroutine, returning immediately.
With two matching owner-list nodes and inert releases, both waits are erased
but only the first surviving list node is explicitly detached. Actual release
destruction can itself change that list, so this does not predict how many
coroutines survive a real call without the remaining lifetime audit.

## Reproducible bounded validation

[`audit_unityplayer_cancellation.py`](../../scripts/audit_unityplayer_cancellation.py)
checks 78 cancellation relationships in addition to the registration audit's
52 relationships, then executes 26 synthetic cases in isolated Unicorn memory.
It runs the wrappers, matching helpers, simple unlink path and native tree
operations. Only node allocation, release bodies, GC field writes and GC-target
lookup are synthetic. Both GC exports are independently identified from their
resolved export-name/storage bindings.

The corpus covers matching filters, marked links, null handles/enumerators,
cached and handle-resolved enumerator identity, all-match queue removal, saved
cursor updates, null callback/release slots, empty-list guards and one- versus
all-node owner-list unlinking. Surviving tree records and invariants are checked.
The [report](../../reports/f530404b0f3f_807de4a83df4_unity_cancellation.json)
contains selected addresses, authored inputs/results and no native bytes.

```powershell
python -m pip install unicorn==2.1.4 pefile capstone
python reverse_engineering/scripts/audit_unityplayer_cancellation.py `
  'B:\SteamLibrary\steamapps\common\Demon Bluff Playtest\UnityPlayer.dll' `
  --output reverse_engineering/reports/f530404b0f3f_807de4a83df4_unity_cancellation.json
```

Invalid managed-wrapper/exception paths, release-body mutation, nested coroutine
stop branches and real reference-count destruction remain unresolved. These
findings do not change managed-method coverage or add live solver behavior.

Validation passed 26 native cases, 78 cancellation relationships plus 52
registration relationships, 778 Python tests, 32 reverse-engineering tests,
Python compilation and diff checks. Rust/simulations were not repeated for
these offline-only audit files after the tested Rust queue checkpoint.

The subsequent [reference-release audit](unity_coroutine_release.md) connects
the native reference count and secondary-handle branches to managed Coroutine
finalizer cleanup. Nonnull waiter objects and arbitrary lifetime/link graphs
remain outside that bounded extension.
