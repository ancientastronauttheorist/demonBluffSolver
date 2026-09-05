# Native reference release and managed Coroutine cleanup

Evidence: `native-static` and `native-emulated`, pinned to the shipped
UnityPlayer, GameAssembly and metadata hashes recorded in the
[report](../../reports/f530404b0f3f_807de4a83df4_unity_coroutine_release.json).
This narrows the remaining lifetime boundary after the
[creation bridge](unity_coroutine_bridge.md) and
[cancellation audit](unity_coroutine_cancellation.md).

## Two cooperating cleanup paths

Native release routine `0x778bd0` decrements the signed reference count at
payload `+0x60`. A positive remainder returns `1` without cleanup. Otherwise it
sets cleanup flag `+0x64`, recursively releases a nonnull `+0x68` link, clears
the reciprocal `+0x68` of a nonnull `+0x70` link, and unlinks its owner-list node.

With the tested null `+0x78` waiter field, it frees a nonzero enumerator handle
at `+0x10` through `il2cpp_gchandle_free`, then clears that handle and its mode
field at `+0x18`. The secondary handle at `+0x28` selects the final branch:

- If absent, the routine calls the allocation-release boundary `0x17a7808`
  with the payload pointer and size `0x88`, then returns `0`.
- If present, it frees that handle, clears `+0x28` and its mode at `+0x30`,
  and returns `0` without the allocation-release call in this invocation.

That second branch cooperates with managed Coroutine cleanup. The registered
`UnityEngine.Coroutine::ReleaseCoroutine` entry at table index 2225 points to
`0xf88b0`, a jump to `0x778cc0`. If the native reference count is nonzero, this
cleanup clears/frees the secondary handle without decrementing references or
unlinking the active coroutine. If the count is exactly zero and the list links
are already cleared, it calls the `0x88`-byte allocation-release boundary.

The native tests exercise both orders:

1. Managed wrapper cleanup first clears the secondary handle while references
   remain. The final later native release clears the enumerator handle and
   releases the allocation.
2. Native completion first clears both handles and leaves the allocation for
   the later managed cleanup, which releases the detached zero-reference payload.

These are controlled invocation orders. They establish the cooperating native
branches, not the time at which garbage collection runs in an actual game.

## Finalizer and internal-call lookup binding

Pinned metadata identifies `UnityEngine.Coroutine.m_Ptr` at `+0x10` and its
`Finalize` method at GameAssembly RVA `0x1c7ad80`. The native method reads that
field and calls the cached internal call; the separate managed wrapper at
`0x1c7ae00` uses the same cache and tail-dispatches the pointer.

Both request the exact string
`UnityEngine.Coroutine::ReleaseCoroutine(System.IntPtr)`. The engine table
registers the name without the parameter signature. GameAssembly resolver
`0x2b7df0` calls `0x265bc0`, which first searches the full key. On a miss, it
searches for `(`, constructs the preceding prefix and retries the same table.
This fallback connects the actual managed request to the registered entry;
the two strings are not incorrectly asserted to be identical.

## Bounded native validation

[`audit_unityplayer_coroutine_release.py`](../../scripts/audit_unityplayer_coroutine_release.py)
checks 73 native relationships across the images, rechecks 52 registration
relationships, and executes 14 cases in isolated Unicorn memory. Only the
reference-release routine, registered cleanup thunk, cleanup body and the
retained-reference path of the auxiliary release helper execute.
GC-handle free and allocation release are synthetic recording boundaries.

The cases cover positive-reference retention, final release with zero/one/two
handles, low-32-bit handle arguments, reciprocal-link cleanup, recursive release,
both managed/native cleanup orders, and auxiliary objects retained by one or
two other references. Allocation-release sinks deliberately
leave emulated storage mapped so field writes can be inspected; these snapshots
do not imply that reading freed memory would be valid in the real process.

```powershell
python -m pip install unicorn==2.1.4 pefile capstone
python reverse_engineering/scripts/audit_unityplayer_coroutine_release.py `
  'B:\SteamLibrary\steamapps\common\Demon Bluff Playtest\UnityPlayer.dll' `
  --output reverse_engineering/reports/f530404b0f3f_807de4a83df4_unity_coroutine_release.json
```

GameAssembly and metadata must be present alongside the pinned engine in the
normal game layout. Native bytes and decompiler output are never written to
the report.

For a nonnull auxiliary object at payload `+0x78`, cleanup first clears object
fields `+0x10`, `+0x18`, `+0x20` and `+0x28`, then passes object+8 to helper
`0x33cc90`. The helper atomically decrements the reference count at object+12.
With an initial count of at least two, it returns without destruction; cleanup
then clears payload `+0x78` and continues. Two native cases verify counts 2→1
and 3→2 and all those field writes. The emulator excludes the last-reference
branch, which calls the object's vtable destructor and enters allocator paths.

The auxiliary object's concrete type and last-reference destructor, the zero-
reference but still-linked diagnostic path, and validity of arbitrary reference/
link graphs remain unresolved. Full cancellation/release-body mutation and
actual lifetime integration remain separate from the finite queue projection.

Validation passed 14 native cases, 73 cross-image relationships, 52 registration
relationships, 778 Python tests, 32 reverse-engineering tests, Python compilation
and diff checks. The preceding Rust queue checkpoint already passed 625 Rust
library tests, all 34 simulations and the release build; no gameplay code changed
in this audit-only checkpoint.
