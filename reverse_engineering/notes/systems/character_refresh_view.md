# Character.RefreshView: executable presentation caller

Pinned build `f530404b0f3f_807de4a83df4`. This strengthens the existing
[native-static view evidence](gameplay_reveal_view.md) with an executable audit
of the complete `Character$$RefreshView` caller at RVA `0x367B60`. Its normal
return is `0x367DB8`; decoded exclusive end is `0x367DFB`, before the null-helper
trap and trailing alignment. The next managed entry begins at `0x367E00`.
Coverage target is `tdi5487.m0063`, already understood from static evidence.

The audit runs 1,645 individual cases, eight explicit callback-mutation cases,
three retained repeated-call sequences and 29 exact controlled-stop prefixes.
It executes 153 of 155 decoded instructions. Normal returns verify stack,
all eight nonvolatile integer registers and all ten nonvolatile XMM registers.
GameAssembly and Dumper fingerprints, exact method/service signatures, 28
instruction relationships and consumed fields are pinned. Class declarations
are bound to global Character TypeDefIndex 5487 and UnityEngine.Vector3
TypeDefIndex 6699, including their namespaces.

## UI predicate and source boundaries

Native code first hides `pickable` when the signed `pickableUses` count is at
most zero. A positive count preserves its current active state. A required null
control fails only when that hide path is reached; the count is never changed.

Only Dead state (`20`) checks the existing `createdDeadPrefab` through Unity
equality. An absent or authored destroyed object creates a replacement. A live
object skips the entire creation/transform/RIP sequence. Other states retain
the existing raw pointer without consulting its liveness or destroying it.

The optional disguise icon is tested through Unity inequality. An absent or
authored destroyed icon skips its UI setter. A nonzero `killedByDemon` byte also
preserves the prior active state. Otherwise the icon is active exactly when
state is Dead (`20`) or Revealed (`30`) and the raw `bluff` is live. The separate
`revealed` flag does not enter this predicate; the full matrix varies it
independently. Raw pointer fields are reloaded after the liveness callbacks.

This caller reads no apparent/real role, alignment, statuses, resistance,
speech, copied action role, or background/border colors. Those sources belong
to other audited methods. The fixture fills every other actor byte with
nonzero sentinels and verifies preservation, except the explicitly stored new
death object and named authored callback mutations. Pixels and renderer
behavior are not represented.

## Exact creation and transform call order

The creation path consumes these services in order:

1. Capture the current `deadPrefab` pointer, then request the Character's
   Component transform.
2. Call the folded native `Object.Instantiate<object>` body at `0x668010` with
   the pinned `Instantiate<GameObject>()` MethodInfo and parent transform.
3. Store the returned pointer to `createdDeadPrefab`, then call its GC barrier.
   The pointer is already visible when the barrier is entered or fails.
4. Reload that pointer and request its GameObject transform.
5. Request the icon's Component transform and its position.
6. Copy the returned 12-byte Vector3 and set the first created transform's
   position.
7. Reload the current created-object pointer and request its GameObject
   transform again. The caller does not reuse the first transform reference.
8. Copy the 12-byte `Vector3.zeroVector` from supplied static storage and set
   the second transform's Euler angles.
9. Activate `ripView`.

The Vector3 getter uses the Windows x64 structure-return buffer in RCX, the
source Transform in RDX and MethodInfo in R8. Its returned RAX pointer supplies
the copied bytes. Setters receive Transform in RCX, the address of the copied
Vector3 in RDX and MethodInfo in R8. The native body copies two components with
`movsd` and the third as a DWORD; it performs no floating-point arithmetic.
Fixtures preserve exact IEEE-754 bits, including signed zero, subnormals,
infinities and a NaN payload. Authored distinct first/second Transform tokens
verify both getter calls and which setter consumes each identity.

`zeroVector` initialization is an explicit runtime input. The ordinary fixture
supplies all-zero bits. Two authored nonzero static-field probes establish that
the body copies those fields rather than materializing its own zero literal;
they do not claim Unity normally initializes zeroVector to those values.

## Partial state and callback reloads

Null dependencies and API results preserve exact prefixes:

- A null Instantiate result is stored and barriered, then fails the following
  null check; an old destroyed reference is not restored.
- A null first created Transform still permits icon transform/position calls
  before the native check of that captured Transform. No position setter runs.
- A null second created Transform fails after the position setter. Cold
  metadata resolution for Vector3 can occur before that check.
- A null RIP object fails after both position and Euler setters. The newly
  stored death object and completed transform writes remain.

Eight named callback probes preserve actual capture/reload timing. Changing
`deadPrefab` during the Character transform getter does not replace the already
captured Instantiate source. Changing `createdDeadPrefab` during its barrier
changes the later GameObject-transform input. Clearing it during icon position
retrieval still permits the first position setter, then fails at the second
created-object reload. Disguise liveness callbacks can change the later killed
guard, state selection or reloaded icon identity.

These mutations are authored service effects, not claims that Unity's actual
APIs perform them. Likewise the supplied Instantiate service accepts authored
null prefab/parent stress inputs to expose the caller's checks; this does not
claim actual Unity accepts a null prefab. Unity equality and inequality return
only the Boolean consumed through AL while upper return-register bits are
poisoned deliberately.

Repeated-call sequences retain the same actor memory, newly created identity
and UI/transform effects. The second successful refresh reuses the live object,
does not instantiate again, and leaves all actor bytes unchanged. No destroy or
new scheduler action is synthesized.

## Coverage and reproduction

The two unexecuted decoded instructions are class-init calls at `0x367C25` and
`0x367DCD`. Earlier liveness checks have already initialized Unity Object under
the supplied runtime contract. No callback invalidates that class cache. The
report lists both exclusions explicitly; native exception unwinding and the
post-null-helper trap are outside its execution boundary.

All 29 injected service stops reproduce the successful event prefix and full
semantic actor/UI/transform snapshot. They stop emulation before the selected
service effect instead of modeling an exception or rollback. No actual Unity,
Windows, game process, scheduler or renderer service is accessed.

```powershell
$env:PYTHONPATH='B:/CodexTools/DemonBluffReverseEngineering/python-emulation'
python reverse_engineering/scripts/audit_character_refresh_view.py GAME_ROOT DUMPER_ROOT --output reverse_engineering/reports/f530404b0f3f_807de4a83df4_character_refresh_view.json
```

Independent private and repository reports must match as JSON values. No copied
native bytes or decompiled bodies enter the repository. This standalone report
does not broaden the current bounded Rust view projection or the existing
initializer/scheduler composition; their explicit contracts remain unchanged.

The separate `bluff::character_refresh_view` replay now accepts versioned
`character_refresh_view_native_v1` contexts with independently verified native
bindings, Unity liveness, inert services/callbacks and normal completion. It
retains the complete initializer Actor, physical UI/Transform records, exact
Vector3 bits and ordered API/store/barrier events. Fresh allocation identities
and both transform-query outputs are supplied explicitly. The consumed fresh
creation plan is removed so a later refresh reuses the retained live object.

Seven focused tests compare 1,639 supported normal native fixtures and three
retained sequences, plus API arguments/order, physical aliases, field retention
and guarded capacity/provenance rejection. References naming the same physical
object must agree on liveness. Null dependencies, callback mutations and service
failures remain outside the replay. It does not feed the live solver, existing
view projection or scheduler and does not infer real Unity lifetime/rendering.
