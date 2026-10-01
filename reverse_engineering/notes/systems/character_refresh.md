# Character.RefreshCharacter: complete supplied caller

Pinned build `f530404b0f3f_807de4a83df4`. The standalone executable audit covers
`Character$$RefreshCharacter` at RVA `0x367970`, extending the initializer
audit's Hidden-only joined slice. The decoded body ends at exclusive
`0x367B5A`; its normal return is `0x367B4E`. Neighboring managed entry
`Character$$RefreshView` starts at `0x367B60`. Trailing alignment traps are not
treated as method instructions.

The corpus has 2,094 ordinary and null-dependency fixtures, ten explicit
callback-mutation fixtures, and twenty controlled-stop prefixes. It executes
112 of 118 decoded instructions. All eight Windows x64 nonvolatile integer
registers and stack restoration are checked on normal returns. GameAssembly,
Dumper metadata, exact consumed field declarations, enum values, 21 instruction
relationships, and three shared metadata-slot relationships are pinned.

## Ordered writes and reads

First, the native body loads `Character.pickeds` once and deactivates each
referenced GameObject in index order. A null array fails immediately; a null
element fails after the earlier elements were deactivated. Zero elements skip
that loop. The array and its elements are not replaced or cleared.

After those UI calls, native code loads `Gameplay.PrevState` at static field
offset `0x2C`. Only Night (`20`) enters the reset path. Current `GameplayState`
at offset `0x28` is irrelevant: fixtures independently vary both fields.

The first data selection uses `dataRef` when actor state is Dead (`20`) or
Revealed (`30`), or when the separate `revealed` Boolean is nonzero. Otherwise
it asks Unity equality whether `bluff` is Unity-null. An absent or authored
destroyed pointer selects real data; a live bluff selects bluff data. After the
equality callback, the body reloads the selected raw pointer rather than using
the object supplied to that callback. A nonnull authored destroyed bluff pointer
remains stored even when real data was selected.

Only selected data with `abilityUsage == ResetAfterNight` (`10`) writes
`pickableUses = 1`. Other usage values preserve the previous count. The write
occurs before checking actor state: Hidden (`5`) and Dead (`20`) then return
without pickable activation. Other states continue.

The body now performs a second data selection using the same rules, then checks
that selected data's `picking` byte. A nonzero byte activates the actor's
`pickable` GameObject. Null selected data or an eligible null `pickable` fails.
Neither the non-reset path nor the non-picking path deactivates `pickable`; an
already-active authored control can remain active. These two UI surfaces must
not be conflated: every reached `pickeds` element is hidden, while `pickable`
is only conditionally activated.

The only native actor-field write is the four-byte use count. The audit fills
other actor and source-data storage with nonzero sentinels, checks all remaining
actor bytes, all three source-data objects, and the picked-array bytes on both
returns and stops. Explicit service-callback mutations are separately allowed
and named; no native status, resistance, role, copied-role, speech, continuation,
or Start-guard mutation is inferred.

## Repeated selection and authored callback effects

Unity equality is an explicit authored service. Fixtures distinguish absent,
live, and destroyed pointers without claiming to reconstruct engine object
lifetime. Its Boolean result is consumed through `AL`: services poison upper
return-register bits to verify that width.

Ten fixtures permit named actor mutations at the first or second equality
callback. These establish native reload order rather than ordinary game
behavior. In particular:

- A first-callback `revealed` change can reset uses from bluff data, then test
  picking on real data in the second selection.
- Replacing `bluff` during equality makes the native body consume the new raw
  pointer after the callback. Replacing it with zero causes a null failure.
- A second-callback null replacement fails after the reset has already written
  one; there is no rollback.
- Changing state to Dead during the second callback can still activate the
  control: the Hidden/Dead suppression checks happened before that callback.
  Changing state to Dead during the first callback reaches those later checks
  and suppresses activation.

Mutations occur solely at declared service callbacks. They do not claim that
Unity's real equality implementation changes these actor fields.

## Failure prefixes and instruction coverage

Cold fixtures execute metadata resolution and class initialization services.
The two inlined data selectors share the same metadata guard and Unity Object
TypeInfo slot. Once the first selector initializes them, the second selector's
redundant initialization branch is skipped under the authored runtime contract.

Six decoded addresses remain unexecuted:

| RVA | Reason |
| --- | --- |
| `0x367ABB`, `0x367AC2`, `0x367AC7` | Second selector's redundant metadata resolution branch |
| `0x367AFB` | Second selector's already-warmed Unity Object class initialization |
| `0x367B54` | Trap after the controlled null-failure helper |
| `0x367B55` | Bounds helper branch; frozen array length makes it unreachable |

The report explicitly lists these addresses. Concurrent array-length changes,
runtime cache invalidation inside callbacks, and continuation after exception
helpers remain outside the executed corpus. Negative authored array-length
DWORDs demonstrate the signed initial loop comparison; they are stress inputs,
not valid managed-array shapes.

Twenty injected stops reproduce exact event prefixes and actor/UI snapshots.
They include metadata, class initialization, Unity equality and UI calls.
Null-dependency cases additionally retain the native partial state. The
external services stop emulation rather than unwinding a managed exception.

## Composition boundary and reproduction

This report resolves the full standalone caller behavior under the named
services. It does not automatically broaden the existing bounded Rust
initializer replay, whose inert-callback Hidden scope remains unchanged.
Joining arbitrary post-initializer actor states, engine scheduler admission,
or view presentation requires its own explicit composition evidence.

`bluff::character_refresh` adds the independent versioned
`character_refresh_native_v1` replay. Its context retains the complete
initializer Actor, real/bluff data identities, separate raw-bluff liveness,
picked-array identity and ordered occurrences, and unique physical UI records.
Picked occurrences can repeat or alias `pickable`; the replay updates one
shared UI record in actual call order. Real and bluff references can also name
the same data object. Incompatible actor/data/array/UI type collisions reject.

The replay requires independently verified native runtime/metadata, Unity
liveness, inert callbacks/UI, and normal completion. It rejects missing/null
required records, unsupported provenance, ambiguous duplicate identity records,
unknown schema fields and excessive retained data. It represents both reached
data selections and the ordered UI/use writes, preserving every other Actor
field and unreferenced UI/data records. It does not model injected failures,
mutating service callbacks, or malformed negative managed-array lengths.

Six focused tests compare 1,975 supported native normal fixtures (including
three cold-runtime baselines), complete Actor retention, repeated/aliased UI
occurrences, destroyed bluff identity, stable repeated selections, reset before
state suppression, rejected provenance/collisions and capacity boundaries.
This standalone API has no live solver or existing initializer/scheduler caller.

```powershell
$env:PYTHONPATH='B:/CodexTools/DemonBluffReverseEngineering/python-emulation'
python reverse_engineering/scripts/audit_character_refresh.py GAME_ROOT DUMPER_ROOT --output reverse_engineering/reports/f530404b0f3f_807de4a83df4_character_refresh.json
```

The independent private peer process must match the repository report's JSON
values. Reports contain authored fixture data and evidence summaries; copied
native bytes and decompiled bodies remain outside the repository.
