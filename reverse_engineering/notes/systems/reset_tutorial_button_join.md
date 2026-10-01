# ResetTutorialButton caller joined to save storage

The previously unclassified `ResetTutorialButton.ResetTutorials` declaration
(`tdi5726.m0000`, RVA `0x3a6600`) now executes through its actual persistence
callee. The audit pins its exact Dumper signature, complete `0x59`-byte native
body, 13 instruction/slot relationships and relevant class field declarations.
It passes 26 cases and ten exact outer service stops. Normal caller returns
check the stack and all eight nonvolatile integer registers.

The button follows this exact chain:

1. `ProjectContext_TypeInfo`, resolved from slot `0x271f268`.
2. Its supplied runtime static-fields block and `ProjectContext.Instance`.
3. `ProjectContext.gameData`, instance field `+0x20`.
4. `GameData.saveData`, instance field `+0x30`.
5. A native tailjump to `SavedGameData.ResetTutorials`, RVA `0x3eaa10`.

There is no additional context-class initialization call in this body. Cold
metadata initialization precedes the native flag write at `0x288c3b9`. The
receiver is not dereferenced: a null button receiver behaves identically when
the supplied ProjectContext chain is valid. Each managed link has an explicit
null branch to the runtime null-reference gateway. The class and static-fields
block themselves are authored valid runtime inputs, not arbitrary-pointer cases.

The native reset increments the tutorial List version, sets its logical count
to zero and optionally clears its backing array before serializing. Even an
empty List increments its version; `0xffffffff` wraps to zero. It retains the
unlocked-character List and its version. The button join executes actual JSON
writing, public PlayerPrefs SetString, provider acquisition, key formatting and
the registry setter. Subsequent value-level Load fixtures also execute native
queries, native runtime string construction, the generic FromJson wrapper and
engine JSON reading, confirming the reset values in storage.

Null ProjectContext instance, gameData or saveData fails before List mutation,
JSON or storage. Null SavedGameInfo or tutorial List reaches the persistence
callee's failure boundary. A registry-write failure or failed provider write
acquisition occurs after clearing the tutorial List and propagates through the
public preference exception branch; there is no rollback in these caller bodies.
Cold and warm baselines stop at every reached outer service, retaining the exact
attempted event prefix and snapshot. ProjectContext/static-instance and GameData
object bytes remain unchanged even on controlled stops.

This callback does not invoke `TutorialsController.ResetTutorials`. The audit
does not establish Unity Button event routing, UI-note reset behavior or the
producer of ProjectContext.Instance. Singleton objects, metadata services,
runtime/allocator/collection/JSON gateway services and Windows API outcomes are
authored. Values cross between independent emulators; object identities and
allocation ownership do not. Repeated snapshots are verified in memory and
omitted from the compact report. Execution address counts include supplied
service entries. Native exception unwinding, actual OS registry access and live
process interaction remain outside scope.

```powershell
python reverse_engineering/scripts/audit_reset_tutorial_button_join.py GAME_ROOT DUMPER_ROOT --output REPORT
```

Report: `f530404b0f3f_807de4a83df4_reset_tutorial_button_join.json`.
