# Native preference wrappers inside save-data callers

The exact PlayerPrefs GetString and SetString wrapper bodies now execute inside
SavedGameData.Load, Save and ResetTutorials. Load also runs the native generic
FromJson wrapper. Complete unwind ranges, metadata/literal slots, exact request
strings and 23 instruction/operand assertions precede 52 cases, 28 controlled
service stops and six native engine JSON joins.

GetString initializes the empty-string literal separately from resolving its
internal-call cache. It forwards the requested key and that empty default by
tail call. Two reads in one Load share the cached pointer; cold/warm literal and
cache states vary independently. Null/empty keys and supplied null/empty/nonempty
results exercise the caller without inferring the platform backend's policy.

SetString resolves and caches a Boolean-returning internal call, invokes it with
the exact key/value, and tests only AL. Fixtures retain nonzero upper RAX bits
on both success and failure, verifying that operand width. Success returns.
Failure initializes PlayerPrefsException metadata, allocates an exception,
initializes the exact message `Could not store preference value`, resolves the
SetString MethodInfo and reaches the throw gateway. Exception construction and
throw behavior remain explicit services. ResetTutorials has already cleared its
List when the preference write fails; the audit preserves that mutation.

The exact requests are:

- `UnityEngine.PlayerPrefs::GetString(System.String,System.String)`
- `UnityEngine.PlayerPrefs::TrySetSetString(System.String,System.String)`

The repeated `Set` in the second request is present in the pinned native literal.
This audit supplies resolver results; it does not assert equality with an engine
registration name or establish fallback lookup behavior.

Controlled stops independently check every service occurrence in three baseline
calls, preserving complete event prefixes and snapshots, including cached
pointers and allocated/constructed exception state. Normal returns retain the
stack and all eight nonvolatile registers. Six value-level engine JSON joins
execute with preference storage still supplied. They do not project object
identities or private List versions between emulators, invoke real preference
storage or model native exception unwinding.

```powershell
python reverse_engineering/scripts/audit_saved_game_preferences.py GAME_ROOT DUMPER_ROOT --output REPORT
```

Report: `f530404b0f3f_807de4a83df4_saved_game_preferences.json`. Private Unicorn
2.1.4 dependencies are required. These framework wrappers add no Assembly-CSharp
classification; native bytes and decompiled bodies remain private.
