# Saved-game callers joined to native JSON and preference storage

The offline audit composes the actual SavedGameData caller bodies, native
SavedGameInfo constructor and mutation methods, public PlayerPrefs wrappers,
generic FromJson wrapper, native engine JSON field pipeline and the native
preference entry/provider/getter/setter path. It transfers the three public
SavedGameInfo values between independent emulators. The registry service is an
isolated authored dictionary keyed by the value names requested by native code;
no Windows registry or live game is accessed.

It passes 43 cases and 22 exact outer service stops, reaching 410 caller and
1,923 storage execution addresses, including supplied service entries.
Independent report generation verifies
the deterministic fixture results. Repeated event snapshots are checked in
memory and omitted from the report; event arguments, final storage/caller
states and authored API outcomes remain recorded.

Save and ResetTutorials round trips cover null, empty, Unicode, embedded-NUL and
500-character keys, warm/cold public wrapper caches, duplicate/null List values
and native allocation paths. Additional fixtures run AddTutorial, AddCharacter
and both clearing methods before serialization. Native caller Lists preserve
null string elements; the actual JSON writer serializes them as empty strings,
which reload as empty strings. JSON writing also truncates string values at NUL,
while the storage entry converts the complete managed preference key. The
registry's ANSI name boundary can truncate that key before its hash suffix.
The fixture checks each stage independently instead of assuming a round trip
preserves the caller's original values.

Load executes its first native preference read and, when nonempty, its second
read and generic FromJson path. A supplied callback replaces the current save
between reads; native code then uses the replacement key for the second read.
Save's supplied serialization callback similarly replaces the save reference:
the JSON contains the earlier object's values, while native code persists under
the replacement object's key. These callbacks expose existing native sequencing;
the audit does not claim that the live runtime normally performs such changes.

Authored legacy entries exercise native hashed-name failure followed by raw-key
fallback. The supplied missing-value service retains input type and data-buffer
capacity; this is an authored failure-output policy. Native retries consume
those fields, so zeroing the capacity would instead exercise a failed fallback.
UTF-8 binary registry values load, whereas REG_SZ containing non-ASCII
bytes returns the default and causes native SavedGameInfo construction. Missing,
empty and failed read-handle acquisition also construct a fresh default object.
Both cold provider acquisition and matched cached fields/handles execute. Native
provider path construction requests `Software\Studio\Game`; configuration
strings and the cached token value remain authored inputs.

Authored RegSetValueExA failures and failed write-handle acquisition propagate
through the actual native setter return and public PlayerPrefs exception branch.
ResetTutorials has already cleared its tutorial List when that branch is reached.
Exception allocation/construction/throw remain services; the final throw gateway
is a controlled emulator stop, not native exception unwinding. Outer service
stops check exact event prefixes and snapshots, including previously completed
JSON and storage adapters.

All normal caller and storage returns check the stack and eight nonvolatile
registers. Native managed-string inputs and native configuration storage are
retained. The outer caller uses a 120-second wall-time limit and a 100,000-
instruction limit because nested emulators execute synchronously inside its
callbacks; nested engines retain their original limits.

This is a value composition audit, not a single runtime process reconstruction.
Object identities, private List versions and runtime constructor semantics do
not transfer between emulators. Internal-call resolution, non-generic JSON
gateways, managed metadata/object/string/GC services, allocator ownership and
memory primitives remain supplied. Windows registry handle/query/write and
UTF-8 conversion outcomes are authored API services. Security-token discovery
and real configuration initialization are outside this join's scope.

```powershell
python reverse_engineering/scripts/audit_saved_game_storage_join.py GAME_ROOT DUMPER_ROOT --output REPORT
```

Report: `f530404b0f3f_807de4a83df4_saved_game_storage_join.json`.
