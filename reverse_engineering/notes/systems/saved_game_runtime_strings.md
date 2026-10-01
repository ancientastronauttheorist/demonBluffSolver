# Native runtime string construction in saved-game storage

A separate values-only audit replaces the preference getter's supplied
`il2cpp_string_new_len` service with the pinned GameAssembly export and its
actual UTF-8 validation/conversion and managed UTF-16 construction bodies.
The exact backend byte slice crosses into a separate runtime emulator. Only
its returned text crosses back into the preference and save-caller emulators;
object identities, class pointers and allocation ownership remain local.

The audit covers 24 direct native runtime cases, 48 native preference getters,
20 SavedGameData.Load cases, two Save/Load round trips and ten controlled stops.
Its execution address counts include supplied service entries. Input is bounded
to 2,048 backend bytes before reading the explicit-length runtime slice. Native
storage and configuration input retention, normal stack/nonvolatile-register
retention and actual managed-string header/length/terminator checks remain in
the underlying audits. The save caller retains its 120-second/100,000-instruction
outer budget; nested emulators retain their original bounds.

Malformed UTF-8 makes native explicit-length construction return its cached empty
string. It discards an otherwise valid prefix instead of producing replacement
characters or a partially decoded managed string. Fixtures include stray
continuation bytes, overlong encodings, truncated multibyte sequences, encoded
surrogates and values beyond the Unicode maximum. A stored binary JSON document
followed by malformed UTF-8 therefore becomes an empty preference value.
SavedGameData.Load takes its actual empty-value branch, allocates/runs native
SavedGameInfo construction and creates distinct empty Lists. The generic FromJson
path and engine JSON reader do not run in those cases; this outcome occurs before
JSON parsing.

The three byte boundaries differ. Direct `il2cpp_string_new_len` preserves an
embedded NUL and validates following bytes. The native binary preference getter
first applies its C-string boundary, so `a NUL invalid-byte` reaches the runtime
constructor as just `a`. Valid JSON followed by NUL and invalid bytes similarly
loads its valid prefix. For REG_SZ, native storage rejects any high-bit byte in
the returned byte count, including bytes after NUL, and sends the default value
to the runtime constructor instead. The report records the exact constructor
input at every boundary.

Both Save/Load round trips retain native JSON writing, registry name formatting,
provider acquisition, binary writes, queries and JSON reading. Null List string
elements still serialize as empty strings. Windows handle/query/write and UTF-8
conversion API outcomes are authored services in an isolated registry dictionary.
The class/empty-string globals, GC allocation, CRT allocation/free/copy, remaining
managed runtime services and non-generic JSON gateways remain supplied.

Two stops occur at the new runtime adapter boundary. Native runtime service stops
check exact nested event prefixes and snapshots for short and 2,048-byte inputs.
A GC-allocation stop propagates through the native getter adapter into Load while
retaining the earlier save values and avoiding JSON parsing. These are controlled
emulator stops, not native exception unwinding. Repeated snapshots are checked in
memory and omitted from the report.

No live process or actual OS registry is accessed. Arbitrary pointers, oversized
inputs, actual GC, runtime class discovery, object identity across emulators and
native exception unwinding are outside the fixture scope.

```powershell
python reverse_engineering/scripts/audit_saved_game_runtime_strings.py GAME_ROOT DUMPER_ROOT --output REPORT
```

Report: `f530404b0f3f_807de4a83df4_saved_game_runtime_strings.json`.
