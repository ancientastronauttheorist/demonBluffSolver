# Exported IL2CPP string construction

The pinned GameAssembly exports now execute offline through their actual native
UTF-8 validation/conversion, temporary wide-string construction and managed
UTF-16 constructor. The audit passes 720 cases and seven exact controlled-stop
prefixes, executing 548 distinct instruction/service addresses. No
Assembly-CSharp method classification is added.

The PE exports `il2cpp_string_new` and `il2cpp_string_new_wrapper` share RVA
`0x2821F0`, which jumps to `0x29CCB0`. `il2cpp_string_new_len` at `0x282200`
jumps to `0x29CD50`; `il2cpp_string_new_utf16` at `0x282210` jumps to
`0x29CE80`. Exact export names, wrapper targets, unwind families and 29 native
instruction/global assertions are checked against the installed binary.

Both UTF-8 bodies call `0x2435D0`. The wrapper first scans the input to its first
NUL; the lengthful export consumes its explicit low-DWORD byte count. Native
validation at `0x242790` checks the entire selected byte range before conversion.
An invalid sequence returns an empty temporary wide string, so construction
returns the authored cached managed empty string. A valid earlier prefix is
discarded; malformed sequences do not produce replacement characters. Invalid
bytes after a NUL are outside the wrapper's selected range, while the lengthful
export still validates them.

Valid UTF-8 converts to UTF-16, including supplementary surrogate pairs. The
lengthful export preserves embedded NUL characters. BOM and noncharacter
fixtures are retained. The matrix executes every single-byte input, boundary
Unicode values, overlong encodings, isolated continuation bytes, truncated
sequences, bad continuation bytes, encoded surrogates and values above U+10FFFF.
Malformed fixtures also have valid ASCII prefixes/suffixes. Python's strict
UTF-8 decoder supplies the independent expected outcome; no Python conversion
service substitutes for the native validator or converter.

Native `std::wstring` reserve and append bodies execute, including the seven-unit
inline boundary and large temporary allocation alignment. The reserve initially
uses the selected UTF-8 byte count, even when the final UTF-16 unit count is
smaller. Native allocation callers retain the original pointer before an aligned
buffer and recover it during cleanup. CRT allocation, copy and free remain
supplied services; the report records their calls rather than asserting actual
allocator ownership or operating-system release.

The UTF-16 constructor uses the actual unit count at object offset `0x10` and
copies units to offset `0x14`, matching the pinned `System.String` declarations.
For nonempty input it requests `unit_count * 2 + 0x1A` bytes, writes the final
UTF-16 NUL, and calls the native object-header helper `0x2BFD50`. That helper
sets the supplied class pointer, clears the monitor and increments the actual
native allocation counter. Its underlying GC allocation remains a service.
Zero units use the authored cached empty object without allocation. Direct
UTF-16 fixtures preserve NULs and unpaired surrogates; this constructor does not
perform the UTF-8 validator's checks.

The class pointer global `0x289E640`, cached-empty global `0x289DF48`, and
profiler configuration are authored runtime inputs. Profiler bit seven is tested
both clear and set with an empty callback table; the actual native callback-list
traversal executes. Bootstrap, actual GC, profiler callbacks, invalid pointers,
negative/overflowing lengths, allocation failure unwinding and native exceptions
remain outside this report. Every normal return preserves input bytes, the stack
and all eight general nonvolatile registers. Both short and aligned long
baselines stop at every supplied-service occurrence with exact prefix/snapshot
equality.

```powershell
$env:PYTHONPATH='B:/CodexTools/DemonBluffReverseEngineering/python-emulation'
python reverse_engineering/scripts/audit_il2cpp_string_creation.py GAME_ROOT --output REPORT
```

Report: `f530404b0f3f_807de4a83df4_il2cpp_string_creation.json`. This remains a
separate runtime audit: existing JSON/preference fixtures still supply their
managed-string exports until an explicit joined audit adopts these constructors.
A separate process produced an identical private peer report. Python compilation
and repository diff checks passed. Native bodies and bytes remain private.

`Machine(game_root).run_new_len(raw_bytes, options=None)` is a values-only
adapter for inputs of at most 2,048 bytes, the largest audited size. It returns
`text` decoded from actual native UTF-16, `utf16_unit_count`, `managed_storage`
(cached-empty selection, allocation byte count and header/terminator checks),
and `native_trace`. A controlled service stop returns no text/storage outcome.
Object identities belong to that emulator and are not transferable to another
runtime. Length overflow, null source pointers and inputs above the adapter
bound are deliberately excluded. Embedded NULs and invalid UTF-8 contents are
accepted within the bound; invalid UTF-8 returns the native empty-string result.
