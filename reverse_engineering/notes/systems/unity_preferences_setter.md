# Engine preference registry setter

The shipped TrySetSetString entry now executes through its native backend helper
at `0x7E6030`, key hash at `0x7E5F80`, format orchestration at `0x4334C0`, argument
dispatch and pointer-backed formatters. The audit passes 132 cases and 41
controlled service stops, executing 1,147 distinct addresses.

Key hashing starts at 5381. Each nonzero byte updates a wrapping 32-bit value as
`hash * 33 XOR signed_byte`. The byte is sign-extended before the XOR, so UTF-8
bytes above 127 differ from an unsigned-byte implementation. Hashing stops at
the first NUL. The exact pinned format is `{0}_h{1}`, with the hash formatted as
an unsigned decimal number. `Tutorials` becomes `Tutorials_h3717727834`.

The native formatter consumes the full explicit key slice. Its descriptor binds
two actual leaf entries: the string-slice formatter at `0x4308B0`, and the integer
formatter wrapper at `0x430850`. Neither is replaced by a Python formatting
service. The first forwards to native string append; the second calls native
integer formatting. Chained unwind families for append, reserve and format
cleanup are included, rather than treating their first fragments as whole bodies.

The resulting name is passed to `RegSetValueExA` through the verified
`ADVAPI32.dll` import. This is a C-string interface: a key containing a NUL has
its requested registry name truncated before the suffix. The value is passed
as type three (`REG_BINARY`) with its explicit converted length plus one final
NUL, including any embedded NULs. The API's low 32-bit status determines the
native helper's byte-sized Boolean; the outer entry normalizes that result.
Poisoned upper return bits do not alter success or failure.

Fixtures cover null/empty inputs, Unicode, malformed UTF-16, embedded NULs,
inline/heap boundaries and long keys/values. Blocked backends bypass the setter.
Five authored Windows statuses exercise success and failure, and every service
occurrence in short and long baselines has exact stopped-prefix/snapshot checks.
Normal returns preserve input storage, the stack and all eight nonvolatile
registers.

Windows registry calls are supplied services; no actual registry access occurs.
Backend acquisition, runtime exports, allocation/ownership, memory copy and
string assign/copy also remain explicit services. Native hashing, formatting,
setter logic and cleanup execute; native exception unwinding and other allocator
modes remain outside this contract. No managed-method classification is added.

```powershell
python reverse_engineering/scripts/audit_unity_preferences_setter.py GAME_ROOT --output REPORT
```

Report: `f530404b0f3f_807de4a83df4_unity_preferences_setter.json`. Requires the
private Unicorn dependencies and pinned installed UnityPlayer binary. Native
bodies and bytes remain outside the repository.
