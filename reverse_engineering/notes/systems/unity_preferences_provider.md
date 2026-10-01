# Native engine preference provider

The preference provider at `0x7E58B0` now executes offline with native path
construction, both handle acquisitions, cache comparison/copy, invalidation and
retry. The audit passes 163 standalone cases, 14 joins into the actual preference
entries and 152 controlled service stops, executing 1,760 distinct addresses.

It compares two supplied configuration strings at offsets `0xE8` and `0xC0`
against cached native strings. A difference closes both nonnull old handles,
clears both records, constructs the new path, acquires both handles and copies
both configuration strings to the cache. The path uses the `0xC0` string first,
then a backslash and the `0xE8` string when the latter is nonempty. Those offsets
are exact native inputs; their runtime initialization remains open.

The cached-token predicate chooses either `Software\` or
`Software\AppDataLow\Software\` as prefix. This report supplies the predicate's
cache DWORD; its complete cold discovery has a separate
[token audit](unity_preferences_token.md). Path bytes enter the actual native
UTF-16 conversion orchestration and reserve helpers. `MultiByteToWideChar` is a
supplied Windows service with explicit UTF-8 input lengths. The registry API's
UTF-16 C-string parameter truncates paths at embedded NULs.

The write record uses `RegCreateKeyW`; the read record uses `RegOpenKeyExW` with
access mask `0x20019`. Both use the signed predefined key `0xFFFFFFFF80000001`.
An empty conversion result or nonzero API status sets that record's blocked
byte. A successful size conversion followed by a failed output conversion can
still reach the registry service with an empty path; fixtures supply the API
result explicitly. No real API success is inferred from that path.

Mode is consumed as a byte: zero returns the read record and any nonzero byte
returns the write record. The selected handle is probed with a null value name
and null output pointers. Only low-DWORD status `0x3FA` triggers cache invalidation
and recursive acquisition. Other authored errors leave the records unchanged.
Repeated `0x3FA` responses exercise multiple retries and normal stack restoration.
Close failures do not prevent native cache clearing.

Matching configuration skips acquisition and preserves both handles/blocked
flags. A supplied empty configuration with matching empty cache and zero handles
only probes the zero handle; it does not initialize one. This is a native state
boundary, not a claim about the game's real bootstrap state.

The entry joins use distinct authored read/write handles, actual native provider
calls, the actual setter/getter/hash/formatting bodies and blocked acquisition
outcomes. The subclass initialization hook runs after per-call allocation reset,
so configuration buffers are retained. Checks preserve both native string headers
and pointed-to payloads, managed input storage, all eight nonvolatile registers
and the stack. Short/long provider and composed-entry baselines stop at every
service occurrence with exact event-prefix/snapshot equality.

Registry and UTF-8 conversion APIs, runtime exports, string assignment/copy,
memory primitives and allocation/ownership remain explicit services. Native
cache-copy callers execute but supplied string assignment does not establish
all underlying storage-reuse policies. Allocator-manager flag zero is exercised;
other modes, native exception unwinding, actual OS access and runtime configuration
loading remain unclaimed. No Assembly-CSharp classification is added.

```powershell
python reverse_engineering/scripts/audit_unity_preferences_provider.py GAME_ROOT --output REPORT
```

Report: `f530404b0f3f_807de4a83df4_unity_preferences_provider.json`. Private Unicorn
dependencies and the pinned installed binary are required. `Machine.run_entry`
also exposes the composed fixture to the values-only save/load adapter.
