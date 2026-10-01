# Engine preference registry getter

The GetString entry now executes its native getter at `0x7E61A0`, query helper
at `0x7E60D0`, key hashing/formatting, temporary buffers and cleanup. The audit
passes 398 cases and 65 controlled service stops, executing 1,385 distinct
addresses. Windows query outcomes remain authored services, with no actual
registry reads.

The getter requests the provider with byte argument zero. A blocked backend
returns the converted default without querying. Otherwise it starts with a
native copy of the default, queries the hashed name for type/size, and retries
the bare key on any nonzero status. Missing or zero-size values keep the default.
The bare and hashed names both use Windows C-string rules.

It allocates `reported_size + 1` bytes and zero-fills the buffer. Counts below
2000 use the native stack-probe path; counts at least 2000 use supplied heap
allocation. Only initial type one or three leads to a data query. The data query
helper independently retries hashed then bare names, so a legacy size result
does not pin the subsequent data read to that same name.

Type three (`REG_BINARY`) requires a successful data query still reporting type
three. Type one (`REG_SZ`) requires success and unchanged type one, then checks
every returned byte for ASCII. A high byte after a NUL still rejects the entire
legacy string. Type/status changes, unsupported types and failed data reads
retain the default. The audit includes growth failures and successful shrinkage;
it does not assert arbitrary Windows race outcomes beyond supplied responses.

Accepted data is scanned to its first NUL and assigned using that byte length.
The extra zero-filled byte handles authored unterminated successful responses.
Embedded NULs therefore truncate stored values at the backend, even though the
outer entry and its managed-string constructor use explicit lengths. Binary
UTF-8 survives; legacy non-ASCII strings fall back to the default. Arbitrary
invalid UTF-8 conversion into managed strings remains outside this audit.

The matrix covers null keys/defaults, embedded NULs, Unicode, long keys, both
accepted types, both lookup routes and the stack/heap boundary. Fourteen special
cases cover blocking, missing values, size/type changes, independent fallback,
unterminated data and non-ASCII after NUL. Every supplied-service occurrence in
two baselines has exact stopped-prefix and snapshot checks. Normal returns
preserve input storage, the stack and all eight nonvolatile registers. API
return fixtures poison upper bits while supplying the low 32-bit status.

Provider acquisition, runtime exports, string assignment/copy, memory primitives
and allocation/ownership remain supplied services. String assignment preserves
the destination allocator tag, including the blocked getter's native `0x49`.
Native exception unwinding and other allocator-manager modes are unclaimed.
No managed-method classification is added; native bytes and bodies stay private.

```powershell
python reverse_engineering/scripts/audit_unity_preferences_getter.py GAME_ROOT --output REPORT
```

Report: `f530404b0f3f_807de4a83df4_unity_preferences_getter.json`. Requires private
Unicorn dependencies and the pinned UnityPlayer binary. Successful-case records
omit repeated event snapshots; the executable assertions check full traces, and
controlled-stop baselines retain their complete snapshots.
