# Native scalar and string JSON arrays

The pinned engine audit joins native one-dimensional array metadata construction,
registry selection, collection processors, element conversion, JSON writing and
rendering to native parsing and array reload. It contains 72 numeric save/load
cases, 44 integer-array reader cases, two string-array round trips, two shared
array cases and 152 controlled numeric-array service stops. Fifty-eight exact
instruction/export assertions cover eleven complete chained-unwind families and
three pointer-backed leaf wrappers. Required native entries must execute.

Native descriptor construction at `0x77F870` resolves the element class through
`0x784860`, selects its registry element handler through `0x77F620`, and retains
the actual collection processor. The reader registry publishes `0x78F420`; the
writer publishes `0x7906A0`. These processors execute rather than being replaced
with collection services. For the four-byte integer family, the pointer-backed
wrapper at `0xA91810` dispatches to `0xA949E0` and native vector reading at
`0xA94AC0`. The writer wrapper at `0xA92E80` dispatches to `0xA98E50`. Actual
element conversion creates the JSON tree and writes loaded array data.

The twelve numeric families cover widths 1, 2, 4 and 8 bytes, signed/unsigned
boundaries, canonical boolean values, noncanonical flagged bytes, negative zero,
smallest floating values, NaN and infinities. Native normalization matches the
scalar evidence: flagged `FF` reloads as `01`, negative-zero signs are lost, and
the smallest eight-byte floating value underflows on reload. The separate ordinary
byte family preserves `FF`. Compact and pretty output are both tested.

Reader behavior over a supplied four-byte integer array:

- Missing or differently cased fields preserve the pointer and elements.
- A found null or any tested non-array JSON type becomes an empty array.
- An existing array with the requested length is reused; a different length
  causes a runtime allocation request and later field publication. A null old
  field also receives a newly requested array, including length zero.
- Duplicate field names use the first value. Within the array, tested boolean,
  null, object and array elements convert to zero; the string `12` becomes 12,
  and `1.5` truncates to 1.
- Native field writes leave authored array header/trailing guards untouched.

String-array handlers execute their own native reader/writer bodies. Valid
Unicode survives. Embedded NUL truncates, and null elements reload as empty
strings. These retain the explicit runtime string services from the string audit.

Two object fields initially pointing to the same nonempty array save as separate
JSON array values. Reload issues separate allocation requests and writes distinct
fixture arrays with equal contents. This demonstrates native traversal and
allocation requests, with object creation still a supplied runtime service; it
does not claim arbitrary graph cloning or ownership preservation.

Failure fixtures independently stop every observed service occurrence during a
new-array read, a same-length reused-array read and writing. Snapshots retain both
published array data and any allocated arrays not yet attached to the object.
They verify when native writes survive later cleanup stops. Successful returns
preserve the stack and all eight nonvolatile integer registers. Writing preserves
the whole input object and authored input-array storage.

The independently bound runtime exports are `il2cpp_class_get_type`,
`il2cpp_array_length`, `il2cpp_array_new` and `il2cpp_class_array_element_size`.
Metadata, runtime array allocation/length/size, GC stores, cache/classifiers and
allocator/vector cleanup remain explicit fixture services. Source-table labels
do not assert real managed type discovery. Lists, nested arrays, compound/reference
element graphs, callbacks and actual runtime allocation remain open.

```powershell
python reverse_engineering/scripts/audit_unity_json_arrays.py GAME_ROOT --output REPORT
```

Report: `f530404b0f3f_807de4a83df4_unity_json_arrays.json`. Private Unicorn
2.1.4 dependencies are required. No native bytes or decompiled bodies are retained;
this engine audit adds no Assembly-CSharp classification.
