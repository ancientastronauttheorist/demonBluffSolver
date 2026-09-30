# Native JSON reader registry

The pinned UnityPlayer reader registry at `0xA8F090` now executes in 32
fixtures over the actual provider initializer at `0x783F40`. Twenty-six exact
instruction assertions bind those entries, reset ordering, the first handler
pointer and the caller's reader/writer publication slots. The caller at
`0xA8CC80` publishes the reader provider at registry slot `+0x48` and the writer
provider at `+0x40`. Caller binding is decoded evidence; these fixtures execute
the initializer and reader construction, not the complete two-provider caller.

The native constructor creates 33 ordered, 40-byte records. An optional extension
adds a 34th record after querying its native interface slot. Each record carries
a class token, three handler pointers and metadata. Runtime discovery supplies
explicit, distinct class tokens labeled by source table and byte offset. These
labels do not assert public type names. The report records the actual native
handler RVAs for every row, including the repeated handlers for distinct tokens.

Owned old storage is released before the provider's pointer, count and capacity
reset. Borrowed storage skips that release. Missing runtime state calls the
supplied initializer and registers its initialize/cleanup callbacks before
reading the rows. Storage reservation is an explicit always-success service;
growth sizes 1, 4 and 40 produce the same ordered records. Controlled service
stops retain the exact preceding native writes and earlier rows. Successful
returns preserve the stack and all eight nonvolatile integer registers.

The constructor writes four metadata bytes and one feature byte in each row,
while three upper bytes retain earlier stack contents. Explicit zero and `0xA5`
stack fixtures retain those bytes in the report; they must not be mistaken for
independently initialized metadata. No native allocator-failure or exception
unwind claim is made.

This audit recovers the registry that selects field processors. It does not
execute their JSON conversion, discover real managed classes, build descriptors,
resolve managed references or serialize arbitrary objects. The existing
[metadata adapter audit](unity_json_fields.md) still supplies its field bodies.

```powershell
python reverse_engineering/scripts/audit_unity_json_registry.py GAME_ROOT --output REPORT
```

Report: `f530404b0f3f_807de4a83df4_unity_json_registry.json`. Private Unicorn
2.1.4 dependencies are required. No engine bytes or decompiled bodies are retained,
and this engine audit adds no Assembly-CSharp classification.
