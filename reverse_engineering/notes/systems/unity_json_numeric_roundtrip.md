# Native numeric serialization round trips

The pinned engine audit now executes the writer registry, the ToJson serializer
caller at `0xAACA50`, writer construction at `0x10964E0`, native metadata building,
the writer adapter at `0xA8E480`, registry-selected numeric processors, actual tree
rendering and normal writer destruction. The resulting JSON goes through the
actual native parser, metadata builder and reader application. It contains 110
round-trip fixtures and 32 controlled writer-service stops. Thirty-five exact
instruction/export assertions decode seven complete unwind families, including
the serializer's cache-hit branch after its first return. The runtime object-class
service is independently bound to `il2cpp_object_get_class`.

The inventory contains twelve distinct source-table tokens with numeric handler
families of widths 1, 2, 4 and 8 bytes. Tokens and runtime metadata remain supplied;
these labels do not claim real managed type discovery. Every serializer field
descriptor is built by native code. Actual native handlers create the tree and
actual native rendering produces the saved text. No Python JSON writer or scalar
conversion substitutes for either native stage. Compact and pretty output both
reload through the native reader.

Of the 110 fixtures, 102 retain their field bytes exactly. Four observed value
changes occur once in each output mode:

- The flagged byte source at core `+0x130` serializes raw `FF` as `true`, then
  reloads as `01`. Its ordinary canonical zero/one values remain exact. The
  separate core `+0x170` byte source preserves `FF` as numeric 255.
- The four-byte floating source at core `+0x188` loses the negative-zero sign
  when saved as `0.0` and reloaded.
- The eight-byte floating source at core `+0x198` also loses negative zero.
- That eight-byte source saves its smallest positive value as `5e-324`; native
  parsing reloads zero, matching the earlier parser underflow evidence.

The tested large signed/unsigned integers, canonical NaN and infinities retain
their bytes. A mixed inherited object retains its parent/child numeric fields,
floating field and canonical flagged byte in both output modes. The writer leaves
the managed input and all its guard bytes untouched. Successful returns preserve
the stack and all eight nonvolatile integer registers. Controlled metadata/cache/
cleanup stops retain the corresponding native prefix and do not claim exception
unwinding.

The writer's cache call sets its output pointer and class-key pointer after
constructing the writer. It leaves RCX as constructor scratch; the cache entry
homes that register before replacing it with its lock address. The supplied cache
service interprets only its output/class arguments. A decompiler's inferred
constructor-return/cache-context value is not treated as semantic evidence.

Runtime metadata and object-class exports, cache lookup, type classifiers,
allocation, GC reference storage and vector cleanup remain explicit services.
Cache fixtures take the metadata-build route. Reference processing remains
disabled; callback interface tests return false. Strings, collections, aliases,
compound/reference serialization, real class discovery and allocator failures
remain outside this boundary. This proves numeric serializer/reader composition
over supplied inventories, not arbitrary game-state copying.

```powershell
python reverse_engineering/scripts/audit_unity_json_numeric_roundtrip.py GAME_ROOT --output REPORT
```

Report: `f530404b0f3f_807de4a83df4_unity_json_numeric_roundtrip.json`. Private
Unicorn 2.1.4 dependencies are required. No native bytes or decompiled bodies are
retained; this engine audit adds no Assembly-CSharp classification.
