# JSON metadata adapter and descriptor traversal

Pinned build `f530404b0f3f_807de4a83df4` and the existing UnityPlayer fingerprint.
The native audit executes `0xA8E030`, reference-context construction at
`0x784BC0`, and descriptor traversal at `0x79BB60`. It contains 152 fixtures,
including cache matrices, cold initialization, controlled descriptor changes and
gateway failure prefixes. Nineteen exact instruction assertions cover the three
complete entry ranges, the code after the adapter's first return, traversal
writes and the independently resolved reference-store export. Successful returns
preserve the stack and eight nonvolatile integer registers.

The supplied auxiliary slot is checked first. If it contains zero and the class
is present, the adapter requests a cache lookup and publishes the returned cache
pointer. It then searches that cache's records in order. A match requires both
direction byte 9 and a zero second byte; direction `0x109` does not match.
The first exact matching record wins, including an empty first record followed
by another matching record. Missing records request metadata construction for
direction 9. Cache discovery and metadata construction remain supplied services;
these fixtures do not establish which fields a real class includes.

The adapter captures the selected descriptor pointer and count before traversal.
The native reference context retains the exact target object and class. Its
reference-store callback is independently bound to
`il2cpp_gc_wbarrier_set_field`; the fixture store behavior is an explicit service.
Cold runtime initialization and cleanup registration occur once in either the
cache-hit constructor route or the metadata-build route. Failure after the
initializer returns preserves its supplied runtime publication.

Each descriptor occupies `0x80` bytes. The loop loads the current descriptor,
advances its cursor and updates the remaining count before invoking the function
pointer at descriptor `+8` with payload `+0x10` and the native context. A stopped
field-service call therefore retains that cursor advance and all earlier field
effects. The next function pointer and payload are reread from descriptor memory;
a controlled earlier callback changes the subsequent dispatch. Changing the
cache's stored count during traversal does not change the already captured end.
Setting the parser's field-error flag does not terminate this descriptor loop.

The individual field callbacks write authored integers solely to make ordering
and retention observable. They do not implement native member lookup, field
offset discovery, numeric conversion, strings, arrays, nested construction or
managed references. Reference-scope and metadata-storage cleanup are inert
gateways. The fixtures keep the managed-reference processing flag clear; its
registry/finalization branches remain outside this boundary. Runtime discovery
and exception unwinding are also unclaimed.

The input tree is produced by the [actual native parser](unity_json_parser.md),
not a supplied parser outcome. Descriptor selection and traversal execute as
native code over that tree, but supplied field bodies do not prove a semantic
JSON-to-managed-object copy. That distinction remains necessary before replacing
the engine copy boundary in ascension setup.

Reproduce with private Unicorn 2.1.4 dependencies:

```powershell
python reverse_engineering/scripts/audit_unity_json_fields.py GAME_ROOT --output REPORT
```

Report: `f530404b0f3f_807de4a83df4_unity_json_fields.json`. No native bytes or
decompiled bodies are retained. This engine audit adds no Assembly-CSharp
classification. Actual field processors, metadata building, managed-reference
resolution and the writer-side adapter remain the next boundaries.
