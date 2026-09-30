# Engine JSON parser and parsed-tree rendering

Pinned UnityPlayer SHA `B5D48235E7CC02FF9496FB33A07D5921ADFC4B40DED1BC64C96A7A7C10B4DFB2`.
The executable audit runs actual parse entry `0xAACC80`, constructor `0xAAC870`,
in-place recursive parsing, native tree construction and parsed-tree rendering
through `0x1096690`. It contains 89 natural input cases and 18 separately labelled
controlled error-code cases, executing 4,710 distinct instruction addresses.
Nineteen exact instruction assertions verify entry calls, parser flags, object
root checking, error dispatch and all outer return paths. The error jump table
is interpreted as data after the final parser return, rather than decoded as
instructions. Successful parse and render returns preserve the stack and all
eight nonvolatile integer registers under explicit MXCSR `0x1f80`.

The entry allocates a `0x158` parser object. Its constructor receives `0x4000`,
the literal fourth argument 9 and the in-place selector 1. The native parser error code is read at
`+0x120`; success additionally requires the root tag at `+0xD8` to be object 3.
Array, null, Boolean, number and string roots are rejected with the object-type
diagnostic. Empty and malformed documents produce their specific error strings.
The native tree and parser cleanup occurs before diagnostic assignment or
formatting. Allocation/free services record this order but do not establish
allocator ownership or behavior after allocation failure.

Objects retain ordered member occurrences, including duplicate names. Arrays
retain occurrence order and nested containers. Strings point into the mutated
input buffer, with explicit lengths and recorded offsets: escape decoding and
zero termination are native in-place writes. The report keeps byte strings as
hex so malformed UTF-8 is never silently replaced or rejected by the harness.

The corpus demonstrates byte-preserving acceptance of malformed raw UTF-8,
including overlong encodings and encoded surrogate bytes. An isolated escaped
low surrogate becomes its three encoded bytes; a high surrogate followed by an
invalid partner reports error 9. A valid high/low pair produces the combined
code point. Embedded escaped zero is retained with length one and rendered as
an escape. A literal zero after a complete document terminates the native input;
the following bytes are not parsed. A leading UTF-8 BOM is rejected by this
direct parser boundary. These observations do not recover the earlier managed
string-conversion policy.

Numbers retain native flags and the exact 64-bit payload. The fixtures distinguish
signed/unsigned 32-bit and 64-bit ranges, overflow to double, exponent overflow,
small values and signed zero. `NaN`, `Infinity` and `-Infinity` are accepted and
rendered as those tokens. Integer negative zero renders as zero. Floating negative
zero parses with its sign bit set, renders as `0.0`, and loses that sign when
reparsed. This is the only semantic tree change in the 56 compact round trips
after removing input-storage offsets. Large unsigned overflow rounds to the observed
double bits. The default parser's tiny exponent path can underflow to zero even
when a different general-purpose JSON parser retains a subnormal. The report
records exact results rather than substituting Python's numeric conversion.

Rendering consumes the actual parsed native tree. Compact and pretty output
bytes are retained and compact output is reparsed. This proves behavior for these
trees; it does not prove field inclusion or output for an arbitrary managed
object. Supplied native-string assignments, always-success bounded allocation,
reallocation, free and diagnostic formatting remain services. Core parse/tree,
native buffer growth, traversal and number/string emission execute as native code.

All 17 known diagnostic codes and the unknown-code arm are executed using
controlled post-parse code overrides. They are separate from natural syntax
inputs: code 12, 16 or 17 in this table does not imply natural reachability under
the audited parser mode. The report contains authored inputs and observations,
not proprietary executable bytes or decompiled bodies.

Reproduce with private Unicorn 2.1.4 dependencies:

```powershell
python reverse_engineering/scripts/audit_unity_json_parser.py GAME_ROOT --output REPORT
```

This extends the [FromJson gateway](unity_fromjson_gateway.md) and
[ToJson gateway](unity_json_gateway.md). Metadata-directed field application at
`0xA8E030`, managed-reference resolution, general object serialization at
`0xAACA50`, allocation failures and exception unwinding remain separate boundaries.
It adds no Assembly-CSharp method classification. The subsequent
[metadata adapter audit](unity_json_fields.md) executes cache selection and
descriptor traversal with supplied individual field bodies; concrete field
conversion and managed-object copy semantics remain open.
