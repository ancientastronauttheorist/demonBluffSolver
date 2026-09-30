# Native JSON numeric field readers

The pinned engine audit executes nine field readers selected by the actual
[reader registry](unity_json_registry.md). It contains 333 standalone fixtures
and sixteen composed batches through the actual FromJson metadata adapter,
descriptor loop and normal reference-scope cleanup. Seventy-two exact instruction
assertions cover all nine complete field wrappers, in addition to the registry's
26 and adapter's 23 assertions. The audit executes 1,465 distinct addresses during
numeric application. Successful returns preserve the stack and eight nonvolatile
integer registers; guard bytes outside each fixture field remain intact.

| Class-token source | Native field reader | Field width in fixtures |
| --- | --- | --- |
| core `+0x120` | `0xA917D0` | 4 bytes |
| core `+0x130` | `0xA91900` | 1 byte |
| core `+0x188` | `0xA91A20` | 4 bytes |
| core `+0x118` | `0xA91DB0` | 2 bytes |
| core `+0x128` | `0xA91ED0` | 8 bytes |
| core `+0x178` | `0xA92000` | 2 bytes |
| core `+0x168` | `0xA92420` | 1 byte |
| core `+0x108` | `0xA92540` | 4 bytes |
| core `+0x110` | `0xA92670` | 8 bytes |

Source labels are byte offsets in the runtime tables, not claims about discovered
managed class names. The report retains raw output bytes and native handler RVAs.
Native lookup and conversion execute over the tree produced by the actual parser;
Python does not substitute a JSON decoder or numeric conversion routine.

The field wrappers compute reference-object addresses as object plus signed field
offset. Value-object fixtures use object plus signed inner offset minus `0x10`
plus signed field offset. Both paths produce the same field values. The wrappers
pass the descriptor's key and flags to their native conversion kernels.

Member lookup is case-sensitive in these fixtures. A missing key preserves the
old field and clears the parser's field-found flag. The first exact duplicate
member wins. Nested members do not replace a matching member in the current root.
Combining parser flag bit 1 with descriptor flag bit 19 suppresses the read;
either flag alone still permits it. The kernels restore the previous node,
diagnostic type pointer and traversal-stack depth after each call.

The reader at `0xA917D0` converts `"12"` to integer 12 and `"-12xyz"` to -12.
Booleans, null, arrays, objects and a nonnumeric string produce zero in the tested
cases. Fractions truncate toward zero. Integer 2147483648 produces bytes
`00000080`, while -2147483649 produces `ffffff7f`. The report also preserves the
distinct results for much larger integers, infinities and NaN; those must not be
generalized into an unrestricted wrapping rule. No tested input raises the
parser's field-error flag. These are observed native conversions, not recommended
input validation behavior.

The reader at `0xA91A20` converts JSON `-0.0` to float bytes `00000080`, preserving
the negative-zero sign. NaN produces `0000c07f`; Infinity and `1e100` produce
`0000807f`. Exact numeric bits are retained, including overflow and underflow.

The composed fixtures use cache-selected descriptors with actual numeric handler
pointers. Native traversal performs the field calls, native scope cleanup clears
its links, and the resulting fields match standalone native calls. The GC
reference store and vector-storage cleanup remain explicit services. Class
discovery, field inclusion, descriptor construction, alias registries, managed
references, strings, arrays, nested object construction and writer conversion
remain outside this boundary. This is a numeric application join over authored
descriptors, not arbitrary managed-object copying.

```powershell
python reverse_engineering/scripts/audit_unity_json_primitives.py GAME_ROOT --output REPORT
```

Report: `f530404b0f3f_807de4a83df4_unity_json_primitives.json`. Private Unicorn
2.1.4 dependencies are required. No native bytes or decompiled bodies are retained.
This engine audit adds no Assembly-CSharp classification.
