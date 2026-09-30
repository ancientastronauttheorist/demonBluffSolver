# Native numeric descriptor construction

The pinned engine audit executes the descriptor factory at `0x77FF80`, ordinary
registry lookup at `0x77F620`, and supported noncollection paths through the
fixed-buffer and lazy-reference predicates. It has 105 construction/failure
fixtures and 36 numeric copies joining factory output to the actual parser,
metadata adapter and numeric handlers. Forty-four instruction/export assertions
decode the complete chained unwind families, including the factory's code after
its first return. Nine IL2CPP service slots are independently bound to exact
registration strings. Successful returns preserve the stack and all eight
nonvolatile integer registers.

Metadata services supply a field identity, name, parent, type, class, enum value,
offset and value-type result. The actual native reader registry retains distinct
class tokens. The factory searches that registry natively, chooses its first
matching record and creates a 128-byte descriptor containing the actual handler
pointer, field/parent/key identities, type, offset, flags and zero tail pointers.
The fixtures use type enum 8 with explicit class tokens; they establish native
dispatch over supplied metadata, not discovery of real managed classes or a
complete mapping between class tokens and type enums.

The descriptor's flags combine the supplied description flags with the selected
registry's metadata. Its value-type byte follows the runtime metadata service.
Reserved and newly reserved descriptor storage produce identical records.
Controlled stops at collection classification, metadata exports or reservation
retain the exact prior state. A missing handler suppresses the ordinary numeric
descriptor. A later duplicate record with a valid handler does not override that
first match. Requiring a feature rejects a registry entry whose feature byte is
clear; enabling that byte allows the same native handler.

The 36 joined fixtures take actual factory output directly into direction-9 cache
selection and native descriptor traversal. No authored descriptor replaces it.
Each numeric field is filled from the actual parsed tree, including missing-key
retention, duplicate-key selection and a changed supplied field offset. Results
match independent standalone native reader calls. Native scope cleanup clears
its links; the GC reference store and vector cleanup remain supplied services.

The metadata exports, collection predicate and storage reservation are explicit
services. Descriptor buffers and stack bytes start at zero in these fixtures;
unspecified stack-derived descriptor bytes are not assigned semantic meaning.
Field eligibility/enumeration, actual managed metadata discovery, inheritance,
enums, collections, fixed buffers, lazy references and compound descriptor paths
remain unclaimed. This closes a numeric descriptor/application join, not the
complete managed JSON copy.

```powershell
python reverse_engineering/scripts/audit_unity_json_descriptors.py GAME_ROOT --output REPORT
```

Report: `f530404b0f3f_807de4a83df4_unity_json_descriptors.json`. Private Unicorn
2.1.4 dependencies are required. No native bytes or decompiled bodies are retained;
this engine audit adds no Assembly-CSharp classification.
