# Native field enumeration and eligibility

The pinned engine audit now executes metadata construction at `0x784120`, field
eligibility at `0x7827A0`, supported callback checks at `0x783FE0`, actual numeric
descriptor construction, and joined numeric FromJson application. It has 226
fixtures and executes 1,269 addresses during metadata/application. Eighty-two
instruction/export assertions cover the complete unwind families and independently
bind fifteen runtime service slots. The constructor's exact class-name literals
bind `SerializeField` to runtime `+0xCD0` and `SerializeReference` to `+0xCD8`.
The existing numeric descriptor audit supplies its additional 44 assertions.

Runtime services provide an explicit class/field inventory. The native builder
enumerates each class's fields into a temporary vector before querying each
field's metadata and applying eligibility. Parent fields are built before child
fields. A parent equal to any of the three pinned runtime stop classes at
`+0x520`, `+0xCB0` or `+0x548` is skipped. The supplied excluded-type and subclass
predicates can also stop parent traversal. These stop fixtures verify that parent
fields are never enumerated. Runtime discovery remains supplied; no fixture token
is asserted to be a real managed class.

Eligibility rejects field-flag low-byte bits `0x10`, `0x20` and `0x80`. Among the
supported scalar fixtures, low visibility bits equal to 6 permit an ordinary
field. Otherwise, either supplied serialization attribute permits it. A flag
above the low byte does not by itself suppress an eligible field. Actual native
name filtering rejects a name containing `.`; empty names and differently cased
names remain eligible in the construction fixtures. The later native JSON member
lookup is independently case-sensitive.

Native registry lookup and the descriptor factory produce each accepted numeric
descriptor. The complete joined path starts with an actual parsed object, reaches
the adapter's direction-9 metadata-build route, enumerates and filters the supplied
fields, builds descriptors, then calls the actual numeric readers. Parent/child
ordering and missing-field retention survive this join. No supplied metadata
builder or authored descriptor replaces those native stages.

The 226 fixtures also include controlled stops at each metadata-service occurrence
in representative ordinary, inherited and joined runs. Snapshots retain the exact
earlier enumeration, descriptors and field writes. A stop at later cleanup retains
already applied values. Empty inventories construct no descriptors and preserve
all field bytes. Successful returns preserve the stack and all eight nonvolatile
integer registers. Boolean service arguments are validated at the native byte
width; their upper register bits are not assumed initialized.

Class/field exports, excluded-type and collection predicates, allocation,
descriptor reservation, GC reference storage and vector cleanup remain explicit
services. Callback interface checks return false; callback method discovery and
execution are unclaimed. These fixtures use numeric class tokens with supplied
type enum 8. Compound types, enums, aliases, actual managed metadata discovery,
strings, collections, nested construction, references and writer conversion remain
outside this boundary. This is a native numeric copy over a supplied class
inventory, not the complete engine copy or general reflection semantics.

```powershell
python reverse_engineering/scripts/audit_unity_json_metadata.py GAME_ROOT --output REPORT
```

Report: `f530404b0f3f_807de4a83df4_unity_json_metadata.json`. Private Unicorn
2.1.4 dependencies are required. No engine bytes or decompiled bodies are retained;
this engine audit adds no Assembly-CSharp classification.
