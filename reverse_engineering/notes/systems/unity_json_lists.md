# Native scalar and string List copying

The pinned engine audit executes the native List backing-field locator, collection
context helper, constructor wrapper, scalar/string element processors, JSON
writing/rendering and reload. It contains 28 numeric/null/empty save/load cases,
56 integer-List reader cases, two string-List cases and 373 controlled service
stops. Twenty-five exact instruction/export assertions cover seven complete
chained-unwind families. Required native entries must execute.

Native `0x75F310` enumerates the supplied List fields and selects the first field
whose runtime offset is `0x10`. Both `_size`-first and `_items`-first inventories
are tested. The helper at `0x784860` follows the List path to `0x784680`, obtains
or constructs the List, resolves its backing array's element class, and retains
the outer field location for later publication. The optional backing-lookup cache
flags at `0x1CC6209` and `0x1CC620A` are explicitly disabled.

Observed behavior over coherent supplied List layouts:

- An existing List retains its object identity. Missing or differently cased
  JSON members preserve its data and backing array.
- A null List field is constructed and published as an empty List even when its
  JSON member is missing. This differs from the array path, where a missing null
  field stays null.
- A found null or tested non-array value produces zero logical elements.
- Equal old/new logical counts reuse the backing array, preserving its spare
  capacity and tail bytes. A changed count requests a new backing array of the
  exact required length, even when the old capacity could have held the result.
- Native overwrite retains the existing version field. It updates the backing
  reference and logical size directly rather than invoking List Add/Clear.
- Saving writes logical elements only and leaves both the input List and all
  backing storage untouched. Spare capacity contains explicit nonzero sentinels;
  string-List tail slots contain pointers that would fail if traversed.
- Reloaded newly constructed Lists have the fixture constructor's zero version
  and exact logical capacity. Native conversion retains the previously observed
  boolean and floating normalization, string null-to-empty conversion and NUL
  truncation.

The null path executes native constructor wrapper `0x75F5E0`. Its exact export is
`il2cpp_runtime_object_init_exception`, independently bound alongside
`il2cpp_object_new`. The actual managed constructor is a supplied service that
initializes coherent empty List storage and returns no exception. Runtime
exception-object processing remains outside this boundary.

Failure checks independently stop every observed service occurrence for new,
same-count and resized reads, and null/existing writes. They verify complete
event prefixes and the retained published/unpublished List and array contents.
The report stores each baseline stream once; a failure's `baseline_event_count`
identifies its verified prefix. This avoids repeating the same event snapshots
in every failure record. Successful returns preserve the stack and all eight
nonvolatile integer registers. Authored object header/trailing guards remain
untouched.

Runtime List classification, class/field/type metadata, object allocation and
managed construction, array allocation, GC stores, cache and allocator services
remain explicit fixtures. Generic Dumper field offsets are not treated as runtime
List layout evidence; the engine's consumed offsets and authored metadata are
kept distinct. Actual managed constructors, enabled backing lookup caches, nested
containers, compound elements and real runtime discovery remain open.

```powershell
python reverse_engineering/scripts/audit_unity_json_lists.py GAME_ROOT --output REPORT
```

Report: `f530404b0f3f_807de4a83df4_unity_json_lists.json`. Private Unicorn
2.1.4 dependencies are required. No native bytes or decompiled bodies are retained;
this engine audit adds no Assembly-CSharp classification.
