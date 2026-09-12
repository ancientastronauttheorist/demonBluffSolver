# Characters singleton, pool hiding and construction

Build `f530404b0f3f_807de4a83df4`. The audit covers five exact declarations at
three distinct RVAs: `Awake`, `OnEnable`, `OnDisable`, `HideAll` and `.ctor`.
All three private baseline exports completed at their exact entries. The
script pins GameAssembly, Dumper script and field declarations, then executes
264 authored native cases with explicit engine/runtime service boundaries.

## Singleton publication

`Awake` stores its receiver into the static `Characters.Instance` field before
tail-calling the GC write barrier. It does not first hide pools, create lists,
subscribe events or check an existing instance. A null receiver also becomes
the stored singleton. Cold metadata resolution precedes the store: failure
there preserves the old singleton, whereas barrier failure retains the new
value. No receiver fields change.

## Folded lifecycle callbacks

All three declarations `OnEnable`, `OnDisable` and `HideAll` bind `0x36ca80`.
They traverse the receiver's `characterPool` array at +0x30 in increasing index
order, fetch each pool component's GameObject, then call `SetActive(false)`.
Duplicate pool references cause repeated calls; no list of board characters or
`currentPool` is consulted. There is no event subscription operation in this
body. Null arrays, entries or returned GameObjects stop at the native null
gateway, preserving the earlier completed hiding operations. The helper keeps
the initial array reference and rereads its length during traversal.

The fixture's pool arrays stay stable during engine callbacks. `get_gameObject`
and `SetActive` are explicit services, so it establishes the caller's ordering
and arguments rather than Unity's object-lifetime or activation semantics.

## Constructor ordering

Construction allocates and initializes four distinct lists, storing each only
after its list constructor returns:

1. `characters` (+0x20), a `List<Character>`.
2. `UniquePool` (+0x40), a `List<CharacterData>`.
3. `DuplicatesPool` (+0x48), another `List<CharacterData>`.
4. `BluffMustInclude` (+0x50), another `List<CharacterData>`.

Every reference store precedes its GC barrier. Only after all four does the
native caller tail-dispatch the MonoBehaviour constructor. Array fields,
`currentPool`, `onSetup`, and the singleton remain unchanged in the caller.
The native metadata slots distinguish the first list's type and constructor
from the following three, despite their shared native list-constructor body.

Failures at each allocation, list constructor, barrier and final base
constructor retain exactly the already stored prefix. Synthetic receiver
sentinels verify that no other receiver bytes change; these are caller-write
checks, not a claim that real allocation supplies nonzero object memory.

## Verification boundary

The report checks fourteen exact instruction relationships and every normal
return's stack and eight nonvolatile integer registers. Metadata resolution,
allocation, list initialization, GC barriers, MonoBehaviour construction and
Unity object operations remain explicit gateways. Their injected failures stop
without modeling managed exception unwinding. No live state is read or changed.

With private Unicorn 2.1.4 on `PYTHONPATH`:

```powershell
python reverse_engineering/scripts/audit_characters_lifecycle.py GAME_ROOT DUMPER_ROOT --output reverse_engineering/reports/f530404b0f3f_807de4a83df4_characters_lifecycle.json
```
