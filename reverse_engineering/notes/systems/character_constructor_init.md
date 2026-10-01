# Constructor-produced Character initialization

Build `f530404b0f3f_807de4a83df4`. This audit executes Character construction,
Init or InitWithNoReset, Hidden-state RefreshCharacter, RefreshView, and the
explicit first invocation of DelayReveal.MoveNext in one Unicorn emulator.
It covers 148 normal fixtures, three repeated initialization sequences, four
supplied callback probes and 119 controlled stops. The corpus reaches 569
distinct native instructions, with 31 instruction/literal assertions and
complete actor-byte checks. Private bytes and native bodies remain outside
the repository.

## Produced inputs and fixture boundary

The [standalone constructor](character_constructor.md) supplies the actual
physical actedInfos and onHoverInfo List allocations, pickableUses=1,
act=true and the pinned empty savedAct string. They pass directly to the
initializer without reset, reconstruction from a semantic report, or an
authored replacement of any produced field or list. The List constructor
remains an explicit supplied service which publishes empty count/version
storage. This audit inherits its exact generic MethodInfo and allocation
identity checks, including poisoned void constructor return registers.

UI components, data assets, active-status components/lists, delegate objects,
Gameplay.PrevState and object liveness are authored **before entry**. Their
native field declarations and metadata are pinned. These fixtures do not
establish Unity scene deserialization order, actual constructor invocation
order or provenance for the status objects. The constructor's unrelated-byte
retention check verifies these component bindings survive construction.

| Executed owner | Native entry | Instructions reached |
| --- | --- | --- |
| Character..ctor | 0x3697c0 | 46 |
| Character.Init | 0x365a20 | 187 |
| Character.InitWithNoReset | 0x365720 | 166 |
| Character.RefreshCharacter, Hidden slice | 0x367970 | 74 |
| Character.RefreshView, resulting Hidden slice | 0x367b60 | 50 |
| Character.DelayReveal.MoveNext, first invocation | 0x3756b0 | 45 |

The remaining instruction is the verified folded `ret 0` body at `0x33ed50`.
The report supplies decoded and reached counts separately; this composition
does not imply complete method coverage for the three bounded callees.

## Physical list reuse and retained defaults

Each initializer clears the **constructor-produced actedInfos object**. Its
initial count is zero, so the native caller increments its version without
calling Array.Clear. The first initialization leaves count zero/version one.
Repeated initialization increments that same object's version to two or
three. Neither the initializer nor either refresh routine replaces the list.

The produced onHoverInfo object retains its physical identity, backing-array
identity, empty count and version zero throughout all sequences. The
constructor's empty savedAct string and act=true byte also survive. No
initialization step supplies replacement defaults for them.

The normal corpus separately checks the earlier initializer behavior:
ordinary Init clears trailer/runtime/register-as and sets starting alignment,
while NoReset preserves their existing values. Both clear raw bluff and
revealed, set Hidden after preserving the prior state, clear killedByDemon
and the Start guard, and handle the `-100` stored-ID sentinel. Destroyed death
object references remain stored; live objects are destroyed and cleared.
The ordinary active-status clear occurs after the Hidden-state callback,
while NoReset preserves active statuses. Resistance, target and status
backing values remain unchanged.

RefreshCharacter now executes its actual body in this join. The corpus
distinguishes previous gameplay state zero/Night, usage zero/ResetAfterNight
and zero/two picked controls. It hides picked controls in occurrence order.
Hidden state prevents activation of pickable; a Night usage reset retains one.
RefreshView also executes natively. Required normal uses remain one, Hidden
skips death creation, and a live disguise icon is hidden. An absent or
destroyed disguise icon preserves its supplied active state. Existing
pickable/RIP state otherwise survives the native predicates.

## First yield and repeated construction consumers

The caller publishes its new generated iterator with state zero and the same
physical Character receiver. The explicit StartCoroutine service invokes
native MoveNext once to its first yield. The callee sets state -1, clones the
current data role, writes the supplied clone result to the actor's shared
role pointer, allocates/constructs a wait using exact float32 bits
`0x3e99999a`, and stores current/state one. Null clone fixtures still publish
the null role pointer and reach that yield.

The repeated sequences are Init→NoReset, NoReset→Init→NoReset, and
Init→Init→Init. No actor or produced-list memory is reset between calls.
Earlier iterator identities, captured receivers, yielded states and wait
identities survive later initializations. The actor's role pointer changes
when a later first invocation publishes another clone. No existing iterator
is resumed or cancelled, and no coroutine readiness is inferred.

## Callback and failure ordering

Events carry the owning phase and initialization occurrence, distinguishing
Construct, Init/NoReset, RefreshCharacter, RefreshView and FirstYield. All
normal service arguments are checked at their native widths. Normal return
checks include all eight integer nonvolatiles, XMM6–15 and the original
stack. Services poison volatile registers under the Windows x64 contract.

The inert state-change callback observes Hidden state, the already-cleared
actedInfos with version one, and the still-active old status list. Explicit
replacement callbacks show that ordinary Init reloads the status component
and clears the supplied replacement, while NoReset leaves both active lists
alone. A separate callback sets uses to zero with previous gameplay state
zero: Hidden RefreshCharacter preserves the zero, and native RefreshView
hides pickable. These mutations are authored callback effects, not fixture
patches applied between construction and initialization.

Cold live-death baselines for both initializer methods cover 119 stops at
metadata, allocation, construction, barriers, diagnostics, UI, callbacks,
native refresh entries, coroutine handoff and first yield. Each stopped
event sequence, semantic snapshot and complete actor bytes exactly equals
its successful baseline prefix. A stopped constructor does not proceed to
initialization, and a failed nested first invocation does not fabricate
successful scheduler return. Managed exception unwinding remains outside
the audit.

All runtime/metadata operations, List construction, GC barriers, diagnostics,
Unity object/UI effects, callback effects, role cloning and wait construction
remain explicit supplied services. The scheduler handoff is the existing
explicit synchronous first-yield contract. This producer join does not
broaden scheduler registration, readiness, second resume, arbitrary
post-initializer state or scene callback claims.

Reproduce with Unicorn 2.1.4 and the private emulation PYTHONPATH:

```powershell
python reverse_engineering/scripts/audit_character_constructor_init.py GAME_ROOT DUMPER_ROOT --output reverse_engineering/reports/f530404b0f3f_807de4a83df4_character_constructor_init.json
```

## Guarded Rust producer replay

The separate `bluff::character_constructor_init` module compares 153 supported
normal profiles: 148 fixtures, three retained sequences and two cold baselines.
Five focused tests retain complete intermediate/final Actor fields, physical
Lists/defaults, old storage, status backing values, UI controls and continuations.
Constructor APIs and the initializer/RefreshCharacter/RefreshView semantic
projections are compared separately; no merged raw metadata trace is claimed.

Serialized components and transform/template bindings are explicit valid inputs.
Base, constructors, callbacks and UI are inert; role clone outcomes and synchronous
first yield require independent provenance. New empty Lists share their supplied
static array, while retained backing arrays are unaliased. Known typed/callback
collisions and fresh allocation aliases reject. Whole-producer projection work is
reserved before maps/clones. Scene loading, mutations/failures, later resumes and
real readiness remain outside the contract. All 827 Rust library tests and the
release build passed.
