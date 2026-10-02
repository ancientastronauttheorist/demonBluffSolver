# Character event lifecycle

The pinned native audit executes complete `Character.OnEnable` and
`Character.OnDisable` callers. It reconstructs their event-storage chronology
around explicitly supplied delegate and runtime services. It does not execute
the subscribed callbacks, CLR multicast operations, Unity object admission or
scene lifecycle, or exception unwinding.

Evidence is produced by
`reverse_engineering/scripts/audit_character_event_lifecycle.py` and saved as
`reverse_engineering/reports/f530404b0f3f_807de4a83df4_character_event_lifecycle.json`.
The extraction manifest, `dump.cs`, `script.json`, exact class declarations and
method signatures are pinned through the frozen DeckCharacter surface audit.
The immutable method inventory resolves both unique direct concrete entries;
there are no shared-RVA aliases promoted by this evidence.

## Exact methods and exception bounds

| Stable ID | Declaration | RVA interval, exclusive end | Next managed entry | Bytes / instructions |
| --- | --- | --- | --- | --- |
| `tdi5487.m0017` | `Character::private void OnEnable()` | `0x366E40..0x3674DE` | `0x3674E0` | 1,694 / 384 |
| `tdi5487.m0018` | `Character::private void OnDisable()` | `0x3667A0..0x366E3E` | `0x366E40` | 1,694 / 384 |

Dumper gives both `void Character__OnEnable/OnDisable(Character_o* __this,
const MethodInfo* method)` and type signature `vii`. The explicit original
caller supplies an arbitrary full-width MethodInfo argument. The bodies are
independently decoded from verified entries, through all native return/failure
paths; complete final instructions and trailing `int3` alignment are checked
against the next managed entry. Each body has one containing unwind record,
with shared unwind-data RVA `0x2520C38` and Flags `0`. There is no method-local
exception handler, catch or finally path in these records.

Null/cast runtime gateways and every authored service failure stop at the
supplied entry before its effects. No actual throw, unwind, runtime handler,
or caller-side exception processing is modeled. The native Win64 epilogue and
nonvolatile integer/XMM preservation are verified for completed calls only.

## Fields and seven channels

Exact `Character` TypeDefIndex 5487 fields consumed here are
`public CardInteraction cardInteraction; // 0xA0` and
`public bool disableAnimated; // 0x164`. Exact `CardInteraction` TypeDefIndex
5468 fields are `Action onHover` at `0x60` and `Action onHoverExit` at `0x68`.
All channels use the nominal `System.Action` class; callback metadata tokens
bind exact unique Character methods, without executing those methods.

| Order | Storage | Offset | Callback | Callback RVA | MethodInfo slot RVA |
| --- | --- | --- | --- | --- | --- |
| 1, optional | `CardInteraction.onHover` | `0x60` | `ShowAnimatedArt` | `0x368B40` | `0x270FFA0` |
| 2, optional | `CardInteraction.onHoverExit` | `0x68` | `HideAnimatedArt` | `0x365260` | `0x270FC70` |
| 3 | `GameEvents.OnGameplayStateChange` | `0x0` | `RefreshCharacter` | `0x367970` | `0x270FE90` |
| 4 | `GameplayEvents.OnShowEyeOracleInfo` | `0xB8` | `OracleEyeActive` | `0x3675A0` | `0x270FD80` |
| 5 | `GameplayEvents.OnHideEyeOracleInfo` | `0xC0` | `HideOracleInfo` | `0x3654F0` | `0x270FCF8` |
| 6 | `UIEvents.OnCharacterPrefChanges` | `0x8` | `ReInitPreferences` | `0x367890` | `0x270FE08` |
| 7 | `UIEvents.OnUIUpdate` | `0x0` | `RefreshView` | `0x367B60` | `0x270FF18` |

Exact static class declarations are GameEvents 5518, GameplayEvents 5519 and
UIEvents 5523. Their 6, 29 and 21 static Action fields are captured by the
report, including unrelated fields that must retain their authored bytes.
Runtime class storage pointers at `+0xB8`, and the Unity Object initialization
DWORD at `+0xE0`, are explicitly authored consumed runtime storage. This is not
a general managed class-layout or scene-admission proof.

`OnEnable` uses whole supplied `System.Delegate.Combine`; `OnDisable` uses
whole supplied `System.Delegate.Remove`. Their otherwise corresponding native
instruction sites differ by `0x6A0`; actual opcode/operand assertions verify
each method separately.

## Capture and reload chronology

Each method's cold metadata path calls the whole supplied resolver twelve
times and sets its native flag byte only after all calls complete. The flag
slots are `OnEnable: 0x288C167` and `OnDisable: 0x288C168`. A nonzero flag,
including `0x80` and `0xFF`, bypasses that path.

Any nonzero `disableAnimated` byte skips both instance channels and the Object
class initializer/inequality call. The five static channels still run. When
the byte is zero, native code captures `cardInteraction` before a possible
Object class-initialization helper, then supplies that captured pointer to
Object inequality. Only AL decides whether the instance channels run. Full
return bits are authored as `0xFEDCBA9876543200 | low_byte`; `0x00`, `0x01`,
`0x80` and `0xFF` exercise the exact byte test. A captured nonnull pointer can
produce true while a helper clears the owner's current pointer, causing the
following native null guard.

For each instance channel, native code reloads the current interaction,
captures its old Action before allocation, and retains that interaction as
the store/barrier destination across allocation, Action construction and
Combine/Remove. The second channel reloads the owner's interaction after the
first barrier. A helper can therefore replace the owner interaction while
the first store still targets the previously captured instance.

Each static channel captures its old Action before allocation. It reloads
the class static-storage pointer after the supplied allocation, constructor
and delegate operation when storing the returned Action and computing the
barrier address. Authored helpers can swap between two static blocks: the
operation receives the old captured Action while its result enters the later
block. The final UI update barrier is a native tail jump after the epilogue;
its supplied caller return is the original caller's return address.

Every accepted or null result is stored by actual native instructions before
barrier entry. A foreign nominal class fails its first exact class equality
test before any field store. A nonnull accepted result undergoes a redundant
second exact equality test after the store; no supplied service occurs between
the tests, and the authored class metadata remains stable. Those seven
redundant second-failure stubs per body are explicitly excluded rather than
claimed as executed.

## Supplied ABI and diagnostic contracts

Every supplied-entry event, including failure/guard entries, records full
RCX/RDX/R8/R9 bits, cumulative service ordinal, full return address and decoded
caller RVA. Relative per-invocation ordinals select authored mutation/stop
phases; retained service histories continue accumulating.

| Service | RVA | Consumed call-site setup |
| --- | --- | --- |
| Metadata resolver | `0x2B7B40` | RCX is exact metadata slot; remaining volatile registers retain entry/prior-return bits |
| Object class initializer | `0x281D90` | RCX is nominal Object class; initialization DWORD is zero at entry |
| Object inequality | `0x1C82480` | RCX captured interaction; full RDX/R8 zero; R9 retained |
| Action allocation | `0x2B7D40` | RCX nominal Action class; other volatile bits retained |
| `System.Action..ctor` | `0x4D5170` | RCX fresh allocation, RDX owner, R8 exact callback MethodInfo, full R9 zero |
| `System.Delegate.Combine` | `0x116BCC0` | RCX captured old Action, RDX captured new allocation, full R8 zero, poisoned R9 |
| `System.Delegate.Remove` | `0x116E070` | Same captured operands and widths |
| GC barrier | `0x2B6FF0` | RCX exact reached field address, RDX stored result; instance R8 poisoned, static R8 result pointer; R9 poisoned |
| Null gateway | `0x2B7D90` | Exact reached volatile bits; no supplied effects |
| Cast gateway | `0x2B7040` | RCX foreign result, RDX nominal Action class; instance R8 poisoned, static R8 foreign result |

Supplied returns poison RCX/RDX/R8/R9 with
`0xFACE123456789000..003`, along with other volatile registers. The void
constructor's poisoned RAX cannot replace the allocation held in a native
nonvolatile register. The complete raw ABI verifies both that capture and
full-width zero writes. Supplied allocation and delegate operations may own
fresh 128-byte diagnostic buffers; unallocated arena slots retain their exact
initial `0xA5` bytes until a reached supplied effect owns them.

The exact Delegate declaration pins `invoke_impl +0x18`, `m_target +0x20`,
`method +0x28` and `method_code +0x40`; Action is the exact sealed
MulticastDelegate subclass and its private delegate-array field is pinned at
`+0x78`. These caller bodies do not invoke the constructed Action or consume
that array. Constructor-written identity fields and ordered invocation lists
are supplied diagnostics. Append/last-sequence-removal describes the authored
Combine/Remove contract used by this corpus, not reconstructed CLR delegate
semantics or general event dispatch. Explicit result modes include normal,
null, captured source, captured new allocation, other nominal Action, and
foreign class. Supplied helper effects include their fresh diagnostic
representations even when an alternate authored result overrides the normal
candidate. No helper or subscribed callback gains method coverage here.

Input storage uses a nonnull owner, nonnull class/static blocks and stable
metadata slots, with nullable owner interaction and nullable channel Actions.
Arbitrary null static blocks or mutable native class headers are outside this
diagnostic domain. This audit supplies Object validity and initialization
results; it does not infer whether a real Unity object has been destroyed.

## Verification and retained evidence

The corpus contains 282 cases: 258 completed returns, 10 native null guards,
and 14 first-cast guards. It crosses cold/warm metadata, initialized/zero
Object state, all four Boolean/AL bytes, nullable and aliased channels,
prior/own/duplicate/mixed invocation diagnostics, every channel's alternate
result modes, and mutations at metadata, class initialization, predicate,
allocation, constructor, delegate and barrier phases.

Four retained `OnEnable -> OnDisable -> OnEnable -> OnDisable` sequences
preserve complete state, allocated buffers and cumulative ordinals between
calls. Ten baselines include normal, skipped-instance, mutated-storage,
post-first-store null-guard, and final-channel cast-guard paths. Their 252
stopped rows exactly equal each complete preceding event prefix and the
stopped service's full entry snapshot, including native stores already
performed. All 560 case, retained, baseline and stopped rows are checked by
an independent ordered semantic model built from authored initial state and
options. It does not read current native memory or service capture oracles.
It derives full ABI, complete snapshots, partial guard residues, return/error
disposition, effects and cumulative service counts.

Reached-only retention allows exactly the native field stores, the exact
authored mutation fields, and fresh buffers owned by reached supplied effects.
Owner padding, class bytes, MethodInfo tokens, unused static fields, previous
delegate records and unallocated buffers remain byte-for-byte stable unless
an explicitly reached effect writes their precise range.

Both bodies decode 768 instructions and execute 698. The exact remaining set
is 30 terminal `int3` traps and 40 instructions in stable-class excluded
redundant second-cast stubs. First-cast failure paths and all null gateways are
executed. The combined native execution set contains 708 addresses, including
10 supplied service entries. The report contains ranges/counts and selected
assertion totals, without native byte or complete disassembly exports.

The saved report uses two lossless encodings: authored raw `memory` windows
are pooled by `audit_character_oracle_reveal_join.pool_memory`, then complete
initial/final/service snapshots by `audit_report_snapshots.pool_snapshots`.
Recover exact values using
`expand_memory(expand_snapshots(report))`. Both encodings verify SHA-256 keys,
collision equality and full expanded-value equality; no fields, byte windows,
event prefixes or rows are removed. `audit()` still returns the fully expanded
evidence. Both successful syntax-preceded final producers were byte-identical;
all 36 reverse-engineering infrastructure tests passed before freeze.

This proves the bounded native callers under explicit service contracts. It
does not prove actual runtime subscriptions, callback behavior, scene object
validity, CLR lifecycle, event scheduling, exception unwinding, or arbitrary
managed-object admission.
