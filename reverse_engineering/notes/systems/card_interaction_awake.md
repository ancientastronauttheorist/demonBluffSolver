# CardInteraction Awake and hover gates

The exact native audit executes `CardInteraction.Awake`, `BlockHover`, and
`UnblockHover`. It reconstructs the caller's metadata, field-store, instance-ID
width, boxing-input, formatting-input and Boolean-gate chronology under
explicitly supplied engine/runtime services. It does not reconstruct those
services or infer Unity/CLR scene admission.

Source:
`reverse_engineering/scripts/audit_card_interaction_awake.py`.
Pinned report:
`reverse_engineering/reports/f530404b0f3f_807de4a83df4_card_interaction_awake.json`.
The frozen DeckCharacter surface audit supplies pinned GameAssembly/Dumper
validation, exact metadata/declaration reads, raw-backed full-range decode,
Unicorn runtime and volatile-return poisoning. All graph state, models and
execution targets in this audit are independent of its original UI fixtures.

## Exact targets and aliases

| Stable method | Exact declaration | Native body, exclusive end | Next managed entry | Bytes / instructions |
| --- | --- | --- | --- | --- |
| `tdi5468.m0000` | `CardInteraction::private void Awake()` | `0x35F3E0..0x35F496` | `0x35F4A0` | 182 / 44 |
| `tdi5468.m0003` | `CardInteraction::public void BlockHover()` | `0x35F4A0..0x35F4A5` | `0x35F4B0` | 5 / 2 |
| `tdi5468.m0004` | `CardInteraction::public void UnblockHover()` | `0x360070..0x360075` | `0x360080` | 5 / 2 |

All three exact Dumper signatures return `void` and take `CardInteraction_o*
plus `const MethodInfo*`, type signature `vii`. Awake has one complete unwind
chunk ending at the supplied exclusive end. Both gates are pointer-backed
leaves without unwind records. Decode covers all native paths and complete
last instructions; padding is checked through each next managed entry.

UnblockHover shares its RVA with the compatible declaration
`UnityEngine.Timeline.TrackAsset.OnClipMove`. That alias is recorded explicitly
as unpromoted. The evidence concerns the exact CardInteraction declaration and
its field only. The folded CardInteraction constructor at `0x33E820`, lifecycle
registration pair, MouseOver/MouseExit, clicks and animation behavior are not
targets here.

Exact CardInteraction TypeDefIndex 5468 fields consumed are:

- `private Character character; // 0x20`
- `private string animationId; // 0x40`
- `private bool blockHover; // 0x70`

The two gate bodies write precisely one byte: 1 or 0 at `+0x70`. Initial bytes
0, 1, `0x80` and `0xFF` exercise both writes while retaining all other bytes.
They perform no supplied calls or class/metadata initialization.

## Awake chronology and ABI widths

The cold flag byte at `0x288C137` gates three whole supplied metadata requests.
Their native order and exact RIP-relative slots are derived from the decoded
LEA/call pairs. The flag becomes 1 only after all complete. Warm bytes 1,
`0x80` and `0xFF` bypass the requests. Metadata binds exactly
`Method$UnityEngine.Component.GetComponent<Character>()`, `int_TypeInfo`, and
the pinned string literal `cardAnim_{0}`. The generic metadata record's
MethodAddress is zero; the body explicitly calls the existing generic entry
at `0x606FC0` with that supplied nominal MethodInfo token.

Actual native code retains the original owner in RBX throughout Awake. It
passes that owner and exact generic token to whole supplied GetComponent,
stores the returned Character pointer at `+0x20`, then enters its reference
barrier. A null supplied Character is valid in this diagnostic domain: the
store/barrier still completes and GameObject lookup follows. A barrier
mutation can replace the owner's current Character without altering the
original owner used by subsequent lookups.

Whole supplied Component.get_gameObject receives the original owner and full
RDX zero. A null result enters the actual native null gateway after the prior
Character store/barrier and completed getter effects. A nonnull returned
GameObject is passed directly to whole supplied Object.GetInstanceID, again
with full RDX zero. It is not inferred from the owner's stored Character.

Native `mov dword ptr [rsp+0x30], eax` stores exactly the low DWORD of the
supplied GetInstanceID return. That caller-home location is original entry
RSP plus 8; its authored high DWORD is retained. The boxing service receives
RCX nominal Int32 class, RDX the full exact stack address, and the low DWORD
plus full backing QWORD are captured at entry. The corpus covers 0, 1,
`0x7FFFFFFF`, `0x80000000`, `0xFFFFFFFF` and high-bit-poisoned 64-bit returns.
The observed native operation preserves 32-bit Int32 bit patterns; no
64-bit identity or formatter interpretation is inferred from the return.

Whole supplied boxing returns a nominal boxed pointer or null. Native code
reloads the current literal slot only after boxing, so an authored boxing
mutation replacing that slot reaches the formatter. The boxed return is
passed as full RDX to whole supplied `String.Format`; native R8 is explicitly
zero. The formatter returns a supplied string pointer or null, which actual
native code stores at owner `+0x40` before its second reference barrier.
Formatter mutations of the old animation ID occur before that native store;
barrier mutations occur afterward and remain in the final state.

| Whole supplied boundary | RVA | Exact caller return RVA |
| --- | --- | --- |
| Metadata resolver, three requests | `0x2B7B40` | decoded request sites |
| Component.GetComponent generic entry | `0x606FC0` | `0x35F42C` |
| Character-field reference barrier | `0x2B6FF0` | `0x35F43B` |
| Component.get_gameObject | `0x1C79FD0` | `0x35F445` |
| Object.GetInstanceID | `0x1C81060` | `0x35F454` |
| Int32 boxing | `0x282580` | `0x35F469` |
| String.Format | `0xF74DF0` | `0x35F47B` |
| Animation-ID reference barrier | `0x2B6FF0` | `0x35F48A` |
| Native null gateway | `0x2B7D90` | `0x35F495` |

Every supplied entry records full RCX/RDX/R8/R9 bits, full/decoded return
address and cumulative ordinal. Return helpers poison those registers with
`0xFACE123456789000..003`. The model verifies which arguments are replaced
or fully zeroed and which preserve incoming/prior-helper bits. Win64
nonvolatile integer and XMM registers and final stack restoration are checked
for completed calls. All targets are void; arbitrary RAX contents are not
modeled as a managed return value.

## Authored services and scope

Engine lookup outputs are explicit nominal pointer contracts, including
alternate Character/GameObject identities. Generic lookup, boxing and string
formatting bodies remain supplied. Boxing writes an explicitly owned 128-byte
diagnostic window with nominal class and a four-byte payload; its layout and
nullable result do not establish CLR allocation behavior. Fixture strings
contain authored header/length/UTF-16 bytes, but no actual String decoder or
formatter body executes. Formatting outputs can alias prior or literal
strings, and the literal can be replaced by the authored `alternate_{0}`
fixture at a reached service mutation. This is diagnostic caller evidence,
not language/formatter output admission.

Null owner profiles record actual unmapped write faults at the reached field
store. Awake can complete cold metadata and whole supplied GetComponent
before its write at `0x35F433` faults; both gates fault at their first byte
store. These intentionally supplied getter contracts retain all completed
effects before the fault. They do not assert real Unity behavior on a null
receiver. Runtime null exceptions and authored failures stop at entry before
effects; no real throw, managed handler or exception unwinding is executed.

## Verification and report encoding

The corpus has 67 cases crossing cold/warm metadata, nullable service outputs,
Int32 bit patterns, alternate/aliased outputs, null-owner faults, exact Boolean
bytes, and metadata/getter/boxing/formatter/barrier mutations. Four retained
four-call sequences preserve complete graph, home-slot bytes, boxed diagnostics
and cumulative histories. They include Awake/gate interleaving and changed
Int32/box/string outputs across later Awake calls. Three baselines produce
21 complete stopped prefixes, including a native null guard. Each stop equals
the full baseline prefix and exact stopped entry snapshot, preserving native
stores and the DWORD home-slot write already performed.

Every case, retained call, baseline and stop is checked against an independent
ordered model built from authored initial bytes and options. It does not read
current emulator memory. It derives all service ABI/events/snapshots, partial
fault/guard states, effects, cumulative counts and final full diagnostic state.
Reached-byte retention allows exactly the two native pointer stores, the gate
byte, reached authored mutation fields and reached boxing buffers. All other
owner, class, MethodInfo, GameObject, Character, string and boxed bytes remain
stable. The high DWORD of the boxing home slot is always retained.

All 48 body instructions decode; 47 execute. The sole unexecuted instruction
is terminal trap `0x35F495`, after the supplied null gateway. Eight supplied
service entries bring the complete execution set to 55 addresses. No native
byte or full disassembly export enters the report.

Authored memory windows use `audit_character_oracle_reveal_join.pool_memory`;
full snapshots then use `audit_report_snapshots.pool_snapshots`. Each producer
asserts individual round trips and exact
`expand_memory(expand_snapshots(report)) == fully_expanded_evidence`.
SHA-256/collision equality is verified without removing fields, bytes, rows or
prefixes. `audit()` returns fully expanded evidence. Two successful final
syntax-preceded producers were byte-identical; all 36 reverse-engineering
infrastructure tests passed before freeze.
