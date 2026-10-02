# CardInteraction lifecycle registration

The pinned audit executes complete `CardInteraction.OnEnable` and
`CardInteraction.OnDisable` callers. It reconstructs visual-service ordering,
capture/reload chronology and the Character click-field update around supplied
engine/runtime/delegate services. It does not reconstruct MouseExit, tween or
coroutine behavior, callbacks, CLR multicast semantics or runtime admission.

Source:
`reverse_engineering/scripts/audit_card_interaction_lifecycle.py`.
Report:
`reverse_engineering/reports/f530404b0f3f_807de4a83df4_card_interaction_lifecycle.json`.
The frozen Awake audit supplies pinned build/Dumper validation, native runtime,
full-range decoding and exact full-ABI/caller event wrappers. The lifecycle
graph, targets and ordered model are independent of the Awake fixtures.

## Exact methods and complete bounds

| Method | Stable ID | Body, exclusive end | Next managed entry | Bytes / instructions |
| --- | --- | --- | --- | --- |
| `CardInteraction::private void OnEnable()` | `tdi5468.m0001` | `0x35FE70..0x35FF8D` | `0x35FF90` | 285 / 68 |
| `CardInteraction::private void OnDisable()` | `tdi5468.m0002` | `0x35FD20..0x35FE6F` | `0x35FE70` | 335 / 79 |

Both unique exact Dumper signatures return void, accept `CardInteraction_o*
and `const MethodInfo*`, and have type signature `vii`. Neither entry is a
shared-RVA alias. All instructions are decoded from their verified entries,
including the complete final instructions; raw backing and padding through
the next managed entry are checked.

Each body has six verified unwind chunks. OnEnable chunks end at
`0x35FEE9`, `0x35FF02`, `0x35FF4C`, `0x35FF70`, `0x35FF7B`, `0x35FF8D`.
OnDisable chunks end at `0x35FDCB`, `0x35FDE4`, `0x35FE2E`, `0x35FE52`,
`0x35FE5D`, `0x35FE6F`. The first chunk's end is not a method boundary. Each
root has unwind Flags 0; following chunks chain to that root. There is no
method-local exception handler. Supplied null/cast gateways and failures stop
at their entry before effects; actual throws and runtime unwinding are outside
the audit. Completed calls verify native Win64 epilogues and all nonvolatile
integer/XMM registers.

Exact CardInteraction TypeDefIndex 5468 fields consumed are private Character
`character +0x20` and private string `animationId +0x40`. The exact Character
TypeDefIndex 5487 field is `private Action <onClick>k__BackingField; // 0x100`.
The native code directly consumes that backing field, without executing a
property accessor. Nominal runtime Action, Unity Object and DOTween class
records, including class-initialization DWORD `+0xE0`, are explicitly supplied
diagnostic storage. They do not establish a general managed-class layout.

## Native chronology

Cold metadata paths resolve exact slots for System.Action, Unity Object and
`Method$CardInteraction.Click()`; OnDisable also resolves DG.Tweening.DOTween.
The callback MethodInfo record's exact MethodAddress is `0x35F4B0`. Flags are
OnEnable `0x288C138` and OnDisable `0x288C139`; only after all three/four whole
resolver calls finish does native code set the corresponding byte to 1.
Warm nonzero bytes, including `0x80` and `0xFF`, skip the path.

OnEnable first supplies whole `CardInteraction.MouseExit(owner, 0)`. Its
effects and any reached authored mutation complete before the owner's current
Character is captured for Object validity.

OnDisable captures the owner's animation-ID pointer before a possible whole
DOTween class initializer. Whole DOTween.Kill receives that captured pointer
even when class initialization replaces or clears the owner's current field.
Native XOR instructions clear full RDX and R8, supplying false completion and
null MethodInfo. The supplied Int32 kill result is ignored; zero, one, negative
low-DWORD and poisoned high-bit outputs all continue identically. Whole
MonoBehaviour.StopAllCoroutines follows with the original owner and full RDX
zero. Mutations by these services complete before later Character capture.

Both bodies capture the current Character before a possible Unity Object
class initializer, then supply that captured identity to whole Object
inequality with full RDX/R8 zero. Only AL gates subscription/removal; poisoned
full returns with low bytes 0, 1, `0x80`, `0xFF` test the actual byte condition.
If false, native code reaches the normal epilogue without constructing an
Action. If true, it reloads the owner's current Character. A helper can make
the captured pointer valid while clearing the reloaded pointer, causing the
actual native null gateway.

The reloaded Character and its old click Action are captured before whole
allocation. Whole Action construction receives a fresh nominal allocation,
the original CardInteraction owner, exact Click MethodInfo and full R9 zero.
Native RBX retains the allocation across the void constructor's poisoned RAX.
Whole Combine on enable or Remove on disable receives the captured old Action
and captured new allocation, full R8 zero and prior poisoned R9.

The resulting pointer undergoes one exact Action class check when nonnull.
A foreign nominal class reaches the actual cast gateway before storing the
field. Null and accepted results are stored by actual native instructions at
captured Character `+0x100`, then passed to the reference barrier. Allocation,
constructor and delegate helper mutations may change the owner's current
Character or current field while the native operation still uses its captured
old operand and captured target. Barrier mutations occur after the native
store and are retained. Unlike the separate Character lifecycle audit, these
bodies have no redundant second class checks and no tail barrier.

## Supplied ABI and contracts

Every supplied entry records full RCX/RDX/R8/R9 bits, full and decoded caller
return, cumulative service ordinal and complete state before any effects.
Per-call relative ordinals select authored failures/mutations, while service
histories persist between retained invocations. Return helpers poison volatile
registers with `0xFACE123456789000..003`; exact full-width zero writes and
preserved incoming/prior-return arguments are independently modeled.

| Whole supplied boundary | RVA |
| --- | --- |
| Metadata resolver | `0x2B7B40` |
| Class initializer | `0x281D90` |
| CardInteraction.MouseExit | `0x35F810` |
| DOTween.Kill | `0x5044D0` |
| MonoBehaviour.StopAllCoroutines | `0x1C7F4A0` |
| Unity Object inequality | `0x1C82480` |
| Nominal Action allocation | `0x2B7D40` |
| System.Action constructor | `0x4D5170` |
| System.Delegate.Combine | `0x116BCC0` |
| System.Delegate.Remove | `0x116E070` |
| Reference barrier | `0x2B6FF0` |
| Null gateway | `0x2B7D90` |
| Cast gateway | `0x2B7040` |

The barriers receive exact captured field address/result, poisoned full R8
and R9, after the actual eight-byte store. Constructor and delegate caller
sites, field-store operands and complete native boundary paths are pinned by
selected opcode assertions and execution coverage.

Delegate buffers use the exact nominal Action class and supplied identity
fields compatible with the pinned Delegate declarations. Ordered invocation
entries and append/last-matching-sequence removal are authored diagnostic
contracts, not reconstructed CLR algorithms. Constructor writes and fresh
helper-owned buffers are explicit effects; a normal candidate may be authored
before an alternate null/source/new/other/foreign result overrides it. These
diagnostic allocations do not establish CLR heap allocation or event dispatch.
Click itself remains whole and unexecuted. MouseExit, Kill and coroutine-stop
requests are modeled as explicit effect histories, without inventing visual,
tween or scheduler implementation.

Owner, class and MethodInfo records are nonnull and stable in this input
domain. The Character, animation-ID and old click pointers can be null. Object
validity and class initialization are supplied, rather than inferred from
diagnostic object bytes or actual destroyed Unity objects. Arbitrary object
admission and runtime initialization behavior remain unresolved.

## Independent model, retention and prefixes

The corpus has 130 cases, including 122 completed returns, six native null
guards and two native cast guards. It crosses cold/warm metadata, both
initialization DWORDs, all four AL bytes, nullable and aliased storage,
prior/own/duplicate/mixed invocation entries, all six result modes, ignored
kill-result bit patterns, and mutations at metadata, visual service, class
initialization, predicate, allocation, constructor, delegate and barrier entry.

Four retained `OnEnable -> OnDisable -> OnEnable -> OnDisable` sequences
preserve full state, previous buffers and cumulative histories. Ten baselines
produce 61 complete stopped prefixes across normal, skipped, foreign-cast,
reloaded-null and captured-target mutation paths. Every stop equals the
baseline's entire preceding event prefix and complete stopped service-entry
snapshot, including native stores already reached.

All 217 case, retained, baseline and stopped rows are checked against an
independent ordered model derived from authored initial bytes and options.
The model does not read native current memory, capture oracles or output
snapshots to select expected behavior. It derives full ABI/caller/ordinal,
complete service-entry snapshots, guard residues, effects, final state and
return/error disposition. Raw snapshots include both Character windows,
class storage, callback tokens, animations, every existing delegate and
fresh buffer, plus complete external effect histories.

Reached-only retention allows the exact native captured click-field store,
reached authored mutation fields, reached class-initialization DWORD and
fresh buffers owned by reached supplied effects. It checks every previously
owned byte. Unallocated arena slots are additionally required to remain exact
`0xA5` windows until a supplied effect owns them; they are outside the logical
object graph until that point. This permits no blanket writable owner/class
window. Stable unrelated bytes and complete modeled values are required at
every supplied entry and final state.

All 147 instructions decode and 143 execute. The four remaining instructions
are exact terminal `int3` traps after supplied null/cast gateways. All native
guard setup, joins and normal paths execute. Thirteen supplied service entries
bring the execution set to 156 addresses. No native bytes or complete
disassembly exports enter the report.

The report pools authored raw windows with
`audit_character_oracle_reveal_join.pool_memory`, then full snapshots with
`audit_report_snapshots.pool_snapshots`. Individual round trips and exact
`expand_memory(expand_snapshots(report)) == fully_expanded_evidence` are
asserted. SHA-256/collision equality preserves every original field, byte,
row and prefix. `audit()` returns the fully expanded evidence. Both final
syntax-preceded producers succeeded and were byte-identical; all 36
reverse-engineering infrastructure tests passed before freeze.
