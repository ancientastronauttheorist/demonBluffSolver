# CardInteraction MouseExit

The audit executes the complete pinned native MouseExit caller, including its
hover callback, tween cleanup, captured highlight loop, Y movement and scale
setup. Metadata, callback, DOTween class initialization/Kill, Unity SetActive,
DOLocalMoveY, DOScale, SetId and the null gateway are supplied whole services.
Their implementation and runtime admission remain outside this evidence.

Source: `reverse_engineering/scripts/audit_card_interaction_mouse_exit.py`.
Report: `reverse_engineering/reports/f530404b0f3f_807de4a83df4_card_interaction_mouse_exit.json`.
The frozen audio machine supplies pinned PE/Dumper/header validation, runtime
mapping, native decode, full ABI/caller wrapper and return poisoning. The
MouseExit graph, body targets, services, snapshots and ordered model are
independently authored. The frozen audio audit remains unchanged.

## Complete entry and declarations

The exact method is `CardInteraction::public void MouseExit()`, stable ID
`tdi5468.m0006`, unique native RVA `0x35F810`. Its complete exclusive end is
`0x35FA2A`; the next managed entry is `0x35FA30`. All 538 bytes and 122
instructions are decoded from the verified entry, with raw section backing,
complete final instructions and trailing alignment verified. Dumper signature
is `void CardInteraction__MouseExit(CardInteraction_o* __this,
const MethodInfo* method)`, type signature `vii`.

Four unwind chunks cover `0x35F810..0x35F84B`, `0x35F84B..0x35FA18`,
`0x35FA18..0x35FA1E`, `0x35FA1E..0x35FA2A`. The first end is not the method
end. The root has unwind Flags 0; later chunks chain to it. There is no
method-local exception handler. Native null dispatch and authored unmapped
reads stop before throwing or unwinding. Normal completion checks stack cleanup
and all Win64 nonvolatile integer/XMM registers, including the saved/restored
XMM6 duration register.

Exact TypeDefIndex 5468 fields consumed are GameObject[] highlights `+0x28`,
Transform card `+0x30`, Transform shadow `+0x38`, string animationId `+0x40`,
Action onHoverExit `+0x68` and bool blockHover `+0x70`. The pinned Delegate
fields are invoke_impl `+0x18`, method `+0x28`, method_code `+0x40`. Unused
m_target `+0x20` deliberately differs from the dispatched code pointer.
The full owner byte window retains unconsumed fields as diagnostics.

Vector3 TypeDefIndex 6699 has binary32 x/y/z at offsets 0/4/8 and readonly
static oneVector at `+0xC`. Nominal Vector3 class `+0xB8` points to its authored
static record. DOTween class initialization reads the whole DWORD at `+0xE0`.
Class and static records have diagnostic sizes 256 and 128; these consumed
slots do not establish general runtime object admission or arbitrary layouts.

## Callback, cleanup and captured loop

Cold entry flag `0x288C13B` resolves exact DOTween metadata and the generic
SetId MethodInfo slot, setting its byte only after both resolver calls return.
Warm nonzero bytes, including `0x80` and `0xFF`, skip resolution. The separate
Vector3 helper flag is `0x288C102`.

Native code checks blockHover once. A nonzero byte returns immediately after
entry metadata. For zero, it captures current onHoverExit and optionally calls
its invoke_impl with method_code in RCX, method in RDX, retained R8/R9 and
native caller return `0x35F869`. Callback changes to blockHover do not re-run
the already completed gate. Callback changes to later owner fields are visible
to their subsequent loads; a replacement hover Action is used only by later
retained calls.

The code loads the current DOTween class and captures animationId before a
possible whole class initializer. Thus initialization can replace the owner's
current animation string while Kill still receives the earlier capture. The
initializer completes by setting the captured class DWORD to 1; class-slot
replacement affects later calls rather than changing that captured class.

Kill receives the captured animation pointer, complete=true and null
MethodInfo. `MOV DL,1` preserves the other 56 RDX bits; `XOR R8D,R8D` clears
all 64 R8 bits. Full incoming or supplied volatile RDX is independently checked
at every call. The supplied Int32 result, including poisoned high bits and
negative low-DWORD profiles, is discarded.

After Kill, native code captures the current highlights array in RDI. A
service replacing owner.highlights after this capture does not redirect the
loop. The length's low DWORD and each next element are reloaded from that
captured array on each iteration. Whole SetActive receives the current
nonnull element, full RDX=false and full R8=null. Callbacks can shorten or
extend the captured array's declared length within the authored three-element
window, replace upcoming elements or introduce a null element. Their completed
effects remain visible to the next iteration.

The loop first compares signed EAX=index against signed low-DWORD length,
exiting on JGE. It then compares the same EBX=index against the same low-DWORD
length using unsigned JAE, with no call or store between the two comparisons.
At each iteration EAX and EBX are equal and nonnegative. A negative length
exits at index zero; a nonnegative length exits before any index overflow.
Therefore the JAE gateway cannot be reached in this synchronous domain.
It is retained in the complete decode and excluded explicitly, rather than
given an invented fixture. Asynchronous mutation between the two comparisons
is not admitted. High-QWORD length poison and negative low-DWORD lengths test
the actual consumed width without creating huge physical arrays.

A null captured array or null visited element reaches the native null gateway
at `0x35FA1E`; its trailing int3 is not executed because the supplied exception
entry stops. The unsigned bounds call `0x35FA24` and its int3 `0x35FA29` are
unreachable under the demonstrated invariant. The audit's array contracts have
at most three elements; arbitrary enormous/concurrent runtime arrays remain
outside the fixtures.

## Movement, scale, captures and raw ABI

`0x50FB60` is the exact DOLocalMoveY service, not DOScale. The first call loads
current owner.card and supplies endY=positive zero. The second reloads current
owner.shadow and supplies endY=-15.5, raw bits `0xC1780000` from literal slot
`0x1F34CA8`. Both use duration binary32 0.2, exact bits `0x3E4CCCCD` from
`0x1F34B10`, retained in native XMM6. MOVSS memory loads clear remaining XMM
bits; XORPS makes the first endpoint full 128-bit zero. XMM2 receives the
duration, full R9 is zero snapping and the fifth stack argument is full QWORD
null MethodInfo. Full RCX/RDX/R8/R9, XMM0 through XMM6 and the fifth argument
are preserved in service-entry diagnostics, even when a register is not a
semantic parameter of that overload.

Each movement result, including nullable/aliased supplied nominal records, is
passed to exact generic SetId at `0x6BC9D0`. Its RDX reloads current
owner.animationId and R8 reloads current generic MethodInfo slot. Thus class
initialization, movement or prior SetId effects can change the next ID without
changing Kill's earlier captured ID. SetId's returned pointer is discarded.
The exact metadata name is
`Method$DG.Tweening.TweenSettingsExtensions.SetId<TweenerCore<Vector3, Vector3, VectorOptions>>()`;
Dumper MethodAddress is 0, so this is not promoted into a game-owned method.

`0x513BF0` is exact DOScale. Before each possible Vector3 metadata resolution,
native code captures card or shadow. A resolver can replace the owner field
while the current scale still uses that earlier capture. It reloads the
current Vector3 class and static block after resolution, loads oneVector's
first eight bytes with MOVSD and its final DWORD, then copies exactly 12 bytes
to the stack by-reference argument. The adjacent fourth DWORD remains intact.
XMM0's upper 64 bits clear; full XMM2 receives duration and full R9 is null
MethodInfo. Vector bytes with NaN, negative zero and infinity are preserved
without float conversion or decimal rounding.

The first Vector3 resolver normally warms the shared flag, so the second
resolution is skipped. An authored reached SetId mutation can clear that flag
and make the second complete resolver path execute. Fixtures show that its
shadow capture also precedes resolver effects. Each scale's current SetId ID
and token reloads are independent of the captured transform and vector copy.

Supplied transform/tween aliases and nullable outputs are diagnostic service
contracts. They do not prove that Unity or DOTween admits those objects. There
is no Image call in this native method. Native null owner/class/static faults
are captured at their exact read instructions, with all earlier completed
service effects retained and no runtime unwinding.

## Full model and exact prefixes

The independent model starts from authored initial bytes, options and raw
incoming registers. It predicts every service event, cumulative ordinal,
decoded caller return, complete pre-effect snapshot, reached mutations,
histories, flags, native stack vector copy, saved XMM6 and fifth argument,
guard/fault residue and final bytes. It never derives expected effects from
current emulator memory. All 37 nominal windows are included in every
snapshot. The native caller stores no owner field. Retention permissions are
per invocation and cover only exact reached authored mutation ranges plus a
completed captured class-initialization DWORD. The two literal DWORDs are
rechecked after every call, and the vector argument's adjacent DWORD is retained.

The suite includes 58 standalone cases and eleven retained sequences with
33 calls. Cases cover cold/warm flags, blocked bytes, callback/class-init
gates, nullable storage, array lengths/elements, ignored Kill return bits,
aliased/nullable tween results, unusual vector bits and service mutations.
Seven retained sequences stop at metadata, callback, class initialization,
SetActive, movement, SetId or scale, then resume with retained state. Other
sequences change actions, arrays, flags and supplied tween results. Every
preceding final snapshot equals the next initial snapshot exactly.

Seven baselines produce 67 stopped prefixes, including cold normal paths,
blocked early return, null arrays/elements, changed captured arrays, the
second Vector3 metadata path and actual native storage faults. Every reached
supplied entry stops before its effects. Each entire event sequence equals the
matching baseline prefix, and final state equals the corresponding entry
snapshot. The independent model validates all standalone, retained, baseline
and stopped rows. There are 165 modeled rows overall. Native execution covers
119 of 122 body instructions and 129 native/service addresses; the only
unexecuted instructions are the null trailing trap and the proved unreachable
bounds call/trap. Forty selected operand assertions supplement full
range, declarations, literal and service-signature validation.

The report losslessly pools complete raw memory using `pool_memory` from
`audit_character_oracle_reveal_join.py`, then complete snapshots using
`pool_snapshots` from `audit_report_snapshots.py`. Both expanded forms are
asserted equal to the original full report. Decoding order is
`expand_memory(expand_snapshots(report))`; no fields or prefixes are dropped.
No full native body bytes or disassembly exports are included.

Final freeze requires two successful syntax-preceded native producers with
byte-identical output and all 36 RE infrastructure tests. A matching first
private producer remains in the pinned artifact directory for peer review.

## Scope remaining supplied

Callback/delegate/runtime admission, DOTween service algorithms and state,
Unity GameObject behavior, tween scene effects, exception throwing/unwinding
and asynchronous scene mutation remain supplied. Other CardInteraction
methods, the folded constructor and lifecycle-to-MouseExit composition are
separate families; this audit establishes the previously supplied MouseExit
body only.
