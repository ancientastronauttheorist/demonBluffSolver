# CardInteraction RevealInteraction and TapCard audio callers

This audit executes both complete native card audio callers. The reconstructed
behavior is metadata gating, one float RNG draw, AudioEvents/delegate reloads,
integer sound selection and tail dispatch. Metadata resolution, the whole float
Random.Range service and the whole Action<ESFX> callback remain supplied.

Source: `reverse_engineering/scripts/audit_card_interaction_audio.py`.
Report: `reverse_engineering/reports/f530404b0f3f_807de4a83df4_card_interaction_audio.json`.
The frozen DeckCharacter surface machine supplies pinned PE/Dumper validation,
native runtime mappings, complete-range decoding and volatile return poisoning.
The frozen Character InitReward helper decodes native caller returns. The audio
graph, full-state fixtures, services and independent model are newly authored.

## Exact declarations, bounds and fields

| Method | Stable ID | Native body, exclusive end | Next managed entry | Bytes / instructions |
| --- | --- | --- | --- | --- |
| `CardInteraction::public void RevealInteraction()` | `tdi5468.m0010` | `0x35FF90..0x35FFF8` | `0x360000` | 104 / 22 |
| `CardInteraction::private void TapCard()` | `tdi5468.m0009` | `0x360000..0x360068` | `0x360070` | 104 / 22 |

Each entry has exactly one Dumper ScriptMethod row, a void return, owner pointer
and hidden MethodInfo pointer, and type signature `vii`. Neither is a shared-RVA
alias. The entire body is decoded from its verified entry; each final ret and
both nullable-delegate paths are included. Raw backing and alignment padding up
to the next managed entry are verified. Each body has one matching unwind range
with Flags 0 and no method-local exception handler.

Exact AudioEvents TypeDefIndex 5524 declares static `Action<ESFX>
OnPlaySfxOneShot` at offset 0. Its metadata slot points to a supplied class
record whose static pointer is at `+0xB8`. The independent class and static
records have diagnostic sizes 256 and 128 bytes. The complete bytes also retain
the unused other audio-event channels. Neither class initialization nor an
audio-event registration body is executed.

Exact ESFX TypeDefIndex 5467 has Int32 storage and values CardClick 110 and
CardTap 120. Pinned Delegate fields are invoke_impl `+0x18`, m_target `+0x20`,
method `+0x28` and method_code `+0x40`; Action<ESFX> inherits through
MulticastDelegate. The emitted callback uses method_code, method and invoke_impl.
The unrelated m_target value is deliberately distinct and unused. Nominal
delegate records and callback addresses establish authored diagnostics, not CLR
delegate admission or invocation-list behavior. Source validates dump.cs,
script.json and il2cpp.h against the pinned extraction manifest.

## Exact chronology and float ABI

RevealInteraction uses metadata flag `0x288C13E`; TapCard uses `0x288C13D`.
A zero byte resolves the single AudioEvents metadata slot. The native byte is
set to 1 only after that supplied resolver returns. Nonzero bytes `0x80` and
`0xFF` skip resolution. The two flags remain independent across retained calls.

Both methods load literal floats into XMM0/XMM1 and clear full R8 using XOR
R8D before the exact float overload `UnityEngine.Random.Range` at `0x1C86640`:
`float UnityEngine_Random__Range(float minInclusive, float maxInclusive,
const MethodInfo* method)`, type signature `fffi`.

| Method | XMM0 minimum slot / bits | XMM1 maximum slot / bits |
| --- | --- | --- |
| RevealInteraction | `0x1F34C60` / `0x3F19999A` | `0x1F34B18` / `0x3F800000` |
| TapCard | `0x1F34C64` / `0x3F4CCCCD` | `0x1F34C74` / `0x3FB33333` |

Their actual binary32 values are approximately 0.6000000238418579 / 1.0 and
0.800000011920929 / 1.399999976158142. The exact bits and raw backing are
asserted from RIP-relative operands; no decimal re-rounding is used. Memory
MOVSS clears the remaining 96 XMM bits. Full raw RCX/RDX/R9 survive as entry or
resolver-return values at RNG entry; they are not semantic float arguments.
Every service event includes full RCX/RDX/R8/R9, six 128-bit volatile XMM
registers, cumulative service ordinal, decoded caller return and a complete
pre-effect authored-state snapshot.

The supplied RNG returns arbitrary full XMM0 bits, including zero, negative
zero, infinity, NaN and high-bit poison profiles. The native callers never use
that result for a branch or semantic argument. Its 128 bits survive in the raw
delegate-entry diagnostic. It is not a third callback argument or an authored
volume parameter. RNG history is retained, so the draw itself remains observable.

After RNG, native code reloads the metadata class, its current static block and
the current OnPlaySfxOneShot pointer. RNG mutations can therefore replace the
class, static block or action; clearing the action reaches the ordinary ret.
For a nonnull action, native code captures method `+0x28` in R8, zero-extends
EDX to full RDX 110 or 120, captures method_code `+0x40` in RCX, restores the
stack and tail-jumps via invoke_impl `+0x18`. R9 retains supplied RNG poison.
The callback sees the original authored caller return, not a synthetic native
return site. A callback mutation happens after these captures and only affects
later retained calls. Null or aliased nominal method/code records are explicit
diagnostic callback contracts, not proof of runtime validity.

There is no owner field read or owner dereference in either body. Null owner
profiles complete under these supplied contracts; this proves the native
instruction behavior, not that Unity would issue such a call. A null class or
static block instead faults at the actual native read, after completed RNG
effects. These authored storage faults are explicitly stopped without
executing exception unwinding.

## Full-state model, retention and validation

The model begins only from initial authored bytes, options and incoming raw
registers. It independently predicts flags, all complete ordered service
events, their ABI and snapshots, effect histories, partial completion, faults,
final bytes and cumulative counts. It does not infer expected effects from
current emulator memory or truncate snapshots to selected fields.

All 13 diagnostic storage windows are included in every snapshot. Per-call
retention permissions cover only exact eight-byte writes by a reached authored
mutation. Native callers write only their reached metadata flag; the model
checks both flags and the metadata slot independently. Owner, class, static,
delegate, code and method bytes remain unchanged outside those explicit ranges.
The four literal DWORDs are checked again after each call. Native completion
verifies stack cleanup plus all Win64 nonvolatile integer and XMM registers.

The suite contains 82 standalone cases and seven retained sequences with 25
calls. Four sequences cover mixed methods, independent cold flags, null owner,
post-callback class/action changes and RNG class replacement. Three sequences
stop at metadata, RNG or callback, then resume another call with retained
bytes, histories and ordinals. Every preceding final snapshot equals the next
initial snapshot exactly.

Ten baselines cover both methods with normal callback, null callback, changed
callback ABI, null class and null static storage. Each reached supplied entry
has a stopped replay: 24 complete prefixes. A stopped event is recorded before
effects, its entire event sequence equals the baseline prefix, and its final
snapshot equals the matching service-entry snapshot. The independent model
also verifies all stopped and retained rows. Across all fixtures, 44 of 44
native instructions and 48 native/service addresses execute. There are no
unexecuted body stubs or traps. Sixteen selected operand assertions supplement
complete-body, branch, flag, literal and declaration checks.

The report losslessly interns memory hex using `pool_memory` from
`audit_character_oracle_reveal_join.py`, then complete snapshots using
`pool_snapshots` from `audit_report_snapshots.py`. Both layers assert exact
expanded equality to the full original report. Decoding order is
`expand_memory(expand_snapshots(report))`; hash validation and independent
container reconstruction are preserved. No fields, prefixes or fixture cases
are removed, and no native body byte/disassembly exports are included.

Final verification requires two successful syntax-preceded native producers
with byte-identical outputs and the 36 reverse-engineering infrastructure tests.
The frozen report is copied into its canonical pinned build path only after
producer success. The private peer output retains the matching first producer.

## Explicit limits

Whole metadata, RNG, delegate invocation, audio playback, CLR/runtime storage
admission and exception unwinding remain supplied. Other CardInteraction
methods and its folded constructor are excluded. Actual audio callback targets
may use the integer sound ID and additional state; their implementation is not
established by these callers or the deliberately retained XMM diagnostics.
