# CharacterData skin lookup callers

This audit executes the exact `LoadSkin(string)` and
`CheckIfSkinUnlocked(string)` native callers offline. Whole iterator, string
comparison, SkinData unlock, metadata, barrier, guard and rethrow services are
supplied. A separate ordered model compares all events, complete physical
snapshots and final volatile registers before lossless report pooling.

## Exact declarations and complete extents

The owner is `CharacterData : ScriptableObject, ICharacterLocData, ICardData`,
TypeDefIndex 5845. Exact declarations, signatures and immutable symbol keys are
bound separately; no generic/shared service alias is classified by RVA.

| ID | Method | Type signature | Body | Next managed entry | Instructions |
| --- | --- | --- | --- | --- | --- |
| tdi5845.m0012 | LoadSkin(string skinId) | viii | `[0x3B4F10, 0x3B5061)` | `0x3B5070` | 77 |
| tdi5845.m0013 | CheckIfSkinUnlocked(string skinId) | iiii | `[0x3B43B0, 0x3B4511)` | `0x3B4520` | 84 |

Each body occupies one matching complete unwind range, with EH/UH flags 3 and
handler RVA `0x30CD28`; neither has chained method chunks. All file-backed bytes
decode through the final one-byte `int3`, and each body is followed by exactly
15 `CC` padding bytes. Repository evidence records byte lengths, instruction
counts and SHA-256 rather than complete proprietary bytes/disassembly. Twenty-two
selected operand assertions and all consumed call-site lists are authored in
the source.

The exact fields are `CharacterData.currentSkin` at `+0xC0`,
`CharacterData.skins` at `+0xC8`, and `SkinData.skinId` at `+0x18`.
SkinData is the exact TypeDefIndex 5945 declaration. The generic enumerator's
field names are pinned to TypeDefIndex 1509 and its native reference-type layout
to the hash-pinned Dumper header: 24 bytes, list pointer at 0, index DWORD at 8,
version DWORD at 12 and current pointer at 16. The generic dump's zero offsets
are not treated as the specialized native offsets.

## Iterator ABI and ordinary behavior

Both methods capture the supplied query reference and owner before metadata
services. A zero method-specific flag initializes the same four exact
SkinData-enumerator MethodInfo slots in order: Dispose, MoveNext, get_Current,
GetEnumerator. Native code writes flag 1 only after all four services complete.
Any nonzero flag skips the prefix. The get_Current MethodInfo is initialized,
but current values are read directly from the enumerator; its method body is
not called.

GetEnumerator's Win64 ABI is a hidden 24-byte output pointer in RCX, current
skins List receiver in RDX and exact generic MethodInfo in R8. The native caller
copies its output from frame `+0x28` into the active enumerator at frame `+0x40`
using a 16-byte MOVUPS and an eight-byte MOVSD. It then clears the exception
slot at frame `+0x28` and stores the active enumerator address at frame `+0x30`.
The original hidden-output storage and complete active state remain visible in
every snapshot. Supplied GetEnumerator RAX bits are independent inputs; the
caller copies the produced storage even when those bits are zero.

MoveNext receives the active state pointer in RCX and reloaded exact MethodInfo
in full RDX. Only AL selects whether to continue. The supplied service's
ordered entries, 24-byte output, cursor/current writes and raw return bits are
explicit diagnostic inputs; actual generic collection semantics are not
established here. Its default fixture captures entries from the supplied List
at GetEnumerator time. Replacing the owner's skins reference afterward does
not replace that captured service input.

LoadSkin first clears currentSkin, invokes a reference barrier with that field
address and full zero RDX, then reloads the owner's skins reference. Consequently
a barrier callback can change which List is enumerated. For every current
entry, it captures the SkinData pointer, reads that skin's current skinId, and
supplies String.op_Equality with skinId in RCX, captured query in RDX and full
zero R8 MethodInfo. Nonzero AL stores the captured skin pointer into currentSkin
and invokes its barrier. The loop continues after a match: every match is
stored, so the last matching supplied entry wins unless a later callback changes
the field. Zero matches leave the initial clear or a reached callback's value.

CheckIfSkinUnlocked does not clear currentSkin. It supplies String.op_Equality
with captured query in RCX, current skinId in RDX and zero R8. The first nonzero
AL match supplies SkinData.CheckIfUnlocked with the already captured skin in
RCX and full zero RDX MethodInfo. After that service, native MOVZX EDI,AL
captures the raw byte, calls Dispose, then MOVZX EAX,DIL zero-extends that byte
into the final return. Diagnostic `00`, `01`, `80` and `FF` outputs are retained
exactly; no Boolean canonicalization is imposed.

With no match, CheckIfSkinUnlocked calls Dispose then uses XOR AL,AL. Its final
low byte is zero while the supplied Dispose upper RAX bits survive. LoadSkin's
void return likewise carries supplied Dispose RAX residues. Those register
patterns are ABI evidence, not additional public return values.

## Capture, reload and retained state

Every boundary records raw RCX/RDX/R8/R9, all seven integer volatile registers,
XMM0 through XMM5, native caller return address, method and entry mode. Services
poison all volatile registers independently of supplied RAX outputs. The model
accounts for MOVUPS/MOVSD effects on the first MoveNext entry and compares all
final volatile integer/XMM values on returns, service stops and native faults.
Normal entry returns preserve Win64 stack discipline, all eight integer
nonvolatile registers and XMM6 through XMM15.

Callbacks replace generic MethodInfo roots, owner List references, active
enumerator current pointers, SkinData IDs and currentSkin. A MoveNext callback's
current pointer is captured afterward. A String equality callback changing that
active pointer does not change the captured skin used for the next store or
unlock call. Repeated references to the same skin reload its changed skinId on
the next iteration. Barriers can overwrite just-stored currentSkin; Dispose
callbacks occur after the lookup's final decision. Owner receivers, duplicate
skin entries and cleanup receiver aliases remain explicit nominal identities.

Complete diagnostic storage includes both owners, supplied Lists and arrays,
skins, strings, MethodInfo records, opaque exception storage, a second
enumerator and the 64-byte native scratch window. These are diagnostic windows,
not asserted managed object extents. Only actual reached native stores and
completed supplied writes permit byte changes. Metadata roots and both flag
bytes are checked separately, including phases skipped by nonzero flags.

Ten retained four-call sequences preserve one physical state. They cover
success, a stopped equality prefix, recovery and later callbacks; two sequences
also start with a failed cold metadata prefix over a null List, restore the List
in the next reached metadata callback, then continue with changed queries.
Every prior final snapshot equals the next initial snapshot.

## Cleanup diagnostics and exclusions

The native cleanup entries are `0x3B501B` for LoadSkin and `0x3B44CC` for
CheckIfSkinUnlocked. Ordinary entry does not reach them. Separate direct probes
supply a synthetic existing frame, saved nonvolatile registers, active
enumerator receiver and null/live exception pointer. They execute the actual
cleanup instructions, call supplied Dispose, reload the exception pointer from
frame `+0x28`, and either take the normal epilogue or call supplied rethrow.
Dispose callbacks can clear a previously live exception or introduce one.
The receiver can alias the separately supplied enumerator. Synthetic frame
return checks verify the authored frame only; actual managed exception dispatch,
stack unwinding and the handler body do not execute.

Native null List/current guards and owner read/write faults are recorded with
full preserved state; guard and rethrow gateways are explicitly nonreturning
stops. Of 161 decoded instructions, 153 execute. Eight exclusions are pinned:
the post-guard/rethrow NOPs or traps, plus CheckIfSkinUnlocked's second null
captured-skin guard and its following NOP. That second guard cannot be reached
through normal service boundaries preserving RDI: the captured pointer passed
the preceding null check and equality does not replace it. Exclusions do not
claim an executed unwind or a complete runtime exception implementation.

## Final verification

The corpus has 352 cases, ten retained four-call sequences, 22 baselines and
151 exact supplied-boundary stops. Every stopped result equals the complete
baseline event prefix and that boundary's full pre-effect snapshot. Two
syntax-preceded independent producers yield byte-identical reports, and all 36
reverse-engineering infrastructure tests pass.

Source: `reverse_engineering/scripts/audit_character_data_skin_lookup.py`.
Report: `reverse_engineering/reports/f530404b0f3f_807de4a83df4_character_data_skin_lookup.json`.
Private peer report:
`B:/CodexTools/DemonBluffReverseEngineering/artifacts/f530404b0f3f_807de4a83df4/character_data_skin_lookup_peer.json`.

Preferences persistence, skin unlock policy, actual generic iterator/string
bodies and managed exception machinery remain separate boundaries.
