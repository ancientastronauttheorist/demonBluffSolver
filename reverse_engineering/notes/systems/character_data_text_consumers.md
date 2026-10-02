# CharacterData flavor, translated text and name writes

Pinned build `f530404b0f3f_807de4a83df4`. The audit executes six exact native
CharacterData callers in one authored Data/String/array/CharacterLoc graph.
Whole Random.Range, CharacterLoc translation, StringHelper conversion,
Unity Object.get_name and reference-barrier implementations remain supplied.
GetDescription is a separate caller and does not gain evidence here.

Producer: `reverse_engineering/scripts/audit_character_data_text_consumers.py`.
Report: `reverse_engineering/reports/f530404b0f3f_807de4a83df4_character_data_text_consumers.json`.

## Exact scope and consumed storage

| Stable method | Entry | Complete body end, exclusive | Next managed entry |
| --- | --- | --- | --- |
| `tdi5845.m0003` GetFlavorText | `3B4CA0` | `3B4CEB` | `3B4CF0` |
| `tdi5845.m0004` GetTranslatedName | `3B4D60` | `3B4D97` | `3B4DA0` |
| `tdi5845.m0005` GetIWasTranslated | `3B4D10` | `3B4D47` | `3B4D50` |
| `tdi5845.m0016` GetIfLies | `3B4D50` | `3B4D5E` | `3B4D60` |
| `tdi5845.m0017` GetHints | `3B4D00` | `3B4D0B` | `3B4D10` |
| `tdi5845.m0019` UpdateCharacterName | `3B5070` | `3B5094` | `3B50A0` |

Each exact declaration has one unique managed entry. Complete unwind ranges,
leaf absence of unwind entries, file backing, following entries, CC padding,
full decoder consumption and exact Dumper signatures are verified. Tracked
body fingerprints contain byte lengths, instruction counts and SHA-256 only;
complete native bytes/disassembly remain private. Twenty-two authored operand
assertions pin the critical captures, loads, unsigned branch, stores and tails.
Both flavor guard traps are included in the complete decode and explicitly
remain unexecuted. No folded alias is promoted.

Exact CharacterData TypeDefIndex 5845 fields are characterName `+28`,
flavorText `+68`, additionalFlavorTexts `+70` (`string[]`), hints `+78`, ifLies
`+80` and translation `+148` (`CharacterLoc`). The two Data windows are 512
bytes; arrays/classes are 256 bytes and remaining opaque records 128 bytes.
These complete authored windows are diagnostics, not object-size or runtime
admission assertions. Unused sentinel references and buffer contents retain
their full bytes without being treated as valid managed objects.

## Native chronology

GetFlavorText captures additionalFlavorTexts in RBX. A null array reaches the
native null gateway. An exactly zero QWORD at array `+18` returns the current
flavorText pointer directly, including null. No RNG callback occurs on this
empty branch, so a configured RNG mutation has no effect or write allowance.

A nonzero QWORD requests supplied integer Random.Range with full ECX zero,
zero-extended EDX from the length's low DWORD and full R8 zero MethodInfo.
The result's EAX is sign-extended by CDQE. The caller then compares that low
DWORD against the captured array's reloaded length DWORD using unsigned JAE.
Thus a callback replacing or clearing the Data's array field does not change
this invocation's captured array, while a changed length or selected element
inside that array is observed. An out-of-bounds authored result reaches the
exact bounds gateway before element access. Negative result DWORDs and a
nonzero high QWORD with zero low length are explicit diagnostic probes; they
do not assert legal Unity RNG output or arbitrary managed array admission.
Within the bounded three-element corpus, selected pointer/null and aliases
are returned exactly. No weighted RNG distribution is reconstructed here.

GetTranslatedName and GetIWasTranslated capture translation `+148`, and pass
the original locale pointer, including null, to their exact supplied
CharacterLoc methods with zero R8 MethodInfo. Any nonnull returned pointer is
returned directly. Null translation or null translation output falls back to
supplied Unity Object.get_name using the original Data owner, full EDX zero,
and a native tail jump. A callback changing translation does not replace the
captured receiver or original fallback owner. The fallback is Unity's object
name; this body does not read characterName or iWasName as that fallback.

GetHints and GetIfLies load their exact String pointer and tail-call supplied
StringHelper.ConvertTextToTextWithTooltips, clearing full EDX and preserving
unused R8/R9 residues. Both nullable input and nullable result are represented.
Actual conversion/markup semantics remain outside this caller audit.

UpdateCharacterName calls supplied Object.get_name, captures its full returned
pointer in RAX, stores it at the original owner's characterName `+28`, and
tail-calls the reference barrier with the exact field address and stored
value. A name-service callback that first changes characterName is overwritten
by the native store; a barrier callback can subsequently change it. A null
owner with an explicitly supplied name result reaches the exact native write
fault at `3B5087`, retaining the earlier completed service history. This is
not a claim that Unity accepts that null receiver in a real scene.

## Independent comparison and retained calls

Every supplied and native-guard event retains full RCX/RDX/R8/R9, exact decoded
call/jump site, full stack return address and complete service-entry snapshot.
Whole supplied returns poison volatile integer/XMM registers. Native returns
verify the restored stack, eight integer nonvolatiles and XMM6-XMM15. Void
UpdateCharacterName's observed RAX is diagnostic supplied-barrier residue,
not a managed return value.

An independent ordered semantic model starts solely from each initial raw
buffer, authored options and incoming registers. It predicts every full event
snapshot, raw ABI/caller, native return/guard/fault disposition, complete final
bytes, logical field/array projections and prior/native service histories.
It does not read native output to generate expected values. Retention permits
only completed authored eight-byte callback writes and the exact reached
native characterName store. Stopped or skipped services receive no effects or
write permissions. Every other diagnostic byte remains unchanged.

The corpus contains 159 cases: 130 normal returns and 29 native guard/fault
outcomes. Four retained three-call sequences preserve full storage/history;
each next initial snapshot equals the previous final snapshot. One sequence
changes the Data's array during the first RNG call, proves the first call still
uses the old array, and proves later invocations load the other array. Inputs,
array elements and supplied result pointers include physical String aliases.
Twelve baselines yield 16 exact stopped prefixes; every stopped event list and
full final snapshot equal the baseline prefix and stopped entry snapshot.

All 77 nontrap instructions execute out of 79 decoded. Unexecuted traps are
`3B4CE4` and `3B4CEA`. No exception unwinding, allocation/admission, renderer,
real localization/conversion, engine input or scheduling is inferred.

## Reproduction and storage

Compile the script with `python -m py_compile` before execution. Set PYTHONPATH
to the private `python-emulation` dependency directory. Invoke the script with
positional pinned game and Dumper directories and `--output`.

Raw memory is pooled before complete snapshots. Expand with
`audit_character_oracle_reveal_join.expand_memory` after
`audit_report_snapshots.expand_snapshots`. Both codec round trips and explicit
dual expansion equality retain every field and byte. There are 29 raw memory
blobs and 199 full snapshots; no native behavior or fixture is discarded.

Two separately syntax-preceded successful native producer processes emitted
byte-identical 733,652-byte reports, SHA-256
`946a1740049ba1a4e535b6d5aa2157b7a6de628c98b49301a6f154e5a7d6762c`.
Private peer:
`B:/CodexTools/DemonBluffReverseEngineering/artifacts/f530404b0f3f_807de4a83df4/character_data_text_consumers.peer.private.json`.
Read-only peer review found no blockers after the alias/captured-array,
unsigned-branch and dual-codec checks. Rust/bridge/simulation and live gameplay
are unaffected by this offline native caller family.
