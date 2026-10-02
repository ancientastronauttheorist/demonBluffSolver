# CharacterLoc translated-name callers

Pinned build `f530404b0f3f_807de4a83df4`. These two actual native callers
consume explicit whole FindLocaleLoc and String.IsNullOrEmpty services. Neither
locale search nor the managed string implementation executes in this family.

Source: `reverse_engineering/scripts/audit_character_loc_text.py`.
Report: `reverse_engineering/reports/f530404b0f3f_807de4a83df4_character_loc_text.json`.

| Exact method | Stable ID | Complete native body | Next managed entry |
| --- | --- | --- | --- |
| CharacterLoc.GetIWasTranslated | tdi5972.m0003 | 3F5690..3F56C7 | 3F56D0 |
| CharacterLoc.GetTranslatedName | tdi5972.m0002 | 3F5920..3F5957 | 3F5960 |

Each unique binding has exact `iiii` Dumper signature with CharacterLoc owner,
localeCode String and hidden MethodInfo, returning String. Each complete body
has one unwind range, Flags 0, 55 file-backed bytes, twenty instructions and
two ordinary return paths. Nine trailing CC alignment bytes are excluded from
each body. Full raw extent, following managed entry, decoder consumption,
body SHA-256 and twenty selected operand assertions are verified. Full body
bytes and disassembly remain private.

Exact LocaleLoc TypeDefIndex 5969 storage has localeCode at +10, translatedName +18,
iWasTranslated +20 and entries +28. Every fixture preserves fourteen complete
nominal diagnostic windows: 256 bytes for class records and 128 for all other
records. These sizes are diagnostic contracts, not managed object extents.
Unused owner, class, entries and String sentinel bytes remain represented.

Both methods clear full R8 before supplied FindLocaleLoc at 3F5540, preserving
owner RCX, localeCode RDX and unused raw R9. Null owner and localeCode probes
complete under that explicit supplied service; the getter itself never reads
owner storage. This does not establish that actual FindLocaleLoc admits null
owners. The returned LocaleLoc is captured in RBX. A null result returns full
RAX zero without an emptiness request.

For a nonnull result the caller loads its selected field into RCX, clears full
EDX and calls supplied String.IsNullOrEmpty at F76390. Only AL determines the
branch: a nonzero low byte returns full zero; zero AL reloads the selected
field from the captured LocaleLoc and returns that pointer, including null.
The input text passed to the emptiness service need not equal the later
returned text after a supplied callback changes that field. Native code does
not rerun search or replace its captured record. Boolean high-bit poison and
0x80 low bytes explicitly verify consumed width, without claiming legal
managed Boolean representations beyond the authored diagnostic contract.

Callbacks from either service can change selected or unselected locale fields.
The supplied search output is a captured nominal identity, independent of its
callback writes. The later emptiness input reflects search callback changes;
the final getter load reflects emptiness callback changes. Aliased text and
nullable fields/results remain physical identities in one graph.

Every event includes cumulative ordinal, full RCX/RDX/R8/R9, seven volatile
integer registers, six volatile XMM registers, exact native site/return caller
and the complete pre-effect snapshot. Whole services poison volatile registers
and supply explicit RAX results. The independent model starts solely from
initial bytes, options and incoming registers; it predicts every complete
event, final bytes/history/counts, result, stopped disposition and final
integer/XMM residues. Expected state is not generated from emulator output.

Normal completion verifies stack cleanup, all eight integer nonvolatiles and
XMM6 through XMM15. The native callers write no object fields. Per-invocation
retention permits only completed reached eight-byte callback mutations; stopped
services have no effects. Each retained call starts from its predecessor's
exact final snapshot, including earlier native entries and service history.

The corpus contains 134 standalone cases, six retained three-call sequences,
six baselines and ten exact stopped prefixes. Two sequences stop at search or
emptiness then continue with retained history. Each stopped event list equals
the baseline's entire prefix, and its final state equals that boundary's
pre-effect snapshot. All forty decoded native instructions execute.

Raw memory is losslessly pooled before complete snapshots. Both codec layers
and their combined expansion assert full equality. Decode with
`expand_memory(expand_snapshots(report))`, using
`audit_character_oracle_reveal_join` and `audit_report_snapshots`. No cases,
fields or diagnostic bytes are removed.

Final freeze requires two separately syntax-preceded successful producers with
byte-identical output and the 36 RE infrastructure tests. The matching private
peer is `B:/CodexTools/DemonBluffReverseEngineering/artifacts/f530404b0f3f_807de4a83df4/character_loc_text.peer.private.json`.
Whole search/emptiness implementations, locale contents, runtime admission,
engine localization, exception unwinding and other CharacterLoc methods remain
open. Separate CharacterData caller evidence does not establish an actual
CharacterData-to-CharacterLoc composed native join.
