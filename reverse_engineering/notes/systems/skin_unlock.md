# SkinData unlock callers

This family executes the two actual installed SkinData callers offline. Save
get/set, generic List Contains/Add, metadata and null guards are whole supplied
services. No live process or preference file is used or changed.

Source: `reverse_engineering/scripts/audit_skin_unlock.py`.
Report: `reverse_engineering/reports/f530404b0f3f_807de4a83df4_skin_unlock.json`.

## Binding and native extent

| Method | Inventory | Entry | Complete end, exclusive | Instructions |
| --- | --- | --- | --- | --- |
| CheckIfUnlocked | `tdi5945.m0001` | `3EBAF0` | `3EBB8E` | 42 |
| UnlockSkin | `tdi5945.m0002` | `3EBE50` | `3EBEF0` | 43 |

Exact SkinData TypeDefIndex 5945 declarations, ordinals within its four methods,
complete ScriptMethod signatures and type signatures are checked. Each body has
one complete unwind range with flags zero. The first body is 158 bytes and has
two CC alignment bytes before the next managed entry. The second is 160 bytes
and ends at the next managed entry. Raw backing, complete instruction boundaries,
final one-byte INT3, selected operands and complete gateway call-site lists are
verified. Tracked evidence retains 16 selected assertions and body fingerprints;
complete native exports remain private.

Consumed managed fields are SkinData.skinId `+18`, SkinData.unlockWith `+68`,
and SavedSkins TypeDefIndex 5551 ids `+10`. String length is a signed DWORD at
`+10`, bound to the generated System_String header. Three metadata roots are
AutoUnlocked_TypeInfo and the exact List<string> Contains/Add MethodInfo slots.
Each caller has its own cold flag and two ordered metadata calls. Only after both
complete does it write flag 1; all nonzero flags skip the prefix.

## Check and mutation ordering

CheckIfUnlocked first examines the current unlockWith. A null value skips the
inline class gate. Otherwise it loads that object's class, the current
AutoUnlocked root, the root's raw byte at `+130`, and the candidate class's raw
byte at `+130`. A candidate byte below the root byte skips the indexed test.
Otherwise it loads the candidate's `+C8` pointer and compares the QWORD at
`pointer + root_byte*8 - 8` with the captured AutoUnlocked root. Equality returns
through MOV AL,1 without calling save services.

The runtime name of the native class byte is unresolved. Independent generated
header layout places typeHierarchyDepth at `+12C` and naturalAligment at `+130`.
The producer retains this discrepancy and treats the decoded byte/index operands
as opaque native storage. It does not relabel the generated header or claim a
runtime typeHierarchyDepth binding. Fixtures use bounded nonzero indexing bytes;
zero-byte indexing and runtime class admission are outside this corpus.

When the inline gate does not match, CheckIfUnlocked calls whole
SavesGame.get_UnlockedSkins with zero MethodInfo, loads the returned ids List,
reloads the original owner's skinId, and calls whole Contains. Null saved/List
results reach the supplied null guard. Only AL controls the final truth branch.
A true branch writes AL=1 while preserving the upper RAX bits; a false result
retains the supplied Contains residue. The full return register is compared.

UnlockSkin captures the original owner and the returned SavedSkins identity.
It loads ids and the owner's ID for Contains. A true AL skips Add and save.
After false AL it reloads the original owner's current skinId. Null reaches a
guard; signed string length at most two skips Add/save. A positive length greater
than two reloads ids from the already captured SavedSkins, loads the current Add
MethodInfo, and requests Add. It then saves the original captured SavedSkins with
zero MethodInfo, even if an Add callback changed that record's ids reference.
Its void RAX is only a supplied-service residue.

The supplied Add gateway is `2EB0`. It has no ScriptMethod declaration. Evidence
binds the exact caller, List<string>.Add MI root and three selected raw-backed
entry instructions; it does not invent a managed declaration or claim that the
gateway's body executes. The authored Add service appends a QWORD and increments
the diagnostic count. Its other behavior, allocation/version policy, generic
implementation and barriers remain excluded. Getter/Contains/Add/set callbacks
and metadata/MI replacements are explicit completed fixture effects. Wrong-site
callback plans and stopped services have no such effects.

## Physical state and verification

There are 27 complete diagnostic windows: owners, unlock/classes, hierarchy,
SavedSkins, Lists/arrays, strings, MethodInfo records, metadata slots/flags and a
112-byte native stack window. The stack spans entry SP minus 40 through entry SP
plus 30, hexadecimal, including return cells, saved RBX/RDI and untouched bytes.
These are diagnostic storage sizes rather than inferred managed object extents.

Every native entry, supplied boundary, stop and final state retains all authored
GPR/XMM registers. Seven volatile GPRs and XMM0–5 are compared; service boundaries
also retain raw four arguments, exact caller/SP and per-invocation ordinals.
XMM values use canonical 32-digit hex, preserving all 128
bits in future JSON consumers. Successful supplied services explicitly poison
volatile registers; normal outer returns preserve eight integer nonvolatiles,
XMM6–15 and stack discipline.

A separate byte-addressed instruction model starts from initial complete bytes
and incoming registers. It independently evaluates addresses, widths, AL upper
bits, DWORD zero extension, signed/unsigned comparisons, branches, calls, stores
and returns. It uses authored service contracts but never observed native events
or final values. The graph write hook records reached native stores; every event,
all complete physical bytes, histories, native writes and final ABI must equal
the independent model. No broad changed-byte allowance copies native outputs.

The frozen corpus contains 91 cases, five retained three-call sequences, 43
baselines and 183 full stopped prefixes. All 83 non-trap instructions execute;
only terminal guard INT3 at `3EBB8D` and `3EBEEF` are excluded. Every stopped row
matches the whole baseline event prefix and exact pre-effect storage/register
snapshot. Retained final/initial snapshots match completely, including stack and
chronological native-entry/write/service history. Cold metadata recovery and
cross-method warm flags are checked without rollback.

Both separately syntax-preceded final producers succeeded and matched byte for
byte. The report is 3,744,387 bytes, SHA-256
`b2cff80aa8f0db06768e7d0ca181415201bf5b6f42f5e56639f76b4db5d8904a`.
The matching private peer is `skin_unlock.peer.private.json` beneath the pinned
build artifact directory. Decode using
`audit_character_oracle_reveal_join.expand_memory(audit_report_snapshots.expand_snapshots(report))`.
Each producer checks exact combined lossless expansion. Local primitive checks
cover AL/EDX writes, signed/unsigned flags and canonical register round trips.
Actual saves, collection algorithms, runtime class layout/admission and exception
unwinding remain outside this bounded caller evidence.
