# Actual locale search and translated getter composition

This family executes exact CharacterLoc.FindLocaleLoc as a standalone caller,
separate direct synthetic cleanup diagnostics, and both actual translated-text
getters calling actual FindLocaleLoc. All executions use one physical graph.
Whole generic iterator, string, metadata, guard and rethrow services remain
supplied. A separate complete ordered model predicts events, snapshots, results
and volatile ABI before lossless report pooling.

## Exact declarations and complete bodies

| Stable ID | Exact declaration | Complete native body | Next managed entry | Instructions |
| --- | --- | --- | --- | --- |
| tdi5972.m0001 | private LocaleLoc FindLocaleLoc(string localeCode) | `[0x3F5540, 0x3F5684)` | `0x3F5690` | 76 |
| tdi5972.m0002 | public string GetTranslatedName(string localeCode) | `[0x3F5920, 0x3F5957)` | `0x3F5960` | 20 |
| tdi5972.m0003 | public string GetIWasTranslated(string localeCode) | `[0x3F5690, 0x3F56C7)` | `0x3F56D0` | 20 |

All three exact Dumper signatures have type signature `iiii`. CharacterLoc is
TypeDefIndex 5972 and LocaleLoc is TypeDefIndex 5969; both declaration blocks
are bounded by their exact closing brace. The consumed owner field is
`localizedLocs +0x20`. LocaleLoc fields are `localeCode +0x10`,
`translatedName +0x18`, `iWasTranslated +0x20`, and retained `entries +0x28`.

FindLocaleLoc occupies 324 file-backed bytes in one complete unwind range,
with EH/UH flags 3 and handler RVA `0x30CD28`. It has no chained range. Its
cleanup entry is `0x3F5645`; its final one-byte `int3` at `0x3F5683` is pinned.
Exactly 12 following `CC` bytes precede the next managed entry. Each getter has
55 bytes, one Flags 0 unwind range, nine following `CC` bytes, two return paths
and a pinned final one-byte RET. Repository reports retain SHA-256, byte lengths
and instruction counts, without full native bytes or disassembly. Thirty-seven
selected operand assertions and exact call-site lists are authored in source.

The frozen getter source class supplies its hash-pinned declarations/body setup
to the new family. Its source, note and report remain unchanged. Native getter
calls enter actual FindLocaleLoc at each getter's `start + 9`; no supplied
FindLocaleLoc boundary or report trace is spliced into this family.

## Actual search, capture and first match

FindLocaleLoc captures the input code and owner before metadata calls. A zero
method flag initializes four exact LocaleLoc enumerator MethodInfo slots in
order: Dispose, MoveNext, get_Current, GetEnumerator. Only after all four
services complete does native code set flag 1. Raw nonzero flags skip that
prefix. The get_Current slot is initialized but its body is not called;
current pointers are loaded directly from native stack storage.

The owner.localizedLocs reference is reloaded after metadata callbacks. A null
owner reaches an actual field-read fault at `0x3F5595`; a null List reaches
the supplied nonreturning guard at `0x3F5672`. This execution does not establish
runtime null admission by an actual locale provider.

Supplied GetEnumerator receives a hidden output pointer in RCX, the current
List in RDX and exact GetEnumerator MethodInfo in R8. It writes a complete
24-byte reference-type enumerator: list pointer at 0, index DWORD at 8,
version DWORD at 12 and current pointer at 16. That layout is pinned to the
hash-verified Dumper header and generic declaration, without treating the
generic dump's zero field offsets as specialized offsets. Actual MOVUPS and
MOVSD copy its output into active stack state, followed by clearing the
exception slot and saving the active enumerator address. Supplied GetEnumerator
RAX bits may be zero; the copied storage remains the caller's source.

MoveNext receives the active state pointer and reloaded full MethodInfo. Its
AL alone decides whether to inspect an entry. The native caller then captures
current LocaleLoc in RDI; a null current reaches the guard at `0x3F5678`.
String.op_Equality receives that captured record's current localeCode in RCX,
the originally captured query in RDX and full zero R8 MethodInfo. Only AL
selects a match. The first match calls supplied Dispose, then returns captured
RDI, regardless of a callback replacing the active current pointer.

With no match, Dispose executes and native XOR EAX,EAX forces the entire RAX
to zero. Supplied Dispose upper bits do not survive this no-match return.
The standalone method returns the nominal captured LocaleLoc pointer on match;
the caller does not produce or initialize a new LocaleLoc.

The supplied iterator contract explicitly records its 24-byte output, captured
ordered entries, cursor/current writes and raw return bits. String equality is
also supplied with explicit outputs. Actual collection/search policy, string
implementation and engine locale behavior are not established.

## Actual getters consume the captured result

Both getters clear full R8 before calling actual FindLocaleLoc. A null result
returns full zero and skips IsNullOrEmpty. A nonnull result is captured in the
getter's RBX. GetTranslatedName reads `+0x18`; GetIWasTranslated reads `+0x20`.
The loaded field is passed to supplied String.IsNullOrEmpty in RCX with full
zero RDX MethodInfo. Nonzero AL returns full zero. Zero AL reloads that field
from the same captured LocaleLoc after the callback and returns the current
pointer, including null.

Consequently, Dispose callbacks can change a selected locale field before the
getter's first load. An IsNullOrEmpty callback can change it again before the
return load. Replacing the active enumerator current pointer during equality or
Dispose does not replace either the search's captured LocaleLoc or the getter's
captured result. Different records with equal codes remain separate identities;
the first matching record wins. Duplicate records, aliased text/query pointers,
nullable fields and two owner receivers remain explicit in one graph.

## Shared stack bytes and full ABI

Let S be the fixture entry stack pointer. Standalone FindLocaleLoc's frame is
`S-0x68`; nested FindLocaleLoc enters at `S-0x30` and has frame `S-0x98`.
Both use frame `+0x28` for hidden output and `+0x40` for active state. A single
112-byte diagnostic window spans `[S-0x78, S-0x08)`, retaining the overlapping
standalone and nested scratch regions.

This window also includes standalone helper-call return-address pushes, the
getter's FindLocaleLoc/IsNullOrEmpty return-address pushes and nested search
prologue saves of caller RBX/RSI/RDI. Those bytes are predicted as actual native
effects rather than treated as independent buffers or ignored padding. Mixed
retained sequences preserve and reuse this overlapping physical storage.
Memory-write hook values are compared at the emitted byte width, preserving
high-bit saved register patterns even when Unicorn reports them as signed.

Every native entry and supplied boundary records full RCX/RDX/R8/R9, all seven
integer volatile registers, XMM0 through XMM5 and its caller information. Events
are qualified by native phase, entry API and synthetic cleanup status. Native
entry caller kinds distinguish the fixture return sentinel, an actual nested
return, and a synthetic frame slot. All supplied boundaries have actual native
call return addresses. Services poison volatile integer/XMM registers
independently of authored RAX outputs. The model also accounts for MOVUPS/MOVSD
register effects and compares all final volatile values on returns, stops and
faults. Normal returns verify stack discipline, all eight integer nonvolatiles
and XMM6 through XMM15.

The complete graph includes both owners, three LocaleLoc records, supplied
Lists/arrays, strings, classes, four MethodInfo roots and alternative records,
opaque exception storage, an aliased cleanup enumerator and the shared stack
window. Sizes are diagnostic contracts, not managed object extents. Only actual
reached native stack/flag writes and completed supplied callback writes grant
mutation permission. Metadata roots and the flag are verified separately.
Callbacks requested for skipped or stopped phases grant no permission.

## Synthetic cleanup and exception limits

Separate direct probes enter actual cleanup at `0x3F5645` with an explicit
synthetic frame, saved nonvolatiles, null/live exception and active or aliased
enumerator receiver. They execute supplied Dispose, reload the exception slot,
then either return full zero through the ordinary epilogue or call supplied
rethrow at `0x3F567E`. Dispose callbacks can clear a live exception or introduce
one. The initial synthetic frame slot is qualified as opaque frame storage,
not claimed to be a real call return address.

These probes verify the authored frame and actual cleanup instructions. They
do not execute managed exception dispatch, a Windows unwinder or handler
`0x30CD28`. Guard and rethrow services are explicitly nonreturning stops.
Of 116 decoded instructions, 113 execute. The only three exclusions are the
two post-guard NOPs and final post-rethrow `int3`; no cleanup instruction is
silently counted as ordinary-entry evidence.

## Verification

The corpus has 504 cases, thirteen retained four-call sequences, 27 baselines
and 166 exact stopped prefixes. Retained calls cover success, a stopped search
comparison, recovery, callbacks and a mixed standalone/name/i-was sequence
with a stopped outer emptiness request. Every prior final snapshot equals the
next initial snapshot. Every stop matches the full baseline event prefix and
the boundary's complete pre-effect snapshot.

Two syntax-preceded independent final producers yield byte-identical reports;
all 36 reverse-engineering infrastructure tests pass. Source:
`reverse_engineering/scripts/audit_character_loc_search_join.py`.
Report:
`reverse_engineering/reports/f530404b0f3f_807de4a83df4_character_loc_search_join.json`.
Private peer:
`B:/CodexTools/DemonBluffReverseEngineering/artifacts/f530404b0f3f_807de4a83df4/character_loc_search_join_peer.json`.

Generic iterator/string implementations, runtime admission, engine localization,
exception machinery and other CharacterLoc providers remain separate boundaries.
