# CharacterData identity-generation caller

This audit executes the installed current-build `CharacterData.GenerateCharacterId`
body offline. All metadata, array allocation, Unity object-name, integer RNG,
boxing, type-check, formatting, reference-barrier and exception services remain
whole supplied boundaries. An independent ordered model predicts every event,
physical snapshot and final volatile ABI from the authored initial state.

## Declaration and complete native extent

The sole binding at `0x3B4520` is immutable `tdi5845.m0010`, symbol key
`CharacterData::public void GenerateCharacterId()`. Its exact Dumper signature is
`void CharacterData__GenerateCharacterId (CharacterData_o* __this, const MethodInfo* method);`
with type signature `vii`. The owner is `CharacterData : ScriptableObject,
ICharacterLocData, ICardData`, TypeDefIndex 5845. The consumed field is
`public string characterId; // 0x18`.

The complete native extent is `[0x3B4520, 0x3B4982)`: 1,122 file-backed bytes
and 290 decoded instructions. Its four contiguous unwind chunks are:

| Start | Exclusive end | Flags |
| --- | --- | --- |
| `0x3B4520` | `0x3B4581` | 0 |
| `0x3B4581` | `0x3B48DF` | CHAININFO (4) |
| `0x3B48DF` | `0x3B48E6` | CHAININFO (4) |
| `0x3B48E6` | `0x3B4982` | CHAININFO (4) |

Each chained entry resolves to the verified first root. The next managed entry
is `0x3B4990`, separated by 14 `CC` padding bytes. The producer freshly hashes
`GameAssembly.dll` and `global-metadata.dat`, all three consumed Dumper outputs,
the exact declarations and inventory binding. It derives and checks the complete
gateway call-site lists and selected instruction operands from the pinned body.
The report retains the body hash, length and bounds, without exporting complete
native bytes or disassembly.

## Caller behavior and service chronology

The caller captures its original receiver in RDI and the eventual ID destination
in RSI. A cold flag at `0x288C4DC` performs three metadata calls in order:

| Root | Slot RVA |
| --- | --- |
| `int_TypeInfo` | `0x2707130` |
| `object[]_TypeInfo` | `0x2720110` |
| Exact format literal | `0x26EEE60` |

The literal is `{0}_{1}{2}{3}{4}{5}{6}{7}{8}`. Only after all three calls
complete does the native caller store flag byte 1. Every nonzero flag, including
diagnostic `80` and `FF`, skips this prefix.

It loads the receiver's current `characterId` into the whole supplied
`System.String.IsNullOrEmpty` call at `0xF76390`, with full zero RDX. The native
branch tests AL only. A zero AL skips all generation; upper return bits and
volatile residues remain recorded. The supplied result is explicit and does not
derive from the seeded nominal string bytes.

For nonzero AL, the caller reloads the current array TypeInfo root, requests
exactly nine entries from the supplied allocation gateway `0x2B7080`, and captures
the result in RBX. It then invokes the whole supplied `UnityEngine.Object.get_name`
at `0x1C82250` on the original captured receiver, with zero RDX. This call occurs
before the null-array guard, so even a supplied null allocation has a reached
name-service boundary.

The nine array elements are the captured object name followed by eight captured
boxing results. Each integer RNG call at `0x1C86600` receives exact zero-extended
RCX 0, RDX 10 and R8 0. The caller consumes only EAX: it writes four bytes to a
stack scratch slot and passes the slot address, plus the freshly reloaded Int32
TypeInfo root, to whole supplied boxing at `0x282580`. Arbitrary upper RAX bits
and diagnostic integer values are retained without asserting RNG policy. Supplied
boxing returns a nominal identity and does not initialize its physical bytes
unless an explicit reached callback says so.

Every nonnull name or box triggers a whole supplied type-check at `0x2B7010`.
The array identity remains captured, while its current class pointer and that
class's `element_class` at `+0x40` reload before each type-check. The returned
pointer is tested for zero; a nonzero return identity is otherwise discarded.
The native store uses the original captured name or box, even when the supplied
successful type-check returns a different nominal object. Null name and box
results skip type-checking and are stored as null.

Before every element store, the caller reloads the low DWORD of array
`max_length` at `+0x18` and applies unsigned `JBE` against that element's index.
Its following eight-byte store at `+0x20 + index*8` precedes the reference-barrier
boundary. The exact `System_Object_array` and `Il2CppClass` header declarations
bind these offsets; the audit preserves the full QWORD length so its unused high
DWORD is visible. Callback length changes before the current gate and class/root
changes before subsequent calls are modeled independently.

After all nine barriers, the caller reloads the current literal root and supplies
it with the captured array and full zero R8 to `System.String.Format` at
`0xF74B10`. Its supplied result is captured and written to the original receiver's
`characterId` before the final reference barrier. Format callbacks that write the
ID first are overwritten by the native store; final barrier callbacks can replace
the newly stored ID. Formatting output text and saved identity policy remain
outside this caller reconstruction.

## Complete stack, ABI, aliases and reached effects

The frame is entry RSP minus `0x58`, and every ordinary supplied call arrives at
frame minus 8 with its exact native return address. The complete diagnostic stack
window spans entry RSP minus `0x68` through entry RSP plus `0x30` (152 bytes).
It includes the call-return cell, saved RSI/RDI/RBX, caller return sentinel,
all eight scratch slots and adjacent unused bytes. Scratch offsets relative to
the frame are `60`, `70`, `78`, `20`, `24`, `28`, `2C`, `30`, each hexadecimal.
Each four-byte native write preserves adjacent bytes, including nonzero high
halves of QWORD storage. The independent model predicts the same prologue,
call-return writes, saved registers and scratch bytes, without consulting native
phase memory for expected values.

Every supplied entry, including failures, records raw RCX/RDX/R8/R9, all seven
integer volatile registers, XMM0 through XMM5, caller return and raw RSP. Each
successful supplied return independently poisons the integer/XMM volatile state
and supplies full RAX bits. Normal completion verifies original caller stack
discipline, all eight integer nonvolatile registers and XMM6 through XMM15.
The managed method is void; final RAX is a supplied-service residue.

All 30 nominal object windows and the stack window retain every physical byte.
These are diagnostic window sizes and identities, not managed extent or type
admission. The corpus includes repeated box identities, text/box/result aliases,
reused arrays, changed arrays across retained calls, null results and different
successful type-check return identities. Whole supplied services permit these
authored diagnostics without claiming that the CLR would allocate or admit them.

The write hook only allows exact decoded native stores and completed authored
callback writes. Permissions are populated when each write is actually reached;
future callback phases and supplied failures grant no permission. Full raw
snapshots retain both receivers, both arrays, all class/element roots, every
nominal string/box/exception record and complete accumulated service history.
Cold and warm retained calls preserve partial metadata and barrier history; every
previous final snapshot equals the following initial snapshot, including stack.

## Exceptions, fault boundaries and coverage

A null original receiver reaches an actual read fault at `0x3B455E`, after any
completed cold metadata prefix. A null captured array reaches supplied null
exception gateway `0x2B7D90`. Each failed nonnull type-check reaches its own
exception-maker call at `0x2B78A0`, followed by `0x2B7D50` with captured exception
in RCX and full zero RDX. Every length guard reaches the common supplied bounds
exception gateway `0x2B7D80`. A callback that nulls the current array class reaches
the actual element-class read fault at the corresponding cast call site minus
four; every such position is covered.

Supplied raising/null/bounds services normally stop at the whole nonreturning
boundary after explicit completed callback effects. Failure probes instead stop
at the pre-effect service entry. Separate diagnostic profiles author a returned
exception service and stop at the following native INT3 boundary before executing
that instruction. They demonstrate all 11 boundaries, without executing traps
or claiming exception construction, raising or Windows/managed unwinding.

Of 290 decoded instructions, 279 non-trap caller instructions execute. The 11
excluded INT3 addresses are `0x3B48EB`, `0x3B48FB`, `0x3B490B`, `0x3B491B`,
`0x3B492B`, `0x3B493B`, `0x3B494B`, `0x3B495B`, `0x3B496B`, `0x3B497B`,
`0x3B4981`. Report coverage separates those directly reached trap-boundary probes
from executed instructions.

## Verification and artifact scope

The corpus has 141 cases, six retained three-call sequences, 44 baselines and
1,053 complete supplied-boundary stopped prefixes. Every stop matches the entire
baseline event prefix and exact pre-effect full snapshot. Every row also matches
the independent ordered full-byte model and complete final volatile ABI.

Canonical source: `reverse_engineering/scripts/audit_character_data_identity.py`.
Canonical report: `reverse_engineering/reports/f530404b0f3f_807de4a83df4_character_data_identity.json`.
The independent matching peer lives privately at
`B:/CodexTools/DemonBluffReverseEngineering/artifacts/f530404b0f3f_807de4a83df4/character_data_identity_peer.json`.

The report uses both lossless codecs: `pool_memory`/`expand_memory` from
`audit_character_oracle_reveal_join.py`, then `pool_snapshots`/`expand_snapshots`
from `audit_report_snapshots.py`. Each producer asserts
`expand_memory(expand_snapshots(report)) == original_report`, preserving every
field, complete prefix, stack byte and history. Two syntax-preceded independent
producers emitted byte-identical reports; all 36 reverse-engineering
infrastructure checks passed. The report is 32,045,328 bytes, SHA-256
`848b2d1a7c8dfd50caa4b4e332c707315af37bc23ebf8e35ad21b29864d1ab4c`.
No actual allocation, RNG, boxing, cast, formatting,
metadata resolver, collector, Unity name access or exception runtime executes;
their whole supplied inputs/effects remain explicit.

## Guarded Rust normal caller replay

The guarded Rust [identity-generation replay](notes/systems/character_data_identity.md)
compares 46 inert normal native cases, 15 retained rows and three full sequences.
Five tests check the complete stack, captured array/values, integer scratch
writes, metadata, histories and raw ABI, with atomic context rejection.

The normal replay retains all 31 physical windows, including the complete
152-byte stack, and all thirteen explicit categorized service histories. It
does not invent ordering across prior categories; current steps are ordered.
Original captured values are stored even when the supplied successful cast
returns another nominal identity. EAX writes preserve neighboring scratch bytes.
The low DWORD array length must admit all nine stores before cloning.
Nominal ranges cannot overlap each other or the three native roots/flag, and
all exclusive endpoints are checked. Canonical lowercase XMM diagnostics and
required nullable scalar fields prevent lossy or missing evidence. Future work
reserves 452 units per call (200 for an entry plus 42 complete history records),
43 snapshots per call plus two, and trace/final ABI storage separately; the
checked work limit is 4,194,304. Callbacks, stopped/exceptional calls and traps
remain native evidence rather than supported future replay behavior.

Source: [character_data_identity.rs](../../../crates/solver-core/src/bluff/character_data_identity.rs).
All 928 Rust library tests and the release build pass; all 36 RE infrastructure
tests passed at the native checkpoint. Independent read-only review found no
remaining blockers. Simulation and Python bridge suites were not rerun for
these offline caller modules.
