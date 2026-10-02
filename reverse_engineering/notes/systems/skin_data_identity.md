# SkinData identity-generation caller

This audit executes the installed current-build `SkinData.GenerateSkinId` caller
offline. Metadata, array allocation, Unity name, integer RNG, boxing, type checks,
formatting, reference barriers and exception services are whole supplied
boundaries. An independent ordered model starts from the authored initial bytes
and predicts all service events, snapshots, final storage and volatile ABI.

## Exact declaration and extent

The unique binding is immutable `tdi5945.m0000`, symbol key
`SkinData::public void GenerateSkinId()`, at `0x3EBB90`. Dumper declares
`void SkinData__GenerateSkinId (SkinData_o* __this, const MethodInfo* method);`
with type signature `vii`. Its owner is `public class SkinData : ScriptableObject
// TypeDefIndex: 5945`; the consumed field is `public string skinId; // 0x18`.

The complete extent is `[0x3EBB90,0x3EBE42)`: 690 file-backed bytes and 178
decoded instructions. Four contiguous unwind chunks resolve to the first root:

| Start | Exclusive end | Flags |
| --- | --- | --- |
| `0x3EBB90` | `0x3EBBF1` | 0 |
| `0x3EBBF1` | `0x3EBDDF` | CHAININFO (4) |
| `0x3EBDDF` | `0x3EBDE6` | CHAININFO (4) |
| `0x3EBDE6` | `0x3EBE42` | CHAININFO (4) |

The next verified managed entry is `0x3EBE50`, after fourteen `CC` padding bytes.
Each producer independently verifies the pinned DLL and metadata inputs, exact
Dumper output hashes, method row, declaration, immutable inventory binding,
unwind chain and selected decoded operands/call-site lists. The report contains
the complete body hash and bounds, without complete byte or disassembly exports.

## Caller chronology

The original owner is captured in RDI and its ID destination in RSI. Cold flag
`0x288C681` initializes three metadata roots in order, then stores byte 1 only
after all calls finish. Any nonzero flag skips this prefix.

| Root | Slot RVA |
| --- | --- |
| `int_TypeInfo` | `0x2707130` |
| `object[]_TypeInfo` | `0x2720110` |
| Literal `{0}_SKIN_{1}{2}{3}{4}` | `0x26EE6F0` |

The owner ID at `0x18` is passed to supplied `String.IsNullOrEmpty` (`0xF76390`)
with full RDX zero. Only AL controls generation: zero AL returns immediately,
including when upper return bits are nonzero. String contents do not determine
this diagnostic supplied result.

For nonzero AL, the caller reloads the array TypeInfo and requests five elements
from supplied allocation (`0x2B7080`). It captures that array in RBX. Supplied
`UnityEngine.Object.get_name` (`0x1C82250`) receives the original owner and full
RDX zero before the null-array guard, so a null allocation still reaches name.

The five entries are the captured name followed by four captured boxing results.
Each supplied integer `Random.Range` (`0x1C86600`) receives full zero-extended
RCX 0, RDX 9 and R8 0. The installed upper bound is **9**, rather than the
CharacterData caller's 10. EAX is stored as exactly four scratch bytes before
supplied boxing (`0x282580`) receives the current reloaded Int32 root and scratch
address. Upper RAX bits never enter the integer scratch.

Each nonnull captured value triggers a supplied type check (`0x2B7010`) using
the allocated array's current class and current element-class pointer at `0x40`.
The returned pointer is tested for null but otherwise discarded; the native
array store uses the original captured value even when a successful check
returns a different nominal identity. Null name/boxes skip type checking. Before
each of the five stores, the caller compares the array length's unsigned low
DWORD against that item's index and takes `JBE` to the bounds boundary.

The current format root is reloaded after the five array stores. Supplied
`String.Format` (`0xF74B10`) receives that root, the original captured array and
R8 zero. The returned nullable identity is stored in the original owner's ID
field before the final supplied reference barrier (`0x2B6FF0`). Barrier return
bits are opaque void residue and remain recorded.

## Independent state and ABI checks

Twenty-six nominal diagnostic records plus a 136-byte stack window are retained
in every snapshot. These seeded record widths are diagnostic storage windows,
not reconstructed managed object extents. The frame is entry RSP minus `0x48`;
the stack window spans entry RSP minus `0x58` through entry RSP plus `0x30`.
Four boxing scratch offsets relative to that frame are `0x50`, `0x60`, `0x68`,
and `0x20`. Complete bytes preserve high scratch bytes, native call-return slots
and saved nonvolatile registers. All normal returns preserve Win64 nonvolatile
GPRs and XMM6..15 and advance entry RSP by eight.

Every service entry includes all seven volatile GPRs, six canonical lowercase
128-bit XMM values, raw RCX/RDX/R8/R9, native caller and RSP, along with the
complete state snapshot before effects. The model derives these independently
from initial state and authored supplied outputs/callbacks. It does not use
observed native state to calculate expected events. Histories are chronological
and categorized across retained invocations.

Native writes are checked against decoded instruction sites, destinations and
widths. Retention admits only actual reached native stores or reached authored
callback mutation ranges; untouched records, classes, stack bytes, metadata
slots and flags must remain byte-identical. Callback profiles exercise owner-ID
mutation, array class/element reloads, current length gates, metadata root reloads,
captured values versus successful cast aliases, literal changes and final
barrier effects. Nullable supplied outputs, upper integer/AL bits, changed
owners/arrays, seeded patterns and warm flags have separate profiles.

## Guards, stops and explicit exclusions

Null-owner reads and null current-array-class reads are bounded actual native
read faults. Null allocated arrays, insufficient low-DWORD lengths and failed
type checks reach supplied exception boundaries. The five failed type-check
paths call supplied exception creation (`0x2B78A0`) then raise (`0x2B7D50`), with
full RDX zero. Null and bounds gateways are `0x2B7D90` and `0x2B7D80`.

Authored normally-nonreturning exception services stop before downstream native
instructions. Separate synthetic returning-service probes reach seven `INT3`
entries, where the code hook stops **before execution**. These are qualified
trap-boundary observations, not executed trap instructions. Normal/guard paths
exercise all other 171 instructions. No exception unwinder or runtime exception
policy is reconstructed.

For every reached service in each of 28 baseline profiles, a stop before service
effects verifies the complete event prefix and final snapshot against that
baseline entry snapshot. There are 445 stopped prefixes. Six retained three-call
sequences require each final state to equal the next initial state. The final
producer reports 117 cases and 18 retained calls.

## Artifact and reproducibility

Source: `reverse_engineering/scripts/audit_skin_data_identity.py`.
Canonical report:
`reverse_engineering/reports/f530404b0f3f_807de4a83df4_skin_data_identity.json`.
Matching private peer:
`B:/CodexTools/DemonBluffReverseEngineering/artifacts/f530404b0f3f_807de4a83df4/skin_data_identity.peer.private.json`.
Fresh selected discovery:
`B:/CodexTools/DemonBluffReverseEngineering/artifacts/f530404b0f3f_807de4a83df4/skin_data_identity.discovery.private.json`.

Reports use the lossless `pool_memory`/`expand_memory` codec from
`audit_character_oracle_reveal_join.py`, followed by `pool_snapshots`/
`expand_snapshots` from `audit_report_snapshots.py`. Every producer asserts the
combined expanded report equals its complete original value. No fields,
histories, bytes or stopped prefixes are dropped.

Both final independent producers completed successfully, each immediately after
syntax compilation, using the pinned installed game/Dumper and private
`python-emulation` dependency path. Their reports match byte-for-byte: 12,994,045
bytes, SHA-256 `fbe265defdb62b86321f0649ad1db7e97b8435cf811a85fac783e0a2510a9782`.
There are 1,444 complete snapshot blobs and 166 memory blobs, with 53 selected
instruction assertions. All 36 reverse-engineering infrastructure tests passed.

This family reconstructs only this caller's dataflow; whole RNG, formatting,
boxing, metadata, Unity and runtime policies remain supplied.
