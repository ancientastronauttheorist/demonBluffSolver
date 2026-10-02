# SkinData constructor caller

Status: frozen after two successful identical final producers and fresh
serialized model verification. The exact declaration is `public void
.ctor()` in the closed `public class SkinData : ScriptableObject`, TypeDefIndex
5945, method ordinal 3 (`tdi5945.m0003`). Its exact metadata signature is
`void SkinData___ctor (SkinData_o* __this, const MethodInfo* method);` (`vii`).
Twenty-nine declarations share RVA `373A40`; only this nominal SkinData binding
is selected. No other folded declaration, runtime object construction,
allocation default, or base implementation is promoted.

The complete caller is a seven-byte, two-instruction leaf wrapper, with no
containing unwind record. Its end is `373A47`. The uniquely verified next managed
entry is `SfxController.Awake` at `373A50`; nine following `CC` bytes are alignment
padding, not terminal guards. Complete body SHA-256 is
`20e9f56f8427c04c0925a2326da4e8aa3579d033d7c316b96cb5b2782fae51df`.
The producer checks raw section extent, complete read/decode/length, exact leaf
boundary, next declaration, and the full input fingerprints. The tracked report
contains one authored operand-width assertion and the tail gateway identity,
not a complete native export. A discovery export remains private.

The actual caller zeroes EDX, clearing the upper DWORD of RDX too, then tail
transfers to the exact whole supplied `UnityEngine.ScriptableObject..ctor`
gateway at `1C8A5C0`. It retains the original RCX receiver, R8/R9 and other unused
volatile residues, all incoming XMM values, original caller return address, and
entry SP. It reads or writes no SkinData field. No metadata, Color, allocator,
barrier, renderer, or scheduler service is reached by this caller.

Fresh SHA-256 and size checks cover installed GameAssembly/global metadata and
all Dumper dump/header/script outputs. Exact closed SkinData fields, ERarity,
EArtType, Color's four float fields, ScriptableObject declaration/constructor,
and the exact SkinData method order are bound before execution. Full authored
windows retain both SkinData owners, their nominal string/art/unlock/data/class
dependencies, and the complete consumed external stack window. These are
diagnostic windows, not assertions about Unity native object extents.

The independent model begins with complete initial byte storage and all entry
registers. It models the DWORD clear and tail handoff separately from Unicorn,
then applies only the explicitly completed supplied base effects/return. It
compares full pre-effect service state, four raw arguments, seven GPR integers
and canonical 16-digit hex strings, six canonical 32-digit XMM strings, all
remaining registers, caller/SP, native-entry history, completed service history,
and final bytes/registers. Native caller writes are forbidden by the hook.
Normal inner and outer Win64 preserve eight integer nonvolatiles, XMM6–15, and
restore SP. Void return residues are authored diagnostics, not a managed return.

The bounded corpus varies two physical owners, zero/A5/FF storage, nullable
reference fields, full raw incoming method bits, arbitrary enum/color bit
patterns (including negative zero and NaNs), and supplied return residues. Inert
base profiles preserve every SkinData byte, including glowColor and art type;
this does not imply how an actual runtime-created object initializes them.
Additional profiles explicitly attribute a glow-color or locked-art write to
the completed whole supplied base contract. A stop before that base effect
leaves every byte untouched. A null receiver is examined only at a supplied
stop; no legal runtime construction or native base behavior is inferred.

Retained calls preserve every physical record and all chronological histories;
the leaf has no caller-local stack stores, so adjacent complete snapshots match.
Each invocation supplies its register inputs and resets its service ordinal to
one. The report includes normal repeated calls, alternating owners, stopped-call
recovery, and retention of an earlier supplied base write. Full base stopped
prefixes retain unchanged caller storage plus the native entry and cleared-RDX
service register state.

Five frozen lossless codecs retain all raw bytes, memory maps, ordered histories,
state maps, and complete snapshots. `expand_report` is the full decoder;
`expand_memory` accepts input already expanded by `expand_snapshots`. Local codec
hash/deep-copy/order/corruption assertions and complete report round trips run in
the producer. The 36 RE infrastructure tests remain separate from these local
assertions.

Final evidence contains 33 cases (32 normal returns and one supplied stop), four
retained two-call sequences, three baselines and three full stopped prefixes.
Both decoded instructions execute; all 47 rows pass the independent model with
14 complete physical windows. Each final producer followed successful Python
syntax compilation. Their complete serialized files are byte-identical:
311,420 bytes, SHA-256
`3420909abd3f4419b9ab847f68326e62adfd10a520e0a85f1b813873bdeb23e7`.
Source SHA-256 is
`dccfb1bc6eb558b13321674135b97dd475d04d19ec5ec86f9f757fa5553d2ee2`.
Report schema is `skin_data_constructor_native_v1`.

A separate fresh process decoded the serialized report, reran the model for
every row, checked every full window, raw argument/caller identity, retained
snapshot continuity and complete stopped register/state prefix, and confirmed
the post-snapshot decoder adapter equals full expansion. Local codec tests
passed again. All 36 RE infrastructure tests passed. Complete native discovery
remains private at the pinned artifact directory; independent producer outputs
are `skin_data_constructor.first.private.json` and
`skin_data_constructor.peer.private.json` there.

Reproduce from the repository root with the pinned emulator dependencies on
`PYTHONPATH`, compiling the script first and stopping on failure:

```powershell
python -m py_compile reverse_engineering/scripts/audit_skin_data_constructor.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
$env:PYTHONPATH='B:/CodexTools/DemonBluffReverseEngineering/python-emulation'
python reverse_engineering/scripts/audit_skin_data_constructor.py `
  'B:/SteamLibrary/steamapps/common/Demon Bluff Playtest' `
  'B:/CodexTools/DemonBluffReverseEngineering/artifacts/f530404b0f3f_807de4a83df4/il2cppdumper-v6.7.46' `
  --output 'B:/CodexTools/DemonBluffReverseEngineering/artifacts/f530404b0f3f_807de4a83df4/skin_data_constructor.peer.private.json'
```

No actual ScriptableObject base implementation, Unity allocation/registration,
managed admission, unrelated folded alias, or wider constructor provenance is
asserted by this small caller audit.

## Guarded Rust normal caller replay

The guarded Rust [SkinData constructor replay](notes/systems/skin_data_constructor.md)
compares all 32 normal native cases and seven normal retained calls, including
stopped-call recovery. Five tests check complete storage, tail-call ABI, explicit
supplied base writes and atomic schema, identity and capacity rejection.

The bounded replay retains all fourteen physical windows, complete native-entry
and completed-service histories, all sixteen GPRs and sixteen XMM values. It
clears RDX before the tail gateway, keeps the original receiver/caller/SP, and
applies only the two explicitly admitted supplied base write shapes. The
constructor itself writes no owner or stack bytes. Three complete normal
two-call sequences and the normal recovery suffix compare full retained state.
Supplied stops remain native evidence and reject as future calls.

Nominal field references, complete disjoint window extents, canonical XMM
strings, caller sentinel, aligned external stack and exclusive endpoints validate
before cloning. Nested records/calls reject unknown fields; retained ABI/history
objects validate exact keys and values. Future history and snapshot growth is
reserved under checked work bounds. No actual ScriptableObject implementation,
allocation default or Unity runtime construction is inferred.

Source: [skin_data_constructor.rs](../../../crates/solver-core/src/bluff/skin_data_constructor.rs).
All 933 Rust library tests, the release build and 36 RE infrastructure tests
passed. Independent read-only peer review found no remaining blockers.
Simulation and Python bridge suites were not rerun for this offline module.
