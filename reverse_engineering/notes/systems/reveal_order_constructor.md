# RevealOrder constructor caller

This audit binds the exact `RevealOrder` constructor declaration (`tdi5735.m0002`)
to the pinned installed GameAssembly and Il2CppDumper metadata. It executes the
complete game-owned wrapper and selected retained presentation joins offline.
Unity constructor, component, formatting and TMP effects are explicitly supplied.
There is no live process read, rendering or scheduler execution.

## Exact declaration and shared body

`RevealOrder` is a `MonoBehaviour` with one declared custom field,
`public TextMeshProUGUI text; // 0x20`. Its constructor metadata is exactly:

```c
void RevealOrder___ctor (RevealOrder_o* __this, const MethodInfo* method);
```

The Dumper type signature is `vii`. The verified native entry is `0x33E820`;
its complete body ends at `0x33E827` exclusive. The next managed entry is
`0x33E830`, with nine `CC` alignment bytes between the body and that entry.
The PE raw section extent and complete read are checked before decoding.
Containing unwind entries, when present, are recorded rather than assumed.

The two instructions are:

```asm
33E820  xor edx, edx
33E822  jmp 1C79770
```

The `EDX` write zeroes all of `RDX`. The tail transfer preserves the original
owner in `RCX` and the original stack return address, and discards the incoming
MethodInfo. `R8` and `R9` remain unchanged at the supplied service entry. There
are no native object writes, field initialization, allocations, metadata gates,
class initialization, or calls to text/renderer services in this wrapper.

There are 218 metadata declarations at this wrapper RVA. This audit establishes
only the exact RevealOrder declaration; it does not classify every folded alias.
The gateway at `0x1C79770` has five folded Unity constructor declarations. The
nominal base for RevealOrder is `UnityEngine.MonoBehaviour`, whose exact `vii`
signature and metadata row are checked separately. The gateway body is supplied,
not claimed as executed or reconstructed here.

## Supplied base boundary and diagnostics

The fixtures explicitly supply normal base acceptance or a stop at its entry.
Acceptance can retain storage, clear `text`, or replace `text` with an already
authored reference. Those field changes are supplied effects, not native wrapper
behavior. Null owners are accepted only by these authored service profiles; this
does not establish actual Unity null-receiver behavior or successful engine
construction. The incoming MethodInfo can be null, a retained fixture pointer or
an invalid poison value, because the executed wrapper never reads it.

Fresh zeroed, serialized-input and reused owner storage are retained in complete
diagnostic windows. Class/monitor and consumed reference slots are authored;
sentinel bytes outside consumed fields are diagnostics, not inferred valid
typed managed fields. The RevealOrder diagnostic window is `0x80` bytes, not an
asserted managed object size. Referenced fixture components, virtual tables,
MethodInfos, strings and the owner class each have a complete retained memory
window in every initial, service-entry and final snapshot. The stack's `0x40`
byte window is also retained. Thus a supplied stop checks the full event and
snapshot prefix, rather than only its last method name.

Normal supplied returns preserve the native Win64 return stack, all eight
integer nonvolatile registers and XMM6–XMM15. Supplied services poison RCX, RDX,
R8–R11 and XMM0–XMM5, and return a distinctive RAX poison. The wrapper is void;
that RAX value is not an allocation result or initialized object identity.

## Retained joins

The optional joins execute the frozen `RevealOrder.Init` and `Hide` native
callers from `audit_reveal_order_presentation.py` after this constructor wrapper.
They use explicit retained serialized component input and the same supplied
base effects. `Init` receives exact signed-low-DWORD order inputs, and `Hide`
uses the supplied component GameObject path. Component GameObject lookup,
SetActive, Int32 formatting and the TMP virtual setter remain supplied services.
The inherited presentation harness pins their exact metadata and instruction
relationships and validates complete retained memory, service-entry snapshots
and normal Win64 return preservation.

The joined examples establish that the constructor does not install `text`.
Fresh zero storage and an explicitly cleared text reference cause `Init` to
reach its native null guard after GameObject activation and formatting; `Hide`
can still return because it does not consume that text field. Reused component
storage can retain its other text reference across repeated constructor calls.
These are authored component graphs and explicit call sequences, not evidence
of Unity serialization order, engine allocation, lifecycle admission or scene
readiness. No engine exception unwinding is modeled by a supplied entry stop.

## Reproduction

```powershell
python -m py_compile reverse_engineering/scripts/audit_reveal_order_constructor.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
$env:PYTHONPATH = 'B:/CodexTools/DemonBluffReverseEngineering/python-emulation'
python reverse_engineering/scripts/audit_reveal_order_constructor.py `
  --game-root 'B:/SteamLibrary/steamapps/common/Demon Bluff Playtest' `
  --dumper-root 'B:/CodexTools/DemonBluffReverseEngineering/artifacts/f530404b0f3f_807de4a83df4/il2cppdumper-v6.7.46' `
  --output 'B:/CodexTools/DemonBluffReverseEngineering/artifacts/f530404b0f3f_807de4a83df4/reveal_order_constructor_peer.json'
```

The pinned build is `f530404b0f3f_807de4a83df4`; the producer validates the
installed GameAssembly and Dumper artifact hashes through their checked-in
manifests. Reports contain symbolic fixture references and diagnostic fixture
bytes; no copied private native method bytes are emitted.

The final corpus contains 54 standalone contexts, six baselines, six exact
supplied base-entry stops, and 11 retained sequences. Both constructor
instructions are executed. The complete report is stored at
`reports/f530404b0f3f_807de4a83df4_reveal_order_constructor.json`.

Two successful independent final producer processes, each preceded by Python
syntax compilation, produced byte-identical reports: 3,383,323 bytes, SHA-256
`9887eb2b37484bdc7d38f14ed8a997c9bde484751b1ee7be5ade5102deb6d663`.
The independent private output is
`B:/CodexTools/DemonBluffReverseEngineering/artifacts/f530404b0f3f_807de4a83df4/reveal_order_constructor_peer.json`.
All 32 reverse-engineering infrastructure tests passed. These checks do not
rerun solver simulations, live games, or the Python bridge regression suite.
