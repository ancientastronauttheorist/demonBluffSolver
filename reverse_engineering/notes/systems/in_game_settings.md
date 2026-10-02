# InGameSettings native callers

This offline audit executes all three nonconstructor `InGameSettings` caller
bodies from the pinned GameAssembly. Input and GameObject services are explicit
supplied effects. It does not read real keyboard input, admit Unity lifecycle
calls, open the live menu, render the settings object, or infer scheduler timing.

## Exact bindings

The declaration is `InGameSettings : MonoBehaviour`, TypeDefIndex 5769, with
one custom field, `public GameObject settings; // 0x20`. All three metadata
signatures are exact `void` instance methods with `vii` type signatures:

| Method | Method ID | RVA | Body end exclusive | Next managed entry |
| --- | --- | --- | --- | --- |
| Update | tdi5769.m0000 | 0x3A22A0 | 0x3A22EE | 0x3A22F0 |
| OnEnable | tdi5769.m0001 | 0x3A2270 | 0x3A2291 | 0x3A22A0 |
| ManageShowSettings | tdi5769.m0002 | 0x3A2220 | 0x3A226F | 0x3A2270 |

Each body has a verified unwind entry with these exact bounds. The audit checks
raw section backing and a complete read, decodes from each verified entry through
the entire final instruction, and verifies `CC` alignment padding to the next
managed entry. No metadata flag, class-init gate or literal lookup is present.

`OnEnable` shares its native RVA with `NightStep.Disable`. Both metadata rows
are recorded, but only the exact InGameSettings declaration is bound here;
NightStep's fields and lifecycle are not inferred from the shared implementation.
The folded constructor at `0x33E820` is outside this audit.

The three supplied service bindings are independently checked against exact
metadata signatures: `Input.GetKeyDown` at `0x1CD3C80`, `GameObject.get_activeSelf`
at `0x1C7DC50`, and `GameObject.SetActive` at `0x1C7D810`. The input gateway's
`GetKeyDownInt` folded alias is recorded without promoting its declaration.
The caller consumes key 27; the exact pinned `UnityEngine.KeyCode` enum,
TypeDefIndex 6685, identifies `Escape = 27`.

## Native behavior and widths

`OnEnable` loads `settings`, requires it to be nonnull, and tail-transfers to
SetActive with a canonical false value and zero R8 MethodInfo. It resets the
authored menu GameObject's supplied active state.

`ManageShowSettings` requires a nonnull settings reference, obtains its supplied
active result, then reloads `settings`. A nonzero AL selects false with an
entire EDX clear. A zero AL selects true with `DL = 1`, preserving the supplied
upper RDX bits. Both branches pass zero R8 MethodInfo and tail-transfer to
SetActive after restoring the native frame.

`Update` first obtains the supplied Escape GetKeyDown result. A zero AL returns
without reading the owner or settings reference. A nonzero AL obtains activeSelf
from the current settings object, reloads `settings`, inverts the supplied AL
with `sete dl`, and tail-transfers to SetActive with zero R8 MethodInfo. Unlike
ManageShowSettings' false branch, Update preserves supplied upper RDX bits for
both Boolean outcomes. The fixtures assert complete physical RDX values, its
canonical low byte, and the zero MethodInfo arguments. Key and active gates use
only AL, including zero/noncanonical bytes with poisoned upper RAX bits.

Input, activeSelf, and SetActive services poison integer volatile registers and
XMM0–XMM5 before returning. Normal caller returns verify the original Win64
return stack, all eight integer nonvolatiles, and XMM6–XMM15. This establishes
caller ABI behavior under supplied normal completion, not the underlying Unity
implementations.

## Retention, receiver reloads and guards

The fixture graph retains an owner, two GameObjects and a class diagnostic
window, with full `0x80` byte snapshots at initial state, every service entry,
and final state. These are diagnostic windows, not asserted managed object
sizes or valid typed interpretations of unconsumed sentinel bytes. Only the
exact settings reference and supplied service state are consumed. All other
bytes are checked for exact retention.

Service callbacks can explicitly replace, restore or clear the owner's settings
reference, or retain the same receiver. The getter records and returns the
captured source object's result; the following native SetActive uses the
reloaded destination. A read-time replacement can therefore invert one object's
supplied result and write that value to the other object. Aliasing the replacement
to the captured receiver preserves the same-object path. Key-time replacements
are consumed by Update's subsequent read, and write-time replacements remain
visible to later explicitly invoked calls. These are authored callback effects,
not engine races or automatic lifecycle transitions.

Clearing settings during a read reaches the native second-reference null guard
after the read has completed. Clearing it during the key service reaches the
first settings guard after the key result. Initial null owners fault at verified
field loads in OnEnable and ManageShowSettings, or after a nonzero key in Update.
A zero-key Update can return without consuming a null owner or null settings
argument; this does not claim Unity admission of those null receivers.

Supplied service-entry stops match the entire baseline event/snapshot prefix
and the stopped final snapshot. Both active result branches, an idle Update,
and read-time receiver replacements have independent stop baselines. No engine
exception unwinding or rollback is inferred from these controlled stops.

## Reproduction and scope

The corpus contains 151 standalone normal, edge and mutation profiles, four
retained sequences, nine baselines and 18 exact service-entry stops. The retained
sequences include toggles, an idle key result, resets and receiver changes in
one continuing supplied state. All 61 nontrap instructions of the 64 decoded
instructions execute. The three remaining instructions are recorded terminal
`int3` traps after native null-guard gateways; those gateways stop explicitly.
There are 20 direct instruction assertions plus decoded call-site count checks.

```powershell
python -m py_compile reverse_engineering/scripts/audit_in_game_settings.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
$env:PYTHONPATH = 'B:/CodexTools/DemonBluffReverseEngineering/python-emulation'
python reverse_engineering/scripts/audit_in_game_settings.py `
  --game-root 'B:/SteamLibrary/steamapps/common/Demon Bluff Playtest' `
  --dumper-root 'B:/CodexTools/DemonBluffReverseEngineering/artifacts/f530404b0f3f_807de4a83df4/il2cppdumper-v6.7.46' `
  --output 'B:/CodexTools/DemonBluffReverseEngineering/artifacts/f530404b0f3f_807de4a83df4/in_game_settings_peer.json'
```

The producer validates GameAssembly and both Dumper artifacts through the
checked-in manifests for build `f530404b0f3f_807de4a83df4`. The public report is
`reports/f530404b0f3f_807de4a83df4_in_game_settings.json`. No private native method
bytes are copied into that report.

Two successful independent final producers, each preceded by Python syntax
compilation, produced byte-identical reports: 1,085,353 bytes, SHA-256
`bcfb5918e8e4f4ac0137d305eaf8a409a8b4f1eec75c3bf4064f53ab830556d8`.
The independent private peer is
`B:/CodexTools/DemonBluffReverseEngineering/artifacts/f530404b0f3f_807de4a83df4/in_game_settings_peer.json`.
All 32 reverse-engineering infrastructure tests passed. No Rust build,
simulation suite, live game or Python bridge regression was run by this audit.
