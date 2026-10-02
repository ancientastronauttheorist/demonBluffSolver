# CardTokens native callers

This offline audit executes the exact `CardTokens.OnEnable` and `Update` bodies
from pinned build `f530404b0f3f_807de4a83df4`. Input.GetKeyDown, GameObject
activeSelf and SetActive are explicitly supplied services. It does not collect
live keyboard input, render tags, infer Unity lifecycle admission, or implement
engine exception unwinding.

The exact class is `CardTokens : MonoBehaviour`, TypeDefIndex 5734. Its Character
reference is at `+0x20`; tagGood, tagExcl, tagUnsure and tagBad references are at
`+0x28`, `+0x30`, `+0x38` and `+0x40`. The exact methods are `tdi5734.m0000`
(`OnEnable`, `0x3971A0..0x3971FF`) and `tdi5734.m0001` (`Update`,
`0x397200..0x397436`), with exclusive end addresses. Both have the pinned `vii`
signature. Complete decode, raw backing, next managed entries and trailing
alignment bytes are verified; all 194 caller instructions execute. The folded
constructor and other classes sharing runtime entries are not promoted.

OnEnable disables Good, Unsure, Bad, then Exclamation in that order. Every field
is reloaded immediately before use; the final SetActive is a native tail call.
Earlier effects survive when a later reference reaches the native null guard.
It does not consume Character, its placement, or its hover flag.

Update first guards Character, then requires its **placement** DWORD at `+0xE8`
to equal Gameplay 10 and its hover byte at `+0x190` to be nonzero. The exact enum
is ECharacterPlacement, TypeDefIndex 5490, with None 0 and Browsing 20. This is
distinct from Character.state at `+0xE4`. The entry gates are checked once;
supplied callbacks changing Character, placement or hover after the first key
request do not suppress later requests.

The native key query order is 53, 49, 51, 50, 52. Each branch is independent,
and all five can execute in one Update:

| Key value | Ordered effect |
| --- | --- |
| 53 | Toggle Exclamation |
| 49 | Toggle Good, disable Unsure, disable Bad |
| 51 | Disable Good, disable Unsure, toggle Bad |
| 50 | Disable Good, disable Bad, toggle Unsure |
| 52 | Disable Good, Unsure, Bad and Exclamation |

These numeric values are exact supplied Unity key arguments; the audit does
not establish platform keyboard collection. Both GetKeyDownInt and GetKeyDown
metadata declarations bind the supplied native input entry. Decisions consume
only returned AL, including noncanonical and zero low bytes with nonzero upper
return bits. An independent values-only oracle verifies complete ordered
service arguments and final physical GameObject states for every returned
unmutated profile, including all 32 key masks and sequential alias writes.

Toggling captures activeSelf from one receiver, then reloads the field before
SetActive. A callback can therefore redirect the setter to another physical
GameObject while retaining the earlier active result. Clearing the field stops
at its native guard without the setter. Explicit disables zero all of RDX.
Good's `sete dl` retains the supplied upper register bits for both values;
Exclamation, Bad and Unsure retain upper bits when enabling, but clear EDX when
disabling. The harness verifies full RDX at each exact call site, zero MethodInfo
arguments, and exact zero-extended key arguments.

Every initial, service-entry and final snapshot retains complete authored
diagnostic byte windows for owner, Character, GameObjects and the opaque class
record. Character's `0x200` window and the other `0x80` windows are diagnostic
storage, not proven runtime object extents or valid unused typed fields. Native
callers do not change these records; callbacks can change only their declared
field/placement/hover bytes. Supplied active-state and key/read/write ledgers
are retained separately. All eight integer nonvolatile registers, XMM6–15 and
the native return stack are verified after normal completion; supplied services
poison integer/XMM volatile registers. Null-owner diagnostics require the exact
unmapped-read error and exact entry instruction, not an arbitrary emulator fault.

The final report contains **302 cases** (275 normal returns), four service-stop
baselines, **73 exact full stopped prefixes**, and two retained four-call
sequences. Each stopped final state equals its original service-entry snapshot,
and all earlier events equal the complete baseline prefix. Mutation fixtures
assert the captured/read and reloaded/write recipients, partial effects and
unchanged diagnostic bytes. This does not model engine exception unwinding.

Report memory windows use the frozen lossless `pool_memory`/`expand_memory`
helpers from the Oracle-to-RevealOrder audit. Every memory blob is SHA-256 keyed;
each producer asserts full round-trip equality after native checks. Two
successful independent final producers emitted byte-identical **7,118,113-byte**
reports, SHA-256
`ca3e5e6b32f71b5ebb5515116cbbe0876f6cc42373a481ec7c324a4536c1f1af`.
The private peer is
`B:/CodexTools/DemonBluffReverseEngineering/artifacts/f530404b0f3f_807de4a83df4/card_tokens_peer.json`.
An independent peer review checked the branch recipes and authored corpus.
Python syntax compilation and the 32 reverse-engineering infrastructure tests
pass; this checkpoint does not rerun solver simulations or live gameplay.

Source: [audit_card_tokens.py](../../scripts/audit_card_tokens.py).
Report: [card_tokens.json](../../reports/f530404b0f3f_807de4a83df4_card_tokens.json).

```powershell
python -m py_compile reverse_engineering/scripts/audit_card_tokens.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
$env:PYTHONPATH='B:/CodexTools/DemonBluffReverseEngineering/python-emulation'
python reverse_engineering/scripts/audit_card_tokens.py `
  --game-root 'B:/SteamLibrary/steamapps/common/Demon Bluff Playtest' `
  --dumper-root 'B:/CodexTools/DemonBluffReverseEngineering/artifacts/f530404b0f3f_807de4a83df4/il2cppdumper-v6.7.46' `
  --output 'B:/CodexTools/DemonBluffReverseEngineering/artifacts/f530404b0f3f_807de4a83df4/card_tokens_peer.json'
```

## Guarded Rust caller replay

The guarded Rust [CardTokens replay](notes/systems/card_tokens.md) compares
267 supported normal native profiles and two retained sequences in five tests.
Complete represented physical storage and supplied ledgers retain service-entry
chronology, ordered tag aliases and exact byte/register widths. Future snapshot
and log work validates before cloning; engine effects and failure paths are excluded.

Five guarded Rust tests compare 267 supported normal CardTokens native profiles and two retained four-call sequences against complete represented physical storage, service-entry snapshots, exact arguments, ledgers and final state. Independently supplied key/active low-byte results and volatile register bits preserve five ordered branch effects and physical aliases. Nominal identities, consumed references and complete future snapshot/log budgets validate before maps/clones; replay leaves input unchanged. Pointer mutation, native guard/failure paths, engine lifecycle/input/rendering and exception unwinding are excluded.

Source: [card_tokens.rs](../../../crates/solver-core/src/bluff/card_tokens.rs).
Five focused tests, all 877 Rust library tests, the release build and 32
reverse-engineering checks passed. The simulation and Python bridge suites
were not rerun for this offline caller reconstruction.
