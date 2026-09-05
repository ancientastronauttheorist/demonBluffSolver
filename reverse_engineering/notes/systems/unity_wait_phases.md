# Default PlayerLoop wait-dispatch bindings

Evidence: `native-static` and `native-emulated` against UnityPlayer SHA-256
`B5D48235E7CC02FF9496FB33A07D5921ADFC4B40DED1BC64C96A7A7C10B4DFB2`.
This follows the [coroutine bridge](unity_coroutine_bridge.md),
[clock audit](unity_icall_bindings.md) and [wait consumer](unity_wait_boundary.md).

## Bound callbacks

| Qualified PlayerLoop node in `UnityEngine.PlayerLoop` | Mask passed to wait consumer | Native callback RVA | Default node index |
| --- | ---: | --- | ---: |
| `EarlyUpdate/ScriptRunDelayedStartupFrame` | `4` | `0x5b7ce0` | 34 |
| `FixedUpdate/ScriptRunDelayedFixedFrameRate` | `1` | `0x5b7d20` | 56 |
| `Update/ScriptRunDelayedDynamicFrameRate` | `2` | `0x5b7d50` | 70 |
| `PostLateUpdate/ScriptRunDelayedDynamicFrameRate` | `2` | `0x5b7d50` | 89 |
| `PostLateUpdate/PlayerSendFrameComplete` | `32` | `0x5b7af0` | 113 |

The fixed-step callback dispatches only when the byte at engine RVA `0x1cd5908`
is nonzero. PlayerSendFrameComplete dispatches only when the dword at
`0x1cd5904` is zero. The other three node bindings dispatch unconditionally
within their callback. These are recovered guard locations; no public identity
or complete lifecycle interpretation is assigned to the guard fields here.

Both dynamic-frame nodes use the same callback, which loads the queue owner
from `0x1c6e720`, passes mask `2`, and jumps through vtable slot `0xb8`. The
existing constructor/vtable audit binds that slot to consumer `0x43bd90`.

WaitForSeconds records use mask `0xA`, so they intersect the two recovered
mask-2 callbacks. The callbacks' different node positions matter: there can
be multiple consumer generations within a frame. The generation exclusion
alone therefore does not mean "wait until next frame". The separate signed
frame threshold still applies, including to zero and negative durations.

This establishes the constructed default-loop bindings, not actual dispatch
timestamps. Phase bit `8` remains unresolved; no claim is made that these two
callbacks exhaust every possible dispatch path. PlayerLoop replacement or
mutation, guard-state changes and full clock policy also remain outside scope.

## Binding chain and independent construction check

CoreModule cache initialization at `0x81a880` passes the full qualified node
name, `UnityEngine.PlayerLoop` namespace and `UnityEngine.CoreModule.dll`
assembly to lookup helper `0x75fc00`, unwraps the result and stores the selected
cache entry. Bare basename references elsewhere in the engine are not used
as evidence of a binding.

Initialization at `0x59bfe0` installs native callback addresses in the selected
callback cells. Default-loop builder `0x5a6810` calls type-array builder
`0x42f6b0`, then joins type entries and callback-cell addresses into `0x68`-byte
nodes. Some joins occur in loops, so the audit does not infer them merely from
nearby global addresses.

[`audit_unityplayer_phases.py`](../../scripts/audit_unityplayer_phases.py)
checks 67 native instruction/reference relationships and executes the two
builders in isolated Unicorn memory with an already-initialized synthetic type
cache. Every cache offset receives a unique tag. The native builders construct
131 nodes; the audit then requires each selected callback cell to occur exactly
once with its independently resolved type tag. The run executes 973 distinct
instructions across these builders. No OS calls, managed lookup implementations,
or callbacks execute in this emulation.

Four CI tests exercise the join validator with correct distinct cells, wrong
types, missing/duplicate cells, truncation and invalid counts. Native libraries
and game inputs are unnecessary for those tests.

## Reproduce and remaining scope

```powershell
python -m pip install unicorn==2.1.4 pefile capstone
python reverse_engineering/scripts/audit_unityplayer_phases.py `
  'B:\SteamLibrary\steamapps\common\Demon Bluff Playtest\UnityPlayer.dll' `
  --output reverse_engineering/reports/f530404b0f3f_807de4a83df4_unity_phases.json
python -m unittest discover -s reverse_engineering/tests
```

The [report](../../reports/f530404b0f3f_807de4a83df4_unity_phases.json) contains
selected bindings and authored results, without native bytes. Complete consumer
mutation/lifetime behavior is still needed before automatic continuation
admission. Managed coverage and live solver behavior are unchanged.

Validation passed the native audit, 778 Python tests, 29 reverse-engineering
tests, Python compilation and diff checks. Rust and simulations were not rerun
for this offline audit-only checkpoint.
