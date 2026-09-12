# Mask-16 wait-manager lifecycle

The pinned UnityPlayer routine at `0x59F5E0..0x59F743` calls wait-manager
vtable slot `+0xB8` with mask `16` at `0x59F692`. This is unconditional within
the routine assuming its earlier callees return normally. The optional service
at global `0x1C6E728` gates only its own preceding call: its null branch lands
on the wait-manager load, so it does not suppress dispatch.

The earlier `0x59F67B` boundary is an unwind chunk inside this method, not its
entry. Six contiguous unwind chunks cover the complete method. The native
method also empties an optional tree through a mocked cleanup helper and native
sentinel writes, invokes further services, and finally writes clock `+0xF0`
to the sign-bit negation of clock `+0x60` and sets clock bytes `+0xC0/+0xC1`
to one. The clock object comes from the established global `0x1C6E718`.
The second argument equal to `2` skips a later call at `0x59F708`; it never
skips mask-16 dispatch. These observations describe a reset/transition
boundary without assigning a public lifecycle name.

Three direct native call sites were found and verified on decoded instruction
boundaries:

| Containing method | Call sites | Recovered gate |
| --- | --- | --- |
| `0x5C62D0` | `0x5C63B4`, `0x5C6403` | Input mode at `+0x3E0` equals `2` or `0`, respectively. Modes `1`, `5`, and other values skip these calls. |
| `0xBD81B0` | `0xBD84F1` | Nested inside the object byte `+0x78 != 0` path; second-argument mode at `+8` must equal zero. |

Both caller families also invoke `0x59EFC0` earlier in their selected mode-zero
path; `0x5C62D0` does so for mode two as well. No verified public identity is
assigned to those modes. In particular, the evidence does not establish a
pause, manual-step, ordinary per-frame, or named scene-loading callback.
The direct-call search is not proof against indirect callers.

## Reproduction and limits

[`audit_unityplayer_wait_dispatch16.py`](../../scripts/audit_unityplayer_wait_dispatch16.py)
requires the pinned UnityPlayer SHA-256
`B5D48235E7CC02FF9496FB33A07D5921ADFC4B40DED1BC64C96A7A7C10B4DFB2`,
Capstone, pefile, and Unicorn 2.1.4. It checks 33 instruction/reference/constant
relationships, all six unwind bounds, and executes the complete enclosing
method in 32 cases spanning nullable tree/service, modes `0/1/2/5`, and
clock values `0/123.25`. Every case observes exactly one mask-16 dispatch and
the native final clock writes, including negative zero. Optional service and
tree cleanup calls and the mode-two notification exclusion are also checked.
All callees are explicit mocked gateways; their side effects and exceptions
are outside this audit. The two caller methods receive static checks only.

```powershell
python reverse_engineering/scripts/audit_unityplayer_wait_dispatch16.py `
  'B:\SteamLibrary\steamapps\common\Demon Bluff Playtest\UnityPlayer.dll' `
  --output reverse_engineering/reports/f530404b0f3f_807de4a83df4_unity_wait_dispatch16.json
```

The [report](../../reports/f530404b0f3f_807de4a83df4_unity_wait_dispatch16.json)
contains authored results only. Native private decompilations completed for
entries `0x59F5E0`, `0x5C62D0`, and `0xBD81B0`; none are checked in.

This extends the [previous mask-16 boundary](unity_clock_phase.md) and
[default wait-phase bindings](unity_wait_phases.md). It does not resolve bit
`8`: WaitForSeconds mask `0xA` does not intersect `0x10`. It does not establish
live timestamps, complete engine lifecycle provenance, helper semantics, or
scheduler admission. No solver or managed coverage changes are made.
