# Unity clock normalization and cached reciprocal

Pinned `UnityPlayer.dll` SHA-256:
`b5d48235e7cc02ff9496fb33a07d5921adfc4b40ded1bc64c96a7a7c10b4dfb2`.

The clock vtable at RVA `0x1954e38` contains two separate operations:

| Slot | Native range (end exclusive) | Observed operation |
| --- | --- | --- |
| `+0x10` | `0x5520c0..0x5520dc` | Copy fixed delta `+0x48` into `+0x50`; store single-precision `1 / fixedDelta` at `+0x58`. |
| `+0x20` | `0x5520e0..0x5521aa` | Normalize fixed delta `+0x48`, then floor the two limits `+0x100` and `+0x104` at the normalized delta. |

The second operation first maps NaN fixed delta to the exact binary32 value
`0x38d1b717` (approximately 0.0001), clamps smaller values to that minimum,
and clamps values above 10 (`0x41200000`) to 10. Positive infinity becomes
10; negative infinity and either zero become the minimum. Each limit is then
independently checked with the native NaN classifier: NaN is temporarily
replaced by positive zero and the result is floored at fixed delta. Consequently
NaN and negative infinity limits become fixed delta, while positive infinity
limits remain infinite. There is no upper cap on either limit and no coupling
between the two limits beyond their common floor.

The `+0x104` field is independently identified as `maximumParticleDeltaTime`:
the pinned internal-call registration table binds
`UnityEngine.Time::get_maximumParticleDeltaTime` to RVA `0x10e320`. Its entire
three-instruction body loads the clock pointer from global RVA `0x1c6e718`,
loads the float at `+0x104`, and returns at `0x10e32f`. The harness asserts the
registration and all three instructions through the exclusive `0x10e330`
boundary. It also verifies registration of the corresponding setter at
`0x10e330`; the [follow-up audit](unity_particle_timing.md) covers that setter
and its interaction with fixed-delta writes.

Normalization preserves the cached fields `+0x50` and `+0x58`. Refresh preserves
`+0x48` and both limits; it does no clamping or NaN rejection. Refresh copies the
input bits unchanged to `+0x50`, including a signaling NaN payload. Its division
quietens signaling NaN at `+0x58` while preserving the payload and sign in these
fixtures. Signed zero gives correspondingly signed infinity, signed infinity
gives signed zero, and very small subnormals can overflow the reciprocal to
infinity. These are default-MXCSR results, not a promise for altered floating
point control modes.

## Verification

`reverse_engineering/scripts/audit_unityplayer_clock_normalization.py` checks
the pinned file hash, native vtable pointers, all three literal constants,
complete instruction boundaries through both returns and the native NaN
classifier, and nine critical opcode/operand assertions. Unicorn 2.1.4 executes
the native instructions with synthetic 512-byte objects. Every object byte is
compared against an independent projection, including randomized unrelated
bytes; Windows nonvolatile integer and XMM registers and the return stack are
also checked. No native service is stubbed.

The committed report records 14,673 passing cases: 2,022 refresh and 12,651
normalization cases. Normalization covers the complete three-input Cartesian
product of 22 selected binary32 patterns, plus 2,000 seeded random patterns and
three documented examples. Refresh covers all selected patterns and 2,000
random patterns. Cases include both signed zeros, finite extremes, minimum
normal and subnormal values, both infinities, signed quiet and signaling NaNs,
and adjacent binary32 values around both clamp limits. All decoded instructions
in both operations and the classifier execute at least once. Native bodies and
Ghidra exports remain private.

Run with the pinned local dependency directory on `PYTHONPATH`:

```powershell
python reverse_engineering/scripts/audit_unityplayer_clock_normalization.py 'B:\SteamLibrary\steamapps\common\Demon Bluff Playtest\UnityPlayer.dll' --output reverse_engineering/reports/f530404b0f3f_807de4a83df4_unity_clock_normalization_audit.json
python -m py_compile reverse_engineering/scripts/audit_unityplayer_clock_normalization.py
```

## Remaining boundary

This audit identifies state effects of two virtual slots; it does not establish
when their callers invoke them, whether the lifecycle invokes refresh after
normalization, or when the engine consumes `maximumParticleDeltaTime`. No live object
was modified. Exception status flags, unmasked exceptions, nondefault rounding,
DAZ/FTZ, and full engine execution are outside this bounded audit.
