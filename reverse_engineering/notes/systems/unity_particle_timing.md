# Maximum particle delta and immediate timing writers

The pinned `UnityEngine.Time::set_maximumParticleDeltaTime` registration resolves
to RVA `0x10e330`. Its entire native body spans `0x10e330..0x10e35b` (exclusive
end), with ten instructions and one return. It loads the engine clock through
global `0x1c6e718` and writes only the float at `+0x104`.

Its exact rule is: choose the existing fixed delta at `+0x48` if that value is
greater than the argument; otherwise retain the argument bits. This is a strict
`COMISS`/`CMOVA` comparison, with no call to the general normalizer or reciprocal
refresh. For finite valid fixed delta it floors particle delta at fixed delta.
However, an unordered comparison retains the argument, including the exact sign
and payload of a quiet or signaling NaN. Positive infinity remains positive
infinity. With an invalid NaN fixed delta, even a negative or zero argument is
retained. Equal signed zeros retain the argument's sign. The public
`maximumDeltaTime` setter at `0x10e2e0..0x10e30b` uses the same operation, writing
`+0x100` instead.

The neighboring `fixedDeltaTime` setter at `0x10e240..0x10e2c9` clamps its
argument to approximately 0.0001 through 10, maps NaN to the minimum, refreshes
`+0x50/+0x58`, and floors `+0x100` at the selected fixed delta. It leaves particle
delta at `+0x104` untouched. Therefore increasing fixed delta from approximately
0.02 to 1 preserves a particle delta of approximately 0.03: this public setter
does not maintain the particle floor. It also preserves an existing NaN at
`+0x100`, because the final strict comparison is unordered.

The separate normalizer at `0x5520e0` repairs these cases: it sanitizes fixed
delta and maps NaN limits to the fixed floor, independently for both limits.
It preserves positive-infinite limits and does not refresh `+0x50/+0x58`.
The separate virtual refresh at `0x5520c0` then updates those cached fields.
These are verified effects of explicit operation sequences, not evidence that
startup or a PlayerLoop invokes the operations in that order.

## Verification

`reverse_engineering/scripts/audit_unityplayer_particle_timing.py` pins
UnityPlayer SHA-256
`b5d48235e7cc02ff9496fb33a07d5921adfc4b40ded1bc64c96a7a7c10b4dfb2`, verifies the
three exact setter registrations, both virtual slots, all clamp/reciprocal
constants, complete native instruction ranges and every instruction/operand in
the particle setter. Unicorn 2.1.4 executes all five routines and the native
NaN classifier without service stubs.

The audit passes 8,438 native invocations: 2,486 fixed setters, 2,484 maximum
delta setters, 2,488 particle setters, 490 normalizations and 490 refreshes.
Every invocation compares all 512 synthetic object bytes against a bit-aware
projection, verifies Windows nonvolatile integer and XMM registers, and checks
the return stack. All 118 decoded native instructions execute at least once.
Coverage includes a 22-by-22 matrix of selected current-fixed and argument bit
patterns, 2,000 seeded random fixtures per direct setter, six documented writer
sequences and a normalization/refresh matrix. Patterns include signed zero,
subnormals, finite extremes, both infinities, signed quiet/signaling NaNs and
adjacent clamp values.

The authored report records each sequence's exact intermediate field bits:
`reverse_engineering/reports/f530404b0f3f_807de4a83df4_unity_particle_timing_audit.json`.
Run with the local pinned reverse-engineering dependency directory on
`PYTHONPATH`:

```powershell
python reverse_engineering/scripts/audit_unityplayer_particle_timing.py 'B:\SteamLibrary\steamapps\common\Demon Bluff Playtest\UnityPlayer.dll' --output reverse_engineering/reports/f530404b0f3f_807de4a83df4_unity_particle_timing_audit.json
python -m py_compile reverse_engineering/scripts/audit_unityplayer_particle_timing.py
```

This is a bounded writer interaction audit. It does not enumerate all writers,
execute startup, mutate the live engine, or establish deserialization and
callback ordering. Tests use default MXCSR `0x1f80`; exception status flags,
unmasked exceptions, DAZ/FTZ and alternate rounding modes remain outside scope.
