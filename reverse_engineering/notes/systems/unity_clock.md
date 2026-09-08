# Native Unity frame-clock update and fixed selection

Engine SHA-256:
`B5D48235E7CC02FF9496FB33A07D5921ADFC4B40DED1BC64C96A7A7C10B4DFB2`.
The [audit](../../scripts/audit_unityplayer_clock.py) executes frame updater
`0x551D70` and fixed selector `0x5A6690` in Unicorn 2.1.4. Its
[report](../../reports/f530404b0f3f_807de4a83df4_unity_clock_audit.json) checks
1,437 updates and 180 selections against complete synthetic clock objects.
No external calls occur inside these two bounded routines.

## Frame update order

The updater first increments the 64-bit frame counter at `+0xC8` and 32-bit
rendered counter at `+0xD0`, with native wrapping. A nonzero `+0xF8` flag then
returns immediately. Thus frame advancement alone does not prove time changed.

Otherwise it subtracts the unscaled offset `+0xE8` from the supplied double
timestamp. The difference from the old frame unscaled time is rounded to float.
If that float is at least `float(0.00001)`, it publishes the new unscaled time;
otherwise it retains the old unscaled time. The unscaled delta is floored at
that same constant. These writes happen before the next early return.

Scaled frame-time selection follows this precedence:

1. Positive captureDeltaTime (`+0xD8`): add the float-rounded product of capture
   delta and timeScale to the old double frame time.
2. Nonzero `+0xC0`: clear that one-shot flag and return. This retains the earlier
   counter and unscaled writes, but does not publish a new public timing block.
3. Nonzero first-frame flag `+0xC1`: add float-rounded `timeScale * 0.02`.
4. Otherwise form a double candidate from timestamp minus scaled offset
   `+0xE0`, and compare its elapsed difference against maximumDeltaTime. Above
   the maximum, add float-rounded `maximumDeltaTime * timeScale`. Below the
   minimum, add float-rounded `timeScale * float(0.00001)`.
5. Within those bounds, a float-rounded distance of timeScale from one no
   greater than `float(0.000001)` preserves the double candidate directly.
   Other scales round elapsed to float before multiplication, then promote
   the product and add it to old time.

Capture therefore bypasses the one-shot skip without clearing it. The clamp
precedes scaling. Negative scales and maximum values are accepted by this
routine when supplied directly; their setter policies are separate boundaries.

The updater stores current and previous frame times and a float delta. It
stores reciprocal delta only when delta exceeds the minimum, otherwise one.
Smoothing uses float operations: weight becomes `oldWeight * 0.8 + 0.2`, blend
is `0.2 / weight`, and smooth delta is the weighted old smooth/current delta.
The exact float rounding points are preserved by the differential oracle.

It copies all 48 bytes of the frame block (`+0x60..+0x8F`) to the public block
(`+0x90..+0xBF`), then updates scaled offset to timestamp minus new frame time.
Finally a set first-frame flag is cleared and the frame smoothing weight is
zeroed. The already-copied public weight retains its pre-reset value. This
cleanup also occurs when capture took precedence over the first-frame branch.

The four public getter registrations independently confirm deltaTime at
`+0xA8`, smoothDeltaTime at `+0xB0`, maximumDeltaTime at `+0x100`, and
captureDeltaTime at `+0xD8`, all through the same engine singleton.

## Fixed selection and delayed Reveal

The selector compares fixed time plus promoted fixedDeltaTime against frame
time. A later candidate selects the frame block unless the first-fixed flag
is set. Equality selects fixed time. The first-fixed path retains the existing
fixed time; subsequent steps advance it. With nonzero timeScale, fixed unscaled
time is derived from the fixed/frame difference divided by scale plus frame
unscaled time. Zero scale preserves both old fixed unscaled fields.

It copies the selected 48-byte block to public time and sets inFixedTimeStep.
It never changes the retained frame block. Consequently a new WaitForSeconds
deadline uses the frame clock even when the consumer's public snapshot is
currently fixed. A suppressed frame update can satisfy a wait's next-frame
gate while leaving its time gate unchanged. Passing both gates still depends
on phase, generation and owner/callback state in the other scheduler modules.

## Offline reconstruction and precision

[`bluff::clock`](../../../crates/solver-core/src/bluff/clock.rs) exposes the
versioned `unity_clock_native_v1` contract for explicit finite clock states and
timestamps. It returns updated state and the selected path, rejects nonfinite
inputs/results, preserves opaque timing-block tail bits and uses full-width
wrapping counters. Errors do not mutate the caller's state. It is an offline
projection; no live or scenario caller is introduced.

Seven Rust tests compare native full-state fixtures, connect clock snapshots
to wait gates, verify version/finite guards and preserve JSON timestamp bits.
The default JSON parser shifted some fixture doubles by one representable
step. Enabling serde_json's `float_roundtrip` feature resolves that discrepancy;
the native comparisons remain exact rather than accepting a tolerance.

The native audit checks all object bytes, normal return stack, nonvolatile
registers, eight constants and four getter registrations. Its finite matrix
uses explicit policies, deterministic randomized fields, counter rollover and
early-return precedence cases with default round-to-nearest MXCSR. Native
NaNs, alternate floating-point environments, timestamp production, clock
initialization, policy setters and complete PlayerLoop composition remain open.
UnityPlayer is outside the Assembly-CSharp coverage denominator.

The subsequent [clock source audit](unity_clock_source.md) traces QPC conversion,
the process-relative baseline, construction and reset through 62 native cases.
Provider/pause callbacks, remaining configuration writers
and full-loop sequencing remain outside the combined boundary.

Validation passed all 669 Rust library tests, all 34 simulation tests covering
the unchanged 426-fixture corpus (992.37 seconds), 778 Python tests and
32 reverse-engineering tests. The release build also passed.

The [public setter audit](unity_clock_setters.md) subsequently verifies 72
native cases. It distinguishes fixedDeltaTime's 0.0001 floor from this updater's
0.00001 floor and confirms that several public setters admit nonfinite values.
