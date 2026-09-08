# Public Unity timing setters

Pinned engine SHA-256 is unchanged. The
[audit](../../scripts/audit_unityplayer_clock_setters.py) independently resolves
four internal-call registrations, executes their native setters and the native
NaN classifier, and compares every supplied clock-object byte in 72 cases.
The [report](../../reports/f530404b0f3f_807de4a83df4_unity_clock_setters_audit.json)
includes negative values, signed zero, clamp boundaries, infinities and NaN.

| Public setter | Recovered behavior |
| --- | --- |
| fixedDeltaTime (`0x10E240`) | NaN and values below `float(0.0001)` become that minimum; larger values cap at 10. Writes fixed delta at `+0x48`, the float at `+0x50`, and its reciprocal at `+0x58`. Raises maximumDeltaTime if the new fixed step exceeds it. |
| maximumDeltaTime (`0x10E2E0`) | Stores the requested value unless the current fixed step is strictly greater, in which case it stores the fixed step. NaN requests pass through the unordered comparison. |
| timeScale (`0x10E370`) | Rejects NaN and negative values through the error-log path, leaving clock fields unchanged. Accepts both signed zeros, positive finite values and positive infinity without an upper clamp. |
| captureDeltaTime (`0x10E600`) | Stores the supplied float directly at `+0xD8`, including negative and nonfinite values. |

The fixed-step setter's minimum is **0.0001**, distinct from the frame updater's
**0.00001** elapsed-time floor. It retains fixed unscaled delta (`+0x4C`) and the
current frame/public timing snapshots. Its maximum-step adjustment is one-sided:
reducing fixed delta does not reduce an existing larger maximum.

These setters therefore do not guarantee finite clock policy. The offline
Rust clock projection intentionally remains a finite-input contract rather
than treating NaN, infinity or alternate floating-point modes as ordinary time.
Direct runtime fields can also come from other configuration writers; public
setter restrictions should not be applied indiscriminately to every native
clock snapshot.

The harness uses successful short-string formatting/logging gateways for
rejected timeScale inputs. The optional object-notification subsystem after a
successful timeScale write is disabled, so its further side effects remain
outside this audit. Native return stack and nonvolatile registers are checked;
no DLL is loaded into the host process and no live engine settings are changed.

This offline audit does not change Rust behavior. Python compilation and diff
checks pass; the preceding solver checkpoint remains validated by 669 Rust
tests, 34 simulations over 426 fixtures, 778 Python tests and a release build.
