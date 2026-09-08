# Unity timestamp source, construction and reset

Same pinned UnityPlayer SHA-256 as the [clock arithmetic audit](unity_clock.md).
The [source harness](../../scripts/audit_unityplayer_clock_source.py) passes
62 native cases with explicit OS, thread-initialization, allocation and reset
tail gateways. The [report](../../reports/f530404b0f3f_807de4a83df4_unity_clock_source_audit.json)
also records eight static relationships in the outer frame caller.

## Timestamp chain

`0x12F78D0` calls the import-table entry at `0x1825680`, independently identified
as KERNEL32.QueryPerformanceCounter. It subtracts the stored 64-bit origin with
wrapping arithmetic, converts the result as **unsigned** to double, multiplies
by a runtime conversion factor at `0x1CC58D8`, then divides by the file-backed
double constant 1,000,000,000. The factor's effective units are nanoseconds per
counter tick. Its two calibration initializers are audited below.

The unsigned conversion is significant when the supplied counter precedes the
origin or wraps. That path produces a large positive elapsed value rather than
clamping or returning a negative time. The native harness covers both sides of
the signed boundary and values above exact integer representation in a double.

A TLS epoch and process guard protect first-use origin initialization. When
the thread-header service grants initialization, the routine reads one counter
into the stored origin, invokes the thread-footer service, then performs a
second counter read for the returned timestamp. When initialization was already
completed by another participant, the header path skips origin replacement.
The harness supplies those service outcomes; it does not model thread races.

The baseline initializer `0x5CE5B0` stores one native timestamp into its supplied
double allocation and returns that allocation. The normal outer frame caller
`0x5B72C0` lazily resolves this baseline, gets another timestamp and subtracts
the baseline in double precision. An alternate provider branch calls virtual
slot `+0x690` instead. The provider and the flags choosing it remain separate.

The caller forwards the resulting double from XMM0 to XMM1 and invokes the
clock object's virtual slot `+0xB8`. Constructor vtable `0x1954E38` places the
audited updater `0x551D70` at precisely that slot. A caller-level suppression
branch can skip this invocation entirely, unlike updater flag `+0xF8`, which
still allows frame-counter increments. The caller's final tail service runs
after either path. Only these caller ABI/order relationships are statically
checked here; the complete pause/mode/provider branches are not emulated.

The first decompiler rendering represented this double as a truncated
undefined value. Native XMM argument forwarding establishes the actual width;
the rendered C is not the authority for this ABI boundary.

## Frequency calibration

The native pointer table at `0x182BBA0` / `0x182BBA8` identifies two adjacent
initializer entries, `0x3FE70` and `0x3FEB0`. The first zeroes a local counter,
calls the verified QueryPerformanceFrequency import, stores the returned output
value and stores the integer numerator 1,000,000,000. The second converts both
integers as unsigned to double, divides numerator by frequency and publishes
the factor used by the timestamp routine. Their execution order is a required
startup dependency; this audit does not replay the whole global initializer list.

The first routine does not branch on the OS service Boolean. A zero output
therefore produces positive infinity during factor calculation under the
audited default floating-point environment. Seven native cases verify ordinary
and large unsigned frequencies plus failure-service outcomes. This recovers
the native calibration arithmetic without asserting such failures occur during
normal play. The finite Rust clock contract continues to reject nonfinite input.

## Reset behavior

`0x551F90` zeros frame current, previous and unscaled time, smoothing fields,
fixed current/previous/unscaled time, both frame counters, captureDeltaTime and
the eight bytes at `+0xF0`. It sets all three one-shot flags at `+0xC0..+0xC2`.

With the initialization argument true, frame delta and unscaled delta become
`float(0.02)` and reciprocal delta becomes 50. With it false, frame delta and
reciprocal become zero, while the old **unscaled delta is retained**. Fixed
delta is retained in either mode; reset copies it to fixed unscaled delta and
stores a float reciprocal. The finite harness uses nonzero fixed steps.

The frame block is copied to public timing before reset samples the current
process-relative timestamp. That timestamp is then stored in both scaled and
unscaled offsets. The routine finally tail-calls a separate service with the
global at `0x1C706E8`. That service's effects are not reconstructed here.

Reset preserves timeScale, maximumDeltaTime, suppression/inFixedTimeStep flags,
fixed smoothing fields and opaque timing-block tail bytes. It is not a full
zero-initialization operation. Twelve native cases vary reset mode, baseline
availability and fixed step, comparing all 272 supplied clock-object bytes.

## Construction is a distinct boundary

The actual constructor is `0x551630`; nearby `0x551600` is a deleting-destructor
path and must not be treated as initialization merely because it writes a
vtable. The constructor installs the clock vtable, initializes selected object
header/timing fields, masks the supplied allocation flags, sets fixedDeltaTime
to `float(0.02)`, clears 800 bytes starting at `+0x110` and calls reset(true).

Four native cases verify every byte of the 1,072-byte object and its returned
identity. Fields not explicitly written retain their supplied bytes, including
timeScale and maximumDeltaTime. Their initialization depends on other engine
configuration paths; this constructor alone does not establish usable defaults.

All proprietary instructions remain private. The repository contains authored
assertions, synthetic inputs and numeric results. The existing Rust clock
projection still accepts explicit state and timestamps; this audit does not
silently infer startup state or add a complete PlayerLoop driver.
