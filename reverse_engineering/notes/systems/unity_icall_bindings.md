# UnityPlayer internal-call registration and clock bindings

Evidence level: **native-static**, pinned to the UnityPlayer SHA-256 in
[the wait-boundary audit](unity_wait_boundary.md). This audit does not use live
process state or change the solver.

## Registration proves the table pairing

Routine `0xFA1110` obtains the image base through an image-relative LEA, starts
a byte offset at zero, then loads a function pointer from table `0x1894FC0`
and a name pointer from table `0x189BB80` using that same offset. It passes
`(name, function)` to each registered observer and to the common registration
sink, advances the offset by eight, and repeats exactly `0xD77` (3,447) times.
The sink is the same pointer populated by the loader's named
`il2cpp_add_internal_call` resolution at `0x76CC59..0x76CC65`.

This independently resolves the table pairing that the earlier wait audit
left open. The preceding pointer region is not the function-table start, and
counting apparently non-null pointers past the table end is not a valid way
to determine the registration count. Image-base-indexed memory operands also
explain why a RIP-relative-only reference scan missed these arrays.

The reproducible audit validates all 3,447 name/target pairs: names must be
bounded file-backed strings, target RVAs must be in executable sections, and
names must be unique. Shared native targets remain legal; they do not collapse
registered name identities. Only the following selected bindings are emitted
in the public report.

| Registered name suffix | Index | Native RVA |
| --- | ---: | --- |
| `MonoBehaviour::StartCoroutineManaged` | 2273 | `0x1007F0` |
| `MonoBehaviour::StartCoroutineManaged2` | 2274 | `0x100CE0` |
| `Time::get_time` | 2476 | `0x10E100` |
| `Time::get_timeAsDouble` | 2477 | `0x10E120` |
| `Time::get_fixedTime` | 2481 | `0x10E180` |
| `Time::get_fixedTimeAsDouble` | 2482 | `0x10E1A0` |
| `Time::get_frameCount` | 2498 | `0x10E510` |
| `Time::get_renderedFrameCount` | 2499 | `0x10E520` |
| `Time::get_fixedDeltaTime` | 2489 | `0x10E230` |
| `Time::get_timeScale` | 2496 | `0x10E360` |
| `Time::get_inFixedTimeStep` | 2504 | `0x10E610` |

All names above have the `UnityEngine.` prefix. The native
StartCoroutineManaged2 wrapper reaches creation helper `0x77BC80` on its valid
owner path. The complete creation and managed continuation bridge remain the
next audit boundary; a name binding alone does not reconstruct those methods.
The subsequent [coroutine bridge audit](unity_coroutine_bridge.md) recovers the
normal valid-owner creation, managed-step and WaitForSeconds path.

## What the clock getters establish

The selected time/frame getters access the same engine object global
`0x1C6E718` used by the wait producer and consumer.

- Double field `0x90` backs both `Time.timeAsDouble` and, after conversion to
  float, `Time.time`. The queue consumer samples this field once per drain.
- The low 32 bits of field `0xC8` back `Time.frameCount`. The wait producer and
  consumer use a **64-bit** value at this offset; the consumer comparison is
  signed. This audit does not discard the upper bits or substitute a u32 gate.
- The low 32 bits at `0xD0` back `Time.renderedFrameCount`.
- Wait production uses the distinct frame clock at double field `0x60`. The
  selector below copies this clock into the public time snapshot outside a
  fixed step, and copies fixed time into that snapshot during a fixed step.
- Field `0x48` backs fixedDeltaTime, `0xFC` backs timeScale, and byte `0xF9`
  backs inFixedTimeStep. All three getters use the same engine object.

## Frame versus fixed-step clock selection

Native selector `0x5A6690` calculates `fixed_time + double(fixed_delta)`, using
fields `0x30` and `0x48`. For finite values, if this candidate is **greater than**
the frame clock at `0x60` and the initial-fixed-step flag at `0xC2` is false,
it copies the frame timing block `0x60..0x8F` to public timing block
`0x90..0xBF`, clears inFixedTimeStep and returns false. Equality takes the fixed
branch. The native unordered comparison also takes that branch; the finite
Rust eligibility projection does not model that floating-point case.

Otherwise the selector records the previous fixed time at `0x38`. It advances
fixed time by the promoted float delta only when `0xC2` was false. The first
fixed step therefore uses the existing fixed time and then clears that flag.
It also updates the unscaled fixed fields when timeScale is nonzero, copies
fixed timing block `0x30..0x5F` to public timing block `0x90..0xBF`, sets
inFixedTimeStep and returns true. The frame clock at `0x60` is retained through
both selector branches.

Clock-update routine `0x551D70` increments the native 64-bit frame counter and
32-bit rendered-frame counter. Its frame-update path writes `0x60` and copies
the frame timing block into the public timing block. Full time accumulation,
capture settings, clamping, initialization and timeScale policy remain outside
this audit. The checked selection relationship is sufficient to reject the
assumption that producer `0x60` and consumer `0x90` are always interchangeable.

The selector was a leaf function not recognized by the first private Ghidra
export. Defining its independently verified entry in a read-only analysis
session produced a complete export, consistent with the direct native audit.
Per-target export success is now required explicitly by AGENTS.md.

This narrows the readiness contract without supplying a runtime clock snapshot,
identifying the queue's numerical phase masks, or establishing mutation-safe
equal-deadline dispatch. Existing replay APIs retain their explicit provenance.

## Reproduce

```powershell
python reverse_engineering/scripts/audit_unityplayer_icalls.py `
  'B:\SteamLibrary\steamapps\common\Demon Bluff Playtest\UnityPlayer.dll' `
  --output reverse_engineering/reports/f530404b0f3f_807de4a83df4_unity_icall_bindings.json
```

The audit pins the engine before decoding, checks 52 instruction/relative-
reference relationships and 11 selected bindings, and validates all 3,447
table pairs. Four synthetic tests cover shared native aliases, malformed table
sizes, out-of-image/nonexecuting pointers, and duplicate/invalid names. Native
dependencies remain lazy imports and are unnecessary for those tests.

Validation passed 778 Python tests, 20 reverse-engineering tests, native audit,
Python compilation and diff checks. Rust/simulations were not rerun because
this checkpoint changes only offline audit tooling and documentation.

UnityPlayer functions remain outside the managed-method coverage denominator:
532 classified managed methods and 276 evidence records are unchanged.
