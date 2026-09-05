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

All names above have the `UnityEngine.` prefix. The native
StartCoroutineManaged2 wrapper reaches creation helper `0x77BC80` on its valid
owner path. The complete creation and managed continuation bridge remain the
next audit boundary; a name binding alone does not reconstruct those methods.

## What the clock getters establish

The selected time/frame getters access the same engine object global
`0x1C6E718` used by the wait producer and consumer.

- Double field `0x90` backs both `Time.timeAsDouble` and, after conversion to
  float, `Time.time`. The queue consumer samples this field once per drain.
- The low 32 bits of field `0xC8` back `Time.frameCount`. The wait producer and
  consumer use a **64-bit** value at this offset; the consumer comparison is
  signed. This audit does not discard the upper bits or substitute a u32 gate.
- The low 32 bits at `0xD0` back `Time.renderedFrameCount`.
- Wait production still uses the distinct double field `0x60`. Its relationship
  to the public time getter remains unresolved, including fixed-step behavior.

This narrows the readiness contract without supplying a runtime clock snapshot,
identifying the queue's numerical phase masks, or establishing mutation-safe
equal-deadline dispatch. Existing replay APIs retain their explicit provenance.

## Reproduce

```powershell
python reverse_engineering/scripts/audit_unityplayer_icalls.py `
  'B:\SteamLibrary\steamapps\common\Demon Bluff Playtest\UnityPlayer.dll' `
  --output reverse_engineering/reports/f530404b0f3f_807de4a83df4_unity_icall_bindings.json
```

The audit pins the engine before decoding, checks 24 instruction/relative-
reference relationships and eight selected bindings, and validates all 3,447
table pairs. Four synthetic tests cover shared native aliases, malformed table
sizes, out-of-image/nonexecuting pointers, and duplicate/invalid names. Native
dependencies remain lazy imports and are unnecessary for those tests.

Validation passed 778 Python tests, 20 reverse-engineering tests, native audit,
Python compilation and diff checks. Rust/simulations were not rerun because
this checkpoint changes only offline audit tooling and documentation.

UnityPlayer functions remain outside the managed-method coverage denominator:
532 classified managed methods and 276 evidence records are unchanged.
