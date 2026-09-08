# Clock callback in the default PlayerLoop

The [audit](../../scripts/audit_unityplayer_clock_phase.py) joins the clock
caller `0x5B72C0` to the fully qualified managed type
`UnityEngine.PlayerLoop.TimeUpdate/WaitForLastPresentationAndUpdateTime` in
UnityEngine.CoreModule.dll. It checks the type lookup, cache unwrapping/store
at offset `+0xA28`, and callback installation into cell `0x1CAA1E8`.

The native default-loop builder then independently pairs that type-cache tag
with the callback cell at node index 2 of 131. The existing five wait callbacks
retain indices 34, 56, 70, 89 and 113. Both mask-2 callbacks remain distinct
nodes, and the new clock binding does not resolve phase bit 8.

The [report](../../reports/f530404b0f3f_807de4a83df4_unity_clock_phase_audit.json)
records nine static relationships and execution of 973 distinct native builder
instructions. Type tags are synthetic, unique, initialized cache inputs; the
builder and type-array builder execute native instructions. Existing phase
auditing also reran successfully: 67 relationships and five wait bindings.

This identifies the default clock callback's constructed placement and links
it to the [source/reset audit](unity_clock_source.md). It does not assert an
unmodified live PlayerLoop, override the callback's provider/pause guards, or
infer recorded timestamps. Complete delayed-Reveal scheduling still requires
the relevant runtime provenance and remaining phase/lifetime boundaries.

The reusable builder harness now accepts an explicit selection of nodes to
check while preserving its original default selection. All 32 reverse-
engineering tests passed. This checkpoint changes offline audit tooling and
does not add Assembly-CSharp coverage or alter solver behavior.
