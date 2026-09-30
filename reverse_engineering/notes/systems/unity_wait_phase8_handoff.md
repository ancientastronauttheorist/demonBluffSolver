# Phase-eight boundary follow-up handoff

This follow-up establishes additional instruction boundaries and complete chained
unwind ownership. It does **not** establish a phase-eight dispatcher or admission
order. The earlier inventory and wait-consumer evidence remain unchanged.

Reproducible script: `reverse_engineering/scripts/audit_unityplayer_wait_phase8_handoff.py`.
Report: `reverse_engineering/reports/f530404b0f3f_807de4a83df4_unity_wait_phase8_handoff.json`.
The script accepts the pinned UnityPlayer DLL, the existing phase8 inventory JSON,
and `--output`. Use the private emulation Python dependency directory on PYTHONPATH.
It verifies the engine SHA before interpreting any addresses. Outputs contain no
native bytes or reconstructed method bodies.

The 23 prior global loads belong to 18 chained-unwind families plus three already
verified PlayerLoop leaves. Following actual chained-unwind records, rather than
adjacency, produces 59 chunks and 3,024 decoded instructions. Every byte in each
selected chunk is consumed. Full-family memory operands at displacement +0xb8
occur at the two known unwind-backed dispatches and at 0x779605. The latter loads
an argument into RDX; the following indirect call takes its target from the
function-pointer global 0x1cd6310. The earlier RSI load names global 0x1c6e708.
These exact sites are asserted, but path-sensitive RSI provenance is not yet
automated. The argument load is not itself a loaded virtual call target.

Three previously unverified raw candidates now have independently supported leaf
entries:

| Raw candidate | Entry evidence | Receiver global | Branch instruction |
| --- | --- | --- | --- |
| 0x304a63 | aligned code pointer at 0x1932bf0 points to 0x304a50 | 0x1cd5710 | 0x304a62 |
| 0x6d0bcb | aligned code pointer at 0x196a7c8 points to 0x6d0bc0 | 0x1c6e728 | 0x6d0bca |
| 0xefaa40 | verified LEA at 0xefcb7b takes 0xefaa30; 0xefcb86 publishes it at +0x550 | 0x1cd1f28 | 0xefaa3f |

All three forward through receiver vtable slot +0xb8 without assigning EDX. Their
argument therefore remains caller-supplied at this boundary. Different global
addresses do not prove distinct runtime objects: their publication and alias
relations to wait-manager global 0x1c6e720 remain open. The first and third leaves
also have a null-return path. No phase mask is assigned to any of these leaves.

Raw candidate 0x1467e7f remains without a verified containing entry. Existing
Ghidra DumpContaining requests found no containing function for any of the four
raw candidates; that does not prove they are unreachable. No Ghidra project was
modified by this follow-up.

Remaining work is path-sensitive inter-chunk dataflow, including register-loaded
targets, stack aliases, computed or combined masks, forwarding-global publication,
and dispatcher callers. The expanded ownership inventory cannot replace those
proofs. No scheduler model or Rust phase behavior changed.
