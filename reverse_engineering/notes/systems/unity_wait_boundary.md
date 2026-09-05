# UnityPlayer wait producer and deadline insertion

This native-static audit follows the shipped engine's WaitForSeconds diagnostic
into a wait-record producer and its deadline-tree insertion helper. It does
**not** close the scheduler readiness or dispatch-order contract.

## Artifact identity and reproducibility

The local shipped `UnityPlayer.dll` is 30,671,384 bytes with PE timestamp
1694569221, preferred image base `0x180000000`, and SHA-256
`B5D48235E7CC02FF9496FB33A07D5921ADFC4B40DED1BC64C96A7A7C10B4DFB2`.
This is a separate engine fingerprint from GameAssembly and metadata. Existing
build notes identify Unity 2022.3.10f1; offsets below are pinned to this exact
engine binary, not every player built with that Unity version.

Reproduce the 15 checked instruction-semantic/constant relationships with:

```powershell
python reverse_engineering/scripts/audit_unityplayer_wait.py `
  'B:\SteamLibrary\steamapps\common\Demon Bluff Playtest\UnityPlayer.dll' `
  --output reverse_engineering/reports/f530404b0f3f_807de4a83df4_unity_wait_boundary.json
```

The audit uses local pefile and Capstone, verifies the engine hash before
decoding, checks executable-section membership, and writes only authored
findings/addresses and verification status. Dependencies load only for native
auditing; the three redistributable helper tests use synthetic data. No game
process is read or mutated, and no new Ghidra project was created.

## Located boundary

The internal-call name `UnityEngine.MonoBehaviour::StartCoroutineManaged2`
is stored at RVA `0x18BFE08`, with a data reference at `0x18A0290`. Adjacent names
include StopCoroutineManaged and StopCoroutineFromEnumeratorManaged. This
confirms an engine-side naming surface; it does not bind a function pointer.
A neighboring pointer region did not yield a verified one-to-one mapping.
Positional pairing with that region is rejected as evidence.

The WaitForSeconds NaN diagnostic string at `0x1978260` has a direct reference
at `0x779419`. The PE exception/unwind entry spans `0x7793F3..0x7794FE`;
this is a native **chunk boundary**, not necessarily the full containing
function. The chunk reads a float at yielded-object offset `0x10`, matching
the managed WaitForSeconds duration field, and promotes it to double. The
diagnostic path rejoins the common wait-record construction path.

The common path reads an engine-object double at offset `0x60`, applies the
maximum-finite-double cap stored at RVA `0x1A72FD8`, and adds the promoted float
duration. It stores the resulting double as the first field of a temporary
record. It also reads an engine-object 64-bit value at offset `0xC8`, increments
it by one, and stores it in the record. **The owning engine type and exact
meaning of these fields have not been independently resolved.** Calling them
scaled time and frame count would currently be an inference, not a recovered
field identity.

Two callback addresses, `0x778B30` and `0x778BD0`, are stored in that record.
The first compares ownership-like state and can tail-call `0x778D90`; the
second decrements a counter before continuing into another native chunk.
Resume/release are candidate interpretations only. The queue consumer and
callback invocation protocol have not yet been traced.

## Deadline insertion

The producer calls `0x440F00`. This helper allocates a 0x60-byte node, copies
the 0x40-byte input record to node offset 0x20, initializes tree links, and
walks a tree by comparing the existing node's double key with the new key.
For ordered values, an existing key greater than the new key takes the left
link; less-than-or-equal takes the right link. The unordered floating-point
branch also takes the right link. It then calls the tree-link/balance helper
at `0x366CB0`.

This establishes a deadline-keyed insertion operation and its equality branch.
It does not establish equal-deadline FIFO execution: removal, traversal,
readiness predicates, cancellation, frame gates, queue swaps and invocation
ordering remain untraced. The bounded ready-batch explorer therefore continues
to require explicit sealed-ready provenance and assigns no schedule weights.

## Next native work and coverage

Trace the owner of the engine time/counter globals, the queue consumer and its
eligibility comparisons, and then the callback dispatch chain. Independently
resolve the StartCoroutineManaged2 binding rather than using table proximity.
Keep any subsequently recovered engine-specific order separate from Unity's
public portability guarantees and from observed live event provenance.

UnityPlayer methods are outside the existing Assembly-CSharp denominator, so
this audit does not add managed classifications. Coverage remains 532 classified
methods and 276 evidence records. The engine report is separately fingerprinted;
no replay/live solver behavior was changed by these findings.

Validation passed all 15 pinned native checks, 778 Python tests, 16 reverse-
engineering tests, Python compilation, diff checks and coverage integrity.
Rust and simulation suites were not rerun because this checkpoint changes
only audit tooling, evidence and documentation. AGENTS.md now also requires
resolving uncertain script/test filenames before reading after failed guessed
paths during this audit.
