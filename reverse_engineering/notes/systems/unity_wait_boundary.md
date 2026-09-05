# UnityPlayer wait producer, deadline insertion and consumer

This native-static audit follows the shipped engine's WaitForSeconds diagnostic
into a wait-record producer, its deadline-tree insertion helper and the native
queue consumer. It closes the consumer's local eligibility comparisons, but
does **not** yet close automatic Reveal scheduling or engine field identities.

## Artifact identity and reproducibility

The local shipped `UnityPlayer.dll` is 30,671,384 bytes with PE timestamp
1694569221, preferred image base `0x180000000`, and SHA-256
`B5D48235E7CC02FF9496FB33A07D5921ADFC4B40DED1BC64C96A7A7C10B4DFB2`.
This is a separate engine fingerprint from GameAssembly and metadata. Existing
build notes identify Unity 2022.3.10f1; offsets below are pinned to this exact
engine binary, not every player built with that Unity version.

Reproduce the 52 checked instruction-semantic/constant relationships with:

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
Positional pairing with that region is rejected as evidence. The subsequent
[registration-loop audit](unity_icall_bindings.md) establishes the correct
table bases/count and resolves StartCoroutineManaged2 to `0x100CE0`.

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
field identity. The subsequent registration audit identifies consumer field
`0x90` as the double backing Time.time and the low 32 bits of `0xC8` as
Time.frameCount; producer field `0x60` remains distinct and unresolved.

Two callback addresses, `0x778B30` and `0x778BD0`, are stored in that record.
The consumer below establishes their dispatch/release slots. The dispatch
callback compares payload field `0x58` against the resolved owner and, when
equal, tail-calls `0x778D90` with the payload and a zero second argument. The
release callback decrements payload field `0x60`; a nonpositive result enters
cleanup that unlinks references and can free the 0x88-byte payload. The full
continuation dispatcher and managed MoveNext bridge remain untraced.

## Deadline insertion

The producer calls `0x440F00`. This helper allocates a 0x60-byte node, copies
the 0x40-byte input record to node offset 0x20, initializes tree links, and
walks a tree by comparing the existing node's double key with the new key.
For ordered values, an existing key greater than the new key takes the left
link; less-than-or-equal takes the right link. The unordered floating-point
branch also takes the right link. It then calls the tree-link/balance helper
at `0x366CB0`.

This establishes a deadline-keyed insertion operation and its equality branch.
The consumer establishes in-order traversal, but a complete equal-deadline
FIFO claim still requires the tree linking/balancing and erasure helpers,
callback mutations and all relevant producers to be accounted for. The bounded
ready-batch explorer continues to require explicit sealed-ready provenance.

## Queue consumer and local eligibility

Constructor `0x43B700` assigns vtable `0x1941950`, initializes the tree at
owner offset `0x30`, zeroes the dispatch generation at `0x48`, and stores the
sentinel as the initial traversal cursor at `0x40`. Vtable slot `0xB8` points
to consumer `0x43BD90`. Engine call sites dispatch this slot with phase masks
including 1, 2, 4, 16 and 32. The WaitForSeconds producer stores mask 10
(`0xA`), so intersection admits masks containing bit 2 or bit 8. Public phase
names are not assigned from these numbers alone.

The consumer samples engine field `0xC8` into a 64-bit register and **field
`0x90`** into a double register, then increments its 32-bit generation. Note
that producer time came from field `0x60`: equating those two fields would be
an unsupported simplification. It starts at the tree's leftmost node and:

1. Stops traversal when sampled field `0x90` is less than the deadline, or the
   floating-point comparison is unordered (`COMISD` followed by `JB`).
2. Calculates and saves the in-order successor before invoking any callback.
3. Skips the node when its phase mask has no bits in common with this call.
4. Skips the node when its insertion generation equals the current generation.
5. Skips the node when its **signed** 64-bit threshold exceeds sampled field
   `0xC8`. The producer supplies field `0xC8 + 1`; other producers use -1.
6. Resolves the stored 32-bit owner key through an engine lookup structure.
   Missing lookup state or a missing owner removes the record with release.

The record fields, relative to node offset `0x20`, are:

| Offset | Recovered use |
| --- | --- |
| `0x00` | double deadline |
| `0x08` | signed 64-bit eligibility threshold |
| `0x10` | float repeat interval |
| `0x14` | repeat flag |
| `0x18` | callback payload |
| `0x20` | dispatch callback |
| `0x28` | release callback |
| `0x30` | 32-bit owner lookup key |
| `0x34` | phase mask |
| `0x38` | insertion generation |

For a nonrepeating wait with a resolved owner, helper `0x43BC60` erases the
node **before** the callback receives `(owner, payload)`. A non-null release
callback then receives the payload only if the dispatch result is exactly 1.
Helper `0x43BB00` combines erasure and unconditional optional release for the
missing-owner/cancellation path. Both helpers can advance the saved cursor
when erasing the node it names, so the traversal is not a precomputed list.

Repeating records follow a separate branch that updates their deadline,
optionally updates the threshold when mask bit 8 is set, reinserts, erases the
old node and dispatches. WaitForSeconds sets the repeat flag to zero; no
standalone repeating-wait reconstruction is claimed here.

Records created during a callback carry the current generation and fail the
same-generation gate if encountered later in this drain. This is a native
local exclusion; it does not prove which records become eligible at the next
engine phase, nor supply the timing of writer-created Reveal continuations.

## Next native work and coverage

Trace the producer clock's relationship to Time.time and the callback dispatch
chain; recover tree mutation effects and phase provenance. The separate
registration-loop audit resolves StartCoroutineManaged2 without table proximity.
Keep any subsequently recovered engine-specific order separate from Unity's
public portability guarantees and from observed live event provenance.

UnityPlayer methods are outside the existing Assembly-CSharp denominator, so
this audit does not add managed classifications. Coverage remains 532 classified
methods and 276 evidence records. The engine report is separately fingerprinted;
no replay/live solver behavior was changed by these findings.

Validation passed 52 native relationships, 778 Python tests, 16 reverse-
engineering tests, Python compilation and diff checks. Rust/simulations were
not rerun for this audit-only checkpoint. UnityPlayer auditing does not add
managed coverage. Native PE inspection guidance in AGENTS.md also distinguishes
virtual zero-filled data and verified unwind containment.
