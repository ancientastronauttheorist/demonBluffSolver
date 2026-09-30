# Round construction to selector ledger

`bluff::pool_ledger_bridge` supplies the offline versioned
`pool_ledger_bridge_native_v1` composition. It executes the existing unique-pool
weighted kernel, then the actual duplicate-candidate weighted kernel, then the
successful-selector ledger. This bridges previously separate numeric asset
identities and canonical role names under explicit shared-state provenance.
It does not execute ManageCharacters, its Init/Start writers or engine scheduling.

## Identity and chronology

The caller supplies a bijection from each live asset ID to one exact canonical
role. The bridge verifies every asset's real faction against its role, rejects
distinct same-name assets and aliases/case variants, and preserves repeated
references as separate list occurrences. Null final pool entries and unsupported
bluff roles reject the complete operation. Starting alignment remains separate
from faction: the native builder can admit starting-Evil Villagers and Outcasts.
The mapping's current-build liveness and actual asset identity are required
provenance, not inferred from these synthetic fixture IDs.

The unique kernel's supplied script must equal the duplicate kernel's four
current rosters concatenated in 10/20/30/100 order. Both kernels share identical
asset predicates and the same initial Gameplay singleton/class state. Their
pool identities and intermediate collections must be distinct. Successful unique
construction carries completed Gameplay initialization into duplicate construction;
the class initializer is not rerun from the initial cold state.

Unique getters remain supplied in this version. Their stronger actual source
composition is documented in [unique_source_composition.md](unique_source_composition.md)
and is a subsequent integration boundary. Controlled singleton replacement and
the separate predicate-only API are rejected by this bridge. The caller must
exclude intervening pool/script writers and establish the selector acquisition
order independently. Board position does not generate that order.

## Probability and failure

Each outer outcome retains its unique/duplicate construction trace indices.
Construction traces are stored once. A successful outcome additionally contains
the selector ledger's conditional path probability; the outer probability
multiplies both construction histories and the selector path. The unique kernel
is run once, and duplicate support is prepared once after completed unique
construction. No earlier script draw or selected pool is reconstructed again.

Unique failures retain their full incoming mass and prevent both later stages.
Duplicate failures retain the product of the two construction probabilities and
prevent selectors. These failures retain partial lists/versions and attempted RNG
requests through their referenced traces. The successful-selector ledger's
unsupported or empty-support error rejects the entire bridge invocation; successful
branches are never conditioned or renormalized around a failed branch.

Uniform occurrence draws and independent conditional service responses are an
explicit contract. This is weighted support rather than a Unity PRNG seed/state
transcript. The Minion branch roll and marginalized must-include probe remain
represented by the ledger's draw count; marginalization does not recover the
discarded probe's specific RNG result.

## Bounds and validation

The bridge accepts the existing 32-input/32-asset kernel bounds and at most 32
must-include occurrences. Joint outcomes are capped at 1,024, with 1,048,576
retained logical trace units shared by construction traces and selectors. JSON
event snapshots count their actual nested value units. Each construction kernel
applies its own working budget before returning; the bridge charges its retained
traces afterward. This does not promise one aggregate peak allocation limit
across the subordinate construction calls.

The ledger's bounded internal entrypoint receives the bridge's remaining path
and retention budgets. It checks pending, current and proposed cloned paths
before allocating a new branch. Rejection leaves inputs immutable and publishes
no partial distribution.

Six focused bridge tests compare duplicate construction to the eighteen native
support traces, then independently check ninety joint Minion/Demon/Drunk outcomes,
their exact conditional/unconditional fractions and acquisition ordinals. They
also cover first-equal must-include removal with repeated occurrence indices,
resistant versus accepted Drunk corruption attempts, construction failure mass,
invalid identity/provenance atomicity and bounded expansion. A separate ledger
regression checks its shared working budget including the current path.

Evidence: [unique-pool replay](round_bluffs_replay.md),
[duplicate-candidate replay](round_candidate_replay.md), the existing selector
ledger, and the authored bridge tests. These are composed offline contracts,
not a new native execution of the full joined setup/selector transaction.
