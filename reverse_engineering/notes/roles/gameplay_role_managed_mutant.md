# Managed Mutant selector and role boundary

The managed `Mutant : Role` declaration at TypeDefIndex 5903 is distinct from
the shipped public Mutant card. The public asset at `sharedassets0.assets`
path ID 21592 binds `Skinwalker`; its exact object fingerprint and serialized
role-name offset are rechecked in this audit. No shipped binding is assumed
for the managed class described here.

The [target set](../../targets/gameplay_role_mutant.json) covers all six managed
declarations and the inherited BluffAct dispatcher and ActedInfo constructor.

| Method | RVA | Direct behavior |
| --- | --- | --- |
| get_Description | `0x3E4DB0` | Empty string |
| GetInfo | `0x3E4D50` | New clue, empty text, null references |
| GetBluffInfo | `0x3E4CF0` | Independently constructed empty clue |
| Act | `0x33ED50` | Immediate return for every trigger |
| GetBluffIfAble | `0x3E4BB0` | Good/bluffable selection after a Mad attempt |
| Constructor | `0x3CFFF0` | Empty base-constructor forwarding chain |

Inherited Role.BluffAct tail-dispatches the actual Act slot, so that path is
also inert for this managed role. The base GetRegisterAsRole is unchanged.

## Selector order and failure state

With live singletons and receiver, the selector obtains the combined script
list, filters exact serialized alignment Good (`10`), then filters `bluffable`.
Both filters preserve order and repeated asset references. Public role names,
declared role faction and distinct-name counts cannot replace these fields.

The actor's status owner receives `AddStatus(Mad=20, source=actor, target=null)`
before the final list check and random indexed selection. The generic status
contract means an accepted attempt adds Mad only if absent and clears the shared
target even when Mad already exists. Resistance preserves membership and target.

A null source element fails during filtering before the status call. An empty
eligible list reaches the status call, one zero-width random range call, and
indexed access failure. Accepted Mad is not rolled back. No selected asset is
removed from a pool or added to a script list by this method.

## Offline implementation

[`mutant_selector.rs`](../../../crates/solver-core/src/bluff/mutant_selector.rs)
accepts version `managed_mutant_selector_native_v1`, an explicit asset-identity
table, a combined script occurrence list and ordered status/resistance state.
It retains each eligible occurrence as a separate equal-probability result,
including different occurrences of the same asset. Asset identity is independent
of a public name and can represent a Good/bluffable asset of any declared type.

Native null-element and empty-support failures are explicit result variants.
Malformed provenance, missing asset definitions, duplicate definition keys and
capacity violations remain atomic API errors. No result mutates caller input.
The context bounds input and retained branch state; it does not infer setup,
singletons, allocation, caller dispatch, Unity RNG state or interleaving.
There is no live/scenario caller and no change to existing ledger versions.

## Evidence and verification

[`audit_mutant.py`](../../scripts/audit_mutant.py) pins the native image,
metadata and Dumper outputs, verifies the six-method declaration boundary and
22 native instruction/literal relationships, and rechecks the public asset's
Skinwalker binding. The report contains authored observations, not native bytes
or decompiler bodies.

Its 16 isolated executions run the actual warmed selector body. Explicit service
gateways provide script lists, filters, status handling, random indices and
indexed access. This verifies native caller order and exact arguments; it does
not claim native execution of those gateways. Cases cover repeated identities,
mixed alignment/bluffability, empty lists and early/late null elements. The
Rust comparison checks each native result, status-attempt boundary and RNG call
count. Separate regressions check probability, accepted/resisted status effects,
target clearing, retained failure state, strict input decoding and capacity.

```powershell
python reverse_engineering/scripts/audit_mutant.py `
  'B:\SteamLibrary\steamapps\common\Demon Bluff Playtest' `
  --dumper-root 'B:\CodexTools\DemonBluffReverseEngineering\artifacts\f530404b0f3f_807de4a83df4\il2cppdumper-v6.7.46' `
  --output reverse_engineering/reports/f530404b0f3f_807de4a83df4_mutant_audit.json
cargo test --release -p solver-core --lib mutant_selector
```

Baseline and typed exports each completed all eight targets. The rebuilt typed
union contains 43 sets, 899 memberships, 553 exact FunctionDefinitions and 451
native RVAs. Read-only validation checked all memberships and 2,638 parameter
storage locations with zero mutations. Application imported no new reachable
datatypes. The [quality report](../../reports/f530404b0f3f_807de4a83df4_typed_quality_gameplay_role_mutant.json)
passes its gates: placeholder parameters fall from 24 to zero, raw field offsets
from 13 to six and unresolved type tokens from 17 to four. The inherited virtual
tail-jump diagnostic remains, resolved by the explicit native slot/ABI checks.

Validation passed 647 Rust library tests, the release build, 778 Python tests,
32 reverse-engineering tests, native caller comparison, typed-union validation,
coverage integrity and formatting checks. The new module has no live/scenario
caller. The ongoing full simulation pass is recorded separately when complete.
