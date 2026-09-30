# Actual setup pool prefix to selector ledger

`bluff::manage_pool_ledger_bridge` composes the actual ManageCharacters pool
prefix with the successful-selector ledger. The version is
`manage_pool_ledger_native_v1`. It executes the weighted source and both-builder
history once, retaining its shared cache, callbacks, pool identities and RNG
chronology. It does not execute the intervening Init/Start passes or generate
selector dispatch order.

The caller supplies a bijection from numeric asset identities to exact canonical
roles, verified against each asset's real faction. Current-build identity and
liveness, completion of intervening setup without pool/script writers, actual
selector dispatch order and independent uniform selector draws are explicit
requirements. Synthetic fixture IDs do not establish shipped asset identities.
Null, unknown, ambiguous or cross-faction final identities reject the operation.

Each successful native prefix uses its final current roster providers for the
ledger's faction script lists. These may differ from either builder's earlier
captured script snapshot. A callback after duplicate script capture can replace
a current provider without changing either completed pool. The tests preserve
that distinction rather than rebuilding a captured list. A missing final
provider or vanished Gameplay singleton prevents the selector bridge.

Every native prefix failure retains its unconditional probability and prevents
selectors. Successful prefixes stop before first Init, or before empty-board
publication with no selectors. The conditional selector probability is multiplied
by the already combined construction probability exactly once. Unsupported
positive-mass selector branches reject the full invocation; no surviving branch
is renormalized. Construction traces are retained once and outcomes reference
them by index.

The shared output bound is 1,024 outcomes and 4,194,304 logical retained trace
units. Nested JSON values count toward that bound. Construction applies its own
working budget before returning; this is not one aggregate peak-memory limit.
The selector kernel receives the remaining path and retention budgets and checks
current, pending and proposed branch allocations before cloning.

Five tests cover 96 joint Minion/Demon outcomes from eight mixed construction
histories, exact conditional and unconditional fractions, four fallback paths,
sixteen inline paths and fifty-six nullable-inline paths. They also cover final
provider replacement, native-valid aliases rejected by canonical faction rules,
singleton disappearance and retained failure mass. This is an offline contract
composition, not a new native complete setup/selector execution or live solver
integration.

Evidence: [actual shared pool prefix](manage_pool_composition.md),
[supplied-source predecessor](pool_ledger_bridge.md), and the authored
`manage_pool_ledger_bridge_tests.rs` tests. Supported actor/continuation production
is documented separately in [setup initialization](setup_initialization_batch.md)
and [setup actions](character_action_setup.md).
