# Spy managed native contract

This closes the six-method managed `Spy : Minion` declaration boundary at
TypeDefIndex `5911` in build `f530404b0f3f_807de4a83df4`. Spy has one added field,
`CharacterData chData` at `+0x48`. The target set also includes three semantic
callees: inherited `Minion.GetBluffInfo`, inherited `Role.BluffAct`, and the
`ActedInfo` constructor. There is no claim that a Spy asset belongs to the
current normal deck.

## Native boundary

| Managed method | RVA | Behavior |
| --- | --- | --- |
| `Spy.get_Description` | `0x3ed7d0` | Returns a cached description literal |
| `Spy.GetInfo` | `0x3ed640` | Creates empty-text info with a null reference list |
| `Spy.Act` | `0x33ed50` | Shared immediate return; no state or callback writes |
| `Spy.GetBluffIfAble` | `0x3ed4b0` | Existing-bluff shortcut, then shared Villager cache |
| `Spy.GetRegisterAsRole` | `0x3ed6a0` | Shared Villager cache for apparent identity |
| `Spy..ctor` | `0x3cfff0` | Empty constructor chain; no cache initialization write |

The description says that Spy can register as a Good Townsfolk and that the
Demon will kill the best targets. This is description text; the Spy methods
contain no corresponding night-target-selection implementation. Effects in
other game systems must be established through those systems' own code.

`Spy.GetInfo` and inherited `Minion.GetBluffInfo` at `0x3e4ac0` each allocate an
`ActedInfo` with the same empty string and null character-reference list. They
perform no RNG selection, status mutation or role-local speech construction.

The folded `Act` body is a native immediate return with a zero stack-pop operand.
Its decompiler output is empty because the declared return type is void. Spy
does not override `BluffAct`; neither does Minion. Inherited `Role.BluffAct` at
`0x3c4ca0` forwards to the concrete object's Act slot, so both truthful and lying
dispatch reach the same inert Spy body at every trigger, including Start.

The constructor forwards through `0x357920` to the same immediate-return body.
It does not clear `chData`, install an action subscriber, or add a role-specific
initialization hook. Allocation or clone semantics must still be supplied by
the caller; the constructor alone does not establish a fresh empty cache.

## Cache behavior

The two cache methods retain the contract already established in the
[bluff-acquisition audit](../systems/gameplay_bluff_acquisition.md#role-selectors-and-script-mutations):

- Register-as returns a live `chData` or filters the combined script list to
  exact Villagers, selects uniformly over retained occurrences and caches it.
- Bluff selection returns `charRef.registerAs` immediately when a live raw
  bluff already exists. Otherwise it uses that same live-cache/draw path.
- Selection does not mutate the script or unique/duplicate bluff pools. Null
  dependencies and empty candidate support are not silently repaired.
- Null and destroyed Unity references differ from a live cache. Distinct
  native role instances need distinct logical cache provenance; two bodies
  sharing one data-role instance share its cache.

Reveal's register-as call precedes the optional bluff acquisition, so its normal
uncached path draws only once. Start neither resets nor populates this cache.
An action-role clone's copied cache is inert under the audited Spy dispatch.

## Serialized-name observations

The audit pins and scans `sharedassets0.assets`, `level0`, `resources.assets`
and `globalgamemanagers.assets`. None contains an ASCII `Spy` occurrence. The
report records their exact hashes and sizes. No current shipped Spy asset
binding has been established; these negative name observations alone do not
prove global unreachability or rule out unrelated construction paths. This
boundary is classified from its managed declarations and native methods.

## Offline replay

The existing `reveal` v3 and `character_start` contracts now accept reached Spy
Start calls in either real or copied slots. Character still sets its one-shot
Start latch before dispatch and still enforces its subscription and provenance
guards. Spy dispatch changes neither statuses nor cache state and creates no
continuations. Earlier context versions retain their existing Spy/version rules.

The ordered writer and queue-driven Reveal adapter inherit this support through
the same kernels. Synthetic tests cover a copied Spy after a weighted Twin swap,
real Spy with empty/live shared caches, both standalone Start slots, repeated
Reveal cache reuse, and a scheduled uncached Spy acquisition followed by inert
Start. Unreviewed asset construction is not added to live solver state.

## Reproduction

[`gameplay_role_spy.json`](../../targets/gameplay_role_spy.json) contains the
nine-function target boundary. The native audit checks all six declarations,
the cache field, 17 instruction/literal relationships and the four pinned
serialized-name observations without emitting native code.

```powershell
python reverse_engineering/scripts/audit_spy.py `
  'B:\SteamLibrary\steamapps\common\Demon Bluff Playtest' `
  --dumper-root 'B:\CodexTools\DemonBluffReverseEngineering\artifacts\f530404b0f3f_807de4a83df4\il2cppdumper-v6.7.46' `
  --output reverse_engineering/reports/f530404b0f3f_807de4a83df4_spy_audit.json
cargo test --release -p solver-core --lib spy
```

Baseline and typed exports both completed all nine targets. The rebuilt union
contains 42 sets, 891 memberships, 548 exact FunctionDefinitions and 448 native
RVAs. Its read-only validation checked all memberships and 2,613 parameter
storage locations with zero program mutations. The Spy application imported
no additional reachable datatypes.

The [typed-quality report](../../reports/f530404b0f3f_807de4a83df4_typed_quality_gameplay_role_spy.json)
passes its non-regression gates: placeholder parameters fall from 30 to zero,
raw field offsets from 23 to 11 and unresolved type tokens from 29 to four.
The existing virtual-tail-jump diagnostic remains; native register/slot checks
establish the inherited dispatch rather than treating the decompiler's omitted
call arguments as a zero-argument managed call.

Validation passed 639 Rust library tests, the release build, 778 Python tests,
32 reverse-engineering tests, the native audit, full typed-union validation,
coverage integrity, formatting and diff checks. The earlier queue checkpoint's
34-test/426-fixture simulation pass remains the latest full simulation run;
this Spy extension is confined to offline replay with no live/scenario caller.
