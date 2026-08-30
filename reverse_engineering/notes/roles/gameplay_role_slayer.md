# Gameplay role: Slayer

Build: `f530404b0f3f_807de4a83df4`

Evidence status: **native-static** for all ten methods in the checked boundary.
The note records authored behavior summaries only; native bodies and decompiler
output remain outside the repository.

The checked target set is
[`reverse_engineering/targets/gameplay_role_slayer.json`](../../targets/gameplay_role_slayer.json).
Its read-only baseline Ghidra export completed at 10/10 functions with no
failures.

## Target boundary

| Managed identity | RVA | Boundary role |
| --- | ---: | --- |
| `Slayer.get_Description` | `0x3ED480` | Display description |
| `Slayer.GetInfo` | `0x3ED110` | Real-role passive information |
| `Slayer.GetBluffInfo` | `0x3ED0B0` | Bluff passive information |
| `Slayer.Act` | `0x3EBF50` | Normal Day picker setup |
| `Slayer.CharacterPicked` | `0x3EC9C0` | Normal target resolution |
| `Slayer.BluffAct` | `0x3EC1D0` | Bluff/disabled picker setup |
| `Slayer.StopPick` | `0x3ED170` | Picker callback cleanup |
| `Slayer.CharacterPickedDrunk` | `0x3EC570` | Disabled target resolution |
| `Slayer.ConjourInfo` | `0x3ECFD0` | Synthetic result text |
| `Slayer.ctor` | `0x3CFFF0` | Base-only construction |

## Passive text and construction

`GetInfo` and `GetBluffInfo` each return a fresh empty `ActedInfo`; Slayer has no
passive clue on either surface. `ConjourInfo` formats a failed-kill result for
Good, a successful Evil-kill result for Evil, and an empty string for any other
alignment value.

The current description literal says that Slayer dies after choosing Evil.
That text is stale: no Slayer method in this boundary kills the acting card.
The constructor only follows the ordinary base-role initialization path; its
native body is shared with many other otherwise empty role constructors.

## Day picker and cleanup

`Act` does work only for the Day trigger. It stores the acting card, starts a
one-character pick, then subscribes the normal completion callback and the
stop callback. Other trigger phases return without changing picker state.

`BluffAct` has the same Day gate and one-character picker. It inspects the
acting card's raw `WorkingAbility` status: present selects the normal completion
callback, while absent selects `CharacterPickedDrunk`. The stop callback is
installed in either branch. This status test is separate from the higher-level
truth/lying dispatch that chose `BluffAct`.

`StopPick` removes both possible completion callbacks and its own stop callback,
so cancellation cleans up either setup path. A successful normal or disabled
callback removes only its own completion handler plus the stop handler. Missing
picker singletons, actor status storage, or a usable first picked card follow
the native failure paths rather than producing an empty result.

## Callback resolution

The normal callback resolves only the first picked card and asks that card for
its **registered alignment**. Any value other than exact Evil produces a
failed-kill result and no death. Exact Evil selects the kill branch. A raw
`Lying` status on the acting Slayer changes the displayed result to a failed
kill, but does not cancel that selected kill branch.

After constructing the result, the callback checks the target's live state. A
target already marked Dead returns before the acted callback, debug log, or
kill. Otherwise the emitted `ActedInfo` contains the one selected target, and
the Evil branch calls the existing `Character.KillAndReveal` lifecycle entry.

`CharacterPickedDrunk` always emits the failed-kill result for the first target
and never calls a kill method. Unlike the normal callback, this disabled path
has no target-state gate before reporting its result.

## Joined Wretch interaction

The public Wretch role is the managed `Recluse` class. Its audited
`GetRegisterAsRole` selects Minion data, which internal `Character.Reveal`
stores as `registerAs`. `Character.GetRegisterAlignment`, used by Slayer's
normal callback, reads that registered record before the card's runtime
alignment. The shipped Minion records are Evil-aligned, so Slayer selects its
kill branch against a Good Wretch.

The live regression case
[`tests/cases_v2/asc26_v8.json`](../../../tests/cases_v2/asc26_v8.json)
records that exact interaction and the separate 5-HP consequence. HP mutation
belongs to the existing
[execution-resolution boundary](../systems/gameplay_execution_resolution.md#wrong-execution-damage-and-value-reduction),
not to the Slayer callback itself.

## Linked boundaries and metadata cautions

This target intentionally does not duplicate `Character.GetRegisterAlignment`,
internal `Character.Reveal`, or `Character.KillAndReveal`; their established
behavior joins the role-specific methods above. The Slayer constructor's RVA is
a shared native body, so coverage must preserve the exact managed identity and
signature rather than relying on its primary native symbol label.
