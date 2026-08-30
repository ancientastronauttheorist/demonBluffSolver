# Gameplay role: Witch (managed `Cipher`)

Build: `f530404b0f3f_807de4a83df4`

Evidence status: **native-static** for all five methods declared by the shipped
role and for the Start dispatch, global block value, hidden-card count, ordinary
click gate, reset, and both death paths that determine its observable boundary.
Serialized asset evidence establishes the public binding, exact player text,
and ordered Start slot. Native bodies and decompiler output remain outside the
repository.

The checked target set is
[`reverse_engineering/targets/gameplay_role_witch.json`](../../targets/gameplay_role_witch.json).
Its read-only baseline and typed Ghidra exports each completed at 19/19
functions with no failures. The body-free
[`quality report`](../../reports/f530404b0f3f_807de4a83df4_typed_quality_gameplay_role_witch.json)
passes its regression check: unresolved-type tokens fall from 119 to 30 and
raw field-offset accesses from 241 to 89. Decompiler error-marker counts remain
seven on both sides; the typed export adds one nonfatal warning marker.

## Public asset binding and authored text

The shipped `sharedassets0.assets` `CharacterData` at path ID `21602` is named
`Witch`, has `characterId` `Witch_25286521`, and binds its SerializeReference
role to managed `Cipher` in `Assembly-CSharp`. Its raw object SHA-256 is
`6B4353F67AF9999D0D4BC7E88791B37B5E2436380B6013DF26B7A6F4F04B838D`.
`Cipher` is TypeDefIndex `5918`, declares no fields, and declares exactly the
five native methods in this boundary.

The asset is an Evil Minion: `characterType == 30`,
`startingAlignment == 20`, `abilityUsage == 0` (`Once`), `bluffable == false`,
`usuallyDisguised == false`, and `picking == false`. Its roguelike point value
is 10, multiplier is 1.0, and income is 1. `additionalStatuses`, tags,
`canAppearIf`, and the localized description override fields are empty.

The exact authored description is:

```text
You can not Reveal the last card.

I Lie and Disguise.
```

The exact authored hint is:

```text
You can reveal the last card after I die.
```

The hint is serialized on `CharacterData`; no `Cipher` method constructs or
mutates it. The managed role's separate `get_Description` literal is:

```text
You can reveal 1 less card
```

`Cipher.GetInfo` returns a fresh `ActedInfo` with an empty description and a
null character list. Witch's effect therefore has no target reference, speech
result, saved blocked identity, or active picker result.

The `level0` `Characters` component at path ID `137026` references Witch path
ID `21602` at zero-based index 4 of `startGameActOrder`: after Drunk and before
Marionette/Twin Minion.

## Target boundary

| Managed identity | RVA | Boundary role |
| --- | ---: | --- |
| `Characters.ManageCharacters` | `0x36CE30` | Serialized ordered Start and ordinary duplicate handling |
| `Character.Act` | `0x3645C0` | Start latch and truth-aware dispatch |
| `Character.RoleAct` | `0x368790` | Concrete `Act`/`BluffAct` virtual call |
| `CharacterHelper.CheckLying` | `0x397750` | Evil-Witch dispatch condition |
| `Role.BluffAct` | `0x3C4CA0` | Inherited virtual forwarder back to concrete `Act` |
| `Character.GetHiddenCardsAmount` | `0x364EC0` | Exact global Hidden-state count |
| `Character.OnClick` | `0x366270` | Reveal quota and bypassing picker/execution branches |
| `Character.Kill` | `0x366130` | Ordinary/Slayer death transition and cleanup callback |
| `Character.DelayedDemonKill.MoveNext` | `0x3757F0` | Lilis/night death transition and cleanup callback |
| `Gameplay.ResetPlayerInfo` | `0x37FC90` | Between-village block reset |
| `SimpleValue.Add` | `0x38A780` | Increment global block quota |
| `SimpleValue.GetValue` | `0x38A7A0` | Read quota for the click gate |
| `SimpleValue.Reduce` | `0x38A7B0` | Death decrement with zero clamp |
| `SimpleValue.Reset` | `0x38A7E0` | Reset quota to zero |
| `Cipher.get_Description` | `0x3D65A0` | Managed description surface |
| `Cipher.GetInfo` | `0x3D6540` | Empty passive information surface |
| `Cipher.Act` | `0x3D64C0` | Start-only block increment |
| `Cipher.ActOnDied` | `0x3D6450` | Unconditional block decrement |
| `Cipher.ctor` | `0x3CFFF0` | Folded base-role construction |

`Role.BluffAct`, `SimpleValue.GetValue`, and `Cipher.ctor` share native bodies
with other managed identities. The manifest preserves the exact Witch-facing
definitions and applies the established canonical prototypes for those bodies.

## Ordered Start and the Evil-dispatch forwarder

`Characters.ManageCharacters` initializes and publishes every board card,
runs `Character.Act(Init)` on all of them, and then walks the exact serialized
Start order. For each entry it compares exact `CharacterData` objects and calls
`Character.Act(Start)` (`5`) on the matching card.

Witch initially satisfies `CharacterHelper.CheckLying` because it is Evil and
does not have `HealthyBluff`. With no copied bluff role yet,
`Character.Act` sends the real role through its `BluffAct` slot. That does not
skip the Witch effect: `Cipher` does not override `BluffAct`, and the inherited
`Role.BluffAct` body is a virtual tail-forwarder to the same object's concrete
`Act` slot. It therefore reaches `Cipher.Act(Start)`. If a truth-state or live
bluff instead makes `Character.Act` choose the normal slot, it reaches the same
`Cipher.Act` body directly.

`Cipher.Act` is a no-op for every trigger except Start. On Start it resolves
`PlayerController.PlayerInfo.blocks.value` and calls virtual `Add(1)`. It does
not select a character, inspect card state, store an identity, or check the old
quota.

`Character.Act` sets `characterStartActed` before role dispatch, so a second
Start call on the same initialized `Character` is ignored. The ordered roster
scan also stops after the first card matching an ordinary `CharacterData`.
Only Alchemist, Poisoner, and Puzzlemaster are duplicate exceptions; Cipher is
not. Consequently two ordinary cards backed by the same shipped Witch asset
still produce one Start increment, from the first matching card in scan order.

The value implementation itself is additive and has no maximum or uniqueness
guard. Separate successful calls to `Cipher.Act(Start)` would stack, but the
ordinary shipped duplicate path above does not create them. Conversely,
`Cipher.ActOnDied` is not paired to the particular card that supplied the
increment. In a hypothetical duplicate-Witch board, either real Witch death
can decrement the single ordinary quota.

## Global block storage and scalar semantics

Metadata fixes the storage chain as `PlayerInfo.blocks` at `+0x20`, then the
inherited `Resource.value` reference at `+0x10`. The target audits the shipped
`SimpleValue` behavior behind the virtual resource calls:

- `Add(n)` performs `current += n` with no cap, then invokes `onValueChanged`;
- `GetValue()` returns `current`;
- `Reduce(n)` performs `current -= n`, clamps a negative result to zero, and
  invokes `onValueChanged`; and
- `Reset()` writes zero and invokes `onValueChanged`.

There is no collection of blocked cards. The only persistent Witch rule state
is this player-global integer quota. Multiple increments therefore stack as a
number, not as target records; multiple cleanup calls cannot underflow it.

## Exact hidden-card count and reveal predicate

`Character.GetHiddenCardsAmount` iterates `Gameplay.CurrentCharacters` and
increments for each exact `Character.state == Hidden` (`5`). It applies no
role, alignment, identity, reveal-history, `killedHidden`, `killedByDemon`, or
ability-use filter.

A killed-hidden card does **not** remain in the count. Both kill paths change
its state to `Dead` (`20`); the `killedHidden` flag records history but is not
consulted by `GetHiddenCardsAmount`. This is the exact Lilis/Witch interaction:
a hidden Lilis victim leaves the hidden total as soon as its state becomes
Dead, whether or not the victim was Witch.

Outside Night, picker mode, and execution mode, `Character.OnClick` computes
the hidden count `H`, reads the block quota `B`, and changes this card from
`Hidden` (`5`) to `Alive` (`10`) only when:

```text
B < H && this.state == Hidden
```

Equivalently, ordinary clicks can reduce the hidden population only until
`H == B`. With one living Witch quota, the last hidden card cannot be revealed;
with quota two, the final two hidden cards cannot be revealed. The card identity
is determined entirely by click/reveal order. A hidden Witch counts like every
other card, so Witch can be the last card blocking itself.

When this guard fails on a Hidden card, the later active-ability branch returns
because the card is still Hidden. Thus a blocked hidden card cannot fire its own
unused Day ability. After the quota is removed, its first ordinary click makes
it Alive and returns; normal active-card handling is available on a later click.

## Picker, execution, and programmatic bypasses

The global quota gates only the ordinary Hidden-card click transition:

- `PickCharacters` routing runs before the quota and state checks, so a picker
  can select a blocked Hidden card if its board object receives the click;
- execution mode runs a real reveal and `Kill` without consulting the quota,
  so the final hidden card can be executed directly;
- death and other programmatic state changes do not call the quota predicate;
  and
- a quota decrement does not auto-flip anything. A surviving Hidden card must
  receive a later ordinary click.

These bypasses reinforce that Witch does not attach a status to one target.
An external `blocked_positions` entry is only a snapshot of which card happened
to remain Hidden under the reveal order.

## Death cleanup and blocked-identity absence

`Cipher.ActOnDied` ignores its `charRef`, resolves the same global block value,
and calls `Reduce(1)` unconditionally. The role stores no flag saying whether
this instance acted and no pointer to a blocked card.

`Character.Kill` is a no-op when the card is already Dead. Otherwise it records
`killedHidden` when appropriate, changes the state to Dead, refreshes the card,
then calls `dataRef.role.ActOnDied` before publishing `OnCharacterKilled`.
Ordinary execution and Slayer therefore clean up the exact real Witch role even
when the card was Hidden or displayed another identity. A mere Witch bluff does
not decrement the quota because the callback comes from `dataRef.role`.

The delayed demon-kill path used by Lilis first rejects an already-Dead or
unkillable target. On success it sets `killedByDemon`, changes state to Dead,
applies its death statuses and Died trigger, then calls the same exact
`dataRef.role.ActOnDied` before the kill event/UI completion. A night-killed
real Witch therefore removes itself from the hidden count and decrements the
quota during the same coroutine. A night-killed non-Witch leaves the quota
unchanged but still leaves the hidden count because its state is Dead.

After either real Witch death, the remaining formerly blocked card is merely
eligible for a later click; the game does not remember or reveal a particular
identity automatically.

## Between-village reset

`Gameplay.ResetPlayerInfo`, called from the HandOut path, resets player mana
and then blocks. It also resets health except in Roguelike mode (`10`). The
block reset calls the same virtual `Reset`, so the quota becomes zero and its
change event fires before the next board is set up. Witch block state does not
persist between villages.

## Active corpus surface

A deterministic read of the 426 checked-in `tests/cases_v2` fixtures found:

- 67 Witch-deck cases and 66 recorded true Witch identities; `asc74_v7` names
  Witch #9 in notes but lacks that truth entry;
- 38 nonempty `blocked_positions` cases, all and only in Witch decks;
- every nonempty blocked list has exactly one position, while 29 Witch-deck
  cases end with no stored block;
- four self-block observations: `asc52_v7`, `asc70_v7`, `asc75_v3`, and
  `asc79_v5`;
- blocked truth among those 38 snapshots: 30 Good/unknown, four Witch, two
  Baa, one Lilis, and one Puppeteer; and
- among the 66 recorded true Witches, 60 were ordinary-executed, three were
  Slayer-killed, and three were Lilis/night-killed.

The fixture distribution confirms that self-block and Evil-card block outcomes
are legal and that `blocked_positions` tracks the last-hidden consequence, not
a Witch target predicate. No current fixture demonstrates a stacked quota or
an ordinary duplicate Witch, so those edge semantics remain native-static.

## Reconstruction consequences and bounded unknowns

- Bind public Witch to managed `Cipher`, not to `Illuzionist` (public Shaman).
- Model one global integer reveal quota, not a chosen card or per-card status.
- Count only exact current Hidden states; a killed-hidden card is Dead and no
  longer contributes, regardless of its history flags.
- Permit Witch self-block and any other identity as the final hidden card.
- Allow picker selection and direct execution of a blocked Hidden card, but do
  not fire that card's unused active ability through the failed reveal click.
- Clear one quota on every exact real Witch death, including Slayer and Lilis;
  do not clear it merely because an apparent Witch bluff dies.
- Do not auto-reveal a card when the quota falls. Require the subsequent click
  transition.
- Reset the quota between villages and clamp duplicate cleanup at zero.

No native semantic uncertainty remains in the shipped one-Witch Start,
last-card predicate, Hidden membership, lack of blocked identity, or ordinary
and night death cleanup. Two edges remain bounded rather than observed in the
current corpus: the additive value would support multiple independently
successful Cipher Start calls, while the shipped ordinary duplicate scan emits
only one; and Unity presentation can still determine whether a particular
scene skin exposes a clickable board object even though `Character.OnClick`
itself applies the bypass and reveal rules above.
