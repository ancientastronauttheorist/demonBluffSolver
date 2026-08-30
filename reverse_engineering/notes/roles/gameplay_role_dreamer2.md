# Gameplay role class: `Dreamer2` (unbound alternate)

Build: `f530404b0f3f_807de4a83df4`

Evidence level: `native-static`. This audit covers the complete managed
`Dreamer2` class plus the full two-method `GetDreamerClue` virtual-provider
set. Serialized asset evidence independently proves that the public Dreamer
card uses the distinct managed `Dreamer` class. `Dreamer2` is retained as
understood alternate code, not as the current live solver contract.

The private native exports are not committed. Exact metadata names,
signatures, and RVAs are preserved in
[`../../targets/gameplay_role_dreamer2.json`](../../targets/gameplay_role_dreamer2.json).

## Audited boundary

| Managed method | RVA | Surface |
| --- | ---: | --- |
| `Role.GetDreamerClue` | `0x3C4CE0` | Base virtual clue text |
| `Scout.GetDreamerClue` | `0x3C5F90` | Only override of that virtual slot |
| `Dreamer2.get_Description` | `0x3D8EE0` | Display description |
| `Dreamer2.GetInfo` | `0x3D8B70` | Real-role passive record |
| `Dreamer2.GetBluffInfo` | `0x3D8B10` | Bluff passive record |
| `Dreamer2.Act` | `0x3D74F0` | Normal picker setup |
| `Dreamer2.CharacterPicked` | `0x3D8310` | Normal result generation |
| `Dreamer2.BluffAct` | `0x3D7760` | Bluff picker setup |
| `Dreamer2.StopPick` | `0x3D8BD0` | Callback cleanup |
| `Dreamer2.CharacterPickedDrunk` | `0x3D79D0` | Bluff result generation |
| `Dreamer2.ConjourInfo` | `0x3D8A50` | Result formatting |
| `Dreamer2.ctor` | `0x3CFFF0` | Folded base-role construction |

## Passive text and construction

`GetInfo` and `GetBluffInfo` each return a fresh empty `ActedInfo`. The current
description says to pick one character, but both action paths request exactly
two characters. The description is therefore stale relative to this class's
native picker behavior.

`Role.GetDreamerClue` and `Scout.GetDreamerClue` both return the same short
fallback text, `I forgot my dream`. Metadata contains no other override of
virtual slot 6, and no `Dreamer2` method calls either provider. They are
preserved because they are the complete virtual-provider set, not because they
participate in the result path below.

The constructor has no role-local mutation. Its RVA is compiler-folded with
536 other distinct metadata names; the target retains the exact `Dreamer2`
identity while applying the already canonicalized base-role constructor
prototype to the shared body.

## Serialized public-card binding

The public `Dreamer` CharacterData is `sharedassets0.assets` path ID `21615`.
Its raw serialized object hash is
`6AD39DC4E26ADBCE0A722D4D92FBD0FB6084667A1BED4D4533CD3AFF0C1B6C35`, and its
SerializeReference role names managed type `Dreamer` in `Assembly-CSharp`.
It does not name `Dreamer2` or `DreamerOld`.

The same asset's authored description says: `Pick 2 characters: Learn a role
that's among them, and one that's not.` That binding and wording agree with the
observed concrete-role-pair clues. A read-only scan of the current
`sharedassets0.assets` and `level0` gameplay assets found no serialized
`Dreamer2` or `DreamerOld` role binding.

Consequently, this class's type-exclusion clue must not replace the public
Dreamer's role-pair parser, validator, or recommendation model. The bound
11-method `Dreamer` implementation and its five generated helpers are audited
separately in [`gameplay_role_dreamer.md`](gameplay_role_dreamer.md).

## Two-character picker and cleanup

Both `Act` and `BluffAct` are clean no-ops outside the Day trigger. On Day they
call `CharacterPicker.StartPickCharacters(2, charRef)`, subscribe one completion
handler, and subscribe `StopPick`:

- the normal path uses `CharacterPicked`;
- the bluff path uses `CharacterPickedDrunk`.

There is no role picker in either path. `StopPick` removes both possible
completion handlers and its own stop handler, so cancellation is safe no matter
which action path installed the callbacks.

## Normal result generation

After removing its callbacks, `CharacterPicked` reads selected characters zero
and one. It creates the unique type pool `Demon`, `Minion`, `Outcast`,
`Villager`, then removes each target's registered character type. The queried
type is the native `Character.GetCharacterType` surface:

```text
registered type = (registerAs is live ? registerAs : dataRef).type
```

One remaining type is selected uniformly and rendered as its enum name. Since
the initial pool contains one copy of each type, targets of two distinct types
leave two possible outputs. Two targets of the same type remove that type only
once and leave three possible outputs.

The English format is:

```text
#<first id>, <second id>:
None of them is
<Villager|Outcast|Minion|Demon>
```

Only the first ID has a literal `#` in this build's format string. The emitted
`ActedInfo` stores both selected character references even though the second
displayed ID lacks that prefix. The statement is true by construction.

## Bluff result generation

`CharacterPickedDrunk` preserves the same two targets, output format,
`ActedInfo` storage, callback invocation, and logging surface, but constructs a
false result pool:

1. For each selected character whose real `dataRef.usuallyDisguised` flag is
   set, append that character's registered type.
2. If that pool is empty, append the registered type of both targets.
3. Uniformly select one list entry. Duplicate types remain duplicate entries
   and therefore retain their random weight.

The selected type is consequently present at one or both targets, making
`None of them is <type>` false. The `usuallyDisguised` filter affects which
false type is sampled and its probability; it does not change the truth
predicate.

A read-only audit of all 46 shipped core CharacterData assets found
`usuallyDisguised == false` on every record. In the current roster the first
step is therefore always empty, so the unbound alternate's lying path falls
back to sampling the two selected registered types with equal entry weight.
This current-build fact does not make the unbound clue format part of the
public Dreamer contract.

## Typed-analysis corroboration

The target validates 12 of 12 metadata entries and baseline-exports all 12
without failure. The deterministic typed union then applies and read-only
validates all 12 signatures and 33 Windows x64 parameter-storage locations;
typed export again completes 12 of 12.

The body-free quality report at
[`../../reports/f530404b0f3f_807de4a83df4_typed_quality_gameplay_role_dreamer2.json`](../../reports/f530404b0f3f_807de4a83df4_typed_quality_gameplay_role_dreamer2.json)
records no decompiler-error markers or policy regressions. Typed analysis
removes all 37 placeholder-parameter tokens, reduces unresolved-type tokens
from 105 to 80, and reduces raw field-offset accesses from 167 to 156.

## Reachability and remaining uncertainty

This class's local control flow is understood, but no current gameplay asset
binds it. The separate public `Dreamer` audit reconstructs the shipped
concrete-role selection and random weighting; solver parity must use that bound
implementation rather than this alternate.

Picker-level rejection of self-selection or duplicate selection belongs to the
shared `CharacterPicker` boundary and is not inferred here. English output is
statically pinned; other localization variants and live presentation remain
behaviorally unverified.
