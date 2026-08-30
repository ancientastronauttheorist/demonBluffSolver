# Gameplay role class: `Dreamer` (shipped public card)

Build: `f530404b0f3f_807de4a83df4`

Evidence level: `native-static`, with serialized asset binding for the public
card. This audit covers the complete 11-method `Dreamer` class and all five
methods in its compiler-generated `Dreamer.<>c` helper type. The private native
exports are not committed. Exact metadata identities, signatures, RVAs, and
shared-body prototype choices are preserved in
[`../../targets/gameplay_role_dreamer.json`](../../targets/gameplay_role_dreamer.json).

## Public asset binding and variant split

The public `Dreamer` CharacterData is `sharedassets0.assets` path ID `21615`.
Its raw serialized object SHA-256 is
`6AD39DC4E26ADBCE0A722D4D92FBD0FB6084667A1BED4D4533CD3AFF0C1B6C35`, and its
SerializeReference role type is `Dreamer` from `Assembly-CSharp`. Its authored
description is:

```text
Pick 2 characters: Learn a role that's among them, and one that's not.
```

No serialized binding for `Dreamer2` or `DreamerOld` was found in the current
`sharedassets0.assets`, `level0`, or `resources.assets`. The distinct
[`Dreamer2` audit](gameplay_role_dreamer2.md) therefore remains an unbound
alternate and must not replace this public role-pair contract.

## Audited boundary

| Managed method | RVA | Surface |
| --- | ---: | --- |
| `Dreamer.<>c..cctor` | `0x3F0A90` | Generated singleton initialization |
| `Dreamer.<>c..ctor` | `0x357920` | Generated empty construction |
| `Dreamer.<>c.<ConjourInfo>b__10_0` | `0x3F07B0` | Identity key for ID sorting |
| `Dreamer.<>c.<ConjourInfo>b__10_1` | `0x392D50` | Random key for ID tie-breaking |
| `Dreamer.<>c.<ConjourInfo>b__10_2` | `0x392D50` | Random key for role-name ordering |
| `Dreamer.get_Description` | `0x3B9110` | Managed description literal |
| `Dreamer.GetInfo` | `0x3B8AC0` | Real-role passive record |
| `Dreamer.GetBluffInfo` | `0x3B8A60` | Bluff passive record |
| `Dreamer.Act` | `0x3B5ED0` | Normal picker setup |
| `Dreamer.CharacterPicked` | `0x3B7530` | Truthful result generation |
| `Dreamer.BluffAct` | `0x3B6140` | Bluff picker setup |
| `Dreamer.StopPick` | `0x3B8E00` | Callback cleanup |
| `Dreamer.CharacterPickedDrunk` | `0x3B63B0` | Lying result generation |
| `Dreamer.GetRandomNonRepeatedFakeCharacter` | `0x3B8B20` | Unique board-role fallback draw |
| `Dreamer.ConjourInfo` | `0x3B8650` | ID/name ordering and result formatting |
| `Dreamer.ctor` | `0x357920` | Folded base-only construction |

The two random-key lambdas share one native body. The two constructors also
share one heavily folded no-op body. Both managed identities remain explicit;
the target applies one ABI-compatible canonical prototype per shared RVA.

## Passive text and two-character picker

The managed `get_Description` literal is stale one-target wording: `Pick a
player. Learn an Evil role. If Evil player picked, learn correct info`. The
serialized public-card description and native action code are current: both
`Act` and `BluffAct` are no-ops outside Day and call
`CharacterPicker.StartPickCharacters(2, charRef)` on Day.

The normal path subscribes `CharacterPicked`; the bluff path subscribes
`CharacterPickedDrunk`. Both subscribe `StopPick`. Completion removes its own
handler and the stop handler, while cancellation removes both possible
completion handlers plus the stop handler. `GetInfo` and `GetBluffInfo` each
return a fresh empty `ActedInfo`.

## Truthful result generation

`CharacterPicked` preserves both selected `Character` references for the
emitted `ActedInfo`. Before normal role-pair generation, it checks the real
`dataRef.role` object of both targets. If either is the managed `Recluse` class
(the public Wretch), it immediately emits:

```text
Between
#<first id>, #<second id>
there is:
a Cabbage
```

This short circuit invokes `onActed` but bypasses the normal result logger.

Otherwise it uniformly chooses one of the two targets as the real-role anchor.
The first role name is the anchor's real `dataRef.characterName`; this path
does not use `registerAs` or the visible bluff name as the real role.

The second role is selected in this exact priority order:

1. If the unchosen target has a live `bluff` whose character name differs from
   the anchor's real name, use that bluff.
2. Otherwise, collect current-script CharacterData entries whose
   `usuallyDisguised` flag is set, remove both selected real CharacterData
   objects, and uniformly draw one when the list is nonempty.
3. Otherwise, copy `Gameplay.CurrentCharacters`, remove every board entry whose
   real `dataRef` or `bluff` equals the anchor role (and the prior fake candidate
   when one exists), uniformly draw a remaining board entry, and use that
   entry's real `dataRef.characterName`.

The fallback in step 3 is weighted by board entries, not unique role names.
Duplicate real roles therefore retain multiplicity. It also does not remove
the unchosen target merely because it was selected.

### Current-build authored-description violation

A read-only serialized audit found 46 current core CharacterData records at
`sharedassets0.assets` path IDs `21590` through `21635`; every record has
`usuallyDisguised == false`, including all Minions and Demons. No additional
CharacterData records were found in the other scanned gameplay assets.
Consequently step 2 is empty for the shipped roster and step 3 is a normal,
reachable path.

When two ordinary unbluffed targets have distinct roles, the fallback can draw
the unchosen target itself. The clue can therefore honestly contain both
selected roles, despite the authored promise that one role is not among them.
For a chosen anchor, that event has probability `1 / eligible board entries`.
Solver validation must preserve the native truth predicate of **at least one**
matching option; requiring exactly one would reject reachable truthful clues.

## Lying result generation

`CharacterPickedDrunk` starts a unique-by-object result list with each selected
target's live `bluff`. It then builds the same current-script
`usuallyDisguised` pool, excluding both selected real roles and any already
chosen bluff, and draws without replacement until it has two entries or the
pool is exhausted. That authored pool is empty in the current shipped assets.

Any missing result is filled by
`GetRandomNonRepeatedFakeCharacter(nonRepeatList)`. The helper walks every
entry in `Gameplay.CurrentCharacters`, considers its real `dataRef` followed by
its live `bluff`, excludes the supplied identities, deduplicates CharacterData
objects, and uniformly draws one identity. The first fill excludes both
targets' real and bluff identities. The second fill excludes those identities
plus the first result. Helper-generated options are therefore distinct and
absent from both targets' real and bluff identities. It is identity-uniform,
unlike the truthful board-entry fallback.

The initial target bluffs are appended before those exclusions. If one target's
bluff is the same CharacterData as the other target's real role, that option is
retained. Duplicate-bluff acquisition makes this a reachable cross-target
collision. The native body therefore does not itself enforce the authored hint
that both lying options are fake; zero underlying-role matches is the ordinary
case, not a complete reachability predicate.

The special Cabbage branch exists only in truthful `CharacterPicked`; the
lying callback always follows its fake-role pipeline.

## Formatting and random order

`ConjourInfo` sorts the two IDs ascending through an identity key, then applies
a `Random.value` secondary key. The random key matters only for duplicate IDs.
It independently orders the two role-name strings by `Random.value`, making
their displayed order approximately 50/50. The exact role-pair format is:

```text
Among
#<lower id>, #<higher id>
there is:
<role A> or <role B>
```

The role order carries no mapping to the target order. Both selected character
references are stored in the emitted `ActedInfo`, and the result is passed to
`onActed` and logged.

## Solver implications

- The public parser should require exactly two displayed target IDs and two
  nonempty, distinct role options, while retaining the historical one-target
  shape only for old builds and fixtures.
- Validation must compare against underlying `dataRef` roles while also using
  the known apparent bluffs to model native output reachability. A truthful pair
  needs at least one match, and the reachable two-match fallback must remain
  valid. A lying pair normally has zero matches, but a selected bluff can equal
  the other target's real role; an unconditional zero-match rule is too strict.
- A Cabbage clue proves a truthful Dreamer selected at least one Wretch.
- The recommender cannot treat the output as the sorted roles of both targets
  or create one deterministic `lie_<roles>` branch. The native observation is
  a weighted distribution over anchor choice, target bluffs, board-entry draws,
  and unique-identity draws.

## Typed-analysis corroboration

The target validates and baseline-exports 16 of 16 managed identities. The
deterministic typed union applies and read-only validates all 16 signatures and
43 Windows x64 parameter-storage locations, importing 40 reachable datatypes;
typed export also completes 16 of 16.

The body-free quality report at
[`../../reports/f530404b0f3f_807de4a83df4_typed_quality_gameplay_role_dreamer.json`](../../reports/f530404b0f3f_807de4a83df4_typed_quality_gameplay_role_dreamer.json)
passes its policy check. Typed analysis removes all 48 placeholder-parameter
tokens, reduces unresolved-type tokens from 190 to 92, reduces raw field-offset
accesses from 370 to 186, and introduces no decompiler-error or warning-count
regression.

## Remaining uncertainty

The role-local control flow and current serialized pools are closed. Shared
`CharacterPicker` behavior for attempted self-picks, repeated clicks, and
cancellation UI presentation remains a separate boundary. Localization variants
and the probability of the truthful two-match edge have not yet been
live-measured, although the native path and current asset reachability are
static facts.
