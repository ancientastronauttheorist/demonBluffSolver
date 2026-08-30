# Gameplay role: Fortune Teller

Build: `f530404b0f3f_807de4a83df4`

Evidence status: **native-static** for all 11 methods declared by the shipped
managed `FortuneTeller` role, all six methods in its compiler-generated
`FortuneTeller.<>c` helper, and the exact dispatch, click, picker,
registered-alignment, and acted-record helpers needed to close the active
ability. Serialized asset evidence fixes the public role binding, two-target
contract, `ResetAfterNight` usage, and attached achievement. Native bodies and
decompiler output remain outside the repository.

The checked target set is
[`reverse_engineering/targets/gameplay_role_fortune_teller.json`](../../targets/gameplay_role_fortune_teller.json).
Its read-only baseline and typed Ghidra exports each complete at 25/25
functions with no failures. The body-free
[`quality report`](../../reports/f530404b0f3f_807de4a83df4_typed_quality_gameplay_role_fortune_teller.json)
passes its regression check: unresolved-type tokens fall from 171 to 67, raw
field-offset accesses from 286 to 194, placeholder parameter tokens from 163
to zero, and indirect-call patterns from 15 to five. Both exports contain zero
decompiler-error markers and 43 warning markers.

## Public asset binding and authored contract

The shipped `sharedassets0.assets` `CharacterData` at path ID `21619` is named
`Fortune Teller`, has `characterId` `Fortune Teller_74565681`, and binds its
SerializeReference role to managed `FortuneTeller` at TypeDefIndex `5880` in
`Assembly-CSharp`. Its raw object SHA-256 is
`923B4EA4F61D42945A0C376A45C3D48027890775FACC6BCBE8E5F22867BE2344`.

The card is a Good Villager (`characterType == 10`,
`startingAlignment == 10`), is bluffable, is not usually disguised, and has
`picking == true`. Its `abilityUsage` is enum value `10`,
`ResetAfterNight`. The exact public description is:

```text
<b>Pick 2 characters:</b>
Learn if any of them is Evil
```

The managed description getter retains slightly older wording:

```text
Pick 2 players. Learn if any of them is Evil
```

The asset carries one achievement, `Twice as Evil`, at path ID `21579`. Its
authored condition is `Pick exactly 2 Evil characters with a non-Lying Fortune
Teller.`, and its native unlock key is `FTeller_ACHIV_7689`.

## Audited boundary and shared bodies

| Group | Methods | Observable purpose |
| --- | ---: | --- |
| `FortuneTeller.<>c` | 6 | Singleton/cache setup, ID keys, and random secondary keys |
| `FortuneTeller` | 11 | Complete passive surface, normal/bluff picker paths, output, and achievement |
| `Character` | 4 | Actual truth dispatch, picker-first click routing, and registered alignment |
| `CharacterPicker` | 3 | Setup, exact-reference toggling, completion, and cancellation |
| `ActedInfo` | 1 | Exact result text and reference-list storage |

The 25 target memberships select 25 distinct managed FunctionDefinitions but
only 23 native RVAs. The normal and bluff ID selectors share `0x3F0790`, and
their random secondary-key selectors share `0x392D50`. Six selected identities
have bodies shared with methods outside this target:

- the generated constructor at `0x357920` uses the canonical
  `Dreamer..ctor` prototype and has 3,052 aliases;
- each ID selector uses a body with eight aliases;
- each random secondary-key selector uses the canonical Dreamer helper
  prototype at a body with 15 aliases; and
- `FortuneTeller..ctor` at `0x3CFFF0` uses the canonical Slayer constructor
  prototype and has 537 aliases.

Those folded bodies are ABI-compatible implementation reuse. They do not
collapse the six compiler-generated identities or the role constructor in the
method ledger.

## Dispatch and picker setup

`GetInfo` and `GetBluffInfo` each return a fresh `ActedInfo` with an empty
description and a null character-reference list. Fortune Teller's meaningful
information is produced only by the active callbacks.

`Character.Act` and `Character.RoleAct` select the apparent role's normal or
bluff action from the acting card's actual truth state. An ordinary clean
public Fortune Teller reaches `FortuneTeller.Act`; a corrupted Good Fortune
Teller reaches `FortuneTeller.BluffAct`. A lying Evil card presenting Fortune
Teller likewise uses the bluff action. Target appearance, target corruption,
and the historically named `CharacterPickedDrunk` method do not choose this
dispatch.

Both action methods are no-ops unless the trigger is Day (`30`). On Day they:

1. call `CharacterPicker.StartPickCharacters(2, actor)`;
2. subscribe either `CharacterPicked` or `CharacterPickedDrunk` to completion;
   and
3. subscribe `StopPick` to cancellation.

The normal action installs `CharacterPicked`; the bluff action installs
`CharacterPickedDrunk`. There is no random choice between callbacks and no
special test for a Drunk actor or target.

## Legal targets, toggling, and completion order

`StartPickCharacters` clears the shared selection list, records count two and
the exact actor, enters `PickCharacters`, and supplies no candidate list or
predicate. `Character.OnClick` routes the `PickCharacters` state before the
ordinary killed-by-Demon, card-state, remaining-use, and active-pickable
checks. Therefore every board `Character` whose click reaches `OnClick` is a
native-legal Fortune Teller target:

- self is legal;
- ordinary dead and killed-by-Demon cards are legal;
- Hidden or otherwise unrevealed cards have no picker rejection;
- a card with its own unused active ability is selected instead of activating
  that ability; and
- no role, faction, alignment, corruption, execution, reveal, or prior-use
  filter is applied.

Unity presentation can still determine whether a board object is physically
clickable, but none of those properties is rejected at the native picker
boundary.

`ClickedCharacter` toggles the exact object reference and preserves insertion
order while selection is in progress. Clicking the same exact card again
removes it. Completion requires the list count to reach two, so a normal
result contains two distinct `Character` objects; malformed distinct objects
with duplicate displayed IDs are not rejected.

Once the count reaches two, the picker first dispatches `OnPicked` (`70`) to
each selected object in insertion order except the exact current picker. It
then invokes Fortune Teller's completion delegate while the shared selection
list is still populated. Only after the callback returns does the normal
completion path restore Day, clear the selection, hide the picker panel, and
clear the current picker. Registered alignment is consequently evaluated
after any reached `OnPicked` actions, not at initial click time.

## Truth and lying registered-alignment semantics

The normal callback starts with `False` and changes the result to `True` if
**any** selected card has `Character.GetRegisterAlignment() == Evil (20)`.
The helper prefers a live `registerAs.startingAlignment`; if no register-as
record is live, it returns the physical card's runtime alignment. It does not
query the selected card's displayed bluff, truth appearance, or Corrupted
status.

The bluff callback is the exact logical complement. It starts with `True` and
changes the result to `False` if any selected card registers Evil. The output
is therefore deterministic for the selected pair:

| Selected registered alignments | Normal callback | Bluff callback |
| --- | --- | --- |
| neither is Evil | `False` | `True` |
| one or both are Evil | `True` | `False` |

`CharacterPickedDrunk` consumes one unconditional floating-point
`Random.Range(0, 1)` draw before this scan, but discards the returned value.
That call advances Unity random state without making the lie probabilistic.
Both callbacks later evaluate random secondary sort keys as described below.

Consequences of using registered alignment rather than apparent role include:

- a normal Corrupted Good target still registers Good and does not by itself
  make the normal result True;
- an ordinary real Drunk is Corrupted and lying but still runtime Good, so it
  does not count as Evil;
- an Evil card's Good-looking bluff does not by itself make it register Good;
  and
- public Wretch's Evil Minion register-as record makes it count as Evil even
  though its physical runtime alignment is Good.

The `Drunk` suffix is thus historical naming for the lying actor path, not a
Drunk-target exception.

## Exact ordering, output, and reference shape

Each callback copies the two selected references, sorts them by displayed
integer ID ascending, applies `Random.value` as a secondary key, and materializes
that order as a new list. The secondary key is evaluated as part of sorting but
can change order only when displayed IDs tie. Normal boards have unique IDs,
so their speech and stored references are deterministically low-ID first;
malformed distinct objects with the same ID are randomly tie-broken.

`ConjourInfo` has exactly two case-sensitive templates:

```text
Is #{0} or #{1} Evil?: False
```

```text
Is #{0} or #{1} Evil?: True
```

For every successful ordinary result, the stored shape is:

```text
characters.count == 2
characters[0].id == first displayed ID
characters[1].id == second displayed ID
characters[0] and characters[1] are the exact selected Character references
```

The callback emits one `ActedInfo(resultText, sortedReferences)` through
`onActed` and logs the same text. Initial picker insertion order is not the
result order. Neither callback stores a role identity, raw faction value, or
hidden evil position outside those exact references and the Boolean text.

## Cancellation, reset, and history

`StopPick` removes both possible Fortune Teller completion handlers and its own
stop handler. Explicit `CancelPick` restores Day, clears the partial selection,
invokes that stop delegate, and hides the panel. Cancellation emits no
`ActedInfo`, consumes no completed result, and cannot leave a Fortune Teller
completion handler subscribed.

The serialized `ResetAfterNight` value uses the already audited generic
`Character.RefreshCharacter` boundary. Returning from Night restores
`pickableUses` to one and re-enables the active marker when the apparent card
is neither Hidden nor Dead and remains configured as picking. It does not clear
`actedInfos` or `savedAct`.

Every successful use therefore appends another two-reference observation in
chronological history, while the displayed saved action becomes the newest
text. A reused Fortune Teller can legitimately have more than one historical
record; parsers should validate every record's exact two-target shape and use
the final record for a newly completed action.

## Achievement boundary

After the truthful callback emits and logs its result, it checks both sorted
reference slots independently with `GetRegisterAlignment`. If both are exact
Evil, it requests unlock key `FTeller_ACHIV_7689`. The separate
`CheckAchievementsAndUnlockIfAble` method performs the same two-index test on
the supplied `ActedInfo`.

The bluff callback never calls the achievement path. Thus the authored
“non-Lying” requirement means the actor reached the normal callback, while
“exactly 2 Evil” means both of the picker's two registered-alignment records
are Evil. Display bluffs and selected-card corruption do not substitute for
that test.

## Typed-union accounting

Eight target memberships are exact managed-identity overlaps with the previous
20 target sets: the four `Character` methods, all three `CharacterPicker`
methods, and `ActedInfo..ctor`. The target therefore adds 17 distinct managed
FunctionDefinitions. Pairwise folding of the four generated ordering helpers
reduces those definitions by two RVAs inside the target, while the generated
constructor, random-key body, and role constructor already have canonical RVAs
in the earlier union. The exact target delta is 12 new native RVAs.

The deterministic 21-set union contains 479 memberships, 324 distinct selected
FunctionDefinitions, and 299 unique native RVAs. All 17 new managed identities
remain explicit even when they reuse an earlier canonical body.

## Corpus and reconstruction implications

A deterministic scan of the 426 checked-in `tests/cases_v2` fixtures finds:

- 115 Fortune Teller deck entries across 115 cases;
- 128 apparent Fortune Teller cards across 107 cases;
- 95 parsed active results across 84 cases; and
- zero parsed Fortune Teller results whose target list is not exactly two.

Reconstruction and solver validation should therefore:

- generate two distinct exact board targets without state, self, or
  unused-active exclusions;
- evaluate the OR predicate from registered alignment at completion time;
- leave that predicate unchanged for a truthful actor and deterministically
  invert it for a lying actor;
- require the speech IDs and `ActedInfo` references to be the same ascending-ID
  pair;
- preserve every chronological record across Night reuse; and
- treat the discarded bluff-path random draw and secondary sort keys as RNG
  state effects, not output uncertainty for ordinary unique-ID boards.

The remaining uncertainty is presentation-only: the native boundary proves
what happens after a click reaches `Character.OnClick`, but it does not prove
that every future scene skin exposes every state of card as an enabled UI
target.
