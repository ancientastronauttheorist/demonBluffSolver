# Gameplay role: Plague Doctor (managed `Puzzlemaster`)

Build: `f530404b0f3f_807de4a83df4`

Evidence status: **native-static** for all 11 methods declared by the shipped
role and for the picker, status, alignment, activation, and dispatch helpers
that determine its complete observable boundary. Native bodies and decompiler
output remain outside the repository.

The checked target set is
[`reverse_engineering/targets/gameplay_role_plague_doctor.json`](../../targets/gameplay_role_plague_doctor.json).
Its read-only baseline and typed Ghidra exports each completed at 23/23
functions with no failures. The body-free
[`quality report`](../../reports/f530404b0f3f_807de4a83df4_typed_quality_gameplay_role_plague_doctor.json)
passes its regression check: unresolved-type tokens fall from 190 to 123 and
raw field-offset accesses from 329 to 223.

## Public asset binding and ordered Start slot

The shipped `sharedassets0.assets` `CharacterData` at path ID `21606` is named
`Plague Doctor`, has `characterId` `Plague Doctor_49312486`, and binds its
SerializeReference role to managed `Puzzlemaster` in `Assembly-CSharp`. Its raw
object SHA-256 is
`68B2EC4E9ECD5C9C5EC8451B981971DADE0279922EED88274DB25BA31C2BBAFB`.
The serialized card is Outcast/Good and has a one-use active ability.
Its current authored description is:

```text
[1 Villager is Poisoned] Pick 1 character: if its Poisoned I learn an Evil character.
```

The asset's hint also states that selecting self always reports Not Corrupted,
even if the actor is Evil. The native role uses the `Corrupted` status despite
the older player-facing word “Poisoned.” Runtime-name normalization must map
managed `Puzzlemaster` to public Plague Doctor.

The `level0` `Characters` component at path ID `137026` references path ID
`21606` at zero-based index 8 of `startGameActOrder`. Plague Doctor therefore
runs after Baa and before Shaman and Alchemist. The global dispatcher supports
multiple matching Puzzlemaster data entries, highest displayed ID first, but
the unique Standard role pool and the pre-Plague-Doctor transformations do not
naturally create two. Multiple Start calls are a latent custom/modded or
externally retriggered state, not a normal Standard assumption.

## Target boundary

| Managed identity | RVA | Boundary role |
| --- | ---: | --- |
| `Character.Act` | `0x3645C0` | Truth-aware role dispatch |
| `Character.OnClick` | `0x366270` | Active-ability and picker click gates |
| `Character.RoleAct` | `0x368790` | Real-versus-bluff virtual dispatch and `onActed` hook |
| `CharacterStatuses.AddStatus` | `0x363AA0` | Resistance-aware Start status insertion |
| `CharacterStatuses.Contains` | `0x363C40` | Raw active-status query used by both callbacks |
| `Characters.FilterCharacterMissingStatus` | `0x36A8C0` | Start status exclusion |
| `Characters.FilterCharactersWithoutResistance(Character)` | `0x36AE50` | Start resistance exclusion |
| `Characters.FilterCharacterType(Character)` | `0x36AC30` | Apparent-Villager Start pool |
| `Characters.FilterAlignmentCharacters(Character)` | `0x36A030` | Apparent Good/Evil reveal pools |
| `CharacterPicker.CancelPick` | `0x378CC0` | Cancellation and stop dispatch |
| `CharacterPicker.StartPickCharacters` | `0x379260` | One-target picker setup |
| `CharacterPicker.ClickedCharacter` | `0x378DB0` | Target toggling and completion order |
| `Puzzlemaster.get_Description` | `0x3E85B0` | Managed active description |
| `Puzzlemaster.GetInfo` | `0x3E80A0` | Passive truthful information surface |
| `Puzzlemaster.GetBluffInfo` | `0x3E8040` | Passive bluff information surface |
| `Puzzlemaster.Act` | `0x3E6D90` | Truthful Start and Day orchestration |
| `Puzzlemaster.PoisonRandomVillager` | `0x3E8100` | Start corruption helper |
| `Puzzlemaster.StopPick` | `0x3E82A0` | Normal/bluff callback cleanup |
| `Puzzlemaster.CharacterPicked` | `0x3E7960` | Truthful Day result |
| `Puzzlemaster.BluffAct` | `0x3E7190` | Lying Day orchestration |
| `Puzzlemaster.CharacterPickedDrunk` | `0x3E7410` | Lying Day result; not a Drunk-target rule |
| `Puzzlemaster.ConjourInfo` | `0x3E7EB0` | Exact clue formatting and self override |
| `Puzzlemaster.ctor` | `0x3CFFF0` | Folded base-role construction |

`Puzzlemaster` is TypeDefIndex `5909`. It declares exactly one field,
`private Character self` at offset `0x48`, and exactly the final 11 methods in
the table. All 11 have native bodies.

## Dispatch and ability activation

`Character.Act` selects the role surface from the actor's live truth state. In
the normal reachable cases, a truthful real Plague Doctor reaches
`Puzzlemaster.Act`; a lying non-Evil Plague Doctor reaches
`Puzzlemaster.BluffAct`; and an Evil actor carrying Puzzlemaster as its real
role still uses the real action while its disguise role receives the bluff
action. `Character.RoleAct` installs the character's acted-information handler
on the role immediately before virtual dispatch.

Neither Puzzlemaster action checks `WorkingAbility` or `BrokenAbility`.
`Character.OnClick` directly requires `killedByDemon == false`, at least one
`pickableUse`, a non-Hidden card, and an active `pickable` GameObject before
it dispatches `Day` (`30`). Any additional presentation-level enablement is
outside the role. This boundary must not invent a status gate inside
Puzzlemaster.

`Puzzlemaster.Act` handles only two triggers:

- `Start` (`5`) runs the corruption selection below.
- `Day` (`30`) stores `charRef` in `self`, starts a one-character picker, then
  combines `CharacterPicked` and `StopPick` onto the static picker delegates.

Every other trigger is a no-op. `Puzzlemaster.BluffAct` handles only `Day`: it
stores self, starts the same picker, and registers `CharacterPickedDrunk` plus
`StopPick`. It has no `Start` branch. Consequently a lying non-Evil real Plague
Doctor does not perform the role's Start corruption.

Both Day methods call `StartPickCharacters` before subscribing their handlers.
Normal picker setup does not complete synchronously, but an externally repeated
Day dispatch can accumulate handlers until completion or cancellation removes
them. The ordinary once-per-round click path prevents that repetition.

## Truthful Start corruption

At Start, `Puzzlemaster.Act` makes a shallow copy of
`Gameplay.CurrentCharacters` and applies these filters in order:

1. apparent type is Villager (`10`);
2. active Corrupted (`10`) is absent; and
3. exact Corrupted resistance is absent.

The apparent-type helper uses a Unity-live `registerAs` record when present and
falls back to real `dataRef`; it does not inspect the current bluff. The filters
preserve source order and duplicate occurrences. There is no alignment,
liveness, self, or `MessedUpByEvil` predicate.

An empty valid pool is a clean no-op. Otherwise the method samples one uniform
index and calls `CharacterStatuses.AddStatus(Corrupted, charRef, null)` on the
selected card. It does not add `MessedUpByEvil`. Self and dead cards remain
eligible if their apparent type and status/resistance state pass the filters.
Malformed singleton/list/status dependencies reach native failure paths; work
already performed is not transactional.

`PoisonRandomVillager` exposes the same selection and insertion behavior as a
separate declared helper. The optimized `Act` body contains the corresponding
logic directly rather than making an observable managed helper call.

## Picker lifecycle and legal Day targets

`CharacterPicker.StartPickCharacters(1, actor)` updates the panel, records a
one-target requirement, clears the shared picked list, changes gameplay state
to PickCharacters (`30`), and stores the current picker. It silently returns if
gameplay is already in state `50`. It supplies no candidate list and applies no
role, alignment, status, self, or liveness restriction.

While gameplay is in PickCharacters, `Character.OnClick` forwards the clicked
card to `ClickedCharacter` before its ordinary `killedByDemon` check. The picker
toggles that exact reference in the shared list. When the count reaches one it:

1. invokes trigger `70` on every selected card except the current picker;
2. invokes `OnCharactersPicked` while the shared picked list is still intact;
3. changes gameplay state to Day (`10`), clears the picked list, invokes
   `OnStopPick`, and hides the panel; and
4. clears `CurrentPicker` on this completed-selection path.

Self is therefore a legal target and merely skips the target's trigger-70
dispatch. No native picker or click-path test rejects a dead/executed card; if
its board object remains clickable, it is legal. `CancelPick` performs the Day
transition, list clear, stop callback, and panel hide, but does not itself clear
`CurrentPicker`.

`Puzzlemaster.StopPick` removes both possible Puzzlemaster completion handlers
and its stop handler. Each completion callback first removes its own completion
handler and the stop handler before reading the selected card.

## Truthful Day result

`CharacterPicked` reads `PickedCharacters[0]` and directly queries that card's
active status collection with `Contains(Corrupted)`. It does not inspect real
role, apparent role, status source, shared cure target, or any Drunk marker.

For a clean target, the result character remains null. For a Corrupted target,
the callback copies `Gameplay.CurrentCharacters`, filters to apparent Evil
alignment (`20`), and samples one uniform index. Alignment filtering prefers
the live `registerAs.startingAlignment` and otherwise uses the runtime
character alignment. It does not read the current bluff or filter self,
liveness, death, execution, or the selected target. Source multiplicity and
order remain in the random pool.

There is no empty-pool guard after alignment filtering. A Corrupted target with
no apparent-Evil candidate reaches the native random/list failure path; it does
not degrade to a clean clue.

## Lying Day result

`CharacterPickedDrunk` is the completion callback installed by
`Puzzlemaster.BluffAct`. Its name describes the historical lying-role path; it
does **not** test whether the selected character is Drunk.

The callback performs the same raw `Contains(Corrupted)` query with the
opposite random branch:

- a clean selected card causes a uniform draw from apparent Good alignment
  (`10`), and that Good card is falsely named Evil while the selected card is
  called Corrupted;
- a Corrupted selected card leaves the result null and reports the selected
  card as Not Corrupted.

The Good pool uses the same register-as-first alignment rule and has no
liveness, self, selected-target, or empty-pool guard. A clean target with no
apparent-Good candidate therefore reaches the native failure path.

## Clue construction and acted-information shape

Both callbacks pass their result and selected card to the intentionally spelled
`ConjourInfo`. It checks whether the selected card is `self` before it checks
the result pointer. The exact outputs are:

```text
#{picked} is
Not Corrupted
```

or:

```text
#{result} is Evil
#{picked} is Corrupted
```

Selecting self always takes the first form, even when a random result was
already drawn. This formatting override does not erase that result pointer.

Each callback builds a fresh character list by adding the picked list and then
adding the result pointer unconditionally. The emitted `ActedInfo.characters`
shape is therefore always `[picked, result-or-null]`; a clean result contains a
real null second entry rather than a one-element list. A non-null `onActed`
handler receives a fresh `ActedInfo` with the formatted string and this list,
and the string is logged independently.

`GetInfo` and `GetBluffInfo` are semantically identical passive surfaces: each
returns a fresh `ActedInfo` with an empty string and null character list. The
active callbacks, not those getters, produce the player-facing result.

## Drunk and the asc84_v2 clean observation

The native Plague Doctor query has no blanket Drunk exception. A Drunk with an
active Corrupted status enters the truthful Corrupted branch and can reveal an
apparent-Evil character. Drunk's own Start action normally attempts
`AddStatus(Corrupted, self, self)`: exact resistance can prevent insertion, and
the self shared-cure target vetoes later Alchemist removal when insertion
succeeds. Plague Doctor ignores that cure provenance and observes only whether
the active status is present at click time.

The live `asc84_v2` result is consistent with this boundary rather than an
exception. Chancellor generated Drunk on physical card #6. Chancellor's
`Character.Init` clears active statuses but preserves the physical resistance
collection; if the replaced Villager was Alchemist, its already-installed
Corrupted resistance survives on the new Drunk. Drunk's Start insertion is then
resisted, so truthful Plague Doctor correctly reports #6 as Not Corrupted even
though Drunk still lies intrinsically and retains its special execution damage.
A solver must retain that clean resistant history and must not force every
Drunk clean on the Plague Doctor surface.

## Construction and implementation consequences

The constructor uses the ordinary folded base-role body at `0x3CFFF0`; the
target retains exact `Puzzlemaster.ctor` identity while applying the established
canonical constructor prototype for the shared native body.

## Solver checkpoint: ordered duplicate Start replay

The clean-room Start simulator now preserves the global dispatcher's duplicate
exception instead of reducing Plague Doctor presence to a boolean. Scenario
construction enumerates the exact selected physical PD actors, including
natural duplicate pool copies and a Chancellor-generated copy beside a natural
one. Puppeteer-overwritten characters are removed before the PD slot. Actors
run from highest displayed ID to lowest; each actor rebuilds the eligible live
Villager pool, must choose when it is nonempty, and records an explicit no-op
when earlier actors exhausted it.

Each pure outcome carries an internal ordered `(actor, target-or-none)` trace.
That trace survives Alchemist convergence without exposing a hidden target as
public solver knowledge. Uniform target-history multiplicity is used as exact
conditional mass only when the whole result has one structural Evil root, one
Start context, exactly one PD actor, and no Chancellor, Shaman, Poisoner, Twin,
or Puppeteer writer. Opaque or multiple roots still deduplicate to equal
logical worlds; assigning them an absolute scalar weight would be unsound
because their hidden identity priors are grouped rather than generative.

Focused regressions cover descending duplicate order, live candidate removal,
pool exhaustion, duplicate authored PD selection, Chancellor overlap,
Puppeteer replacement, Alchemist convergence, and the exact three-target case
where a Knight's native corruption/execution risk is `1/3` rather than `1/2`.
The pure post-Twin corruption boundary now also replays a latent Shaman-copied
PD on a caller-proven ordinary runtime-Good destination: it runs after the
global history, rebuilds the live pool, records selected/no-candidate
provenance separately, and preserves three-way target mass through later
Alchemist convergence. Normal shipped scenario generation deliberately does
not emit that trace because initial bluffs and register-as pointers are still
null at Shaman's pre-Reveal slot, so an ordinary Villager source cannot supply
Outcast PD data. Runtime-Evil/stale-bluff composition and general provenance-
bearing probability factors remain separate replay frontiers.

The clean-room behavioral contract is:

- Start corrupts one eligible apparent Villager and only the real truthful
  `Act(Start)` surface owns that behavior.
- Day may target self or any other clickable board card, including a dead card;
  target legality is not restricted to Villagers.
- Truth and bluff results branch on raw active Corrupted presence, with no
  role-specific Drunk override.
- Truthful Corrupted results draw apparent Evil; lying clean results draw
  apparent Good; both pools include dead/executed entries and fail when empty.
- Self always formats Not Corrupted, but the two-entry acted-information list
  can still retain the random result.
- Validation must preserve random candidate support and register-as alignment
  rather than substitute real alignment or an alive-only pool.
