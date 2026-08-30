# Gameplay role: Judge (managed `Judge2`)

Build: `f530404b0f3f_807de4a83df4`

Evidence status: **native-static** for all ten methods declared by the shipped
role and for the dispatch, truth-appearance, click, and picker helpers that
determine its observable active-ability boundary. Serialized asset evidence
also establishes the public binding and `ResetAfterNight` usage. Native bodies
and decompiler output remain outside the repository.

The checked target set is
[`reverse_engineering/targets/gameplay_role_judge.json`](../../targets/gameplay_role_judge.json).
Its read-only baseline and typed Ghidra exports each completed at 18/18
functions with no failures. The body-free
[`quality report`](../../reports/f530404b0f3f_807de4a83df4_typed_quality_gameplay_role_judge.json)
passes its regression check: unresolved-type tokens fall from 146 to 80 and
raw field-offset accesses from 268 to 175, with no decompiler errors on either
side.

## Public asset binding and alternate split

The shipped `sharedassets0.assets` `CharacterData` at path ID `21623` is named
`Judge`, has `characterId` `Judge_87202475`, and binds its SerializeReference
role to managed `Judge2` in `Assembly-CSharp`. Its raw object SHA-256 is
`A7AE7EE62E9ABD6322514799BB66B931C3D3F1EC86ECF87684ED0C5EB1881BED`.
The serialized card is a Good Villager, is bluffable, is not usually
disguised, and has an active picker. Most importantly, its `abilityUsage` is
the enum value `10`, `ResetAfterNight`, not the one-use value.

The authored public description is:

```text
<b>Pick 1 character:</b>
Learn if they're Lying.
```

The managed role's separate description literal is:

```text
Pick 1 character.
Learn if he is lying
```

`Arbiter` is a distinct TypeDefIndex `5897` with ten separate native methods;
it is not the public Judge implementation in this build. A read-only scan of
`sharedassets0.assets`, `level0`, and `resources.assets` found the single
`Judge2` SerializeReference on path ID `21623` and no serialized `Arbiter`
reference. Runtime-name normalization must therefore map `Judge2` to public
Judge and must not substitute `Arbiter` merely because that alternate has a
similar method surface.

## Target boundary

| Managed identity | RVA | Boundary role |
| --- | ---: | --- |
| `Character.Act` | `0x3645C0` | Actor truth-aware role dispatch |
| `Character.OnClick` | `0x366270` | Picker-first click routing and ordinary ability gates |
| `Character.RoleAct` | `0x368790` | `Act`/`BluffAct` dispatch and acted-result hook |
| `CharacterHelper.CheckLying` | `0x397750` | Actual actor truth state |
| `CharacterHelper.CheckLyingAppearance` | `0x397630` | Target truth appearance queried by Judge |
| `CharacterPicker.CancelPick` | `0x378CC0` | Cancellation and stop dispatch |
| `CharacterPicker.StartPickCharacters` | `0x379260` | One-target picker setup |
| `CharacterPicker.ClickedCharacter` | `0x378DB0` | Exact-reference toggle and completion ordering |
| `Judge2.get_Description` | `0x3DFA00` | Managed active description |
| `Judge2.GetInfo` | `0x3DF690` | Truthful passive information surface |
| `Judge2.GetBluffInfo` | `0x3DF630` | Bluff passive information surface |
| `Judge2.Act` | `0x3DE600` | Truthful Day picker orchestration |
| `Judge2.StopPick` | `0x3DF6F0` | Normal/bluff callback cleanup |
| `Judge2.CharacterPicked` | `0x3DF020` | Truthful active result |
| `Judge2.BluffAct` | `0x3DE870` | Lying Day picker orchestration |
| `Judge2.CharacterPickedDrunk` | `0x3DEAE0` | Lying active result; not a Drunk-target rule |
| `Judge2.ConjourInfo` | `0x3DF560` | Exact result formatter |
| `Judge2.ctor` | `0x3CFFF0` | Folded base-role construction |

`Judge2` is TypeDefIndex `5898`. It declares no fields and exactly the final
ten methods in the table; every one has a native body. Its constructor shares
the same folded native body used by Slayer and many other fieldless roles.

## Dispatch and active setup

`Character.Act` decides whether the apparent role gets its normal or bluff
action from the actor's actual `CheckLying` result. For the shipped Good Judge,
an ordinary truthful actor reaches `Judge2.Act`, while a corrupted actor reaches
`Judge2.BluffAct`. `AppearTruthfull` and `AppearLying` do not alter this actor
dispatch; they affect only appearance queries.

Both Judge action methods are clean no-ops unless the trigger is `Day` (`30`).
On Day they:

1. call `StartPickCharacters(1, actor)`;
2. add their own completion callback to `OnCharactersPicked`; then
3. add `StopPick` to `OnStopPick`.

The normal path installs `CharacterPicked`; the bluff path installs the
historically named `CharacterPickedDrunk`. There is no target-role test and no
special handling for Drunk in either method.

`GetInfo` and `GetBluffInfo` each return a fresh `ActedInfo` whose description
is the empty string and whose character list is null. Judge is active-only;
those passive surfaces do not contain a hidden clue.

## Picker lifecycle and legal targets

`StartPickCharacters` clears the shared selected list, records the requested
count and exact actor reference, enters `PickCharacters`, and supplies no
candidate list or predicate. In `Character.OnClick`, the `PickCharacters`
branch calls `ClickedCharacter` before the ordinary `killedByDemon`, card
state, remaining-use, and active-pickable checks.

Consequently the native selection boundary accepts every board `Character`
whose click reaches `Character.OnClick`:

- self is legal;
- an ordinary dead card and a `killedByDemon` card are legal;
- a Hidden/unrevealed card has no native picker rejection;
- a target with its own unused active ability is selected rather than firing
  that target's Day ability, because picker routing has already taken
  precedence; and
- no role, alignment, reveal, corruption, execution, or prior-use filter is
  applied.

Unity presentation still determines whether a particular board object can be
physically clicked, but none of those properties makes it illegal once its
`OnClick` handler runs. Automation must not reject self, dead, hidden, or an
unused-active target on Judge's behalf.

`ClickedCharacter` toggles the exact object reference in insertion order. The
picker was freshly cleared and requires one target, so the first accepted
click immediately reaches completion with exactly one selected object. Before
the completion delegate runs, the picker sends trigger `OnPicked` (`70`) to
each selected target except the exact current picker; self therefore skips
that incidental trigger without being removed from the result.

Completion invokes Judge while the selected list is still populated. Judge
removes its own completion and stop handlers, emits its result, and returns.
The picker then returns to Day, clears the shared list, hides the panel, and
clears `CurrentPicker`. Explicit cancellation instead returns to Day, clears
the selection, invokes `StopPick`, and hides the panel; `Judge2.StopPick`
removes both possible Judge completion handlers and itself, producing no
acted result.

## Truth appearance and corrupted-Judge behavior

Judge queries `CharacterHelper.CheckLyingAppearance` on the target, not its
stored role, registered alignment alone, or the simpler actual-dispatch truth
helper. With:

- `baseLie = target.alignment == Evil || target.bluff is live`;
- `H = HealthyBluff`;
- `C = Corrupted`;
- `T = AppearTruthfull`; and
- `L = AppearLying`,

the exact target predicate is:

```text
appearsLying = L || (!T && (C || (!H && baseLie)))
```

The normal and bluff callbacks use the same predicate. Their only semantic
difference is whether the formatted answer is inverted:

| Actor callback | Target `appearsLying` | Public answer |
| --- | --- | --- |
| `CharacterPicked` | false | `saying Truth` |
| `CharacterPicked` | true | `Lying` |
| `CharacterPickedDrunk` | false | `Lying` |
| `CharacterPickedDrunk` | true | `saying Truth` |

This closes the corrupted-Judge question: corruption on a shipped Good Judge
forces the actor's `CheckLying` true, so `Character.Act` selects
`Judge2.BluffAct`, which deterministically negates the target's lying
appearance. Native behavior does **not** support treating a corrupted Judge
result as either value. Appearance overrides on the target still apply before
that inversion.

Self selection has no hard-coded output override. In the common states a
truthful Judge sees truthful self and a corrupted Judge sees lying self but
then inverts it, so both say `saying Truth`; explicit appearance overrides can
change that outcome through the same formula.

## Exact output and acted-information shape

The only two result templates are case-sensitive native literals:

```text
#{0} is
saying Truth
```

```text
#{0} is
Lying
```

The callbacks format `{0}` from the selected character's displayed integer
ID. `ConjourInfo` exposes the same mapping independently; the active callbacks
inline the formatting instead of calling it. There is no random branch.

Each successful use creates a new `List<Character>` by copying the still-live
picker list and constructs one `ActedInfo(resultText, copiedList)`. Because the
picker count is exactly one, the emitted character list has exact shape:

```text
characters.count == 1
characters[0] == the exact selected Character reference
```

The reference order is picker insertion order; for Judge there is only index
zero. The visible `#X` and `characters[0].id` name the same target. Cancellation
emits no `ActedInfo`.

## ResetAfterNight and result history

The serialized `ResetAfterNight` value is enforced by
`Character.RefreshCharacter` when gameplay returns from Night. It sets
`pickableUses` back to one and, when the apparent card is neither Hidden nor
Dead and is configured as picking, re-enables its active marker. That reset
does not clear `actedInfos` and does not clear `savedAct`.

The general acted-result coroutine appends every non-empty Judge `ActedInfo` to
the character's existing list. The list therefore remains chronological:

```text
actedInfos[0]               oldest successful use
actedInfos[actedInfos.Count - 1]  newest successful use
```

Every Day result decrements the current use counter. The display coroutine
then overwrites `savedAct` with that result's text, so `savedAct` corresponds
to the newest appended entry. The game's own `GetCurrentActedInfo` likewise
returns `actedInfos[Count - 1]`. Only character reinitialization or an explicit
memory-interference path clears/removes entries; the ordinary Night reset does
neither.

This matters to memory parsing after reuse. The outer `actedInfos` history may
legitimately contain two or more records, so a parser must not require
`len(actedInfos) == 1`. It should validate that **each** Judge record has one
reference, preserve older observations if the solver can use them, and compare
the newly clicked target and `savedAct` against the final record. The current
memory reader enumerates list indices from zero upward, so its returned order
is oldest to newest and the newest record is `acted_infos[-1]`.

## Active corpus surface

A deterministic read of the 426 checked-in `tests/cases_v2` fixtures found:

- 108 Judge deck entries across 108 cases;
- 105 apparent Judge cards across 87 cases;
- 78 parsed active results across 65 cases;
- 79 used Judge entries across 66 cases;
- 37 `Lying` and 41 `saying Truth` parsed results; and
- three self checks, all recorded as `saying Truth`.

`asc62_v7` is the one used Judge entry with empty parsed information. No
current v2 fixture stores a multi-night Judge history, so the reuse contract is
native-static rather than fixture-demonstrated. The role is nevertheless a
high-frequency live surface, and the deterministic inversion and permissive
picker materially affect scenario filtering and automation.

## Reconstruction consequences

- Bind public Judge to `Judge2`; keep `Arbiter` as an unbound alternate unless
  a future asset fingerprint proves otherwise.
- Validate Judge against target truth **appearance**, then leave it unchanged
  for the normal callback or invert it for the bluff callback. Corrupted Good
  Judge is the deterministic bluff case.
- Accept exactly one target per result, but accept one or more chronological
  result records after Night reuse.
- Cross-check both the exact speech target and the one stored character
  reference. For a new action, use the final history record.
- Candidate generation may include every board position, including Judge,
  executed/night-killed cards, hidden cards, and cards with unused active
  abilities.
- Match the two native result phrases rather than interpreting every unknown
  string as `saying Truth`.

The remaining uncertainty is presentation-only: this static boundary proves
what happens when a card click reaches `Character.OnClick`, but it does not
guarantee that every custom scene skin or future UI prefab exposes an enabled
click target. No native semantic uncertainty remains in the shipped Judge2
result, target predicate, reference shape, or Night-reuse behavior.
