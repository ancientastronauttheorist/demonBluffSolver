# Gameplay Status, Corruption, Truth, And Bluff Boundary

Build: `f530404b0f3f_807de4a83df4`

Evidence status: **native-static** for the first 27 methods and the three base
bluff-storage/query methods in the checked boundary. No statement here is
based on live dynamic observation.

The checked target set is
[`reverse_engineering/targets/gameplay_status_corruption_truth.json`](../../targets/gameplay_status_corruption_truth.json).
Its baseline Ghidra export completed read-only at 40/40 functions with no
failures. The first slice covers status storage and removal, the selection
helpers used by corruption sources, actual and apparent lying, and disguised
appearance. The second slice covers the Pooka, Poisoner, Puzzlemaster, Drunk,
and Alchemist status lifecycle. The remaining 13 role and bluff-orchestration
methods are mapped but not yet claimed as native-audited here. Decompiled
bodies remain outside the repository.

## Status storage and cure

`CharacterStatuses.AddStatus` checks the exact enum in `resistances` first. A
matching resistance makes the entire call a no-op. Otherwise it unique-adds the
enum to `statuses`, then always replaces the one shared `targetCharacter`
reference with the supplied `targetRef`, even when the status was already
present. `sourceRef` is unused. The call does not consume resistance, dispatch a
callback, or retain one target per status.

`CharacterStatuses.AddResistance` unique-adds the exact enum to `resistances`.
It ignores both character arguments and neither removes an active status nor
changes `targetCharacter`.

`CharacterStatuses.CheckIfCanCurePoisonAndCure` first tests the shared
`targetCharacter` with Unity object semantics. When that target is live, it
virtually calls the predicate on the role declared by `targetCharacter.dataRef`,
`CheckIfCanRemoveStatus(Corrupted)`. A false result returns false without
removal. A null or destroyed target bypasses the role veto. The successful path
removes Corrupted and returns true even if the status was already absent; it
does not clear `targetCharacter`.

The base `Role.CheckIfCanRemoveStatus` always returns true. Drunk's override
returns false only for Corrupted. `CharacterStatuses.Contains` is a thin exact
membership check over the active `statuses` list, with no normalization or
special cases.

This shared-target design is consequential. A later non-resisted duplicate
`AddStatus` can replace an earlier role-aware target with null even though it
does not append the status again. Any later cure check sees only the most
recently supplied `targetRef`.

## Selection filters and circular ordering

All four selected filters allocate a fresh list, preserve source order and
duplicate occurrences, and leave the input unchanged:

- `FilterCharacterMissingStatus<Character>` retains entries whose raw active
  status list lacks the requested enum.
- `FilterCharactersWithoutResistance<Character>` retains entries whose raw
  resistance list lacks the requested enum.
- `FilterCharacterType<Character>` compares the apparent type: a Unity-live
  `registerAs` record takes precedence over `dataRef`.
- `FilterRealCharacterType<Character>` always compares `dataRef.type` and
  ignores `registerAs`.

These helpers do not silently discard malformed entries. A null input, element,
status container, inner list, or required character-data record reaches the
native failure path.

`Characters.GetAdjacentCharacters` copies `Gameplay.CurrentCharacters`, finds
the requested card with Unity equality, and returns the circular previous and
next entries in that order. An absent target or empty board returns an empty
list. A one-card board returns the same card twice; a two-card board returns the
other card twice. Preserved duplicate entries can therefore weight a later
random choice.

`CharactersHelper.GetSortedListWithCharacterFirst` starts with a copy of its
supplied list, but its rotation count and appended entries come from enumerating
the global `Gameplay.CurrentCharacters`. For every global entry before
`firstCharacter`, it appends that global entry and removes result index zero.
When the argument is the global list, this is an ordinary left rotation. With a
different list or ordering, global cards can be injected while supplied entries
are dropped. An absent `firstCharacter` processes the whole global list.

## Actual truth, apparent truth, and disguise

For the native predicates below, define:

- `baseLie = Unity-live bluff || alignment == Evil`;
- `H = HealthyBluff`;
- `C = Corrupted`;
- `T = AppearTruthfull`; and
- `L = AppearLying`.

`CharacterHelper.CheckLying`, which drives actual role dispatch, is exactly:

```text
C || (!H && baseLie)
```

It ignores `AppearTruthfull`, `AppearLying`, and the stored `Lying` status.
Corruption therefore outranks HealthyBluff; otherwise HealthyBluff suppresses
the ordinary evil-or-bluff lie.

`CharacterHelper.CheckLyingAppearance` is exactly:

```text
L || (!T && (C || (!H && baseLie)))
```

Its override precedence is thus AppearLying, AppearTruthfull, Corrupted,
HealthyBluff, then the base evil-or-bluff condition. This changes perceived
truth only; it does not change `Character.Act` dispatch.

`CharacterHelper.GetCharacterConditions` ignores its `condition` argument and
returns exactly one condition: `Lying` when the appearance expression is true,
otherwise `Truthfull`.

`CharacterHelper.CheckIfDisguisedAppearance` starts from whether `bluffRole` is
non-null, then walks active statuses in stored order. `AppearDisguised` sets the
result true and `AppearHonest` sets it false. When both exist, the later list
entry wins. Status insertion prevents a duplicate of each enum but does not
make these two statuses mutually exclusive.

## Bluff storage and fallback queries

`Character.GiveBluff` always stores the supplied `CharacterData` in `bluff`.
When that reference is live under Unity object semantics, it clones the
referenced role into `bluffRole`. A null or destroyed bluff does **not** clear
an existing `bluffRole`; it leaves that separate field untouched. The method
does not change alignment, register-as data, statuses, or reveal state.

`Character.GetCharacterBluffIfAble` returns `bluff` only when all four native
conditions hold: state is neither Dead nor Revealed, the separate `revealed`
flag is false, and the bluff reference is Unity-live. Every other path returns
`dataRef`. It does not consult `bluffRole`, lying/disguise status, alignment, or
the role's virtual bluff selector.

The base `Role.GetBluffIfAble` always returns null and mutates nothing. Its
constant-null native body is shared with hundreds of unrelated managed
identities, so the managed method identity and signature remain part of the
evidence even though the machine-code body is not unique to Role.

## Corruption producers

### Pooka Start neighbour corruption

`Pooka.Act` calls `PoisonNeighboursIfAble` only for `Start` (trigger 5); every
other trigger is a no-op. The helper rotates `Gameplay.CurrentCharacters` with
Pooka first, removes Pooka, then examines result index zero and the last result
entry. Each entry whose real `dataRef.type` is Villager receives Corrupted and
then MessedUpByEvil. Both calls use Pooka as the source and null as the shared
cure target.

Pooka does not prefilter an existing status, resistance, liveness, or
`registerAs`. The two status additions have independent resistance checks, so
Corrupted resistance can block Corrupted while the later MessedUpByEvil attempt
still succeeds. On a two-card board the sole other card is processed twice. A
board with fewer than two cards reaches the native failure path after the actor
is removed because index zero is read without a count guard.

### Poisoner Start adjacent corruption

`Poisoner.Act` does work only at `Start`. It obtains the circular adjacent pair,
then retains real Villagers that lack both active Corrupted and exact Corrupted
resistance. An empty candidate list is a clean no-op. Otherwise it makes one
random index draw and applies Corrupted followed by MessedUpByEvil, with
Poisoner as source and a null shared cure target.

The filters preserve duplicate occurrences and there is no liveness filter. On
a two-card board the one neighbour therefore occurs twice in the random pool,
although either index selects the same character.

### Puzzlemaster Start corruption and Day picker

`Puzzlemaster` is the internal role type embedded by the serialized display
`Plague Doctor` CharacterData; it is not a separate public role. At `Start`,
`Puzzlemaster.Act` copies every current card and filters by apparent
Villager type, missing Corrupted, and missing Corrupted resistance. It includes
self or a dead card when that entry otherwise passes. An empty pool is a clean
no-op; a non-empty pool produces one random draw and adds Corrupted with
Puzzlemaster as source and null as target. It does **not** add
MessedUpByEvil.

At `Day` (trigger 30), the role stores `charRef`, starts a one-character picker,
then combines `CharacterPicked` and `StopPick` handlers onto the two static
picker delegates. The picker starts before those subscriptions, and repeated
Day calls can combine additional handlers until external cleanup. The eventual
callback and clue-building bodies are outside this slice.

### Drunk self-corruption and bluff selection

`Drunk.Act` attempts one self-targeted Corrupted insertion at `Start`: Drunk is
both source and shared cure target. Exact Corrupted resistance makes the whole
call a no-op. A non-resisted duplicate still restores the shared target to
Drunk even though status insertion remains unique.

`Drunk.GetBluffIfAble` performs the same status mutation first, then requests a
random unique Villager bluff. When both the result and Gameplay singleton are
live, it registers that CharacterData through `AddScriptCharacterIfAble` and
returns it. Missing prerequisites have no fallback or clean null-return path,
and a later selection failure does not undo the earlier status mutation.

## Alchemist resistance, cure, and reporting

`Alchemist.OnInit` unique-adds exact Corrupted resistance. `Alchemist.Act` has
three relevant branches:

- `Init` (trigger 3) virtually dispatches the initialization hook.
- `Start` calls `CurePoisons` unless the character has BrokenAbility.
- `Day` obtains real info and invokes `onActed` when that delegate is non-null.

`Alchemist.BluffAct` has a different gate. At `Start`, WorkingAbility permits a
cure; either way, successful Start handling then replaces
`charRef.runtimeData` with a fresh zero-count `AlchemistRuntimeData`. At `Day`,
a non-null `onActed` delegate receives bluff info. BluffAct has no Init branch.

`GetPoisonedCharactersAroundMe` rotates the global current-character list with
the Alchemist first and removes self. It scans up to the first two entries,
then up to two entries from the end while stopping before index zero. It keeps
only entries whose raw active-status list contains Corrupted. It does not
prefilter cure permission, resistance, role, type, or liveness, and it does not
deduplicate scan overlap. With three or four total cards, result index one is
visited twice when Corrupted.

`CurePoisons` enumerates that returned list in order. Before every cure attempt,
it increments the role's cumulative `corruptions` counter; it ignores the cure
Boolean and does not reset the counter. Drunk vetoes, an already-removed status
revisited through overlap, and any other failed removal are therefore counted.
The accumulated counter is an attempt/list-entry count, not a distinct-
character or successful-cure count.

The scan reads live status at the moment each Alchemist cures, so a later cure
call sees mutations left by earlier calls. The shipped serialized order runs
all Init hooks first, then Pooka, Poisoner, Drunk, Puppeteer conversion,
Plague Doctor/Puzzlemaster, and Alchemist in that order. Duplicate Alchemists
act synchronously from highest displayed ID to lowest. The configured order
therefore makes Plague Doctor corruption visible to the first Alchemist, and
each later Alchemist sees cures left by the prior one rather than a shared
immutable snapshot.

## Native relationship boundary

The first-slice primitives join as follows without implying that status
insertion synchronously invokes cure:

```text
status producer -> CharacterStatuses.AddStatus -> stored status/shared target
later cure consumer -> CheckIfCanCurePoisonAndCure
                    -> dataRef.role.CheckIfCanRemoveStatus

Pooka     -> Corrupted + MessedUpByEvil on two real-type neighbours
Poisoner  -> Corrupted + MessedUpByEvil on one eligible real-type neighbour
Puzzle/PD -> Corrupted only on one eligible apparent-type current card
Drunk     -> Corrupted on self with the self role as cure-veto target
Alchemist -> live Corrupted scan -> counted cure attempts

Character.Act
  -> CharacterHelper.CheckLying
  -> not lying:      real Act,      then bluff-role Act
  -> lying non-Evil: real BluffAct, then bluff-role BluffAct
  -> lying Evil:     real Act,      then bluff-role BluffAct
```

`CheckLyingAppearance` and `CheckIfDisguisedAppearance` are query surfaces
separate from actual action dispatch.

## Managed-reconstruction corrections

Native review corrected four material managed-output gaps in this slice:

- Drunk's status-removal predicate is the simple `status != Corrupted` rule;
  the recovered managed body was structurally mangled.
- `GetSortedListWithCharacterFirst` is coupled to the global current-character
  list even when a different list is supplied; that dependency was missing from
  the apparent high-level reconstruction.
- Pooka and Poisoner use real character type while Puzzlemaster uses apparent
  type; treating all three candidate pools alike changes valid worlds.
- Alchemist increments its clue counter before each cure attempt and preserves
  overlap duplicates, rather than counting distinct or successfully cured
  characters.

## Metadata and prototype cautions

- `Role.CheckIfCanRemoveStatus` shares a constant-true native body with 864
  other metadata identities. Evidence must preserve the requested managed
  identity and exact signature, not only the primary native symbol.
- The selected `List<Character>` and earlier `List<CharacterData>` overloads of
  `Characters.FilterRealCharacterType` emit the same C identifier in
  Il2CppDumper output. The target's explicit `prototype_name` aliases only its
  GDT FunctionDefinition; `signature` remains the exact metadata signature.
- Eight methods in the full 40-method target are intentional exact overlaps
  with earlier checked boundaries and reuse their existing evidence where
  applicable.
