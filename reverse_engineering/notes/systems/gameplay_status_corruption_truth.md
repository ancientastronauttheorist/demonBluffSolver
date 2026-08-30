# Gameplay Status, Corruption, Truth, And Bluff Boundary

Build: `f530404b0f3f_807de4a83df4`

Evidence status: **native-static** for the first 16 methods in the checked
boundary. No statement here is based on live dynamic observation.

The checked target set is
[`reverse_engineering/targets/gameplay_status_corruption_truth.json`](../../targets/gameplay_status_corruption_truth.json).
Its baseline Ghidra export completed read-only at 40/40 functions with no
failures. This first slice covers status storage and removal, the selection
helpers used by corruption sources, actual and apparent lying, and disguised
appearance. The remaining 24 role and bluff-orchestration methods are mapped
but not yet claimed as native-audited here. Decompiled bodies remain outside
the repository.

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

## Native relationship boundary

The first-slice primitives join as follows without implying that status
insertion synchronously invokes cure:

```text
status producer -> CharacterStatuses.AddStatus -> stored status/shared target
later cure consumer -> CheckIfCanCurePoisonAndCure
                    -> dataRef.role.CheckIfCanRemoveStatus

Character.Act
  -> CharacterHelper.CheckLying
  -> not lying:      real Act,      then bluff-role Act
  -> lying non-Evil: real BluffAct, then bluff-role BluffAct
  -> lying Evil:     real Act,      then bluff-role BluffAct
```

`CheckLyingAppearance` and `CheckIfDisguisedAppearance` are query surfaces
separate from actual action dispatch.

## Managed-reconstruction corrections

Native review corrected two material managed-output gaps in this slice:

- Drunk's status-removal predicate is the simple `status != Corrupted` rule;
  the recovered managed body was structurally mangled.
- `GetSortedListWithCharacterFirst` is coupled to the global current-character
  list even when a different list is supplied; that dependency was missing from
  the apparent high-level reconstruction.

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
