# Gameplay Status, Corruption, Truth, And Bluff Boundary

Build: `f530404b0f3f_807de4a83df4`

Evidence status: **native-static** for all 40 methods in the checked boundary.
No statement here is based on live dynamic observation unless explicitly
identified as a scheduling inference.

The checked target set is
[`reverse_engineering/targets/gameplay_status_corruption_truth.json`](../../targets/gameplay_status_corruption_truth.json).
Its baseline Ghidra export completed read-only at 40/40 functions with no
failures. The first slice covers status storage and removal, the selection
helpers used by corruption sources, actual and apparent lying, and disguised
appearance. The second slice covers the Pooka, Poisoner, Puzzlemaster, Drunk,
and Alchemist status lifecycle. The final slice covers bluff storage and
lookup, Puppet/Puppeteer conversion, Doppelganger selection, Confessor
appearance, and the internal Reveal boundary. Five exact overlaps reuse the
earlier native audits of `Character.Reveal`, `Character.Init`,
`Characters.ManageCharacters`, `Character.Act`, and `Character.RoleAct`.
Decompiled bodies remain outside the repository.

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

## Puppet bluff and Start statuses

`Puppet.GetBluffIfAble` ignores its character argument. It obtains the current
script roles in Villager, Outcast, Minion, then Demon pool order, retains
`CharacterData` whose stored alignment is Good and whose `bluffable` flag is
set, and returns one random entry. It does not require Villager type or current
board presence. An empty pool or missing singleton/list dependency reaches the
native failure path rather than returning no bluff.

`Puppet.Act` handles only Start. It attempts `HealthyBluff`, then
`BrokenAbility`, then `MessedUpByEvil`, using Puppet as source and null as the
shared target for every insertion. Each exact resistance independently blocks
its own status. Other triggers are clean no-ops; a missing character or status
container fails.

## Puppeteer conversion

The public Puppeteer role is managed type `Mezepheles`; its configured
`minionId` resolves the public Puppet data. At Start it rotates current cards
with Puppeteer first, removes Puppeteer, and examines only the first and last
remaining entries--the two physical circular neighbours. It retains each
occurrence whose real `dataRef.type` is Villager, ignoring register-as data,
alignment, statuses, resistance, and liveness.

It then removes only the first retained neighbour whose real role object is
`SaintVillager`, the role embedded by the shipped Villager named Saint, and
stops searching for another Saint. Bombardier instead embeds `Saint` and has
real type Outcast, so it never enters this Villager candidate pool. If no
candidate remains, conversion is a clean no-op. Otherwise one candidate is
selected randomly and conversion is mandatory. Preserved multiplicity matters
on a two-card board: the same Saint neighbour occurs twice, so removing one
occurrence still leaves a convertible candidate.

For the selected target, the native mutation order is:

1. save its old `CharacterData`;
2. resolve Puppet data through `minionId`;
3. call `Character.Init(puppetData, -100)`;
4. call `GiveBluff(savedOldData)`;
5. add `HealthyBluff`;
6. add `BrokenAbility`;
7. add `AlteredCharacter`; and
8. add `MessedUpByEvil`.

All four statuses use Puppeteer as source and null target. `Init` clears active
statuses, bluff, register-as, runtime/acted state, and the Start latch while
preserving physical ID and resistance storage. The Puppet therefore displays
the victim's former true Villager identity, not Puppet data. Malformed
neighbours, missing game data, a failed Puppet lookup, or missing status
storage can fail after earlier mutations; the operation is not transactional.

The reset Start latch is consumed later at Puppet's serialized Start slot.
HealthyBluff makes dispatch truthful: real `Puppet.Act(Start)` runs, followed
by the copied Villager role's `Act(Start)`. `BrokenAbility` is already present
when that copied action runs, which prevents a copied Alchemist from becoming
an ordinary extra cure actor.

## Doppelganger bluff selection

The native managed spelling is `Doppleganger`. Its clean selection branch acts
in this order:

1. add `HealthyBluff` to self;
2. copy current board cards;
3. retain sources whose real `dataRef.bluffable` is set;
4. retain apparent Villagers by `(live registerAs ?? dataRef).type`;
5. retain runtime alignment Good; and
6. randomly select a source and return its real `dataRef`.

There is no explicit self-removal. The real Outcast normally excludes itself
because Doppelganger's base register-as result is null. An empty source pool
fails rather than yielding no disguise.

When Doppelganger is Corrupted, it does not add HealthyBluff. It instead keeps
real-bluffable sources with runtime alignment Evil, with no type filter, then
returns the selected source's `GetCharacterBluffIfAble()` result. That helper
returns a live unrevealed source bluff when available and otherwise the
source's real data. A corrupted Doppelganger can therefore copy either an
evil's current disguise or its real role according to setup/reveal timing.

Doppelganger is absent from `startGameActOrder`. During internal Reveal, a
clean Doppelganger's new HealthyBluff causes its first Start dispatch; the real
Doppelganger action is a no-op, but the cloned role's Start action runs before
Reveal's Init. A clean Doppelganger-as-Alchemist can consequently perform a
late cure after the ordered Alchemist pass. A corrupted Doppelganger receives
no such Start dispatch. Cross-card `registerAs` eligibility is schedule-
sensitive because another card publishes it only when that card's delayed
Reveal resumes; this slice does not establish coroutine resume order. The
dedicated
[`Doppelganger/Drunk` audit](../roles/gameplay_roles_doppelganger_drunk.md#setup-and-delayed-reveal-chronology)
closes the stronger Start boundary: Puppeteer conversion completes before any
initial delayed Reveal resumes, and the resulting real non-bluffable Puppet is
excluded from both Doppelganger branches regardless of sibling resume order.

## Confessor appearance and internal Reveal

`Confessor.OnInit` attempts to add `AppearTruthfull` with self as source and
null target. Resistance can block it. The status affects perceived truth only;
actual action dispatch ignores it. Both the real and bluff Confessor Init path
call this hook, so any displayed Confessor identity can appear truthful while
still actually lying because of Evil alignment or Corruption.

`Character.Reveal` performs setup and presentation in this exact order:

1. store the real role's `GetRegisterAsRole` result, including null;
2. when bluff is Unity-null, obtain the ordinary role bluff or trailer bluff
   and pass it to `GiveBluff`, including a null result;
3. when HealthyBluff is active, dispatch Start;
4. always dispatch Init;
5. always dispatch AfterRoundStart (`7`); and
6. present real data when bluff is null, otherwise present the bluff name,
   color, art, optional background, and refreshed view.

It does not change player-facing card state, the separate `revealed` flag,
reveal order, or reveal delegates. Role/status actions precede most view
dependency checks, so a later failure does not roll them back. `Character.Init`
starts a new delayed-Reveal coroutine without cancelling an earlier one; the
possibility of multiple resumptions after Puppeteer reinitialization follows
the native call structure, while exact Unity frame scheduling remains to be
validated dynamically.

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

The separate private `PoisonClosestNeighbours` body is not this shipped path.
It would select one random real-Villager occurrence from the circular adjacent
pair and add only Corrupted, but a native executable-xref scan finds no caller;
its only absolute reference is the ordinary IL2CPP method-registration pointer.
`Pooka.Act` instead has the sole executable edge to
`PoisonNeighboursIfAble`. The complete public asset binding, direct-xref
evidence, dormant-helper classification, duplicate lifecycle, and solver
consequences are closed in the dedicated
[`Pooka` role audit](../roles/gameplay_role_pooka.md).

### Poisoner Start adjacent corruption

The shipped public Minion asset binds exact managed `Poisoner` and occupies
ordered-Start index two, after Pooka and before Drunk. `Poisoner.Act` does work
only at `Start`. It obtains the circular previous-then-next adjacent pair, then
retains current real Villagers that lack both active Corrupted and exact
Corrupted resistance. The filters preserve order and duplicate occurrences;
they do not inspect registered or apparent type, alignment, liveness, dead
state, MessedUpByEvil, or its resistance.

An empty candidate list is a clean no-op with no RNG consumption. Otherwise
Poisoner makes exactly one max-exclusive integer random-index draw and applies
Corrupted followed by an independent MessedUpByEvil attempt, with Poisoner as
source and a null shared cure target. Corrupted or Corrupted-resistant
neighbours never enter the pool and therefore receive neither attempt. A
MessedUpByEvil-resistant selected card still receives Corrupted but blocks only
the marker.

Poisoner is an explicit all-matches lifecycle exception: duplicate assets act
synchronously from highest displayed ID to lowest and each later actor sees
the earlier status mutations. On a two-card board the sole neighbour occurs
twice in the random pool, although either index selects the same character. On
a one-card board the self-reference pair is removed by the real-Villager
filter, producing a clean no-op. Managed `Poisoner` has no alternate private
helper; its older `1 adjacent good character is Poisoned` getter is stale text,
not dormant gameplay. The complete asset binding, xref route, resistance and
small-board composition, and solver implications are closed in the dedicated
[`Poisoner` role audit](../roles/gameplay_role_poisoner.md).

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
callbacks, alignment pools, exact clue construction, acted-information shape,
and raw-status Drunk boundary are closed by the dedicated
[Plague Doctor role audit](../roles/gameplay_role_plague_doctor.md).

### Drunk self-corruption and bluff selection

The dedicated
[`Doppelganger/Drunk` audit](../roles/gameplay_roles_doppelganger_drunk.md#drunk-selection-and-registration)
adds the exact two-draw must-include priority, fallback reuse, duplicate
weighting, bounded not-in-play guarantee, and registration consequences.

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

Puppeteer -> eligible real-Villager neighbour -> Puppet Init
           -> saved former data as bluff -> four setup statuses
Puppet Start -> Puppet status action -> copied Villager Start action

clean Doppel -> HealthyBluff -> good apparent-Villager source's real data
              -> internal Reveal -> copied role Start
corrupt Doppel -> evil bluffable source -> current bluff-or-real data
                -> internal Reveal without copied role Start

Character.Act
  -> CharacterHelper.CheckLying
  -> not lying:      real Act,      then bluff-role Act
  -> lying non-Evil: real BluffAct, then bluff-role BluffAct
  -> lying Evil:     real Act,      then bluff-role BluffAct
```

`CheckLyingAppearance` and `CheckIfDisguisedAppearance` are query surfaces
separate from actual action dispatch.

## Managed-reconstruction corrections

Native review corrected seven material managed-output gaps in this slice:

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
- Puppeteer passes the victim's saved former data to `GiveBluff`; the recovered
  managed output appeared to pass the new Puppet data.
- Puppeteer conversion is mandatory when a candidate remains and removes only
  the first Saint/`SaintVillager` occurrence, rather than treating no conversion
  or all Saint occurrences as equivalent branches. Bombardier is an Outcast and
  never enters the real-Villager candidate pool.
- Clean and corrupted Doppelganger use different source predicates and return
  surfaces, and only the clean copied role receives the internal-Reveal Start
  action.

## Metadata and prototype cautions

- `Role.CheckIfCanRemoveStatus` shares a constant-true native body with 864
  other metadata identities. Evidence must preserve the requested managed
  identity and exact signature, not only the primary native symbol.
- Base `Role.GetBluffIfAble` shares its constant-null body with 322 other
  metadata identities. The exact managed identity and return signature remain
  evidence even when the program uses one canonical prototype for that RVA.
- The selected `List<Character>` and earlier `List<CharacterData>` overloads of
  `Characters.FilterRealCharacterType` emit the same C identifier in
  Il2CppDumper output. The target's explicit `prototype_name` aliases only its
  GDT FunctionDefinition; `signature` remains the exact metadata signature.
- Eight methods in the full 40-method target are intentional exact overlaps
  with earlier checked boundaries and reuse their existing evidence where
  applicable.
