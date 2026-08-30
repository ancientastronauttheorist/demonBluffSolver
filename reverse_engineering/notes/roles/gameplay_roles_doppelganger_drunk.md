# Gameplay roles: Doppelganger and Drunk

Build: `f530404b0f3f_807de4a83df4`

Evidence status: **native-static** for all methods declared by the shipped
managed `Doppleganger` and `Drunk` types and the exact setup, delayed-reveal,
filtering, pool, registration, status, and execution helpers needed to close
their disguise lifecycle. Serialized asset evidence binds both public cards
to those managed types. Native bodies and decompiler output remain outside the
repository.

The checked target set is
[`reverse_engineering/targets/gameplay_roles_doppelganger_drunk.json`](../../targets/gameplay_roles_doppelganger_drunk.json).
Its read-only baseline and typed exports each complete at 39/39 functions with
no failures. The body-free
[`quality report`](../../reports/f530404b0f3f_807de4a83df4_typed_quality_gameplay_roles_doppelganger_drunk.json)
passes its regression check: unresolved-type tokens fall from 216 to 72, raw
field-offset accesses from 266 to 54, placeholder parameter tokens from 291 to
zero, and indirect-call patterns from nine to zero. The one error marker is
unchanged and the typed export gains one nonfatal warning marker. The report
contains no private decompilation bodies or artifact paths.

## Public asset bindings and authored contracts

The public Doppelganger `CharacterData` is `sharedassets0.assets` path ID
`21604`. It is named `Doppelganger`, has character ID
`Doppleganger_52694042`, and binds its SerializeReference role to managed
`Doppleganger` at TypeDefIndex `5908`. The misspelling is present in both the
serialized ID and managed type. Its raw object SHA-256 is
`720335AB6E1B23938405822FF49450834B9B38CFE7906A26BC598F87AE3EBADB`.
It is a Good Outcast (`characterType == 20`, `startingAlignment == 10`), has
Once ability usage, is not bluffable or usually disguised, has no picker, and
authors no starting status.

Its exact public description is:

```text
<b>Game Start:</b>
I Disguise as a Good Villager currently in play.
```

The managed description is the older bracketed string
`[I am a Good Villager role currently in play]`. The serialized text is the
current player-facing contract.

The public Drunk `CharacterData` is path ID `21605`. It is named `Drunk`, has
character ID `Drunk_15369527`, and binds managed `Drunk` at TypeDefIndex
`5904`. Its raw object SHA-256 is
`82A407F22608D01F46AAB9C5E1A24D6F173B9E828CC7D99775E3586F10D3AB6B`.
It is also a Good Outcast, has Once ability usage, is not bluffable or usually
disguised, and has no picker or authored starting status. Its exact public
description is:

```text
I Disguise as a random not in play Villager.
I am Corrupted and I Lie.
I can not be Cured.
```

Its exact hint is `You receive 2 damage instead of 5 when you Execute me.`.
The managed description getter returns the empty string, so the serialized
description is again authoritative.

## Audited boundary

| Group | Methods | Observable purpose |
| --- | ---: | --- |
| `Drunk` | 8 | Complete role, status veto, two-damage override, and bluff selector |
| `Doppleganger` | 9 | Complete role, clean/corrupted selector, and copied protection |
| `Character` and delayed reveal | 8 | Setup timing, register-as fallback, bluff storage/query, and dispatch |
| `CharacterStatuses` | 2 | Mutation and raw status membership |
| `Characters` | 8 | Round setup, unique/must-include draws, and exact source filters |
| `Gameplay` | 2 | Current-card publication and script-role registration |
| Base `Role` | 2 | Default status-removal permission and wrong-execution damage |

The 39 memberships select 37 distinct RVAs. `Doppleganger.Act` and
`Doppleganger.BluffAct` are separate managed identities folded to the shared
no-op body at `0x33ED50`; both role constructors share the base-only body at
`0x3CFFF0`. Seven selected identities also use bodies shared with methods
outside this target, so the target preserves metadata identity rather than
mistaking native-body folding for semantic equivalence.

## Setup and delayed-reveal chronology

`Characters.ManageCharacters` selects the round bluff and duplicate pools
before initializing cards. It then initializes every physical card, publishes
a shallow copy as `Gameplay.CurrentCharacters`, dispatches Init to every card,
and runs the serialized Start order synchronously. Each initial
`Character.Init` clears bluff, register-as, status, runtime, acted, and Start-
latch state, assigns the new real data, and starts `DelayReveal`. The iterator
clones the current real role before its first `0.3`-second yield.

The relevant serialized Start order is:

```text
Chancellor -> Pooka -> Poisoner -> Drunk -> Witch -> Marionette
-> Puppeteer -> Baa -> Plague Doctor -> Shaman -> Alchemist
-> Bounty Hunter -> Puppet -> Rambler -> Lilis
```

Doppelganger is absent. Consequently:

1. the first matching public Drunk attempts Corrupted before Puppeteer runs;
2. Puppeteer's neighbour scan, chosen-card reinitialization, saved-old-role
   bluff assignment, and four status attempts all complete in the ordered
   Start call stack; and
3. no initial delayed Reveal can resume until that synchronous setup stack has
   returned.

Puppeteer's `Character.Init(puppetData, -100)` starts a second delayed iterator
on the converted physical card. It clears `registerAs`, makes Puppet the real
`dataRef`, and the second iterator clones Puppet before yielding. Puppeteer
then stores the former Villager only as `bluff`/`bluffRole` and attempts
HealthyBluff, BrokenAbility, AlteredCharacter, and MessedUpByEvil. The earlier
iterator is not cancelled, but every later `Character.Reveal` reads the card's
current real data. Neither iterator restores the erased Villager.

On resumption, internal `Character.Reveal` performs this order:

1. store the real role's register-as result, including null;
2. if bluff data is null, call the real role's bluff selector and store its
   result;
3. if HealthyBluff is present, dispatch Start;
4. dispatch Init;
5. dispatch AfterRoundStart; and
6. present the selected real or bluff identity.

Thus Drunk and Doppelganger acquire their disguises only in delayed Reveal,
after Puppeteer conversion and all other ordered Start mutations. The exact
resume order among sibling delayed iterators is still a Unity scheduling
surface. It can affect another source's live `registerAs`, but it cannot make a
Puppeteer-erased card eligible: public Puppet is a non-bluffable Evil Minion,
and both Doppelganger branches reject it on real `dataRef.bluffable` before
consulting apparent identity.

## Doppelganger passive surface and dispatch

`Doppleganger.GetInfo` and `GetBluffInfo` each return a fresh
`ActedInfo("", null)`. `ConjourInfo` returns the empty string. Both `Act` and
`BluffAct` are no-ops for every trigger; copied-role behavior comes from
`Character.Act` dispatching the cloned `bluffRole`, not from either
Doppelganger action.

During ordinary internal Reveal the real selector runs once while the bluff
field is null. A clean actor takes the clean branch; active Corrupted selects
the corrupted branch. No branch retries, removes a selected board source, or
deduplicates by role identity.

## Clean Doppelganger source pool

The clean branch performs these operations in order:

1. attempt `HealthyBluff` (`30`) on self;
2. copy `Gameplay.CurrentCharacters` in physical order;
3. retain sources whose **real** `dataRef.bluffable` is true;
4. retain apparent Villagers, where a Unity-live `registerAs` takes precedence
   over real `dataRef`;
5. retain apparent/runtime Good, where live
   `registerAs.startingAlignment` takes precedence over the physical runtime
   alignment; and
6. uniformly choose one list occurrence and return that source's **real**
   `dataRef`.

The status attempt is not transactional. Resistance can reject HealthyBluff
without preventing the later draw, and a later missing dependency or empty
pool does not undo an accepted status.

There is no explicit self, Hidden, Alive, Revealed, Dead, killed-by-demon,
public-revealed, status, or duplicate filter. The ordinary public actor excludes
itself because its real Doppelganger asset is not bluffable, not because of a
self rule. Hidden and dead sources are equally eligible when the three stated
predicates pass. Duplicate physical entries and malformed repeated list
entries survive and weight the uniform index draw. Multiple eligible cards
with the same role therefore make that role proportionally more likely.

## Corrupted Doppelganger source pool

The Corrupted branch does **not** attempt HealthyBluff. It:

1. copies current physical cards;
2. retains real-bluffable sources;
3. retains apparent/runtime Evil by the same register-as-first alignment
   helper; and
4. uniformly chooses one occurrence and returns that source's
   `GetCharacterBluffIfAble()` identity.

There is no Villager-type filter. Selection itself still ignores card state,
but the returned identity is state-sensitive: a live bluff is returned only
while the source state is neither Dead nor Revealed, its separate `revealed`
flag is false, and its bluff object is live. Otherwise the selected source's
real data is returned. A hidden Evil will normally contribute its disguise;
a dead, fully revealed, or forced-real-revealed Evil contributes its real role.
The random source distribution remains multiplicity-weighted and there is no
reroll when several sources collapse to the same returned identity.

Both branches fail on an empty final list through the zero-width random/index
path. Null cards, null data, null lists, or missing required singletons follow
their native failure paths rather than producing a clean no-disguise result.

## Erased roles and source examples

| Physical source at Doppelganger Reveal | Clean branch | Corrupted branch |
| --- | --- | --- |
| ordinary bluffable Good Villager | Eligible; returns its real Villager | Excluded unless live register-as/runtime alignment is Evil |
| bluffable Good Villager, Hidden or Dead | State does not matter; eligible | Alignment-dependent; state changes returned real/bluff identity only |
| ordinary Evil with a Villager bluff | Excluded by Good alignment | Eligible if its real data is bluffable; normally returns live bluff while Hidden |
| Puppeteer-converted former Villager | Excluded by real Puppet `bluffable == false` | Excluded by the same first filter |
| public Drunk or another public Doppelganger | Excluded by real non-bluffable asset | Excluded by real non-bluffable asset |
| duplicate eligible physical cards | Every occurrence remains | Every occurrence remains |

Puppeteer's saved old Villager exists only on that Puppet's display surface; it
is not still a Good Villager source. If a separate physical copy of the same
Villager asset remains, that other copy is independently eligible. Drunk's
asset-pool selector likewise does not recover an erased physical role, though
it may select the same Villager `CharacterData` for unrelated pool reasons.

## Doppelganger killability

`Doppleganger.CheckIfCanBeKilled` returns true immediately when HealthyBluff is
absent. With HealthyBluff present it delegates to the current cloned bluff
role's `CheckIfCanBeKilled(charRef)`. Missing required character, status,
bluff-role, or role dependencies follow the native failure path rather than a
protected fallback.

This makes a normal clean Doppelganger displaying Knight inherit Knight
protection, while the normal Corrupted branch is immediately killable because
it never installs HealthyBluff. The selector and protection rules are distinct:
a resisted HealthyBluff can still leave a selected disguise but no delegated
protection.

## Drunk status, information, and damage

`Drunk.GetInfo` and `GetBluffInfo` each return a fresh
`ActedInfo("", null)`. `Drunk.Act` is a no-op except at Start (`5`), where it
attempts Corrupted (`10`) with the Drunk as both source and shared cure target.
An exact Corrupted resistance blocks insertion. Repeating the accepted status
does not duplicate the status entry but restores the shared cure target to the
Drunk.

`CheckIfCanRemoveStatus` returns false only for Corrupted and true for every
other status value. `GetDamageToYou` returns exactly `2`; the base role value is
`5`. Those overrides belong to the real Drunk even when the displayed bluff is
Knight or another role.

Only the first matching ordinary Drunk data occurrence receives the serialized
ordered Start dispatch. Every physical Drunk nevertheless repeats the same
Corrupted attempt inside its own `GetBluffIfAble` during delayed Reveal. Under
normal setup all instances therefore become Corrupted before public card
flips; duplicates can differ only during the interval before their individual
Reveal continuations or when resistance blocks insertion.

## Drunk selection and registration

`Drunk.GetBluffIfAble` has this mutation order:

1. require the actor and status container;
2. attempt self Corrupted again;
3. call `Characters.GetRandomUniqueVillagerBluff`;
4. require a live selected asset and Gameplay singleton;
5. call `Gameplay.AddScriptCharacterIfAble(selected.type, selected)`; and
6. return the selected `CharacterData`.

The status mutation precedes selection and registration and is never rolled
back by a later failure.

The unique-Villager helper gives eligible `BluffMustInclude` entries absolute
priority. It first calls the typed must-include helper with `remove == false`.
If that probe returns a live Villager, it calls the helper a second time with
`remove == true` and returns/removes the second draw. The two calls are
independent uniform random selections, so the probed object need not be the
returned object. The helper's nominal type parameter is ignored in this build;
it always filters real Villager data. An empty eligible must-include copy
returns null cleanly.

When the probe is null, the selector copies `UniquePool`, filters real
Villagers, and uniformly returns one occurrence without removing it. An empty
fallback list fails through random/index access. All filters preserve order and
duplicate occurrences, so both branches are weighted by list multiplicity,
not uniform over distinct role names.

With `k` eligible must-include occurrences, the final selected occurrence has
probability `1/k`; the first call consumes an extra independent RNG draw but
does not bias the second. One matching occurrence is then removed. Multiple
Drunks can drain the must-include Villagers in reveal order. Once none remain,
fallback draws do not consume `UniquePool`, so later Drunks can reuse the same
asset. `AddScriptCharacterIfAble` routes the selected Villager into the script
Villager list and suppresses an already-contained exact object; it does not
alter these draw probabilities or retry after a duplicate registration.

## The authored “not in play” promise

The round-pool builder normally enforces the description upstream rather than
inside Drunk's selector. It builds candidate data, removes the current script
assets, filters bluffable roles, and forms `UniquePool` from up to four
Villagers plus one Outcast. An always-in-deck role routed to
`BluffMustInclude` is placed there instead of the ordinary roster. Under that
ordinary construction, both Drunk sources are intended to be absent from the
selected script.

The guarantee is not an unconditional selector postcondition. Neither
`GetRandomUniqueVillagerBluff` nor its must-include helper rechecks current
board presence. When the normal unique pool has one or fewer entries, the
builder's all-ascension Good/bluffable-Villager fallback does not repeat the
not-in-play subtraction. Duplicated or externally modified pool entries are
also preserved. Solver reconstruction should therefore use exact prepared
pool support when available and treat “not in play” as the normal construction
invariant, not add a second rejection or reroll that native code does not have.

## Register-as, script counts, and HUD implications

Neither managed role overrides the base register-as selector. During internal
Reveal, `Character.Reveal` therefore stores null for both cards. Later
`Character.GetRegisterAs` falls back to real `dataRef`, and
`GetRegisterAlignment` falls back to the physical runtime alignment. A normal
public Doppelganger and Drunk consequently register as Good Outcasts for
native filters even while presenting Villager disguises.

`GiveBluff` writes only `bluff` and a cloned `bluffRole`; it does not change
real data, runtime alignment, or `registerAs`. Drunk's later
`AddScriptCharacterIfAble` mutates the role-asset faction list, not the Drunk
card and not the already-cloned `Gameplay.CurrentScript` count record.
Doppelganger performs no script-list registration at all.

Visible header handling is therefore a separate pre-card roster/count surface.
Observed rules such as Doppelganger contributing to the Villager allowance or
some Drunk setups appearing in the Villager header cannot be implemented as a
runtime register-as change, nor can Drunk's delayed registration retroactively
change that header. This boundary proves the separation; exact upstream
ascension/header composition remains part of the deck-construction audit.

## Typed-union accounting

This target has 26 exact identity overlaps with the previous 19 target sets
and introduces 13 new managed definitions. Four of those definitions use RVAs
already selected by the union: the two Doppelganger no-op identities at
`0x33ED50` and both constructors at `0x3CFFF0`. The exact delta is therefore
nine new RVAs, not eleven.

The deterministic 20-set union contains 454 memberships, 307 distinct selected
FunctionDefinitions, and 287 unique native RVAs. All exact managed identities
remain represented in the GDT and coverage ledger despite folded bodies.

## Reconstruction implications

- Model clean and Corrupted Doppelganger as distinct physical-source draws;
  never deduplicate eligible sources by role name.
- Do not exclude hidden, dead, or self by a generic state rule. Public self is
  excluded only by real non-bluffability, and corrupted returned identity has
  its own state-sensitive bluff guard.
- A Puppeteer-erased Villager occurrence is not a Doppelganger source. Only a
  separate surviving occurrence of that identity can remain eligible.
- Drunk must-include selection consumes two RNG draws and removes the second
  selected occurrence; unique fallback consumes one draw and removes nothing.
- Preserve status-before-selection failure effects, two-damage execution, the
  Corrupted cure veto, and registration deduplication.
- Do not derive header counts from displayed bluff or `registerAs`; both cards
  retain real Good-Outcast registration in this lifecycle.
