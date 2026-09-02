# Gameplay bluff acquisition

Build: `f530404b0f3f_807de4a83df4`

Evidence status: **native-static** for all 23 methods in the checked boundary.

The checked target set is
[`reverse_engineering/targets/gameplay_bluff_acquisition.json`](../../targets/gameplay_bluff_acquisition.json).
It closes the common path from per-card bluff assignment through role-specific
selection, exact round-pool construction, must-include and pool draws,
script-list registration, and fresh-card creation. `Character.Init`,
`Character.Act`, internal `Character.Reveal`,
`Character.DelayReveal.MoveNext`, `Characters.ManageCharacters`, and the
Chancellor/Baron transformation remain linked lifecycle evidence rather than
duplicate members of this target.

## Target boundary

| Managed identity | RVA | Role in the boundary |
| --- | ---: | --- |
| `Character.AssignBluff` | `0x364880` | Select and store the card bluff |
| `Character.GetCharacterBluffRoleIfAble` | `0x364CD0` | Guarded real/bluff role query |
| `Character.GiveBluff` | `0x365160` | Store bluff data and clone its role |
| `Demon.GetBluffIfAble` | `0x3D6AC0` | Unique-Villager draw and registration |
| `Minion.GetBluffIfAble` | `0x3E49F0` | Duplicate-versus-unique draw |
| `Spy.GetBluffIfAble` | `0x3ED4B0` | Cached Villager bluff selection |
| `Spy.GetRegisterAsRole` | `0x3ED6A0` | Cached Villager apparent identity |
| `Mutant.GetBluffIfAble` | `0x3E4BB0` | Good/bluffable draw and Mad attempt |
| `Characters.GetRandomUniqueVillagerBluff` | `0x36C940` | Villager must-include or unique-pool draw |
| `Characters.GetRandomUniqueBluff` | `0x36C810` | General must-include or unique-pool draw |
| `Characters.GetRandomDuplicateBluff` | `0x36C7A0` | Duplicate-pool draw |
| `Characters.PickRoundBluffs` | `0x36D3A0` | Build the round unique pool |
| `Characters.PickRoundDuplicates` | `0x36D720` | Build the round duplicate pool |
| `Characters.<>c__DisplayClass22_0.<PickRoundBluffs>b__0` | `0x377170` | Remove starting data already contained in the script |
| `Characters.GetARandomBluffMustIncludeOfType` | `0x36BE50` | Filtered must-include draw/removal |
| `Characters.GetARandomBluffMustInclude` | `0x36BF80` | General must-include draw/removal |
| `Characters.FilterRealCharacterType(CharacterData)` | `0x36B9C0` | Exact real-type filter |
| `Characters.FilterCharacterType(CharacterData)` | `0x36AAB0` | Exact data-type filter |
| `Characters.FilterAlignmentCharacters(CharacterData)` | `0x369EB0` | Exact data-alignment filter |
| `Characters.FilterBluffableCharacters(CharacterData)` | `0x36A550` | Bluffable-data filter |
| `Gameplay.AddScriptCharacterIfAble` | `0x37B370` | Faction-list registration |
| `Helpers.RollDice` | `0x396840` | Inclusive one-based die roll |
| `CharactersPool.CreateAndGetCharacters` | `0x3698F0` | Destroy old cards and instantiate a fresh board |

## Assignment, storage, and action surface

In ordinary play, `Character.AssignBluff` asks the real
`dataRef.role` for a bluff and passes the result to `GiveBluff`. Trailer mode
instead obtains the bluff from the trailer record for the card ID. Missing
required card, role, trailer-singleton, or trailer-record references take the
runtime null-reference path. Selector failures propagate. A null selector
result is still passed to `GiveBluff`.

There is no direct native call reference to the retained `AssignBluff` method
in this build. Internal `Character.Reveal` contains the same operation after it
resolves register-as data, and only performs it while the current `bluff` is
Unity-null. This is the normal bluff-acquisition caller and timing.

`GiveBluff` always writes the supplied `CharacterData` to `bluff`. When that
Unity object is live, it clones the data's `role` into `bluffRole`. A null or
destroyed data object does not clear `bluffRole`. This makes the storage method
non-transactional: the data field can become Unity-null while an earlier copied
role remains.

`GetCharacterBluffRoleIfAble` returns `bluffRole` only while the card is not
Dead or Revealed, its separate `revealed` flag is false, and `bluff` is
Unity-live. Otherwise it returns the real `role`. It does not validate the
returned `bluffRole`, and no direct native caller was found.

The guarded query is not the action-dispatch contract. `Character.Act` reads
`bluffRole` directly after dispatching the real role. A truthful card dispatches
the copied role normally; a lying card dispatches it in bluff mode. The existing
[execution-resolution audit](gameplay_execution_resolution.md#action-and-lying-dispatch)
documents the full precedence and call order.

## Role selectors and script mutations

`Demon.GetBluffIfAble` requests a unique Villager bluff, registers the returned
data in Gameplay by its actual character type, and returns it. A missing
Characters or Gameplay singleton, a null selection, or an invalid destination
list throws rather than producing a no-bluff result.

`Minion.GetBluffIfAble` rolls `Helpers.RollDice(10)`, whose exact result range
is one through ten. Results one through four return a duplicate-pool draw
without script registration. Results five through ten request a unique bluff,
register it, and return it. The branch probabilities are therefore exactly 40%
duplicate and 60% unique. The duplicate branch returns its helper result
without a post-selection null check.

Spy maintains one `chData` cache:

- `GetRegisterAsRole` returns a live cached value or uniformly samples a
  script `CharacterData` whose exact type is Villager, caches it, and returns
  it.
- `GetBluffIfAble` first returns `charRef.registerAs` when the card already has
  a live bluff, without separately validating that register-as result.
  Otherwise it uses the same live cache or the same Villager sampling path.

Internal Reveal invokes register-as selection first, then bluff selection when
the card has no live bluff. Ordinary setup consequently uses the same cached
Villager data for Spy's apparent identity and copied bluff role. Neither Spy
method adds that data to a script list.

`Mutant.GetBluffIfAble` starts with the combined script-character list, filters
it to exact Good alignment and `bluffable == true`, and calls
`AddStatus(Mad, source=charRef, target=null)` before validating and drawing from
the final list. Resistance can block the status under the generic status rules;
an accepted status is not rolled back if the later draw fails. Mutant does not
register the selected bluff.

`Gameplay.AddScriptCharacterIfAble` maps Villager, Outcast, Minion, and Demon
data to their four corresponding script lists. It suppresses an already
contained object, adds otherwise, throws for a null selected destination list,
and returns without mutation for an unsupported type.

## Bluff pools and filter semantics

The unique selectors give the must-include pool priority:

- `GetRandomUniqueVillagerBluff` probes for a must-include Villager. If one is
  live, it performs another random selection and removes/returns that result.
  Otherwise it copies the unique pool, filters exact real type Villager, and
  draws from the copy.
- `GetRandomUniqueBluff` similarly probes the general must-include list, then
  calls its removal helper for the returned draw. If the probe is not live, it
  draws directly from the unique pool.
- `GetRandomDuplicateBluff` draws directly from the duplicate pool and never
  removes the result.

The probe and remove operations are separate random calls. With several
eligible must-include entries, the initially probed object need not be the one
ultimately returned and removed. Fallback unique-pool and duplicate-pool draws
also do not remove their selected entry.

`GetARandomBluffMustInclude` returns null for an empty, non-null must-include
list. Its typed counterpart copies the source, filters it, and likewise returns
null for an empty filtered list. Despite accepting a `type` parameter, the
current native typed helper ignores it and always filters exact real type
Villager; its only current caller also asks for Villager. Optional removal
deletes one matching occurrence from the original must-include list.

All four `CharacterData` filters allocate a new result list, enumerate the
input in order, preserve duplicate entries, and select on the exact field named
by the method. A null input or null element reaches the null-reference path.
This means random selection remains multiplicity-weighted.

Empty duplicate, unique fallback, Spy, or Mutant candidate lists are not a
supported fallback. They reach a zero-width random index followed by indexed
list access and fail. Null lists and missing singletons likewise throw. The two
must-include helpers are the exception: an empty non-null eligible list returns
null cleanly.

## Round unique-pool construction

`Characters.ManageCharacters` calls `PickRoundBluffs` once, immediately after
updating board positions and before `PickRoundDuplicates` or any per-card
`Character.Init`. The builder obtains two new occurrence lists:

```text
allStarting = starting Demons ++ Outcasts ++ Minions ++ Villagers
script      = current Townsfolk ++ Outcasts ++ Minions ++ Demons
```

The first list comes from `Gameplay.GetAscensionAllStartingCharacters`; the
second comes from `Gameplay.GetScriptCharacters`. The captured predicate at
`0x377170` is exactly `script.Contains(cd)`. `RemoveAll` therefore removes
every starting occurrence whose data object is already contained in the
script. The method then clears `UniquePool`, filters the remaining occurrences
by `CharacterData.bluffable`, and forms exact-real-type Villager and Outcast
lists.

It uniformly samples at most four Villager occurrences without replacement
from the local list, appending them to `UniquePool` in draw order. It then
uniformly samples one Outcast occurrence when any are available. Local
`Remove(item)` calls reduce one matching occurrence after each draw; all
probability is by list multiplicity rather than distinct public role name.

The normal selected count is:

```text
min(4, eligible not-in-script Villager occurrences)
+ indicator(eligible not-in-script Outcast occurrences > 0)
```

When that value is zero or one, the method appends one fallback occurrence. It
calls `Gameplay.GetAllAscensionCharacters`, whose current native body
concatenates Townsfolk, Outcasts, Minions, and Townsfolk a second time while
omitting Demons. It then filters bluffable, exact starting-alignment Good, and
exact real-type Villager before drawing. There is no second script subtraction,
no identity deduplication, and no removal from `UniquePool`. The fallback can
therefore re-admit an in-script role, duplicate an already selected identity,
and gives ordinary Townsfolk occurrences their duplicated source weight. An
empty fallback support fails at random/index access.

Despite its name, `UniquePool` is not an identity set. Duplicate input
occurrences and the fallback can produce repeated exact data objects. The
builder neither reads nor mutates `BluffMustInclude`, so its code does not
forbid the same data from being represented in both collections.

## Round duplicate-pool construction

`PickRoundDuplicates` runs immediately after the unique builder. It obtains a
fresh combined script list in the faction order above, clears `DuplicatesPool`,
filters bluffable occurrences, and forms exact-real-type Villager and Outcast
lists. It also calls `FilterAlignmentCharacters(..., Good)` but discards the
returned list. Starting alignment has no effect on the actual candidate pool;
a bluffable starting-Evil data object with real type Villager or Outcast is
eligible.

The Villager sampler is an unguarded `do` loop. It uniformly appends one local
Villager occurrence, removes one matching occurrence, and continues until the
local list empties or four draws have completed. It then appends one uniformly
selected Outcast occurrence when available. The completed pool preserves draw
order and multiplicity.

The native body contains a final `DuplicatesPool.Count == 0` branch that would
draw one bluffable, Good occurrence of any real type from all starting data.
That branch is unreachable after any successful execution:

- one or more eligible Villagers forces the first loop iteration to append an
  entry, so the count cannot remain zero; and
- zero eligible Villagers reaches `Range(0, 0)` and indexed access before the
  count test, so setup fails before the apparent fallback.

Reconstruction must therefore require at least one bluffable real-Villager
script occurrence. It must not use the emitted Good/all-type branch as an
empty-Villager recovery. Once built, `DuplicatesPool` is sampled with
replacement and never consumed; `UniquePool` is likewise reusable after the
must-include list is exhausted. Later unique registration mutates the script
faction lists but does not rebuild either round pool.

## Round recreation and stale-role reachability

`Character.Init` clears `bluff` but not `bluffRole`, so reuse of an already
bluffed Character would be hazardous. A failed later `GiveBluff` could preserve
the copied role, and direct `Character.Act` dispatch could execute it even
though the guarded query would return the real role.

The shipped board lifecycle prevents that stale value across ordinary rounds.
`CharactersPool.CreateAndGetCharacters` destroys every old child GameObject,
instantiates a new Character prefab under each board position, stores a new
array, and returns that array. `Characters.ManageCharacters` publishes and
dispatches only these newly created Character references; it does not recycle
the old runtime components.

The normal Chancellor/Baron path is also ordered before bluff acquisition.
Every fresh card is initialized first. Each `DelayReveal` coroutine clones the
current real role before yielding `0.3` seconds. Chancellor is the first ordered
Start role, so all of its `Init(..., -100)` transformations finish while the
fresh `bluffRole` fields are still null and before any delayed Reveal assigns a
bluff. Multiple delayed-Reveal continuations can later exist for a reinitialized
card, but the first successful bluff assignment is then observed by the later
Reveal calls.

A latent nonstandard path remains: invoking Baron Start after a card already
has a copied bluff role would preserve it through `Init`. If the currently
executing card is reinitialized, the outer `Character.Act(Start)` could read
and dispatch that role as soon as Baron returns. A later null/destroyed bluff
selection could preserve it again. This path is present in the native control
flow but excluded by the shipped Start ordering. Puppeteer's other gameplay
reinitialization immediately follows `Init` with a live `GiveBluff(dataRef)`,
overwriting the field.

The linked lifecycle evidence is in the
[gameplay lifecycle audit](gameplay_lifecycle.md#character-construction-and-publication),
[per-card initialization section](gameplay_lifecycle.md#per-card-initialization-and-internal-reveal),
and [Chancellor/Baron section](gameplay_lifecycle.md#chancellorbaron-start-replacement-and-relocation).

## Global delayed-Reveal ordering

Ordinary `Character.Init` starts one `DelayReveal` continuation per freshly
created board card in board-list order. Each continuation clones the
then-current `dataRef.role` before yielding the same `0.3f` wait. All ordinary
initial continuations therefore exist before the ordered Start pass. Writers
such as Chancellor, Twin Minion, Puppeteer, and Shaman can synchronously add
more continuations without cancelling the earlier ones. Twin's two
`InitWithNoReset` calls start the selected neighbour's continuation before the
original Twin body's continuation; a self-swap adds both to the same card.

The resume-side iterator retains only its Character reference. It calls
`Character.Reveal` against current shared card state rather than a captured
data or role identity. Reveal recomputes register-as data, acquires and stores a
bluff only while the raw `bluff` is Unity-null, optionally dispatches Start for
`HealthyBluff`, and then always dispatches Init and AfterRoundStart. Thus the
first continuation that successfully assigns a live bluff fixes that card's
selection for later siblings, while every sibling still repeats the later
hooks. A null or destroyed result remains eligible for retry.

Native static analysis fixes continuation creation order but not Unity's
resume order after equal waits. Unity's coroutine contract does not guarantee
that coroutines completing in one frame finish in their start order. A solver
must not infer global Reveal order from board order, Twin endpoint order, or
coroutine creation order.

This matters directly when a runtime-Good Twin recipient reaches Reveal with
current Minion data. Its first successful selection rolls the exact 40/60
Minion branch, but earlier resumed Minion, Demon, or Drunk cards can consume
must-include occurrences and advance the shared RNG stream. Later sibling
continuations reuse its live bluff. If another ordered writer changed the
recipient away from Twin before that first resume, the dynamic Reveal instead
uses the replacement role. If the recipient preserved `HealthyBluff`, Reveal
dispatches the current Twin Start again and can cause another board swap and
additional continuations.

Exact replay therefore requires either the realized global continuation order
or a complete enumeration of relevant interleavings. It must also retain the
ordered/multiplicity-bearing `UniquePool`, `DuplicatesPool`, and
`BluffMustInclude` state at each event; current script-list mutations; every
physical card's current data, runtime alignment, statuses, resistance, raw
bluff, copied bluff role, and register-as data; and all intervening random
outcomes. A bounded reconstruction can instead require a known pool snapshot,
a recipient that remains runtime-Good/current-Twin with a null bluff and no
`HealthyBluff`, and proof that no unresolved earlier continuation can consume
or replace its selection state. Any missing provenance must fall back
atomically rather than assume an ordering.

Confessor is a concrete appearance-sensitive example. It can be selected from
the duplicate pool when already scripted, from the unique pool when present in
that realized support, or from the remaining must-include list. `GiveBluff`
clones its role, and both truthful `Confessor.Act(Init)` and lying
`Confessor.BluffAct(Init)` attempt `AppearTruthfull` on the physical recipient.
Resistance can reject the status; otherwise repeated sibling Reveals repeat
the unique-add attempt and reset the shared status target pointer to null.
This changes appearance queries, not actual `Character.Act` truth routing, and
an existing `AppearLying` status retains appearance precedence.

### Bounded solver integration

The Rust solver's first exact consumer is deliberately narrower than a global
coroutine replay. An optional offline-only
`twin_recipient_bluff_context` uses rule marker
`twin_recipient_bluff_native_v1` and records the moved recipient, a global
acquisition ordinal, the occurrence-preserving duplicate and unique pools, and
the must-include list as it existed at that recipient's first successful
Minion bluff acquisition. Fresh live sessions never populate this hidden
field, and public `reveal_order` is never interpreted as its substitute.

Within the exact Twin-then-Shaman slice, the context is accepted only for a
distinct runtime-Good recipient that remains current Twin, has no modeled
`HealthyBluff`, and exposes one exact current Scout, Witness, or Confessor
card. The enumerator assigns every duplicate occurrence mass `2/(5D)` and
every active unique occurrence mass `3/(5U)`; a nonempty must-include snapshot
replaces the ordinary unique source. It reduces those fractions to equal
integer tickets without deduplicating repeated roles. Each surviving scenario
retains a tagged `twin_recipient_bluff_trace` with its pool source and
occurrence index, so later validation cannot replace the selected role with an
existential raw-bluff guess.

Missing context preserves archived behavior. Malformed, over-cap, unsupported,
or structurally incompatible context causes wholesale fallback rather than
partially mixing weighted exact worlds with the legacy approximation. The
current live regression corpus has no acquisition-event snapshots, so this
checkpoint changes no archived case result; it establishes the guarded input
boundary for future post-mortem fixtures and eventual global interleaving work.

## Typed import, overlaps, and shared identity

Two target memberships intentionally overlap earlier checked boundaries:

- `Character.GiveBluff` reuses the exact status/corruption/truth identity.
- `Characters.FilterRealCharacterType(CharacterData)` reuses the exact
  roster-helper identity.

The CharacterData overloads of `FilterCharacterType` and
`FilterAlignmentCharacters` share their emitted Il2CppDumper C identifiers
with existing `List<Character>` overloads. Their target entries use explicit
`prototype_name` aliases for Ghidra while retaining the unmodified metadata
signatures and RVAs for validation.

`Helpers.RollDice` and `Calculator.RollDice` are two managed method identities
bound to the same native RVA. Only `Helpers.RollDice` is a member of this
23-method target, but coverage records both exact managed identities against
the shared body. The typed program applies the selected Helpers signature; it
does not collapse the Calculator method into the target membership count.

The expanded boundary exports all 23 functions from both the baseline and typed
projects with zero failures. Its three new memberships are three new exact
FunctionDefinitions at three distinct native RVAs. Across all 41 checked target
sets, deterministic discovery now yields 874 memberships, 541 distinct selected
FunctionDefinitions, and 443 unique native RVAs. The 333 repeated exact
identities explain the membership/definition difference; folded or shared
bodies explain the remaining 98-definition RVA gap. The Calculator alias is
outside the target union, so it adds a coverage classification but no target
membership, FunctionDefinition, or selected RVA.

The rebuilt GDT contains 151,678 datatypes and all 541 selected definitions.
The six-batch no-analysis refresh imported six additional reachable datatypes,
then its separate read-only pass validated all 874 memberships and 2,561
membership-level parameter-storage locations with zero program mutations. The
checked quality report covers all 23 baseline/typed pairs: unresolved type
tokens fall from 131 to 65, raw field-offset accesses from 187 to 70, and
placeholder parameter tokens from 155 to zero. It records no decompiler-error
markers and passes its directional regression policy.

## Reconstruction implications

- Normal solver state does not need a cross-round or shipped-Chancellor stale
  bluff-role transition.
- Minion acquisition must retain its exact 40/60 split and registration
  difference.
- Demon and Minion-unique draws can expand a script faction list; Minion-
  duplicate, Spy, and Mutant draws do not.
- Spy's apparent identity and bluff share one cached Villager selection.
- Mutant attempts Mad before a multiplicity-weighted Good/bluffable draw.
- Unique-pool construction samples at most four off-script Villager
  occurrences and one off-script Outcast, then uses its non-subtracting
  all-ascension fallback only when the resulting count is at most one.
- Duplicate-pool construction requires at least one bluffable real-Villager
  script occurrence; its emitted zero-count fallback is not a reachable
  recovery after the unguarded first draw.
- Equal-delay Reveal continuations require explicit order provenance or full
  interleaving enumeration whenever their pool consumption or current-role
  mutations can affect an observation.
- Invalid or empty pool configurations generally fail rather than yielding a
  playable card with no bluff.
