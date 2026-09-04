# Gameplay bluff acquisition

Build: `f530404b0f3f_807de4a83df4`

Evidence status: **native-static** for all 31 methods in the checked boundary,
plus the version-matched Unity 2022.3 coroutine API contract where explicitly
identified below.

The checked target set is
[`reverse_engineering/targets/gameplay_bluff_acquisition.json`](../../targets/gameplay_bluff_acquisition.json).
It closes the common path from fresh-card initialization and delayed-Reveal
registration through first-bluff assignment, role-specific selection, exact
round-pool construction, must-include and pool draws, script-list registration,
and the GameAssembly-to-Unity coroutine/RNG handoff. `Character.Act`, concrete
Init/AfterRoundStart role hooks, the unused emitted `Character.DelayReveal`
wrapper, UnityPlayer's scheduler implementation, and the Chancellor/Baron
transformation remain linked or explicitly external evidence rather than
members of this target.

## Target boundary

| Managed identity | RVA | Role in the boundary |
| --- | ---: | --- |
| `Character.AssignBluff` | `0x364880` | Select and store the card bluff |
| `Character.GetCharacterBluffRoleIfAble` | `0x364CD0` | Guarded real/bluff role query |
| `Character.GiveBluff` | `0x365160` | Store bluff data and clone its role |
| `Character.Init` | `0x365A20` | Fresh/replacement initialization and continuation registration |
| `Character.InitWithNoReset` | `0x365720` | Status-preserving replacement and continuation registration |
| `Character.<DelayReveal>d__84.MoveNext` | `0x3756B0` | Shared role clone, wait, and resume dispatch |
| `Character.Reveal` | `0x368410` | Atomic register-as and first-bluff acquisition event |
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
| `Characters.ManageCharacters` | `0x36CE30` | Deterministic creation and ordered-Start registration order |
| `Characters.GetARandomBluffMustIncludeOfType` | `0x36BE50` | Filtered must-include draw/removal |
| `Characters.GetARandomBluffMustInclude` | `0x36BF80` | General must-include draw/removal |
| `Characters.FilterRealCharacterType(CharacterData)` | `0x36B9C0` | Exact real-type filter |
| `Characters.FilterCharacterType(CharacterData)` | `0x36AAB0` | Exact data-type filter |
| `Characters.FilterAlignmentCharacters(CharacterData)` | `0x369EB0` | Exact data-alignment filter |
| `Characters.FilterBluffableCharacters(CharacterData)` | `0x36A550` | Bluffable-data filter |
| `Gameplay.AddScriptCharacterIfAble` | `0x37B370` | Faction-list registration |
| `Helpers.RollDice` | `0x396840` | Inclusive one-based die roll |
| `UnityEngine.Random.Range(int, int)` | `0x1C86600` | Shared engine RNG internal-call boundary |
| `UnityEngine.MonoBehaviour.StartCoroutine(IEnumerator)` | `0x1C7F160` | Validation and engine scheduler handoff |
| `UnityEngine.MonoBehaviour.StartCoroutineManaged2` | `0x1C7F0B0` | Direct engine scheduler internal-call stub |
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

`Characters.ManageCharacters` calls ordinary `Character.Init` once per fresh
card in board-list order, publishes the board, and only then begins the Init and
ordered Start passes. Both `Character.Init` and `Character.InitWithNoReset`
clear raw `bluff`, allocate a `<DelayReveal>d__84` directly, store state zero
and the physical Character reference, and pass that iterator to
`MonoBehaviour.StartCoroutine`. Neither caller retains the returned Coroutine
handle or cancels an earlier continuation. The separately emitted
`Character.DelayReveal` wrapper at `0x364A20` merely constructs the same state
machine and has no retained native call reference; it is not part of the
shipped setup path.

The version-matched
[Unity 2022.3 `StartCoroutine` contract](https://docs.unity3d.com/2022.3/Documentation/ScriptReference/MonoBehaviour.StartCoroutine.html)
returns when the iterator reaches its first yield. The first `MoveNext` call is
therefore synchronous inside each Init call. It clones the then-current
`dataRef.role` into the physical card's shared `Character.role` field, creates
the common `0.3f` `WaitForSeconds`, stores it as iterator current, changes the
iterator to state one, and returns true. This is not a per-continuation role
snapshot: a later reinitializer starts another iterator and overwrites the same
`Character.role` field before either sibling resumes. The iterator itself has
only state, current-yield object, and Character-reference fields.

All ordinary first yields consequently finish before the ordered Start pass.
Chancellor, Twin Minion, Puppeteer, and Shaman can synchronously add more
continuations during that pass without cancelling the ordinary ones. Twin's
two `InitWithNoReset` calls register the selected neighbour before the original
Twin body; a self-swap registers two more iterators on the same card. Those
game-side creation and registration orders are exact. `WaitForSeconds` starts
its scaled wait at the end of the current frame and permits resumption on the
first frame after the duration has elapsed, so the registrations in this one
synchronous setup call share the same frame boundary.

The checked `StartCoroutine(IEnumerator)` body validates non-null input and a
live MonoBehaviour, then resolves
`UnityEngine.MonoBehaviour::StartCoroutineManaged2(System.Collections.IEnumerator)`
and tail-dispatches it. The separately emitted `StartCoroutineManaged2` stub
resolves the same engine internal call. GameAssembly contains no queue, sort,
or resume policy beyond that handoff; the implementation lives in
`UnityPlayer.dll` 2022.3.10f1. Unity's version-matched contract explicitly does
not guarantee that coroutines finishing in one frame finish in start order.
Board order, writer call order, and iterator registration order are therefore
not valid substitutes for realized resume order.

At state one, `DelayReveal.MoveNext` calls `Character.Reveal` against current
shared card state and completes. Reveal contains no yield, so one resume is an
atomic game-code event: it recomputes `registerAs` through the current
`dataRef.role`, tests raw `bluff` with Unity-object equality, and only when that
field is null or destroyed calls the current role's `GetBluffIfAble` and then
`GiveBluff`. The selector, every RNG draw, must-include removal, script-list
addition, and bluff write all finish before another continuation can resume.
A live first result suppresses later sibling bluff selectors; null or destroyed
results remain retryable. `GiveBluff` still does not clear an older
`bluffRole` when the new result is not live.

Register-as runs before the raw-bluff guard and can itself consume RNG or mutate
a role cache even on a sibling whose bluff is already live. Current Twin
Minion/Marionette inherits the folded default `Role.GetRegisterAsRole`, which
returns null without RNG, but a global replay cannot generalize that fact to
other current roles. After acquisition, Reveal optionally dispatches Start for
`HealthyBluff`, then always dispatches Init and AfterRoundStart. Those hooks can
mutate card or global state; a `HealthyBluff` Start can reenter an ordered writer
and register additional continuations in the middle of the resume stream.

All checked integer selections, including `Helpers.RollDice`, converge on
`UnityEngine.Random.Range(int, int)` and its engine internal call
`UnityEngine.Random::RandomRangeInt`. The GameAssembly stub exposes the exact
arguments but not the engine RNG state. For a current Minion acquisition, the
native event consumes one 1-through-10 branch draw plus one duplicate-pool draw
on the 40% branch. The 60% unique branch then consumes one pool draw when the
must-include list is empty, or a probe plus a second independently selected and
removed must-include occurrence when a live entry exists. Thus an ordinary
Minion event consumes exactly two or three shared RNG draws. Demon selection
uses one unique-Villager pool draw or two must-include draws, then registers the
selected identity in the current script. Pool draws do not consume pool
entries; the second must-include draw removes one matching occurrence.

This matters directly when a runtime-Good Twin recipient reaches Reveal with
current Minion data. Its first successful selection rolls the exact 40/60
Minion branch, but earlier resumed Minion, Demon, or Drunk cards can consume
must-include occurrences and advance the shared RNG stream. Later sibling
continuations reuse its live bluff. If another ordered writer changed the
recipient away from Twin before that first resume, the dynamic Reveal instead
uses the replacement role. If the recipient preserved `HealthyBluff`, Reveal
dispatches the current Twin Start again and can cause another board swap and
additional continuations.

Exact global replay therefore requires the realized resume order, or a bounded
explicit branch over an independently justified set of scheduler outcomes. At
each atomic event it must retain the ordered and multiplicity-bearing
`UniquePool`, `DuplicatesPool`, and `BluffMustInclude`; current script lists;
the physical card's current `dataRef`, shared cloned `role`, runtime alignment,
statuses, resistance, runtime data, raw bluff, copied bluff role, register-as,
and reveal state; and either the intervening RNG outcomes/state or a complete
weighted branch over those draws. Creation ordinal alone is insufficient. A
bounded reconstruction can instead require a known acquisition-time pool
snapshot, a recipient that remains runtime-Good/current-Twin with a null bluff
and no `HealthyBluff`, and proof that no unresolved earlier resume can consume
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

### Postmortem capture surface

`memory_reader.py --postmortem-bluff-snapshot-json` exposes the deliberately
offline-only schema `demon_bluff.postmortem_bluff_snapshot` version 1. It
strictly reads the occurrence-preserving unique and duplicate pools, the
remaining must-include list, all four current script lists, and board identity
records containing board index, displayed position, current data, runtime
alignment, state, raw bluff, and register-as data. Its provenance includes the
validated DLL fingerprint and explicitly sets `live_solver_input` false.

This snapshot is useful structural evidence, not an acquisition ledger. The
reader is sequential rather than atomic, the caller asserts the postmortem
phase, settled pool state is not verified, and the must-include list is only
the remainder at capture time. Version 1 deliberately records no continuation
creation/resume order, acquisition ordinal, RNG state or results, shared
`Character.role`, `bluffRole`, statuses, resistance, or runtime data. It must
not be converted into `twin_recipient_bluff_context` or
`twin_recipient_bluff_prefix_context`, fed into a live solver decision, or used
to infer that board order was resume order. A future exact
bridge needs event-time capture immediately before `Character.Reveal` (or its
first successful bluff write), including a monotonic resume/acquisition
ordinal and every omitted mutation-relevant field.

### Bounded solver integration

The Rust solver's first exact consumer is deliberately narrower than a global
coroutine replay. An optional offline-only
`twin_recipient_bluff_context` uses rule marker
`twin_recipient_bluff_native_v1` and records the moved recipient, a global
acquisition ordinal, the occurrence-preserving duplicate and unique pools, and
the must-include list as it existed at that recipient's first successful
Minion bluff acquisition. Fresh live sessions never populate this hidden
field, and public `reveal_order` is never interpreted as its substitute.

An optional `twin_recipient_bluff_prefix_context` composes exactly one earlier
Lilis acquisition with that recipient snapshot. Rule marker
`twin_recipient_bluff_one_lilis_prefix_native_v1` requires exactly three
strictly ordered acquisition events: Lilis, the moved Twin recipient, then
Shaman. It also records the occurrence-preserving must-include list immediately
before Lilis. The replay accepts this prefix only when the round has exactly
one board Demon and the deck's only Demon is Lilis, and when the three
positions are distinct, in range, and agree with the scenario's current-role
evidence.

Lilis uniformly selects one supported Villager occurrence from the pre-prefix
must-include list when any exists. Native then calls `List.Remove(selected)` on
the original list, so it removes the first equal `CharacterData` occurrence
even when a later duplicate occurrence supplied the selected source index.
Otherwise Lilis uniformly selects a supported Villager occurrence from the
immutable unique pool and leaves must-include unchanged. Only branches whose
resulting must-include list equals the independently captured recipient
snapshot survive. The recipient then retains the ordinary Minion 40/60
occurrence weights. Its `twin_recipient_bluff_trace.prior_acquisitions` stores
the exact Lilis position, ordinal, selected bluff, pool source, and occurrence
index. When a public Lilis card is present, it must show that selected bluff
before the branch is admitted.

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

Missing context preserves archived behavior. A prefix without the recipient
snapshot, malformed ordinals or positions, a non-one-Lilis round, an
unsupported pool occurrence, an over-cap ticket product, or any other
structurally incompatible context causes wholesale fallback rather than
partially mixing weighted exact worlds with the legacy approximation. The
current live regression corpus has no acquisition-event snapshots, so this
checkpoint changes no archived case result; it establishes the guarded input
boundary for future event-time postmortem fixtures and eventual global
interleaving work.

### Composable offline selector ledger

The next clean-room boundary is `bluff::ledger::replay_selectors` in
`crates/solver-core/src/bluff/ledger.rs`. Its versioned
`bluff_selector_ledger_native_v1` input composes independently ordered calls
to the exact `Demon.GetBluffIfAble`, `Minion.GetBluffIfAble`, and
`Drunk.GetBluffIfAble` selectors. This is **behavioral** reconstruction tested
against the native-static contracts above and the
[Drunk selector audit](../roles/gameplay_roles_doppelganger_drunk.md#drunk-selection-and-registration).
It adds no new native target or coverage classification.

The input contains immutable occurrence-preserving unique/duplicate pools,
mutable must-include occurrences, all four ordered script lists, and explicit
selector events with physical positions and acquisition ordinals. Names denote
one live canonical current-build asset each; repeated names denote repeated
references to that same asset. Distinct same-name assets and null/destroyed
assets are outside the supported model. Selector dispatch is explicit: a
Minion-faction asset with an override, such as Spy, cannot be assumed to call
the base Minion selector.

Each output retains its unconditional reduced rational probability, final
pools/script lists, and every selected source occurrence. In particular:

- Demon and Drunk consume only eligible Villager must-include entries, falling
  back to the immutable unique pool's Villager occurrences when none remain.
- Ordinary Minion keeps the exact 2/5 duplicate and 3/5 unique split. Its
  must-include selector is untyped and can consume an Outcast.
- Must-include removal deletes the first equal asset from the original list,
  while the trace preserves the actual selected occurrence index. Neither
  unique-pool nor duplicate-pool selection consumes a pool entry.
- Demon, Drunk, and Minion-unique selections append an absent selected asset
  to the correct script faction list. Existing occurrences are preserved and
  do not cause rerolls. Minion-duplicate never registers its selection.
- Drunk emits a pre-selection Corrupted-attempt effect: accepted unique-add
  with self as target, or resisted with existing status/target unchanged.
  This is an effect record, not a reconstruction of the whole status container.
- RNG draw counts include the Minion branch roll and must-include probe. The
  discarded probe's independent outcomes are marginalized. These counts are
  not engine RNG state, and the ledger does not claim seed-exact replay.

Rational path mass is necessary once branches leave different pool widths.
For example, start with must-include `[Scout, Bard]` and one duplicate option.
A Minion duplicate followed by a Demon has two paths of mass `1/5` each;
a Minion unique followed by a Demon has two paths of mass `3/10` each.
Multiplying independently reduced per-event ticket counts would lose this
relationship. Tests also condition this general ledger on the old captured
post-Lilis remainder and verify agreement with the existing one-Lilis ticket
distribution, including separated duplicate source indices.

This version accepts at most 16 events with strictly increasing ordinals and
distinct nonzero physical positions. It rejects a repeated body's acquisition
because reinitialization or failed first assignment would require additional
state. Ordinal gaps are allowed only under the caller's proof that omitted
events do not alter the modeled state. Every needed support must be nonempty
on every positive-mass path: if even one path fails, the whole operation returns
an error rather than dropping that path and biasing the distribution. Native
failure side effects are not emulated. Path-count, cloned-entry, and checked
integer-arithmetic bounds also reject the complete operation without returning
partial results.

This API is deliberately separate from `GameState`, Python session ingestion,
and live recommendations. It models selector calls only. It does not execute
`Character.Reveal`, register-as callbacks, `GiveBluff`, HealthyBluff Start,
Init/AfterRoundStart hooks, scheduler choices, or intervening writers. Those
effects must be proved irrelevant before composing these calls; full Reveal
interleaving remains unresolved. The version-1 postmortem snapshot still cannot
be promoted to this event ledger because it has no event-time/order evidence.

Checkpoint validation: 524 Rust library tests (including 13 new ledger tests),
778 Python tests, 13 reverse-engineering tests, release build, formatting, and
the byte-for-byte method-coverage check pass. The long simulation suite was
not rerun for this isolated API: scenario generation, validators, and live
entry points have no new calls or behavior changes.

## Typed import, overlaps, and shared identity

Thirteen target memberships intentionally overlap earlier checked boundaries:

- `Character.GiveBluff`, `Character.Init`, `Character.InitWithNoReset`,
  `Character.<DelayReveal>d__84.MoveNext`, `Character.Reveal`, and
  `Characters.ManageCharacters` reuse exact lifecycle or role-boundary
  identities.
- `Characters.GetRandomUniqueVillagerBluff`,
  `Characters.GetARandomBluffMustIncludeOfType`,
  `Characters.FilterRealCharacterType(CharacterData)`,
  `Characters.FilterCharacterType(CharacterData)`,
  `Characters.FilterBluffableCharacters(CharacterData)`, and
  `Gameplay.AddScriptCharacterIfAble` reuse exact pool, roster, or role-helper
  identities.
- `UnityEngine.Random.Range(int, int)` reuses the exact engine-RNG identity
  already selected by multiple role boundaries.

The CharacterData overloads of `FilterCharacterType` and
`FilterAlignmentCharacters` share their emitted Il2CppDumper C identifiers
with existing `List<Character>` overloads. Their target entries use explicit
`prototype_name` aliases for Ghidra while retaining the unmodified metadata
signatures and RVAs for validation.

Three selected identities have a second metadata identity at the same RVA:
`Helpers.RollDice` shares with `Calculator.RollDice`,
`UnityEngine.Random.Range(int, int)` shares with `RandomRangeInt`, and
`UnityEngine.MonoBehaviour.StartCoroutine(IEnumerator)` shares with
`StartCoroutine_Auto`. Only the first identity in each pair is a member of this
31-method target. The exact alias remains visible in `script.json`, but does
not add a target membership, FunctionDefinition, or selected RVA.

The expanded boundary exports all 31 functions from both the baseline and typed
projects with zero failures, and all 31 select distinct signatures and RVAs
within this target. Of the eight scheduler-boundary additions, six reuse exact
definitions already in the target union; the two StartCoroutine identities add
two FunctionDefinitions at two distinct native RVAs. Across all 41 checked
target sets, deterministic discovery now yields 882 memberships, 543 distinct
selected FunctionDefinitions, and 445 unique native RVAs. The 339 repeated
exact identities explain the membership/definition difference; folded or
shared bodies explain the remaining 98-definition RVA gap.

The rebuilt GDT contains 151,680 datatypes and all 543 selected definitions.
The scheduler expansion required no additional reachable datatype imports
during its six-batch no-analysis refresh. Its separate read-only pass validated
all 882 memberships and 2,585 membership-level parameter-storage locations
with zero program mutations; this target accounts for 31 memberships and 89
storages. The checked quality report covers all 31 baseline/typed pairs:
unresolved type tokens fall from 244 to 74, raw field-offset accesses from 361
to 88, indirect-call patterns from ten to zero, and placeholder parameter
tokens from 320 to zero. Three decompiler-error markers remain unchanged in
the baseline and typed exports, and the directional regression policy passes
with zero regressions.

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
- The bounded exact prefix can replay one Lilis acquisition before a moved
  Twin recipient only from event-time order and both pre-Lilis and
  recipient-time must-include snapshots; its `prior_acquisitions` trace is
  part of the exact scenario claim.
- The version-1 postmortem bluff snapshot is a strict offline pool/script/board
  capture, not acquisition-order provenance and not live solver input.
- `Character.Reveal` must remain one atomic event through register-as, pool/RNG
  mutation, `GiveBluff`, and synchronous HealthyBluff/init hooks. Unsupported
  hook re-entry or missing event-time state requires atomic fallback.
- Invalid or empty pool configurations generally fail rather than yielding a
  playable card with no bluff.
