# Gameplay bluff acquisition

Build: `f530404b0f3f_807de4a83df4`

Evidence status: **native-static** for all 20 methods in the checked boundary.

The checked target set is
[`reverse_engineering/targets/gameplay_bluff_acquisition.json`](../../targets/gameplay_bluff_acquisition.json).
It closes the common path from per-card bluff assignment through role-specific
selection, must-include and pool draws, script-list registration, and fresh-card
creation. `Character.Init`, `Character.Act`, internal `Character.Reveal`,
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
20-method target, but coverage records both exact managed identities against
the shared body. The typed program applies the selected Helpers signature; it
does not collapse the Calculator method into the target membership count.

With this target added, deterministic target discovery yields six checked sets,
137 target memberships, 127 distinct selected function definitions, and 126
unique native RVAs. Ten repeated exact identities across the six sets account
for the membership/definition difference; this target introduces two of those
overlaps. The previously selected `Role.CheckIfCanBeKilled` and
`Role.CheckIfCanRemoveStatus` identities share one native body and account for
the definition/RVA difference. The Calculator alias is outside the target
union, so it adds a coverage classification but no target membership,
FunctionDefinition, or selected RVA. Existing typed artifacts for five sets
are intentionally stale; `build-types` must rebuild the header/GDT union before
`typed-refresh` reapplies it to the preserved project and enables typed export
of this boundary.

## Reconstruction implications

- Normal solver state does not need a cross-round or shipped-Chancellor stale
  bluff-role transition.
- Minion acquisition must retain its exact 40/60 split and registration
  difference.
- Demon and Minion-unique draws can expand a script faction list; Minion-
  duplicate, Spy, and Mutant draws do not.
- Spy's apparent identity and bluff share one cached Villager selection.
- Mutant attempts Mad before a multiplicity-weighted Good/bluffable draw.
- Invalid or empty pool configurations generally fail rather than yielding a
  playable card with no bluff.
