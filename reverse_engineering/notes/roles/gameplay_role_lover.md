# Gameplay role: Lover (managed `Empath`)

Build: `f530404b0f3f_807de4a83df4`

Evidence status: **metadata** for the public `CharacterData` binding and
**native-static** for every method declared by managed `Empath`, the circular
adjacency and registered-alignment helpers used by its clue, both Day action
bodies, and the complete `LoversAchivementHelper` side-effect boundary. Native
bodies and decompiler output remain outside the repository.

The target manifest is
[`reverse_engineering/targets/gameplay_role_lover.json`](../../targets/gameplay_role_lover.json).
It selects 15 managed FunctionDefinitions at 15 distinct target-local native
RVAs.

Its read-only baseline and typed Ghidra exports each complete at 15/15
functions with no failures. The body-free
[`quality report`](../../reports/f530404b0f3f_807de4a83df4_typed_quality_gameplay_role_lover.json)
passes its regression check: unresolved-type tokens fall from 92 to 38, raw
field-offset accesses from 59 to 24, raw integer type tokens from 60 to 15,
placeholder parameter tokens from 82 to zero, and indirect-call patterns from
four to zero. Both exports retain two decompiler-error and 23 warning markers.

## Public asset binding and managed identity

The shipped `sharedassets0.assets` `CharacterData` at path ID `21626` is named
`Lover`, has `characterId` `Lover_91302708`, and binds its managed-reference
role to exact `Empath` at TypeDefIndex `5863` in `Assembly-CSharp`. The object
is 5,816 bytes at file offset `23,695,640` and has SHA-256
`C342FC5445655FD7A3723C9C89563542B5FCA872D6DA82D15F82126C59302CAA`.
The containing asset has the build-manifest SHA-256
`E0D239C03FA47EC138F9C2797779E83B65AEF41F30C5AD84D4CA76767A93E967`.

Lover is a Good Villager (`characterType == 10`,
`startingAlignment == 10`). It is bluffable, is not usually disguised, has
`picking == false`, and uses `abilityUsage == 0`. It has no bundled character,
additional status, tag, appearance condition, or per-card achievement entry.
The Halloween achievement behavior described later is installed through the
global helper rather than this asset's achievement list.

The authored asset description is:

```text
Learn how many Evil characters I am adjacent to
```

The managed description getter retains a small wording difference:

```text
Learn how many Evil characters are adjacent to me
```

Neither string ends in punctuation. The asset establishes that public Lover is
managed `Empath`; a guessed public-name class is not the executable role
identity. Its authored hint and if-lies fields are both empty.

## Audited boundary and shared bodies

| Group | Methods | Observable purpose |
| --- | ---: | --- |
| `Empath` | 9 | Description, truth/bluff generation, adjacent-Evil count, exact text, achievement registration, Day dispatch, and construction |
| `Characters` | 1 | Circular previous/next Character references |
| `Character` | 1 | Register-as-first alignment projection |
| `LoversAchivementHelper` | 4 | Neighbor storage, execution subscription, unlock test, reset, and construction |

The class name `LoversAchivementHelper` and method name `ConjourInfo` preserve
the shipped spellings. All 15 target memberships are:

| Managed identity | RVA | Boundary purpose |
| --- | ---: | --- |
| `Empath.get_Description` | `0x3B98E0` | Managed description string |
| `Empath.Act` | `0x3B09F0` | Day-only truthful callback |
| `Empath.BluffAct` | `0x3B33E0` | Day-only bluff callback |
| `Empath.GetInfo` | `0x3B9810` | Truthful count, text, references, and achievement registration |
| `Empath.GetBluffInfo` | `0x3B9620` | Authored-domain false-count generation |
| `Empath.CheckAdjacentEvils` | `0x3B9290` | Registered-Evil occurrence count |
| `Empath.ConjourInfo` | `0x3B9560` | Exact public clue formatting |
| `Empath.CheckAchievementsAndUnlockIfAble` | `0x3B9140` | Truth-reference registration |
| `Empath..ctor` | `0x357920` | Fieldless base-role construction |
| `Characters.GetAdjacentCharacters` | `0x36C2E0` | Previous-then-next circular references |
| `Character.GetRegisterAlignment` | `0x365030` | Registered alignment lookup |
| `LoversAchivementHelper.AddCharacter` | `0x3A2D20` | Store one occurrence and subscribe once |
| `LoversAchivementHelper.CheckTrigger` | `0x3A2F10` | Current-alignment execution unlock test |
| `LoversAchivementHelper.Reset` | `0x3A2F80` | Per-occurrence unsubscribe and clear |
| `LoversAchivementHelper..ctor` | `0x3A31B0` | Fresh tracked-Character list |

`Empath.Act` is folded with the previously audited Witness, Oracle, Poet,
Scout, and Hunter real-action body. `Empath.BluffAct` is folded with the
corresponding bluff-action body. `Empath..ctor` uses the broad fieldless-role
constructor body already selected under several managed identities. The target
preserves exact Empath metadata even when a typed export uses an ABI-compatible
canonical name such as Witness or Dreamer.

`Characters.GetAdjacentCharacters`, `Character.GetRegisterAlignment`, and the
exact `Empath..ctor` FunctionDefinition were already present in earlier target
sets. The remaining Lover-specific generation and achievement methods add the
new clean-room boundary.

## Circular adjacency and exact reference order

`Characters.GetAdjacentCharacters` starts each call by copying the complete
live `Gameplay.CurrentCharacters` list. It searches that snapshot for the first
Character Unity-equal to the supplied actor and, when found, appends exactly
two occurrences to a fresh result list in this order:

```text
previous Character, then next Character
```

Both indices wrap around the circular list. The helper does not remove the
actor, deduplicate equal references, or filter by alignment, identity, status,
corruption, state, death, reveal, or visibility. Its exact small-board topology
is therefore:

| Current Character count | Returned references |
| ---: | --- |
| 1 | `[self, self]` |
| 2 | `[other, other]` |
| 3 or more | `[previous, next]` |

If no Character Unity-equal to the actor occurs in an otherwise valid current
list, the helper returns a fresh empty list. Null global dependencies and
null neighbor occurrences consumed by the later count can instead reach native
failure paths.

The duplicated one- and two-card results are observable data, not an
implementation detail. They affect the clue count, the `ActedInfo` references,
and the number of achievement subscriptions.

## Registered alignment and truthful counting

For each adjacent occurrence, `Empath.CheckAdjacentEvils` uses
`Character.GetRegisterAlignment`. That helper returns:

```text
live registerAs != null ? registerAs.startingAlignment : Character.alignment
```

It does not consult current `dataRef`, displayed bluff, corruption, any other
status, or the Character's state. Consequences include:

- a Wretch with a live Evil-Minion `registerAs` counts as Evil;
- an ordinary runtime-Evil Character, including a Puppet, counts as Evil;
- corruption alone does not change the count;
- dead, hidden, and revealed Characters remain eligible; and
- moved Evil-role data on a runtime-Good body remains Good unless that body has
  a live Evil `registerAs`.

`CheckAdjacentEvils` first locates the actor in the current physical list, then
counts every occurrence returned by the adjacency helper whose registered
alignment is Evil. It does not deduplicate. On a stable one-card board the
actor therefore contributes either zero or two; on a stable two-card board the
other Character contributes either zero or two. On ordinary boards the result
is zero, one, or two.

`Empath.GetInfo` performs the following observable sequence:

1. compute the registered-Evil adjacent occurrence count;
2. format that count through `ConjourInfo`;
3. call `GetAdjacentCharacters` again for the result references;
4. construct a fresh `ActedInfo` from the text and exact references;
5. register those reference occurrences with the Lover achievement helper; and
6. return the same `ActedInfo`.

The count and returned references consequently come from two separate live
adjacency-helper invocations rather than one shared list object. Normal
synchronous play does not mutate the circle between them, but a reconstruction
should not invent one persistent internal adjacency field. Truth makes no
random draw.

## Exact truthful text and references

`ConjourInfo` has three formatting branches:

```text
count == 0:  NO Evils\nadjacent to me
count == 1:  1 Evil\nadjacent to me
otherwise:   {count} Evils\nadjacent to me
```

The capital `NO`, singular `Evil`, plural `Evils`, embedded newline, lack of
punctuation, and casing are exact. The formatter itself accepts any signed
integer; ordinary native adjacency produces only zero through two.

The text does not print target IDs. The returned `ActedInfo` nevertheless
stores the exact previous-then-next Character occurrences. On one- and
two-card boards those are duplicate references. A zero clue still stores both
ordinary neighbor references; zero is not a zero-reference sentinel.

If the actor is absent from an otherwise valid current list, the count path
uses zero and the second helper call returns no references, yielding the zero
text with an empty list. Current live ingestion deliberately rejects such an
out-of-board event instead of treating that malformed native edge as a normal
observation.

## Bluff candidate construction, removal, and RNG

`Empath.GetBluffInfo` does not choose an arbitrary number different from truth.
It constructs a fresh ordered integer pool from the authored current script.
Let:

```text
S = Gameplay.CurrentScript.minion + Gameplay.CurrentScript.demon
limit = S + 1
```

The method appends ascending integers beginning at zero and stops when it has
appended `limit` entries or when the next integer would be three. For ordinary
nonnegative authored counts the initial pool is exactly:

| Authored `S` | Initial bluff pool |
| ---: | --- |
| 0 | `[0]` |
| 1 | `[0, 1]` |
| 2 or more | `[0, 1, 2]` |

This is the serialized script Minion-plus-Demon count. It is not a recount of
the live board, not the HUD total-Evil objective, and does not automatically
include a Puppet, a Wretch's registered identity, or another runtime-added
Evil.

After building the pool, bluff computes the real adjacent count and removes
one equal integer occurrence. The removal result is ignored. It then makes
exactly one integer `UnityEngine.Random.Range(0, remainingCount)` draw and
indexes the surviving ascending list. The values are unique, so every
surviving authored candidate has equal index probability. There is no shuffle,
retry, float draw, reference draw, or extra tie-break.

If runtime/register-as state makes the real count fall outside the authored
pool, removal changes nothing. Native selection continues from the complete
authored pool, whose values still all differ from that real count. For example,
with `S == 0` and one or two adjacent registered Evils, bluff can emit only
zero.

The critical empty-pool edge is `S == 0` with real count zero. Removing zero
leaves no candidate; native still reaches the zero-size random/index path, so
it cannot produce a valid clue. A null current script or a malformed negative
script count can also reach failure rather than a fallback sentence. For
ordinary `S >= 1`, removing a reachable real count leaves at least one
candidate.

After the one integer choice, bluff formats it with `ConjourInfo`, calls the
adjacency helper again, and returns the same deterministic previous-then-next
reference shape as truth. It does not register achievement callbacks. The
correct native predicate is therefore authored-pool reachability after
remove-one, not merely `claimed != actual`.

## Day dispatch and callback surface

The folded real and bluff action bodies recognize only trigger `0x1E`, which
metadata fixes as `ETriggerPhase.Day == 30`. Every other trigger returns
without generating a result.

On Day, each body captures the inherited `onActed` delegate. If it is null, no
clue is generated and truthful achievement registration does not run. If it is
non-null, `Act` calls virtual `GetInfo` once and invokes that captured callback
once with the exact returned object. `BluffAct` analogously makes one
`GetBluffInfo` call and one callback.

The separately audited generic Character routing chooses the real or bluff
body from the actor's current truth state. A clean truthful Lover uses truth;
a corrupted Good Lover and an ordinary lying Evil disguise use bluff.
Poet's fresh Empath provider receives the Poet Character as `charRef`, so a
Poet/Lover result counts and references the Poet's own physical neighbors under
the identical generation rules.

## `LoversAchivementHelper` side effects

The helper constructor creates a fresh `List<Character>`. Truthful
`GetInfo` visits every occurrence in its newly constructed `ActedInfo`
reference list and calls `AddCharacter` before the Day callback is invoked.
There is no alignment filter at registration time: both physical neighbor
occurrences are registered. Despite its name,
`CheckAchievementsAndUnlockIfAble` does not itself test or unlock an
achievement; it only performs these registrations.

For each occurrence, `AddCharacter`:

1. appends the exact Character to `AdjacentCharacters` without deduplication;
2. combines one bound `CheckTrigger` delegate into that Character's
   `onTrigger`; and
3. logs the resulting stored-list count on the normal path.

On a one- or two-card board, the same Character is appended and subscribed
twice. Repeated truthful Lover generation can likewise add further entries and
subscriptions; the helper has no contains check.

`CheckTrigger` ignores every trigger except `ETriggerPhase.OnExecuted == 40`.
On execution it calls the triggered Character's **current**
`GetRegisterAlignment`. If that value is Evil, it calls the achievement unlock
surface with exact identifier:

```text
Lover_Halloween_ACHIV_7657
```

The decision is not based on alignment captured when Lover spoke. A later
register-as/alignment change can therefore change the unlock result. Duplicate
subscriptions invoke the check once per stored occurrence and can request the
same unlock more than once; idempotence belongs to the achievement system, not
this helper.

`Reset` iterates every stored occurrence, removes one matching bound
`CheckTrigger` delegate from that Character for each entry, then clears the
list. Duplicate subscriptions are consequently removed symmetrically. Empath
does not itself call `Reset`; whole-game lifecycle timing for that call is
outside this role target.

Null helper/list/Character dependencies follow native failure paths. Notably,
`AddCharacter` appends before its null-Character failure. A truthful result
with ordinary nonempty references also depends on the global Lover helper
being present, while bluff has no such dependency.

## Current bridge and Rust schema

Current direct Lover observations carry an exact provenance marker:

```json
{
  "evil_adjacent": 1,
  "lover_variant": "public_current"
}
```

Their `CardInfo.info_text` must independently equal the matching exact native
sentence. Poet/Lover uses the same count surface plus exact provider metadata:

```json
{
  "copied_role": "Lover",
  "evil_adjacent": 1,
  "poet_variant": "public_current"
}
```

Memory ingestion requires a positive current board size, an in-board actor,
the newest acted event, one exact native sentence, and exact
previous-then-next acted references. It preserves `[self, self]` and
`[other, other]` on one- and two-card boards. Poet/Lover applies the same check
relative to the Poet position. Wrong case, punctuation, plurality, newline,
event age, reference order, reference count, or board bounds fail closed.

Manual current entry cannot independently recover native references, but it
requires board context and an in-board position, restricts the count to zero,
one, or two, stamps the exact corresponding text, and adds the current marker.
The bare `card_lover` builder remains unmarked by default so loading archived
fixtures does not silently reinterpret them.

The Rust current schema requires exactly the fields shown above, the exact
marker value, exact `copied_role: "Lover"` for Poet, an in-board actor, an
integer count in the exact text domain, and an exact text match. Extra, mixed,
future, wrong-typed, or role-inconsistent current provenance fails closed.
Direct and Poet observations delegate to the same current Lover predicate.

Truth uses the same registered-alignment projection as native and counts both
entries of a duplicated tiny-board adjacency pair. Explicit current Wretch
data registers Evil. For anonymous natural Wretch candidates, the validator
enumerates exact required/forbidden hidden assignments consistent with the
shared Outcast multiset and HUD budget, then retains every reachable adjacent
occurrence count. Duplicate adjacency occurrences share one hidden identity
choice but contribute twice when that choice is Wretch.

For a lying current source, Rust derives the authored Minion-plus-Demon count
from the public `n_evil` objective and subtracts one only in a Scenario branch
where Puppeteer generated the extra Puppet. It deliberately does not use the
candidate deck pool, whose undealt Evil roles are not authored board slots,
and it does not subtract Wretch. The validator requires the claim to be no
greater than `min(2, authoredCount)` and requires at least one reachable real
count different from the claim. This models construction followed by
remove-one, including rejection of the `S == 0`, actual-zero empty-pool case
and acceptance of zero when an extra registered/runtime Evil makes actual
nonzero.

Unmarked direct and Poet/Lover observations retain the historical validator:
truth requires equality, lying requires inequality, and a missing count remains
non-constraining. That permissive path intentionally protects the frozen corpus
and is not a statement about current native schemas. Once any current passive
provenance marker is present, malformed or mixed provenance cannot fall back to
legacy behavior.

## Corpus and compatibility implications

A deterministic scan of the 426 checked-in `tests/cases_v2` fixtures finds:

- 141 direct apparent Lover cards across 120 fixtures;
- 140 complete direct `evil_adjacent` observations across 119 fixtures;
- one non-count direct observation, `asc82_v5` #1, whose redesigned Rambler
  event is stored as `shut_up_target: 10`;
- direct count distribution: 59 zero, 48 one, and 33 two;
- six Poet/Lover observations across six fixtures, all complete;
- Poet/Lover count distribution: three zero, two one, and one two;
- 58 complete direct observations whose apparent Lover position is a known
  true-Evil position, plus one such Poet/Lover source;
- zero `lover_variant` or `poet_variant` markers on these historical records;
- zero stored exact Lover `info_text` values or acted-reference arrays; and
- relevant board sizes from six through ten for direct Lover and seven through
  nine for Poet/Lover, so no historical fixture exercises the one- or two-card
  duplicate-reference topology.

The corpus protects broad historical equality/inequality behavior, all three
public count values, ordinary circle wraparound, Evil disguises, and the
Rambler replacement. It does not independently protect the exact current text,
event-reference shape, authored bluff-domain restriction, empty-pool failure,
current provenance schema, or duplicate tiny-board behavior. Those surfaces
are covered by build-pinned bridge and Rust regressions derived from the native
boundary.

Reconstruction, solver, and live tooling should therefore:

- bind public Lover to managed `Empath`;
- count registered alignment rather than displayed role, current data role,
  corruption, or reveal/death state;
- preserve exact previous-then-next references and their multiplicity;
- count duplicated one- and two-card neighbor occurrences twice;
- preserve exact `NO`/singular/plural text and newline formatting;
- restrict bluff output to the authored Minion-plus-Demon domain after
  removing the real count;
- treat `S == 0`, actual zero as a native generation failure rather than a
  legal zero lie;
- retain truth-only achievement registration, duplicate subscriptions, and
  current-alignment execution checks; and
- keep every unmarked historical observation on the legacy predicate.

## Typed union accounting

Three Lover target memberships are exact managed-definition overlaps with the
previous 28 target sets: `Empath..ctor`,
`Characters.GetAdjacentCharacters`, and
`Character.GetRegisterAlignment`. Five target memberships use native bodies
already present in that union: those three definitions plus the folded
`Empath.Act` and `Empath.BluffAct` bodies.

The boundary therefore adds 12 selected FunctionDefinitions and ten unique
native RVAs. The deterministic 29-set union now contains 628 memberships, 405
distinct selected FunctionDefinitions, and 348 unique native RVAs. The rebuilt
GDT contains 151,530 datatypes.

Lover signature application and read-only validation each close all 15 target
memberships, validate 43 membership-level parameter-storage locations, and
canonicalize three shared bodies. Application imports 12 newly reachable
datatypes; the final validation performs zero program mutations. Across the
whole 29-set union, the read-only pass validates all 628 memberships and 1,838
membership-level parameter-storage locations.

## Remaining uncertainty

- Unity's global PRNG implementation and statistical quality are outside this
  target; the role boundary proves one uniform integer-index request over the
  surviving ordered pool.
- Native failure paths are identified for null dependencies and empty required
  pools, but their exact managed exception classes, UI presentation, and
  recovery behavior are not reconstructed here.
- Destroyed-object behavior beyond ordinary Unity-null comparison is outside
  the clean-room model.
- The target proves `LoversAchivementHelper.Reset` behavior but not every
  whole-program lifecycle call site that schedules it.
- Anonymous natural Wretch placement is solved exactly per observation but is
  not stored as one joint hidden identity map across every passive clue in a
  scenario.
- The checked corpus has no current-provenance Lover record, exact acted
  references, empty authored bluff pool, or tiny-board result; a paired live
  capture would add behavioral corroboration beyond native-static proof and
  synthetic tests.

This checkpoint closes the shipped Lover clue-generation and achievement
side-effect boundary. It does not claim that every remaining Poet provider or
every game role is decompiled.
