# Gameplay role: Oracle (managed `Investigator`)

Build: `f530404b0f3f_807de4a83df4`

Evidence status: **metadata** for the public `CharacterData` binding and
**native-static** for every method declared by managed `Investigator`, all six
methods in its compiler-generated helper, the truthful and bluff candidate
filters, registered/current identity lookup, script Minion pool, all-ascension
fallback, result ordering, exact text, and Day-only output surface. Native
bodies and decompiler output remain outside the repository.

The checked target set is
[`reverse_engineering/targets/gameplay_role_oracle.json`](../../targets/gameplay_role_oracle.json).
Its read-only baseline and typed Ghidra exports each complete at 19/19
functions with no failures. The body-free
[`quality report`](../../reports/f530404b0f3f_807de4a83df4_typed_quality_gameplay_role_oracle.json)
passes its regression check: unresolved-type tokens fall from 116 to 78, raw
field-offset accesses from 139 to 101, raw integer type tokens from 116 to 62,
placeholder parameter tokens from 69 to zero, and indirect-call patterns from
four to zero. Both exports retain two decompiler-error and 27 warning markers.

## Public asset binding and managed identity

The shipped `sharedassets0.assets` `CharacterData` at path ID `21628` is named
`Oracle`, has `characterId` `Oracle_07039445`, and binds its managed-reference
role to exact `Investigator` at TypeDefIndex `5893` in `Assembly-CSharp`. The
raw object is 10,228 bytes at file offset `23,711,736` and has SHA-256
`EC20FFD271AF822618EC182849C46FEFA8EB00519287E954C8F794697790251C`.
The containing asset has the build-manifest SHA-256
`E0D239C03FA47EC138F9C2797779E83B65AEF41F30C5AD84D4CA76767A93E967`.

Oracle is a Good Villager (`characterType == 10`,
`startingAlignment == 10`). It is bluffable, is not usually disguised, has
`picking == false`, and uses `abilityUsage == 0`. It carries no additional
status, tag, bundle, or appearance condition. Its authored description and hint
are:

```text
Learn that 1 out of 2 characters is a specific Minion role.

Both characters in my info are Good
```

The managed description getter retains older wording:

```text
Learn that 1 of 2 players is a particular Minion
```

The asset establishes that public Oracle is managed `Investigator`. It also
contains the localized `oracle_0` and `oracle_1` format records, including the
English result template and no-Minions sentence, but the executable selection
and formatting contract below comes from native code.

## Audited boundary and shared bodies

| Group | Methods | Observable purpose |
| --- | ---: | --- |
| `Investigator.<>c` | 6 | Singleton/cache setup, displayed-ID keys, and random secondary keys |
| `Investigator` | 7 | Description, truth, bluff, formatting, Day dispatch, and construction |
| `Character` | 1 | Register-as-first current `CharacterData` lookup |
| `Characters` | 3 | Current-Character type/alignment filters and `CharacterData` type filter |
| `Gameplay` | 2 | Current script and all-ascension role pools |

The target contains 19 managed FunctionDefinitions and 17 unique native RVAs.
The normal and bluff displayed-ID selectors share RVA `0x3F0790`; their random
secondary selectors share `0x392D50`. Eight memberships use bodies already
shared with other managed identities: the generated constructor, four ordering
selectors, real and bluff action bodies, and the role constructor. Exact
metadata identities remain preserved even when the typed export's canonical
body label names Fortune Teller, Dreamer, Witness, or Slayer.

The six principal role bodies are:

| Managed identity | RVA | Boundary purpose |
| --- | ---: | --- |
| `Investigator.get_Description` | `0x3DE5D0` | Managed description string |
| `Investigator.GetInfo` | `0x3DDFA0` | Truthful two-pool generation |
| `Investigator.Act` | `0x3B09F0` | Day-only real callback |
| `Investigator.BluffAct` | `0x3B33E0` | Day-only bluff callback |
| `Investigator.GetBluffInfo` | `0x3DD8E0` | Reachable bluff generation |
| `Investigator.ConjourInfo` | `0x3DD810` | Exact positive/sentinel text |

`Investigator..ctor` is a fieldless base-role construction path at
`0x3CFFF0`. The generated helper's static constructor installs its singleton;
its ordinary constructor is empty. Each primary sort key returns
`Character.id`, while each secondary key returns `UnityEngine.Random.value`.
The secondary key only decides ordering when displayed IDs compare equal.

## Truthful candidate pools

Each `Investigator.GetInfo` invocation copies the complete current physical
`Character` list twice. The first copy is filtered to exact current type
`Minion`; the second is filtered to registered alignment `Good`.

The type filter uses:

```text
character.registerAs ?? character.dataRef
```

and compares that `CharacterData.type` with `Minion`. The alignment filter
instead compares:

```text
character.registerAs.startingAlignment ?? character.alignment
```

with `Good`. These are independent projections. Neither filter removes the
actor, dead cards, hidden cards, revealed cards, corrupted cards, duplicate
data identities, or repeated physical-list occurrences. Both preserve source
order and multiplicity.

When the current Minion pool is empty, truth immediately returns a fresh
`ActedInfo` with exact text `There are no minions` and no Character references.
It does not construct or index the Good pool and therefore does not require a
Good candidate for this sentinel.

When at least one current Minion exists, truth makes one integer
`Random.Range(0, minionCount)` draw and independently makes one integer
`Random.Range(0, goodCount)` draw. It does not remove the first selection from
the second pool. An empty Good pool therefore reaches native indexed-list
failure rather than a sentinel or partial result.

## Truthful result identity and duplicate references

The selected Minion occurrence supplies the public role label through
`Character.GetCharacterData()`, which returns `registerAs` when present and
otherwise direct current `dataRef`. A revealed Wretch consequently participates
as its sampled registered Minion identity and emits that exact Minion name.
The label is not the card's stable physical origin or displayed bluff.

The selected Minion and selected registered-Good occurrences are appended to
one list, ordered by displayed integer ID, secondarily ordered by a random float
key for an ID tie, and stored as the exact two references in the returned
`ActedInfo`. The text is:

```text
#{id1} or #{id2} is a {minionCharacterName}
```

Normally the two pools are disjoint. A current `Twin Minion` data identity can,
however, move onto a runtime-Good physical body without changing that body's
alignment. With no live `registerAs`, that same physical Character is both
Minion-type and registered Good. The two independent draws can then choose it
twice, producing an exact duplicate reference and text such as
`#5 or #5 is a Twin Minion`. This is reachable truth, not malformed data.

A Wretch cannot produce the same overlap: once its `registerAs` is populated,
both its current type and registered alignment come from the sampled Evil
Minion record. The actor itself remains eligible in either truth pool when its
current type/alignment qualifies.

The public sentence is not a universal proposition over all cards. It records
one sampled Minion identity and one independently sampled Good identity. Solver
validation therefore checks whether the exact result was reachable under one
world rather than treating every unreferenced card as negative evidence.

## Bluff generation

`Investigator.GetBluffInfo` begins with a fresh copy of all current physical
Characters filtered to registered alignment `Good`. It then:

1. draws one Good occurrence uniformly by integer index;
2. appends it to the result list;
3. removes that exact `Character` object from the candidate list;
4. draws a second Good occurrence uniformly by integer index; and
5. appends and sorts both references by displayed ID with the same random
   secondary key.

The two bluff references are therefore distinct Character objects. Self, dead,
hidden, revealed, corrupted, duplicate-role, and moved-data candidates are not
otherwise excluded. Fewer than two registered-Good Characters reaches native
indexed-list failure.

The false path does **not** compute truth and invert the sentence. After choosing
two Good references it independently draws a Minion `CharacterData` label from
the current script pool described below, then emits the same positive template.
The correct clean-room predicate is generation reachability: both references
must be distinct registered-Good Characters, and the named canonical Minion
must occur in the active label pool. A logically true-looking sentence can be a
native bluff result.

Because current type is irrelevant to the bluff reference filter, a
runtime-Good body carrying moved Twin Minion data remains selectable as one of
the two Good references. A registered Wretch is Evil-aligned and is not.

## Script Minion label and fallback pools

`Gameplay.GetScriptCharacters` creates one list by appending the four live
script lists in this order:

```text
current Townsfolk, current Outcasts, current Minions, current Demons
```

Bluff copies that list and filters exact `CharacterData.type == Minion`,
preserving order and occurrence multiplicity. If the filtered list is nonempty,
it is the label pool.

Only when that exact filtered script pool is empty does bluff call
`Gameplay.GetAllAscensionCharacters`. The helper appends the current
ascension's authored arrays in the shipped order Townsfolk, Outcast, Minion,
Townsfolk. The repeated Townsfolk array and omitted Demon array are real native
behavior. The subsequent exact Minion filter reduces this concrete fallback to
the authored all-ascension Minion occurrences.

One final integer index draw selects the label occurrence. There is no role-name
deduplication, bluffability test, in-play exclusion, retry, or no-Minions
sentence on the bluff path. An empty fallback Minion pool fails at indexed
lookup.

This is also the exact candidate hierarchy used by Wretch's independently
audited register-as selection: current script Minions first, all-ascension
starting Minions only when the first pool is empty. A stable revealed Wretch
must therefore retain one sampled Minion identity across Oracle and Scout
observations in the same represented world.

## Trigger dispatch and exact output

The folded real and bluff action bodies recognize only trigger `0x1E`, which
metadata fixes as `ETriggerPhase.Day == 30`. Every other trigger returns without
generating a result. On a successful Day invocation with a non-null `onActed`
delegate, `Act` calls virtual `GetInfo` once and invokes the delegate once with
that exact object. `BluffAct` does the analogous single `GetBluffInfo` call and
single callback.

The previously audited generic Character dispatch selects these bodies from the
actor's actual truth state. An ordinary clean public Oracle uses truth; a
corrupted Good Oracle and an Evil card presenting Oracle as its bluff use bluff
generation. Poet delegates to the identical `Investigator.GetInfo` or
`GetBluffInfo` body after its fresh provider draw, so direct Oracle and
Poet/Oracle have the same result semantics.

`ConjourInfo` has exactly two public results:

```text
#{0} or #{1} is a {2}
There are no minions
```

The positive branch formats both sorted displayed IDs and the selected Minion
data's canonical public `characterName`. The sentinel branch ignores IDs and
Minion data. It is reachable only from truthful `GetInfo`'s empty-Minion test;
bluff never requests it.

## Live schema and solver model

Current direct Oracle observations carry:

```json
{
  "targets": [2, 3],
  "minion_role": "Witch",
  "oracle_variant": "public_current"
}
```

or the exact truthful sentinel:

```json
{
  "no_minions": true,
  "oracle_variant": "public_current"
}
```

Poet/Oracle uses the same provider payload plus `copied_role: "Oracle"` and
`poet_variant: "public_current"`. The bridge requires a current board size,
exact native text, the newest acted event, exact ordered references, and a
canonical Minion label. Positive references are nondecreasing rather than
strictly increasing so the moved-Twin duplicate remains representable. The
sentinel requires an exact zero-reference event. Partial, mixed, stale,
out-of-board, noncanonical, wrong-case, or extra-field current payloads fail
closed.

The Rust validator models the truthful Minion/Good orientation existentially,
current-data moves, truthful duplicates, the truth-only sentinel, and the
distinct-Good bluff path. It enforces the current script Minion label pool when
that pool is present. Direct and Poet payloads delegate to the same predicate.
Explicit current-data Wretch positions share one stable register-as choice
across Scout and Oracle observations.

Anonymous natural Wretch identities are still grouped hidden state in
`Scenario`. Each current observation uses the exact required/forbidden
assignment budget, but those anonymous physical assignments are not persisted
jointly across separate observations. This is a documented conservative
false-positive boundary, not a claim that native Wretch registration rerolls.
Unmarked archived Oracle observations retain their historical validator.

`GameState` does not carry the native all-ascension CharacterData arrays. When
the current script Minion list is empty, the validator therefore accepts any
canonical knowledge-base Minion label as a conservative approximation of the
native all-ascension fallback. It can retain a world whose named Minion was not
available at that specific ascension. This affects Wretch register-as labels
and lying Oracle labels; it is a solver representation boundary, not native
behavior.

## Corpus and compatibility implications

A deterministic scan of the 426 checked-in `tests/cases_v2` fixtures finds:

- 120 direct apparent Oracle cards across 106 fixtures;
- 117 complete direct observations across 103 fixtures;
- three empty direct observations: `asc28_v3` #5, `asc67_v6` #3, and
  `asc69_v2` #8;
- eight Poet/Oracle observations, all unmarked legacy records;
- no historical no-Minions sentinel and no current provenance marker;
- all complete historical reference pairs distinct and ascending;
- 16 direct and four Poet observations that include the actor itself; and
- 14 direct and one Poet observation that target an apparent Wretch.

The corpus protects broad legacy behavior, self-selection, and Wretch exposure,
but it does not independently exercise the current duplicate-Twin result,
no-Minions sentinel, or exact provenance schema. Those edges are covered by
synthetic bridge and Rust regressions derived from the native boundary.

Reconstruction, solver, and live tooling should therefore:

- bind public Oracle to managed `Investigator`, never a guessed `Oracle` class;
- preserve the independent truthful Minion and registered-Good draws;
- allow exact duplicate truth only when one physical card qualifies for both
  pools, notably moved Twin Minion data on a runtime-Good body;
- require two distinct registered-Good Character objects for bluff;
- test bluff-result reachability instead of inverting the sentence's truth;
- use current script Minions first and the all-ascension Minion fallback only
  when that pool is literally empty;
- treat Wretch's sampled registered Minion identity as stable within a world;
- preserve exact nondecreasing acted-reference order and public text; and
- keep all unmarked historical observations on the legacy path.

## Typed union accounting

Five target memberships are exact managed-identity overlaps with the previous
27 target sets. This boundary adds 14 selected FunctionDefinitions and seven
new native RVAs. The deterministic 28-set union now contains 613 memberships,
393 distinct selected FunctionDefinitions, and 338 unique native RVAs. Its 220
exact membership overlaps and 55-definition folded/shared-body gap remain
explicit.

The rebuilt GDT contains 151,518 datatypes. Oracle signature application and
read-only validation both close 19/19 functions and 57 membership-level
parameter-storage locations, canonicalize eight shared bodies, import six newly
reachable datatypes on apply, and perform zero validation-time program
mutations. Across the whole union, the final read-only pass validates all 613
memberships and 1,795 parameter-storage locations.

## Remaining uncertainty

- Unity's global PRNG implementation and statistical relationship between
  successive integer and float draws are outside this native target.
- The target proves the role body's Day-only gate and one callback per
  successful invocation; whole-program reveal/callback reentrancy is inherited
  from the separately audited generic dispatch boundary.
- Native failure paths are identified for empty required pools and null global
  dependencies, but their UI presentation and exception recovery are not
  reconstructed here.
- Anonymous natural Wretch assignments are not yet stored as one joint hidden
  identity map across every current observation.
- `GameState` lacks the exact all-ascension Minion arrays, so an empty current
  script pool uses the conservative canonical-Minion approximation described
  above.
- Current fixtures do not contain the truthful sentinel or duplicate-reference
  surface, so a paired live capture would add behavioral corroboration beyond
  the native-static proof and synthetic tests.

This checkpoint closes the shipped Oracle clue-generation boundary. It does not
claim that every remaining Poet provider or every game role is decompiled.
