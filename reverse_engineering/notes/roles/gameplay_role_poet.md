# Gameplay role: Poet

Build: `f530404b0f3f_807de4a83df4`

Evidence status: **metadata** for the public `CharacterData` binding and
**native-static** for every method declared by managed `Gossip`, its exact
twelve-entry provider pool, the fresh provider-selection boundary, Day-only
real/bluff role bodies, and the generic `Character.Act`/`RoleAct` dispatch
needed to route an ordinary Poet result. The individual providers' complete
clue-generation bodies, Unity's RNG implementation, the generic acted-history
callback body, and whole-program writes to the public provider-list field are
outside this target and remain explicit scope limits. Native bodies and
decompiler output remain outside the repository.

The checked target set is
[`reverse_engineering/targets/gameplay_role_poet.json`](../../targets/gameplay_role_poet.json).
Its read-only baseline and typed Ghidra exports each complete at 20/20
functions with no failures. The body-free
[`quality report`](../../reports/f530404b0f3f_807de4a83df4_typed_quality_gameplay_role_poet.json)
passes its regression check: unresolved-type tokens fall from 74 to 9, raw
field-offset accesses from 144 to 21, raw integer type tokens from 86 to 33,
placeholder parameter tokens from 98 to zero, and indirect-call patterns from
eight to one. Both exports retain three decompiler-error and twelve warning
markers.

## Public asset binding and managed identity

The shipped `sharedassets0.assets` `CharacterData` at path ID `21629` is named
`Poet`, has `characterId` `Gossip_85354100`, and binds its managed-reference
role to exact `Gossip` at TypeDefIndex `5856` in `Assembly-CSharp`. The raw
object is 920 bytes at file offset `23,721,968` and has SHA-256
`E6D16F73B52975E3BBF8CD01190ED56B428433ABB24BDFB222391484850FD741`.
The containing asset has the build-manifest SHA-256
`E0D239C03FA47EC138F9C2797779E83B65AEF41F30C5AD84D4CA76767A93E967`.

Poet is a Good Villager (`characterType == 10`, `startingAlignment == 10`).
It is bluffable, is not usually disguised, and has `picking == false`. Its
serialized `abilityUsage` is enum value zero (`Once`), but the managed role has
no player picker and its shipped action body contains no once-use latch. The
asset carries no additional status, tag, appearance condition, or role-local
serialized bundle. Its public description is:

```text
Learn random info.
```

The asset establishes the public-name/managed-class binding. The provider
pool and its order come from `Gossip..ctor`, not from serialized asset data.

## Audited boundary and shared bodies

| Managed identity | RVA | Boundary purpose |
| --- | ---: | --- |
| `Gossip.get_Description` | `0x3BB490` | Managed description string |
| `Gossip.GetInfo` | `0x3BAD30` | Fresh real-information provider draw |
| `Gossip.Act` | `0x3B09F0` | Day-only real-result callback |
| `Gossip.BluffAct` | `0x3B33E0` | Day-only bluff-result callback |
| `Gossip.GetBluffInfo` | `0x3BAC00` | Fresh bluff-information provider draw |
| `Gossip..ctor` | `0x3BADC0` | Exact ordered provider-list construction |
| `Character.Act` | `0x3645C0` | Real role plus optional bluff-role routing |
| `Character.RoleAct` | `0x368790` | Callback installation and Act/BluffAct branch |

The other twelve memberships retain the exact provider constructor identities
referenced by `Gossip..ctor`:

| Public provider | Managed constructor | TypeDefIndex | RVA |
| --- | --- | ---: | ---: |
| Lover | `Empath..ctor` | 5863 | `0x357920` |
| Scout | `Scout..ctor` | 5854 | `0x357920` |
| Oracle | `Investigator..ctor` | 5893 | `0x3CFFF0` |
| Bounty Hunter | `BountyHunter..ctor` | 5871 | `0x357920` |
| Medium | `Lookout..ctor` | 5881 | `0x3CFFF0` |
| Knitter | `Knitter..ctor` | 5855 | `0x357920` |
| Hunter | `Tracker..ctor` | 5891 | `0x3CFFF0` |
| Enlightened | `Shugenja..ctor` | 5890 | `0x3CFFF0` |
| Empress | `Noble..ctor` | 5883 | `0x3CFFF0` |
| Bishop | `Bishop..ctor` | 5888 | `0x3CFFF0` |
| Gemcrafter | `Archivist..ctor` | 5884 | `0x3CFFF0` |
| Bard | `Acrobat2..ctor` | 5896 | `0x3CFFF0` |

The target contains 20 distinct managed FunctionDefinitions but only ten
unique native RVAs. `Gossip.Act` is folded with 16 other managed methods and
`Gossip.BluffAct` with 14; their canonical applied prototypes are the
ABI-compatible Witness methods even though the target preserves Gossip's exact
metadata identity. Every selected provider constructor is also folded into one
of two broad fieldless-constructor bodies. Provider identity is proved by the
exact `*_TypeInfo` allocations in `Gossip..ctor`, not by misleading names in
those folded bodies.

## Exact provider constructor pool

On the normal successful construction path, `Gossip..ctor` allocates a fresh
`List<Role>`, constructs one fresh role object for each row below, and appends
them in this exact order:

```text
1. Empath         -> Lover
2. Scout          -> Scout
3. Investigator   -> Oracle
4. BountyHunter   -> Bounty Hunter
5. Lookout        -> Medium
6. Knitter        -> Knitter
7. Tracker        -> Hunter
8. Shugenja       -> Enlightened
9. Noble          -> Empress
10. Bishop        -> Bishop
11. Archivist     -> Gemcrafter
12. Acrobat2      -> Bard
```

No gameplay predicate, deck lookup, state check, or conditional branch controls
inclusion. After all twelve additions, the list is written to the constructed
Gossip object's public `infoRoles` field at offset `0x48`, then base-role
construction completes. This proves a fresh list per successfully constructed
Gossip role object. It does not prove one role object per physical Character,
nor whole-program immutability of the public field without a complete field-xref
audit.

Architect, Fortune Teller, Confessor, and Baker do not appear in the current
constructor pool. Bounty Hunter remains a distinct provider from current
Hunter. A broad list of Villagers with superficially compatible clue fields is
therefore not a valid current Poet whitelist.

## Truthful and bluff delegation

Each `Gossip.GetInfo` invocation:

1. reads the current `infoRoles` list;
2. calls the integer overload of `UnityEngine.Random.Range(0, Count)` once;
3. retrieves exactly that list entry;
4. invokes its virtual `GetInfo` with the original `charRef`; and
5. returns the provider's `ActedInfo` object directly.

`Gossip.GetBluffInfo` repeats the same fresh selection sequence independently
at the call boundary, logs the character ID and selected provider type, invokes
the selected provider's virtual `GetBluffInfo` with the same `charRef`, and
returns that result directly. Neither method caches an index, writes the list,
removes an entry, retries, rerolls, or substitutes a fallback result. Null list,
entry, or required character state follows the native failure path rather than
a clean no-op.

The strongest safe probability statement is **one fresh uniform-index draw per
invocation under Unity's integer `Random.Range` contract**. Consecutive calls
share Unity's global PRNG state, whose implementation is outside this export,
so the native target alone does not prove mathematical independence. A selected
provider may consume additional RNG or state in its own `GetInfo` or
`GetBluffInfo`; those provider bodies are also outside this target.

## Trigger dispatch and output history

The folded Gossip real and bluff action bodies recognize only trigger `0x1E`,
which metadata fixes as `ETriggerPhase.Day == 30`. For every other trigger they
return without selecting a provider. On one successful Day invocation with a
non-null `onActed` delegate, `Act` calls the role's virtual `GetInfo` once and
invokes the delegate once with that exact result. `BluffAct` does the analogous
single `GetBluffInfo` call and single delegate invocation.

`Character.RoleAct` installs a fresh callback into `role.onActed` before role
dispatch. `EBluffCase.Act == 0` routes to virtual `Act`; any nonzero case used
here (`BluffAct == 10`) routes to virtual `BluffAct`. `Character.Act` may then
dispatch both the current real `role` and a separate `bluffRole`:

- when `CharacterHelper.CheckLying` is false, the real role receives `Act`, and
  a present bluff role also receives `Act`;
- when it is true, the real role normally receives `BluffAct`;
- the exception is a physical Evil character with a non-null bluff role: its
  real role receives `Act`, while the bluff role receives `BluffAct`.

Thus an ordinary clean Good Poet reaches `Gossip.Act`/`GetInfo`, a corrupted
Good Poet reaches `Gossip.BluffAct`/`GetBluffInfo`, and an ordinary Evil using
Poet as its displayed bluff reaches the Poet bluff path while its real Evil
role can still act. Current-data replacement and dual-role cases must use this
full matrix rather than a blanket “truthful Character calls GetInfo” shortcut.

The Day picking guard in `Character.Act` can suppress dispatch while the real
or bluff data is picking. Gossip itself has no Day once-guard. This target does
not include the generic `RoleAct` callback closure, Day/reveal caller topology,
or direct-call xrefs to `GetInfo`; it therefore proves one callback per
successful role-body invocation, not exactly one stored or displayed event per
reveal. Live `savedAct`/`actedInfos` behavior remains behavioral evidence for
the ingestion boundary.

## Sentinels and schema boundary

Current live ingestion marks only a fully parsed result from the exact native
provider pool with:

```json
{
  "poet_variant": "public_current",
  "copied_role": "<canonical public provider>"
}
```

The remaining provider-specific keys, JSON types, target counts, target
distinctness, role/faction spellings, direction spellings, and current-board
bounds are exact and fail closed in both the Python bridge and Rust validator.
Manual values are canonicalized before state is saved. `auto_card` requires the
newest acted event's description to equal the current public text and checks
the exact native reference shape; it may replace only an empty same-role Poet
placeholder, never a nonempty legacy/manual observation.

The audited current Scout one-Evil sentence and Oracle no-Minions sentence now
have exact provenance-marked schemas. Hunter has no one-Evil sentinel: its
truthful exhaustion result is the numeric distance `N - 1`. A Poet no-info
observation or Rambler `shut up!` replacement still has no current marker. The
strict schema accepts Scout's explicit sentinel or a positive 1-through-3
distance, Oracle's exact zero-reference sentinel or a complete two-reference
result, Hunter's positive distance, and either Bard's `-1` no-corruption
sentinel or a positive distance.

Historical fixtures omit `poet_variant`. That missing marker preserves their
legacy permissive behavior byte-for-byte, including obsolete providers and
empty observations. The old direct-Scout `distance: 0` encoding remains the
legacy one-Evil sentinel, while current direct and Poet-Scout payloads use
`one_evil: true` with the exact native sentence. An explicit malformed,
unknown, or partial current marker is rejected rather than falling back to
legacy interpretation.

## Typed union accounting

Two target memberships, `Character.Act` and `Character.RoleAct`, are exact
managed-identity overlaps with the previous 25 target sets. The boundary adds
18 newly selected FunctionDefinitions but only four newly selected RVAs because
the Gossip action bodies and every provider constructor are folded with already
selected native code.

The deterministic 26-set union contains 568 memberships, 360 distinct selected
FunctionDefinitions, and 316 unique native RVAs. Its 208 exact membership
overlaps and 44-definition folded/shared-body gap remain explicit. The GDT
contains 151,484 datatypes. Poet signature application and read-only validation
both close 20/20 functions and 50 membership-level parameter-storage locations,
canonicalize 14 shared bodies, import 17 newly reachable datatypes on the first
apply, and perform zero validation-time program mutations. Across the whole
union, the final read-only pass validates all 568 memberships and 1,658
parameter-storage locations.

## Corpus and reconstruction implications

A deterministic scan of the 426 checked-in `tests/cases_v2` fixtures finds:

- 123 Poet deck entries across 123 cases;
- 130 apparent Poet cards across 110 cases;
- 124 parsed `copied_role` records;
- 121 records using the current twelve-provider whitelist;
- three obsolete records: Architect twice and Fortune Teller once;
- six empty/no-info Poet observations; and
- zero historical fixtures carrying `poet_variant: public_current`.

The corpus gives broad exposure to Poet but predates the exact provenance
marker. It is therefore a compatibility corpus, not independent proof of the
current provider list or every provider's current public wording.

Reconstruction, solver, and live tooling should therefore:

- use the exact ordered twelve-provider pool, not a guessed Villager list;
- model one fresh provider draw separately for real and bluff result calls;
- delegate truth and lying to the selected provider's corresponding validator;
- retain Bounty Hunter independently from Hunter;
- fail closed on partial current payloads, out-of-board targets, noncanonical
  nested roles/types, and impossible positive-distance surfaces;
- preserve unmarked historical observations without relabeling them current;
- treat the newest public acted event and its references as one atomic capture;
  and
- use only an audited provider's exact sentinel schema; keep all remaining
  unaudited sentinel wording manual.

The Rust solver now enforces the exact current schema and fixes Scout's definite
false branches: when a named Evil role is certainly absent, or that role is the
only Evil while a distance sentence was captured, truthful Poet-Scout rejects
the world and a lying/corrupted Poet accepts it. Current Poet-Oracle additionally
models the exact truthful Minion/Good draw orientation, possible moved-Twin
duplicate, no-Minions sentinel, and distinct-Good bluff pair. A historical
executed Evil with unknown original role remains conservative rather than being
falsely pruned.

## Typed analysis corroboration

The typed export resolves every placeholder parameter and seven of eight
indirect-call patterns. It exposes `Gossip.infoRoles`, list size/item access,
`Character` and `Role` fields, GetInfo/GetBluffInfo vtable slots, the Day trigger
comparison, callback delegate storage, and the Act/BluffAct eCase branch. The
quality report's regression policy passes with no directional regression.

Decompiler presentation still has known limits. Folded action bodies are named
Scout or Witness rather than Gossip, folded constructors carry unrelated
canonical names, `Character.RoleAct` has imperfect register/parameter recovery,
and virtual/delegate sites retain an unrecovered-jumptable warning. Exact
metadata signatures, entry RVAs, vtable slots, and control-flow branches make
the bounded conclusions above reliable; concrete local pseudocode types at
those folded sites do not establish new managed identity.

## Remaining uncertainty

- Provider `GetInfo`/`GetBluffInfo` bodies other than the separately audited
  Scout, Oracle, and Hunter boundaries are not selected here; their exact clue-
  generation pipelines and sentinels remain later milestones.
- Unity's RNG implementation and statistical relationship between consecutive
  global PRNG draws are outside `GameAssembly.dll`'s selected boundary.
- The generic `RoleAct` callback closure is not exported, so native append,
  display, overwrite, and reentrancy behavior for acted history remains open.
- Day/reveal caller topology and direct calls to Gossip Get methods are not
  exhaustively xref-audited; the role body has no internal once-use guard.
- `infoRoles` is public. These six Gossip methods never mutate it after
  construction, but whole-program field immutability is not claimed.
- Role-object ownership and sharing across physical Characters is outside this
  constructor target; “fresh” applies to each successfully constructed Gossip
  role object.

These boundaries leave the roadmap's broad clue-generation and every-role
milestones open. This checkpoint fully closes the shipped Poet selector and
dispatch boundary, not every provider implementation it can call.
