# Gameplay role: Poisoner

Build: `f530404b0f3f_807de4a83df4`

Evidence status: **native-static** for every method declared by managed
`Poisoner` and the complete ordered-Start, dispatch, adjacency, filtering,
integer-RNG, status, and resistance boundary needed to reproduce its shipped
behavior. Serialized asset evidence fixes the public binding and exact Start
slot. Native bodies and decompiler output remain outside the repository.

The checked target set is
[`reverse_engineering/targets/gameplay_role_poisoner.json`](../../targets/gameplay_role_poisoner.json).
Its read-only baseline and typed Ghidra exports each complete at 17/17
functions with no failures. The body-free
[`quality report`](../../reports/f530404b0f3f_807de4a83df4_typed_quality_gameplay_role_poisoner.json)
passes its regression check: unresolved-type tokens fall from 104 to 46, raw
field-offset accesses from 105 to 26, raw integer type tokens from 106 to 9,
placeholder parameter tokens from 141 to zero, and indirect-call patterns from
five to zero. Both exports retain two decompiler-error markers; warning markers
increase by one, from 34 to 35.

## Public asset binding and ordered Start slot

The shipped `sharedassets0.assets` `CharacterData` at path ID `21597` is named
`Poisoner`, has `characterId` `Poisoner_64796285`, and binds its
SerializeReference role to exact managed `Poisoner` at TypeDefIndex `5912` in
`Assembly-CSharp`. Its raw object SHA-256 is
`63CCF9E795472EFBE7C0198C9665300D3FBFF7A1F6473961D8ED5AC72643329E`.

The card is an Evil Minion (`characterType == 30`,
`startingAlignment == 20`), is not bluffable, is not usually disguised, and
has `picking == false`. Its `abilityUsage` is enum value zero (`Once`), but it
has no Day picker or player-selected target. It serializes no additional
statuses or appearance conditions and carries the `Corrupt` tag. The exact
current public description is:

```text
<b>Game Start:</b>
One adjacent Villager is Corrupted (if possible).

I Lie and Disguise.
```

The `level0` object at path ID `137026` references Poisoner's path ID as the
third entry (zero-based index 2) in `startGameActOrder`, immediately after
Pooka and before Drunk. The order entry is reached only after every board card
has completed `Init`. Poisoner is one of three explicit all-match exceptions in
`Characters.ManageCharacters`: every physical card whose current
`CharacterData` equals this asset receives `Start`, in current-list order.
Because normal construction assigns descending displayed IDs in that same
order, duplicate real Poisoners act from highest displayed ID to lowest.

## Audited boundary and shared bodies

| Managed identity | RVA | Boundary purpose |
| --- | ---: | --- |
| `Poisoner.get_Description` | `0x3E66D0` | Older managed wording surface |
| `Poisoner.GetInfo` | `0x3E6670` | Empty passive information record |
| `Poisoner.Act` | `0x3E64F0` | Complete Start-only role action |
| `Poisoner..ctor` | `0x3CFFF0` | Fieldless base-role construction |
| `ActedInfo..ctor` | `0x35D5D0` | Exact empty output shape |
| `Characters.ManageCharacters` | `0x36CE30` | Init/Start timing and duplicate exception |
| `Character.Act` | `0x3645C0` | Truth/lie and real/bluff action dispatch |
| `Character.RoleAct` | `0x368790` | Virtual real-role and apparent-role routing |
| `CharacterHelper.CheckLying` | `0x397750` | Evil actor truth-state input |
| `Role.BluffAct` | `0x3C4CA0` | Inherited empty bluff action surface |
| `Alchemist.OnInit` | `0x3B05F0` | Pre-Start Corrupted resistance |
| `Characters.GetAdjacentCharacters` | `0x36C2E0` | Previous-then-next circular pair |
| `Characters.FilterRealCharacterType(Character)` | `0x36BB40` | Current real `CharacterData` type filter |
| `Characters.FilterCharacterMissingStatus` | `0x36A8C0` | Raw active-status exclusion |
| `Characters.FilterCharactersWithoutResistance(Character)` | `0x36AE50` | Exact resistance exclusion |
| `CharacterStatuses.AddStatus` | `0x363AA0` | Independent status insertion and shared target |
| `UnityEngine.Random.Range(Int32)` | `0x1C86600` | One max-exclusive integer draw |

The 17 memberships select 17 distinct managed FunctionDefinitions and 17
native RVAs inside this target. `Poisoner..ctor` is one of 537 managed aliases
of the fieldless construction body and uses the established ABI-compatible
canonical prototype `Slayer___ctor`. `Role.BluffAct` has two managed aliases.
The selected integer `UnityEngine.Random.Range` body likewise has two managed
aliases, but the target retains the exact overload identity and applies the
named `UnityEngine_Random__Range_Int32` prototype. Shared native code is not
treated as shared managed identity.

## Start dispatch and output surface

`Poisoner.Act` has exactly one meaningful trigger: `Start` (enum value 5).
Every other trigger returns without effect. The normal `Character.Act` truth
matrix initially sends a lying Evil actor with a null `bluffRole` through the
real role's `BluffAct`. Poisoner inherits base `Role.BluffAct`, which forwards
to its virtual `Act`, so a fresh real Poisoner still reaches this branch during
its serialized pre-Reveal slot. If a nonstandard path has already populated
`bluffRole`, the Evil exception instead selects real `Act` directly. The
apparent bluff role remains a separate dispatch surface; it neither replaces
nor suppresses the real Poisoner Start action.

`Poisoner.GetInfo` allocates a fresh `ActedInfo` with an empty description and
a null character-reference list. Poisoner does not override `GetBluffInfo` or
`BluffAct`. It creates no speech, acted-history entry, picker, target reference,
reset callback, runtime-data object, or achievement request. The fieldless
constructor contributes no role-local state.

The managed description getter returns the older text
`1 adjacent good character is Poisoned`. That wording is not the current
authored contract: it says `good` rather than exact real `Villager`, and uses
the retired `Poisoned` name rather than `Corrupted`. The serialized public text
and native filters above control the shipped behavior.

## Target eligibility, order, and RNG

At Start, Poisoner obtains the circular adjacent pair around the acting
physical card. In ordinary boards with at least three cards, the pair is
ordered previous neighbour first and next neighbour second. It then applies
three stable filters in this exact order:

1. keep current real `dataRef.type == Villager` (`10`);
2. keep cards whose raw active-status list lacks `Corrupted` (`10`); and
3. keep cards whose exact resistance list lacks `Corrupted` (`10`).

Every filter preserves input order and repeated references. There is no check
of apparent or registered type, runtime or registered alignment, bluff role,
visibility, liveness, dead state, `MessedUpByEvil`, or
`MessedUpByEvil` resistance. A dead or hidden circular neighbour remains
eligible when its current real data is Villager and it passes the two
Corrupted checks. A displayed Villager backed by Drunk, Doppelganger, or any
other non-Villager real data does not qualify.

An empty filtered list is a clean no-op and consumes no random value. A
nonempty list causes exactly one integer `UnityEngine.Random.Range(0, count)`
draw; the upper bound is exclusive. There is no retry or alternate target
after selection. One eligible occurrence is deterministic. Two distinct
eligible neighbours are equiprobable. Repeated occurrences remain separate
draw indices and therefore retain their native probability weight even when
they denote the same physical card.

Executable relative-reference scanning independently closes the helper chain:
the Poisoner body calls adjacency at `0x3E655C`, real-type filtering at
`0x3E6585`, missing-status filtering at `0x3E65AE`, resistance filtering at
`0x3E65D7`, integer Range at `0x3E65F0`, and status insertion at
`0x3E662D` and `0x3E6651`.

## Corrupted and MessedUpByEvil composition

For the selected exact occurrence, Poisoner attempts these insertions in
order:

1. `Corrupted` (`10`);
2. `MessedUpByEvil` (`50`).

Both calls pass the acting Poisoner as `sourceRef` and null as `targetRef`.
`AddStatus` does not retain the source. For each call independently it checks
the exact matching resistance, unique-adds the status when absent and not
resisted, then replaces the one shared cure-target reference with the supplied
null value. A resisted insertion returns before changing either status or the
shared target.

The candidate was already proven to lack Corrupted and Corrupted resistance,
so its first insertion succeeds in ordinary synchronous gameplay. The second
attempt is independent: exact `MessedUpByEvil` resistance can block the marker
without undoing Corrupted. An existing `MessedUpByEvil` status does not exclude
the card from selection; a non-resisted duplicate attempt leaves membership
unique but still rewrites the shared target to null.

This differs materially from Pooka. Pooka does not prefilter Corrupted or its
resistance and always attempts its marker after the first insertion, whereas a
Corrupted or Corrupted-resistant neighbour never enters Poisoner's random pool
and therefore receives no Poisoner marker attempt at all.

All `Init` hooks run before ordered Start, so an actual Alchemist has already
installed exact Corrupted resistance and is removed by Poisoner's third
filter. Chancellor and Pooka have already acted: Poisoner observes current
post-Chancellor real data and position, while any neighbour Pooka successfully
Corrupted is removed by the second filter. Drunk, Puppeteer conversion, Plague
Doctor, Shaman, and Alchemist Start actions occur later and observe or mutate
the statuses left by Poisoner according to their own audited boundaries.

## Duplicate Poisoners and small boards

Duplicate Poisoner assets are the explicit exception to the ordinary
first-match Start scan. Every match acts synchronously, highest displayed ID
first. Each later actor rebuilds its adjacent pool from live current data and
status lists, so a Villager successfully Corrupted by an earlier Poisoner is
ineligible for every later one. The role therefore does not choose all targets
from one immutable snapshot.

`GetAdjacentCharacters` has these bounded board-shape results when its actor is
present:

- with at least three cards, previous and next are distinct circular entries;
- with two cards, the sole other card appears twice in the candidate pair;
- with one card, the Poisoner itself appears twice, then the real-Villager
  filter removes both occurrences; and
- with an absent actor or an empty current list, the helper returns an empty
  pair.

Thus a two-card eligible Villager produces a two-entry random pool whose two
indices both select that same card; statuses are attempted once after the draw.
A one-card board produces a clean Poisoner no-op. Missing singleton or list
dependencies still follow native failure paths; those malformed-runtime cases
are not authored board shapes.

## Reachability and legacy surface

Managed `Poisoner` declares exactly four methods: the description getter,
`GetInfo`, `Act`, and the constructor. There is no private alternate selection
helper analogous to Pooka's dormant `PoisonClosestNeighbours` and no unused
random-one-neighbour implementation to classify.

An executable-section direct-edge scan finds no static call or jump to
`Poisoner.Act`, which is the expected IL2CPP shape for virtual role dispatch.
Its sole absolute native pointer is the ordinary method-registration entry at
RVA `0x26A50D8`; the preceding description and information methods occupy
`0x26A50C8` and `0x26A50D0`. Public asset binding, the serialized Start entry,
`Characters.ManageCharacters`, and `Character.Act`/`RoleAct` establish the
live route to this registered virtual method. Zero direct xrefs therefore do
not make Poisoner unreachable.

The older managed description is the only legacy surface found in the class.
It is stale text, not a dormant gameplay body. The direct helper edges listed
above all originate inside the one reachable `Poisoner.Act` implementation.

## Typed-union accounting

Thirteen target memberships are exact managed-identity overlaps with the
previous 23 target sets. The boundary adds the description getter, `GetInfo`,
the Poisoner constructor, and exact integer Range overload as four newly
selected FunctionDefinitions. The constructor body already exists under its
canonical folded-body identity, so the getter, information method, and integer
Range body are the three newly selected RVAs.

The deterministic 24-set union contains 528 memberships, 337 distinct selected
FunctionDefinitions, and 308 unique native RVAs. Its 191 exact membership
overlaps and 29 folded-body differences remain explicit. The GDT contains
151,448 datatypes. Poisoner signature application and read-only validation
both close 17/17 functions and 58 membership-level parameter-storage
locations with zero imported datatypes and zero program mutations. Across the
whole union, the final read-only pass validates all 528 memberships and 1,547
parameter-storage locations.

## Corpus, solver, and live implications

A deterministic scan of the 426 checked-in `tests/cases_v2` fixtures finds:

- 77 Poisoner deck entries across 77 cases, with no duplicate-Poisoner deck;
- 77 true-Evil Poisoner records and 71 executed-Evil Poisoner records;
- no apparent-role Poisoner board entries; and
- 29 notes mentioning Poisoner, all of which also mention corruption or
  poisoning.

The corpus broadly exercises ordinary Poisoner placement and corruption but
does not independently prove the duplicate or two-card edges. Native lifecycle
and helper control flow supply those boundaries.

Reconstruction, solver, and live tooling should therefore:

- execute every real Poisoner in descending displayed-ID order after Pooka and
  before Drunk;
- build each live candidate pool as previous then next, retaining only current
  real Villagers that lack both Corrupted and Corrupted resistance;
- branch over every distinct logical selected target while retaining duplicate
  occurrences only when probability weighting matters;
- apply Corrupted and then independently apply MessedUpByEvil, with no marker
  attempt for a card excluded by the Corrupted checks;
- keep dead and hidden real Villagers eligible;
- emit no Poisoner-local clue, picker, result history, reset, or achievement;
  and
- treat the two-card duplicated neighbour and one-card empty result as bounded
  native behavior.

The current Rust ordered-Start simulator already matches these logical rules:
it processes all Poisoners high-ID first, observes prior Pooka/status changes,
filters real Villagers by live Corrupted state and Init resistance, branches on
the surviving target, and tracks the independent marker resistance. It
deduplicates repeated references because scenario enumeration models logical
possibility rather than RNG probability. This checkpoint therefore requires
no solver or live-tool change.
