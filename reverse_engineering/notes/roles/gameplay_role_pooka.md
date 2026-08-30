# Gameplay role: Pooka

Build: `f530404b0f3f_807de4a83df4`

Evidence status: **native-static** for all five methods declared by managed
`Pooka`, the active deterministic Start helper, the dormant random-neighbour
helper, and the exact status and ordering helpers needed to distinguish them.
Serialized asset evidence fixes the public binding and ordered-Start slot.
Native bodies and decompiler output remain outside the repository.

The checked target set is
[`reverse_engineering/targets/gameplay_role_pooka.json`](../../targets/gameplay_role_pooka.json).
Its read-only baseline and typed Ghidra exports each complete at 9/9 functions
with no failures. The body-free
[`quality report`](../../reports/f530404b0f3f_807de4a83df4_typed_quality_gameplay_role_pooka.json)
passes its regression check: unresolved-type tokens fall from 53 to 31, raw
field-offset accesses from 42 to 28, raw integer type tokens from 45 to 11,
and placeholder parameter tokens from 49 to zero. Both exports retain zero
decompiler-error markers and 11 warning markers.

## Public asset binding and ordered Start slot

The shipped `sharedassets0.assets` `CharacterData` at path ID `21593` is named
`Pooka`, has `characterId` `Pooka_13445289`, and binds its SerializeReference
role to exact managed `Pooka` at TypeDefIndex `5922` in `Assembly-CSharp`. Its
raw object SHA-256 is
`D7113C5C8FD18D560433C50FF010C3663883A83C37D8DB0DAFFCA426794EDEFE`.

The card is an Evil Demon (`characterType == 100`,
`startingAlignment == 20`), is not bluffable, is not usually disguised, and
has `picking == false`. Its `abilityUsage` is enum value zero (`Once`), but it
has no Day picker or player-selected target. It serializes no additional
statuses or appearance conditions and carries the `Corrupt` tag. The exact
public description is:

```text
<b>Game Start:</b>
Villagers adjacent to me are Corrupted (if possible).

I Lie and Disguise.
```

The `level0` object at path ID `137026` references Pooka's path ID as the
second entry (zero-based index 1) in `startGameActOrder`, immediately after
Chancellor and before Poisoner. Ordinary ordered-Start roles stop after their
first exact CharacterData match. Because `CurrentCharacters` is constructed in
descending displayed-ID order, duplicate real Pookas cause only the
highest-ID matching physical card to receive this ordered Start action.

## Audited boundary and shared constructor

| Managed identity | RVA | Boundary purpose |
| --- | ---: | --- |
| `Pooka.GetRules` | `0x3E6720` | Empty special-rule surface |
| `Pooka.Act` | `0x3E6700` | Start-only dispatch |
| `Pooka.PoisonClosestNeighbours` | `0x3E6780` | Dormant older random-one-neighbour path |
| `Pooka.PoisonNeighboursIfAble` | `0x3E6870` | Active deterministic two-neighbour path |
| `Pooka..ctor` | `0x3CFFF0` | Fieldless base-role construction |
| `CharacterStatuses.AddStatus` | `0x363AA0` | Resistance, uniqueness, and shared cure target |
| `Characters.GetAdjacentCharacters` | `0x36C2E0` | Circular pair used only by the dormant path |
| `Characters.FilterRealCharacterType(Character)` | `0x36BB40` | Exact current real-type filter |
| `CharactersHelper.GetSortedListWithCharacterFirst` | `0x398AF0` | Global-list rotation used by the active path |

The nine memberships select nine distinct managed FunctionDefinitions and nine
distinct RVAs within this target. `Pooka..ctor` is one of 537 managed aliases
of the same fieldless construction body. The target preserves exact Pooka
metadata while applying the established ABI-compatible canonical prototype
`Slayer___ctor`; shared native code is not treated as shared managed identity.

## Rule, action, output, and construction surfaces

`Pooka.GetRules` allocates and returns a fresh empty `List<SpecialRule>`.
Pooka's corruption is therefore an ordered role action, not a registered
special rule.

`Pooka.Act` has exactly one meaningful trigger: `Start` (enum value 5) routes
to `PoisonNeighboursIfAble`; every other trigger returns without effect. The
normal `Character.Act` truth matrix invokes the real role's `Act` for an Evil
actor even though that actor lies, so the public real Pooka reaches this branch
during its ordered slot. The apparent bluff role remains a separate dispatch
surface and does not replace Pooka's real action.

No Pooka method creates `ActedInfo`, writes speech, opens a picker, retains
references, schedules a reset, or requests an achievement. The fieldless
constructor performs only the shared base-role initialization and introduces
no runtime role-local state. Pooka consequently has no result-history or
between-Night reset contract beyond the status effects placed on other cards.

## Active deterministic neighbour algorithm

The active `PoisonNeighboursIfAble` path does **not** call
`GetAdjacentCharacters` and does not make a random draw. It rotates the exact
global `Gameplay.CurrentCharacters` list until the acting Pooka is first,
removes that first entry, then examines result index zero followed by the last
result entry. With the normal global list these are the two circular physical
neighbours: the occurrence after Pooka in global-list order, then the
occurrence before it.

Each endpoint is tested independently against its current real
`dataRef.type`. Exact `Villager` (`10`) qualifies. Apparent or registered type,
runtime or registered alignment, display bluff, state, liveness, active
status, and resistance do not affect this eligibility test. Chancellor acts
one ordered slot earlier, so Pooka observes any current real-data changes that
Chancellor has already completed.

For every qualifying endpoint, Pooka attempts these status insertions in
order:

1. `Corrupted` (`10`);
2. `MessedUpByEvil` (`50`).

Both calls pass the acting Pooka as `sourceRef` and null as `targetRef`.
`AddStatus` ignores the source, checks exact resistance separately for every
call, unique-adds the status when not resisted, and then replaces the one
shared cure-target reference with null even when the status was already
present. A Corrupted resistance can therefore block only the first attempt;
the later MessedUpByEvil attempt still occurs. Likewise, an existing
Corrupted entry does not prevent the later marker and can still have its
shared cure target overwritten.

The branch performs no fallback when a neighbour is not a real Villager. It
also performs no post-resistance reroll or alternate selection: the two
physical endpoints are deterministic.

## Small-board and malformed-state edges

After removing the actor, the active helper reads result index zero without a
count guard. An original board with fewer than two entries therefore reaches
the native failure path rather than returning a clean empty outcome.

On a two-card board, the sole other card is both result index zero and the last
entry, so it is processed twice. Status membership remains unique, but every
non-resisted repeat can rewrite the shared cure target to null. On an ordinary
board of at least three cards, each circular endpoint is processed once.

The rotation helper has a broader malformed-input quirk: it copies the
supplied list but derives rotation steps and appended entries from the global
current-character list. Pooka supplies that exact global list, making its
normal path an ordinary rotation. A caller supplying a different list could
inject or drop entries, but that is not the Pooka call shape.

## Dormant `PoisonClosestNeighbours` path

`PoisonClosestNeighbours` implements a materially different older algorithm.
If invoked, it asks `Characters.GetAdjacentCharacters` for the circular
previous/next pair, filters that list to current real Villagers while
preserving order and duplicate occurrences, and returns cleanly when the
filtered list is empty. Otherwise it draws one uniform list index and attempts
only `Corrupted` on that one occurrence. It does not add
`MessedUpByEvil`.

This helper is present in metadata and native code but is unreachable from the
current shipped gameplay flow:

- an executable-section relative-reference scan finds zero direct call or
  jump edges to `0x3E6780`;
- the active helper at `0x3E6870` has exactly one such edge, from
  `Pooka.Act` at call-site RVA `0x3E670E`;
- the only absolute pointer to each helper is its ordinary contiguous IL2CPP
  method-registration entry (`0x26A5258` for the dormant helper and
  `0x26A5260` for the active helper); and
- the dormant method is private and non-virtual, while the shipped Pooka asset
  serializes no callback or picker binding to it.

The `unreachable` classification is scoped to the shipped Standard/Ascension
game surface. The method remains physically registered and could be invoked by
an external reflection harness; the claim is not that IL2CPP stripped it or
made reflection mechanically impossible.

## Typed-union accounting

Six target memberships are exact managed-identity overlaps with the previous
22 target sets: `Pooka.Act`, the active helper, and the four shared status and
ordering helpers. The target adds `Pooka.GetRules`, the dormant helper, and the
Pooka constructor as three newly selected FunctionDefinitions. The constructor
body already exists under its canonical folded-body identity, so only the
GetRules and dormant-helper RVAs are new.

The deterministic 23-set union contains 511 memberships, 333 distinct selected
FunctionDefinitions, and 305 unique native RVAs. Its 178 exact membership
overlaps and 28 folded-body differences remain explicit. The Pooka signature
application and read-only validation both close 9/9 functions and 29
membership-level parameter-storage locations with zero imported datatypes and
zero program mutations.

## Corpus, solver, and live implications

A deterministic scan of the 426 checked-in `tests/cases_v2` fixtures finds:

- 166 Pooka deck entries across 166 cases;
- 158 executed-Evil Pooka records across 158 cases;
- no apparent-role Pooka board entries; and
- 78 notes mentioning Pooka, 51 of which also mention corruption.

The corpus broadly exercises the live active role but cannot by itself prove
the dormant helper's non-reachability. Native control flow and xrefs supply
that distinction.

Reconstruction, solver, and live tooling should therefore:

- preserve the existing deterministic visit of both circular neighbours,
  qualifying each by current real Villager type;
- preserve independent Corrupted and MessedUpByEvil resistance outcomes;
- run only the highest-ID ordinary duplicate Pooka at the serialized Start
  slot, after Chancellor and before Poisoner;
- avoid adding a one-neighbour random branch from the misleading dormant
  helper;
- emit no Pooka-local clue, active result, reset record, or achievement event;
  and
- treat the two-card duplicate occurrence and fewer-than-two-card failure as
  bounded native edges rather than ordinary authored board shapes.

The existing Rust Start-corruption model already follows the active path: it
chooses the highest-ID Pooka, visits both adjacent positions, filters through
the real-Villager-at-that-stage set, and applies the two resistance outcomes
independently. This checkpoint therefore requires no solver or live-tool
change.
