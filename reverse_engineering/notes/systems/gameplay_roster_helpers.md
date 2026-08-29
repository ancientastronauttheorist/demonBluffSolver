# Gameplay Roster Helper Native Audit

Build: `f530404b0f3f_807de4a83df4`

Evidence level: `native-static`. The findings combine current-build IL2CPP
metadata, Ghidra control flow, and raw instruction review where Ghidra lacked a
float prototype. They have not passed a live or behavioral validation gate.

The private native exports are not committed. Their exact metadata names,
signatures, and RVAs are preserved in
[`../../targets/gameplay_roster_helpers.json`](../../targets/gameplay_roster_helpers.json).

## Always-In-Deck Routing

`Gameplay.ManageAlwaysInDeck(characterCount)` computes:

```csharp
float evilFraction =
    (float)(Gameplay.CurrentScript.demon + Gameplay.CurrentScript.minion) /
    (float)characterCount;
```

It then visits every entry in
`currentTemporaryAscension.alwaysInDeck`. A role with `usuallyDisguised` set is
always added to its real faction roster. For every other role:

```csharp
if (Calculator.GetPercentageChance(evilFraction))
    Characters.Instance.BluffMustInclude.Add(role);
else
    AddCharacterToTheRoster(role);
```

`Calculator.GetPercentageChance(value)` is exactly
`UnityEngine.Random.Range(0f, 1f) < value`. The comparison is strict and the
input is not clamped. `ManageAlwaysInDeck` likewise has no zero guard for
`characterCount`.

Cpp2IL recovered the branch shape but typed the division result as an integer.
The native instructions explicitly convert both operands to `float` before the
division.

## Roster Insertion

`Gameplay.AddCharacterToTheRoster(role)` appends the role to one persistent
list according to `CharacterData.type`:

- Villager (`10`) -> `currentTownsfolks`;
- Outcast (`20`) -> `currentOutsiders`;
- Minion (`30`) -> `currentMinions`; and
- Demon (`100`) -> `currentDemons`.

There is no duplicate check. `None` and unknown values are ignored, while a
null role or required null list follows an IL2CPP null-reference path.

## Empty-Pool Refill

`Gameplay.ManageEmptyCharacterList(data, type)` does not read `data`. After a
guard requiring `Gameplay.Instance`, it builds a fresh list by appending these
current-ascension arrays in this exact order:

1. `townsfolks` (`+0x68`);
2. `outsiders` (`+0x70`);
3. `minions` (`+0x78`); and
4. `townsfolks` (`+0x68`) a second time.

It does not append `demons` (`+0x80`). The repeated Townsfolk source and missing
Demon source are present in the native instructions, so they are current-build
behavior rather than a Cpp2IL rendering error.

The combined list is passed through:

```csharp
var available = Characters.Instance.FilterNotInPlayCharacters(combined);
return Characters.Instance.FilterRealCharacterType(available, type);
```

`FilterNotInPlayCharacters` clones its input, iterates
`Gameplay.CurrentCharacters`, and removes one occurrence of each character's
`dataRef` when present. `FilterRealCharacterType` returns a new list containing
entries whose `CharacterData.type` equals the requested type. Both operations
preserve remaining order and duplicates.

Consequently, duplicate Townsfolk candidates can survive the not-in-play
filter. Under faction-consistent ascension data, a Demon refill returns an
empty list because no Demon source array was added. Whether either case is
reachable during normal selection remains behaviorally unverified.

## Typed-Analysis Corroboration

The isolated typed Ghidra project completed its full analysis pass without a
timeout, saved, and then passed a read-only reopen check for all six helper
signatures and all 19 parameter-storage locations. In particular, the typed
`ManageAlwaysInDeck` decompilation recovers the three-argument managed ABI,
names `CharacterData.usuallyDisguised`, and renders the numerator and
`characterCount` conversions as `float` before division. This independently
corroborates the raw-instruction audit above.

The count-only comparison in
[`../../reports/f530404b0f3f_807de4a83df4_typed_quality_gameplay_roster_helpers.json`](../../reports/f530404b0f3f_807de4a83df4_typed_quality_gameplay_roster_helpers.json)
records 38 to 34 unresolved-type tokens and 76 to 41 raw field-offset accesses,
with no decompiler-error markers in either export. Decompiled bodies remain in
the private artifact store.

## Remaining Uncertainty

The target-local logic and immediate helper boundary are understood. Exact
exception classes on null paths remain represented by unnamed IL2CPP runtime
helpers, and the practical reachability of empty Demon pools or duplicate
Townsfolk refills still needs a fixture or live validation.
