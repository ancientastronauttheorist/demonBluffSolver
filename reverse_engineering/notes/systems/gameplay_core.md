# Gameplay Core Native Audit

Build: `f530404b0f3f_807de4a83df4`

Evidence level: `native-static`. The statements below combine current-build
IL2CPP metadata with Ghidra control flow. They have not yet crossed the
`live-validated` or `behavioral` gates.

The local native exports are intentionally not committed. Their requested
metadata names, signatures, and RVAs are preserved in
[`../../targets/gameplay_core.json`](../../targets/gameplay_core.json).

## Resource wrappers

- `Health.Damage(value)` dispatches `Reduce(value)` through the resource value.
- `Health.Heal(value)` dispatches `Add(value)`.
- `Health.ResetHp()` dispatches `Reset()`.
- `CurrentMaxValue.GetValue()` returns the `current` field at `+0x1C`.

The first three native bodies are folded with equivalent Gold or Mana helpers;
the getter is folded with 47 other trivial integer getters. These are shared
implementations, not evidence that the managed methods are aliases semantically.

## Health Add Max HP

`Health.AddMaxHp(value)` requires the resource value to be compatible with
`CurrentMaxValue`, increments its `max` field at `+0x18`, and then dispatches
the virtual `Add(value)` operation. It subsequently invokes the value's change
delegate when present and invokes `UIEvents.OnUIUpdate` when present. Because
`CurrentMaxValue.Add` itself raises the change delegate, the concrete current-
max path produces the virtual notification followed by the explicit one.

Null or incompatible values enter the IL2CPP runtime exception paths. This
native control flow replaces the malformed Cpp2IL type-test reconstruction.

## Player Resource Management

`PlayerController.ManageResources(character)` has two independent branches:

1. A Good character (`alignment == 10`) damages the player only when it lacks
   `NoDamage` and `killedByDemon` is false. Damage is obtained through the
   role's virtual `GetDamageToYou()` and passed to the health value's virtual
   `Reduce` operation.
2. An Evil character (`alignment == 20`) grants
   `CharactersHelper.GetUnrevealedCharactersCount() + 5` through the gold
   value's virtual `Add` operation.

The native byte check confirms that Cpp2IL's odd-looking
`killedByDemon == flag` expression means `!killedByDemon` after the earlier
`!NoDamage` condition.

## Gameplay State Transitions

`Gameplay.ChangeGameplayState(newState)` returns without an event when:

- `newState` already equals the current state;
- `newState` is `Draw` (`45`) while the current state is `BugReport` (`200`);
  or
- trailer-character mode is enabled and `newState` is `Draw`.

Otherwise it copies current state to `PrevState`, stores `newState`, and invokes
`GameEvents.OnGameplayStateChange` when the delegate is present.

`Gameplay.StartKill()` is a no-op in `Summary` (`50`) and `Night` (`20`). From
`Day` (`10`) it enters `Killing` (`40`); from `Killing` it restores
`PrevState`; other states are unchanged. `Gameplay.StartNight()` unconditionally
requests `Night` through `ChangeGameplayState`.

## Current Script Selection

`Gameplay.GetCurrentScript()` obtains the temporary ascension's character-count
list and the game mode's starting level. If the starting level is greater than
or equal to the list count, it returns the last item; otherwise it obtains the
starting level again and returns that indexed item. Required nulls follow the
runtime null-reference path. An empty list therefore attempts index `-1` rather
than returning a fallback object.

## Characters Count Construction

The `CharactersCount` constructor stores all five arguments exactly:

- `allCharCount` at `+0x10`;
- `towns` at `+0x14`;
- `demons` at `+0x18`;
- `outsiders` at `+0x1C`; and
- `minions` at `+0x20`.

The four `d*` fields remain zero-initialized. Cpp2IL had incorrectly discarded
the two stack-passed `outsiders` and `minions` arguments.

## Standard Roster Setup

`Gameplay.SetupCurrentVillageForStandard()` returns immediately unless the game
mode is Standard (`0`). It processes Outcast, Minion, Demon, then Villager:

1. clone the ascension's starting array for that faction;
2. remove every role already present in the corresponding current-faction list;
3. compute `max(baseCount, disguisedCount) - currentList.Count`; and
4. randomly move up to that many candidates into the current list, stopping if
   the candidate list is exhausted.

The paired base/disguised fields are `outs/dOuts`, `minion/dMinion`,
`demon/dDemon`, and `town/dTown`. This confirms the max-count rule and faction
order that Cpp2IL only partially recovered.

## Random Roster Selection

`Gameplay.GetRandomCharacters(characterCount)` orchestrates selection as
follows:

1. add each ascension `mustInlcude` role to its persistent faction roster;
2. call `ManageAlwaysInDeck(characterCount)`;
3. run Standard roster setup in Standard mode, or add every `roguelikeDeck`
   role to its roster in Roguelike mode (`10`);
4. copy the four persistent faction rosters and create an empty result;
5. choose `CurrentScript.demon`, then `outs`, then `minion`, then `town` roles;
   and
6. shuffle the result with `OrderBy(Random.value).ToList()` and return it.

Each ordinary pick is random without replacement from its temporary faction
copy. If that copy is empty, `ManageEmptyCharacterList` refills it. The chosen
role is also added to the persistent faction roster if absent. In Roguelike
mode, demon selection uses `Map.Instance.pickedDemon` instead of a random demon.

The method uses the base faction counts for the final result; disguised counts
only enlarge the Standard candidate rosters. The `characterCount` argument is
passed to `ManageAlwaysInDeck` and is not used as a loop bound here.

## Remaining Boundaries

This audit classifies the 13 target methods themselves as understood.
`ManageAlwaysInDeck`, `ManageEmptyCharacterList`, and their immediate helper
boundary are now covered by the follow-on
[`gameplay_roster_helpers.md`](gameplay_roster_helpers.md) audit. Role damage
methods and concrete resource implementations remain separate entries in the
complete coverage ledger. The isolated typed Ghidra project now applies the
selected IL2CPP structures and prototypes with validated Windows x64 storage;
its complete auto-analysis pass finished without a timeout, saved, and passed a
read-only reopen check for all 13 signatures and all 36 parameter-storage
locations. The count-only
[`typed quality report`](../../reports/f530404b0f3f_807de4a83df4_typed_quality_gameplay_core.json)
records 78 to 43 unresolved-type tokens and 237 to 144 raw field-offset
accesses, with the same four unrecovered-jumptable warnings as the baseline.
Broader call-graph expansion is the next static-analysis gate. Live behavior
and screenshot-paired memory checks remain pending.
