# Gameplay Lifecycle Native Audit

Build: `f530404b0f3f_807de4a83df4`

Evidence status: **mixed**. Eleven methods in the first setup, board,
per-card, and click/kill slice have native-static confirmation. The remaining
seventeen methods in the 28-method boundary are still metadata/managed
hypotheses and are listed as follow-up work below. No statement here is based on
live dynamic observation.

The checked target set is
[`reverse_engineering/targets/gameplay_lifecycle.json`](../../targets/gameplay_lifecycle.json).
Its baseline Ghidra export completed read-only at 28/28 functions with no shared
native bodies. Decompiled bodies remain in the private artifact tree.

## Boundary

The full target set follows:

- initial and per-village setup from `Gameplay.Init` and `Gameplay.HandOut`
  through `Gameplay.SetupDelay.MoveNext`;
- board-pool selection, `Character` construction, and publication of
  `Gameplay.CurrentCharacters`;
- delayed internal reveal, click dispatch, real/bluff presentation,
  protected/normal/demon kill paths, reveal-order accounting, and dead-character
  bookkeeping; and
- Night phase ordering and the transition back to Day.

Role-specific `Character.Act` and `Role.*` behavior, bluff and duplicate
selection algorithms, UI-only night presentation, and alternate restart entry
points are separate later boundaries.

## Handout entry

`Gameplay.HandOut` performs the following work in native order:

1. It resets player mana and both block counters. It also resets health unless
   the game mode is Roguelike (`10`).
2. It increments `currentLevel` and `currentDay`.
3. It allocates a `SetupDelay` iterator and starts that coroutine.
4. It invokes `OnStartNewLevel` after `StartCoroutine` returns.

The iterator cannot run to its first yield until Unity schedules it, but the
exact scheduling relationship between the coroutine body and the final event is
not proven statically.

## Setup coroutine

On its first `MoveNext`, `Gameplay.SetupDelay`:

1. calls `SetupCurrentAscension`;
2. clears the current picked script and calls `SetupCharactersCount`;
3. resets `CurrentReveal` to zero;
4. clears the current townsfolk, outcast, minion, demon, and
   `BluffMustInclude` collections;
5. obtains character counts and clears `DeadCharacters`;
6. selects a script and clones its `CharactersCount` into `CurrentScript`;
7. runs relic trigger `PreRoundInit` (`10`);
8. runs `SetupStartingCharacters` and the ordinary random-character selection;
9. in trailer mode, replaces that already-generated ordinary roster with the
   trailer roster;
10. runs relic trigger `PreCharactersInit` (`20`);
11. starts the intro and yields a `WaitForSeconds` of `0.1f`.

Standard and Advanced both route through the zero-valued game-mode selection
path, which uses `Random.Range`. Roguelike (`10`) instead clamps the current
level to the last script index. This native path has no local empty-script-list
guard.

On the second `MoveNext`, the iterator:

1. changes state to Day;
2. invokes `OnRestartPlayerInfo`;
3. updates rules;
4. calls `Characters.Init`;
5. schedules `DelayedDeckIntro` for Standard or Advanced when
   `currentDay != 0`;
6. runs relic trigger `AfterCharactersInit` (`30`); and
7. updates the gameplay UI.

This setup method does not directly clear `CurrentCharacters`, the saved
faction lists, the Roguelike deck, relic inventory, score, `currentLevel`, or
`currentDay`.

## Board-pool selection

`Characters.Init` first clears its current character list and disables every
configured board pool. It then scans all pools. Each pool whose placeholder
count equals the requested roster count is enabled, assigned as `currentPool`,
and copied into the working character list. The scan does not break after a
match, so duplicate-sized pools are all enabled and the final matching pool
wins. If no pool matches, the old working list has already been cleared, every
pool remains disabled, and `currentPool` is left stale. The function still
tail-calls `ManageCharacters`.

## Character construction and publication

`Characters.ManageCharacters` orders board construction as follows:

1. update board positions;
2. choose round bluffs;
3. choose round duplicates;
4. pair board cards with the supplied roster and assign descending IDs from
   roster count to one;
5. call `Character.Init` for each pair;
6. call `Gameplay.UpdateCharacters`, which publishes a new shallow `List` copy
   of the board-card references;
7. call `Character.Act(Init)` (`3`) on every card;
8. run `Character.Act(Start)` (`5`) for every card whose data equals an entry in
   `startGameActOrder`, using Unity object equality;
9. invoke `onSetup`; and
10. shuffle the deck.

The ordinary `Init` pass therefore precedes all ordered `Start` actions, and
the global character list is visible before either action pass. Every matching
entry in `startGameActOrder` is processed, so duplicate entries can trigger
multiple starts.

## Per-card initialization and internal reveal

`Character.Init` clears trailer text, acted UI, acted-info and runtime-data
collections, destroys an earlier dead-character prefab, clears bluff and
register-as data, assigns the new `CharacterData`, resets reveal/death/activity
flags, initializes alignment and—unless the supplied ID is `-100`—stores the
new ID. It records the old state as `prevState`, changes state to Hidden,
invokes the state-change callback, and only then clears the status collection
and refreshes the card.

The method does not directly reset `killedHidden`, resistance collections, or
`bluffRole`. The `-100` sentinel preserves the old ID.

`Character.DelayReveal.MoveNext` clones `dataRef.role` before yielding
`0.3f`. On resumption it calls `Character.Reveal`.

That internal `Reveal` routine is setup/presentation work, not the player's
face-down-card flip. It resolves register-as data and the card's bluff, allows
`HealthyBluff` to run a `Start` action early, always runs `Init` and
`AfterRoundStart`, presents either real or bluff information, and refreshes the
card. It does not itself change the card state, update reveal order, or invoke
`onReveal`.

## Click dispatcher

`Character.OnClick` returns immediately during Night. In other phases it invokes
the generic click callback before choosing a branch.

During `PickCharacters`, it delegates to the active picker, updates picker
indicators, and returns before checking `killedByDemon`. Outside picker mode, a
demon-killed card returns without further work.

For a non-killing click on a hidden card, the card becomes Alive only when the
hidden-card count exceeds the current block value. Otherwise it is merely made
visible. A revealed active card with pickable uses remaining can run its Day
action (`30`).

During Killing, the role first checks whether the card can be killed. A
protected result logs the attempt, runs the Protected action (`60`), and
restores `PrevState`. A successful ordinary execution performs this exact
sequence:

1. reveal all real information;
2. call `Kill`;
3. invoke `onReveal`;
4. run the Executed action (`40`);
5. run the Died action (`50`);
6. restore `PrevState`; and
7. assign the card's order from `Gameplay.CurrentReveal`.

The `uninteractable` field is not consulted in this native dispatcher, and a
successful execution does not call the internal setup `Reveal` routine.

## Kill and hidden-order accounting

`Character.Kill` is a no-op for an already-dead card. Otherwise it records
whether the old state was Hidden, copies the old state into `prevState`, changes
state to Dead, invokes the state-change callback, refreshes UI, calls the role's
`ActOnDied`, invokes `OnCharacterKilled`, and snapshots
`Gameplay.CurrentReveal` into the card's order.

`Gameplay.IncreaseOrderCountOnHiddenKill` is a normal kill-event subscriber. It
increments `CurrentReveal` synchronously when the killed card's `prevState` was
Hidden, so the subsequent snapshot inside `Kill` observes the increment. A
caller such as the ordinary execution path can later overwrite that order.

The role-level `ActOnDied` call and later `Character.Act(Died)` dispatch are
distinct surfaces. The role validation happens after state and UI mutation, so
a missing role can leave a partially killed card before the native guard fails.

## Native-static anchors

The first slice was checked against these native entry points:

| Method | RVA |
| --- | ---: |
| `Gameplay.HandOut` | `0x37DCD0` |
| `Gameplay.SetupDelay.MoveNext` | `0x390AB0` |
| `Characters.Init` | `0x36CC40` |
| `Characters.ManageCharacters` | `0x36CE30` |
| `Character.Init` | `0x365A20` |
| `Gameplay.UpdateCharacters` | `0x3811B0` |
| `Character.DelayReveal.MoveNext` | `0x3756B0` |
| `Character.OnClick` | `0x366270` |
| `Character.Reveal` | `0x368410` |
| `Character.Kill` | `0x366130` |
| `Gameplay.IncreaseOrderCountOnHiddenKill` | `0x37DE30` |

## Remaining lifecycle targets

The next native audit should close the other seventeen target entries:

- outer initialization and loading: `Gameplay.Init`,
  `Gameplay.InitCoroutine.MoveNext`, and `Gameplay.LoadCharacters`;
- setup helpers: `Gameplay.TriggerRelics` and `Gameplay.UpdateRules`;
- reveal callbacks and concrete kill variants: `Character.OnReveal`,
  `Gameplay.OnCharacterReveal`, `Character.RevealAllReal`,
  `Character.ExecuteAndReveal`, `Character.KillAndReveal`,
  `Character.KillProtected`, `Character.KillByDemon`, and
  `Character.DelayedDemonKill.MoveNext`;
- dead-character bookkeeping: `Gameplay.ManageKilledCharacter`; and
- night flow: `NightPhase.ManagePhase`, `NightPhase.StartPhase.MoveNext`, and
  `NightPhase.ReorderList`.

`Gameplay.ManageKilledCharacter` is the only target in the complete baseline
export carrying a decompiler warning: an unrecovered jump table. Its control
flow needs explicit native inspection rather than a managed-only promotion.
