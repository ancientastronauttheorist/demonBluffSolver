# Gameplay Lifecycle Native Audit

Build: `f530404b0f3f_807de4a83df4`

Evidence status: **native-static** for all 28 methods in the checked lifecycle
boundary. No statement here is based on live dynamic observation.

The checked target set is
[`reverse_engineering/targets/gameplay_lifecycle.json`](../../targets/gameplay_lifecycle.json).
Its baseline Ghidra export completed read-only at 28/28 functions with no shared
native bodies. The isolated typed project also validated and exported 28/28
after full analysis. Typed decompilation reduced unresolved-type tokens from
370 to 149 and raw field-offset accesses from 678 to 289. Decompiled bodies
remain in the private artifact tree.

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
the global character list is visible before either action pass. For ordinary
roles the first matching card ends that role's scan. Alchemist, Poisoner, and
Puzzlemaster are explicit exceptions: every card sharing their data receives
the ordered `Start` action.

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

## Initialization entry

`Gameplay.Init` orders a new run as follows:

1. call `ResetSavedCharacters` (its internals are a separate boundary);
2. request the Init gameplay state (`1`);
3. set `currentDay` to zero;
4. clear the Roguelike deck and global current-relic list;
5. clone each saved faction list into its corresponding current list;
6. replace the global score with a new `ScoreOld`, initialized with 100 points
   for completion, 50 per kill, and 10 per unrevealed card;
7. call the current mode's `GetStartingLevel` twice, storing the results as
   `currentLevel` and `startingLevel`;
8. call `LoadCharacters` when `GetGameMode` returns Standard (`0`); and
9. start `Gameplay.InitCoroutine`.

Advanced and Standard both return zero from `GetGameMode` in this build. The
state-change request precedes the day, relic, faction-list, and score resets, so
synchronous state-change subscribers can observe the old values. This method
does not directly clear `DeadCharacters`, `CurrentReveal`, or `specialRules`.

`Gameplay.InitCoroutine.MoveNext` first yields null once. On resumption it
invokes `OnGameInit`, re-reads the current game mode, and then either invokes
`OnGameStart` for a zero-valued mode or requests the Map state (`70`) for a
nonzero mode. Because the mode read follows `OnGameInit`, that callback can
affect the branch.

## Character loading

`Gameplay.LoadCharacters` first tests `ProjectContext.Instance` with Unity
object equality and returns for a null or destroyed instance. Otherwise it:

1. copies `GameData.GetAllCharactersData` into a temporary list;
2. filters it in source order by `unlockedCharactersId.Contains(characterId)`;
3. appends each remaining entry through `Gameplay.Instance` according to its
   exact faction value: Villager (`10`), Outcast (`20`), Minion (`30`), or
   Demon (`100`).

Other faction values are ignored. The method neither clears nor deduplicates
the destination lists, so these unlocked roles append to the saved-list clones
created by `Init`. The passed receiver is unused; all writes use the global
instance. Only the initial project-context check is a graceful null path.

## Relic and special-rule dispatch

`Gameplay.TriggerRelics` walks `CurrentRelics` in list order and calls every
relic's virtual `Act(trigger)`. The dispatcher performs no trigger filtering.

`Gameplay.UpdateRules` first calls `Remove` on every existing rule, leaving the
old list intact until all removals finish, and then clears it. It next walks the
new roster in order. For each role it calls `GetRules`; when that first result
is nonempty it calls `GetRules` a second time and appends the second result with
`AddRange`. It then calls `Init` on every aggregated rule and finally invokes
`OnSpecialRulesInit`.

Rule order is roster order followed by each role's returned-list order, with no
deduplication. The two `GetRules` calls for a nonempty result are native fact;
static evidence does not assume that both calls return the same list. If the
new roster is null, old rules have already been removed and cleared before the
failure. An empty roster still reaches the final event.

## Normal reveal accounting

`Character.OnReveal` invokes `GameplayEvents.OnCharacterRevealed` and, after
the complete multicast delegate returns, copies `Gameplay.CurrentReveal` into
the card's order. `Gameplay.OnCharacterReveal` simply increments
`CurrentReveal`.

`Gameplay.OnEnable` normally subscribes that increment handler to the reveal
event. Thus the configured path increments the counter before `Character`
snapshots it, while other subscribers can run before or after the increment
according to registration order. This method is distinct from the `onReveal`
UI delegate used by forced reveal helpers.

## Forced real reveal and kill helpers

`Character.RevealAllReal` sets `revealed`, displays the uppercase real role
name, appends Corrupted (`10`) and Mad (`20`) status labels when present, copies
the real color and art, and updates the real view. It does not change state or
`prevState`, invoke either reveal delegate, or update reveal order. Because
`revealed` is stored first, a later missing dependency can leave a partial
visual reveal.

`Character.ExecuteAndReveal` performs the same forced-execution sequence used
by the normal click path:

1. `RevealAllReal`;
2. `Kill`;
3. invoke the `onReveal` UI delegate;
4. run Executed (`40`) and Died (`50`) actions;
5. request the global previous gameplay state; and
6. resnapshot `CurrentReveal` into order.

`Character.KillAndReveal` omits Executed and the final order resnapshot, but
otherwise follows the same sequence. Both helpers bypass
`Role.CheckIfCanBeKilled`. If the card is already Dead, the internal `Kill` is
a no-op while the reveal delegate, actions, and state restoration still run.

`Character.KillProtected` only logs, runs Protected (`60`), and requests the
global previous gameplay state. It performs no protection check and changes no
card state, reveal flag, order, or status. Native `Character.OnClick` contains
the execution and protection sequences directly rather than calling these
helper symbols.

## Demon kill delay

`Character.KillByDemon` immediately invokes the demon-picked-character VFX
event, captures the victim and evil source in an iterator, and starts the
coroutine on the victim. It performs no immediate state mutation or killability
check, so the VFX also fires for an already-dead or protected target.

`Character.DelayedDemonKill.MoveNext` yields `0.45f` and then:

1. exits silently if the victim is already Dead;
2. asks the current real role whether the victim can be killed, also exiting
   silently when it cannot;
3. preserves the old state, sets `killedByDemon`, and changes state to Dead;
4. invokes the state callback;
5. adds MessedUpByEvil (`50`) and KilledByEvil (`55`), both sourced from the
   captured evil card;
6. invokes the UI update;
7. runs the card's Died action (`50`) and then the role's `ActOnDied`;
8. invokes `OnCharacterKilled`; and
9. calls `Character.UpdateUI`, which snapshots `CurrentReveal`.

This path does not call `Character.Kill`, `RevealAllReal`, or `onReveal`, and it
does not restore gameplay state. It also does not directly set `killedHidden`,
although the normal hidden-kill event subscriber still sees the preserved
Hidden `prevState`. Its card-action/role-action order is the reverse of the
ordinary `Kill` plus caller path.

## Dead-character bookkeeping

`Gameplay.ManageKilledCharacter` awards score through the runtime score object
only when alignment is exactly Evil (`20`), appends every killed card to
`DeadCharacters`, invokes the UI update, and finally tail-dispatches
`OnCheckEndGameCondition`. It performs no duplicate check, removal from
`CurrentCharacters`, or direct state/order/counter mutation.

The baseline decompiler's unrecovered-jump-table warning is a false positive.
Raw x64 shows a normal delegate `invoke_impl` tail dispatch for the end-condition
event, not a switch. With the normal subscription order, hidden-order increment
runs before this score/list/UI/end-condition handler. A normal kill therefore
emits one UI update before the role death action and another after the dead-list
append.

## Night flow

`NightPhase.ManagePhase` returns unless the global state is Night (`20`). In
Night it logs and starts a new `StartPhase` coroutine. There is no saved handle,
overlap guard, cancellation, or later state recheck.

`NightPhase.ReorderList` clears its private list, selects every global character
whose `dataRef` occurs in `nightCharactersOrder`, then stably sorts those cards
by the first `IndexOf(dataRef)` in the order list. Selection ignores card ID,
circle position, reveal order, alignment, and state, so Dead or Hidden cards are
included when their data appears in the order. Duplicate order entries do not
duplicate cards; multiple cards sharing a data object keep global-list order
within their equal key.

`NightPhase.StartPhase.MoveNext` has this exact timeline:

1. fade the night canvas to alpha `1.0` over `0.3f`, then wait `0.8f`;
2. rebuild the ordered list;
3. before each selected card, wait `0.4f`, then log its name and run its Night
   action (`20`);
4. after the list is exhausted, wait another `0.8f`;
5. request Day (`10`), set `NightPhase.currentRevealed` to zero, and start an
   unawaited `0.3f` fade to alpha `0.0`.

For `N` selected cards, the explicit waits total `1.6 + 0.4N` seconds. The Day
transition happens before resetting the NightPhase-local counter and before
fade-out. That counter is distinct from `Gameplay.CurrentReveal`. Once started,
the coroutine can force Day even if another path changed the phase while it was
waiting.

## Native-static anchors

| Method | RVA |
| --- | ---: |
| `Gameplay.Init` | `0x37DEF0` |
| `Gameplay.InitCoroutine.MoveNext` | `0x38FD10` |
| `Gameplay.LoadCharacters` | `0x37E240` |
| `Gameplay.HandOut` | `0x37DCD0` |
| `Gameplay.SetupDelay.MoveNext` | `0x390AB0` |
| `Gameplay.TriggerRelics` | `0x381050` |
| `Gameplay.UpdateRules` | `0x3814F0` |
| `Characters.Init` | `0x36CC40` |
| `Characters.ManageCharacters` | `0x36CE30` |
| `Character.Init` | `0x365A20` |
| `Gameplay.UpdateCharacters` | `0x3811B0` |
| `Character.DelayReveal.MoveNext` | `0x3756B0` |
| `Character.OnReveal` | `0x367500` |
| `Gameplay.OnCharacterReveal` | `0x37ED60` |
| `Character.OnClick` | `0x366270` |
| `Character.Reveal` | `0x368410` |
| `Character.RevealAllReal` | `0x367E80` |
| `Character.ExecuteAndReveal` | `0x364B50` |
| `Character.KillAndReveal` | `0x365F10` |
| `Character.KillProtected` | `0x366080` |
| `Character.Kill` | `0x366130` |
| `Character.KillByDemon` | `0x365FB0` |
| `Character.DelayedDemonKill.MoveNext` | `0x3757F0` |
| `Gameplay.IncreaseOrderCountOnHiddenKill` | `0x37DE30` |
| `Gameplay.ManageKilledCharacter` | `0x37EBD0` |
| `NightPhase.ManagePhase` | `0x383540` |
| `NightPhase.StartPhase.MoveNext` | `0x3927C0` |
| `NightPhase.ReorderList` | `0x3838D0` |

## Follow-up boundaries

The lifecycle boundary is closed natively, but several dispatched systems remain
separate work: `ResetSavedCharacters`, gameplay-state transition guards,
bluff/duplicate selection, `Character.Act`, concrete role/relic/rule/status
behavior, score overrides, shuffle/reveal animation, and event subscribers
outside the recovered Gameplay registrations.
