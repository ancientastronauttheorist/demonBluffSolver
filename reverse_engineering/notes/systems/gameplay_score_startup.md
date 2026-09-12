# Gameplay startup and Score selection

Build `f530404b0f3f_807de4a83df4`. The supplemental `scripts/audit_gameplay_score_startup.py` pins GameAssembly, Dumper script and dump hashes, verifies eight exact Gameplay declarations and eight callback method identities, and executes complete native bodies in Unicorn 2.1.4. The report `reports/f530404b0f3f_807de4a83df4_gameplay_score_startup.json` has 246 passing cases and 1,247 executed native instruction addresses. No new target manifest is needed.

## Concrete Score selection

Gameplay.Init (37DEF0) and RestartGame (37FFC0) both unconditionally allocate the registered ScoreOld type. Native inline constructor writes set pointPerKill=50, pointsPerUnrevealed=10 and pointsForCompleting=100 before the object-constructor gateway; the object is then published to Gameplay.Score (+8). The object allocator supplies zero-initialized remaining fields. Neither path selects ScoreNew or base Score based on the game mode.

Init publishes ScoreOld before it asks the mode for starting levels. RestartGame asks for the reset level before allocating/publishing ScoreOld. Both query GetGameMode later, only to decide whether to call LoadCharacters for the zero result. Thus a mode-null failure in Init retains a newly published ScoreOld, while a mode-null failure in RestartGame occurs before replacement. Constructor failure leaves the previous global Score intact while the unpublished allocation already contains its three default writes; report snapshots retain both identities.

The bounded reference inventory decodes all 55 distinct Gameplay-declared method entries. ScoreOld_TypeInfo is referenced by only Init and RestartGame in that inventory: two metadata-warming LEAs and two allocation-type loads, recorded separately. This does not prove the absence of all indirect writers, external reflection writes, or writes outside Gameplay. It does establish these two ordinary initialization/restart paths' concrete selection.

## Lifecycle responsibilities

| Method | RVA | Tested responsibility |
|---|---|---|
| .cctor | 381990 | Allocates empty CurrentRelics, CurrentCharacters and DeadCharacters lists; writes GameplayState, PrevState and CurrentReveal to zero. Preserves Score, Instance and CurrentScript. |
| .ctor | 381B00 | Allocates nine CharacterData lists at instance +20..+60 and a SpecialRule list at +88, then calls the MonoBehaviour constructor gateway. Does not write currentLevel, currentDay or startingLevel. |
| Awake | 37B5B0 | Initializes the Gameplay runtime class if needed and replaces static Instance with this receiver, including a previously distinct instance. |
| Start | 380F20 | Appends this receiver's Gameplay.Init to GameEvents.OnGameInit (+18). Does not invoke it in this body. |
| OnEnable | 37F520 | Appends eight handlers in the order below. |
| OnDisable | 37EDB0 | Removes the last matching occurrence of each of those same eight handlers in the same order. |
| Init | 37DEF0 | Resets saved-character state through a gateway, requests gameplay state 1, clears the deck/relic lists, copies four saved faction lists, publishes ScoreOld, obtains two starting-level results, conditionally loads characters and schedules InitCoroutine. |
| RestartGame | 37FFC0 | Resets player information, resets health through its virtual gateway, copies four saved faction lists, obtains reset level, publishes ScoreOld, clears currentDay, then conditionally loads characters. |

The two GetStartingLevel calls in Init are separate: the first writes currentLevel and the second writes startingLevel. Fixtures deliberately supply different values and signed extrema. RestartGame preserves startingLevel. Init's list clears write size zero and increment version (including wrap) before the array-clear gateway; failures preserve that prefix. The four copied lists replace the current faction-list identities in order. Resets and list-copy behavior remain explicit gateways rather than a claim to execute their complete implementations.

## Callback identity and order

OnEnable/OnDisable operate on GameplayEvents:

| Event | Handler |
|---|---|
| OnNewHandOut (+78) | HandOut |
| OnSameHandOut (+88) | SameHandOut |
| OnRestartGame (+38) | RestartGame |
| OnCharacterRevealed (+50) | OnCharacterReveal |
| OnCharacterKilled (+48) | IncreaseOrderCountOnHiddenKill |
| OnCharacterKilled (+48) | ManageKilledCharacter |
| OnNextChallenge (+18) | UpdateScore |
| OnGameStart (+0) | SameHandOut |

Each delegate targets the supplied Gameplay receiver and exact pinned MethodInfo identity. Duplicate and empty initial invocation lists test append order and last-match removal, including the two distinct handlers sharing CharacterKilled. Start uses GameEvents.OnGameInit, which is a separate static event collection; OnDisable does not remove that Start registration in its audited body. This statement does not infer broader object destruction or event cleanup behavior.

Every reachable gateway occurrence has an injected stop fixture compared with the successful execution's exact event/state prefix. Independently asserted successful outcomes establish publication, constructor values, handler identities/order, allowed reference/counter changes and list-header preservation. Wrong delegate result types at each of 17 registration/removal positions execute native cast failure paths and retain only earlier event stores. Interned snapshot_table entries retain reference labels, counters, invocation lists, list headers and unpublished Score defaults.

## Boundary

The actual eight native caller bodies execute; object/list/delegate allocation and construction, Combine/Remove, casts, GC barriers, class initializers, ResetSavedCharacters, ResetPlayerInfo, health reset, ChangeGameplayState, mode selectors, LoadCharacters and coroutine scheduling remain controlled gateways. In particular, class-initialization gateways do not automatically run the separate .cctor fixture, and registered callback bodies are not invoked by a fabricated Unity schedule. No complete Awake/OnEnable/Start invocation order, live event history, reset-helper effect, actual coroutine scheduling, or native exception unwinding is inferred.

Reproduce with the script's game-root and Dumper-root positional arguments plus `--output`, using the private python-emulation PYTHONPATH. Python compilation passes. Native code and live saves remain private.
