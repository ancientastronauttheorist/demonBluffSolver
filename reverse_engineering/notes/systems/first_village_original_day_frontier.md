# Original first-village Day chronology frontier

Pinned build `f530404b0f3f_807de4a83df4`. This is a static integration triage,
not a new executed native history, public-observation certificate, adapter,
world-set comparison or policy result. Existing native-static bodies and bounded
audits identify the next caller boundary; original Unity lifecycle and callback
composition remain open. All eight full-plan acceptance gates remain open.

## Phase publication precedes physical initialization

The [lifecycle audit](gameplay_lifecycle.md) and its exact
[target manifest](../../targets/gameplay_lifecycle.json) bind
`Gameplay.<SetupDelay>d__47.MoveNext` to `0x390AB0`. Existing private baseline
and typed lifecycle exports cover the complete body. No export or native bytes
are reproduced here.

State zero performs the generation/setup prefix, invokes relic triggers
PreRoundInit `10` and PreCharactersInit `20`, requests global Intro `5`, and
yields the `.1f` WaitForSeconds. It retains the generated CharacterData list.
State one writes iterator state `-1`, requests global Day `10`, invokes
GameplayEvents.OnRestartPlayerInfo, updates rules, and calls `Characters.Init`
`0x36CC40` with that retained list. Characters.Init selects/activates board pools
and tail-dispatches `ManageCharacters` `0x36CE30`. Its physical initialization,
publication, Init/ordered-Start, onSetup animation and Shuffle are the later
retained boundaries; they do not precede this Day request in the outer caller.

After Characters.Init returns, SetupDelay conditionally starts DelayedDeckIntro,
invokes AfterCharactersInit relic trigger `30`, requests the UI update, and
returns false. The proposed composition must verify that actual terminal return
and preserve the iterator's old Current reference rather than invent a cleared
Current or infer completion from an instruction budget.

Exact Dumper declarations are generated iterator TypeDefIndex `5603`, Gameplay
`5604`, and EGameplayState `5607`. The iterator has state at `+0x10`, Current at
`+0x18`, captured Gameplay at `+0x20`, and generated list at `+0x28`. Gameplay's
instance currentDay is `+0x7C`; its static GameplayState and PrevState are
`+0x28/+0x2C`. EGameplayState names Intro `5`, DeckIntro `8`, and Day `10`.
The exact dump contains no `DayChange` declaration used by this triage; the
resolved transition and event are ChangeGameplayState and OnGameplayStateChange.

## Six supplied Hunter checkpoint fields remain provenance obligations

The unpublished draft `first_village_retained_hunter_reveal.md` describes the
positive click/role/history/speech path on acquired actors. Its unpublished
draft source `audit_first_village_retained_hunter_reveal.py`, SHA-256
`3cc3ec153f52fe6bfa1e9b3432c88fd47ed47798c8c2a80e9d8362f06b19da24`,
explicitly writes six supplied checkpoint fields in `admit_supplied_day`.
The published [Hunter checkpoint](../../SOLVER_CONTRACT.md) records the bounded
result and remaining exclusions.
Their values are conditional inputs, even if a write happens to preserve a
previous value. Matching values cannot establish their original writer.

| Supplied field | Supplied value / obligation for an original join |
| --- | --- |
| Gameplay static GameplayState `+0x28` | Day `10`; execute the actual outer transition after its actual Intro request. |
| Gameplay static PrevState `+0x2C` | `1`; the ordinary uninterrupted Intro `5` to Day `10` transition instead records `5`. Preserve actual subscriber effects and derive the final value rather than reuse this supplied phase pair. |
| Character.onClick `+0x100` | Bound RevealCard.Reveal delegate; execute its original installer or retain an explicitly supplied lifecycle boundary and exact invocation list. |
| GameplayEvents.OnCharacterRevealed `+0x50` | Bound Gameplay.OnCharacterReveal delegate; establish Gameplay's actual registration and surviving event-list order. |
| PlayerController runtime class static-storage pointer `+0xB8` | Supplied resource graph; establish the relevant runtime/lifecycle and restart-player-info effects. Supplied mana `1` and blocks `0` remain separate from public HP. |
| Gameplay static CurrentReveal `+0x38` | `0`; execute SetupDelay's earlier reset and preserve subsequent writers. Do not reset it again at the acquired checkpoint merely to make a click fixture pass. |

Thus the PrevState mismatch is one concrete gap; the other five fields also
need their named original producer or an explicit retained supply contract.
The prototype additionally supplies UI/settings, active backside, RevealCard
initReveal false, null info-reveal callback, Tracker GetInfo virtual-slot
hydration, and tween/current-provider boundaries. These remain disclosed; the
six-field table does not imply that all other runtime inputs are original.

## Phase events and role Day actions are different edges

| Boundary | Exact provenance and scope |
| --- | --- |
| ChangeGameplayState `37B620` | [Core note](gameplay_core.md), [exact manifest](../../targets/gameplay_core.json), complete existing private typed body; native-static transition/dispatch proof, not a newly executed composition. |
| Character.OnEnable `366E40` | [Lifecycle note](character_event_lifecycle.md), [caller audit](../../scripts/audit_character_event_lifecycle.py); complete native installer under supplied delegate/runtime services, subscribed bodies excluded there. |
| RefreshCharacter `367970` | [Refresh note](character_refresh.md), [complete-body auditor](../../scripts/audit_character_refresh.py); separately executed UI/use caller, not role Day dispatch. |
| RevealCard.OnEnable `3866B0` | Exact Dumper ScriptMethod plus [Baker selected installer pins](../roles/gameplay_role_baker.md); no complete installer execution claimed by this frontier. |
| OnClick `366270` and Reveal `386A60` | [Baker note](../roles/gameplay_role_baker.md), [exact manifest](../../targets/gameplay_role_baker.json), complete existing private typed bodies; the unpublished retained Hunter auditor identified above executes the positive route with supplied handler/input bindings, as summarized in the published [checkpoint](../../SOLVER_CONTRACT.md). |

The native-static [core audit](gameplay_core.md), exact
[core manifest](../../targets/gameplay_core.json), and existing private typed
`Gameplay.ChangeGameplayState` body bind the transition to `0x37B620`.
It returns for an unchanged phase and has separate Draw guards. Otherwise it
stores old current phase into PrevState, stores the requested phase, and
dispatches GameEvents.OnGameplayStateChange when present. This body does not
directly invoke Character.Act.

The [Character event lifecycle audit](character_event_lifecycle.md) and
[executable script](../../scripts/audit_character_event_lifecycle.py) bind
Character.OnEnable `0x366E40` and its GameEvents.OnGameplayStateChange handler
to RefreshCharacter `0x367970`, with MethodInfo slot `0x270FE90`. This is a named
subscription proof, not a reconstruction of every scene subscriber. Delegate
operations and lifecycle invocation remain supplied in that audit.

The separate [RefreshCharacter audit](character_refresh.md) and
[script](../../scripts/audit_character_refresh.py) execute that complete body.
Its native operations deactivate picked indicators and, only when PrevState is
Night `20`, may reset uses and activate a picking control. It has no direct
Character.Act call. It does not publish role Day information when SetupDelay
requests Day. Other subscribers, their order, mutations and recursion require
their own retained contracts; no global absence claim follows.

The first Hidden-card Day action instead comes through the click delegate.
The native-static [Baker click/reveal evidence](../roles/gameplay_role_baker.md)
records selected RevealCard.OnEnable operands: load Character.onClick `+0x100`
at `0x386715`, construct the Reveal delegate at `0x386737`, and store the
combined delegate at `0x38676D`. Exact Dumper ScriptMethod metadata binds
RevealCard.OnEnable to `0x3866B0`. These selected installer pins are not a new
complete execution of its lifecycle body.

The [Baker target manifest](../../targets/gameplay_role_baker.json) and existing
private complete typed bodies bind Character.OnClick `0x366270` and
RevealCard.Reveal `0x386A60`. OnClick invokes the installed callback at
`0x366306`. Reveal runs the positive reveal predicate, calls Character.Act with
role Day trigger `30` at `0x386BA4`, then Character.OnReveal at `0x386BC2`.
Only after this synchronous callback returns do OnClick's later admitted-click
stores change previous/current character state at `0x36645A/0x366464`.
The unpublished retained Hunter source identified above and the published
[acquisition/publication audit](hunter_acquisition_publication.md) execute that
positive edge under supplied inputs while Hunter is still Hidden. The retained
draft result remains bounded by the published [checkpoint](../../SOLVER_CONTRACT.md).
Global Day `10`, role Day `30`, and relic AfterCharactersInit `30` are distinct
enum spaces and must not be conflated.

## First-start and delayed deck branches

The [startup callback audit](gameplay_score_startup.md) and
[script](../../scripts/audit_gameplay_score_startup.py) establish that Init
clears currentDay to zero and Gameplay.OnEnable binds OnGameStart to SameHandOut.
The [same-hand caller audit](gameplay_roster_reset.md) and
[script](../../scripts/audit_gameplay_roster_reset.py) bind SameHandOut
`0x380260`: it constructs/captures SetupDelay and forwards to StartCoroutine
without incrementing currentDay. HandOut `0x37DCD0` is a different caller and
increments currentLevel/currentDay before starting SetupDelay.

Under the explicitly composed Init to OnGameStart to SameHandOut first-start
path, currentDay stays zero, so SetupDelay's nonzero-day delayed-intro guard
is false. This branch must be proved from the retained path, not selected merely
because the fixture is described as a first village. For a Standard or Advanced
instance with nonzero currentDay, the caller starts DelayedDeckIntro after
Characters.Init. The two runtime-class tests reread the mode; the actual mode
identity and stable/mutating service contract must therefore be retained.

The [iterator factory audit](gameplay_iterator_factories.md) and
[script](../../scripts/audit_gameplay_iterator_factories.py) bind SetupDelay
factory `0x380C70`, DelayedDeckIntro factory `0x37BDE0`, and DelayedDeckIntro
MoveNext `0x38F9C0`. The latter yields exact float32 `1.0`, then reads BlindDeck
on resumption. Only integer `1` suppresses its request for DeckIntro `8`; every
other admitted value requests it without a local current-phase check.
StartCoroutine, actual engine admission/timing and ChangeGameplayState effects
are separate boundaries in that factory audit. A future nonzero-day join must
retain this continuation and its possible phase change, not silently suppress it.

## Smallest composition and finite exit

Start the actual SameHandOut construction/capture and SetupDelay state-zero
call under a declared runtime and mode boundary, retaining the actual generated
row-zero list through the
Intro request, `.1f` first yield, engine admission and state-one resume. Execute
actual ChangeGameplayState and named phase callbacks, OnRestartPlayerInfo,
UpdateRules, and Characters.Init into the existing retained Manage/actor/queue
composition. Complete the rest of state one, including its actual mode/day
branch, relic dispatch and UI notification. Compare exact pre-entry/final
storage, invocation chronology, queue identities and both active machine states.

The finite exit is SetupDelay's verified terminal false return and ABI
completion, with generated-list identity preserved, actual phase/PrevState
derived from the executed stores and callbacks, all Characters.Init/Manage
returns completed, and all resulting waits either completed or explicitly
retained with their owner/timing state. Do not require an invented empty queue.
Executing only state one with a supplied iterator/list could close a smaller
native Day-writer boundary; it would not prove state-zero provenance or the
original outer chronology. Re-entering Characters.Init after the acquired
checkpoint reruns initialization and cannot preserve that checkpoint by fiat.

Scene hydration and Awake/OnEnable/Start order, full event lists, Unity pool
activation, input/raycast/focus, concrete rule effects, resource resets and
rendered capture remain additional dependencies. The
[profile-generation audit](first_village_profile_generation.md) explicitly
schedules its generation calls and does not certify the whole SetupDelay caller;
retain consistent mode/profile identities rather than splice independently
supplied branches into an original history.

Relic trigger `0x381050` enumerates CurrentRelics and invokes each concrete relic
virtual action with the supplied trigger. Empty inventory needs explicit
provenance; nonempty entries and their callbacks cannot be replaced by inert
effects. UpdateRules `0x3814F0` and its concrete callbacks likewise require a
declared scope. The [relic/rule lookup audit](gameplay_relic_rules.md) does not
execute arbitrary relic/rule abilities. Global Day alone certifies neither the
positive reveal predicate nor public click availability. Complete modal deck,
trusted HUD/HP/capture and legal-history admission remain separate gates in the
[mixed N5 frontier](mixed_n5_observation_frontier.md).

Validation for this note is read-only source/metadata/export triage and resolved
repository-link checks. No new native producer, Rust implementation, live action,
proprietary instruction export or hidden input to PlayerHistory accompanies it.
