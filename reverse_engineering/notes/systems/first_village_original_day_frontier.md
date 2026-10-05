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

## Fresh pool and rule bindings for the next composition

2026-10-04. Further independent static review resolves the minimum fresh-board
and rule bindings. This remains a caller-dependency triage, with no newly
executed SetupDelay history or public-domain promotion.

The original Characters instance `137026` has an empty working list, null
currentPool and seven configured pools. Characters.Init calls
CreateAndGetCharacters at `36CDA8`, then tail-dispatches Manage at `36CE11`.
Supplying a prefilled working board does not establish this path. The unique
five-placeholder pool is `139114`; its prefab reference `[2,23406]` resolves
to Character. Its ordered original references are:

| Placeholder | Transform | Acted side |
| --- | --- | --- |
| 185364 | 98170 | Down `30` |
| 177982 | 103254 | Left `20` |
| 177993 | 103260 | Left `20` |
| 177992 | 103261 | Right `40` |
| 177956 | 103286 | Right `40` |

All five serialized transforms have empty child lists; runtime-empty child
enumerators still require an explicit service contract. The native creator
`3698F0..369CE7` first enumerates and destroys every placeholder's children,
then performs a separate instantiation pass. Its call at `369BAD` returns
Character directly with prefab/parent/exact generic MethodInfo in RCX/RDX/R8;
there is no intervening GetComponent call. The empty-child profile would reach
ten transform lookups, five false enumerations and disposals, zero Current or
Destroy calls, five instantiations/appends and one ToArray. Those are branch
expectations for the unexecuted composition, not measured native service counts.

Native placement selects actor `+A8` from its Down `+C8`, Left `+B8` or Right
`+D0` directional Acted reference. Right also writes leftAct `+B0=true`.
Directional component references, clone fields, activation and lifecycle remain
supplied runtime inputs. The finite factory result must be its actual-written
pool `+30` array containing five distinct, fresh unpublished actors in
placeholder order. Before that final store, a stopped caller preserves the old
array pointer; a later barrier failure preserves the new pointer.

The [asset audit](character_asset_flags.md) binds the selected role classes:
Minion `5910`, Confessor `5894`, Enlightened/Shugenja `5890`, Hunter/Tracker
`5891` and Lover/Empath `5863`. Each inherits Role `5853` and declares no
GetRules override. Native base GetRules `3C4D10..3C4D65` allocates and constructs
a fresh empty SpecialRule list without reading a role-instance payload field.
Thus actual UpdateRules can execute under narrow list allocation/construction
providers rather than an arbitrary empty-rule callback.

UpdateRules loads virtual code/MethodInfo from class `+178/+180`, calls the
first getter at `381748`, checks null at `38174A/38174D`, then checks list
count at `381753/381757`. Only a nonnull, nonempty result reaches the second
getter at `381781` and AddRange at `381799`. The declared five-role empty-list
path therefore needs five fresh getter results and zero AddRange calls. Its
enumerator is copied from a hidden return buffer to the live stack cursor;
do not retain the overwritten constructor temporary as that cursor's identity.

The Standard-enum script draw at `390F2E` uses zero minimum and the actual
retained count-list size as maximum, followed by get_Item `390F48` and
CharactersCount CreateCopy `390F57`. These metadata arguments are resolved;
the recorded schedule must account for this draw before roster draws.
Gameplay static `+0` is CurrentRelics and `+10` is Instance. Existing harness
bootstrap aliases must not conflate them. Without executing Gameplay.Init,
day zero remains a supplied entry value rather than original first-start proof.

Independent PE/metadata checks confirm these complete body fingerprints:

| Body interval | SHA-256 |
| --- | --- |
| SameHandOut `380260..3802D0` | `0cfacae96397e8397f595a3bc3d1213cc46d973025fc84997d27c4a397a46d78` |
| SetupDelay.MoveNext `390AB0..39128C` | `e3bede06b2a0bac64219786f3a676d4f818f79098d24f29422a3a36a2f208d71` |
| ChangeGameplayState `37B620..37B74E` | `84f9c5ec23435606837fb16adeef4d908ae873b8add2a9049741e557c6e39b43` |
| Characters.Init `36CC40..36CE21` | `f9c1b63e1f42b2ad35ab40bc866f8a2266c9fea440cc14df618e4925177e0a62` |
| CreateAndGetCharacters `3698F0..369CE7` | `d21757c3705c056f5c005b103d85445187cc0365e7c83d2394496660eae00a95` |
| Role.GetRules `3C4D10..3C4D65` | `ea2eaf82bb232b78f1e6f4726f79a1938252ca74e73f249c28fe0bb7cb54f5cd` |

Root's separate readers confirm the four outer body identities, twelve selected
direct calls, five exact inherited role declarations and ten getter/branch pins.
An initial reader incorrectly assumed a memory-form virtual call; the corrected
reader follows the loaded RAX target and subsequent count check. These static
checks establish neither an executed branch nor a retained normal/stop result.
The original setup producer is still an unpublished, unexecuted draft. Null
event channels, empty relic inventory, concrete mode publication, engine
ownership/timing, prefab hydration and trusted public capture remain separate
contracts. All eight full-plan acceptance gates remain open.

## Executed original setup terminal join

2026-10-04. One normal original setup composition now completes under its
declared entry and runtime providers. This replaces the supplied outer
generation/Day sequence for the recorded row-zero N5 case with actual
SameHandOut, SetupDelay state zero, the first wait, and the same iterator's
state-one continuation through fresh Characters.Init/Manage.

The original first wait stores `.1f` and is admitted at supplied producer time
`1.0`, frame `7`. The engine resumes it at `1.125`, frame `8`, after its exact
deadline. Native transitions produce Intro `5`, then Day `10` / PrevState `5`.
Five empty base Role.GetRules calls, the selected CharactersCount copy, eighteen
recorded draws and the two-pass pool creator execute before five actual
Character.Init occurrences. SetupDelay returns terminal false with state
`FFFFFFFF`, retaining its generated-list and old Current identities.

Ten completed managed bridges are ordered Setup0, five acquisition first
waits, Audio, Animation, Shuffle and Setup1. Nine distinct iterators are
registered. Nineteen engine native calls complete with preserved parent
windows and outside-child stack hashes; all managed bridges satisfy their
return SP, nonvolatile-register and storage checks. Fifty emulator returns
record actual timeout flags of zero. Host timeout is disabled while the
instruction cap remains `500000` per call, with managed/engine handoff caps
`4096`/`1024`; supplied game time is separate.

Only the outer Setup payload is logically released. Eight child waits remain
pending with their actual owners, producer `1.125` / frame `8` and native
duration/deadline/frame fields. No child continuation is resumed in this case.
No borrowed payload, suspended parent, live native frame or managed driver
frame remains at normal exit. Pending records are part of the finite result.

The supplied fresh typed static DeadCharacters list receives only the native
version increment and zero-count store, preserving its other bytes. The
separate static CurrentCharacters field starts at zero and is subsequently
published by native Gameplay.UpdateCharacters from Manage. Its fresh list
identity differs from the working board, while both contain the same five
fresh actor pointers in order. The list-copy constructor remains a supplied
provider. This publication establishes neither modal deck nor pixel visibility.

Three failed native normal attempts preceded this result and remain private:
a missing DeadCharacters entry binding; an acquisition guard using the
iterator-constructor return rather than the inherited tail-dispatch caller;
and an undeclared emulator budget return whose timeout cause was not measured.
The corrected acquisition guard binds actual Manage return `36CFDF`, the live
incomplete Init record, actor, iterator, state and entry SP. Removing the host
timer and recording actual timeout queries did not weaken completion guards.
A review assertion also incorrectly froze CurrentCharacters at entry zero;
the corrected reader follows its actual native publication instead.

The private normal report is 124,972,481 bytes, SHA-256
`ec16c76242ab97bcf72b153f1f700675c6bd4c0f6274f49ab4ca7a8fcae7c0f8`.
The reviewed 91,003-byte source has SHA-256
`fede8a925c9e587673d36f9f41f406c542ace59d151c03b7563fcfe2dad541cd`.
Static preflight verifies forty-five inputs, eleven complete body fingerprints,
forty selected operands, seven pools and eighteen draw entries. Independent
normal readers authenticate 519 declared snapshot blobs and all input hashes,
then check chronology, bridge/frame ABI, collection publication and retained
wait ownership. Fresh failure output is absent on success. No fresh stopped,
pause/reentry, mutation, held-out or outcome campaign accompanies this case.

The entry day, mode, empty relics/rules and null event channels remain explicit
supplied contracts. Gameplay.Init, full first-start lifecycle, Unity cloning,
rendering/input, concrete nonempty callbacks and resources are excluded. No
legal public observation history, complete possible-world set, generation
weights or policy certificate is promoted. Production Rust is unchanged and
its earlier suites were not rerun. Proprietary bytes, full CPU/storage, reports
and producer sources remain private; all eight full-plan gates remain open.

The next end-to-end boundary is to resume these same pending acquisition and
animation continuations, complete deck publication and bind a trusted public
Hunter prefix. Preserve their new producer clocks rather than reuse timings
from a separately supplied acquired checkpoint. Mixed N5 history admission,
independent S2 worlds and legal S3 policy still require their own evidence.

## Acquisition continuation entry requirements

2026-10-04. Independent native operand and source review narrows the next
setup-to-observation blocker. The pending waits can advance chronologically
without admitting Audio or Shuffle early, but acquisition needs a stronger
declared entry graph than the terminal setup witness supplies.

The native consumer admits time and signed-frame equality: its time comparison
rejects sampled time below the deadline, and its frame comparison rejects a
threshold above the sampled frame. The proposed next case drains each actual
minimum Animation record at frames `9` through `13`, using the deadline written
by the preceding native producer. The subsequent acquisition deadline is
`1.425000011920929`, admitted at frame `14`; Audio and Shuffle remain future
at `1.5250000059604645` and `1.625`. These are decoded, conditional schedule
expectations, not measured continuation results. Jumping directly to the
acquisition deadline would change later Animation producer times.

Character.Reveal requests Init `3` and AfterRoundStart `7`; optional Start `5`
requires HealthyBluff status `30`, absent here. Tracker, Empath and Shugenja
Act/BluffAct accept Day trigger `30`, so acquisition triggers do not produce
their clues. Confessor handles Init through its actual OnInit and returns for
trigger `7`; Minion Act is inert. Global Day `10` does not itself authorize a
clue callback. The existing unexpected-onActed rejection must remain in force
for this acquisition slice; no supplied Day/PrevState rewrite is needed.

The original setup fixture provides directional, number, status, pickable and
animation fields. It does not establish the six Character presentation
references needed by the declared RevealReal/SetupArt/UpdateViewReal paths.
The artBg reference is conditional on backgroundArt; borders must be a
nonnull array even when the declared array is empty. A Python-only
pre-native graph comparison finds thirty absent actor references, twenty
absent selector code/context fields and missing source presentation colors
and name contents. The asset parser supplies the original sprite references
and exact Color bits; runtime UI objects and Sprite resolution remain opaque
providers. The comparison executes no native constructor or iterator.

Selected CharacterData, source-role, source-class and hierarchy regions are
already frozen in the setup witness. Adding those bindings after its native
Init and recapturing guard baselines would invalidate retention evidence.
The next runner therefore creates a separate fixture, hydrates and seals its
complete declared presentation/selector graph before any Character.ctor or
Init, and then executes the original outer chronology and continuations.
The existing setup source/report stays immutable. This stronger entry domain
cannot claim byte-exact reproduction of the earlier raw prefix.

Cross-fixture checks must declare all entry differences and use an injective
typed identity map grounded in assets and producer occurrences. Compare the
complete unaffected role/status/list-occurrence/setup/scheduler projection;
do not normalize arbitrary pointer-like integers or erase differing services
to manufacture full-trace equality. Each new run independently enforces raw
CPU/ABI, parent-window/outside-stack and immutable-source guards.

The proposed finite exit is actual terminal Animation and five acquisition
callbacks, preserving Day `10` / PrevState `5`, CurrentReveal zero, Hidden
actors, uses one and empty information histories. Only the original Audio and
Shuffle waits should remain. This establishes an acquisition prerequisite,
not legal Hunter click readiness: input delegates, resources, backside state,
later UI publication and trusted capture still require their own evidence.

The two reviewed archived asc84 captures have exact recorded hashes but lack
capture-time build binding, complete deck occurrences and serial reveal
chronology. They cannot authenticate this N5 prefix. Native text/publication
bindings establish routing; exact-prefix original capture review establishes
public availability without requiring Unity renderer reconstruction.
No new native continuation, public history, world/policy promotion or Rust
change accompanies this triage. All eight full-plan gates remain open.

## Original acquisition normal continuation

2026-10-04. One separate, preentry-hydrated fixture now executes the original
setup chronology through terminal Animation and all five acquisition iterator resumes.
The original setup v3 source/report remains unchanged; this case retains its
own native actors, iterators, CPU and queue throughout. Its private report is
`297645897` bytes, SHA
`82a4e5f89aae7cf461b6a56be45ec5333b7950d2a2ed86ca94e2aaa3b8609ab2`.
The frozen runner SHA is
`5dd3233414b116a8f3c9caec5ccaa2a50ac43eb982241f746cf8712ae388cc2d`;
the independent typed projection helper SHA is
`f2d427bfebcd64628b365d5d2997399140853c0478216c707f44ae6133d28515`.

The producer and independent JSON reader both complete with terminal code zero.
The reader authenticates all 48 inputs and 750 lossless snapshot blobs, and
independently compares the actual new setup prefix against v3: 19 checkpoints,
204 injective typed identities and 1532 typed paths agree, projection digest
`3546dd763131128a64e1ce0bebd3be6a6955a0a05a3592c5deb5d448c1994061`.
This comparison covers unaffected semantics and provenance; it does not claim
cross-fixture raw CPU, allocation, service or presentation equality.

A separate causal reader also completes with terminal code zero. It joins
both supplied selector draws to the two actual `acquisition_rng` services:
`[1,11) -> 1` follows the duplicate branch, then `[0,4) -> 0` selects Confessor.
One field-faithful Confessor clone is retained; no public roster registration
or pool mutation is inferred. Ten Character.Act trigger rows, ordered Init 3
then AfterRoundStart 7 per actor, produce twelve RoleAct/delegate constructor
rows: the Minion dispatches both real and copied roles for each trigger. The
reader checks the real and copied Confessor OnInit paths and the copied actor's
single status-25 addition, rather than inferring the path from final status.
All 76 recorded emulator returns report integer `UC_QUERY_TIMEOUT=0`: 31
managed `drive_managed` and 45 engine `run_native`, retaining the exact prefix
return sequence. These are measured return flags, not a pause/stop certificate
or a policy-performance measurement.

Five successive earliest Animation deadlines execute at frames 9 through 13,
with native returns `[1,1,1,1,0]`. The sixth drain at
`1.425000011920929`, frame 14, executes five acquisition iterator resumes in actor
order, each returning false. Ten managed bridges pass recorded raw ABI, full
`0x2000` parent-window and outside-child guards. Six top-level engine frames
pass raw ABI and outside-child guards; they have no suspended parent window.
The nine registered payloads retain eight owner identities; seven are logically
released. Final generation is 7 and next identity is 13. Only original Audio
record 6 and Shuffle record 8 remain, with unchanged deadlines
`1.5250000059604645` and `1.625`, frame threshold 9, generation 1 and phase mask
10. The independent reader checks their exact retained records and live links.

The stronger entry has 110 declared hydration rows, 82 sealed source regions
and 48 sealed UI/string/array/sprite regions. The report exposes initial seal
bytes and hashes. Their final preservation is an authenticated assertion of
the frozen producer, not a separate comparison of serialized final regions;
those final raw regions are absent. No existing guard baseline is refreshed.

This closes the named original setup-to-acquisition dependency under explicit
UI, clone, selector, runtime and scheduler contracts. It does not establish
Hunter click readiness, public deck completeness, trusted capture, legal
history, world probabilities or policy quality. No pause/reentry, fresh stopped
prefix, held-out family or native mutation certificate is added by this normal
case. Production Rust is unchanged and all eight full-plan gates remain open.

The next solver-relevant exit is the same acquired Hunter's click/reveal and
text/history publication while preserving native Day 10 / PrevState 5 /
CurrentReveal 0. The older supplied Hunter checkpoint's PrevState 1 cannot be
transplanted. Input delegates and resource/backside predicates need explicit
producer-bound contracts; trusted same-prefix public evidence remains separate.
Existing click-path evidence has no queue-empty or Audio/Shuffle-completion
predicate. Preserve those waits rather than extending unrelated continuations
solely because they remain pending.
