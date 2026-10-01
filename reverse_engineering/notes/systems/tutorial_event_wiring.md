# Tutorial controller event registration

Pinned build: `f530404b0f3f_807de4a83df4`. The offline fixture executes the
complete native TutorialsController.OnEnable (`0x38D4A0`, `tdi5650.m0000`) and
OnDisable (`0x38CDC0`, `tdi5650.m0001`) callers. It audits **registration only**;
no event handler is dispatched, no game runs and no Unity object is activated.

GameplayEvents is exactly TypeDefIndex 5519, a static class with 29 declared
Action fields. Both callers visit these seven subscriptions in the same order:

| Exact event field | Static offset | Exact TutorialsController handler | Handler method ID |
| --- | --- | --- | --- |
| OnGameStart | `0x0` | StartTutorials | `tdi5650.m0011` |
| OnCharacterRevealed | `0x50` | OnCharacterReveal | `tdi5650.m0010` |
| OnCharacterInfoRevealed | `0x58` | CharacterInfoNote | `tdi5650.m0012` |
| OnCharacterKilled | `0x48` | CharacterKilledTutorial | `tdi5650.m0006` |
| OnShowTutorial | `0x90` | EnableTutorial | `tdi5650.m0009` |
| OnCloseTutorial | `0x98` | CloseTutorialIfAble | `tdi5650.m0013` |
| OnStartNewLevel | `0x28` | LevelIdTutorial | `tdi5650.m0003` |

The handlers use plain Action, Action<Character>, Action<ETutorialType,
Transform> and Action<ETutorialType>, according to the exact event and metadata
slots. Folded constructor symbol aliases do not determine those types. The
report pins each metadata type/MethodInfo slot and the exact handler signatures
and entry RVAs. It also decodes and verifies all seven constructor-call sites
and Combine/Remove call sites in each caller.

## Supplied delegate boundary

The callers natively allocate and construct a fresh delegate for every field,
using the exact controller receiver and handler MethodInfo. Allocation and
constructor runtime services return authored physical tokens. The constructor
service records the verified target, handler and delegate type; the caller
then invokes Combine on enable or Remove on disable.

Combine and Remove are explicit supplied services. Their ordered invocation
policy appends the new handler and removes the last matching receiver/handler
entry respectively. Absent removal retains the existing pointer; removing the
only handler returns null. Existing prior/trailing foreign handlers retain
their order and receiver identities. This policy is not execution evidence for
the managed runtime multicast implementations.

The native callers write each returned pointer into the exact static field and
then request its write barrier. All other 22 event fields retain their authored
sentinel pointers. The fixture verifies persistence remains unused and records
all delegate allocations even when a Remove operation finds no matching
subscription. Repeated enable/disable sequences preserve the physical tokens
and ordered invocations between calls rather than rebuilding independent
inputs for each invocation.

## Cast and partial-publication behavior

The two **plain Action** event fields use exact native class-header equality
checks. Generic Action fields call an explicit cast helper twice, around the
static-field publication. Compatible foreign-header inputs exercise only that
generic service path; applying them indiscriminately to plain Action would
reach a native exact-type failure instead.

One probe supplies the wrong plain Action header and reaches that native
failure. Generic probes fail the first cast and the second cast separately.
The second-cast failure retains the already stored OnCharacterRevealed pointer
while its write barrier has not occurred. Earlier OnGameStart publication is
also retained. The audit does not roll back any published field or allocated
delegate on a later stop.

Cold/warm metadata flags are independent for enable and disable. The report
retains their publication in every service-entry snapshot, along with supplied
runtime class state. Metadata/runtime services and valid class/static pointers
are authored boundaries; toggling their flag bytes is not a claim about a real
Unity runtime initialization sequence.

## Evidence and bounds

The executable fixture is
[`audit_tutorial_event_wiring.py`](../../scripts/audit_tutorial_event_wiring.py)
and the values-only report is
[`f530404b0f3f_807de4a83df4_tutorial_event_wiring.json`](../../reports/f530404b0f3f_807de4a83df4_tutorial_event_wiring.json).
It contains **228 cases, 100 exact controlled stops, 28 pinned constructor and
Combine/Remove call-site relationships, and 660 native caller/service execution
addresses**. Independent repo/private runs were byte-identical (13,405,317
bytes; SHA-256
`dd2eeeb6932628d2dda6cabe5c2cd3567b997e8b672aaebacca137116e6c40ba`).
All 32 reverse-engineering tests, Python compilation and whitespace checks
passed.
Each normal case checks all seven expected final invocation lists; retained
step snapshots show the individual calls in repeated sequences. Controlled
stops compare the exact event prefix and complete projected registration/save
state against the baseline service-entry snapshot before compact output omits
repeated event snapshots.

The cases cover prior and trailing foreign handlers, preexisting own handlers,
absent removals, duplicate enables, repeated disables, cold/warm metadata,
exact-header versus supplied-cast objects, and native cast-failure publication
boundaries. Runtime allocation, metadata, delegate construction, Combine/Remove,
casts and write barriers remain explicit services. No controller gameplay
callback, delegate invocation, Unity scheduling, actual registry operation or
native exception unwinding is inferred.

Physical token allocation stays inside the authored arena with an explicit
upper bound. The longest matrix sequence has three calls. Each call retains the
100,000-instruction cap and the existing joined caller's bounded 120-second
wall budget. Native bytes and bodies remain outside the repository.

The separately verified
[tutorial persistence join](tutorial_persistence_join.md) reconstructs the
EnableTutorial target's normal caller chain into Note.Show, AddTutorial and
Save; it does not establish automatic dispatch from these event registrations.
