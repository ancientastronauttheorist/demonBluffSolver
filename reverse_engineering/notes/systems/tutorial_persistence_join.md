# Tutorial presentation and persistence caller join

Pinned build: `f530404b0f3f_807de4a83df4`. This offline audit executes the
game-owned callers below against authored managed records. The existing native
SavedGameInfo, SavedGameData, engine JSON, preference provider, setter/getter and
runtime-string fixtures supply the persistence composition. No live game,
actual Windows registry or Unity renderer is used.

| Exact metadata method | Native entry | Coverage identity |
| --- | --- | --- |
| TutorialsController.ResetTutorials | `0x38DF80` | `tdi5650.m0002` |
| TutorialNote.ResetTutorial | `0x38BD80` | `tdi5633.m0000` |
| TutorialsController.EnableTutorial | `0x38C950` | `tdi5650.m0009` |
| TutorialsController.ShowTutorial | `0x38E1A0` | `tdi5650.m0023` |
| TutorialsController.CheckIfAllTutorialsClosed | `0x38C430` | `tdi5650.m0026` |
| TutorialNote.Show | `0x38BE20` | `tdi5633.m0001` |
| TutorialQueue..ctor | `0x3A88D0` | `tdi5651.m0000` |

All 29 concrete methods declared directly on TutorialsController (`tdi5650`)
were initially without authored coverage classifications. This subset does not
claim the remaining methods or their generated coroutine bodies. OnEnable and
OnDisable have separately verified metadata references for seven matching
delegate subscriptions/removals; executable event wiring remains outside this
audit. Folded delegate-constructor symbol aliases do not establish the delegate
types: the tutorial call sites and exact generic type/MethodInfo slots do.

## Exact ownership

TutorialsController is TypeDefIndex 5650, a MonoBehaviour. Its four fields are
`allTutorials` at `0x20`, `queuedTutorials` at `0x28`, `showedTutorials` at `0x30`
and `onTutorialShow` at `0x38`. TutorialNote is TypeDefIndex 5633, a MonoBehaviour.
The script pins its string ID, starting/current state, type, stop-time flag,
onHide delegate, stage-array reference, current stage and clickable flag to
their exact declarations. `toturialStages` retains its metadata spelling.
ETutorialState values are ShowOnce 0, Shown 10, AlwaysShow 20 and None 100.

The persistence path uses ProjectContext TypeDefIndex 5546 static Instance,
its GameData field, GameData TypeDefIndex 5928 saveData, and SavedGameData.save.
These are distinct authored physical objects with valid supplied runtime
class/static records. They use the same exact chain established by the
[reset button join](reset_tutorial_button_join.md).

## Presentation reset

Controller.ResetTutorials clears only the **in-memory showedTutorials count**
and increments its version, including unsigned wrap. Its existing backing
values remain retained. It then performs the native inline equivalent of
Note.ResetTutorial for each array entry. It does not invoke SavedGameData,
clear persisted completed tutorials, change queuedTutorials or hide the note's
own GameObject.

Each reset publishes current stage 0 first. With a nonempty stage array, it
requests every stage inactive in order, then reloads the stage-array reference
and activates its first stage. An empty stage array skips those UI operations.
It restores the current state from startingState unless startingState is None.
Duplicate note references repeat the work; duplicate stage references receive
repeated writes to the same physical UI record. Null-array/element failures
retain all earlier list, note and UI changes.

## Showing and saving

EnableTutorial directly transfers control to ShowTutorial. ShowTutorial scans
all notes with the requested type. CheckIfAllTutorialsClosed reads every note's
own GameObject activeSelf and stops at the first active one. The queued list
does not participate in this predicate. If an active note exists, ShowTutorial
constructs and appends a new TutorialQueue with the requested type, exact pivot
reference and restriction false. It does not deduplicate these entries.

If all notes are inactive, it calls Note.Show. Note.Show first replaces a None
startingState with the current state, then immediately returns for state Shown.
Otherwise it queries the actual completedTutorials list via an explicit
Contains service. An already completed ID returns before presentation.

For state ShowOnce, an uncompleted ID reaches actual native AddTutorial and
Save, including native JSON field writing and preference provider/key/setter
execution. Only after that save completes does the note become Shown. A
registry/provider write failure preserves the newly appended completion while
the note remains ShowOnce and its UI/coroutine operations have not occurred.

For eligible presentation, a live authored pivot supplies exact 12-byte
Vector3 position bits to the note transform. A null pivot skips that copy.
The note GameObject is activated. stopTime requests timeScale 0. The caller
allocates and publishes a CloseCooldown continuation with state 0 and the exact
note reference to an explicit StartCoroutine service; no MoveNext or timing is
inferred.

After **any normally returning Note.Show**, including its Shown/already-completed
early returns, Controller.ShowTutorial combines an OnTutHidden delegate and
appends that note's type to showedTutorials. The audit preserves this distinction
between a caller publishing a type and an actual note being presented. Multiple
matching notes are processed in array order; a first activation causes later
matches to queue. Repeated completed-note aliases can accumulate delegates and
repeated type entries without activation. A supplied inert onTutorialShow
callback observes the newly appended type; its complete state also participates
in controlled service-entry stops.

## Evidence and explicit limits

The values-only report is
[`f530404b0f3f_807de4a83df4_tutorial_persistence_join.json`](../../reports/f530404b0f3f_807de4a83df4_tutorial_persistence_join.json),
produced by
[`audit_tutorial_persistence_join.py`](../../scripts/audit_tutorial_persistence_join.py).
It contains **474 cases, 64 exact controlled stops, 13 pinned instruction
relationships and 701 outer caller/service execution addresses**. Independent
repo/private producer runs were byte-identical (4,509,584 bytes; SHA-256
`5468a59cf67aef35e1ceee626a3f0f2555dee2dd4362f98df8848010b503625c`).
All 32 reverse-engineering tests, Python compilation and whitespace checks
passed.
It records exact metadata signatures/slots and native ranges, authored inputs,
ordered service calls, physical list/backing values, note fields, UI records,
Vector3 bits, delegate/queue/continuation records and persistence results.

Unity GameObject/Transform/liveness, timeScale, runtime allocation/metadata/
barriers, delegate construction/combine/cast, List append and coroutine
publication are **explicit services**. String Contains/growth retain their
existing native save-audit service contract. The actual game-owned caller
bodies perform all other control flow and stores. Delegate combination returns
authored ordered composition records; it does not execute the managed runtime
multicast implementation. onTutorialShow is supplied absent or an explicit
inert callback with verified target/type/MethodInfo arguments. Runtime Unity
Object class state is supplied initialized, independently of the caller's cold
metadata flags.

Notes/stages are bounded to 32; authored enum/reference List backing capacity
is 32, with failure prefixes and backing slots retained. The existing save
fixtures retain their 256-slot and string/runtime byte limits. The outer caller
uses the join's bounded 120-second wall budget and 100,000-instruction budget;
nested emulators keep their existing bounds. Object identities are never
transplanted into the JSON/storage/runtime emulator; only explicit public
values cross those adapters.

The report checks untouched note fields, controller references and stage-array
bytes. Controlled service stops compare the complete current state and exact
event prefix to the baseline service-entry snapshot before compact output drops
repeated snapshots. Successful Save/Load cases additionally use the native
getter/runtime-string/JSON read pipeline to verify persisted completion values.
Malformed-runtime-string behavior is covered by its separate existing audit.
No native exception unwinding, callbacks that mutate records, asset button
wiring or automatic Unity event/coroutine dispatch is claimed.
