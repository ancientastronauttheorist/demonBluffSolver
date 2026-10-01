# Tutorial close/reveal callback join

Pinned build: `f530404b0f3f_807de4a83df4`. This values-only offline audit invokes
the actual game-owned callbacks with authored arguments and joins their native
callees into the previously verified tutorial presentation/persistence pipeline.
It does not claim that a Unity event has dispatched or that the game is ready
for any coroutine to run.

| Exact metadata method | Native entry | Coverage identity |
| --- | --- | --- |
| TutorialsController.CloseTutorialIfAble | `0x38C8C0` | `tdi5650.m0013` |
| TutorialsController.OnCharacterReveal | `0x38CC30` | `tdi5650.m0010` |
| TutorialNote.HideNote | `0x38BCB0` | `tdi5633.m0002` |
| TutorialsController.OnTutHidden | `0x38DB80` | `tdi5650.m0024` |
| TutorialsController.ProcessTutorialQueue | `0x38DDB0` | `tdi5650.m0025` |
| Character.GetCharacterBluffIfAble | `0x364C40` | `tdi5487.m0013` |

The complete native shared integer-state coroutine constructor at `0x357700`
also executes, through its verified end `0x357724`, including its folded
Object base constructor. The HoverCharacterRoutine (`tdi5642`) and
PickableCharacterType (`tdi5645`) declarations pin state `0x10`, current `0x18`,
controller `0x20` and exact Character argument `0x28`. Their MoveNext bodies
remain outside this audit.

## Reveal ordering and appearance source

OnCharacterReveal first allocates, initializes and publishes a
HoverCharacterRoutine with the exact controller and incoming Character
reference. Only afterward does it check for a null incoming Character and
invoke the actual GetCharacterBluffIfAble body. Null-character failure thus
retains the already published routine with its null captured argument.

Character is exactly TypeDefIndex 5487; real CharacterData is `dataRef` at
`0x50`, bluff at `0x58`, revealed at `0xD8` and state at `0xE4`.
GetCharacterBluffIfAble returns dataRef for state 20/30 or nonzero revealed.
Otherwise it performs supplied Unity-null comparison of the retained bluff:
absent/destroyed bluff returns dataRef, while live bluff returns the exact bluff
record. No role, alignment or status field participates in this getter.

The callback then reads **CharacterData.picking at `0x13E`**, pinned to
TypeDefIndex 5845. Any nonzero value causes an additional native
PickableCharacterType allocation/publication, after HoverCharacterRoutine.
These are coroutine publication services; no synchronous first yield, scheduling
or gameplay ability use is inferred.

Finally, OnCharacterReveal visits all notes in array order and invokes native
HideNote for active notes whose type is RevealCard (10). It queries activeSelf
before checking the note type. CloseTutorialIfAble follows the same active-note
pattern for the caller's supplied type, without the Character routines/getter.

## HideNote and exact partial effects

HideNote does nothing when clickable is false. When clickable is true, it
first requests timeScale 1 if stopTime is set; the native float32 literal is
pinned to `0x3F800000`. This restoration also happens on intermediate stages.
It then increments currentTutStage before validating the stage-array reference.

If another stage remains, it hides the old stage and activates the next one,
with separate array reload/checks. The note's own GameObject remains active and
onHide is not invoked. If no stage remains, it hides the note GameObject, then
tail-invokes its supplied single-method onHide delegate when present. Invalid
stage indices, signed overflow and null-array failures retain the increment and
any earlier timeScale/UI changes. A close request can therefore advance a
tutorial rather than close it.

The callback fixture supplies an explicit delegate object whose method pointer,
MethodInfo and target bind the actual OnTutHidden body. This is a declared
single-method call interface, not evidence for engine or runtime multicast
dispatch. OnTutHidden constructs/removes its own handler through supplied
delegate services, publishes the resulting onHide reference and write barrier,
then tail-calls native ProcessTutorialQueue.

## Queue selection and persistence join

ProcessTutorialQueue counts **every restriction=true record** using an explicit
List enumerator. That count becomes the index selected from the original
physical list. It does not search for the first unrestricted record or partition
the list. Mixed-order fixtures explicitly check both get_Item calls and
RemoveAt receive that exact count-based index. All-restricted and empty queues
return without showing anything.

For a selected record it loads type, reloads the list/item to obtain pivot, and
calls actual ShowTutorial. Only after ShowTutorial normally returns does it
reload the queue and request RemoveAt at the same index. If another note is
active, ShowTutorial can append a new queued record before that original record
is removed. If all notes are inactive, the join reaches actual Note.Show,
SavedGameInfo.AddTutorial, SavedGameData.Save, native JSON writing and native
preference provider/key/setter code, as established by the separate
[tutorial persistence join](tutorial_persistence_join.md).

A supplied registry/provider write failure occurs after the prior note has been
hidden and its onHide handler removed. It retains the newly appended completed
tutorial, leaves the next note in ShowOnce state, and retains the original queue
record because RemoveAt was not reached. No rollback is modeled.

## Evidence and boundaries

The executable fixture is
[`audit_tutorial_close_reveal_join.py`](../../scripts/audit_tutorial_close_reveal_join.py),
and its values-only report is
[`f530404b0f3f_807de4a83df4_tutorial_close_reveal_join.json`](../../reports/f530404b0f3f_807de4a83df4_tutorial_close_reveal_join.json).
The report pins exact metadata fields/signatures/slots, native ranges and
caller/callee relationships. It records authored inputs, physical queue and
backing slots, note stages/state/delegate fields, exact Character references,
partially constructed routine records, ordered UI/runtime/list services and
native persistence output. Controlled stops compare complete projected state
and exact baseline service-entry event prefixes before compact output drops
repeated event snapshots.

List enumeration/get_Item/RemoveAt, runtime allocation/metadata/barriers,
Unity object/liveness/UI/timeScale, delegate construction/removal/casts and
coroutine publication are explicit supplied services. Queue service MethodInfo
arguments are verified against exact TutorialQueue generic metadata slots;
enumerator version and item bounds are checked. Native exception unwinding and
try/finally recovery are not executed. No generator MoveNext, event readiness,
asset wiring or actual registry access is claimed.

The input Character, both CharacterData records, controller references,
stage-array bytes and all unconsumed note fields retain their original authored
storage. Snapshot decoding is gated until handler fixtures are initialized;
routine snapshots preserve null controller/Character fields before publication.
Note/stage/queue fixture capacities stay bounded to 32. Existing save/runtime
fixture bounds and the outer 120-second/100,000-instruction budget remain in
force. Only public values cross the isolated JSON/storage/runtime adapters.

StartTutorials, CharacterInfoNote, CharacterKilledTutorial and LevelIdTutorial
are identified follow-on publication boundaries; this report does not claim
their bodies or their routine scheduling.

Validation: 445 fixtures passed, including 179 controlled failure stops, with
13 exact native assertions and 832 observed caller/service instruction addresses.
Two independent full producers emitted identical 3,922,189-byte reports (SHA-256
`fc4294a937a5628230e96c520102598e714e1cde3e5273e3e927a408fd535b05`).
Python compilation, all 32 reverse-engineering tests and the owned-file whitespace
check passed. No native engine or operating-system registry was accessed.
