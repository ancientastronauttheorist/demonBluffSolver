# Character tutorial generator and persistence composition

Pinned build: `f530404b0f3f_807de4a83df4`. This values-only offline audit executes
the actual HoverCharacter and PickableCharacter generator factories and manually
ordered MoveNext calls. Their true native ShowTutorial calls reach the existing
[presentation/persistence join](tutorial_persistence_join.md), and their queue
branch joins the [close/hide/queue pipeline](tutorial_close_reveal_join.md).

| Exact native method | Coverage identity | Entry |
| --- | --- | --- |
| TutorialsController.HoverCharacterRoutine(Character) | `tdi5650.m0022` | `0x38C960` |
| TutorialsController.PickableCharacterType(Character) | `tdi5650.m0021` | `0x38DC90` |
| TutorialsController.<HoverCharacterRoutine>d__26.MoveNext | `tdi5642.m0002` | `0x3AA0D0` |
| TutorialsController.<PickableCharacterType>d__25.MoveNext | `tdi5645.m0002` | `0x3AAA50` |
| TutorialsController.<>c__DisplayClass25_0.<PickableCharacterType>b__0 | `tdi5638.m0001` | `0x3AD4F0` |
| UnityEngine.WaitForSeconds..ctor(float) | `tdi6799.m0000` | `0x1C961F0` |

All exact metadata signatures, decoded native ranges and containing unwind
families are asserted. Pickable MoveNext spans five contiguous unwind records;
the first ends at `0x3AAAD5`, whereas the complete body ends after its final
null-reference helper at `0x3AACF2`, excluding the final trap. The remaining
records correspond to normal state branches, not supplied catch/finally
execution. Neighboring managed entries and padding establish complete bounds.

The native factories request allocations, invoke the actual shared state
constructor, and store exact controller and Character captures. Both generator
declarations use state `0x10`, current `0x18`, controller `0x20`, Character
`0x28`. Unlike the separate poison routine, these captures are not reversed.
The yielded WaitForSeconds object executes its actual native constructor,
including the folded Object base constructor and float32 store at `0x10`.

## Exact yields and continuation routes

Hover state 0 sets state to -1, allocates/constructs WaitForSeconds with literal
bits `0x3F000000` (0.5), stores current and performs its supplied write barrier,
then sets state 1 and returns true in AL. State 1 sets state -1 before checking
the captured Character and its **Character.icon Transform field at `0x20`**.
It requests that component's Transform, checks the captured controller and calls
actual ShowTutorial(HoverCharacter = 40, returned Transform), then returns false.
Other states return false without changing state or current.

Pickable state 0 follows the same yield pattern with bits `0x3D4CCCCD`
(float32 approximately 0.05), then state 1. On the state-1 resume it sets state
-1 and asks the supplied physical showedTutorials list whether it contains
CharacterInfo = 30. This predicate uses the exact native generic MethodInfo.
It does not ask whether CharacterInfo is currently active.

When CharacterInfo was already shown, Pickable constructs a second wait with
bits `0x3F000000`, changes state to 2 and returns true. State 2 sets state -1,
reads Character.icon, requests its Transform and calls actual
ShowTutorial(PickableCharacter = 80, returned Transform), then returns false.
Current retains the last yielded wait after completion; it is not cleared.

When CharacterInfo was not shown, state 1 allocates a native closure record,
requests the Character icon Transform, allocates a TutorialQueue and stores
type 80, that exact pivot and restriction true. It stores the **same queue
record** in closure.newTut at `0x10`. It creates an authored Action<ETutorialType>
delegate whose target and method point to the real native closure body, then
**replaces controller.onTutorialShow** at `0x38`. No Combine occurs. Finally it
appends that same queue reference through an explicit physical List service and
returns false. A supplied existing callback is retained in the report as the
replaced reference, without an inferred unsubscribe or callback restoration.

The actual closure compares the incoming tutorial type to CharacterInfo = 30.
Only that type clears the captured queue's restriction byte at `0x20`. Other
types leave it unchanged. This callback does not process the queue or show a
tutorial by itself.

## Native callback and save composition

The restricted-queue fixture explicitly invokes native ShowTutorial(30) after
the generator has completed. The supplied CharacterInfo note shows and saves
its completion. Actual ShowTutorial appends the shown type and directly invokes
the installed delegate's target/method pointer, executing the real closure and
clearing that same queue restriction.

The next explicitly ordered native CloseTutorialIfAble(30) invokes HideNote,
the installed native OnTutHidden callback and ProcessTutorialQueue. With the
CharacterInfo note now inactive, the unrestricted queue reaches actual
ShowTutorial(80), Note.Show, SavedGameInfo.AddTutorial, SavedGameData.Save and
the verified native JSON/preferences/storage pipeline. Only afterward does
ProcessTutorialQueue remove the original queue. The fixture retains the queue
object itself and its restriction value in snapshots after list removal.

This sequence establishes actual caller/callee composition with authored
inputs. It does not claim that the real engine admitted any resume, waited for
the requested duration or dispatched a gameplay/UI event.

## Evidence and supplied boundaries

The executable is
[`audit_tutorial_character_generators.py`](../../scripts/audit_tutorial_character_generators.py)
and the report is
[`f530404b0f3f_807de4a83df4_tutorial_character_generators.json`](../../reports/f530404b0f3f_807de4a83df4_tutorial_character_generators.json).
The report records complete generator states/current references, exact stored
wait bits, manual resume order and byte-width Boolean returns. It preserves
partially written queue/closure/delegate objects, native metadata flags,
prior/current callback references, existing tutorial snapshots and joined save
outputs. Controlled service-entry stops compare exact event prefixes and
complete projected state before report compaction drops duplicate snapshots.

Runtime allocation/metadata/write barriers, delegate construction, generic List
Contains/Add/enumeration/item/remove and Unity component/Transform/UI services
are explicitly supplied. Delegate construction projects a declared single-method
native callback interface; no engine multicast dispatch is inferred. Exception
helpers provide controlled stops; native exception unwinding is not run.
Character storage and physical stage-array storage are checked for retention.
The existing bounded fixture arena and outer 120-second/100,000-instruction
budget apply, with nested JSON/preferences/runtime limits preserved.

No actual scheduler, elapsed time, live game, registry or OS process state is
used. The public tutorial type names and floating-point magnitudes come from
their exact enum declarations and decoded literal slots.

Validation: 47 fixtures and 206 controlled service-entry failure stops passed,
with 36 exact metadata/native assertions and 923 observed caller/service
instruction addresses. Two independent full producers emitted identical
792,715-byte reports (SHA-256
`8b3f97ba7b89fb9413ee16686288c6ea542c01e6fed0f2fdc045ea555dbeeaea`).
Python compilation, all 32 reverse-engineering tests and the owned-file
whitespace check passed.
