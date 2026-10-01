# CharacterInfo native publication and tutorial queue join

Pinned build: `f530404b0f3f_807de4a83df4`. This values-only audit extends the
[four-handler publication audit](tutorial_handler_publication.md) by explicitly
resuming the actual CharacterInfo generator and composing its true native
ShowTutorial call with the existing tutorial/storage audits.

| Exact method | Coverage identity | Native range, exclusive end |
| --- | --- | --- |
| TutorialsController.CharacterInfoNote(Character) | `tdi5650.m0012` | `0x38C170..0x38C202` |
| TutorialsController.<CharacterInfoTutorial>d__18.MoveNext | `tdi5640.m0002` | `0x3A9210..0x3A92E1` |

Exact metadata signatures and TypeSignatures, complete decoded ranges,
containing unwind entries and next managed entry/padding are asserted. The
native handler requests a generator allocation, invokes its native integer-state
constructor, stores the incoming controller and Character, then tail-publishes
it through a supplied StartCoroutine service. The allocation service starts
the state field with a sentinel, so publication's state 0 verifies the actual
constructor store. Null captures are retained at publication and checked later
by the generator rather than rejected by an invented handler guard.

The exact generated class (`tdi5640`) uses state `0x10`, current `0x18`,
controller `0x20` and Character `0x28`. State 0 changes state to -1, constructs
a WaitForSeconds with float32 bits `0x3D4CCCCD`, stores current and requests a
write barrier, then publishes state 1 and returns true in AL. The literal is
resolved from the decoded operand to `0x1F34C3C`, approximately 0.05 seconds.
No elapsed-time or scheduler admission is inferred from that value.

On the state-1 resume, the native generator changes state to -1 before checking
Character and loading **Character.acteds at `0xA8`**. This field's exact type is
Acted (`tdi5477`), a MonoBehaviour. It requests that component's Transform,
checks its captured controller and calls actual ShowTutorial(CharacterInfo =
30, returned Transform), then returns false. It does not use Character.icon or
hintPivot. Other states return false and retain state/current. Normal completion
retains the yielded WaitForSeconds reference.

WaitForSeconds executes the same actual native constructor already pinned in
the [Character generator audit](tutorial_character_generators.md). This audit
also explicitly pins System.Object..ctor at `0x33ED50` to its exact metadata
signature and three-byte native `ret 0` leaf through `0x33ED53`. Exact call
operands at `0x1C96203` (WaitForSeconds constructor) and `0x357711` (generator
state constructor) are asserted. The inherited emulator executes that native
leaf; it is not replaced by a base-constructor service.

## Actual restricted-queue composition

The joined fixture first invokes the actual PickableCharacter factory and two
manually ordered native MoveNext calls with CharacterInfo absent from the
supplied showedTutorials list. That produces the same restricted queue and
installed onTutorialShow closure as the separately verified Character generator
audit. It then invokes actual CharacterInfoNote and manually resumes its
published generator twice.

That second resume obtains the Transform from the supplied Acted component and
calls actual ShowTutorial(30). The CharacterInfo note shows and persists its
completion; ShowTutorial appends the shown type and directly executes the
installed native Pickable closure. That closure clears restriction on the same
physical queue reference retained by its captured newTut field. The replaced
prior callback remains uncalled in the authored profile.

An explicitly invoked native CloseTutorialIfAble(30) then joins HideNote,
OnTutHidden, ProcessTutorialQueue, ShowTutorial(80), Note.Show, AddTutorial,
Save, native JSON and native preferences/storage. It persists Pickable completion
and removes the original queue after normal ShowTutorial return. This replaces
the previous explicit Show30 invocation with its real game-owned generator
caller; closing the note and admitting resumes remain explicit fixture inputs.

## Evidence and limits

The executable is
[`audit_tutorial_character_info_join.py`](../../scripts/audit_tutorial_character_info_join.py)
and the report is
[`f530404b0f3f_807de4a83df4_tutorial_character_info_join.json`](../../reports/f530404b0f3f_807de4a83df4_tutorial_character_info_join.json).
It records exact publication captures, partially constructed routines, native
metadata flags, complete manual resume snapshots, yielded references/bits,
physical closure/queue identity and joined storage output. Cold/warm metadata,
invalid completed states, null Character/Acted/controller and null returned
Transform profiles retain their exact native partial effects. Every service-entry
prefix of direct CharacterInfo save and the full queue composition is replayed
with a controlled stop and compared to complete projected state before report
compaction drops duplicate snapshots.

Runtime allocation, metadata resolution, write barriers, coroutine publication,
delegate construction, generic List operations and Unity Transform/UI access
are supplied services. A null Transform is an authored getter result accepted
by ShowTutorial's native optional-pivot path; it does not claim actual Unity
null-receiver behavior. Character and Acted bytes and stage-array storage are
checked for retention. Existing capacities and the outer
120-second/100,000-instruction budget apply; nested native JSON/preferences/runtime
limits are retained. No real scheduling, event dispatch/readiness, exception
unwinding, registry or live-process operation is performed.

Validation: 15 fixtures and 164 controlled service-entry failure stops passed,
with 24 exact metadata/native assertions and 890 observed caller/service
instruction addresses. Two independent full producers emitted identical
428,624-byte reports (SHA-256
`97509f336ac3685ad25c6e1dac85a1141321c311313189b22d37dc0fd17f6a2f`).
Python compilation, all 32 reverse-engineering tests and the owned-file
whitespace check passed.
