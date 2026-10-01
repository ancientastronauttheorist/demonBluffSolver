# Native death tutorial generators and save composition

Pinned build: `f530404b0f3f_807de4a83df4`. This values-only audit extends actual
CharacterKilledTutorial publication into both native generated MoveNext bodies,
the native CharacterStatuses.Contains wrapper and real tutorial UI/queue/save
callers. Resumes are explicitly supplied; no engine scheduler admission or death
event dispatch is inferred.

| Exact method | Coverage identity | Native range, exclusive end |
| --- | --- | --- |
| TutorialsController.CharacterKilledTutorial(Character) | `tdi5650.m0006` | `0x38C330..0x38C42F` |
| TutorialsController.<CharacterKillRoutine>d__11.MoveNext | `tdi5641.m0002` | `0x3A9330..0x3A9436` |
| TutorialsController.<PoisonKilledRoutine>d__12.MoveNext | `tdi5646.m0002` | `0x3AAD40..0x3AAE6F` |
| CharacterStatuses.Contains(ECharacterStatus) | `tdi5488.m0003` | `0x363C40..0x363C91` |

Exact metadata signatures and TypeSignatures, complete decoded ranges,
containing unwind entries and next managed entries/padding are asserted. The
final native exception-helper calls are included; padding traps are excluded.
No generated catch/finally or native exception unwinding is supplied.

The native handler constructs and publishes CharacterKillRoutine first, then
PoisonKilledRoutine. Runtime allocations begin with a sentinel state field;
actual native generator constructors replace it with state 0 before publication.
Both declarations use state `0x10` and current `0x18`. CharacterKillRoutine
captures controller `0x20` and Character `0x28`; **PoisonKilledRoutine captures
Character `0x20` and controller `0x28`**. Snapshot decoding follows the exact
declarations, including partial writes before barrier stops.

## Native waits and show gates

Both state-0 resumes change state to -1, allocate and construct an actual
WaitForSeconds with bits `0x3E4CCCCD`, store current and request a supplied write
barrier, then set state 1 and return true in AL. Both decoded literals resolve
to `0x1F34B10`, float32 approximately 0.2 seconds. The native WaitForSeconds
constructor and Object base leaf execute as established in the
[Character generator audit](tutorial_character_generators.md). This audit also
pins the exact System.Object..ctor metadata signature, three-byte `ret 0` at
`0x33ED50..0x33ED53`, and both constructor call operands.

On state 1 both routines set state -1 before resolving Gameplay class
initialization and static storage. They read **Gameplay.GameplayState at static
offset `0x28`**; Summary = 50 suppresses the tutorial before Character checks or
Transform requests. The native class-initialization request is a named supplied
service that sets its authored class flag and can explicitly update GameplayState.
The actual game class initializer is not executed. Gameplay.Instance, currentLevel
and player-owned fields are not used by these generator bodies.

CharacterKillRoutine then checks its captured Character, reads the exact
**Character.icon Transform field at `0x20`**, requests that component's Transform,
checks its captured controller and calls actual ShowTutorial(KilledCharacter =
100, returned Transform). It returns false. Other completed/invalid states
return false without repeating the show; current retains its yielded wait.

PoisonKilledRoutine first reads **Character.statuses at `0xF0`** and invokes
actual CharacterStatuses.Contains with **Corrupted = 10**. The status container
is exactly TypeDefIndex 5488; the wrapper reads its physical statuses List at
`0x10` and tail-calls a supplied generic List<ECharacterStatus>.Contains service
with the exact native MethodInfo. Membership is checked against explicit physical
entries, including duplicates; the resistance list is not consulted or changed.

If Corrupted is absent, PoisonKilledRoutine returns false without reading icon
or showing anything. If present, it requests Character.icon's Transform and
calls actual ShowTutorial(Poison = 45, returned Transform), then returns false.
Despite the method name, this native gate does not inspect a death-reason field,
killedByDemon, alignment or role. It does not infer how a Character reached death
or which prior gameplay action set the Corrupted status.

## Explicit resume orders and actual queue/save route

With authored inactive notes for types 100 and 45, one explicitly resumed
continuation shows its matching note and persists completion through native
Note.Show, AddTutorial, Save, JSON and preferences/storage. While that note is
active, the other native continuation calls ShowTutorial and queues its own
type/pivot through the actual caller path.

An explicitly invoked native CloseTutorialIfAble for the active type then joins
HideNote, OnTutHidden and ProcessTutorialQueue. Queue processing reaches native
ShowTutorial for the other type, Note.Show and a second native save, then removes
the original queue. Both continuation orders are verified: kill-first saves
kill-t then poison-t; poison-first saves poison-t then kill-t. Publication order
remains kill then poison in either fixture; no scheduling priority follows from
the supplied continuation order. The existing authored onTutorialShow callback
remains an explicit inert observation service.

## Evidence and boundaries

The executable is
[`audit_tutorial_death_generators.py`](../../scripts/audit_tutorial_death_generators.py)
and the report is
[`f530404b0f3f_807de4a83df4_tutorial_death_generators.json`](../../reports/f530404b0f3f_807de4a83df4_tutorial_death_generators.json).
Fixtures cover cold/warm metadata, initialized/uninitialized Gameplay class,
Day/Summary/Shop states, absent/present/duplicate/unrelated statuses, null
Character/icon/controller/status container/status list, summary bypass of null
captures, a supplied initializer changing the gate to Summary and repeated
completed-state resumes. Service-entry stop replays compare exact event prefixes
and complete projected state for direct kill save, direct poison save and the
full two-save queue composition.

Runtime allocation, metadata, write barriers, class initialization, publication,
delegate construction, generic List operations and Unity Transform/UI access
are supplied services. Native caller state changes, gates, constructor stores,
status wrapper, UI/queue/persistence calls execute. Character/status bytes,
physical status list, stage arrays and unconsumed Gameplay static bytes are
checked for retention. Bounds remain the existing 32-entry fixture capacities
and outer 120-second/100,000-instruction budget with original nested limits.
No elapsed time, live game/registry/process state, event readiness or exception
unwinding is claimed.

Validation: 64 fixtures and 220 controlled service-entry failure stops passed,
with 41 exact metadata/native assertions and 907 observed caller/service
instruction addresses. Two independent full producers emitted identical
1,135,689-byte reports (SHA-256
`e729543a561b2b8e0fe3689f1dde86cfe390a387c544020fe817994644f6ddf8`).
Python compilation, all 32 reverse-engineering tests and the owned-file
whitespace check passed.
