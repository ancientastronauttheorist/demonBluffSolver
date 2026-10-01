# Native tutorial handler publication

Pinned build: `f530404b0f3f_807de4a83df4`. This values-only executable audit
completes the four publication handlers left separate from the
[close/reveal join](tutorial_close_reveal_join.md). All four actual caller bodies
and the native shared generator constructor execute; runtime and Unity services
are explicitly supplied.

| Exact method | Coverage identity | Native range, exclusive end |
| --- | --- | --- |
| TutorialsController.CharacterInfoNote(Character) | `tdi5650.m0012` | `0x38C170..0x38C202` |
| TutorialsController.CharacterKilledTutorial(Character) | `tdi5650.m0006` | `0x38C330..0x38C42F` |
| TutorialsController.LevelIdTutorial() | `tdi5650.m0003` | `0x38CAD0..0x38CC2A` |
| TutorialsController.StartTutorials() | `tdi5650.m0011` | `0x38E3D0..0x38E560` |

Ranges include every return/tail path and the final exception-helper call,
excluding alignment traps. Metadata supplies exact signatures, generated
declarations, TypeDefIndex identities and RIP-relative class slots. The shared
integer-state generator constructor at `0x357700..0x357724` is verified against
its unwind entry and runs its folded Object constructor, then writes the actual
state field at `0x10`.

## Publication order and captured arguments

StartTutorials initializes and publishes four routines in this order:
RevealCardTutorial (`tdi5647`), KillTutorial (`tdi5644`), HoverObjective
(`tdi5643`), CancelKillTutorial (`tdi5639`). Each starts at state 0 with null
current and its exact supplied controller captured at `0x20`. KillTutorial and
HoverObjective also retain their initial null closure field at `0x28`.

CharacterInfoNote publishes CharacterInfoTutorial (`tdi5640`), capturing the
exact controller at `0x20` and incoming Character at `0x28`.
CharacterKilledTutorial publishes CharacterKillRoutine (`tdi5641`) with that
same layout, then PoisonKilledRoutine (`tdi5646`). **PoisonKilledRoutine reverses
the captures: Character is at `0x20`, controller at `0x28`.** The audit decodes
each routine according to its exact declaration, including partial stores
visible at write-barrier failure. Null Character inputs are captured without a
caller-side dereference or guard. Null controllers are likewise retained and
passed to the authored publication service; this does not establish that Unity
accepts StartCoroutine on a null receiver.

Neither handler inspects alignment, status, killed role or poison in this
publication prefix. Those decisions belong to generated MoveNext bodies.
StartTutorials does not reset tutorials or invoke EnableTutorial/ShowTutorial
directly. No persistence call is reached in any of these four native prefixes.

## Level reload and supplied callback effects

LevelIdTutorial resolves Gameplay (`tdi5604`), checks its class initialization
field, reads the class static storage at `0xB8`, then static Instance at `0x10`.
The exact instance `currentLevel` field is `0x78`. A null instance reaches the
explicit native null-reference helper before any routine publication.

Level 1 publishes SecondLevelNoteCoroutine (`tdi5648`). After the publication
service returns, the native caller reloads Gameplay class initialization,
static Instance and currentLevel. It then publishes ThirdLevelNoteCoroutine
(`tdi5649`) if that newly read level is 2. With inert publication, level 1 or 2
publishes one corresponding routine and other integers publish none.

Authored publication callbacks demonstrate the reload: changing 1 to 2 causes
both publications; changing 1 to 3 retains only the second-level routine;
clearing Instance causes a native null-reference stop after that first
publication. Clearing the class initialization field causes a second supplied
class-initialization service call. A supplied class initializer can also change
the level before the first read. These are controlled service effects, not
claims about the game's real class initializer or StartCoroutine scheduling.

## Fixtures, evidence and limits

The executable is
[`audit_tutorial_handler_publication.py`](../../scripts/audit_tutorial_handler_publication.py)
and the values-only report is
[`f530404b0f3f_807de4a83df4_tutorial_handler_publication.json`](../../reports/f530404b0f3f_807de4a83df4_tutorial_handler_publication.json).
Normal fixtures cover cold/warm metadata, absent/present supplied controller and
Character, initialized/uninitialized Gameplay class, level extremes, null
Gameplay and callback mutations. Every service-entry prefix of four meaningful
baselines is replayed with a controlled stop and compared to the exact complete
projected snapshot, including allocation order, metadata flags, partially
written captures, publication order and changed Gameplay state.

Runtime allocation, metadata resolution, class initialization, write barriers
and coroutine publication are supplied services. The actual callbacks do their
native allocations requests, constructor calls, reference stores, class/level
guards, reloads and branch/tail order. Character bytes, controller references
and unconsumed Gameplay/static bytes are checked for retention. Save values,
native storage and existing tutorial state remain unchanged. Fixture allocation
uses the existing bounded authored arena; the outer native run keeps its
120-second and 100,000-instruction limits.

No generated MoveNext, event dispatch, engine readiness, synchronous first
yield, exception unwinding or real UI/registry/process operation is claimed.
Joining these publications to EnableTutorial/ShowTutorial requires the separate
generated routine bodies and explicit scheduling inputs; the audit does not
invent that route.

Validation: 55 normal/null/callback fixtures and 42 controlled service-entry
failure stops passed, with 33 exact metadata/native assertions and 292 observed
caller/service instruction addresses. Two independent full producers emitted
identical 498,584-byte reports (SHA-256
`223a1fabe061614834e72f1820d4188e74c1ca1590c8fd6dd4e0293a4de54de7`).
Python compilation, all 32 reverse-engineering tests and the owned-file
whitespace check passed.
