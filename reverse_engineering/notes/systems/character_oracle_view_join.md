# Character Oracle callers joined to CharacterView

Pinned build `f530404b0f3f_807de4a83df4`. This separate offline audit executes
Character.OracleEyeActive (`tdi5487.m0030`, `0x3675A0..0x36778D`) and
HideOracleInfo (`tdi5487.m0031`, `0x3654F0..0x36563D`) with actual
CharacterView.AnimateIn (`0x363D50`), AnimateOut (`0x363DF0`), Init (`0x363F10`)
and SetupArt (`0x3643F0`) bodies. It replaces only those whole-game-owned supplied
View boundaries in the frozen Oracle caller audit. RevealOrder.Init/Hide remain
explicitly supplied here; the separate Oracle-to-RevealOrder report establishes
that different composition. Neither earlier source/report family changes.

The frozen verifiers pin exact extraction hashes, metadata declarations,
signatures, consumed fields, native bounds, following managed entries or unwind
families, literal bytes and decoded operands. This join retains 36 exact operand
assertions. All 192 Oracle caller instructions and 231 of 235 decoded View
instructions execute. The four unexecuted View addresses are `0x3640E5`,
`0x3640E6`, `0x3640EB`, `0x3644B1`: the frozen bounds-gateway call and terminal
traps. Their absence is asserted and is not promoted to executed coverage.
The 447 observed addresses include supplied gateways, not extra implemented
methods. View unwind ranges end at `0x363DE4`, `0x363E7F`, `0x3640EC`,
`0x3644B2`, respectively.

The corpus contains 253 cases (221 normal returns, 32 native guards), six
retained Oracle/Hide/Oracle sequences, six service-stop baselines and 168
complete stopped-prefix fixtures. At each supplied service boundary the entire
current state is retained. Each stop compares all earlier events to its
baseline prefix and its final state to that exact service-entry snapshot.
Normally returned invocations verify all eight integer nonvolatile registers,
XMM6-15 and restored stack position. Supplied services poison caller-saved
integer registers and XMM0-5.

## Physical storage and shared services

The Character's physical view field names the same View receiver for animation
and initialization. Additional View images, text, canvas, border array, string/
sprite records, virtual classes and MethodInfo storage occupy a separate
allocation region from Oracle's Actor/Acted/history/arrow records. View gateway
functions, supplied tween pointers and new runtime metadata records likewise
use separate regions. Oracle arrow colors retain their original list-shaped
bookkeeping; View images have distinct color/sprite records. They share no
accidental Python attribute or physical class identity.

The one GameObject state map represents both families. Shared
Component.get_gameObject calls route by physical receiver: CharacterView
versus View's art/clipping Image. SetActive routes by return positions derived
from the decoded native call sites, including SetupArt's tail-call return into
Init. This distinguishes service roles even when View art/clipping GameObjects
alias each other or an Oracle description, picker or view GameObject. The
current active state can therefore reflect a later disable on an aliased
physical object. Aliased View Images and duplicate border slots retain their
actual repeated receiver identity and chronology.

Snapshots preserve Oracle diagnostic Actor, Acted, List/backing array, info,
data/bluff, arrow and View bytes, plus all additional authored View fields,
components, classes, MethodInfo, sprites/strings, GameObjects and metadata
records. Every byte outside precise native View.dataRef/saved-arrow-color
stores, supplied initializer DWORD updates and explicit callback mutations is
checked unchanged. Physical field identity is serialized separately from
supplied component values. Sentinel data and larger diagnostic windows do not
establish complete valid managed runtime objects or extend their real layouts.

The report pools only diagnostic byte windows through SHA-256 memory references.
Its 90 distinct blobs retain every original byte; the imported
`expand_memory(report)` verifies blob hashes and restores full snapshots.
Each producer asserts exact expanded round-trip equality after all native
prefix checks. Logical state, event order and field identities remain explicit.

## Actual caller composition

Only exact Dead state 20 with killedByDemon byte zero and nonzero supplied
inequality AL enters Oracle's view path. Actual AnimateIn captures animId before
the supplied DOTween class initializer/Kill, requests fade to exact `1.0f`
with `0.2f` bits `0x3E4CCCCD`, then reloads animId for SetId. Actual Oracle
instructions then reload both Character.charBluff and bluff before Init.
Clearing the Character view during the supplied Kill still permits the captured
animation to finish, then the parent stops before Init. Replacing the Character
bluff during Kill reaches the later actual Init argument.

Init stores that argument at View.dataRef and enters the supplied GC barrier.
A distinct field replacement during that barrier remains observable in the
physical final View, while captured original data feeds colors, background,
name, GetArt and GetArtType. Exact sixteen-byte color values preserve signed
zero and NaN payloads. The border array receiver is captured; clearing the View
field during a border color service leaves that earlier array usable. Shrinking
its length changes the actual subsequent iteration. Duplicate Images and border
slots keep their repeated color effects.

The text receiver is captured before supplied uppercase. Clearing the View
text field inside uppercase still calls the previously captured component.
Art/type services receive the captured data receiver. Actual SetupArt tests the
low type DWORD for exact 10, obtains/enables the selected art or clipping
GameObject, reloads its Image for sprite publication, then disables the other
GameObject. Clearing art during its SetActive preserves that earlier enable,
then reaches the following Image reload guard. Upper return bits and type
values including `0x8000000A` do not substitute for exact type 10.

HideOracleInfo's parent path obtains the physical View GameObject and tests only
supplied activeSelf AL. A nonzero result executes actual AnimateOut, requesting
fade to exact zero with the same duration; zero skips that callee. Both animation
directions retain old animId capture and new SetId identity under mutations at
class initialization, Kill and DOFade. Six retained sequences preserve View
fields, colors/sprites, border and GameObject aliases, metadata/class state,
supplied text and complete request chronology. RevealOrder and all earlier
parent Acted/history/color/picker logic still execute at their explicit supplied
boundaries with actual parent field loads and ABI.

## Remaining boundaries

RevealOrder.Init/Hide, Acted.Act, Character.GetCharacterBluffIfAble and folded
List.GetItem remain supplied. Runtime metadata/class/GC effects, Unity Object/
Component/GameObject calls, DOTween Kill/DOFade/SetId, Image/TMP methods,
uppercase and CharacterData.GetArt/GetArtType remain supplied with exact caller
receivers and arguments. The View and Oracle native bodies execute; those
service implementations do not. No renderer, scheduler, actual animation
completion, engine lifetime/event admission, concurrent array-size race or
native exception unwinding is reconstructed. No live game/process/UI access is
used, and private native bytes remain outside the repository.

Executable: [audit_character_oracle_view_join.py](../../scripts/audit_character_oracle_view_join.py).
Report: [f530404b0f3f_807de4a83df4_character_oracle_view_join.json](../../reports/f530404b0f3f_807de4a83df4_character_oracle_view_join.json).
Two successful independent final producers emitted identical 61,856,055-byte
reports, SHA-256 `f6f4bfaec3adeb351b29219e4c76c4e30eaccc52e402f502069d5f4a8eacf91d`.
Python syntax compilation, all 32 reverse-engineering tests and focused diff
checks passed.

```powershell
$env:PYTHONPATH='B:/CodexTools/DemonBluffReverseEngineering/python-emulation'
python reverse_engineering/scripts/audit_character_oracle_view_join.py GAME_ROOT DUMPER_ROOT --output REPORT
```

The reviewed ABI checks assert full Kill RDX from each animation entry
argument and its exact metadata-byte/class-DWORD gates. Warm entry with zero
RDX and no reached initializer sends `0x1`; reached supplied metadata/class
services poison the upper bits and send `0xFACE123456789001`. Both are retained
and checked. Routed Image SetActive requires full `0xFACE123456789001` for
the DL-only true write or whole zero for false. Metadata-record retention
permits E0..E4 updates only on exact physical classes whose initialization
service completed; MethodInfo and unrelated class records stay unchanged.
