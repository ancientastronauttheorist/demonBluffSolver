# Character art preferences and animation selection

Build `f530404b0f3f_807de4a83df4`. This audit executes four complete Character
native bodies and their actual internal SetupArt calls. It passes 518 fixtures
(483 normal returns and 35 native null guards), five retained three-call
sequences, and 110 exact stopped prefixes against nine baselines. All 251
non-trap instructions are reached; 25 operand assertions pin the important
capture, reload, branch and ABI relationships. The 262 observed addresses
include the explicitly supplied service entries.

Exact declarations are `tdi5487.m0020` ReInitPreferences (`0x367890`),
`m0021` SetupArt (`0x3688b0`), `m0022` ShowAnimatedArt (`0x368b40`), and
`m0023` HideAnimatedArt (`0x365260`). Complete ends are respectively
`0x367963`, `0x3689c0`, `0x368c13`, and `0x365333`, excluding each terminal
`int3` and subsequent alignment padding. Bounds are checked against the next
verified managed entry and file-backed section extent; they are not inferred
from a first return. GameAssembly and both Dumper inputs are hash-pinned.
The consumed Character fields are exactly Image `art` at `0x28`, Image
`clippingArt` at `0x30`, CharacterData `dataRef` at `0x50`, and CharacterData
`bluff` at `0x58`. The exact EArtType declaration has Default 0 and Clipping 10.

ReInitPreferences and HideAnimatedArt have the same ordered behavior.
ShowAnimatedArt differs only in requesting GetAnimatedArt instead of GetArt.
Each first compares the captured dataRef against Unity-null. Only a true AL
causes a separate, later bluff load and Unity-null comparison. Both Unity-null
results suppress all art fetching and SetupArt, preserving existing Images and
GameObjects. A live dataRef therefore skips the bluff comparison altogether.
The dataRef/bluff captures occur before their corresponding class-init service;
a class-init callback clearing the field does not replace the captured receiver.

On the active path, the caller requests Character.GetCharacterBluffIfAble,
guards its returned raw pointer, and fetches a Sprite. It then requests
GetCharacterBluffIfAble again, guards that separately returned pointer, and
fetches the art type. The first Sprite remains captured even when callbacks
change the second data selection. SetupArt receives that Sprite and only the
low DWORD of the supplied type return; fixtures give the getter a nonzero upper
RAX sentinel and give direct SetupArt calls a nonzero upper R8 sentinel.

SetupArt returns without UI changes when Sprite equality has true AL, including
a destroyed non-null Sprite. Otherwise exact type DWORD 10 selects clippingArt;
every other DWORD selects art. The selected Image's GameObject is enabled, then
the selected Image field is reloaded before set_sprite. The other Image's
GameObject is then disabled. Every component and getter return has its own
raw-pointer guard, so a secondary failure retains the earlier activation and
sprite assignment. Enabling writes only DL and retains supplied upper RDX bits
(`0xface123456789001` here); disabling clears EDX and passes full RDX zero.
R8 MethodInfo is explicitly zero at Unity setters. Full RCX/RDX/R8/R9 and exact
native return addresses are retained for all service entries, including stops.

Fixtures distinguish captured GameObjects from reloaded Images, field clearing
after SetActive, a later component replacement, physical Image/GameObject/data
aliases, and shared GameObject remapping after sprite assignment. Aliased Images
and shared GameObjects retain their actual sequential effects: the final
disable can deactivate the same object just enabled. A callback resets the
Object class-init DWORD between equality checks to reach all three second
initialization paths. Shared metadata/class gateway callbacks are qualified by
the exact allowed native caller return addresses.

An independent ordered semantic model verifies every normal/guard fixture,
callback fixture, baseline, and retained invocation, including receiver
captures and all final logical state. Every stopped fixture must match its
baseline's entire event prefix and full service-entry snapshot, with no supplied
effect applied at the stopping service. Normal returns preserve stack position,
all eight nonvolatile integer registers, and XMM6–XMM15. Supplied returns poison
caller-saved integer and XMM registers. All authored memory bytes are retained
unless an exact reached callback writes them or a completed class-initialization
service writes its E0 DWORD. Metadata slots are unchanged, and metadata flag
writes require the matching completed metadata service. Raw diagnostic storage
is retained for actor, data, Images, GameObjects, Sprites, and classes.

The boundary deliberately supplies Character.GetCharacterBluffIfAble
(`0x364c40`), CharacterData.GetArt (`0x3b4ab0`), GetAnimatedArt (`0x3b4990`),
and GetArtType (`0x3b4a20`). Their own role/preference/asset-selection bodies
are not executed here. Unity equality, component getters, SetActive and
Image.set_sprite, metadata initialization and class initialization are also
named supplied services. No Unity rendering, scheduler, live game state, or
complete preference event subscription is claimed.

The report uses lossless SHA-256 pooling only for complete authored memory
buffers; every pooled buffer is hash-verified and an expansion round trip must
reproduce the original report exactly. Logical and raw ABI evidence remains
unpooled. Two independent final producer processes emitted identical
42,962,680-byte reports, SHA-256
`4ceb344314d059c9bb6187724bc2c5ab5ffa31e071489bdb8f3971f245c37ad1`.
Python syntax compilation, all 36 reverse-engineering infrastructure tests,
and `git diff --check` passed. The private independent peer is
`B:/CodexTools/DemonBluffReverseEngineering/artifacts/f530404b0f3f_807de4a83df4/character_art_preferences.peer.private.json`.

Reproduce with `audit_character_art_preferences.py`, supplying the pinned game
directory, Dumper directory and output path with `--game-root`, `--dumper-root`,
and `--output`; `PYTHONPATH` must include the private python-emulation runtime.
This source, note and report family is frozen for parent integration.
