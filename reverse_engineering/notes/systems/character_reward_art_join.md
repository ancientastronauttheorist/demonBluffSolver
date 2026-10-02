# Character reward initialization through actual data art and SetupArt

Build `f530404b0f3f_807de4a83df4`. This composition executes actual
Character.SetupObject, InitReward, RevealReal, CharacterData.GetArt,
CharacterData.GetArtType and Character.SetupArt in one physical graph.
The frozen reward, art and data producers provide immutable exact declaration,
field, enum, range and instruction pins. Their files remain unchanged.
GetAnimatedArt and the other Character art preference bodies are excluded.

The reward caller ranges and behavior retain the preceding reward initialization
join's exact evidence. Added bodies have these complete exclusive ranges:

| Method | Identity | Range | Next managed entry | Terminal trap |
| --- | --- | --- | --- | --- |
| GetArt | `tdi5845.m0007` | `3B4AB0..3B4B39` | `3B4B40` | `3B4B38` |
| GetArtType | `tdi5845.m0009` | `3B4A20..3B4AA3` | `3B4AB0` | `3B4AA2` |
| SetupArt | `tdi5487.m0021` | `3688B0..3689C1` | `3689D0` | `3689C0` |

Each added body has a matching complete unwind range. Exact raw backing,
decoder consumption, next managed entry and trailing alignment are checked
before execution. InitReward's terminal `365711` and RevealReal's terminal
`36840E` remain explicitly excluded as well. The decoded set contains 311
instructions, including four terminal traps; RevealReal's separately classified
trap lies just beyond its inherited nontrap decoder range. All 307 nontrap
instructions execute, with 67 selected operand assertions and 321 total native
and supplied gateway addresses. Completeness follows from complete decoded
body sets and reached addresses, not a count of caller services.

All consumers share one physical Object class and the exact metadata slot
`2718BF0`. RevealReal retains its pinned empty String slot separately.
GetArt, GetArtType and SetupArt have independent metadata byte gates at
`288C4D9`, `288C4DB` and `288C16A`. Fixtures vary those bytes independently,
including noncanonical nonzero values, and independently vary RevealReal's gate
and the shared class E0 DWORD. Completed class setup writes only that DWORD.
Callbacks resetting it cause the next native consumer to initialize the same
class again; metadata warmth does not imply class warmth.

The retained reward graph now has 55 physical records. Existing actor/data/TMP,
Acted/GameObject, String/Sprite, Action, MethodInfo and class records remain.
Three distinct 256-byte Skin diagnostic windows, four Image windows and three
Image GameObjects occupy a new nonoverlapping allocation range. Data retains
all 384 diagnostic bytes. Actor art+`28` and clippingArt+`30`, data currentSkin+`C0`
and default art_cute+`98`, and Skin art+`38` and type+`50` are bound to exact
managed field declarations. All unconsumed bytes remain diagnostic; these
windows do not establish complete engine object extents or runtime admission.

GetArt captures currentSkin before class initialization and supplies that
captured pointer to Unity equality. Only AL controls selection. A true AL
returns the current default art_cute field. A false AL reloads currentSkin,
guards that current pointer and returns its current Skin.art field. Thus a
class/equality callback can change the output skin while the equality service
still observes the originally captured skin. Clearing the reloaded skin reaches
the exact native null gateway and retains the full earlier prefix.

RevealReal keeps GetArt's actual full pointer result in native RDI, reloads
actor.dataRef, and calls actual GetArtType. The latter has the same capture,
class and low-AL gate pattern but returns Default 0 for true AL or the reloaded
Skin.type DWORD for false AL. EAX writes zero the upper RAX bits. A GetArt
callback replacing actor.dataRef therefore changes the second getter's owner
without replacing the first produced Sprite. Native SetupArt receives that
first Sprite and the second getter's low DWORD type with exact zero R9.

SetupArt captures Sprite in RDI and type in ESI before its metadata/class work.
Its supplied Unity equality consumes the captured Sprite and only AL. True AL
skips both Image paths and all their pointer reads. False AL selects clipping
only for exact low DWORD 10; every other DWORD selects ordinary art. It loads
the selected Image for Component.get_gameObject, guards the returned GameObject
and activates it. The native `mov dl,1` preserves all upper RDX bits left by
the supplied getter; the report keeps and independently checks the full value.

After activation SetupArt reloads the selected actor Image field and supplies
the original captured Sprite to Image.set_sprite. Callback replacement can
therefore activate one component's GameObject but set another component's
Sprite. It then reloads the other actor Image field, obtains its GameObject
and deactivates it using full RDX zero. Shared Images or GameObjects retain
ordered activation and deactivation, including a final false state when both
operations address the same GameObject. The caller then resumes actual
RevealReal's background comparison/setter and supplied UpdateViewReal tail.

Native nested-frame snapshots retain method identity, complete entry
RCX/RDX/R8/R9, exact entry stack pointer, raw caller return target and captured
skin where reached. Actual getter results and normal callee ABI checks persist
as ordered histories. Each successful native callee verifies its entry stack
and all eight saved nonvolatile integer registers plus XMM6–XMM15 before its
actual return. The outer return verifies the original stack and registers too.
Supplied services poison volatile integer and XMM registers. Interrupted frames
remain represented at their stopped service boundary; no unwind is invented.

The independent ordered model begins from full authored initial bytes and
fixture options. It predicts every full snapshot, all raw service arguments,
native call/jump site and return target, native frame entry/capture/result,
reached pointer/DWORD/byte stores, completed service effects, targeted callback
mutation, metadata/class gate, null guard and final state. It independently
models the two actual getters and SetupArt; it does not import their native
returned snapshots as expected output. Mutation logs record exact completed
phase/action pairs, and planned mutation phases must actually be reached.
Failed supplied entries perform no effect or callback and preserve the exact
full stopped prefix. Byte allowances track only reached exact field widths and
completed class writes; metadata slots and unrelated bytes remain unchanged.

Corpus cases include independent gates, live/dead/null skins, forced low-AL
values with nontrivial upper bits, all relevant/raw art type DWORDs, nullable
Sprite/Image/GameObject outputs, physical aliases, class reset chains and
callbacks at metadata, class, equality, getter, activation and Sprite-setting
phases. Compound cases distinguish captured first Sprite, reloaded second data,
current skin/type and current Image receiver. Retained side/reward/presentation
chains preserve complete storage, class/metadata gates and all prior histories.

GC barriers, metadata/class services, Action callback, String.ToUpper, TMP
setters, Unity equality/inequality, Component.get_gameObject, GameObject.SetActive,
Image.set_sprite and Character.UpdateViewReal remain supplied. Their authored
effects do not prove engine liveness, rendering, allocation, scheduling, CLR
admission, native unwinding or the full acquisition interleaving.

Reports use lossless raw-memory pooling followed by full-snapshot pooling.
Expand snapshots with `audit_report_snapshots.expand_snapshots`, then memory
with `audit_character_oracle_reveal_join.expand_memory`; both codecs verify
exact round trips and retain every field and diagnostic byte.

The final corpus has 394 cases: 342 normal returns and 52 native stops.
Fourteen retained sequences and fourteen normal baselines cover 337 exact full
stopped prefixes. Retained initial/final continuity is checked independently.
Two individually syntax-preceded, independently launched successful native
producers emit identical 65,836,716-byte reports, SHA-256
`75d6758bd67566ae5cda098e94e7118d81a182b3e05d58beb9f417fe8ef47270`.
There are 225 memory blobs and 6300 full snapshot blobs.
Private peer:
`B:/CodexTools/DemonBluffReverseEngineering/artifacts/f530404b0f3f_807de4a83df4/character_reward_art_join.peer.private.json`.
Python syntax, all 36 reverse-engineering infrastructure tests and diff checks
pass. New script, note and report are frozen for parent integration.

Run `reverse_engineering/scripts/audit_character_reward_art_join.py` with pinned
game and Dumper directories as positional arguments and `--output`; PYTHONPATH
must include the private python-emulation directory.
