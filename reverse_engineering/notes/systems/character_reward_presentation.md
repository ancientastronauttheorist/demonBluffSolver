# Character reward object selection and real presentation

Build `f530404b0f3f_807de4a83df4`. This family executes exact SetupObject
(`tdi5487.m0019`, `0x3689d0`) and RevealReal (`tdi5487.m0044`, `0x3682a0`).
InitReward is excluded and belongs to its separate audit. All 113 non-trap
instructions are reached, with 23 operand assertions and 126 observed native
and supplied gateway addresses. The 341 cases comprise 322 normal returns and
19 exact native stops. Three retained four-call sequences preserve one physical
graph across SetupObject, RevealReal, SetupObject, RevealReal. Six baselines
have 61 exact full stopped prefixes.

Bounds are verified before decoding. SetupObject is a 26-instruction leaf
without an unwind record; its complete return ends at `0x368a44`, before
alignment and the next managed entry `0x368a50`. RevealReal is an 87-instruction
body ending after its null-throw call at `0x36840e`; its unwind entry ends at
`0x36840f`, including one terminal int3. That trap and subsequent alignment
before the next managed entry `0x368410` are excluded. Neither method is sized
from a first return. Native bytes and exact Dumper inputs are hash-pinned.

SetupObject tests the low DWORD of its side argument. Exact EActedSide values
Up 10, Left 20, Down 30, and Right 40 select actor fields `upActed` (`0xc0`),
`leftActed` (`0xb8`), `downActed` (`0xc8`), and `rightActed` (`0xd0`) and store
that physical pointer to `acteds` (`0xa8`), followed by a tail GC barrier.
Right 40 alone stores raw byte 1 to `leftAct` (`0xb0`); other sides preserve
the previous byte, including noncanonical authored Boolean diagnostics.
Unknown DWORD values return without touching the actor, even with a null actor.
A recognized side with a null actor has its exact READ_UNMAPPED load fault;
null selected Acted references are stored and passed to the barrier normally.

RevealReal captures dataRef (`0x50`) and chName (`0x40`) after metadata setup.
It guards dataRef and its characterName String (`CharacterData+0x28`), calls
String.ToUpper, substitutes the exact pinned empty String literal for a null
uppercase result, and only then guards the captured chName. Its TMP virtual
text setter uses slot 66: function `0x558`, MethodInfo `0x560`. The component
class is loaded after ToUpper, so a callback can change the MethodInfo while
the original component remains captured. A callback replacing/clearing the
actor's chName does not replace that first captured setter receiver.

After the text setter returns, the actor's dataRef and chName are both
reloaded. Exact 16 raw color bytes at CharacterData+`0xd8` are copied to the
native stack and supplied to TMP's virtual color setter, slot 23, function
`0x2a8` and MethodInfo `0x2b0`. Fixtures distinguish a later component or class
from the original text receiver and preserve signed zero, NaN payloads and
infinities without float normalization.

RevealReal then separately reloads dataRef for GetArt and GetArtType. The
first returned Sprite remains captured while callbacks can replace the data
used for the type query. Only EAX's low DWORD is forwarded in R8D to the
supplied SetupArt; the getter's authored upper RAX sentinel is discarded.
SetupArt returns before another dataRef reload and capture of backgroundArt
(`0xb8`). That Sprite is captured before class initialization and checked
through Unity op_Inequality, consuming only AL. If AL is true, dataRef and
artBg (`0x130`) are reloaded, and the current data's current background Sprite
is passed to Image.set_sprite. Thus the Sprite checked for liveness can differ
from the Sprite actually set. A false AL suppresses those later pointer reads
even when a callback has cleared dataRef/artBg or supplied a new live Sprite.
The final tail call is exactly Character.UpdateViewReal (`0x3694d0`).

All supplied entries retain full RCX/RDX/R8/R9, the exact native call/jump site,
and raw return target. Null-guard register residues are derived independently
at each reached branch without inventing parameters for the throw gateway.
Normal returns preserve the stack, all eight nonvolatile integer registers,
and XMM6–XMM15. Supplied services poison caller-saved integer/XMM registers.
Each native pointer guard, recognized-side null-owner load fault, and partial
presentation state is checked. The actor, three data records, TMP/Image
records, two full authored TMP classes/vtables, all MethodInfo records,
Strings, Sprites, Acted records and Object class retain full diagnostic bytes.
Physical aliases include shared side Acted pointers and a shared TMP/background
component; the latter is an authored diagnostic alias, not engine type admission.

The independent ordered semantic model starts from initial authored storage.
It predicts every event's full snapshot and raw ABI, reached metadata/class
effects, native side writes, supplied UI outcomes and targeted callbacks.
It verifies all returned/partial/stopped final states byte for byte. Runtime
class E0 writes are allowed only after completed initialization; callbacks
permit only their exact reached field widths. Metadata slots remain unchanged;
flag changes are checked by the independent service order. Failed services
retain the complete baseline prefix and service-entry snapshot without effects.

Explicit supplied boundaries are String.ToUpper, TMP text/color setters,
CharacterData.GetArt/GetArtType, Character.SetupArt, Unity op_Inequality,
Image.set_sprite, Character.UpdateViewReal, metadata/class initialization and
GC barriers. No art getter/SetupArt/View body, renderer, acquisition scheduler,
InitReward body, valid runtime object allocation, or native unwinding is claimed.

Full snapshots use lossless SHA-256 pooling with verified expansion round trips.
Two independent final producer processes emitted identical 22,237,039-byte
reports, SHA-256
`49d3d60cdb36d35d639a9cc65801133dedf4f6f61405a4e191fbaa6d3bb7e82c`.
Private peer:
`B:/CodexTools/DemonBluffReverseEngineering/artifacts/f530404b0f3f_807de4a83df4/character_reward_presentation.peer.private.json`.
Python syntax, 36 reverse-engineering infrastructure tests and diff checks
passed. The script, note and report are frozen for parent integration.

Run `audit_character_reward_presentation.py` with positional pinned game and
Dumper directories and `--output`; PYTHONPATH must include python-emulation.

## Lossless authored-memory pooling

The current producer additionally pools raw diagnostic memory before full
snapshots, using `sha256-authored-memory-hex-v1`. Expand full snapshots first
with `audit_report_snapshots.expand_snapshots`, then memory with
`audit_character_oracle_reveal_join.expand_memory`. Every expanded field equals
the prior snapshot-pooled report, including every service-entry snapshot and
stopped prefix. No native execution, coverage, diagnostic bytes or semantics
change. Two newly syntax-preceded successful independent native producers match.

Current size: 6116022 bytes, versus 22237039 previously.
SHA-256: `409a0eb29d26eb67d4ad35857c732928ed2dae2a32d09a8684ac54b36f76129c`.
Memory blobs: 106; full snapshot blobs: 1124.
Private current peer: `B:/CodexTools/DemonBluffReverseEngineering/artifacts/f530404b0f3f_807de4a83df4/character_reward_presentation.pooled.first.private.json`.
The previous peer and report hash above retain the earlier encoding provenance.
