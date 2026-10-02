# Character RevealBluff presentation and refresh ordering

Build `f530404b0f3f_807de4a83df4`. This family executes the exact complete
`tdi5487.m0046`, `Character::public void RevealBluff()`, at `0x368130`.
Its non-trap body ends at `0x368292`; the verified unwind range ends at
`0x368293`, including its terminal int3. That trap and subsequent alignment
before the next managed entry `0x3682a0` are excluded. Bounds, file backing,
complete decoder consumption and exact metadata signature are checked before
execution. All 86 non-trap instructions are reached, with 23 operand assertions
and 99 observed native/supplied gateway addresses.

The frozen reward presentation machine supplies immutable input/type pins,
distinct authored storage, full snapshots, virtual TMP gateway infrastructure
and volatile return poisoning. Its SetupObject/RevealReal instructions are
replaced by the exact RevealBluff instruction set and never execute in this
family. No unfinished source or InitReward implementation is imported.

RevealBluff reads exactly actor.bluff at `0x58`, independently of real dataRef
at `0x50`. Default fixtures give those fields different physical data records;
alias fixtures share them, and a callback clearing only dataRef leaves Bluff
presentation intact. The exact Character, CharacterData, TMP_Text and
TextMeshProUGUI declarations and consumed color/name/background offsets are
pinned through the frozen verifier plus the explicit bluff declaration.

After metadata setup, the caller captures bluff and chName (`0x40`). It guards
bluff and its characterName (`+0x28`), then calls supplied String.ToUpper.
Unlike RevealReal, a null ToUpper result is forwarded directly to the TMP text
setter. No empty String literal is referenced by this native body. The chName
receiver remains the captured component even when a ToUpper callback replaces
or clears the actor field. Its class is loaded after ToUpper. The text virtual
call uses slot 66, function `0x558`, MethodInfo `0x560`, and also leaves the
physical TMP class pointer in R9. Fixtures change that class after ToUpper and
independently verify both the loaded MethodInfo and full R9.

After text, bluff and chName are separately reloaded for color. The exact raw
16-byte Color at bluff+`0xd8` is copied through the native stack and passed to
TMP's color setter, slot 23, function `0x2a8`, MethodInfo `0x2b0`. A text callback
can therefore alter the later data, component, or class. Raw signed zero, NaN
payloads and infinity bits remain intact; no float normalization is performed.

Bluff is then separately reloaded for GetArt and GetArtType. The first Sprite
remains captured while the later type query can use another physical data
record. GetArtType's authored upper RAX sentinel is discarded by the native
R8D write before supplied SetupArt. After SetupArt returns, another bluff read
captures backgroundArt (`+0xb8`) before optional Object class initialization.
Unity op_Inequality checks that captured Sprite and consumes only AL.

For a true AL, the current bluff and artBg (`0x130`) are reloaded; the current
backgroundArt pointer is sent to Image.set_sprite. The checked Sprite can
therefore differ from the Sprite actually set. A false AL suppresses those
later reads even if callbacks clear bluff/artBg or install a new live Sprite.
Metadata byte and Object E0 class DWORD are varied independently across warm
and cold fixtures. Warm skipped services do not execute their callbacks or
receive write allowances.

Finally the caller invokes supplied Character.UpdateView (`0x3695a0`), then
tail-calls supplied Character.RefreshView (`0x367b60`). A callback clearing
bluff during UpdateView does not prevent RefreshView: this caller performs no
bluff recheck between those services. Both receive full RDX zero, while R8/R9
retain the explicitly authored volatile residue. Complete raw RCX/RDX/R8/R9,
native call/jump sites and return targets are retained for every supplied
entry and native null guard. The guard residues are independently predicted
without assigning false parameters to the throw gateway.

The audit passes 144 cases: 129 normal returns and 15 exact native stops.
Three retained three-call sequences preserve one physical graph while changing
bluff and forwarding a nullable uppercase result. Six baselines have 70 exact
full stopped prefixes. A null actor faults with READ_UNMAPPED at `0x368159`;
all other tested null field exits reach the exact `0x36828d` null-throw call.
Normal returns preserve the stack, all eight nonvolatile integer registers
and XMM6–XMM15. Supplied returns poison caller-saved integer/XMM registers.

The independent ordered model starts from the initial authored graph and
predicts each complete service snapshot, exact raw ABI and final byte. It
simulates captured/reloaded receivers, raw color/type handling, reached
callbacks and metadata/class effects for returns, partial faults and stops.
The separate retention check permits only exact callback ranges or a completed
class-init E0 DWORD write. Failed services apply no effect or callback, and their
full event prefix/final state must equal the baseline's service-entry snapshot.

Full raw diagnostic storage is retained for actor, data records, TMP/Image
records, two authored TMP classes/vtables, MethodInfo records, Strings,
Sprites, Acted records and Object class. The inherited unused empty-literal
slot and other diagnostic fields remain unchanged; they are not claimed as
consumed RevealBluff inputs. Shared real/bluff data and TMP/background aliases
use one physical identity; the latter is an authored diagnostic alias, not a
claim of valid engine type admission. No report fields or snapshots are dropped.

Supplied boundaries remain String.ToUpper, TMP setters, CharacterData getters,
Character.SetupArt, Unity inequality, Image.set_sprite, metadata/class runtime,
UpdateView and RefreshView. No renderer, scheduler, InitReward, actual art/View
callee, runtime allocation/admission or native unwinding is reconstructed.

Full snapshots are pooled losslessly by verified SHA-256 with exact expansion
round trips. Two independent final producer processes emitted identical
28,030,683-byte reports, SHA-256
`a8cea4cb4db8d7df44a45e00c5e1d332f56a3784e7b23ed9535593a449d9e381`.
Private peer:
`B:/CodexTools/DemonBluffReverseEngineering/artifacts/f530404b0f3f_807de4a83df4/character_bluff_presentation.peer.private.json`.
Python syntax compilation, all 36 RE infrastructure tests and diff checks
passed. The new script, note and report are frozen for parent integration.

Run `audit_character_bluff_presentation.py` with positional pinned game and
Dumper directories plus `--output`; PYTHONPATH must include python-emulation.

## Lossless authored-memory pooling

The current producer additionally pools raw diagnostic memory before full
snapshots, using `sha256-authored-memory-hex-v1`. Expand full snapshots first
with `audit_report_snapshots.expand_snapshots`, then memory with
`audit_character_oracle_reveal_join.expand_memory`. Every expanded field equals
the prior snapshot-pooled report, including every service-entry snapshot and
stopped prefix. No native execution, coverage, diagnostic bytes or semantics
change. Two newly syntax-preceded successful independent native producers match.

Current size: 7688402 bytes, versus 28030683 previously.
SHA-256: `a5b54d878288e7f8a0e964ca9625a11e1b62ab52d73435e8a009666e5e721e02`.
Memory blobs: 24; full snapshot blobs: 1410.
Private current peer: `B:/CodexTools/DemonBluffReverseEngineering/artifacts/f530404b0f3f_807de4a83df4/character_bluff_presentation.pooled.first.private.json`.
The previous peer and report hash above retain the earlier encoding provenance.
