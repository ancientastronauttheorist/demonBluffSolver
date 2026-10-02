# Character reward art through UpdateViewReal colors

Build `f530404b0f3f_807de4a83df4`. This bounded composition retains the six
actual bodies in the frozen reward-art graph and adds actual
Character.UpdateViewReal as the seventh. The exact native route ends in a
whole supplied Character.RefreshView. UpdateViewReal does not call
CharacterView.Init, SetupArt or an animation body; no CharacterView edge is
inferred from its name. Frozen reward-art/View/Data sources and reports remain
unchanged.

UpdateViewReal is exact `tdi5487.m0065`, declaration
`public void UpdateViewReal()`, RVA `3694D0`, with exact Dumper signature
`void Character__UpdateViewReal (Character_o* __this, const MethodInfo* method);`.
One matching complete unwind is `3694D0..36959C` exclusive. The next managed
entry is `3695A0`, with four trailing `CC` alignment bytes. Full raw backing,
decoder consumption, exact declaration ordinal, complete unwind and next entry
are pinned before execution. Its body fingerprint is retained in the report.

The full seven-body instruction set has 364 decoded instructions. All 357
supported nontrap instructions execute. Six decoded terminal traps and the
bounds dispatch instruction at `369596` remain unexecuted; RevealReal's
separately classified terminal trap is also explicitly pinned. The bounds
gateway follows adjacent native loop/index comparisons with no supplied
callback between them. Stable authored arrays cannot synthesize a concurrent
length change there; that race and exception unwinding remain excluded.
UpdateViewReal has null gateway `369590`, terminal null trap `369595`, bounds
gateway `369596` calling `2B7D80`, and bounds terminal trap `36959B`. There are
84 selected operand assertions and 371 observed native/supplied addresses.

Exact Character fields are Image bg+`120` and Image[] borders+`128`.
CharacterData.cardBgColor+`F8` and cardBorderColor+`108` are exact 16-byte
Color values. The supplied Image virtual color setter is bound to exact
UnityEngine.UI.Graphic TypeDefIndex 9897, virtual slot 23, declared
`public virtual void set_color(Color value)` at `1D41930`. The class function
pointer at `2A8` and full MethodInfo at `2B0` are asserted. Two complete authored
Image class diagnostic windows and separate MethodInfos share the same
supplied gateway as the TMP color setter; exact native site and physical
receiver/class distinguish these entries.

UpdateViewReal loads current actor.dataRef before its background Image field.
It guards both, copies that data's exact raw background color to the native
stack, then loads the captured Image's class and slot 23 MethodInfo. Background
callback mutations occur after this completed request; they do not replace
its captured color or receiver. Only afterward does native code capture the
actor's current border-array pointer once.

The loop reads the signed low DWORD at array+`18`, retaining the independently
authored upper DWORD. Negative low DWORDs skip the loop; high bits do not
change that comparison. Each iteration reloads current actor.dataRef, guards
it, loads the current element from the captured array at `20 + index*8`, guards
that Image, and supplies current data.cardBorderColor. Array length and elements
remain dynamic, while replacing or clearing actor.borders after capture does
not replace the retained array. Shrinking or changing the low length to a
negative value stops later iterations; growing from one to three exposes the
remaining authored slots. All executed indexes stay within three retained
slots; raw header probes do not establish valid managed array allocation.

Callbacks can change data between border requests, alter a later element,
replace an Image class/MethodInfo, or replace raw color storage. Duplicate
Image elements receive repeated ordered requests. Mixed TMP/Image/background
and art aliases preserve one component state: later color writes change that
same physical component without fabricating independent alias copies.
Signed zero, NaN payloads, infinities, subnormal/raw DWORD values and all
diagnostic bytes remain exact without float normalization.

After the loop the native body restores its stack and all nonvolatile state,
then jumps to exact Character.RefreshView `367B60` with actor in RCX and full
RDX zero. The complete original caller return address remains on the stack.
The body has no metadata or class initialization gate of its own. Existing
Data/art consumers still share one Object class/slot and retain their exact
warm/cold gates; the new Image classes do not invent a new class service.

The graph has 66 physical records. New allocations contain two background
Image records, three border Images, two 256-byte array windows, two 1024-byte
Image classes and two MethodInfos. Existing 384-byte data, 512-byte actor/class,
1536-byte TMP classes, Sprite, String, Action, GameObject and Acted windows are
retained in full. Array raw headers, all three physical slots, every class slot
and physical identity are serialized. These windows retain consumed and unused
diagnostic bytes; they do not prove complete runtime object extents or admission.

The independent model starts from complete initial storage and options. It
predicts every full service-entry snapshot, raw RCX/RDX/R8/R9, exact native
call/jump site and return target, nested entry/capture/result history, reached
native store, completed supplied effect/callback and final state. It models
UpdateViewReal separately from the native driver, including background data
capture, array capture, signed length and each later data/element/class load.
Normal native getter/SetupArt returns verify their saved Win64 state; the new
UpdateViewReal tail verifies its entry stack and all eight integer nonvolatiles
plus XMM6–15 before transfer. The outer return independently verifies original
stack/register preservation despite supplied volatile poisoning.

Exact byte allowances cover reached native stores and completed authored
pointer/DWORD/16-byte color/class mutations only. Every planned callback phase
must actually be reached. Failed entries have no effect or callback; each
stopped full state equals the baseline boundary snapshot and its entire event
list equals the complete baseline prefix. Retained sequences assert adjacent
final/initial equality, retaining all physical state and earlier histories.

The new corpus preserves all 394 frozen reward-art profiles and adds raw array
lengths, nullable background/array/elements, physical aliases, callback data and
array replacement, count growth/shrink/negative lengths, class/MethodInfo and
color replacement, compound capture/reload cases and retained continuations.
RefreshView remains a whole supplied method. GC, runtime metadata/class setup,
Action, uppercase, TMP/Image setters and Unity APIs remain supplied. No renderer,
scheduler, full acquisition interleaving, CLR admission or unwind is claimed.

The family preserves every full report row, field, history and raw byte using
four lossless layers. Raw memory is pooled first, complete ordered histories
second, complete named memory-reference maps third, and full snapshots last.
Decode in reverse with this script's `expand_report(report)`. For the root
integration helper, `expand_memory(post_snapshot_report)` is the compatible
adapter after `audit_report_snapshots.expand_snapshots`.

Family-local assertions verify exact round trips, history order, 64-bit/null
values, record-name identity, independent decoded copies, and corruption
rejection at each of all four layers. They run before each native producer and
also pass independently. The existing 36 infrastructure tests are separate
validation; they do not exercise these new family-local codecs.

The final corpus contains 475 cases: 411 normal returns and 64 native stops.
Seventeen retained sequences and nineteen normal baselines produce 534 exact
full stopped-prefix probes. Complete expansion, all adjacent retained states,
all stopped prefixes and per-case independent verification markers are checked
again against the serialized final report.

Two independently launched final native producers each passed preceding Python
syntax and family-local codec assertions, then emitted identical 57,957,772-byte
reports, SHA-256
`7be4d3b02240dff1ab3bc435830af4bca2b0593a689380b648f9add627e7558c`.
The report has 297 raw-memory blobs, 3332 ordered-history blobs, 643 complete
named memory maps and 8736 full snapshot blobs. Complete map interning reduces
the earlier 107,904,148-byte checkpoint without removing any evidence.
Private final peer:
`B:/CodexTools/DemonBluffReverseEngineering/artifacts/f530404b0f3f_807de4a83df4/character_reward_color_join.peer.private.json`.
The 204-byte UpdateViewReal body fingerprint is
`cb69649ac46dc63b3164cfd01bd69150e7534f80df465b8b794c23a50705017d`.
All 36 existing RE infrastructure tests and diff checks pass. Script, note and
report are frozen for parent integration.

Run `reverse_engineering/scripts/audit_character_reward_color_join.py` with
pinned game and Dumper directories as positional arguments and `--output`.
PYTHONPATH must include the private python-emulation directory.
