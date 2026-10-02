# Character reward presentation through actual RefreshView

Pinned build `f530404b0f3f_807de4a83df4`. This composition executes eight
actual native bodies in one physical graph: Character.SetupObject,
Character.InitReward, Character.RevealReal, CharacterData.GetArt,
CharacterData.GetArtType, Character.SetupArt, Character.UpdateViewReal and
Character.RefreshView. The frozen reward-color family and the older standalone
RefreshView audit remain unchanged. The verified route is
RevealReal → UpdateViewReal → RefreshView; it introduces no CharacterView edge.

RefreshView is exact `tdi5487.m0063`, declared `private void RefreshView()`.
Its exact metadata signature is
`void Character__RefreshView (Character_o* __this, const MethodInfo* method);`,
type signature `vii`, unique entry `367B60`. Complete chained unwind chunks are
`367B60..367C08`, `367C08..367C67` and `367C67..367DFC`, all exclusive.
The 668-byte complete body contains 156 instructions: 155 nontraps and the
terminal null-helper trap at `367DFB`. Four following CC bytes are alignment
before the next managed entry `367E00`. Full file backing, every decoded byte,
the exact method ordinal, all 23 decoded call sites, field declarations and
selected instruction operands are asserted before execution. Complete body
SHA-256 is `c327c2e9d510a8c3e1ed803474bf959523524290e77170cbe39faf5dc54ff751`.

Character TypeDefIndex 5487 binds icon `20`, bluff `58`, ripView `78`,
deadPrefab `80`, disguiseIcon `88`, createdDeadPrefab `98`, revealed byte `D8`,
pickableUses DWORD `DC`, state DWORD `E4`, killedByDemon byte `ED` and pickable
`1A8`. Exact ECharacterState TypeDefIndex 5489 names None=0, Hidden=5,
Alive=10, Dead=20 and Revealed=30. The separate revealed byte is diagnostic:
this caller does not use it to select the disguise predicate.

The shared Unity Object slot `2718BF0` and its complete authored class window
remain the same physical storage used by earlier Data/art/reward consumers.
RefreshView metadata flag `288C189` resolves the Instantiate MethodInfo slot
`26D7A30` and that Object slot in order. Vector3 slot `26EA2F8` has its own
metadata flag `288C0E7`; its class static-fields pointer is at `B8`.
Exact UnityEngine.Vector3 TypeDefIndex 6699 binds x/y/z at 0/4/8 and
zeroVector at static offset zero. Both metadata flags are raw bytes, including
noncanonical nonzero warm values. Reached Object class gates read the low DWORD
at class `E0`; completed supplied initialization writes exactly those four bytes.

The MethodInfo metadata is exactly
`Method$UnityEngine.Object.Instantiate<GameObject>()`, with Dumper MethodAddress
zero. The supplied entry is the exact folded
`UnityEngine.Object$$Instantiate<object>` row at `668010`, using original,
parent Transform and opaque MethodInfo arguments. The generic implementation
remains wholly supplied. Neither the MethodInfo name nor the folded entry
promotes another alias or establishes an actual Instantiate<GameObject> body.

RefreshView first checks pickableUses as a signed DWORD. Values at most zero
require the current pickable pointer, then hide that physical GameObject with
full RDX zero. Positive values preserve its current active state. The native
body does not decrement or reset the count.

Only state Dead=20 checks the captured existing createdDeadPrefab through
supplied Unity equality. A live result skips the entire creation/transform/RIP
path and retains the current raw pointer. An absent or authored destroyed result
enters creation. AL alone controls that branch; full supplied RAX retains its
authored upper bits and noncanonical true byte.

Creation captures the current deadPrefab before requesting the actor's Component
transform. A callback changing deadPrefab during that request does not replace
the captured Instantiate source. The parent Transform result is captured across
the following Object class gate, then supplied to Instantiate with the exact
generic MethodInfo. Authored null prefab/parent inputs expose the caller's
unchecked inputs; they do not establish Unity acceptance of those arguments.

The full Instantiate result pointer is stored at actor `98` before the reference
barrier. The barrier entry snapshot already contains the store, including a
nullable result. A barrier callback can change that field, which native code
reloads for its first GameObject transform request. The first Transform is
captured independently of later changes to actor.createdDeadPrefab.

The icon pointer is reloaded after that request. Its supplied Component
transform and position getter run before the captured first Transform is null
checked. The position getter uses the Win64 structure-return buffer in RCX,
Transform in RDX and MethodInfo zero in R8. Its supplied effect writes exactly
12 bytes to that buffer. Native code uses the returned full RAX pointer, copies
eight bytes with movsd and the remaining DWORD, then supplies a separate copied
buffer to set_position. Alternate returned-vector storage proves that the caller
reads RAX rather than assuming the hidden buffer; these are diagnostic supplied
API contracts. The actual getter buffer's written 12 bytes are independently
checked at the next reached native instruction.

After the position setter, native code reloads actor.createdDeadPrefab for a
second GameObject transform request. It resolves/reloads Vector3 metadata and
the class static-fields pointer, then checks the captured second Transform.
The raw first 12 static-field bytes are copied by QWORD plus DWORD to the same
setter buffer and supplied to set_eulerAngles. Nonzero zeroVector fixtures,
signed zero, NaN payloads, infinities and subnormals remain exact bytes without
floating-point arithmetic or parser normalization.

The completed position and Euler requests write 12-byte authored Transform
diagnostic windows at offsets `20` and `30`, and retain the ordered logical
transform state. These supplied effect offsets are fixture storage, not
assertions about Unity Transform's real internal layout. Full 128-byte opaque
Transform records retain every other byte. Distinct first/second identities and
same-Transform aliases preserve one physical record and ordered writes.

After both setters, current ripView is required and activated. This and the
disguise true path use `mov dl,1`, preserving the current upper RDX bits. False
setters clear full EDX. All service-entry RCX/RDX/R8/R9 values, exact native site
and full stack return target are compared independently at their actual widths.

The optional disguise pointer is captured for Unity inequality. A false result
or nonzero killedByDemon byte preserves its current active state. Otherwise the
current state DWORD chooses false for other states, or a current bluff liveness
request for Dead/Revealed. The disguise pointer is reloaded after liveness; a
callback can replace or clear the receiver used by the later setter. Clearing
bluff during its liveness callback does not change the already returned AL.
Later killed/state/bluff fields are read only at their native positions.

All eight bodies share actor, Data, Action, TMP/Image, arrays, class/MethodInfo,
String/Sprite and GameObject storage. New records add opaque icon/parent/created
Transforms, old/new/alternate created GameObjects, prefab/RIP/disguise/pickable,
Vector3 class/static storage, generic MethodInfo and alternate returned vector.
The actor is retained as a full 512-byte authored diagnostic window. Unity
active state is shared with earlier Acted/art GameObjects, so physical aliases
such as pickable=RIP=disguise=an existing reward GameObject observe every ordered
write to one state. Shared service gateways route by their exact decoded caller;
the same Unity entry is not assigned to another owner from its receiver alone.

The independent ordered semantic model begins with complete initial raw storage
and authored options. It predicts every full event snapshot, all four recorded
RCX/RDX/R8/R9 service arguments,
caller/site, native entry/captured frame, completed effect/callback, partial
fault/guard and final state without reading native output to generate expectations.
Callbacks can reset the shared Object class at the final color request or later
Refresh services. This reaches all four native Refresh class-init call sites,
including `367C25` and `367DCD`, which the older standalone fixture left
unexecuted. The full joined path ordinarily warmed the class earlier; a cold
Refresh gate requires an explicitly reached reset.

Native nulls preserve the exact completed prefix. Null Instantiate output is
stored/barriered before its following guard. A null first Transform still permits
icon position retrieval; a null second Transform occurs after position output
and possibly Vector3 metadata resolution. Null RIP occurs after both transform
writes. Supplied null position-return and static-field pointers produce exact
READ_UNMAPPED faults at `367CA7` and `367D15`, respectively. These diagnose the
caller's accesses rather than claim Unity returns those values in a real scene.

Retention allowances cover only reached native stores, completed supplied class
writes, exact 12-byte Transform effects and completed authored callback ranges.
Skipped/stopped services receive no effects or permissions. Every planned
callback phase must be reached except after an explicit controlled stop. Every
stopped event list and full final snapshot equal the baseline prefix and stopped
entry snapshot. Retained sequences assert complete adjacent final/initial equality
and show later refreshes reuse an existing live created object without re-instantiating.

Normal actual getter/SetupArt returns and UpdateViewReal/RefreshView transfers
verify their own entry stack and all eight integer nonvolatiles plus XMM6–15.
The original outer return verifies those registers and stack independently.
All supplied returns poison volatile integer and XMM registers. This family
does not record RAX/R10/R11 or XMM0 through XMM5 at each boundary, or compare
their full final residues. Its ABI evidence covers the four argument registers,
explicit Color/Vector storage and normal nonvolatile/stack preservation.
Complete volatile-register evidence requires a separate strengthening audit.
Terminal traps
are explicitly excluded; the existing adjacent-loop bounds dispatch remains
unexecuted because there is no supplied mutation boundary between its two native
checks. Concurrent array resizing and exception unwinding remain unresolved.

The added corpus covers signed counts/state/liveness/killed predicates, independent
revealed bytes, capture/reload callbacks, nullable dependencies/results, raw vectors,
class resets, physical aliases, retained reward-to-refresh chains and exact full
stopped prefixes. Whole GC/runtime metadata/class services, Action, uppercase,
TMP/Image setters, Unity liveness/getters/SetActive, Instantiate and Transform
services remain named supplied. No renderer, Unity runtime admission, scheduler,
full acquisition interleaving or engine helper implementation is inferred.

Every report row/field/history/byte is retained with five lossless layers:
raw memory, complete ordered histories, complete named memory-reference maps,
complete authored logical-state maps, then full snapshots. The four
existing family codec formats are preserved; the additional state-map codec interns
complete component/liveness/transform/active/field/data dictionaries. Map names
and every ordered list value remain hashed and retained. Decode with this
family's `expand_report(report)` in reverse: full snapshots → state maps →
memory maps → histories → raw memory.
For parent integration, `expand_memory(post_snapshot_report)` is the compatible
adapter after `audit_report_snapshots.expand_snapshots`.
Family-local assertions check complete round trips, ordering, record names,
64-bit/null values, deep-copy independence and corruption at all five layers.
Both producers also assert equality after complete report expansion. The existing
36 infrastructure tests are separate checks and do not exercise these local codecs.

The final native corpus contains 744 cases: 665 normal returns and 79 exact
native guard/fault outcomes. Twenty-one retained sequences and 33 normal
baselines produce 1,129 complete stopped-prefix probes. The eight-body set has
520 decoded instructions, 512 supported executed instructions, 115 selected
operand assertions and 531 observed native/supplied addresses. Seven decoded
terminal traps and the previously classified bounds dispatch remain unexecuted;
RevealReal's separately classified terminal trap is also explicitly retained.
The graph retains all 84 physical record windows.

Two independently launched final native producers, each preceded by successful
Python syntax compilation and local codec checks, completed successfully and
emitted byte-identical 59,228,698-byte reports. SHA-256 is
`abe9e42f56173f93da5d75e929c6c91a493aa0d16302402a20d99137cf1e6bc5`.
The report contains 441 raw-memory blobs, 4,517 ordered-history blobs,
932 complete memory maps, 2,592 complete state maps and 17,735 full snapshots.
Complete state-map interning reduced the earlier 155,154,992-byte development
checkpoint without dropping any evidence. The final report remains below
100 MiB.

Private final peer:
`B:/CodexTools/DemonBluffReverseEngineering/artifacts/f530404b0f3f_807de4a83df4/character_reward_refresh_join.peer.private.json`.
Post-serialization expansion through the integration adapter verifies all
1,996 complete rows and 49,972 full snapshot occurrences, every one of the
1,129 stopped prefixes, all adjacent retained states and every 84-record raw
memory window. All 36 RE infrastructure tests and local five-layer codec
checks pass. Source, note and canonical report are frozen for parent review.

Run `reverse_engineering/scripts/audit_character_reward_refresh_join.py` with
pinned game and Dumper directories as positional arguments and `--output`.
Compile Python syntax first and set PYTHONPATH to the private python-emulation
directory. Frozen earlier source/note/report families remain unchanged.
