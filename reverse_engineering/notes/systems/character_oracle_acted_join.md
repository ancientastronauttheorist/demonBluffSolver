# Character Oracle callers joined to immediate Acted.Act

Pinned build `f530404b0f3f_807de4a83df4`. This separate offline audit executes
Character.OracleEyeActive (`tdi5487.m0030`, `0x3675A0..0x36778D`) and
HideOracleInfo (`tdi5487.m0031`, `0x3654F0..0x36563D`) with actual immediate
Acted.Act(string) (`0x35DD10..0x35DDB4`). It replaces only the whole Acted.Act
supplied boundary in the frozen Oracle audit. RevealOrder.Init/Hide and
CharacterView.AnimateIn/AnimateOut/Init remain named supplied services here.
Their separate joins do not imply this report executes their bodies.

Each producer first runs the frozen immediate Acted surface verifier's 68 native
cases and 200-instruction evidence. It then independently decodes the exact Act
body from the extraction-pinned PE, checks its exact metadata signature and
TypeSignature, following managed entry, raw-backed extent and alignment padding.
Exact Acted (5477) and ActedVersion (5478) declarations bind the field identities;
the Show and ForceRebuildLayoutImmediate metadata entries come from the frozen
verifier. Fourteen new Act instruction/operand assertions join the Oracle's
sixteen pins. All 192 Oracle caller instructions and 44 of 46 decoded Act
instructions execute. Only bounds-gateway call `0x35DDA9` and terminal trap
`0x35DDAE` remain unexecuted, asserted as an exact set. Neither is promoted to
executed coverage. There are 254 observed addresses including supplied gateways.

The corpus contains 331 cases (303 normal returns, 28 native guards), two
retained Oracle/Hide/Oracle sequences, six complete service-stop baselines and
85 exact stopped-prefix fixtures. Stops compare all earlier events to the
baseline prefix and the entire final state to that exact service-entry snapshot.
Normal invocations verify all eight integer nonvolatile registers, XMM6-15 and
restored stack position. Supplied services poison caller-saved integer registers
and XMM0-5; ABI checks require the full pointer/MethodInfo widths used by callers.

## One physical actor, Acted and layout state

The parent captures the physical main Acted before supplied List.GetItem.
Actual Act then uses that same receiver even if the callback clears or replaces
Character.acteds. Two Acted owners have independently retained ActedVersion and
layout-array fields. Their existing highlight, arrow and saved-color fields
remain in the parent state. Two version records retain exact declared size,
blankText/text, animationId and savedScale fields with full diagnostic backing
bytes. Version/array aliases are explicit physical identities. Distinct relocated
component/class/string/RectTransform records do not collide with the Oracle
Actor/history/arrow storage.

Two physical layout arrays retain QWORD length storage and three pointer slots.
Act consumes the low signed length DWORD; upper sentinel bits survive.
Zero/one/two/three lengths, duplicate RectTransforms and null element pointers
are tested. Negative low DWORDs are diagnostic storage, not valid managed array
state. A null RectTransform is passed unchanged to the independently accepted
supplied layout service; no actual engine acceptance is inferred.

Every snapshot retains the parent Actor, Acted owners, List/backing array,
ActedInfo, data/bluff, arrow and View diagnostic windows, plus version/component/
class/string/RectTransform/layout/GameObject/metadata records. Each native
saved-arrow-color store, explicit callback write and completed initializer
DWORD update has an exact allowed byte range. All other bytes remain unchanged.
Only exact classes whose initialization service completed may change E0..E4;
unrelated class and MethodInfo records stay fixed. Sentinel memory and larger
windows are authored diagnostics, not complete valid runtime objects.

The saved report pools only byte windows through 75 SHA-256 memory blobs.
`expand_memory(report)` verifies their hashes and restores full snapshots;
each producer asserts exact round-trip equality after all native prefix checks.
No state, field identity, event or byte window is dropped.

## Capture and reload order

Act preserves the original nullable description argument across metadata work.
It loads/guards Acted.acted and calls supplied ActedVersion.Show with that exact
receiver, original description pointer and zero MethodInfo. Version replacement
inside the Act-only metadata service reaches this load; clearing it there stops
before Show. Clearing/replacing the version during Show leaves its already
captured receiver and later layout traversal intact. Metadata callback effects
are qualified by the verified Act metadata-call return location; parent metadata
requests cannot run them before the Acted entry exists.

Only after Show returns does Act load and capture layoutsToRebuild. A Show
callback replacing that field selects the alternate physical array; clearing it
preserves the completed Show and then reaches the native null guard. Once the
array is captured, later field replacement/clearing during class initialization
or rebuild does not replace the traversal. Signed length is reread on each
iteration, so shrinking that captured array stops subsequent requests. Changing
its second slot during the first rebuild reaches the next RectTransform load.

For each occurrence, Act checks the unsigned bound, captures the element in RSI,
then calls the supplied LayoutRebuilder class initializer if its DWORD is zero.
Replacing the first slot inside that initializer preserves the earlier RSI
capture: the first request still names the old RectTransform, even though the
physical array now contains the replacement. Subsequent elements use current
captured-array contents. ForceRebuildLayoutImmediate receives the exact element
pointer and a fully zero second register/MethodInfo value.

Clearing Character.acteds during Show still permits the actual Act's two rebuild
requests, then the parent stops at its later field reload guard. Replacing it
publishes Show/layout requests through the earlier captured owner and subsequent
Oracle highlight/color requests through the replacement. Parent GetItem and
post-Act mutation profiles retain this distinction. The post-Act callback runs
at the verified native return after all Act body effects, preserving the original
whole-service callback phase without bypassing that body.

Retained Oracle/Hide/Oracle sequences preserve version/array/RectTransform aliases,
parent saved colors, metadata/class state, nullable description identities and
complete supplied Show/rebuild bookkeeping. Independent normal expectations
check first-versus-last history description selection, exact version identity
and ordered array occurrences.

## Remaining boundaries

ActedVersion.Show, runtime metadata/class operations and
LayoutRebuilder.ForceRebuildLayoutImmediate remain supplied. Their represented
effects are description/request and rebuild-count bookkeeping; actual text,
animation, component/layout engine implementations are not reconstructed.
RevealOrder, CharacterView, appearance selection, folded List.GetItem, Unity
Object/Component/GameObject and Image virtual methods likewise remain explicit
supplied boundaries. No constructor, renderer, delayed coroutine/scheduler,
engine event admission or native exception unwinding is claimed. Private native
bytes remain outside the repository; no live game/process/UI is accessed.

Executable: [audit_character_oracle_acted_join.py](../../scripts/audit_character_oracle_acted_join.py).
Report: [f530404b0f3f_807de4a83df4_character_oracle_acted_join.json](../../reports/f530404b0f3f_807de4a83df4_character_oracle_acted_join.json).
Two successful independent final producers emitted identical 29,952,450-byte
reports, SHA-256 `06e28da4e4f08d12e1f3a0bb93a85c6d8aed7cbd7828ad07d3538dc4a3ca8be0`.
Python syntax compilation, the current 36 reverse-engineering tests and focused
diff checks pass.

```powershell
$env:PYTHONPATH='B:/CodexTools/DemonBluffReverseEngineering/python-emulation'
python reverse_engineering/scripts/audit_character_oracle_acted_join.py GAME_ROOT DUMPER_ROOT --output REPORT
```
