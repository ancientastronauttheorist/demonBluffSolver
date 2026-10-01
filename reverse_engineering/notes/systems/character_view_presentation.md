# CharacterView presentation and disguise composition

Pinned build `f530404b0f3f_807de4a83df4`. This separate offline audit executes
four complete CharacterView bodies, including actual SetupArt, and composes
them with the actual Character ShowDisguise/HideDisguise callers from
[the presentation caller audit](character_presentation_helpers.md).
The corpus contains 193 standalone cases, 22 joined cases, two retained
three-call sequences and 92 exact controlled service-stop prefixes.
There are 177 normally returning standalone cases and 17 normal joins;
the remaining cases preserve native null-guard state.

| Method | Method ID | Entry |
| --- | --- | --- |
| AnimateIn | tdi5511.m0007 | 0x363D50 |
| AnimateOut | tdi5511.m0008 | 0x363DF0 |
| Init(CharacterData) | tdi5511.m0003 | 0x363F10 |
| SetupArt(Sprite, EArtType) | tdi5511.m0006 | 0x3643F0 |

Exact signatures and declarations pin CharacterView, CharacterData, consumed
fields and supplied art helpers. Complete verified unwind families decode
235 body instructions; 231 execute. The four remaining instructions are the
native bounds-exception gateway and three terminal traps. Stable arrays and
service mutations do not synthesize a concurrent size change between the
adjacent loop/index guards. That exceptional race and native exception
unwinding remain unclaimed. Twenty instruction assertions and six exact
RIP-relative literal bindings pin stores, field captures/reloads and ABI widths.
Normal calls verify stack balance, all eight integer nonvolatiles and XMM6–15.

AnimateIn/AnimateOut capture the current animId before checking DOTween's
class-init DWORD and calling the supplied initializer when it is zero.
They pass that captured pointer to supplied DOTween.Kill with only DL written
to one; poisoned upper RDX bits remain observable and are not interpreted as
the Boolean argument. Kill's supplied count result is ignored. The native
callers then pass current canvasGroup to supplied DOFade, with exact float32
end bits `0x3F800000` for AnimateIn and positive zero for AnimateOut, and duration
bits `0x3E4CCCCD` (`0.2f`). They reload animId after DOFade and tail-forward the
supplied tween pointer and exact generic MethodInfo to supplied SetId.

Authored initializer/Kill/DOFade mutations replace animId. Kill retains the
earlier captured identifier, while SetId receives the replacement. Null animId,
canvasGroup and supplied tween results are forwarded without a caller null
guard. The supplied services explicitly accept these probes; this does not
establish how the actual DOTween bodies handle them. Warm metadata bytes
`0x80` and class DWORD `0xDEADBEEF` are retained. No alpha change, animation
completion or scheduling is inferred from these request records.

Init writes dataRef before its GC barrier, then rejects a null captured data
argument. A stopped barrier retains that pointer write. The rest of the method
uses its captured data argument even if the authored barrier changes view.dataRef.
It publishes cardBgColor to bgs, then cardBorderColor to each border Image in
physical array order. RGBA units retain their exact DWORD bits, including
signed zero and a NaN payload. Duplicate border pointers produce repeated
virtual colour requests to that same Image. The native loop captures the
array pointer once and rereads its length and each element; clearing the view's
array field during a border callback does not replace the captured array,
while clearing a later element reaches its null guard and shrinking length
skips remaining requests.

The loop reads the signed low DWORD of the supplied array length. Diagnostic
poisoned upper DWORDs do not affect it, and a negative low DWORD skips the loop.
These raw-header probes establish the consumed width, not valid managed array
allocations of those poisoned lengths. The fixture's retained backing-memory
view contains only three authored elements and all executed indexes remain
within it.

Next Init sets the background sprite, reads captured data.characterName and
captures chName before supplied String.ToUpper. It forwards the supplied
uppercase result to the TMP virtual setter, including null. Clearing chName
during ToUpper leaves the captured setter receiver in use. It tints art and
clippingArt with the exact native four-unit white literal, then requests art
and art type from the captured CharacterData through named supplied services.
The type result is forwarded through EAX/R8D, retaining only its low DWORD.
After actual SetupArt returns, Init reloads and tints bg white. Every native
field store and service-entry snapshot is retained; missing UI references do
not roll earlier colour, sprite, text or dataRef effects back.

Actual SetupArt selects clippingArt only when the type DWORD is exactly 10.
Every other value, including 20, `0x8000000A` and `0xFFFFFFFF`, selects art.
It obtains the selected Image's GameObject, activates it, reloads the Image
and sets its supplied sprite, then obtains the other Image's GameObject and
deactivates it. A SetActive mutation can clear the selected Image before the
reload; its activated GameObject persists at the resulting null guard.
Null sprite pointers are forwarded. Aliased art/clipping/bg/bgs Image fields
share physical colour/sprite state and GameObject requests; enabling then
disabling that same GameObject leaves it disabled.

Joined ShowDisguise executes actual AnimateIn and Init before its hint callback.
Init failure retains Character.showDisguise, completed tween requests and
earlier UI/data writes while suppressing the later hint callback. Joined
HideDisguise executes actual AnimateOut before its raw hover/callback test.
Retained ShowDisguise/HideDisguise/ShowDisguise sequences keep class/metadata
state, callback chronology and distinct supplied tween identities. The earlier
caller-only report remains unchanged.

DOTween.Kill/DOFade/SetId, Image/TMP virtual setters, Unity GameObject/active
services, String.ToUpper, CharacterData.GetArt/GetArtType, metadata/class
initialization, GC and hint callback implementation remain explicit supplied
boundaries. Supplied image/text/GameObject/tween state is fixture bookkeeping,
not reconstruction of those implementations. Receivers and supplied class
storage are valid; optional consumed fields and service mutations vary.
The 326 observed execution addresses include supplied native gateways and
three authored virtual/callback gateways. Other actor/view bytes are checked
retained except exact native stores and explicitly authored mutations.
Private native bytes and bodies remain off-repository.

```powershell
$env:PYTHONPATH='B:/CodexTools/DemonBluffReverseEngineering/python-emulation'
python reverse_engineering/scripts/audit_character_view_presentation.py GAME_ROOT DUMPER_ROOT --output REPORT
```

Report: `f530404b0f3f_807de4a83df4_character_view_presentation.json`.
Two independent final producers emitted identical 13,321,805-byte reports
(SHA-256 `6e63e4d368df750087ca40a6fd17120311ab5fb3ea0db78453d8cbb1ecd4e083`).
