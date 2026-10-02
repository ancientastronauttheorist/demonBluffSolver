# Character Oracle presentation callers

Pinned build `f530404b0f3f_807de4a83df4`. This separate offline audit executes
complete Character.OracleEyeActive (`tdi5487.m0030`, `0x3675A0..0x36778D`)
and HideOracleInfo (`tdi5487.m0031`, `0x3654F0..0x36563D`). Exact signatures,
TypeSignatures, managed following entries, unwind families, final instructions
and padding are verified. Sixteen decoded operand assertions bind consumed
fields, capture/reload points and widths. All 192 caller instructions execute,
including the native null-helper call and every normal return path.
The corpus contains 129 cases (104 normal returns, 25 native-null stops), two
retained three-call sequences and 26 exact controlled service-stop prefixes.
Its 209 observed execution addresses include supplied native gateways and
two authored colour virtual-method gateways.

OracleEyeActive skips RevealOrder.Init only for the exact state DWORD Hidden=5.
Other state values request Init with the zero-extended order DWORD. The pinned
enum is Hidden=5, Alive=10, Dead=20, Revealed=30. Both methods read the signed
List count DWORD at `+0x18` directly. They initialize the get_Count MethodInfo
as metadata, but neither calls a Count implementation. When count exceeds one,
supplied GetItem receives the exact physical List and get_Item MethodInfo,
with index DWORD zero for OracleEyeActive and count-minus-one for HideOracleInfo.
The folded GetItem entry is verified at `0xB22150`; its implementation remains
supplied. The selected physical ActedInfo's desc at `+0x10`, including null,
is passed to supplied Acted.Act.

The main Acted receiver is captured before GetItem. An authored GetItem mutation
that replaces it leaves the earlier Acted.Act receiver intact. After Acted.Act,
the caller reloads Character.acteds for highlight and colour publication.
Clearing the main field inside GetItem therefore still permits the captured
speech request, then stops at the later reload guard. Replacing it publishes
speech to the earlier Acted and subsequent UI changes to the replacement.

OracleEyeActive enables the reloaded Acted's highlight GameObject. Its DL-only
write retains the upper RDX bits from the earlier speech pointer; the audit
records both the full register and low-byte Boolean. It then calls the Image
virtual colour getter with the hidden return-buffer pointer in RCX, Image
receiver in RDX and exact MethodInfo in R8. The supplied getter writes sixteen
bytes and returns that buffer pointer. Actual native instructions copy all four
DWORDs through RAX into the captured Acted.savedArrowColor at `+0x40`.
They reload arrowImage after the getter and call the virtual setter with the
exact sixteen-byte native colour literal `[0, 1, 1, 1]`, bits
`[0x00000000, 0x3F800000, 0x3F800000, 0x3F800000]`, at file-backed RVA
`0x1F34CB0`. Clearing arrowImage inside the getter retains the saved colour
write before the next null guard. Signed zero and NaN payloads in the supplied
initial colour remain exact DWORDs.
The getter observation also retains the actual captured RDI Acted identity,
separately from its ABI arguments. Replacing Character.acteds inside the earlier
SetActive service leaves this saved-colour recipient and its arrow receiver
captured, even though the current Character field names the replacement.

Next OracleEyeActive requests supplied GetCharacterBluffIfAble on this exact
Character. Its selected data's picking byte at `+0x13E` must be nonzero and
the Character uses DWORD at `+0xDC` must be signed-positive before enabling
pickHighlight. Only exact Dead=20 with killedByDemon byte zero proceeds to
the Unity bluff inequality. The bluff pointer is captured before the supplied
Object class initializer, when its DWORD is zero. Inequality tests only AL.
On a nonzero result, the caller requests CharacterView.AnimateIn, then reloads
both view and bluff for CharacterView.Init. Authored bluff replacement inside
the inequality service reaches that later Init argument. Clearing view inside
AnimateIn preserves earlier requests and stops before Init.

HideOracleInfo always requests RevealOrder.Hide. When count exceeds one it
restores the last supplied description, hides the reloaded Acted highlight,
and publishes that Acted's saved sixteen-byte arrow colour. It always hides
pickHighlight afterward, even when no description branch runs. It obtains
the current CharacterView's GameObject and tests only AL from supplied
activeSelf. A nonzero result causes a fresh view load for AnimateOut; clearing
the field inside activeSelf reaches the intervening guard. No class initializer
is called by HideOracleInfo. Warm metadata bytes and Object class DWORDs retain
their exact noncanonical values.

The normal corpus varies all four named states, counts zero/one/two, signed
uses negative/zero/positive and canonical/noncanonical picking bytes. Further
probes cover three-entry last-item selection, null text, absent/destroyed and
aliased bluff assets, selected data/bluff outcomes, aliased description/picker
GameObjects, raw AL responses, cold runtime state, native null stops and
service mutations. Negative count DWORDs are explicit diagnostic storage;
they are not valid managed List state. Noncanonical supplied Boolean return
bytes establish the caller's AL consumption only. Positive List counts fit
the independently authored three-slot backing array. The List `+0x10` points
to that distinct array, whose QWORD length and physical ActedInfo slots are
retained along with the List version DWORD.

Snapshots retain the authored memory windows: Character `0x200`, each of two
Acted records `0x80`, List and backing array `0x40` each, three ActedInfo records
`0x30` each, two CharacterData records `0x160` each, two Image records `0x40`
each and CharacterView `0x100`. The pinned Character managed footprint ends
at `0x1B8`; the larger `0x200` snapshot is diagnostic backing storage and does
not enlarge that object. Unconsumed pointer bytes and fixture padding do not
establish complete valid typed runtime objects. Every byte outside exact native
saved-colour writes and explicitly authored mutations is checked retained.
Supplied image colour, active state, speech and reveal records are separate
fixture bookkeeping; they do not reconstruct those services' internal storage.
Two retained Oracle/Hide/Oracle sequences preserve chronology, saved colours,
class/metadata state and physical GameObject aliases. Exact service-stop
prefixes retain all earlier requests and native writes without rollback.

Whole Acted.Act, RevealOrder.Init/Hide, GetCharacterBluffIfAble and
CharacterView.AnimateIn/AnimateOut/Init bodies remain game-owned supplied
boundaries with pinned metadata. GetItem, metadata/class services, Unity
inequality, SetActive, GameObject/activeSelf and Image virtual methods are
also explicitly supplied. Their receiver/argument/result identities and ABI
are verified independently at each caller entry. Actual renderer, scheduler,
engine/service implementations and native exception unwinding remain unclaimed.
Private native code and bodies remain off-repository. No live game is accessed.

Executable: [audit_character_oracle_presentation.py](../../scripts/audit_character_oracle_presentation.py).
Report: [f530404b0f3f_807de4a83df4_character_oracle_presentation.json](../../reports/f530404b0f3f_807de4a83df4_character_oracle_presentation.json).
Two independent final producers emitted identical 9,180,722-byte reports,
SHA-256 `93a608ce21cbc4dc2634b4967b421d8ce2c424d89f7c97deefeaf8146655596c`.

```powershell
$env:PYTHONPATH='B:/CodexTools/DemonBluffReverseEngineering/python-emulation'
python reverse_engineering/scripts/audit_character_oracle_presentation.py GAME_ROOT DUMPER_ROOT --output REPORT
```

## Separate actual RevealOrder composition

The [Oracle-to-RevealOrder audit](character_oracle_reveal_join.md) now executes
those two native callees inside these callers with one physical state. This
standalone audit/report retains its original supplied boundary; the new report
verifies capture/reload chronology, aliases and full stopped prefixes. Other
game-owned and engine services remain supplied in that separate join.
