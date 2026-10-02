# CharacterView to CharacterData art composition

This audit executes CharacterView.Init, CharacterData.GetArt,
CharacterData.GetArtType and CharacterView.SetupArt together in one physical
offline state. Earlier standalone View and Data reports remain unchanged.
Unity Object comparison, runtime metadata/class initialization, the write
barrier, Image/TMP operations and uppercase conversion remain explicit supplied
services. Oracle callers, animation scheduling, rendering, engine object
liveness and localization are outside this composition.

## Exact native declarations

| Method ID | Method | RVA | End exclusive | Next managed entry |
| --- | --- | --- | --- | --- |
| tdi5511.m0003 | CharacterView.Init | 0x363F10 | 0x3640EC | 0x3640F0 |
| tdi5511.m0006 | CharacterView.SetupArt | 0x3643F0 | 0x3644B2 | 0x3644C0 |
| tdi5845.m0007 | CharacterData.GetArt | 0x3B4AB0 | 0x3B4B39 | 0x3B4B40 |
| tdi5845.m0009 | CharacterData.GetArtType | 0x3B4A20 | 0x3B4AA3 | 0x3B4AB0 |

Every declaration has one exact metadata row at its RVA. The pinned owners are
CharacterView TypeDefIndex 5511 and CharacterData TypeDefIndex 5845. Init has
the exact instance `void (CharacterView*, CharacterData*, MethodInfo*)`
signature; SetupArt has `void (CharacterView*, Sprite*, int32_t, MethodInfo*)`.
The Data consumers have their exact `Sprite*` and `int32_t` instance returns.
The dumper's managed SetupArt type is EArtType. Full unwind bodies, next managed
entries and trailing CC alignment are verified. No shared-RVA declaration is
promoted. Existing verifier code contributes instruction/declaration evidence;
its separate fixture allocations are not transplanted into the executing View
machine.

The exact Init call sites are GetArt at 0x36407D, GetArtType at 0x36408A and
SetupArt at 0x36409B. Init reads CharacterData.characterName directly at +0x28;
there is no GetCharacterName call to join. GetAnimatedArt is not consumed by
this path. Its standalone evidence remains separate.

## Captures, reloads and produced sprites

Init stores its incoming data into view.dataRef at +0x28 before the supplied
write barrier, and retains that argument in RSI. Its background sprite, name,
colors and both art getters use this captured argument. A barrier or later
callback can replace view.dataRef with a different record while both native
getters continue to receive the original argument. Profiles also enter Init
with the alternate Data record and share skin records or sprite identities.

The relevant CharacterData fields are characterName +0x28, art_cute +0x98,
backgroundArt +0xB8, currentSkin +0xC0, cardBgColor +0xF8 and
cardBorderColor +0x108. The unused art +0x90 deliberately holds a different
opaque sprite. GetArt's default comes from art_cute, not that other field.
SkinData TypeDefIndex 5945 supplies art +0x38 and raw EArtType DWORD +0x50.
Exact EArtType TypeDefIndex 5946 values are Default 0 and Clipping 10.

Each native getter initializes its own metadata flag, then captures currentSkin
before checking the Object class word and calling supplied initialization.
The comparison receives that captured skin even when initialization changes
the field. A zero comparison AL reloads currentSkin and consumes the reloaded
skin; a nonzero AL chooses the default. Metadata-time changes occur before the
native capture. Instruction observations record both the entry reference and
the actual later captured reference. Callback profiles replace, clear or retain
the same skin, update default/skin sprite fields, change raw type, and replace
view.dataRef. Clearing a reloaded skin reaches the exact native null guard.

GetArt's full pointer result is captured in Init's RBX before GetArtType runs.
Changes to skin sprite storage during GetArtType therefore do not change the
sprite passed to SetupArt. The type result is zero-extended through EAX and
then R8D. SetupArt receives the captured sprite in RDX and a zero R9 MethodInfo;
only an exact DWORD value 10 selects clippingArt. Other raw values, including
negative patterns, select art. Supplied Image and GameObject operations record
ordered outputs and full register values without constructing a renderer.
Produced sprites are opaque authored identities; null and legal aliases remain
null or aliased throughout the join.

Both Object comparisons receive full RDX=0 and R8=0 MethodInfo from native
caller setup. The corpus independently supplies AL values 0, 1, 0x80 and 0xFF
for both getter calls with nonzero upper return bits. The upper bits do not
determine default selection. Supplied services poison RCX, RDX, R8-R11 and
XMM0-XMM5. SetActive consumes DL while retaining the full supplied upper RDX
pattern in diagnostics. Every normal outer return and each native getter
return verifies stack and all integer/XMM Win64 nonvolatiles.
Every supplied event, including stopped entries, also records full raw
RCX/RDX/R8/R9, the observed stack return RVA and its active native phase.
Tracking actual native entry/return transitions preserves the SetupArt phase
at its tail service even though its restored return address points into Init.
Unused registers remain diagnostics rather than inferred extra parameters;
entry R9/R10/R11 are explicitly authored to avoid preceding-fixture residue.

## Physical retention and stopped prefixes

Every profile retains full authored diagnostic windows for Actor, View, both
Data records, both SkinData records, sprites, components, arrays, class/method
records, callback records and statics. These windows are not managed object
extent assertions, and unconsumed sentinel fields are not treated as valid
typed references. Metadata slots and all inherited metadata flags are also
recorded. Only completed callback/class-initializer writes and the exact
reached native view.dataRef/flag writes grant per-invocation write permissions.
Unrelated physical bytes and all metadata slot identities must remain exact.

Repeated retained Init sequences switch supplied data arguments, preserve warm
metadata/class state and previous supplied component effects, and mutate skin
and type storage between calls without allocating a second execution state.
An explicit comparison callback resets the Object class word after GetArt,
allowing GetArtType's own initializer branch to execute. This reset is an
authored supplied effect, not inferred runtime behavior.

Normal, alias, capture/reload and native-guard baselines require exact complete
event/snapshot prefixes at every supplied entry. Stops occur before service
effects and therefore grant no permissions for those effects. Controlled stops
and native guard gateways do not emulate engine exception unwinding.

There are 241 decoded instructions. Five terminal int3 instructions after
guard gateways remain unexecuted. The array-bounds guard call at 0x3640E6 also
remains unexecuted: the signed iteration test and subsequent unsigned bounds
test are adjacent with no service between them, and valid nonnegative
iterations cannot reach that second guard. No artificial instruction-time
race is introduced to force it. All other 235 instructions execute.

## Reproduction and report encoding

```powershell
python -m py_compile reverse_engineering/scripts/audit_character_view_data_join.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
$env:PYTHONPATH = 'B:/CodexTools/DemonBluffReverseEngineering/python-emulation'
python reverse_engineering/scripts/audit_character_view_data_join.py `
  --game-root 'B:/SteamLibrary/steamapps/common/Demon Bluff Playtest' `
  --dumper-root 'B:/CodexTools/DemonBluffReverseEngineering/artifacts/f530404b0f3f_807de4a83df4/il2cppdumper-v6.7.46' `
  --output 'B:/CodexTools/DemonBluffReverseEngineering/artifacts/f530404b0f3f_807de4a83df4/character_view_data_join_peer.json'
```

The GameAssembly and Dumper manifest hashes are verified for build
f530404b0f3f_807de4a83df4. All semantic, physical-retention, ABI and stopped-prefix
assertions run before report pooling. The report interns authored physical
windows by SHA-256, verifies exact expansion, then uses the shared lossless full
snapshot codec and verifies that expansion too. To recover raw reports, first
call audit_report_snapshots.expand_snapshots, then this script's expand_memory.
Every original byte and snapshot field remains available; the report contains
no copied private native method bytes.

The final corpus contains 221 profiles, four retained sequences, 10 baselines
and 173 exact stopped event/snapshot prefixes. It has 31 direct instruction
assertions plus exact decoded call-count checks. Both final producers completed
successfully after independent Python syntax checks, producing byte-identical
14,243,169-byte reports with SHA-256
`d94756b07caefb796df8b9d4a3478f08bbb0b3dcfccf0cb0f462ecac5842fc4c`.
The independent private peer is
`B:/CodexTools/DemonBluffReverseEngineering/artifacts/f530404b0f3f_807de4a83df4/character_view_data_join_peer.json`.
All 36 reverse-engineering infrastructure tests passed. No Cargo build,
simulation suite, Python bridge regression or live game was run by this audit.
