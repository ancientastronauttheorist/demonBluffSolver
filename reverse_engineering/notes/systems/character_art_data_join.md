# Character art to CharacterData composition

This audit executes the four exact Character art bodies with actual
CharacterData.GetArt, GetAnimatedArt and GetArtType in one physical offline
Actor/Data/Skin/Image graph. Existing standalone art, Data and View reports
remain unchanged. Character.GetCharacterBluffIfAble stays an explicitly supplied
appearance service, alongside runtime metadata/class initialization, Unity
Object comparisons, Component/GameObject operations and Image.sprite writes.
No engine lifetime, renderer, animation scheduler or appearance-selection
implementation is inferred.

## Exact native scope

| Method ID | Method | RVA | Complete body end exclusive |
| --- | --- | --- | --- |
| tdi5487.m0020 | Character.ReInitPreferences | 0x367890 | 0x367964 |
| tdi5487.m0021 | Character.SetupArt | 0x3688B0 | 0x3689C1 |
| tdi5487.m0022 | Character.ShowAnimatedArt | 0x368B40 | 0x368C14 |
| tdi5487.m0023 | Character.HideAnimatedArt | 0x365260 | 0x365334 |
| tdi5845.m0007 | CharacterData.GetArt | 0x3B4AB0 | 0x3B4B39 |
| tdi5845.m0008 | CharacterData.GetAnimatedArt | 0x3B4990 | 0x3B4A19 |
| tdi5845.m0009 | CharacterData.GetArtType | 0x3B4A20 | 0x3B4AA3 |

Every selected declaration has one exact metadata row at its RVA. The exact
owners are Character TypeDefIndex 5487 and CharacterData TypeDefIndex 5845.
Character.SetupArt's metadata ABI is `void (Character*, Sprite*, int32_t,
MethodInfo*)`; the other three Character methods are exact void instance
methods. All three Data consumers preserve their original exact instance
signature and return type. Existing source verifiers provide pinned declaration,
field and decoded operand evidence only; their separate fixture allocations are
not joined or transplanted into execution.

The new audit pins complete Character unwind ranges, including each terminal
int3 following the native guard call. These four instructions were excluded
from the earlier standalone caller decode but are included here. Next managed
entries and remaining CC alignment bytes stay separately verified. Each Data
selector uses its complete previously pinned unwind body. No folded owner or
interface declaration is promoted by RVA.

## Two appearance selections and one captured sprite

ReInitPreferences, ShowAnimatedArt and HideAnimatedArt first capture actor.dataRef
at +0x50 before supplied class initialization and equality. If the supplied AL
indicates null/destroyed, they subsequently capture actor.bluff at +0x58 and
perform the analogous gate. Callback writes after either capture do not change
the corresponding comparison argument. A second class initialization after
the first equality is exercised by an explicit authored class-word reset.

When not suppressed, each wrapper calls supplied GetCharacterBluffIfAble twice.
The first returned Data record supplies the sprite. ShowAnimatedArt executes
actual GetAnimatedArt; ReInitPreferences and HideAnimatedArt execute actual
GetArt. The returned sprite is captured in the wrapper's RDI before the second
appearance service. The second selected record supplies actual GetArtType and
can differ from the first, including when a callback at the first getter's
Object comparison changes the supplied appearance selection.

Consequently a skin-normal or skin-animated sprite from the first record can be
paired with the raw type from another record. The native caller preserves that
sprite across the complete second getter and passes the original full pointer
to actual Character.SetupArt. The oracle keeps both selection identities and
ordered getter results; it does not substitute a single coherent Data record
for this two-read behavior.

GetArt and GetAnimatedArt capture CharacterData.currentSkin +0xC0 after metadata
initialization and before supplied Object class initialization. Nonzero
comparison AL selects their defaults at +0x98 art_cute or +0xA8 art_animated.
The separate unused art +0x90 holds another opaque sprite and is not a fallback.
Zero AL reloads currentSkin and consumes SkinData.art +0x38 or animated_art
+0x40. GetArtType follows the same capture/reload sequence and returns zero for
the default, or zero-extended SkinData.type DWORD +0x50 for the skin branch.
SkinData is exact TypeDefIndex 5945; EArtType is exact TypeDefIndex 5946, with
Default 0 and Clipping 10.

Callbacks can replace or clear a skin, retain the same skin, update default or
skin sprite fields, or replace a raw type. Two explicitly authored supplied
callbacks reset the Object class word and then replace/clear currentSkin during
the next initialization, exercising the native old capture and later reload in
both the first sprite getter and later type getter. Entry-time skin and actual
capture observations are separate. Clearing a reloaded skin reaches the exact
native guard; no callback is invented at a pure native getter return.

SetupArt first compares its captured produced sprite for suppression. Only an
exact raw DWORD 10 selects actor.clippingArt +0x30; other values select actor.art
+0x28. It captures each Component's supplied GameObject result for SetActive,
then reloads the actor's Image field before setting its sprite. Aliased Images
or GameObjects retain ordered writes, including a shared GameObject that is
enabled and later disabled during the same call. Nullable sprites and legal
sprite aliases retain their exact identities; no renderer object is synthesized.

## Supplied ABI, physical retention and independent oracle

Actor/sprite and skin comparisons have separately supplied full return patterns.
All gates consume only AL. Actor and skin profiles cover low bytes 0, 1, 0x80
and 0xFF with nonzero upper register bits; skin return bytes are independently
varied across the two getters. GetArtType's raw signed/non-enumerator bit patterns
are zero-extended in EAX and forwarded through R8D, while sprite pointers stay
full width in RDX. Actual SetupArt receives a zero R9 MethodInfo. Object
comparison inputs have full null RDX and zero R8 MethodInfo; Data consumer calls
have full zero RDX MethodInfo. SetActive consumes DL and records the full upper
RDX bits independently.

Every supplied event, including controlled stops, records full raw
RCX/RDX/R8/R9, observed stack return RVA and active native phase. Unused register
values remain diagnostics rather than inferred additional parameters. Supplied
services poison integer volatiles and XMM0-XMM5. Each normal Data return and
outer caller return verifies stack, all eight integer nonvolatiles and
XMM6-XMM15. Phase tracking only observes entries and returns; the method bodies
still execute natively.

The graph retains complete authored windows for Actor, three Data records,
three SkinData records, Images, GameObjects, sprites and diagnostic class
records. These windows are not object-size assertions, and unused sentinel
fields are not valid typed-reference claims. Per-invocation writable ranges
come only from completed supplied callbacks/class effects. Each reached native
flag write is checked at its exact byte width/value and instruction boundary;
all other flags and the metadata slot must remain unchanged. Complete unrelated
memory retention is verified even when a phase is skipped or stopped.
The inherited logical data_types table remains an unused diagnostic surface;
actual getter results come from the separately recorded physical Data/Skin
fields, including alias and mutation profiles, rather than that old supplied
getter table.

An independent ordered model runs before pooling for every profile and retained
call without an injected service stop. It reads initial physical field bytes,
models each exact metadata flag, class word, captured reference, reload and
supplied callback, and independently predicts the complete service sequence,
native getter results, full final physical bytes and ordered UI service states.
Injected-stop profiles additionally require exact whole event/snapshot prefixes
and final state at the stopped entry. Native guards are explicitly modeled;
controlled stops do not emulate engine exception unwinding.

## Reproduction

```powershell
python -m py_compile reverse_engineering/scripts/audit_character_art_data_join.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
$env:PYTHONPATH = 'B:/CodexTools/DemonBluffReverseEngineering/python-emulation'
python reverse_engineering/scripts/audit_character_art_data_join.py `
  --game-root 'B:/SteamLibrary/steamapps/common/Demon Bluff Playtest' `
  --dumper-root 'B:/CodexTools/DemonBluffReverseEngineering/artifacts/f530404b0f3f_807de4a83df4/il2cppdumper-v6.7.46' `
  --output 'B:/CodexTools/DemonBluffReverseEngineering/artifacts/f530404b0f3f_807de4a83df4/character_art_data_join_peer.json'
```

Manifest hashes pin the GameAssembly and Dumper artifacts to build
f530404b0f3f_807de4a83df4. The report uses existing lossless authored-memory and
full-snapshot pooling only after all assertions pass, verifying exact round
trips at both layers. Expand snapshots first, then expand authored memory using
the existing audit_character_oracle_reveal_join codec. Every physical byte and
snapshot field remains recoverable. Reports contain no copied private native
method bytes.

The final corpus contains 535 profiles, six retained sequences, 16 baselines
and 216 exact stopped event/snapshot prefixes. There are 41 direct instruction
assertions and 360 decoded instructions; all 353 nontrap instructions execute.
Seven terminal int3 instructions after native guard gateways remain unexecuted.
Both final processes completed successfully after separate syntax compilations,
producing byte-identical 27,698,537-byte reports with SHA-256
`f7b872cb1f425751218b66ac6361d4a9c44d81c411bd814c97ab4c3417c1cf4b`.
The independent private peer is
`B:/CodexTools/DemonBluffReverseEngineering/artifacts/f530404b0f3f_807de4a83df4/character_art_data_join_peer.json`.
All 36 reverse-engineering infrastructure tests passed. No Cargo build,
simulation suite, Python bridge regression or live game was run by this audit.
