# CharacterData consumer leaves and skin selectors

This audit executes eight exact `CharacterData` consumers offline. It closes
standalone native evidence for art, art type and name consumers that earlier
CharacterView reports supplied. Those earlier reports are unchanged; a composed
CharacterView-to-CharacterData execution remains a separate integration boundary.
Unity Object comparisons and runtime metadata/class initialization remain explicit
supplied services. Localization, flavor selection, hints, preferences, skin
loading/unlocking and other providers are outside this audit.

## Exact declarations and native bounds

The pinned class is `CharacterData : ScriptableObject, ICharacterLocData,
ICardData`, TypeDefIndex 5845. Every selected metadata declaration has an exact
instance signature and `iii` type signature. The body bounds and next managed
entry are verified from file-backed complete reads and metadata; the four skin
selectors have matching unwind entries. The four folded leaves have no unwind
entry, and their complete decoded load/ret bodies are pinned directly.

| Method ID | Method | RVA | Body end exclusive | Shared declarations |
| --- | --- | --- | --- | --- |
| tdi5845.m0000 | GetCharacterName | 0x3B4BE0 | 0x3B4BE5 | 160 |
| tdi5845.m0001 | GetIWas | 0x33E8D0 | 0x33E8D5 | 136 |
| tdi5845.m0002 | GetGender | 0x3B4CF0 | 0x3B4CF4 | 30 |
| tdi5845.m0006 | GetTranslation | 0x3B4DA0 | 0x3B4DA8 | 8 |
| tdi5845.m0007 | GetArt | 0x3B4AB0 | 0x3B4B39 | 1 |
| tdi5845.m0008 | GetAnimatedArt | 0x3B4990 | 0x3B4A19 | 1 |
| tdi5845.m0009 | GetArtType | 0x3B4A20 | 0x3B4AA3 | 1 |
| tdi5845.m0018 | GetArtistName | 0x3B4B40 | 0x3B4BD7 | 1 |

Exact next entries and all `CC` alignment bytes are checked and recorded. The
shared declaration counts identify folded bodies, not additional classified
methods. Only the eight CharacterData declarations are established here; no
other owner or interface declaration is promoted by a shared RVA.

The pure leaves return `characterName` at `+0x28`, `iWasName` at `+0x30`, raw
gender DWORD at `+0x38`, and `translation` at `+0x148`. They perform no metadata,
translation, engine service or allocation. Pointer fields may return null. The
gender load uses EAX and therefore zero-extends its exact 32-bit pattern into
RAX; negative/non-enumerator diagnostic patterns are preserved without filtering.
The exact EGender enum is TypeDefIndex 5956: Female 0, Male 10, They 20. The
translation reference type is the exact CharacterLoc TypeDefIndex 5972 class,
whose methods are not executed.

## Art defaults and current skin

The three art selectors first load the Unity Object TypeInfo and capture
`CharacterData.currentSkin` at `+0xC0`. They then check the Object runtime class
DWORD at `+0xE0` and, when zero, call the explicitly supplied initializer at
`0x281D90`. The already captured skin reference survives that service.

They call the exact supplied `UnityEngine.Object.op_Equality` gateway at
`0x1C822C0`, with captured skin in RCX, null in RDX, and zero R8 MethodInfo.
The branch consumes only AL. Under the supplied null/destroyed-object comparison
profiles, nonzero AL chooses the default:

- GetArt returns **art_cute at CharacterData `+0x98`**. It does not return the
  separate `art` field at `+0x90` and does not seek another fallback if null.
- GetAnimatedArt returns `art_animated` at CharacterData `+0xA8`.
- GetArtType clears EAX and returns Default 0.

For zero AL, each caller reloads currentSkin from CharacterData, checks that
reloaded pointer, and returns the selected SkinData field: art `+0x38`,
animated_art `+0x40`, or raw type DWORD `+0x50`. These offsets are pinned to the
exact SkinData class, TypeDefIndex 5945. Skin sprite outputs may be null. The
art type load uses EAX, including exact zero-extension of raw non-enumerator or
negative bit patterns. The exact EArtType enum is TypeDefIndex 5946: Default 0,
Clipping 10. No class, asset or renderer interpretation is inferred for other
raw DWORD values.

Thus a class-initializer callback can change currentSkin while the comparison
still receives the old captured record. The eventual selected skin fields come
from the reloaded current record. Clearing currentSkin after capture can reach
the native reload guard even though the supplied comparison selected skin.
Replacing it with the same record is an observed receiver-alias path, not a
distinct allocation. Metadata-time changes occur before capture and are
consumed as the initially captured record.

## Artist credit and captured literal

GetArtistName initializes two metadata slots, then captures both the exact
default literal and currentSkin before the supplied Object class initialization.
The default literal slot is `0x2714768`, resolved by decoded RIP-relative
operands to the exact ScriptString value **normandia**. Object TypeInfo is at
`0x2718BF0`. Neither slot is guessed from the display method name.

The caller invokes supplied `UnityEngine.Object.op_Inequality` at `0x1C82480`,
with the same captured-skin/null/zero-MethodInfo ABI. A zero AL returns the
already captured default literal. A nonzero AL reloads currentSkin and returns
that skin's `artistName` at `+0x20`, which may itself be null. A service-time
literal-slot replacement after capture does not change the returned default
identity. A metadata-time replacement precedes capture and does change it.

The fixture's literal has authored length/UTF-16 storage matching the pinned
value; no native string decoder, allocator, interning or valid unused class
header is inferred from the diagnostic window.

## Supplied boundary and verification

Each profile retains complete CharacterData, two SkinData records, the Object
class, output/translation/literal records and metadata slots. These `0x180`,
`0x100` and `0x80` diagnostic windows are not managed object size assertions.
Unconsumed sentinel references are not interpreted as valid typed objects.
Every unrelated byte is checked; pure leaves require the complete initial and
final snapshots to match and emit no services. Legal shared Sprite and string
output references are preserved rather than deduplicated or cloned.
Per-invocation write permissions record only completed class-initializer and
callback effects. Requested callbacks that never execute, or services stopped
before their effects, grant no writable ranges. Metadata slots must match their
initial identities except completed literal replacements; flag transitions must
match the exact reached native byte-1 writes. Skipped initializer and pure-leaf
callback profiles explicitly verify that these requested effects do not occur.

Raw metadata flags and the complete class word are recorded. Nonzero flags,
including `0xFE`, skip metadata services; cold flags become byte 1 only after
the caller's initialization sequence. A nonzero class word skips the supplied
initializer, whose authored normal effect sets the word to 1. The Unity
comparison profiles separately model null, live and destroyed-like results;
they do not execute Unity object lifetime rules. Explicit forced full return
patterns check AL values 0, 1, `0x80` and `0xFF` independently of upper RAX bits.

Callbacks can change the current skin, default sprites, skin outputs, or literal
slot at metadata, class-init or comparison boundaries. Service-entry snapshots
and returned pointer/DWORD assertions distinguish captures from later reloads.
Normal returns verify the original Win64 stack, all eight integer nonvolatiles
and XMM6–XMM15. Supplied services poison integer volatiles and XMM0–XMM5.
All normal, warm, and skin-replacement stop baselines require exact whole
event/snapshot prefixes and matching final stopped state. Native owner-access
faults and null-reload guards are recorded; controlled stops do not emulate
engine exception unwinding.

The corpus contains 354 standalone normal, edge, mutation and alias profiles,
four retained consumer sequences, 20 baselines and 56 exact service-entry stops.
All nontrap instructions execute; the four recorded terminal `int3` instructions
after native null-guard gateways are deliberately unexecuted.

## Reproduction

```powershell
python -m py_compile reverse_engineering/scripts/audit_character_data_consumers.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
$env:PYTHONPATH = 'B:/CodexTools/DemonBluffReverseEngineering/python-emulation'
python reverse_engineering/scripts/audit_character_data_consumers.py `
  --game-root 'B:/SteamLibrary/steamapps/common/Demon Bluff Playtest' `
  --dumper-root 'B:/CodexTools/DemonBluffReverseEngineering/artifacts/f530404b0f3f_807de4a83df4/il2cppdumper-v6.7.46' `
  --output 'B:/CodexTools/DemonBluffReverseEngineering/artifacts/f530404b0f3f_807de4a83df4/character_data_consumers_peer.json'
```

The producer verifies GameAssembly and both Dumper artifacts through the
checked-in manifests for `f530404b0f3f_807de4a83df4`. The public report is
`reports/f530404b0f3f_807de4a83df4_character_data_consumers.json`. It contains no
copied private native method bytes.

Two successful independent final producers, each preceded by Python syntax
compilation, produced byte-identical reports: 13,255,274 bytes, SHA-256
`6a8984cbba9cf66deab23a3d889db5b7e897bf48c4554da62e565d6f2bdd1b23`.
The independent private peer is
`B:/CodexTools/DemonBluffReverseEngineering/artifacts/f530404b0f3f_807de4a83df4/character_data_consumers_peer.json`.
All 36 currently collected reverse-engineering infrastructure tests passed,
including the 32 earlier infrastructure tests and four snapshot codec tests.
There are 149 decoded instructions, 145 executed nontrap instructions, and
26 direct instruction assertions plus decoded call-count checks. No Rust build,
simulation suite, live game or Python bridge regression was run by this audit.
