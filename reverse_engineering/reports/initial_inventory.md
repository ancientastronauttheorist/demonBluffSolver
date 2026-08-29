# Initial IL2CPP Inventory

Build: `f530404b0f3f_807de4a83df4`  
Steam build: `23084916`  
Unity: `2022.3.10f1`, Windows x86-64, IL2CPP metadata 29/29.1

## Il2CppDumper baseline

- 87 IL2CPP images were recovered.
- The complete metadata contains 12,540 types.
- Game-owned `Assembly-CSharp.dll` occupies TypeDef indices 5,422–6,191.
- `Assembly-CSharp.dll` contains 770 types:
  - 696 classes
  - 62 enums
  - 7 structs
  - 5 interfaces
- Those types declare 2,931 offset-bearing fields and 4,207 methods.
- 4,133 of those methods have native RVAs; 74 are abstract/generic/runtime-only.
- `script.json` contains 158,195 method mappings, 97,885 candidate native
  addresses, 9,839 metadata symbols, and 14,380 string literals across all
  images and generic instantiations.

The reviewable type ledger is in
[`../symbols/f530404b0f3f_807de4a83df4/assembly_csharp_types.json`](../symbols/f530404b0f3f_807de4a83df4/assembly_csharp_types.json).

## Higher-level recovery

Cpp2IL release `2022.1.0-pre-release.21` produced only method stubs for this
build. The pinned development commit `cae273a255d3` instead reported generated
IL for 18,325 eligible native methods across the recovered assemblies. ILSpyCmd
11.0 then rendered `Assembly-CSharp.dll` into 558 C# files and roughly 165,000
lines locally.

This is a major navigation aid, not source truth. The generated game C# still
contains 8,993 explicit decompiler-issue markers across 343 files, including
2,240 unresolved helper addresses and 348 unresolved indirect calls. Simple
methods such as `Health.Damage`, `Health.Heal`, `Health.ResetHp`, and
`Gameplay.ChangeGameplayState` recover cleanly; complex Unity event and generic
code can be noisy or structurally wrong. Every important method therefore needs
native Ghidra confirmation or behavioral validation.

## First missing-offset candidates

The dump exposes all three offsets previously listed as blocking full autonomy:

- `Gameplay.GameplayState`: static field `+0x28` with the full state enum.
- `Gameplay.CurrentScript`: static field `+0x30`, pointing to
  `CharactersCount`; `town` is `+0x14` and `outs` is `+0x1C`.
- `PlayerController.PlayerInfo`: static field `+0x0`, then
  `PlayerInfo.health +0x10 -> Resource.value +0x10 -> current +0x1C`.

These are metadata-backed pointer-chain candidates, not yet live-validated.
They must be checked against screenshots and controlled state changes before
`memory_reader.py` treats them as authoritative.

## Next gate

Import the native binary into Ghidra, apply the 158k method mappings and
metadata/string labels, run x64 analysis, then cross-check the three candidate
pointer chains in a live game.
