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
- 4,133 of those methods have direct native RVAs. Of the other 74, 19 generic
  definitions have 25 instantiated native bindings and 55 are abstract methods
  with no body.
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

## Native Ghidra baseline

Ghidra 12.1.3 imported 158,195 method labels at 91,930 unique native entry
points, plus 30,899 metadata labels and 14,380 string labels. No imported RVA
fell outside the image. Full Windows x86-64 analysis completed and saved in
2,131 seconds; an independent second pass completed in 1,290 seconds and wrote
`analysis_timeout_occurred: false`. The resulting private project is roughly
1.5 GB.

The first checked target set contains 13 gameplay/resource methods. Target
metadata names, signatures, and RVAs are validated against `script.json`, and
the exporter additionally requires the expected imported Ghidra symbol at each
address. All 13 native decompilations completed with zero failures. Four target
methods use folded native bodies: `Health.Damage` (3 aliases), `Health.Heal`
(3), `Health.ResetHp` (2), and `CurrentMaxValue.GetValue` (48).

A six-method roster-helper follow-up also exported without failure. It confirms
that always-in-deck probability uses floating-point division and reveals a
current-build refill quirk: `ManageEmptyCharacterList` ignores its input, adds
the Townsfolk source twice, and omits the Demon source. The behavioral
reachability of the resulting duplicate Townsfolk and empty Demon pools remains
unverified.

Native review corrected two material Cpp2IL failures: `CharactersCount` stores
the `outsiders` and `minions` constructor arguments rather than zero, and
`Health.AddMaxHp` increments max, dispatches Add, then explicitly raises its
change and UI notifications. The authored findings and evidence links are in
[`../notes/systems/gameplay_core.md`](../notes/systems/gameplay_core.md).

The deterministic typed-header pipeline parses 151,087 datatypes, restores all
6,159 explicit alignments, validates critical object layouts, and validates 19
selected FunctionDefinitions. The isolated typed project then imported 1,874
reachable datatypes while applying the 13 gameplay-core signatures; the six
roster-helper signatures reused that graph without conflicts. All 19 exact
entry points, metadata labels, prototypes, dynamic `__fastcall` storage, and
the seven-argument constructor ABI passed validation. Typed auto-analysis is
the next gate. The native exports, GDT, and analysis databases remain
local/private.

## Method coverage denominator

The generated coverage ledger accounts for all 4,207 managed methods using
4,158 native bindings and 3,066 unique native bodies. It identifies 107 shared-
body groups containing 1,199 bindings. The first 19 methods are classified
`understood` with eleven `native-static` evidence records; every missing
overlay row remains explicitly `unresolved/not-reviewed`.

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

Run full analysis on the typed project, export and compare the reviewed target
sets, then cross-check the three candidate pointer chains in a live game.
Continue outward from the reviewed gameplay and roster methods while preserving
the 4,207-method coverage denominator.
