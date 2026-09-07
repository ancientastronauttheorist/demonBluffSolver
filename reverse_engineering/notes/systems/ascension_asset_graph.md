# Pinned ascension asset graph

Build `f530404b0f3f_807de4a83df4`, GameData version `v0.630b`.

The [auditor](../../scripts/audit_ascension_assets.py) consumes 61 complete
serialized objects, checking field order against pinned managed declarations,
script bindings against `globalgamemanagers.assets`, external-file mappings,
and exact end-of-object positions. The [report](../../reports/f530404b0f3f_807de4a83df4_ascension_assets_audit.json)
retains list order and duplicate references. It contains configuration values,
not private decompiler bodies or artwork.

## Actual configuration chain

`level0` ProjectContext path `138195`, MonoScript `434`, points through file ID
2 to `sharedassets0.assets` GameData path `21636`, MonoScript `960`. The complete
GameData object is 1,384 bytes. It references:

- Advanced profile `21657`.
- Four Roguelike Standard groups of seven profiles each.
- Two Roguelike profiles, `21671` and `21672`.
- Thirteen Standard profile occurrences, including repeated `21658`.
- All-characters `21673`, debug `21670`, and temporary `21702`.

All references resolve to the 46 complete AscensionsData objects `21657` through
`21702`. Their possible-script references resolve to the 12 CustomScriptData
objects `21703` through `21714`. Inline ScriptInfo, character-count records,
and card-addition records are parsed through the final byte in every object.
The Action delegate in CardAdditionInfo contributes no serialized payload;
its other five integer fields account for every record.

`GameData.allCharacterData` contains 41 entries. This catalogue is distinct
from a profile's starting lists, current pools, possible scripts, and current
picked script. The report preserves all of those separately. For example,
the serialized temporary profile includes legacy Bounty Hunter and Marionette
references absent from the 41-entry catalogue. The catalogue therefore cannot
be treated as an exhaustive runtime candidate universe.

## Compendium provenance correction

Scene path `139347` binds MonoScript `223`, managed **Compendium**. Its 1,080-byte
object contains 27 card-view references, page records with character-data
references, four UI references, and currentPage. Its pages contain the same
41 asset-identity set as GameData's catalogue.

Earlier role notes called this scene object a normal candidate pool. The
reference offsets and identities were correct, but that interpretation was
too broad. Corrected notes and evidence call it the Compendium catalogue.
Membership alone establishes neither a mode's selection policy nor global
runtime reachability. Ordered-Start references are a separate scene object
and are unaffected by this correction.

The already audited native consumers reinforce the distinction:
`Gameplay.GetAllAscensionCharacters` (`0x37C1A0`) reads ProjectContext's
GameData and its temporary ascension, concatenating townsfolks, outsiders,
minions, and townsfolks again. `GetAscensionAllStartingCharacters` (`0x37C3F0`)
uses the same chain and requests type pools 100, 20, 30, 10 in order.
`GetScriptCharacters` (`0x37DC00`) instead concatenates four Gameplay fields.
None of these consumers reads the Compendium object.

## Public Mutant reachability boundary

Public Mutant path `21592`, bound to Skinwalker, has no direct reference in
the decoded GameData, 46 profiles, or 12 custom scripts. This does not prove
absence from dynamic loads, runtime mutations, unexamined scenes, or an entire
game lifecycle. Mode selection, cloning, and temporary-profile writers remain
the next boundary to audit.

```powershell
python reverse_engineering/scripts/audit_ascension_assets.py `
  'B:\SteamLibrary\steamapps\common\Demon Bluff Playtest' `
  --dumper-root 'B:\CodexTools\DemonBluffReverseEngineering\artifacts\f530404b0f3f_807de4a83df4\il2cppdumper-v6.7.46' `
  --output reverse_engineering/reports/f530404b0f3f_807de4a83df4_ascension_assets_audit.json
```
