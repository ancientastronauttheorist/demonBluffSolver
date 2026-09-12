# RoguelikeStandard level and presentation helpers

Pinned build `f530404b0f3f_807de4a83df4`. These six supplemental declarations
complete the remaining level/presentation boundary alongside
[ascension selection](ascension_setup.md),
[lifecycle](roguelike_standard_lifecycle.md), and
[native kill scoring](roguelike_delegate_score.md). No typed targets are added.

## Level bounds

MaxLevel reads ProjectContext's instance, its GameData reference `+0x20`, then
GameData's RoguelikeStandard ascension groups `+0x50`. It selects the group
using currentAscension `+0x14`, upper-clamping to the final group. Negative
indices and an empty outer array fail the native bounds check. A null project,
GameData, outer array, selected group, or selected group's inner array fails.

The result is the selected inner array's length minus one. An empty inner array
therefore returns `-1`; it does not itself fail. This differs from a selector
that must read an actual element from that array.

CheckIfLastLevel captures currentVillage `+0x20`, calls this MaxLevel directly,
and evaluates signed `wrap32(currentVillage - 1) >= wrap32(MaxLevel + mod)`.
Both arithmetic sides can wrap. It does not dispatch the virtual GetMaxLevel
slot, which is a separate helper. Native fixtures exercise these two bodies
together for empty, one-entry and seven-entry inner groups, signed extremes,
and positive/negative modifiers.

## Showcase decision

ShouldShowcaseNewCharacters always boxes and formats four values, in order:
currentAscension, bestAscension, currentVillage, bestVillage. It logs the
formatted message before computing the result. The message labels use “best”
even though the current values are passed first in each pair.

Its result is the conjunction `bestAscension <= currentAscension` and
`bestVillage <= currentVillage`, using signed comparisons. The village
comparison applies even when currentAscension exceeds bestAscension; this is
not a lexicographic comparison of progression. The method neither reads nor
updates showedNewCharacters. The fixture toggles that field while holding the
four inputs constant and checks all instance fields remain unchanged.

Formatting or logging failure stops before a successful result. Logging,
formatting and boxing remain explicit service gateways; no actual Unity log
or UI is changed.

## Score formatting and value projection

| Method | Native inputs to service boundary |
| --- | --- |
| GetKillScore | Returns `wrap32(10 * (count + 5))`, where count comes from CharactersHelper.GetUnrevealedCharactersCount. |
| GetSummaryScores | Boxes `wrap32(ascensionScore + roundScore)` and supplies it to the run-score format. |
| GetScores | First formats `wrap32(bestAscension + 1)`, then bestScore; concatenates the two formatted strings in that order. |

GetSummaryScores' explanatory text about losing a heart is a presentation
literal, not evidence of the actual life/score transition policy. The
[report](../../reports/f530404b0f3f_807de4a83df4_roguelike_presentation.json)
preserves the exact four format templates and all asserted argument values.
The string implementations themselves use gateways, so the audit establishes
which values and templates are passed, not every culture/localization behavior
of String.Format.

## Validation

[`audit_roguelike_presentation.py`](../../scripts/audit_roguelike_presentation.py)
passes 380 native cases and twelve instruction relationships against the
pinned DLL and Dumper script. Cases cover outer/inner array bounds, each nullable
global/reference boundary, signed wrap behavior, showcase comparison pairs,
flag independence, score arguments and formatting/logging failures. All
methods preserve the complete supplied instance field block. Normal returns
also verify preserved registers and stack.

All six private baseline exports completed. Full native method ranges include
the failure paths and exclude trailing alignment padding. There are no virtual
slot calls inside these six methods. Array allocation, boxing, type checks,
format/concat/log services and the unrevealed count are explicit gateways;
metadata is warmed. No solver, live game/save state or shared coverage inventory
is changed.

```powershell
python reverse_engineering/scripts/audit_roguelike_presentation.py `
  'B:\SteamLibrary\steamapps\common\Demon Bluff Playtest' `
  'B:\CodexTools\DemonBluffReverseEngineering\artifacts\f530404b0f3f_807de4a83df4\il2cppdumper-v6.7.46' `
  --output reverse_engineering/reports/f530404b0f3f_807de4a83df4_roguelike_presentation.json
```
