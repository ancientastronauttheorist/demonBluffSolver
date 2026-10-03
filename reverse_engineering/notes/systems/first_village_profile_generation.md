# First-village profile and roster generation: bounded native join

Build `f530404b0f3f_807de4a83df4`. The
[producer](../../scripts/audit_first_village_profile_generation.py) and
[report](../../reports/f530404b0f3f_807de4a83df4_first_village_profile_generation.json)
execute original native profile accumulation, selected-mode profile copying,
script reset/materialization, saved-roster replacement and Standard roster
selection in one retained graph. This establishes conditional generation
semantics for original configured assets. It does not certify complete startup,
physical board construction, original RNG probabilities or public visibility.

## Original first-village configuration

The existing complete-object [asset graph](ascension_asset_graph.md) is reparsed
from the pinned original assets on every run and compared exactly, after JSON
normalization, with its tracked report. Character assets are independently
reparsed and compared too. GameAssembly and Dumper inputs are fingerprinted.

RoguelikeStandard's supplied instance has `currentAscension=0` and
`currentVillage=0`. Its actual native selector reaches group zero, village zero,
profile `21674` (`Ascension_1 1`). This is a RoguelikeStandard profile;
its actual `GetGameMode` returns zero and therefore admits the Standard roster
setup branch. Mode publication, saved-mode loading and zero-field allocation
are separate prerequisites, not outputs of this replay.

Profile `21674` has one inline script and no custom override. Its original
Villager list, in order, is:

| Asset ID | Public role |
| --- | --- |
| 21614 | Confessor |
| 21626 | Lover |
| 21621 | Hunter |
| 21618 | Enlightened |
| 21620 | Gemcrafter |

The Minion list contains only `21596`, public and managed **Minion**, not Spy.
There are no Outcasts or Demons. Both must-include and always-in-deck lists are
empty. The selected count is exactly `N=5`, `town=4`, `outs=0`, `minion=1`,
`demon=0`; the four disguised counts equal their corresponding base counts.
The report retains the profile's original candidate arrays, stored cache,
unlocked IDs, count lists and day-addition records separately.

Its unlocked list is `[21627,21623]`: **Medium and Judge**, not Scout. Those
IDs do not automatically describe selected actors or the unmodified starting
pool. The producer explicitly asserts the current asset-name bindings.

## Accumulation is a partial failing invocation

The pinned executable scan finds zero direct relative calls or jumps to
`GameData.UpdateScriptCharactersFromPreviousLevels` (`3DCE90`) and one
file-backed function-pointer reference at `26A5380`. That registration does
not establish invocation. Indirect, engine or editor callers are not excluded.
No actual accumulation caller is certified by this tranche.

Under an explicitly supplied invocation, the original body visits the actual
four groups in order, appends each profile's unlocked occurrences to one
retained cumulative list, and calls actual
`AscensionsData.AddCharactersToScript` (`3B11C0`). It modifies the first
profile's inline Villager list to
`[21614,21626,21621,21618,21620,21627,21623]`. Its stored starting array remains
the original five entries.

The invocation reaches **22 profile calls**, then takes the native index
failure at profile `21695` (`Ascension_19`), the first profile in group four.
That profile has an empty inline-script array and only custom-script sources.
Earlier profile mutations remain. The first invocation has 972 supplied-service
entries; repeating it has 931 and fails at the same profile without duplicating
Medium or Judge. Managed exception construction/unwinding is not executed.

Continuing from that partial graph requires an explicitly supplied recovery
and later setup invocation. The report marks this condition. Seven entries are
therefore **failed-accumulation state**, not a demonstrated successful normal
startup. Judge also introduces an active-role observation/action boundary.

## Retained native selection and handoff

Actual `GameData.SetupCurrentAscension` calls the installed actual
RoguelikeStandard selector and actual `AscensionsData.CopyData`. The replay
then invokes actual cache clearing, one width-one inline selection and starting
materialization. This sequence is explicitly scheduled; it is not substituted
for the complete `Gameplay.SetupDelay` coroutine.

`CopyData` initially shares the source stored starting array. Materialization
publishes a new array containing the selected inline pool, leaving the source
stored array unchanged. The five-entry baseline and seven-entry partial-state
variant survive typed getter requests in `10,20,30,100` order without another
selection draw. Actual `Gameplay.GetCurrentScript` uses actual
RoguelikeStandard starting-level reads and returns the selected N5 record.

Actual `Gameplay.ResetSavedCharacters` calls the actual GameData typed wrapper
and AscensionsData getter. It replaces the four saved lists with copies of the
selected pools and leaves current rosters unchanged. The roster-selection
family starts with explicitly empty current rosters. Earlier `Gameplay.Init`
saved-to-current copies, `LoadCharacters` unlock appends and the subsequent
`SetupDelay` clears remain separate caller/lifetime conditions. See
[existing lifecycle evidence](gameplay_lifecycle.md) and
[saved-roster evidence](gameplay_roster_reset.md). This replay neither assumes
that loading deduplicates roles nor carries loaded duplicates across an
unexecuted reset.

Actual `SetupCurrentVillageForStandard` then supplies one singleton Minion draw
and four Villager draws without replacement. Exhaustive ordered support is:

| Input condition | Villager pool | Ordered selections | Distinct unordered selections |
| --- | ---: | ---: | ---: |
| Original profile, no accumulation invocation | 5 | 120 | 5 |
| One partial accumulation invocation plus supplied recovery | 7 | 840 | 35 |

All 960 outputs contain four distinct permitted Villagers, no Outcast/Demon,
and exactly Minion `21596`. Draw widths are respectively `[1,5,4,3,2]` and
`[1,7,6,5,4]`. These are native candidate-roster outputs, not a complete
`GetRandomCharacters` return, shuffled physical board or public deck certificate.
No uniform world prior is assigned.

## Runtime services and exit checks

The 16 original caller/getter bodies execute with exact metadata signatures
and per-body fingerprints; complete native bytes stay private. Their combined
executed instruction set contains **1,077 distinct instructions**, not a claim
of complete branch/instruction coverage. Normal returns check stack balance
and all eight Windows nonvolatile integer registers.

Collections, allocation, barriers, Unity reference equality and integer RNG
are explicit services. ClassConv helper outputs are also supplied: one
field-faithful JSON-result contract and one shared-element sensitivity contract.
Neither certifies UnityPlayer JSON serialization or reference reconstruction.
The [copy audit](ascension_helpers.md) establishes the actual JSON mechanism.
Two additional order-sensitivity cases copy first and mutate the source later:
field-faithful output retains five entries, while shared elements expose seven.
The intended pool cannot be established without the copy/lifetime contract.

Validation passes **six retained cases**, **two copy-order sensitivity cases**,
**960 ordered native roster cases**, and **61 stopped service prefixes**.
Every stopped replay compares its complete recorded service prefix and final
modeled profile/roster state against the corresponding service-entry state.
Three selected accumulation stops complement its real native index failure;
this is not exhaustive accumulation-service failure coverage. Other stops
cover all service entries of the unmodified profile setup/materialization,
count handoff and saved-reset calls. The report retains full accumulation
invocation order, cumulative unlocked IDs and final inline pools by profile.

Full authored snapshots are losslessly interned using
`audit_report_snapshots.pool_snapshots`; all **33 snapshot blobs** are verified
by `expand_snapshots`, which reconstructs independent full values. The audit
function returns the expanded report. The CLI writes the compact pooled form.
No physical machine-state or managed-unwinder equivalence is claimed.

## Bounded asset absence and next S1 dependency

The 192 checked count records comprise profile `characterCounts`, inline
`possibleScripts` counts and custom-script counts. Serialized cache copies
are not counted again; the modeled setup clears the cache. None admits N4/5
with `N-1` Villagers, no Outcasts/Minions and one Demon. This is an authored
asset finding, with arbitrary runtime profile mutations, callbacks and trailer
replacement excluded. It does not establish global unreachability.

The previous repeated-Hunter/Baa family remains
[conditional development evidence](conditional_hunter_baa_world_reference.md).
The next genuine first-village join is the unmodified five-role pool plus
ordinary Minion: actual random-roster return/shuffle, round-pool construction
including unique fallback, Minion acquisition, initialization and the complete
retained engine/action schedule. All five role clue domains must be retained.
Reviewed full-deck multiplicity, build binding and legal reveal chronology are
required before projecting public history. No capture, planner, posterior,
village outcome or solver certificate is produced here.

Reproduce after source freeze:

```powershell
$env:PYTHONPATH='B:\CodexTools\DemonBluffReverseEngineering\python-emulation'
python -m py_compile reverse_engineering/scripts/audit_first_village_profile_generation.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python reverse_engineering/scripts/audit_first_village_profile_generation.py 'B:\SteamLibrary\steamapps\common\Demon Bluff Playtest' --dumper-root 'B:\CodexTools\DemonBluffReverseEngineering\artifacts\f530404b0f3f_807de4a83df4\il2cppdumper-v6.7.46' --output reverse_engineering/reports/f530404b0f3f_807de4a83df4_first_village_profile_generation.json
```

Python compilation, the native report producer and compact-snapshot round-trip
verification pass. Existing reports are preserved. No Rust, Cargo, git, live
control or shared process-guide changes are included.
