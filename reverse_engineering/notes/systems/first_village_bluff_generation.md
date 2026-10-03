# Original first-village roster, pool and ordinary Minion acquisition

Build `f530404b0f3f_807de4a83df4`. The
[producer](../../scripts/audit_first_village_bluff_generation.py) and
[schema-v1 report](../../reports/f530404b0f3f_807de4a83df4_first_village_bluff_generation.json)
establish a **conditional native catalogue-support result**: original N5
profile `21674` permits ordinary Minion bluff selection from 24 distinct
Villager assets, including assets outside its five starting Villagers.
Restricting appearances to those five would omit witnessed native selector
outputs. This does not certify complete generation histories, native RNG
probabilities, physical cards, initial Day, public visibility or capture admission.

## Original graph and native boundary

The producer reparses the pinned ascension and character assets, compares both
complete authored reports after JSON normalization, and retains their object
identities. It reuses the
[first-village profile join](first_village_profile_generation.md), with no
accumulation invocation or recovery from a failed accumulation. Supplied
RoguelikeStandard state selects group zero, village zero, original profile
`21674` (`Ascension_1 1`); its actual mode query returns Standard zero.

Actual setup/copy, cache clearing, inline count selection, starting materialization
and `Gameplay.GetCurrentScript` execute before the new tranche. The returned
count object has `N=5`, four Villagers, no Outcasts, one Minion and no Demons.
Publishing that exact returned object into `Gameplay.CurrentScript` is an
explicit supplied write. The complete SetupDelay count-selection/clone/relic
transaction and mode/save publication are not executed here.

The original starting Villager order is Confessor `21614`, Lover `21626`,
Hunter `21621`, Enlightened `21618`, Gemcrafter `21620`. The sole Minion
`21596` is public and managed **Minion**, not Spy. Must-include and
always-in-deck arrays are empty. The original unlocked Medium/Judge records
are retained separately and do not expand this starting list through an
invented accumulation invocation.

Actual `Gameplay.GetRandomCharacters` (`37CE10`) calls actual
`SetupCurrentVillageForStandard` (`3802D0`) exactly once per generation case.
Its integer draws have observed widths `[1,5,4,3,2,1,4,3,2,1]`: Standard
selection retains an ordered four-Villager roster, then GetRandomCharacters
selects the singleton Minion and those four Villagers into its return source.
The native cached key delegate (`392D50`) executes once per source occurrence;
its folded tail jump enters the supplied Random.value service. Supplied deferred
OrderBy/ToList materialization sorts ascending float keys, stably by source
order on ties. Actual managed LINQ implementation and Unity PRNG are excluded.

That actual returned list feeds actual `Characters.ManageCharacters`
(`36CE30`), both actual pool builders (`36D3A0`, `36D720`), actual source/cache
getters and actual predicates/filters. The retained prefix stops immediately
before the first `Character.Init` (`365A20`), preserving the first actor,
returned data object and descending displayed ID. Positions/layout is supplied
inertly. No Init, Start, queue, Act, Reveal or UI publication executes.

The original fallback getter concatenates Townsfolk, Outcasts, Minions and
Townsfolk again. Its 63 source occurrences become 48 eligible occurrences:
24 distinct Good, bluffable, real-Villager assets, each appearing twice.
Rambler `21607` in the authored Townsfolk array is real Outcast type 20 and is
excluded by the exact real-Villager filter. The unique pool has two entries:
the one omitted starting Villager and one fallback draw. Those entries may
share an asset identity; occurrence multiplicity is retained. Duplicate
construction orders the selected four starting Villagers. Observed pool draw
widths are `[1,48,4,3,2,1]`.

Actual ordinary `Minion.GetBluffIfAble` (`3E49F0`) is invoked separately against
each retained pool state, with no intervening writers. Its actual RollDice
body (`396840`) requests integer range `[1,11)`. Values 1 through 4 draw from
the four-entry duplicate pool and do not register a script addition; values 5
through 10 draw from the two-entry unique pool and execute actual
`Gameplay.AddScriptCharacterIfAble` (`37B370`). Registration suppresses an
already-contained exact object and otherwise appends it to current Villagers.
The original four selected Villagers remain the base roster. No probability
law is assigned to these branch/index values.

The actor for these selector calls is a supplied sparse actor at the ordinary
Minion occurrence's position in the native sorted return list. This establishes
selector/pool support under an explicitly supplied invocation. It does not
establish that Character.Init or a concrete later Reveal reaches the same
state. The existing [bluff acquisition audit](gameplay_bluff_acquisition.md)
documents Reveal's first-acquisition caller, and the
[pool composition audit](manage_pool_composition.md) documents the retained
pool prefix and its independent service contract.

## Finite factors and verified results

The successful producer exited 0 after assertions and lossless snapshot-codec
roundtrip. The report has 36 method-body evidence records, 12 selected operand
assertions and 1,921 distinct executed instruction addresses. These are distinct
executed addresses, not complete branch coverage of all recorded bodies.

| Recorded factor | Actual completed cases |
| --- | ---: |
| All observed generation integer-index histories | 2,880 |
| Distinct ordered Standard four-Villager rosters | 120 |
| Distinct four-Villager subsets | 5 |
| Five subset representatives, every fallback occurrence index | 240 |
| Remaining duplicate-draw orders for those representatives | 115 |
| Other ordered Standard rosters, default pool prefix | 115 |
| Sorted placements, default pool prefix | 655 |
| Total pool factor rows | 1,125 |
| Recorded separately supplied Minion invocations | 8,960 |
| Distinct acquired candidate assets, each with full selector witness | 24 |
| Stopped supplied-service prefixes | 573 |

The generation factor checks every Cartesian combination of the ten observed
integer draw domains, with one ascending float-key plan. Each case checks one
actual Standard call, source membership, ordered current roster, five keys and
the complete shared random/sort service ordering. It saves actual native
generation states for all 120 ordered Standard rosters, without carrying a
baseline pool cache into other rosters.

For one actual generated representative of each of the five subsets, all 48
fallback indices run actual pool construction. Each of these 240 retained states
runs all ten die values and every relevant selector index: four duplicate or
two unique indices. These produce 6,720 selector rows. A separate fallback
catalogue probe adds 240 rows. The 23 nondefault duplicate orders per subset add
115 pool rows and 460 selector rows. The other 115 ordered Standard rosters each
run a native default pool prefix and duplicate/unique selector, adding 230 rows.

For each subset representative, all 120 distinct sort-key rank permutations,
one all-equal plan and each of ten single-pair ties execute actual generation
and the supplied stable sort service: 600 distinct plus 55 tie cases. Each feeds
the actual default pool prefix and two Minion branches, adding 1,310 selector
rows. Source order, float bits and native sorted output are recorded; ties
do not establish Unity's random-key collision distribution.

These are exhaustive **separate factors**. Noncanonical final-selection order
times sort plan times nondefault fallback/duplicate indices times selector
histories are uncombined and unverified. Counts are support rows, not weighted
worlds or a Cartesian full-history claim. Baseline runs,24 full catalogue
selector witnesses and interrupted attempts are additional execution records;
they are not included in the 8,960 support-row counter.

Every reached supplied service in the full baseline generation (94), pool
prefix (470), duplicate acquisition (3) and unique acquisition (6) is stopped
separately. The 573 cases compare the exact service prefix and final modeled
snapshot against the successful service-entry snapshot. This establishes the
authored stop boundary, without exception construction, unwinding or rollback.

Snapshots preserve source/profile/cache/current-script identities, roster and
saved-roster contents, pool identities/versions, return order, intermediate
allocated collection views, pending predicate/sort progress and native
occurrence multiplicity. Factor rows retain choices, return order and call
trace hashes. Each catalogue witness retains complete selector services and
links to the actual generated/pool rows that supplied it. Synthetic callback
return site `0x380000200` denotes the harness frame, not an original native
caller address.

## Acquired candidate catalogue

Every row has two eligible fallback occurrences. Public serialized names and
managed identities are separate fields; no internal class is silently treated
as its public alias.

| Asset ID | Authored object / public name | Managed role |
| --- | --- | --- |
| 21609 | Alchemist | Alchemist |
| 21611 | Baker | Baker |
| 21612 | Bard | Acrobat2 |
| 21613 | Bishop | Bishop |
| 21614 | Confessor | Confessor |
| 21615 | Dreamer | Dreamer |
| 21616 | Druid | Librarian |
| 21617 | Empress | Noble |
| 21618 | Enlightened | Shugenja |
| 21619 | Fortune Teller | FortuneTeller |
| 21620 | Gemcrafter | Archivist |
| 21621 | Hunter | Tracker |
| 21622 | Jester | Juggler |
| 21623 | Judge | Judge2 |
| 21624 | Knight | Immortal |
| 21625 | Knitter | Knitter |
| 21626 | Lover | Empath |
| 21627 | Medium | Lookout |
| 21628 | Oracle | Investigator |
| 21629 | Poet | Gossip |
| 21631 | Scout | Scout |
| 21632 | Slayer | Slayer |
| 21634 | Bounty Hunter object; serialized public name empty | BountyHunter |
| 21635 | Witness | Witness |

## Supplied services and remaining gate

Collection allocation/construction, membership, append/copy/remove, indexed
access, snapshots, enumeration/disposal, array clearing, class/metadata
initialization, Unity liveness, GC gateway, mode/copy publication and valid
RNG outputs are supplied services inherited from the preceding join or named
explicitly here. RemoveAll executes the actual native predicate but commits
the supplied collection mutation only after all callbacks. It is not an audit
of managed in-place compaction. Native integer return ABI, stack and eight
nonvolatile integer registers are checked on normal return; general XMM
preservation and whole-runtime heap equivalence are not claimed.

The next full-history gate is the actual retained Init/Start/Reveal acquisition
caller and intervening pool/script writers, with the actual scheduling and
physical board/deck publication contracts. The finite factor witnesses do not
recover Unity seeds, float distributions, joint weights or generation priors.
They do not certify displayed deck semantics, build-matched captures or legal
player-observation chronology. Passive and active behavior for any of these
24 possible copied roles remains part of the eventual public-history domain;
this result must not narrow that obligation back to Hunter-only clues.

No live process/save access, Rust integration, Cargo build, policy objective,
world prior or capture admission is introduced by these artifacts. Complete
proprietary bytes and disassembly remain in the private workspace; tracked
evidence contains selected operand assertions and body/source fingerprints.

## Reproduction and fingerprints

Run from the repository root with the existing private emulator dependencies:

```powershell
$env:PYTHONPATH='B:\CodexTools\DemonBluffReverseEngineering\python-emulation'
python -m py_compile reverse_engineering/scripts/audit_first_village_bluff_generation.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python reverse_engineering/scripts/audit_first_village_bluff_generation.py 'B:\SteamLibrary\steamapps\common\Demon Bluff Playtest' --dumper-root 'B:\CodexTools\DemonBluffReverseEngineering\artifacts\f530404b0f3f_807de4a83df4\il2cppdumper-v6.7.46' --output reverse_engineering/reports/f530404b0f3f_807de4a83df4_first_village_bluff_generation.json
```

`--inspect` lists selected resolved native targets; `--probe` runs only the
baseline and does not perform the normal full asset-report comparisons.
Only the normal invocation establishes the finite results above. Snapshot
pooling happens after expanded assertions and must expand exactly to the
original report. Windows UTF-8 text output preserves the producer's physical
CRLF final newline for byte reproduction.

The final schema-v1 report is 7,916,296 bytes, SHA256
`d4aa00d6b99df42b8140beb7b725305a5520dd32dddc639754315cc4ca6c0c47`.
Producer source SHA256 is
`54214b47ff16336a6a564fbf62a4b0a9697b6cd24fd188ca1b19f7d48cc0dff7`.
GameAssembly SHA256 is
`F530404B0F3F28479A7CE21D5738C4E36C2A0A03E1B5520092975B4150D819EC`,
script.json is
`6c4e6e01a7672b69096e0d52491f3b848237ac40ef3b3817beb8b442ab18f769`,
and dump.cs is
`53f5da90d056ae96970eebc3293fd48190f2c7f70605348fcd38fa3302c438b7`.
Exact original asset and five helper-source hashes are retained under
`source_hashes` in the report. Earlier reports and private source bytes are
preserved unchanged.
