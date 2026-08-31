# Druid current-build native contract

This note records the clean-room native checkpoint for the shipped public
**Druid**, managed as `Librarian` in build
`f530404b0f3f_807de4a83df4`. It closes all 16 declarations owned by
`Librarian` and `Librarian.<>c`, plus the 20 shared functions needed to explain
the picker, registered-role projection, result history, lifecycle, and random
selection. It is not a claim that the whole game is decompiled.

Evidence status is **native-static** for the executable behavior,
**metadata** for the serialized asset binding, and **behavioral** for the
checked-in compatibility corpus. Baseline and typed Ghidra exports remain
private. This note, the target manifest, aggregate typed-quality report, and
coverage records contain no decompiled bodies or private filesystem paths.

## Public asset binding and managed identity

`Demon Bluff_Data/sharedassets0.assets` contains the shipped public Druid
`CharacterData` at path ID `21616`, absolute file offset `23646696`, size
`10264`, and object SHA-256
`46F49127A12A7A5D1158F675E00EC19BD8BCC12C926BAE2F9B0D435C2B5D4CBE`.
The containing asset has SHA-256
`E0D239C03FA47EC138F9C2797779E83B65AEF41F30C5AD84D4CA76767A93E967`.
Its serialized identity is:

- public name `Druid`;
- character ID `Druid_89845092`;
- managed SerializeReference type `Librarian` (`TypeDefIndex 5900`), with
  compiler-generated helper `Librarian.<>c` (`TypeDefIndex 5899`);
- Good Villager (`type == 10`, `startingAlignment == 10`);
- `abilityUsage == 10`, the `ResetAfterNight` category;
- bluffable, not usually disguised, and picking;
- an empty bundled-character and skin collection;
- no role achievements, additional statuses, tags, or `canAppearIf` records;
- exact hint `Wretch does not appear as an Outcast for her.`; and
- an empty `ifLies` field.

The exact authored public description is:

```text
<b>Pick 3 characters:</b>
Learn 1 random Outcast among them (if any)
```

The managed `Librarian.get_Description` declaration independently retains an
older two-target/old-terminology sentence at string-literal RVA `0x2700ED0`:

```text
Pick 2 players. Learn which Outsider is among them (if any)
```

The executable picker and clue bodies agree with the current asset, not that
stale description: they require exactly three distinct Character objects and
use the current `Outcast` enum category.

The normal serialized candidate pool in `Demon Bluff_Data/level0`, path ID
`139347`, contains exactly one file-ID-`2` reference to Druid path ID `21616`,
at pool-object-local offset `488`. The pool object is at file offset
`17578592`, size `1080`, and has SHA-256
`FB9D821AE0A7E3655BEF4A3DD3E544E85B3109258A48DCF68FF0969ACED8D948`.
The containing `level0` file has SHA-256
`B509AC15904F0926419CCCC1D2E86508C69B0E3AF2235A08FF8E9FC4D170C7F1`.
Druid is absent from the 15-entry ordered-Start object at path ID `137026`,
file offset `17310672`, size `332`, and object SHA-256
`544328634CD77D551B5864CDC1B643029F3B30BFFC5BB4350DFCF83C66226BB0`.
It is a direct normal-pool reveal/day role, not an ordered-Start participant.

## Alias implications and obsolete predecessor

An exact serialized type-name census of the current public assets finds one
`Librarian` occurrence and no `RangedEmpath` occurrence. `RangedEmpath`
(`TypeDefIndex 5859`) remains executable but is unbound in the current assets.
Its old managed description and clue are respectively:

```text
Learn character that is adjacent to an Evil
#{0} is adjacent to an Evil
```

Those literals are at RVAs `0x26E0B10` and `0x26F0090`. The code and wording
make `RangedEmpath` a semantic predecessor of the public Druid concept; calling
it a historical public Druid identity is an inference from behavior, not a
current serialized binding. It is not evidence for mapping Druid to Bard.
Current Bard binds `Acrobat2`; the old exact `Acrobat` type is its own Bard
predecessor. For the shipped build the binding to use is unambiguously:

```text
public Druid -> managed Librarian
```

## Exact callable boundary and shared bodies

`Librarian.<>c` declares six methods and `Librarian` declares one field plus
ten methods:

| Managed identity | RVA | Role-local purpose |
| --- | ---: | --- |
| `Librarian.<>c..cctor` | `0x3F0860` | Compiler-helper singleton initialization |
| `Librarian.<>c..ctor` | `0x357920` | Fieldless compiler-helper construction |
| `Librarian.<>c.<CharacterPicked>b__7_0` | `0x3F07B0` | Truth sort key: display ID |
| `Librarian.<>c.<CharacterPicked>b__7_1` | `0x392D50` | Truth secondary key: `Random.value` |
| `Librarian.<>c.<CharacterPickedDrunk>b__9_0` | `0x3F07B0` | Bluff sort key: display ID |
| `Librarian.<>c.<CharacterPickedDrunk>b__9_1` | `0x392D50` | Bluff secondary key: `Random.value` |
| `Librarian.get_Description` | `0x3E36E0` | Stale managed description |
| `Librarian.GetInfo` | `0x3E3320` | Passive empty/null result |
| `Librarian.GetBluffInfo` | `0x3E32C0` | Passive empty/null result |
| `Librarian.Act` | `0x3E1520` | Truth Day picker registration |
| `Librarian.StopPick` | `0x3E3380` | Completion/cancel handler cleanup |
| `Librarian.CharacterPicked` | `0x3E2770` | Truth selection and result delivery |
| `Librarian.BluffAct` | `0x3E1790` | Bluff Day picker registration |
| `Librarian.CharacterPickedDrunk` | `0x3E1A00` | False selection and result delivery |
| `Librarian.ConjourInfo` | `0x3E2FE0` | Exact positive/none formatter |
| `Librarian..ctor` | `0x3E3690` | `drunkId` initialization |

The sole owned instance field is `drunkId` at `Librarian + 0x48`.
Construction initializes it to exact ID `Drunk_15369527`, whose string literal
is at RVA `0x26FE920`.

The 36-function target adds these 20 semantic functions:

- `Character.Act`, `OnClick`, `RoleAct`, `RefreshCharacter`,
  `GetRegisterAs`, and `GetCharacterData`;
- the `Character.RoleAct` callback closure constructor and invoke body;
- `Character.ShowActedDelayed` and its coroutine `MoveNext`;
- `CharacterPicker.CancelPick`, `StartPickCharacters`, and
  `ClickedCharacter`;
- `ActedInfo..ctor`;
- `Characters.FilterCharacterType<CharacterData>`;
- `Gameplay.GetScriptCharacters` and `GetAllAscensionCharacters`;
- `GameData.GetCharacterDataOfId`;
- integer `UnityEngine.Random.Range`; and
- `UnityEngine.Random.value`.

Six Druid-selected FunctionDefinitions use already selected shared native
bodies: the helper constructor and Character callback constructor share the
broad constructor body at `0x357920`, each pair of selector definitions shares
one native body, and the two selector RVAs were already present elsewhere in
the target union. The metadata identities and ABI-compatible prototypes remain
distinct. The target reports seven shared-body flags in total, including the
already-selected integer `Random.Range` body. Unity event internals, generic
list mechanics, allocator/runtime helpers, and actual global PRNG internals
stop outside this role-specific boundary.

## Day picker and target eligibility

`Librarian.Act` and `Librarian.BluffAct` recognize only
`ETriggerPhase.Day == 30`. Every other trigger is a role-local no-op. On Day,
each first calls:

```text
CharacterPicker.StartPickCharacters(howMany = 3, picker = actor)
```

It then registers its completion delegate and registers `StopPick` as the
cancel delegate. Truth uses `CharacterPicked`; bluff uses
`CharacterPickedDrunk`. The role has no separate target predicate. While a
picker is active, `Character.OnClick` routes the clicked Character into
`CharacterPicker.ClickedCharacter` before its ordinary killed, hidden,
ability-use, or local-activation gates. Native-legal targets therefore include:

- the Druid actor itself;
- dead or killed cards;
- hidden or unrevealed cards whose Character object can be clicked; and
- cards that themselves still have an unused active ability.

Selection uniqueness is by Character object reference, not display ID. A
second click on the exact same object toggles it out. Different Character
objects with the same display ID are distinct selections. The picker completes
only when its selected list contains exactly three distinct objects; it does
not auto-complete at one or two and cannot retain more than three through the
normal click path.

A normal board with fewer than three Character objects therefore cannot
complete the picker. A three-object board can complete only by selecting all
three, including the actor when the actor is one of those objects. There is no
padding, repeated-target slot, board-size clamp, or shortened clue.

At completion, the picker iterates selected targets in click order and sends
`OnPicked == 70` to each target other than the physical picker actor. Those
target acts happen before Druid's completion callback. The completion callback
then removes its own completion delegate plus `StopPick`, constructs, and
delivers Druid's result. The cancellation-time `StopPick` helper removes both
the truth and bluff completion handlers plus itself from the stop event.
Canceling emits no Druid result and does not execute the completed-path sorting
or selection RNG.

## Text-ID ordering versus acted-reference ordering

Both completion callbacks enumerate the three selected display IDs through:

```text
OrderBy(display ID).ThenBy(_ => Random.value)
```

The three IDs printed in the clue are therefore nondecreasing. The secondary
selector consumes one `Random.value` float for each of the three selected
elements, even when all IDs differ. It affects only equal-ID tie ordering and
cannot change a clue containing three identical integer values.

The `ActedInfo.characters` list is independent: it is a new list populated by
copying `CharacterPicker.PickedCharacters` in original click order. It is not
sorted to match the text. This distinction is observable and must be retained:

```text
clue IDs       = display-ID order, with random tie keys
acted refs     = click chronology
```

No actor reference is implicitly inserted. A self-target appears in the
references exactly where it was clicked. Equal display IDs do not collapse
references.

## Truthful Outcast selection

For each selected Character, truth calls `Character.GetRegisterAs()` and
tests the returned `CharacterData.type` for exact `Outcast == 20` (`0x14`).
`GetRegisterAs` returns the Unity-live `registerAs` record when present;
otherwise it returns the Character's current `dataRef`. The truth candidate
list preserves one entry per selected physical occurrence and preserves picker
order during collection.

This scan occurs after all non-self targets have received their `OnPicked`
acts. Any synchronous identity or registration mutation caused by those acts
is therefore visible to the Druid result; the role does not snapshot target
data when the picker opens or when each card is clicked.

When at least one selected occurrence registers as Outcast, truth draws:

```text
i = Random.Range(0, candidate_count)
```

It then calls `GetCharacterData()` on that selected occurrence and prints that
record's current `characterName`. `GetCharacterData` uses the same live
`registerAs`-else-current-`dataRef` precedence. Thus both qualification and
named role are projected through current registered data. Druid does not read
an original deck record, a prior identity, a bluff asset directly, or target
runtime data. Duplicate candidate occurrences remain weighted independently.

When no selected occurrence registers as Outcast, truth emits the none result
without an integer draw. Consequences include:

- a normal current Outcast with no override qualifies;
- Wretch registers as Minion and is excluded despite its underlying Outcast
  asset category;
- Spy registers as Villager and is excluded;
- ordinary Doppelganger has no register-as override, falls back to its real
  Outcast `dataRef`, and qualifies; and
- ordinary Drunk has no such forced non-Outcast registration and therefore
  qualifies when its current projection is Outcast.

Multiple qualifying Outcasts are sampled uniformly by selected occurrence,
not by distinct role name. A selected result can be dead, hidden, or the actor
because picker legality and truth categorization impose no later exclusion.

## False selection pools, fallback, and exhaustion

Bluff first scans the selected targets through the same registered-Outcast
predicate. If **any** selected occurrence registers as Outcast, bluff
deterministically emits `there are NO Outcasts`; it does not select a role and
does not consume an integer Range draw.

If no selected target registers as Outcast, it builds the false-positive pool
in the following exact stages:

1. copy the concatenated current script CharacterData lists returned by
   `Gameplay.GetScriptCharacters`, preserving list and entry order;
2. keep exact Outcasts with `bluffable == false`;
3. if empty, repeat that same filter over
   `Gameplay.GetAllAscensionCharacters`;
4. if still empty, keep **all** Outcasts in the ascension-wide list, including
   bluffable records; and
5. if still empty, resolve exact ID `Drunk_15369527` through
   `GameData.GetCharacterDataOfId`.

At each nonempty pool stage it selects one physical list occurrence uniformly
with `Random.Range(0, count)`. It does not remove the actor's current role,
selected roles, in-play roles, duplicate references, duplicate role names, or
the truthful support. There is no collision retry. Repeated CharacterData
occurrences therefore receive repeated probability mass. The exact Drunk
fallback uses no Range draw because its support is a singleton.

In the current authored public role set the six Outcast assets are Bombardier,
Doppelganger, Drunk, Plague Doctor, Rambler, and Wretch. Exactly three have
`bluffable == false`:

```text
Doppelganger
Drunk
Wretch
```

All three occur in the current Standard script pool and in authored ascension
pools, so they are the concrete first-stage false-positive role support for
normal current Standard/Ascension games. The widened all-Outcast fallback adds
Bombardier, Plague Doctor, and Rambler when the nonbluffable pool is empty.
The exact-ID Drunk fallback is reachable only when both live-script and
ascension sources fail to produce any Outcast under those stages, including
malformed/empty configurations.

The native `GetAllAscensionCharacters` helper appends four global list fields
in the observed order `+0x68`, `+0x70`, `+0x78`, `+0x68`; the first list is
therefore repeated. The role performs no deduplication after that helper. This
is relevant to probability mass in custom/malformed configurations, although
the authored current pools already populate the earlier nonbluffable stage.

## Exact clue text and `ActedInfo` shape

The positive format literal is at RVA `0x271F118` and is exactly:

```text
Among #{0}, #{1}, #{2}
there is: {3}
```

The none literal is at RVA `0x271F090` and is exactly:

```text
Among #{0}, #{1}, #{2}
there are NO Outcasts
```

There is one newline and no terminal punctuation. `{0}`, `{1}`, and `{2}` are
the sorted display IDs. `{3}` is the current projected CharacterData
`characterName`, not an enum spelling or raw character ID.

`ConjourInfo(id1, id2, id3, cd)` selects the none literal when `cd` is null;
otherwise it reads `cd.characterName` and selects the positive literal. The
two reachable callback bodies call this formatter. Both then allocate one
fresh `ActedInfo` whose reference list is the click-order copy described
above. The passive declarations `GetInfo` and `GetBluffInfo` are not the active
clue route: each returns an `ActedInfo` with the exact empty-string literal and
a final null Character-list pointer.

## Complete RNG chronology

For every normally completed three-target clue, both truth and bluff first
consume exactly three `Random.value` draws for the secondary ID-sort keys.
The later integer chronology is:

| Path | Integer draw after three floats |
| --- | --- |
| truth, one or more registered Outcasts | one `Range(0, qualifying occurrence count)` |
| truth, no registered Outcast | none |
| bluff, any selected registered Outcast | none |
| bluff, no selected Outcast and nonempty candidate stage | one `Range(0, candidate occurrence count)` |
| bluff, exact hard Drunk fallback | none |

Pool construction, filtering, projected-data lookup, formatting, reference
copying, cancellation, and Night reset consume no role-local RNG. This records
the requested calls and support, not Unity's global PRNG state or statistical
independence across calls.

## Result history, dual-role ordering, and runtime data

`Character.RoleAct` installs a callback closure and invokes the selected role
path. Druid's completion callback passes one result into that closure.
`Character.ShowActedDelayed` then owns framework history and use consumption.
After its delay, the coroutine requires the Character's act flag and a
nonempty description. On success it:

1. invokes `onAboutToAct`;
2. appends the exact `ActedInfo` to chronological `actedInfos`;
3. decrements `pickableUses` for Day trigger `30`;
4. invokes the acted event;
5. displays/saves the result text; and
6. disables the action marker when uses reach zero.

Druid declares no role runtime-data key and neither completion callback reads
or writes `Character.runtimeData`. Targets, the named Outcast, sort keys, and
truth/bluff origin are not serialized into role runtime data; exact references
exist in the appended `ActedInfo` event.

At the outer framework level `Character.Act` can dispatch the current real
role and then a raw/apparent bluff role. If both produce a nonempty clue, the
real result appends first; the later bluff result appends second and overwrites
the currently visible `savedAct`. Druid does not suppress or merge that
framework behavior. Consumers must preserve full `actedInfos` chronology
rather than assuming the one visible speech string is the only result.

## Night reset and other lifecycle surfaces

The Druid asset uses `ResetAfterNight`. `Character.RefreshCharacter` restores
`pickableUses = 1` while the game state is Night (`20`) when its state/reveal
selection chooses a CharacterData with ability usage `10`: it uses `dataRef`
when `state` is Dead (`20`) or Revealed (`30`), or when the separate
`revealed` flag is set; otherwise it uses the raw `bluff` record. This is not a
`registerAs` lookup. Its action marker is enabled only when the chosen record
is picking and the card is not Hidden (`5`) or Dead (`20`). The refresh does
not clear `actedInfos`, `savedAct`, runtime data, statuses, or prior references.

The role itself has no `Init`, `Start`, `AfterRoundStart`, `Night`,
`OnReveal`, `OnExecuted`, `OnDied`, `OnProtected`, or achievement branch. It
has no death callback, cached target list, or local reset field. Picker handler
cleanup occurs at completion or cancellation; use reset remains framework
owned.

## Poet absence and cross-role state

Druid/Librarian is absent from the exact twelve-entry managed `Gossip` /
public Poet provider list. That list is, in order: Lover, Scout, Oracle,
Bounty Hunter, Medium, Knitter, Hunter, Enlightened, Empress, Bishop,
Gemcrafter, and Bard. There is no Druid Poet-provider slot, delegated actor
path, provider RNG draw, or Poet/Druid reference parity in the shipped build.

Cross-role state still matters through shared inputs:

- Wretch's Minion and Spy's Villager `registerAs` override them away from
  Outcast truth support, while ordinary Doppelganger has null `registerAs` and
  falls back to its real Outcast `dataRef`;
- Shaman/other identity replacement can change current `dataRef`,
  `registerAs`, and ability-usage projection;
- corruption and Evil alignment select whether framework dispatch reaches
  truth or bluff, but Druid itself never tests either;
- dead/hidden state does not disqualify an already clickable picker target;
  and
- a second real/raw action can overwrite visible speech while leaving both
  history events.

Those inputs are consumed at result time. Druid does not snapshot them when
the picker opens.

## Corpus and compatibility implications

The checked-in archive contains 160 apparent-Druid records. All have blank
`info_text`; 109 have the complete two-key payload (`targets` plus
`found_outcast`) and 51 have an empty payload.

The 426 active-v2 fixtures contain Druid in 115 deck pools and 118 apparent
records across 100 fixtures: 79 complete and 39 empty. The 137 legacy fixtures
add 42 Druid deck pools and 42 apparent records across 38 fixtures: 30 complete
and 12 empty. Complete results span board sizes six through ten. Every complete
target list has exactly three unique, ascending, in-board IDs; one active-v2
record self-targets (`asc32_v7`, Druid `#6`, targets `#1,#2,#6`).

Across all 109 complete results, `found_outcast` is null 62 times and is
Doppelganger 14, Drunk 15, Plague Doctor six, Wretch five, Bombardier six, and
Rambler once. The active-v2 corpus also has three Baker-original-Druid records
and seven Medium clues naming Druid; neither corpus has a Poet/Druid record.

This corpus protects the historical three-target schema, ascending text-ID
convention, self-target legality, no-Outcast support, and broad role-name
compatibility. It does **not** independently prove current exact text,
click-order references, registered-role projection, truth/bluff route,
candidate probabilities, RNG chronology, reset behavior, or real/raw history
ordering. Native evidence closes those current-build points.

## Reproduction and quality gates

The clean-room checkpoint ran these serialized stages against the saved build
projects:

```powershell
python reverse_engineering/scripts/validate_ghidra_targets.py `
  --targets reverse_engineering/targets/gameplay_role_druid.json `
  --script-json <private-current-build-script.json>

powershell -ExecutionPolicy Bypass -File `
  reverse_engineering/scripts/invoke_ghidra.ps1 `
  -GameRoot <private-game-root> -Stage export-target `
  -TargetSet gameplay_role_druid

powershell -ExecutionPolicy Bypass -File `
  reverse_engineering/scripts/invoke_ghidra.ps1 `
  -GameRoot <private-game-root> -Stage build-types

powershell -ExecutionPolicy Bypass -File `
  reverse_engineering/scripts/invoke_ghidra.ps1 `
  -GameRoot <private-game-root> -Stage typed-refresh

powershell -ExecutionPolicy Bypass -File `
  reverse_engineering/scripts/invoke_ghidra.ps1 `
  -GameRoot <private-game-root> -Stage typed-export `
  -TargetSet gameplay_role_druid

python reverse_engineering/scripts/audit_ghidra_type_quality.py `
  --baseline <private-baseline-export> `
  --typed <private-typed-export> `
  --output reverse_engineering/reports/f530404b0f3f_807de4a83df4_typed_quality_gameplay_role_druid.json `
  --check

python reverse_engineering/scripts/build_method_coverage.py `
  --build-manifest reverse_engineering/manifests/builds/f530404b0f3f_807de4a83df4.json `
  --type-index reverse_engineering/symbols/f530404b0f3f_807de4a83df4/assembly_csharp_types.json `
  --dump-cs <private-current-build-dump.cs> `
  --script-json <private-current-build-script.json> `
  --game-assembly <private-current-build-GameAssembly.dll> `
  --check
```

Results are 36/36 baseline, 36/36 typed, 36/36 signature application, and
36/36 read-only signature validation. Typed application imports 157
additional reachable datatypes, canonicalizes six shared bodies, and the
read-only Druid pass validates 100 ABI parameter storages with zero program
mutations.

The typed-quality check passes. Placeholder-parameter tokens fall from 245 to
zero, raw-field-offset accesses from 424 to 261, raw-integer-type tokens from
374 to 232, unresolved-type tokens from 262 to 101, and indirect-call patterns
from 17 to six. Signature-parameter-name tokens rise from 68 to 270, typed
IL2CPP-type tokens from 57 to 368, and named-field accesses from zero to 22.
Decompiler error and warning markers remain unchanged at two and 75. The
nongating raw-pointer-cast count rises from 585 to 609; the report records no
policy regression.

Adding Druid produces a 39-target-set union with 800 memberships, 512 distinct
selected FunctionDefinitions, 288 exact-definition overlap memberships, and
424 unique native RVAs. Druid contributes 36 memberships: 16 are exact
definition overlaps, 20 are newly selected definitions, and those 20 add 14
new native RVAs because six new definitions use bodies already in the union.
Across all 39 read-only target validations, all 800 memberships and 2,342 ABI
parameter storages validate with zero program mutations.

The rebuilt Assembly-CSharp ledger retains its 4,207-method census. The Druid
rows bring the classification ledger to 505 records and the evidence ledger to
259 records, with terminal native evidence for all 16 Druid-owned declarations.

## Required implementation regressions

A solver/reader bridge that claims current Druid support should pin at least:

- exact three-target cardinality, object-reference duplicate toggling, self,
  dead, hidden, and unused-active target legality;
- sorted clue IDs versus click-order `ActedInfo` references, including three
  float draws and equal-ID objects;
- truth with zero, one, and multiple registered Outcast occurrences, including
  Wretch/Spy exclusion and ordinary Doppelganger/Drunk inclusion;
- current first-stage false support `{Doppelganger, Drunk, Wretch}`, duplicate
  occurrence weighting, widened all-Outcast fallback, and exact Drunk
  exhaustion fallback;
- exact positive and none strings, newline, role-name spelling, and null versus
  non-null `CharacterData` formatting;
- cancellation, Day-only dispatch, one-use consumption, Night restoration, and
  retained history;
- real-then-raw two-event ordering and visible `savedAct` overwrite;
- no Druid runtime data and no Poet provider route; and
- the 109 complete archive records, especially the self-target fixture, as
  legacy compatibility rather than exact-current provenance.

## Remaining uncertainty and solver boundaries

- This target proves requested RNG operations and candidate support, not the
  hidden state or independence of Unity's global PRNG.
- The authored current pools establish concrete first-stage support. Exact
  duplicate probability mass in arbitrary custom scripts depends on their
  live list contents and on the shipped ascension-list duplication described
  above.
- Native click routing proves eligibility when a Character object receives the
  picker click. UI layout or animation can still make a specific hidden object
  practically unclickable in a given scene; that is outside role semantics.
- Null global singletons, null selected records, invalid CharacterData, and
  structurally corrupted picker lists follow native failure paths and were not
  forced in live play.
- Upstream code owns `dataRef`, `registerAs`, identity replacement,
  corruption/truth routing, and board construction. This checkpoint audits the
  exact Druid reads of those surfaces, not every writer.
- The textless archive cannot label any historical result as truth or bluff or
  recover its native reference chronology.
- This is a build-specific Druid checkpoint, not evidence that every role or
  the whole game has been fully decompiled.
