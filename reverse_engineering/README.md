# Demon Bluff Reverse Engineering

This directory tracks the reproducible reconstruction of Demon Bluff's Unity
IL2CPP gameplay code. The objective is not to recover the developer's exact
original C# formatting or comments—IL2CPP compilation discards those—but to
account for every game-owned type and method and reconstruct behavior precisely
enough to validate it against the installed game.

## Public-repository boundary

This repository is public. Commit only our own tooling, manifests, normalized
symbol indexes, offset evidence, behavioral notes, clean-room reconstruction,
and synthetic fixtures.

Do **not** commit the game binaries, Unity assets, bulk dumper output, dummy
assemblies, native-analysis databases, memory dumps, or extracted media. These
are ignored under `work/`, `generated/`, and `private/`. Raw artifacts should be
kept on a separate backed-up private store and keyed by the hashes in the public
build manifest.

## Current build

- Steam app: `3749680` (`Demon Bluff Playtest`)
- Steam build: `23084916`
- Unity: `2022.3.10f1`
- Architecture: Windows x86-64
- IL2CPP metadata: version 29
- Build ID: `f530404b0f3f_807de4a83df4`

The immutable input fingerprints are in
[`manifests/builds/`](manifests/builds/). Tool releases and archive checksums are
in [`toolchain/toolchain.lock.json`](toolchain/toolchain.lock.json).

## Reproduce the metadata dump

From PowerShell at the repository root:

```powershell
python reverse_engineering/scripts/build_manifest.py `
  --game-root 'B:\SteamLibrary\steamapps\common\Demon Bluff Playtest' `
  --steam-manifest 'B:\SteamLibrary\steamapps\appmanifest_3749680.acf'

powershell -ExecutionPolicy Bypass -File `
  reverse_engineering/scripts/invoke_il2cppdumper.ps1 `
  -GameRoot 'B:\SteamLibrary\steamapps\common\Demon Bluff Playtest'
```

The dumper command downloads the pinned release, verifies its checksum, and
writes raw output outside the repository by default:

```text
B:\CodexTools\DemonBluffReverseEngineering\artifacts\
  f530404b0f3f_807de4a83df4\il2cppdumper-v6.7.46\
```

Index the result without committing the raw files:

```powershell
python reverse_engineering/scripts/index_il2cpp_dump.py `
  --dump-cs '<artifact-dir>\dump.cs' `
  --script-json '<artifact-dir>\script.json' `
  --output-dir reverse_engineering/generated/current
```

Recover native methods into managed IL with the pinned Cpp2IL development
commit, then render the recovered assembly as local C# with ILSpyCmd:

```powershell
powershell -ExecutionPolicy Bypass -File `
  reverse_engineering/scripts/invoke_cpp2il.ps1 `
  -GameRoot 'B:\SteamLibrary\steamapps\common\Demon Bluff Playtest'
```

Cpp2IL's “success” count means it emitted IL for a method, not that the output
is source-accurate. Complex methods contain explicit recovery warnings and must
be checked against Ghidra/native instructions and live behavior. The checked-in
quality report keeps those warnings visible.

Create a symbolized Ghidra project and export the first native target set in
three stages:

```powershell
powershell -ExecutionPolicy Bypass -File `
  reverse_engineering/scripts/invoke_ghidra.ps1 `
  -GameRoot 'B:\SteamLibrary\steamapps\common\Demon Bluff Playtest' `
  -Stage import

powershell -ExecutionPolicy Bypass -File `
  reverse_engineering/scripts/invoke_ghidra.ps1 `
  -GameRoot 'B:\SteamLibrary\steamapps\common\Demon Bluff Playtest' `
  -Stage analyze

powershell -ExecutionPolicy Bypass -File `
  reverse_engineering/scripts/invoke_ghidra.ps1 `
  -GameRoot 'B:\SteamLibrary\steamapps\common\Demon Bluff Playtest' `
  -Stage export-core
```

Export any other checked-in target set from the completed baseline project with
the generic, read-only target exporter:

```powershell
powershell -ExecutionPolicy Bypass -File `
  reverse_engineering/scripts/invoke_ghidra.ps1 `
  -GameRoot 'B:\SteamLibrary\steamapps\common\Demon Bluff Playtest' `
  -Stage export-target `
  -TargetSet gameplay_lifecycle
```

The import and analysis stages write explicit completion summaries; the wrapper
rejects cancelled imports, analysis timeouts, missing target symbols, partial
exports, stale files, and filename collisions. Raw Ghidra state remains outside
the public repository under the build-keyed artifact directory. The current
`gameplay-core` export is expected to produce 13 of 13 functions. Folded native
bodies retain every requested managed-method identity in their export headers.
The generic exporter applies the same build, RVA, signature, filename, and
count checks without mutating or reanalyzing the saved project; the current
`gameplay_lifecycle` boundary exports 28 of 28 functions. All 28 entries now
carry native-static behavioral coverage in the
[`gameplay_lifecycle` audit](notes/systems/gameplay_lifecycle.md).
The subsequent `gameplay_execution_resolution` boundary baseline-exports 30 of
30 functions, including four explicitly tracked shared-body identities. Its
first 16-method slice now has native-static coverage for action/lying dispatch,
wrong-execution damage, Knight and Doppelganger protection, and terminal result
selection. The remaining 14-method status-insertion and Striga night slice is
also native-audited, closing all 30 selected methods. The next
`gameplay_status_corruption_truth` boundary maps 40 methods and
baseline-exports all 40 without failure. Eight entries deliberately reuse exact
identities from earlier boundaries. Its `Characters.FilterRealCharacterType`
overload uses the optional `prototype_name` field to give Ghidra's C datatype
parser a unique definition name while preserving the exact metadata signature
in `signature`; target validation still checks that unmodified signature and
RVA against `script.json`. All 40 status-storage, selection, truth/appearance,
corruption-producer, Alchemist, and bluff-orchestration methods now have
native-static coverage in the
[`status/corruption/truth audit`](notes/systems/gameplay_status_corruption_truth.md).
The subsequent `gameplay_bluff_acquisition` boundary adds 20 checked methods
covering common assignment, Demon/Minion/Spy/Mutant selection, bluff pools,
script registration, and fresh-card creation. Two entries are exact overlaps
with earlier sets; CharacterData overloads receive explicit typed prototype
aliases, and the Helpers/Calculator `RollDice` shared native body retains both
managed identities. All 20 are documented in the
[`bluff-acquisition audit`](notes/systems/gameplay_bluff_acquisition.md).
The first per-role boundaries add all ten Slayer methods and all seven Wretch
methods (the latter is managed internally as `Recluse`). Their paired
[`Slayer`](notes/roles/gameplay_role_slayer.md) and
[`Wretch`](notes/roles/gameplay_role_wretch.md) audits join registered
alignment, picker dispatch, the folded Wretch action body, and Slayer's
kill-and-reveal behavior to the live Wretch regression.
The next role-class boundary adds all ten `Dreamer2` methods plus the two
`GetDreamerClue` virtual providers. Its
[`Dreamer2`](notes/roles/gameplay_role_dreamer2.md) audit reconstructs the
two-character randomized type-exclusion result. Serialized asset evidence
binds the public card to managed `Dreamer`, however, and finds no current
gameplay binding for `Dreamer2`; the alternate class therefore does not replace
the live role-pair contract. The subsequent
[`public Dreamer`](notes/roles/gameplay_role_dreamer.md) boundary adds all 11
role methods plus five compiler-generated helpers. It reconstructs the exact
weighted role-pair and Cabbage paths and proves that all 46 shipped core
CharacterData assets currently have `usuallyDisguised == false`. The resulting
board-entry fallback can truthfully emit both selected roles, while a selected
bluff can make a lying clue collide with the other target's real role.
The following [`Baa`](notes/roles/gameplay_role_baa.md) boundary asset-binds
the public card to managed `Imp` and adds all three role methods plus the two
deck-view helpers carrying its visible effect. It proves that Baa obscures one
existing Outcast identity at Start and removes that exact entry on any Baa
death; no gameplay role is added and no board card is flipped.
The next [`Shaman`](notes/roles/gameplay_role_shaman.md) boundary asset-binds
the public card to managed `Illuzionist` and distinguishes it from the public
Witch's managed `Cipher`. Its four role methods plus seven exact selection,
status, and lifecycle helpers show an ordered apparent-Villager source and
destination clone: the source stays in place, the destination is overwritten,
and the destination's truth/status and runtime state survive the replacement.
The following
[`Plague Doctor`](notes/roles/gameplay_role_plague_doctor.md) boundary
asset-binds the public card to managed `Puzzlemaster`. All 11 role methods plus
12 exact dispatch, click, picker, status, and filter helpers establish the
Start corruption pool, truthful and lying Day branches, apparent-alignment
random result pools, two-entry acted-information shape, self override, and raw
Corrupted-status handling. In particular, Drunk has no native blanket-clean
exception: the asc84_v2 generated Drunk was compatible with retained
Alchemist resistance blocking its self-corruption.
The next [`Judge`](notes/roles/gameplay_role_judge.md) boundary asset-binds the
public card to managed `Judge2` and proves that the similar `Arbiter` class is
unbound. All ten Judge2 methods plus eight exact dispatch, truth-appearance,
click, and picker helpers establish its unrestricted one-target selection,
exact one-reference result, deterministic normal/bluff inversion, and
`ResetAfterNight` history. A corrupted Good Judge takes `BluffAct` and
deterministically negates target lying appearance; it is not an unconstrained
result.
The following [`Witch`](notes/roles/gameplay_role_witch.md) boundary
asset-binds the public card to managed `Cipher`. All five Cipher methods plus
14 exact Start, inherited-dispatch, player-value, click, reset, and ordinary/
night-death helpers establish a global scalar reveal quota rather than a stored
target: the ordinary transition is allowed exactly while block count is less
than the number of current Hidden states, killed-hidden cards are excluded once
Dead, picker and execution paths bypass the gate, and exact real Witch death
reduces the quota. Inherited `Role.BluffAct` forwards the Evil Witch's Start
back to concrete `Cipher.Act`, closing the apparent lying-dispatch ambiguity.
The next
[`Chancellor/Baron and Witness`](notes/roles/gameplay_role_chancellor.md)
boundary adds all five managed Baron methods, all eight Witness methods, and
18 exact ordered-Start, selection, status, mutation, and roster helpers. It
proves that Chancellor first replaces an anywhere non-Dead real Villager with
an added Outcast role, then independently marks an apparent-Outcast anchor and
swaps Chancellor role data with one alive circular neighbour. The resulting
`c/v/o/f/a` equations distinguish the first target, marker anchor, final
Chancellor, and generated-Outcast home. Witness reads only surviving current
physical `MessedUpByEvil` status: the first Villager is not marked merely by
replacement, dead markers remain eligible, truthful NO requires zero markers,
and lying NO requires every physical card to be marked.
The combined
[`Lilis/Knight`](notes/roles/gameplay_roles_lilis_knight.md) boundary
asset-binds the public cards to managed `Striga` and `Immortal`. It includes
all three Lilis methods, all ten Knight methods, and 41 exact ordered-Start,
Night-rule, victim-filter, delayed-kill, ordinary-execution, Slayer, HP,
status, and reset helpers. The native audit proves a hard registered-Good
first pass rather than weighted selection, fixed two-HP cost on every live
Lilis Night attempt, no reroll after protected or colliding delayed targets,
and per-physical-duplicate Night actions despite one same-asset Start actor.
Knight protection follows HealthyBluff, Corrupted, then runtime-Evil
precedence. A corrupted runtime-Good Knight costs the ordinary five HP plus a
fixed additional four, for nine total; Lilis and Slayer never run that
OnExecuted hook.
The following [`Rambler`](notes/roles/gameplay_role_rambler.md) boundary
asset-binds the public card to managed `Rambler2` and covers all 14 role
methods, both compiler-generated closure methods, and 20 exact setup,
truth-dispatch, adjacency, reveal, interference, and acted-history helpers.
It proves that interference installs during each physical card's internal
pre-flip AfterRoundStart reveal. Clean real and HealthyBluff fake Rambler
surfaces target appearance-truthful neighbours; corrupted real and ordinary
lying fake surfaces target appearance-lying neighbours. Hidden targets retain
persistent callbacks which replace the imminent acted record with exactly one
Rambler reference, while already non-Hidden targets receive a separate history
entry. User-reveal quotes are constraint-free but carry exact circular
predecessor/successor references, including duplicate small-board entries.

Build the deterministic IL2CPP datatype archive, create the isolated typed
project, analyze it, and export any checked-in target set with:

```powershell
powershell -ExecutionPolicy Bypass -File `
  reverse_engineering/scripts/invoke_ghidra.ps1 `
  -GameRoot 'B:\SteamLibrary\steamapps\common\Demon Bluff Playtest' `
  -Stage build-types

powershell -ExecutionPolicy Bypass -File `
  reverse_engineering/scripts/invoke_ghidra.ps1 `
  -GameRoot 'B:\SteamLibrary\steamapps\common\Demon Bluff Playtest' `
  -Stage typed-import

powershell -ExecutionPolicy Bypass -File `
  reverse_engineering/scripts/invoke_ghidra.ps1 `
  -GameRoot 'B:\SteamLibrary\steamapps\common\Demon Bluff Playtest' `
  -Stage typed-analyze

# Reapply the current canonical signatures to an already analyzed project,
# with analysis disabled, then validate every checked target set.
powershell -ExecutionPolicy Bypass -File `
  reverse_engineering/scripts/invoke_ghidra.ps1 `
  -GameRoot 'B:\SteamLibrary\steamapps\common\Demon Bluff Playtest' `
  -Stage typed-refresh

# Repeat the post-save, read-only signature/ABI validation without reanalysis.
powershell -ExecutionPolicy Bypass -File `
  reverse_engineering/scripts/invoke_ghidra.ps1 `
  -GameRoot 'B:\SteamLibrary\steamapps\common\Demon Bluff Playtest' `
  -Stage typed-validate

powershell -ExecutionPolicy Bypass -File `
  reverse_engineering/scripts/invoke_ghidra.ps1 `
  -GameRoot 'B:\SteamLibrary\steamapps\common\Demon Bluff Playtest' `
  -Stage typed-export `
  -TargetSet gameplay_role_rambler
```

`build-types` normalizes the private `il2cpp.h`, validates 5,830 inheritance
rewrites and 6,159 explicit alignments, and builds one deterministic GDT from
the union of every checked target set. The current archive contains 151,381
datatypes. Its eighteen-set inventory contains 379 target memberships, 274
distinct selected FunctionDefinitions, and 259 unique native RVAs. The typed
project is
separate from the baseline project. It applies only datatype graphs reachable
from the checked-in function signatures and validates exact entry points,
labels, prototypes, dynamic Windows x64 storage, and transaction completion
before writing success summaries. `typed-analyze` and `typed-all` also reopen
the saved program read-only and repeat those checks; `typed-export` requires
their fresh success summaries and opens the project read-only. `typed-refresh`
is the bounded post-analysis path for a changed canonical signature: it reopens
the preserved project with analysis disabled, reapplies all target sets, saves,
and then performs the same exact validations in a separate read-only headless
pass. At eighteen target sets the single all-target `typed-refresh` wrapper can
exceed Windows' command-line limit before Ghidra launches. The current refresh
therefore ran the same apply/save and read-only validation stages in two
serialized nine-target batches; Ghidra commands still must not
overlap on the saved project.

The preserved fully analyzed typed project now covers all eighteen target sets
after a no-analysis refresh. One hundred five memberships are exact overlaps
between boundaries. Folded/shared bodies make the 274 selected definitions
exceed the 259 unique native RVAs by fifteen; each canonical native prototype is explicit
while all exact managed definitions remain in the GDT. The original full
import added 2,032 reachable datatypes and completed its analysis pass in 2,781
seconds without a timeout. Subsequent refreshes imported 121 additional
reachable datatypes, including 40 for the public Dreamer boundary, six for
Baa, eight for Plague Doctor, six for Judge, and 12 for Witch; the latest
Chancellor/Witness refresh imported 12 more, and the first Lilis/Knight refresh
imported 212 more. The Rambler refresh imported 36 more. The two-batch refresh
reapplied and validated all 379 memberships without rerunning auto-analysis.
The final read-only pass validated all 379 memberships (274 exact definitions) and 1,112
membership-level parameter-storage locations with zero program mutations.

The signature-application ABI check now derives each of the first four Win64
register families from the parameter datatype: integer and pointer parameters
use `RCX`/`RDX`/`R8`/`R9` at their ordinal, while `float` and `double` use the
corresponding `XMM0`-`XMM3` family. It also recognizes Ghidra's width-specific
names such as `XMM2_Da` as members of `XMM2`. Rambler's mixed float-argument
`Character.InterfereActed` and `Character.InterfereDelay` signatures exercise
this path. This is an ABI-validator correction, not a gameplay inference.

Compare private baseline and typed exports without putting decompiled bodies or
private paths in the public report:

```powershell
python reverse_engineering/scripts/audit_ghidra_type_quality.py `
  --baseline '<baseline-export-dir>' `
  --typed '<typed-export-dir>' `
  --output reverse_engineering/reports/<report-name>.json `
  --check
```

For the current exports, unresolved-type tokens fell from 78 to 43 in
`gameplay_core`, from 160 to 96 in `gameplay_execution_resolution`, from 370
to 140 in `gameplay_lifecycle`, from 38 to 34 in
`gameplay_roster_helpers`, from 281 to 145 in
`gameplay_status_corruption_truth`, and from 114 to 70 in
`gameplay_bluff_acquisition`. The role boundaries fell from 70 to 51 for
Slayer, from 14 to 5 for Wretch, from 105 to 80 for Dreamer2, and from 190 to
92 for the public Dreamer, from 25 to 17 for Baa, from 95 to 18 for Shaman,
from 190 to 123 for Plague Doctor, from 146 to 80 for Judge, from 119 to 30 for
Witch, from 237 to 105 for Chancellor/Witness, and from 387 to 154 for the
combined Lilis/Knight boundary. Rambler fell from 405 to 103.
Raw field-offset accesses fell from 237 to 144, from
243 to 120, from 678 to 289, from 76 to 41, from 421 to 148, and from 132 to 62
for the six subsystem boundaries, then from 97 to 83 for Slayer, from 20 to 8
for Wretch, from 167 to 156 for Dreamer2, from 370 to 186 for the public
Dreamer, from 33 to 28 for Baa, from 144 to 21 for Shaman, from 329 to 223
for Plague Doctor, from 268 to 175 for Judge, and from 241 to 89 for Witch.
The Chancellor/Witness boundary fell from 294 raw field-offset accesses to
102, Lilis/Knight fell from 581 to 203, and Rambler fell from 699 to 95.
Error-marker counts did not increase;
lifecycle and the status boundary each gained one nonfatal decompiler warning,
eight role reports retained their baseline warning counts, and Witch and the
Chancellor/Witness boundary each gained one nonfatal warning marker. The
Lilis/Knight boundary retained four error markers and gained one nonfatal
warning; placeholder parameters fell from 451 to zero and indirect-call
patterns from 44 to 12. Rambler retained zero error markers, gained one
nonfatal warning, reduced placeholder parameters from 436 to zero, and reduced
indirect-call patterns from 16 to three. The original typed import is
recorded in
[`reports/f530404b0f3f_807de4a83df4_typed_import.json`](reports/f530404b0f3f_807de4a83df4_typed_import.json),
with the new role comparisons in the
[`Slayer typed-quality report`](reports/f530404b0f3f_807de4a83df4_typed_quality_gameplay_role_slayer.json)
and
[`Wretch typed-quality report`](reports/f530404b0f3f_807de4a83df4_typed_quality_gameplay_role_wretch.json),
plus the
[`Dreamer2 typed-quality report`](reports/f530404b0f3f_807de4a83df4_typed_quality_gameplay_role_dreamer2.json)
and
[`public Dreamer typed-quality report`](reports/f530404b0f3f_807de4a83df4_typed_quality_gameplay_role_dreamer.json).
The Baa comparison is in the
[`Baa typed-quality report`](reports/f530404b0f3f_807de4a83df4_typed_quality_gameplay_role_baa.json),
and the Shaman comparison is in the
[`Shaman typed-quality report`](reports/f530404b0f3f_807de4a83df4_typed_quality_gameplay_role_shaman.json).
The Plague Doctor comparison is in the
[`Plague Doctor typed-quality report`](reports/f530404b0f3f_807de4a83df4_typed_quality_gameplay_role_plague_doctor.json).
The Judge comparison is in the
[`Judge typed-quality report`](reports/f530404b0f3f_807de4a83df4_typed_quality_gameplay_role_judge.json).
The Witch comparison is in the
[`Witch typed-quality report`](reports/f530404b0f3f_807de4a83df4_typed_quality_gameplay_role_witch.json).
The Chancellor/Witness comparison is in the
[`Chancellor/Witness typed-quality report`](reports/f530404b0f3f_807de4a83df4_typed_quality_gameplay_role_chancellor.json).
The combined boundary comparison is in the
[`Lilis/Knight typed-quality report`](reports/f530404b0f3f_807de4a83df4_typed_quality_gameplay_roles_lilis_knight.json).
The current Rambler comparison is in the
[`Rambler typed-quality report`](reports/f530404b0f3f_807de4a83df4_typed_quality_gameplay_role_rambler.json).

## Method coverage

[`coverage/`](coverage/) contains the deterministic 4,207-method denominator,
sparse authored classifications, and reusable evidence. Missing classifications
resolve to `unresolved/not-reviewed`; shared native RVAs never collapse managed
method identities. The current overlay contains 275 classifications backed by
112 evidence records. See [`coverage/README.md`](coverage/README.md) for the
generation and byte-for-byte check command.

## Evidence levels

Every behavioral or layout claim should carry one of these labels:

- `metadata`: directly present in IL2CPP metadata, such as a field offset.
- `native-static`: recovered from disassembly/decompilation.
- `live-validated`: confirmed against paired UI and process-memory observations.
- `behavioral`: confirmed through controlled gameplay or regression traces.
- `hypothesis`: plausible but not yet validated.

Memory remains validation-only during solver play, per `AGENTS.md`.

## Directory map

```text
manifests/builds/       Immutable hashes and build identity
toolchain/              Pinned tool versions and configuration
scripts/                Reproducible extraction and indexing tools
offsets/                Versioned field/RVA evidence
symbols/                Small normalized, reviewable symbol indexes
coverage/               Complete method denominator and sparse evidence
notes/systems/           Authored subsystem reconstruction notes
notes/roles/             Authored per-role reconstruction notes
reports/                 Coverage and build-diff summaries
fixtures/synthetic/      Redistributable validation fixtures
work/                    Ignored raw inputs and local analysis projects
generated/               Ignored reproducible bulk output
private/                 Ignored non-redistributable material
```

See [`ROADMAP.md`](ROADMAP.md) for the completion definition and milestones.
