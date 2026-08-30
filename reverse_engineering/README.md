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
the live role-pair contract.

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
  -TargetSet gameplay_bluff_acquisition
```

`build-types` normalizes the private `il2cpp.h`, validates 5,830 inheritance
rewrites and 6,159 explicit alignments, and builds one deterministic GDT from
the union of every checked target set. The current archive contains 151,251
datatypes. Its nine-set inventory contains 166 target memberships, 156 distinct
selected FunctionDefinitions, and 153 unique native RVAs. The typed project is
separate from the baseline project. It applies only datatype graphs reachable
from the checked-in function signatures and validates exact entry points,
labels, prototypes, dynamic Windows x64 storage, and transaction completion
before writing success summaries. `typed-analyze` and `typed-all` also reopen
the saved program read-only and repeat those checks; `typed-export` requires
their fresh success summaries and opens the project read-only. `typed-refresh`
is the bounded post-analysis path for a changed canonical signature: it reopens
the preserved project with analysis disabled, reapplies all target sets, saves,
and then performs the same exact validations in a separate read-only headless
pass. Splitting those phases keeps the growing target inventory below Windows'
command-line limit.

The preserved fully analyzed typed project now covers all nine target sets
after a no-analysis refresh. Ten memberships are exact overlaps between
boundaries. Folded/shared bodies make the 156 selected definitions exceed the
153 unique native RVAs by three; each canonical native prototype is explicit
while all exact managed definitions remain in the GDT. The original full
import added 2,032 reachable datatypes and completed its analysis pass in 2,781
seconds without a timeout. Subsequent refreshes imported 49 additional
reachable datatypes, most recently reapplied and validated all 166 memberships
without rerunning auto-analysis. The final read-only pass validated all 166
memberships (156 exact definitions) and 482 membership-level parameter-storage
locations with zero program mutations.

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
`gameplay_roster_helpers`, and from 281 to 145 in
`gameplay_status_corruption_truth`, from 114 to 70 in
`gameplay_bluff_acquisition`, from 70 to 51 in `gameplay_role_slayer`, and from
14 to 5 in `gameplay_role_wretch`, and from 105 to 80 in
`gameplay_role_dreamer2`. Raw field-offset accesses fell from 237 to 144, from
243 to 120, from 678 to 289, from 76 to 41, from 421 to 148, and from 132 to 62
for the six subsystem boundaries, then from 97 to 83 for Slayer, from 20 to 8
for Wretch, and from 167 to 156 for Dreamer2. Error-marker counts did not
increase; lifecycle and the status boundary each gained one nonfatal
decompiler warning, while all three role reports retained their baseline
warning counts. The original typed import is
recorded in
[`reports/f530404b0f3f_807de4a83df4_typed_import.json`](reports/f530404b0f3f_807de4a83df4_typed_import.json),
with the new role comparisons in the
[`Slayer typed-quality report`](reports/f530404b0f3f_807de4a83df4_typed_quality_gameplay_role_slayer.json)
and
[`Wretch typed-quality report`](reports/f530404b0f3f_807de4a83df4_typed_quality_gameplay_role_wretch.json),
plus the
[`Dreamer2 typed-quality report`](reports/f530404b0f3f_807de4a83df4_typed_quality_gameplay_role_dreamer2.json).

## Method coverage

[`coverage/`](coverage/) contains the deterministic 4,207-method denominator,
sparse authored classifications, and reusable evidence. Missing classifications
resolve to `unresolved/not-reviewed`; shared native RVAs never collapse managed
method identities. The current overlay contains 147 classifications backed by
58 evidence records. See [`coverage/README.md`](coverage/README.md) for the
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
