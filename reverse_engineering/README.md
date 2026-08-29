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
also native-audited, closing all 30 selected methods.

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

# Repeat the post-save, read-only signature/ABI validation without reanalysis.
powershell -ExecutionPolicy Bypass -File `
  reverse_engineering/scripts/invoke_ghidra.ps1 `
  -GameRoot 'B:\SteamLibrary\steamapps\common\Demon Bluff Playtest' `
  -Stage typed-validate

powershell -ExecutionPolicy Bypass -File `
  reverse_engineering/scripts/invoke_ghidra.ps1 `
  -GameRoot 'B:\SteamLibrary\steamapps\common\Demon Bluff Playtest' `
  -Stage typed-export `
  -TargetSet gameplay_roster_helpers
```

`build-types` normalizes the private `il2cpp.h`, validates 5,830 inheritance
rewrites and 6,159 explicit alignments, and builds a 151,120-datatype GDT. The
typed project is separate from the baseline project. It applies only datatype
graphs reachable from the checked-in function signatures and validates exact
entry points, labels, prototypes, dynamic Windows x64 storage, and transaction
completion before writing success summaries. `typed-analyze` and `typed-all`
also reopen the saved program read-only and repeat those checks; `typed-export`
requires their fresh success summaries and opens the project read-only. The
three checked target sets cover 47 methods, all of which survived the complete
2,458-second analysis pass with 121 parameter-storage locations validated.

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
`gameplay_core`, from 370 to 149 in `gameplay_lifecycle`, and from 38 to 34 in
`gameplay_roster_helpers`. Raw field-offset accesses fell from 237 to 144, from
678 to 289, and from 76 to 41, respectively. Error-marker counts did not
increase; lifecycle gained one nonfatal type-propagation warning in
`Characters.ManageCharacters`. The current public counts and artifact
observations are recorded in
[`reports/f530404b0f3f_807de4a83df4_typed_import.json`](reports/f530404b0f3f_807de4a83df4_typed_import.json).

## Method coverage

[`coverage/`](coverage/) contains the deterministic 4,207-method denominator,
sparse authored classifications, and reusable evidence. Missing classifications
resolve to `unresolved/not-reviewed`; shared native RVAs never collapse managed
method identities. See [`coverage/README.md`](coverage/README.md) for the
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
