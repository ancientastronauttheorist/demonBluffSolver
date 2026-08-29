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

Create a symbolized Ghidra project in two steps:

```powershell
powershell -ExecutionPolicy Bypass -File `
  reverse_engineering/scripts/invoke_ghidra.ps1 `
  -GameRoot 'B:\SteamLibrary\steamapps\common\Demon Bluff Playtest' `
  -Stage import

powershell -ExecutionPolicy Bypass -File `
  reverse_engineering/scripts/invoke_ghidra.ps1 `
  -GameRoot 'B:\SteamLibrary\steamapps\common\Demon Bluff Playtest' `
  -Stage analyze
```

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
notes/systems/           Authored subsystem reconstruction notes
notes/roles/             Authored per-role reconstruction notes
reports/                 Coverage and build-diff summaries
fixtures/synthetic/      Redistributable validation fixtures
work/                    Ignored raw inputs and local analysis projects
generated/               Ignored reproducible bulk output
private/                 Ignored non-redistributable material
```

See [`ROADMAP.md`](ROADMAP.md) for the completion definition and milestones.
