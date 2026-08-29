# Method coverage ledger

Each build has a versioned coverage bundle under `coverage/<build-id>/`:

- `manifest.v1.json` pins source hashes, schema vocabulary, and audited counts.
- `methods.v1.jsonl` contains exactly one generated metadata record per managed
  `Assembly-CSharp.dll` method definition.
- `classifications.v1.jsonl` is a sparse authored overlay keyed by method ID.
- `evidence.v1.jsonl` holds reusable evidence and source links.

Missing classification rows resolve to `unresolved` with reason
`not-reviewed`, so every method remains in the denominator. Method IDs combine
the build's `TypeDefIndex` and zero-based dump order, for example
`tdi5604.m0034`. Native bodies use canonical IDs such as
`ga:rva:00380ED0`. Multiple methods may reference the same body, and generic
definitions may reference several instantiated bodies. Body evidence may be
shared, but classifications remain method-specific.

Classification states are `reconstructed`, `understood`, `generated`,
`unreachable`, and `unresolved`. Evidence levels are `metadata`,
`native-static`, `live-validated`, `behavioral`, and `hypothesis`.

Authored overlay records use these compact forms:

```json
{"method":"tdi5604.m0034","state":"reconstructed","reason":"clean-room behavior and tests exist","evidence":["ev.gameplay.start-night.1"]}
{"id":"ev.gameplay.start-night.1","level":"native-static","targets":["tdi5604.m0034","ga:rva:00380ED0"],"claim":"The authored lifecycle description matches native control flow.","sources":[{"kind":"artifact","artifact":"GameAssembly.dll","locator":{"rva":"0x380ED0"}},{"kind":"repo","path":"reverse_engineering/notes/systems/gameplay.md","anchor":"start-night"}]}
```

The public files contain only names, signatures, RVA relationships, hashes,
statuses, and authored source links. Do not add native instructions, decompiler
output, raw bytes, or reconstructed proprietary bodies.

Generate or check the current ledger from the repository root:

```powershell
python reverse_engineering/scripts/build_method_coverage.py `
  --build-manifest reverse_engineering/manifests/builds/f530404b0f3f_807de4a83df4.json `
  --type-index reverse_engineering/symbols/f530404b0f3f_807de4a83df4/assembly_csharp_types.json `
  --dump-cs 'B:\CodexTools\DemonBluffReverseEngineering\artifacts\f530404b0f3f_807de4a83df4\il2cppdumper-v6.7.46\dump.cs' `
  --script-json 'B:\CodexTools\DemonBluffReverseEngineering\artifacts\f530404b0f3f_807de4a83df4\il2cppdumper-v6.7.46\script.json' `
  --game-assembly 'B:\SteamLibrary\steamapps\common\Demon Bluff Playtest\GameAssembly.dll' `
  --check
```

The generator validates every per-type method count, all current-build aggregate
counts, source fingerprints, and that each native RVA belongs to an executable
PE section. It never overwrites the authored classification or evidence files.
