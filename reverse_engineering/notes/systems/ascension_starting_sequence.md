# Starting-character sequence and lazy cache chronology

Build `f530404b0f3f_807de4a83df4`. The supplemental auditor
`audit_ascension_starting_sequence.py` executes four exact managed declarations:
AscensionsData.GetAllStartingCharacters (0x3B18A0),
GetAllStartingtCharacters (0x3B1BE0), GetStartingtCharactersOfType (0x3B1E10),
and GameData.GetAllStartingCharactersFromAscension (0x3DC180).
It also executes List.AddRange's native dispatch at 0xB53F50 through its
InsertRange call boundary. No Ghidra project or typed manifest was changed.

**965 native runs pass**, with ten instruction assertions and 409 distinct
executed instructions. The report's `cases` array holds one normalized
input/expected pair per run, with script IDs 1 and 2, nullable u16 asset IDs,
source arrays, custom records versus null payloads, lists, starting arrays,
consumed RNG source/width/index, every cache write, faction read order, final
cache, and output prefix. The 46 `weighted_native_cases` preserve every consumed
occurrence path for inline [B,B] and custom [null-payload,A]; exact rational
probabilities sum to one. This is an extension of `ascension_setup.md`'s typed
getter evidence, not a second claim to its 412 cases.

## Ordered behavior

AllLazy uses faction order Villager=10, Outcast=20, Minion=30, Demon=100. The
AscensionsData wrapper calls the typed getter for the first two factions and
embeds the lazy-selection logic for the last two. The GameData wrapper calls
the typed getter four times. Both allocate and construct the result list first;
the GameData wrapper then checks its profile argument for null. Its receiver
is unused and may itself be null.

Before each faction lookup, a nonnull currentPickedScript skips both source
arrays and all draws. Otherwise, inline selection happens first, including its
cache write; custom selection follows and overwrites that cache, even with a
null scriptInfo. A null selected CustomScriptData record throws before the
custom cache write and preserves the inline choice. Null source arrays fail
at their respective checks. When both arrays are empty, selection writes null
and emits the barrier each time. Unsupported typed kinds still run selection
before returning null.

A nonnull selected script supplies that faction's list through ToArray. A null
selected list fails and does not fall back. With no selected script, the
original starting array is returned directly, including null. The lazy
concatenators append each result immediately before attempting the next
faction. Native AddRange dispatch obtains the current result-list count and
forwards that insertion index, the returned collection, and exact generic
context into InsertRange. Collection conversion/insertion is an explicit
service boundary in these fixtures.

For inline [B,B] and custom [null-payload,A], the chance of first selecting A
on faction 10/20/30/100 is respectively 1/2, 1/4, 1/8, 1/16. The chance of
using all four fallback arrays is 1/16. Every attempted selection consumes
both draws, even though each inline B choice is discarded. Earlier fallback
output is retained when A is selected later. The duplicate B occurrences
remain separate draw paths. A later null custom record similarly preserves
the completed output prefix and the most recent inline write.

The typo-named AllStored helper directly appends starting arrays +0x40, +0x48,
+0x50, +0x58. It never reads or changes the selected-script cache and ignores
possibleScripts/possibleScriptsData, including null arrays. Neither helper
removes duplicate or null character occurrences.

## Failure and service limits

A failing later selection or selected-list check retains all earlier appends
and cache writes. The output list is local and has not been successfully
returned: retained prefix is diagnostic state, not a result delivered to the
caller. A failure in ToArray or InsertRange likewise stops subsequent factions.
Injected InsertRange failures in this auditor happen before the current
append; it does not claim rollback inside an arbitrary failing collection
implementation. A null fallback array is a successful Typed result but is
passed to the collection service by AllLazy/AllStored, where the fixture
rejects it as a collection failure.

The four method bodies and AddRange dispatch run privately in Unicorn 2.1.4.
RNG draws, allocation, list construction, ToArray snapshots, InsertRange array
appends, and GC barriers have authored service implementations. These services
preserve input arrays, records, and lists. No Unity PRNG state, serializer,
collection-internal failure mutations, concurrent changes, or reentrant source
mutation is reconstructed. SHA-256 checks pin GameAssembly and Dumper inputs.
The repository contains original assertions/results, not proprietary bytes or
native/decompiled bodies.

```powershell
$env:PYTHONPATH='B:\CodexTools\DemonBluffReverseEngineering\python-emulation'
python reverse_engineering/scripts/audit_ascension_starting_sequence.py --game-root 'B:\SteamLibrary\steamapps\common\Demon Bluff Playtest' --dumper-root 'B:\CodexTools\DemonBluffReverseEngineering\artifacts\f530404b0f3f_807de4a83df4\il2cppdumper-v6.7.46' --output reverse_engineering/reports/f530404b0f3f_807de4a83df4_ascension_starting_sequence_audit.json
```

## Offline Rust composition

`crates/solver-core/src/bluff/ascension_starting.rs` composes the existing
`ascension_script` selector under `ascension_starting_native_v1`. Its strict
context requires `services_preserve_inputs: true`, complete script-ID/list
mapping, nullable u16 character occurrences, and an explicit method AllLazy,
AllStored, or Typed. Missing arrays/lists are native nulls, not unknown input.
The two lazy wrappers share AllLazy semantics; null-profile allocation failure
is outside this existing-profile context.

Paths retain unconditional reduced rational probability, every consumed draw,
cache writes, faction read order, typed result or appended prefix, and concrete
failure. Selection failures do not discard their probability. Optional
one-based ToArray/append service-failure positions use the exact before-effect
fixture contract. A null Typed fallback succeeds; a null appended collection
fails. The input context is immutable. The kernel is offline and is not wired
into live solver decisions.

Bounds are 4,096 script IDs/source occurrences/elements per list, 65,536 paths,
and 1,048,576 retained output/trace entries. Integer probability overflow or
excessive support rejects the whole invocation. Unknown provenance, unknown
fields, incomplete script mappings, or missing input-preservation guarantees
are rejected. AllStored does not expand unused selection support.

Validation: `cargo test --release -p solver-core ascension_starting` passed all
six focused tests. They compare 964 native existing-profile cases and all 46
weighted occurrence paths, and check null-result distinctions, unsupported
kinds, service failures, strict provenance, support limits, and immutability.
Both Python auditors compile, and the limited diff whitespace check passes.
