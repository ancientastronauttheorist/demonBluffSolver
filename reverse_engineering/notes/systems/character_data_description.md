# CharacterData description and language reloads

This pinned-build audit executes actual `CharacterData.GetDescription`,
`tdi5845.m0015`, while the text converter and runtime gateways remain supplied.
It completes the description getter independently of the separate
Character.ShowDescription caller audit; their reports do not yet establish
execution of both actual bodies in one graph.

Source: `reverse_engineering/scripts/audit_character_data_description.py`.
Report: `reverse_engineering/reports/f530404b0f3f_807de4a83df4_character_data_description.json`.
Build: `f530404b0f3f_807de4a83df4`.

The exact Dumper signature is
`System_String_o* CharacterData__GetDescription (CharacterData_o* __this, const MethodInfo* method);`,
type signature `iii`. The unique entry is `3B4BF0`; its complete unwind range
ends exclusively at `3B4C98`, with the next managed entry at `3B4CA0`. Raw
section backing, full decoder consumption and eight trailing padding bytes
are verified. The body is 168 bytes and 46 instructions. All 45 nontrap
instructions execute; terminal `int3` at `3B4C97` is asserted and excluded.
Nineteen selected operand assertions and the body SHA-256 pin the chronology
without exporting complete native bytes or disassembly.

Exact declarations bind CharacterData.description at `50`, its unused legacy
descriptionPL/descriptionCHN fields at `58`/`60`, ProjectContext.Instance at
static offset zero and gameData at instance `20`, and GameData.language at
`20`. ELanguage defines English 0 and Polish 10. Metadata uses the exact
ProjectContext slot `271F268` and byte flag `288C4E1`. Cold metadata invokes
one supplied request at `2B7B40`; the flag becomes 1 only after completion.
Any nonzero flag bypasses it. The body performs no class initialization call;
its complete class diagnostic bytes, including E0, remain unchanged unless an
explicit reached callback writes them.

The original Data owner is captured in RBX before metadata. It loads its
current description pointer and calls supplied
`StringHelper.ConvertTextToTextWithTooltips` at `3A83C0`, passing full RDX zero.
This first conversion occurs before reading ProjectContext or its language.
Nullable input and nullable supplied output therefore remain independent of
the later context guards.

After that converter returns, the native body loads the current ProjectContext
class into R8, its static-field block into RCX, its current Instance, and that
Instance's gameData. Null Instance or gameData enters the native null gateway;
null class/static storage produces the exact diagnostic read fault. Neither
guard undoes the completed converter or metadata effects.

English 0 triggers another load of the same owner's current `description`
field and another converter call. The body then reloads the current class
slot into R8. Every other initial language retains the first captured class
in R8. The subsequent context walk reloads that captured class's static-field
pointer, Instance, gameData and language. Its loads use RAX, preserving RCX's
prior bits. A current Polish value 10 triggers a final description reload and
conversion. Thus stable English and stable Polish each convert twice, while
an English-path callback changing the language to Polish can convert three
times. Other stable DWORD values convert once.

All three native conversion sites load `description` at `50`. The legacy PL
and CHN fields are never selected by this current body, even on a Polish
language path. The report preserves their distinct pointer values and all
surrounding bytes. This observation does not establish the behavior of the
supplied converter, localization assets or other translation callers.

Callbacks distinguish replaced class versus replaced static storage,
Instance/gameData changes, exact language DWORD changes and replacement/null
description pointers. The first produced result is kept in RDX; each later
conversion replaces it. Returned RAX is exactly the most recent produced
pointer, including null or an authored alias of the input. A callback after
the final conversion can alter retained graph state without changing that
already captured result.

Every service entry records full RCX/RDX/R8/R9, its exact native call site,
full return address, cumulative ordinal and full snapshot before effects.
Unused entry registers have explicit nonzero seeds; supplied returns poison
volatile integer/XMM registers. The independent model derives the guard
residues: first context traversal sets RCX to the captured static storage;
the second preserves that RCX or converter poison and uses its captured R8.
Normal calls verify stack restoration, eight nonvolatile integer registers
and XMM6 through XMM15.

The authored graph contains full 512-byte Data windows, 256-byte class
windows and 128-byte context, game, static and text windows. Unconsumed
records and all unrelated bytes remain in every snapshot. Exact reached
callback writes are the only permitted record changes. Metadata slot and
flag state are modeled separately and compared in every full final state.
Diagnostic null class/static pointers and unmapped-owner reads are explicit
fixture contracts, not evidence that Unity or CLR admits these graphs.

The corpus contains 298 cases: 259 normal returns and 39 native stops/faults.
It crosses raw metadata bytes, language DWORDs, nullable description/output,
aliases, callback reloads and all three converter phases. Plans at skipped
phases perform no effect. Nine retained three-call sequences preserve full
prior-final to next-initial continuity, including recovery after null-owner,
Instance and gameData failures. Nine baselines generate 27 exact full stopped
prefixes; each stopped event list equals its baseline prefix and each final
state equals the interrupted entry snapshot.

An independent ordered model starts from authored initial bytes and options,
not emulator output. It verifies every complete event snapshot/raw ABI/caller,
all completed effects, partial faults/guards, counts, return bits and final
bytes for cases, retained calls, baselines and stops.

Lossless raw-memory pooling precedes full-snapshot pooling. Expand snapshots
with `audit_report_snapshots.expand_snapshots`, then raw memory with
`audit_character_oracle_reveal_join.expand_memory`. Each producer asserts
`expand_memory(expand_snapshots(report)) == full_evidence`.

Two independently launched, individually syntax-preceded final producers
emit identical 1,522,544-byte reports, SHA-256
`a8b1936a65c2becf8f984fece1ead0127f491d41d3a5c36480fcc00d0ae9cd71`.
Private peer:
`B:/CodexTools/DemonBluffReverseEngineering/artifacts/f530404b0f3f_807de4a83df4/character_data_description.peer.private.json`.
Peer review found no behavior or ABI blocker. Infrastructure checks run at
checkpoint integration. Converter internals, runtime admission, rendering,
scheduling, acquisition interleaving and native exception unwinding remain
outside this bounded audit.

Run the source with the pinned game and Dumper directories as positional
arguments and `--output`; PYTHONPATH must include the private emulation directory.

## Guarded Rust caller replay

The guarded Rust [description getter replay](notes/systems/character_data_description.md)
compares 145 normal native contexts and three complete retained sequences in
five tests. Full storage, raw service arguments, exact callers, histories and
return identities match; nominal input and future snapshot work are bounded.

Five guarded Rust tests compare 145 supported normal native CharacterData.GetDescription contexts and three complete retained three-call sequences against every represented record byte, full service-entry snapshot/raw RCX/RDX/R8/R9, exact caller/site/ordinal, return identity, cumulative history/count and final state. Stable English/Polish reconvert the same description; other raw DWORD languages retain the first result. Nullable input/results and aliases remain explicit whole supplied converter contracts. Typed active Data/class/statics/Instance/game chain, exact diagnostic sizes, disjoint overflow-safe extents, strict schema, counter overflow and complete future snapshot/history budgets validate before maps/clones; unsupported input falls back atomically. Callback mutations, guards/faults/stops, conversion internals, Unity/CLR admission, rendering, acquisition interleaving and native unwinding remain excluded.

Source: [character_data_description.rs](../../../crates/solver-core/src/bluff/character_data_description.rs).
The native fixture graph admits 15 calls and rejects 16 at the complete future
work bound; this is a graph-specific cost boundary, not a universal call limit.
Capacity checks retain prior native/service histories and account for all new
entry/completed snapshots before cloning. Unconsumed diagnostic bytes remain.
Five focused tests, all 902 Rust library tests and the release build pass.
The current reverse-engineering infrastructure suite passes 36 tests. Python
bridge and simulation suites were not rerun for this offline replay.
