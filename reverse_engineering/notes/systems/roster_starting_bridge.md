# Starting-pool to board-exclusion native bridge

Pinned build `f530404b0f3f_807de4a83df4`. The new auditor
`audit_roster_starting_bridge.py` executes Gameplay.GetNotInDeckCharacters
(0x37C8E0), its actual AscensionsData.GetStartingtCharactersOfType callee
(0x3B1E10), and the native board enumerator. **2,122 native cases pass**, with
seven exact instruction assertions and 337 distinct executed instructions.
The report contains exact method metadata, normalized nullable u16 asset IDs,
script lists, inline/custom records, consumed RNG source/width/index, every
cache write, output prefix, list version, board removals, and precise failures.

This closes the specific service gap between the earlier starting-sequence and
roster-composition audits. It adds no new typed targets or proprietary bodies.
No GameData typed wrapper appears in this call chain: GetNotInDeck directly
resolves ProjectContext.GameData.currentTemporaryAscension and calls the
AscensionsData getter. The graph is read again before each faction request.

## Native sequence

GetNotInDeck allocates and constructs one result list, then requests starting
arrays in **Demon 100, Outcast 20, Minion 30, Villager 10** order. Each actual
getter lazily selects a script if the cache is null, converts the selected
faction list through ToArray, or returns the original starting array if no
script is selected. The caller appends that result before requesting the next
faction. Inline selections are written first; custom payloads overwrite them,
including successful null payloads. Null payloads therefore permit another
selection on the next faction.

For inline [B,B] and custom [null-payload,A], the 46 consumed occurrence paths
have exact rational mass summing to one. The first nonnull A selection can occur
on Demon, Outcast, Minion, or Villager, or no faction selects A. The probability
of each position is 1/2,1/4,1/8,1/16; all-null selection also has probability
1/16. Every attempted selection retains both draws, including the discarded
inline occurrence. Earlier fallback output stays in the list when later
factions switch to A.

Only after all four appends succeed does the method initialize Gameplay if
needed and enumerate CurrentCharacters. Each physical Character's data is
passed to Remove. Under the explicit reference-equality service, one first
matching occurrence is removed per board entry; duplicate source entries and
null Character.data references are observable. A null physical Character fails
without processing later board entries. The tests also cover excess duplicate
board references whose removals return false and do not change list version.

## Failure chronology

An early append failure prevents all later selection calls. A later null custom
record leaves the last inline cache write and all earlier appends. A null
selected script list fails inside the typed getter, whereas a null fallback
starting array successfully leaves the getter and fails at range append. A
ToArray failure happens before that faction's append. Global class-initializer,
null CurrentCharacters, physical-character, or Remove failure happens after the
completed getter/append sequence and preserves its cache and output.

The native report distinguishes these failures and records all intermediate
cache writes. Its local output prefix is diagnostic state: the caller receives
no successful result when the method fails. Injected service failures stop
before the current service's effect; managed unwinding/finally is not modeled.

## Services and fixed-profile contract

Actual native selection, source-array indexing/null checks, cache publication,
caller ordering, global pointer reads, and board enumeration execute in
Unicorn 2.1.4. List.ToArray returns an authored snapshot; allocation, AddRange,
Remove comparison, GC barriers, and class initialization are controlled
services. Input arrays/lists, assets, and the temporary-profile graph remain
stable through those services. RNG indices are supplied explicitly and weighted
uniformly by occurrence; no Unity PRNG state is recovered. Runtime comparers,
Unity destroyed-object equivalence, mode selection, profile replacement,
serialization, and the managed exception unwinder remain separate boundaries.

## Offline Rust composition

`crates/solver-core/src/bluff/roster_starting_bridge.rs` composes the existing
starting and roster kernels under `roster_starting_bridge_native_v1`, without
modifying them. A strict context requires an explicitly stable temporary
profile, the existing starting provenance, and the roster equality/callback/
capacity provenance. Its roster starting-pool fields must be empty placeholders;
the bridge supplies the actual harvested arrays, avoiding contradictory inputs.

The bridge invokes Typed starting replay in 100,20,30,10 order and carries the
cache, draws, write trace, and global ToArray-attempt counter across calls.
After each getter, it checks the caller's append boundary before attempting
another selection. The public roster kernel projects the elapsed list operations
and final board-removal phase using those harvested arrays. On a getter failure,
a typed-service stop projects the same completed list prefix, while the bridge
reports the actual getter failure as the cause. No second selection or RNG call
is introduced by this projection.

Paths retain unconditional rational probability, final cache, cache writes,
consumed draws, typed request order, list identities/contents/versions, local
output, successful returned identity, board removal trace, and concrete failure.
Unsupported provenance or support/capacity overflow rejects the entire replay.
The source context remains unchanged. This is an offline model with no live
solver integration.

Six authored Rust tests compare every native case, all 46 weighted paths,
mixed-source board removals, early append/later getter failures, and strict
profile/equality/capacity rejection. The native auditor and Python compilation
pass; Rust validation is coordinated by the parent task.

```powershell
$env:PYTHONPATH='B:\CodexTools\DemonBluffReverseEngineering\python-emulation'
python reverse_engineering/scripts/audit_roster_starting_bridge.py --game-root 'B:\SteamLibrary\steamapps\common\Demon Bluff Playtest' --dumper-root 'B:\CodexTools\DemonBluffReverseEngineering\artifacts\f530404b0f3f_807de4a83df4\il2cppdumper-v6.7.46' --output reverse_engineering/reports/f530404b0f3f_807de4a83df4_roster_starting_bridge_audit.json
```

Validation: the parent broad release library run passed all **726 tests**,
including all six bridge tests and all 2,122 native bridge fixtures plus the
46 weighted paths. This run also included the final score MXCSR guard, roster
capacity correction, and filter query traces. Python compilation and the limited
diff whitespace check pass.

The combined retention limit counts both finished paths and active prefixes.
Boundary regressions cover an exact-fit budget, one-over rejection, and mixed
active/failed paths without publishing partial results on capacity rejection.
