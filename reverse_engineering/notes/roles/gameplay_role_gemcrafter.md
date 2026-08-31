# Gemcrafter / Archivist current-build native contract

This note records the clean-room native checkpoint for the shipped public
**Gemcrafter** role in build `f530404b0f3f_807de4a83df4`. The public asset is
implemented by managed `Archivist` (`TypeDefIndex 5884`). This is a bounded
role checkpoint, not a claim that the whole game is decompiled.

## Public asset binding and managed identity

`Demon Bluff_Data/sharedassets0.assets` contains the shipped Gemcrafter
`CharacterData` at path ID `21620`, object offset `23671032`, size `2908`, and
object SHA-256
`E54A0220E3455FF0D3679180994B45E02321D0E1950AE9C5B8DF08CD6CC96A73`.
The containing asset has SHA-256
`E0D239C03FA47EC138F9C2797779E83B65AEF41F30C5AD84D4CA76767A93E967`.
Its serialized identity is:

- public name `Gemcrafter`;
- character ID `Archivist_34476114`;
- managed role type string `Archivist`, at raw object offset `0xB38`;
- Good Villager (`type == 10`, `startingAlignment == 10`);
- `abilityUsage == 10`, the serialized `ResetAfterNight` usage category;
- bluffable, not usually disguised, and non-picking; and
- empty bundled-character, skin, achievement, additional-status, hint,
  if-lying, tag, and `canAppearIf` collections.

The asset's authored public description is exactly:

```text
Learn 1 Good character.
```

The managed description getter independently returns the same words without
the terminal period:

```text
Learn 1 Good character
```

The normal serialized `level0` candidate pool at path ID `139347` contains one
file-ID-`2` reference to this exact path ID, at object-local offset `536`. The
candidate-pool object has SHA-256
`FB9D821AE0A7E3655BEF4A3DD3E544E85B3109258A48DCF68FF0969ACED8D948`;
`level0` has SHA-256
`B509AC15904F0926419CCCC1D2E86508C69B0E3AF2235A08FF8E9FC4D170C7F1`.
Gemcrafter is therefore live as a direct Standard/Ascension card independently
of Poet. It has zero references in the 15-entry `startGameActOrder` object at
path ID `137026`, whose object SHA-256 is
`544328634CD77D551B5864CDC1B643029F3B30BFFC5BB4350DFCF83C66226BB0`.

## Exact callable boundary and shared bodies

The exact managed declaration boundary contains seven FunctionDefinitions and
no Archivist-owned compiler-generated type:

| Managed identity | RVA | Purpose |
| --- | ---: | --- |
| `Archivist.get_Description` | `0x3D3BA0` | Managed description text |
| `Archivist.GetInfo` | `0x3D3910` | Registered-Good truth selection and result |
| `Archivist.Act` | `0x3B09F0` | Truth Day callback |
| `Archivist.BluffAct` | `0x3B33E0` | Bluff Day callback |
| `Archivist.GetBluffInfo` | `0x3D3680` | Registered-Evil false clue |
| `Archivist.ConjourInfo` | `0x3D3620` | Independently callable one-ID formatter |
| `Archivist..ctor` | `0x3CFFF0` | Fieldless role and Poet-provider construction |

The target also pins four semantic callees: `ActedInfo..ctor`, the Character
overload of `Characters.FilterAlignmentCharacters`,
`Character.GetRegisterAlignment`, and integer `UnityEngine.Random.Range`.
Generic list, string-format, callback-sink, Unity-object, and PRNG internals
stop outside the role-specific boundary.

Four target memberships have globally shared native bodies. The truth and
bluff Day dispatchers have 17 and 15 metadata aliases respectively; the role
constructor has 537, and integer Range has two. Typed application canonicalizes
the three role memberships whose ABI-compatible shared prototypes are already
pinned. Exact managed identities remain distinct even where native code is
folded.

The reachable truth and bluff bodies format their sentence inline. They do not
invoke `ConjourInfo`; that separately declared method returns the same exact
format for an arbitrary supplied integer. This checkpoint retains it because
the assignment audits every Archivist declaration, but does not infer a hidden
second clue path from its existence.

## Registered-alignment projection

Both clue generators synchronously copy `Gameplay.CurrentCharacters` and pass
that copy through the Character overload of
`Characters.FilterAlignmentCharacters`. For each occurrence, the filter uses
the live Unity-non-null `registerAs` CharacterData's `startingAlignment` and
otherwise the Character's stored `alignment`. The independently audited
`Character.GetRegisterAlignment` exposes the same projection.

The filter does not query apparent role, current role/dataRef, runtime faction,
corruption, truth appearance, status, reveal state, death, visibility, or
physical origin. Its candidates are therefore **registered Good** or
**registered Evil** occurrences in the live global list. Dead, executed,
hidden, and unrevealed characters remain eligible if upstream lifecycle code
retains them in `CurrentCharacters`.

Corruption matters upstream because it routes a registered-Good actor to the
bluff action; it does not move that actor into Archivist's registered-Evil
pool. Identity movers and registration writers matter only through the live
`CurrentCharacters`, `registerAs`, stored-alignment, or truth-routing surfaces.
Archivist caches none of them.

## Truth selection and actor handling

`Archivist.GetInfo` filters the copied board to registered Good (`10`). It then
checks the **pre-removal** pool count. Only when that count is greater than one
does it attempt to remove one matching `charRef` occurrence. It next makes
exactly one max-exclusive integer `Random.Range(0, count)` draw and selects the
indexed post-conditional-pool occurrence.

Let `N` be the pool size after that conditional removal. Every occurrence has
conditional probability `1 / N`; the role performs no weighting, shuffle,
retry, replacement, role test, or second draw. Duplicate list occurrences, if
supplied by malformed global state, remain separate index outcomes.

The count guard creates exact actor behavior that a consuming solver must not
normalize away:

- a normally truthful registered-Good Gemcrafter is excluded when at least one
  other registered-Good occurrence exists;
- when the actor is the only registered-Good occurrence, the count is one, no
  removal is attempted, and the clue selects self;
- if `charRef` is absent from a pool of size greater than one, the removal is
  attempted but the pool remains unchanged; and
- a malformed truth call with a non-Good actor selects from all registered-Good
  occurrences because that actor is absent from the pool.

This conditional is pool-wide, not a general `target != actor` rule.

## Bluff selection and actor handling

`Archivist.GetBluffInfo` is structurally identical except that it filters for
registered Evil (`20`). It applies the same pre-removal `count > 1` guard,
makes one integer Range draw, and selects one post-conditional-pool occurrence.
The printed claim still calls that selected Evil character Good.

Consequently:

- a direct registered-Evil actor displaying Gemcrafter is excluded only when
  another registered-Evil occurrence exists;
- if that actor is the sole registered-Evil occurrence, the size-one guard
  preserves it and the false clue selects self;
- a corrupted registered-Good Gemcrafter is absent from the Evil pool, so its
  bluff uniformly selects a registered-Evil occurrence; and
- there is no truth-pool draw followed by inversion, no replacement candidate,
  and no authored Good/Evil-count input.

For a post-conditional Evil pool of size `N`, each occurrence again has
conditional probability `1 / N`. Consecutive calls share Unity's global PRNG
state; this target does not reconstruct that state or claim statistical
independence.

## Exact text, references, and RNG chronology

Both successful paths place the exact selected Character reference in a new
one-element list, read that Character's displayed integer `id`, and format:

```text
#X is Good
```

The executable format is exactly `#{0} is Good`: capital `G`, one space on
each side of `is`, no newline, and no terminal punctuation. The same
one-element list is passed to `ActedInfo..ctor`, so `ActedInfo.characters[0]`
is the exact selected object whose ID is printed. There is no independent
reference ordering, stale role reference, or extra actor reference.

The direct successful chronology is:

```text
copy CurrentCharacters
-> registered-alignment filter
-> conditional actor removal
-> one integer Range draw
-> indexed reference and displayed-ID read
-> text formatting
-> ActedInfo construction
```

The role uses neither `UnityEngine.Random.value` nor `System.Random`, performs
no sort, and writes no runtime data. A size-one pool still consumes the Range
call.

## Day dispatch and state lifecycle

`Archivist.Act` and `Archivist.BluffAct` use separately folded shared action
bodies. Each recognizes only `Day == 30`. With a non-null `onActed` callback,
the body invokes exactly one corresponding virtual `GetInfo` or
`GetBluffInfo` call and delivers that exact result once. A null callback and
every other trigger are no-ops and consume no clue RNG.

There is no role-local Start, Night, Reveal, execution, death, reset, picker,
ability-used, status, resistance, achievement, or runtime-data branch. The
constructor adds no custom state. The asset's serialized
`ResetAfterNight` usage category is a framework-facing descriptor; Archivist
declares no reset method or cached state, so this checkpoint does not turn that
asset value into an unobserved Night action. The empty achievement collection
matches the absence of an achievement branch.

## Poet provider parity

Managed `Gossip` constructs `Archivist` as provider entry eleven in its exact
12-provider list. A successful Poet provider draw forwards the original
physical Poet Character as `charRef` to virtual `Archivist.GetInfo` or
`Archivist.GetBluffInfo` and returns the resulting `ActedInfo` unchanged.

Direct and Poet/Gemcrafter therefore share the same live registered-alignment
pools, conditional-removal rule, exact one-reference text, and zero-runtime-
data behavior. Actor handling is relative to the physical Poet seat. On the
normal truth route a registered-Good Poet is removed only when the Good pool
has more than one occurrence. On a corrupted-Good bluff route, Poet is absent
from the registered-Evil pool.

Poet consumes its uniform provider-index `Random.Range` draw before Archivist
consumes its one target-index Range draw. Constructing the provider object adds
no board Character and runs no Start action.

## Small and malformed pools

Neither clue generator emits a sentinel or short result:

- a non-empty pool succeeds and always returns exactly one reference;
- a size-one pool is intentionally not actor-pruned and selects that sole
  occurrence, including self when it is the actor;
- a size-zero pool reaches `Random.Range(0, 0)` and then the indexed-list
  failure path; and
- null singleton/list/selected-reference surfaces follow native failure paths
  rather than producing `#0`, `none`, or an empty `ActedInfo`.

Board size is not itself the gate. Live registered-alignment composition after
upstream identity chronology is decisive.

## Corpus and compatibility implications

The checked archive inventory is:

| Corpus | Files | Decks containing Gemcrafter | Direct Gemcrafter clues | Poet/Gemcrafter clues | Baker original Gemcrafter |
| --- | ---: | ---: | ---: | ---: | ---: |
| `tests/cases_v2` | 426 | 114 | 114 across 97 fixtures | 7 across 7 fixtures | 1 |
| `tests/cases` | 137 | 31 | 34 across 30 fixtures | 5 across 5 fixtures | 0 |

The v2 corpus has 115 apparent-role Gemcrafter records across 98 fixtures, but
`asc83_v7` position 5 is a Rambler-interrupted `shut_up_target: 4` record, not
an Archivist clue. Removing it leaves the 114 direct clues above. The v2
direct-plus-Poet inventory is 121 records across 104 fixtures; the legacy
inventory is 39 across 34 fixtures.

All 160 direct-plus-Poet clue records have blank `info_text`, exactly one
in-range `good_position`, and board sizes from six through ten. Direct payloads
use `good_position`; Poet payloads add `copied_role`. Exactly one record is
self-referential: the unversioned v2 `asc31_v2` Gemcrafter at position 2 names
position 2. That record is compatible with the native sole-registered-Evil
bluff boundary, but its missing provenance does not independently prove the
historical truth route or alignment state.

The archive protects broad one-position compatibility, Poet delegation, and
the existence of a self outcome. Its blank text and missing current-build
marker do not prove exact punctuation, registered-alignment chronology,
truth-versus-bluff origin, probability weights, live-Day timing, or empty-pool
failure. The Rambler-interrupted record must remain a distinct upstream clue
kind, and the Baker-original record proves identity history rather than an
Archivist action.

## Reconstruction boundary and required tests

This section defines a consuming bridge/solver contract; it is not additional
evidence about hidden native state.

- A current direct observation should carry a closed `public_current`
  Gemcrafter marker, exactly one integer `good_position`, the exact matching
  sentence, and exactly the same newest acted reference.
- A current Poet observation should carry the same closed provider-specific
  contract plus `copied_role: Gemcrafter` and current Poet provenance.
- Truth support is the post-conditional registered-Good pool; bluff support is
  the post-conditional registered-Evil pool. Apparent role, current role, and
  corruption must not replace registered-alignment projection.
- The actor is conditionally removed only when the candidate pool begins above
  one. A blanket no-self validator is incorrect.
- The observation is a live Day snapshot. Identity movers, registration
  writers, and removals from `CurrentCharacters` must be replayed before
  checking support.
- Unmarked archive observations should stay on an explicit legacy predicate;
  a textless historical record must not authenticate current native text.

Minimum focused tests for any implementation are:

1. exact direct and Poet text, capitalization, spaces, hash, no newline, no
   terminal punctuation, and exact one-reference parity;
2. truth selection from registered Good and bluff selection from registered
   Evil using registerAs-first projection rather than apparent/current role;
3. direct and Poet actor exclusion when the relevant pool is above one, plus
   sole-pool self support on both raw truth and direct-Evil bluff boundaries;
4. corrupted registered-Good bluff actor absence from the Evil pool;
5. one Range draw for every non-empty pool, including size one, and no
   Random.value, System.Random, sorting, retry, or runtime-data write;
6. zero-pool failure without inventing an empty or `none` observation;
7. Day-only callback dispatch, null-callback no-op, no Start/Night/reset or
   achievement side effect, and Poet's provider draw before the target draw;
8. current-schema rejection of booleans, zero, out-of-range positions,
   additional fields, wrong text, stale references, and variant confusion; and
9. legacy compatibility for all 160 archived clue records while keeping the
   Rambler interruption and Baker identity record outside Archivist evidence.

## Reproduction, typed quality, and coverage

The clean-room checkpoint ran these serialized stages against the saved build
projects:

```powershell
python reverse_engineering/scripts/validate_ghidra_targets.py `
  --targets reverse_engineering/targets/gameplay_role_gemcrafter.json `
  --script-json <private-current-build-script.json>

powershell -ExecutionPolicy Bypass -File `
  reverse_engineering/scripts/invoke_ghidra.ps1 `
  -GameRoot <private-game-root> -Stage export-target `
  -TargetSet gameplay_role_gemcrafter

powershell -ExecutionPolicy Bypass -File `
  reverse_engineering/scripts/invoke_ghidra.ps1 `
  -GameRoot <private-game-root> -Stage build-types

powershell -ExecutionPolicy Bypass -File `
  reverse_engineering/scripts/invoke_ghidra.ps1 `
  -GameRoot <private-game-root> -Stage typed-refresh

powershell -ExecutionPolicy Bypass -File `
  reverse_engineering/scripts/invoke_ghidra.ps1 `
  -GameRoot <private-game-root> -Stage typed-export `
  -TargetSet gameplay_role_gemcrafter

python reverse_engineering/scripts/audit_ghidra_type_quality.py `
  --baseline <private-baseline-export> `
  --typed <private-typed-export> `
  --output reverse_engineering/reports/f530404b0f3f_807de4a83df4_typed_quality_gameplay_role_gemcrafter.json `
  --check

python reverse_engineering/scripts/build_method_coverage.py `
  --build-manifest reverse_engineering/manifests/builds/f530404b0f3f_807de4a83df4.json `
  --type-index reverse_engineering/symbols/f530404b0f3f_807de4a83df4/assembly_csharp_types.json `
  --dump-cs <private-current-build-dump.cs> `
  --script-json <private-current-build-script.json> `
  --game-assembly <private-current-build-GameAssembly.dll> `
  --check
```

Results are 11/11 baseline, 11/11 typed, 11/11 signature application, and
11/11 read-only signature validation. Typed application imports six additional
reachable datatypes, canonicalizes three shared bodies, and the read-only pass
validates 34 ABI parameter storages with zero program mutations.

The typed-quality check passes. Placeholder-parameter tokens fall from 58 to
zero, raw-field-offset accesses from 53 to 14, raw-integer-type tokens from 36
to 8, unresolved-type tokens from 67 to 27, and indirect-call patterns from
four to zero. Signature-parameter-name tokens rise from 21 to 67 and typed
IL2CPP-type tokens from 22 to 100. Decompiler error and warning markers remain
unchanged at three and 15. The nongating raw-pointer-cast count rises from 66
to 71; the report records no policy regression.

Adding Gemcrafter produces a 36-target-set union with 735 memberships, 474
distinct selected FunctionDefinitions, 261 exact-definition overlap
memberships, and 394 unique native RVAs. Five Gemcrafter memberships are exact
definition overlaps: the role constructor plus the four shared semantic
helpers. Its six newly selected definitions add four new native RVAs because
the two Day dispatcher definitions use bodies already present in the union.
Across all 36 read-only target validations, all 735 memberships and 2,151 ABI
parameter storages validate with zero program mutations.

The rebuilt Assembly-CSharp ledger retains its 4,207-method census while adding
terminal evidence for the six previously unclassified Archivist methods and
strengthening the constructor's existing Poet-provider classification.

## Implementation regression gates

The matching reader, bridge, and solver checkpoint passed these focused and
aggregate gates:

- `python -m py_compile memory_reader.py game_loop.py
  tests/test_gemcrafter_native.py tests/test_poet_native.py` completed cleanly.
  The combined Gemcrafter/Poet focused suite passed 47/47 tests, and the full
  Python discovery suite passed 594/594.
- `cargo test -p solver-core current_gemcrafter --lib -- --nocapture` passed
  5/5 focused tests. They cover exact direct/Poet schema and text, malformed
  and mixed provenance, native pool-wide actor removal, sole-pool self support,
  corrupted-Good bluffing, truthful Puppet, all lifecycle seats, Twin/Shaman
  identity movement, one shared anonymous-Wretch assignment, Baker-to-Spy
  chronology, unmarked Rambler/archive compatibility, and unresolved Start
  identities.
- `cargo test --release -p solver-core --lib` passed 359/359. Both
  `cargo check --all-targets` and `cargo build --release` completed cleanly.
- `cargo test --release --test simulation -- --nocapture` passed 31/31 in
  852.55 seconds. Its 426 active-v2 fixtures produced 303 wins, seven expected
  losses, 21 expected constraint issues, zero unexpected constraint failures,
  six known unexpected simulation losses, 15 hidden-Outcast truth gaps, and 74
  fixtures awaiting ordered Twin traces. Those aggregate counts are unchanged
  from the preceding Empress checkpoint.

The memory reader now maps current managed `Archivist` to public Gemcrafter
while retaining the historical `Gambler` alias. Current direct and Poet
ingestion require the newest coherent exact text/reference event and stamp
source-specific provenance; unrelated stale runtime data is ignored because
Archivist writes none. The Rust validator independently reconstructs the full
registered-alignment pool and joins its support to the existing global hidden-
state search. Unmarked archive clues, the Rambler interruption, and Baker's
reported original identity remain on their legacy paths.

## Remaining uncertainty

- The target proves which candidate pool and RNG operation are requested, but
  does not reconstruct Unity's global PRNG state or claim independence between
  calls.
- The reachable clue bodies inline the formatter. This checkpoint does not
  claim that reflection or malformed external code can never invoke the public
  `ConjourInfo` helper independently.
- The archive is textless and unversioned; it cannot prove current exact text,
  registration state, or the truth/bluff route of any individual record.
- Upstream lifecycle code owns `CurrentCharacters`, registerAs and stored-
  alignment writes, corruption/truth routing, Rambler interruption, callback
  history, and the framework interpretation of `ResetAfterNight`. This role
  checkpoint consumes those surfaces but does not re-audit all of them.
- Null and structurally corrupted global-state failure paths are statically
  bounded here and were not forced in a live game.
- This is a build-specific Gemcrafter checkpoint, not evidence that every role
  or the whole game has been fully decompiled.
