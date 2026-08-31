# Gameplay role: Empress

Build: `f530404b0f3f_807de4a83df4`

Evidence status: **metadata** for the public `CharacterData` binding, normal
candidate-pool membership, and Start-list absence; **native-static** for every
method declared by managed `Noble` and its compiler-generated helper type,
registered-alignment selection, exact clue construction and reference order,
RNG chronology, Day and execution dispatch, and the achievement helper; and
**behavioral** for the archived direct, Poet, and Baker-original corpus. Native
bodies and decompiler output remain outside the repository.

The checked target set is
[`reverse_engineering/targets/gameplay_role_empress.json`](../../targets/gameplay_role_empress.json).
It selects 19 exact managed FunctionDefinitions at 17 distinct target-local
native RVAs. Its read-only baseline and typed exports each complete at 19/19
functions with no failures. Typed application imports six newly reachable
datatypes, and post-save ABI validation records 55 parameter storages with zero
program mutations. The rebuilt GDT contains 151,598 datatypes and 468 function
definitions.

The body-free
[`quality report`](../../reports/f530404b0f3f_807de4a83df4_typed_quality_gameplay_role_empress.json)
passes its regression check. Aggregate unresolved-type tokens fall from 102 to
45, raw field-offset accesses from 122 to 92, raw integer type tokens from 82
to 51, placeholder parameter tokens from 80 to zero, and indirect-call
patterns from four to zero. Both exports retain four decompiler-error and 21
warning markers.

## Public asset binding and managed identity

The shipped `sharedassets0.assets` `CharacterData` at path ID `21617` is named
`Empress`, has `characterId` `Empress_13782227`, and binds directly to managed
`Noble` at TypeDefIndex `5883` in `Assembly-CSharp`. The object is 3,092 bytes
at file offset `23,656,960` and has SHA-256
`253187568696C9EB22380CF4CE098F35759C3D82F2EEB7A460138100CE148677`.
The containing asset has SHA-256
`E0D239C03FA47EC138F9C2797779E83B65AEF41F30C5AD84D4CA76767A93E967`.

Empress is a Good Villager (`characterType == 10`,
`startingAlignment == 10`). It uses passive Once behavior
(`abilityUsage == 0`), is bluffable, is not usually disguised, and has
`picking == false`. Its authored public description is exactly:

```text
Learn 3 characters.
Only 1 is Evil
```

The same asset's exact if-lying hint is:

```text
All characters in my info are Good
```

The managed description getter independently returns:

```text
Learn 3 players. Only 1 is Evil
```

The serialized managed role type string `Noble` occurs at raw object offset
`0xBF4`. The bundled-character, skin-list, achievement, additional-status,
tag, and `canAppearIf` collections are empty.

The normal serialized `level0` candidate pool at path ID `139347` contains one
reference from file ID `2` to this exact CharacterData at path ID `21617`.
`level0` has SHA-256
`B509AC15904F0926419CCCC1D2E86508C69B0E3AF2235A08FF8E9FC4D170C7F1`.
Empress is therefore live as a direct Standard/Ascension card independently of
Poet. It is absent from the 15-entry `startGameActOrder` object at path ID
`137026`, and neither action body has a Start branch.

## Audited boundary and shared bodies

The exact managed declaration boundary contains 14 FunctionDefinitions:

| Managed identity | RVA | Purpose |
| --- | ---: | --- |
| `Noble.<>c..cctor` | `0x3F08D0` | Allocate the singleton closure holder |
| `Noble.<>c..ctor` | `0x357920` | Fieldless closure-holder construction |
| `Noble.<>c.<GetInfo>b__2_0` | `0x3F0790` | Truth ID sort key, `Character.id` |
| `Noble.<>c.<GetInfo>b__2_1` | `0x392D50` | Truth secondary key, `UnityEngine.Random.value` |
| `Noble.<>c.<GetBluffInfo>b__5_0` | `0x3F0790` | Bluff ID sort key, `Character.id` |
| `Noble.<>c.<GetBluffInfo>b__5_1` | `0x392D50` | Bluff secondary key, `UnityEngine.Random.value` |
| `Noble.get_Description` | `0x3E6190` | Managed description text |
| `Noble.GetInfo` | `0x3E5B90` | Truth pools, draws, ordering, text, and references |
| `Noble.Act` | `0x3E53A0` | Truth Day callback and execution achievement branch |
| `Noble.BluffAct` | `0x3E5450` | Bluff Day callback and execution achievement branch |
| `Noble.GetBluffInfo` | `0x3E55F0` | All-registered-Good false clue |
| `Noble.ConjourInfo` | `0x3E5540` | Exact three-ID clue construction |
| `Noble.CheckAchievementsAndUnlockIfAble` | `0x3E5500` | Dormant unconditional unlock helper |
| `Noble..ctor` | `0x3CFFF0` | Fieldless role and Poet-provider construction |

The target also pins five semantic callees: `ActedInfo..ctor`, the Character
overload of `Characters.FilterAlignmentCharacters`,
`Character.GetRegisterAlignment`, integer `UnityEngine.Random.Range`, and
`UnityEngine.Random.get_value`. Generic list, LINQ, delegate, string-format,
Unity-object, callback-sink, achievement-platform, and PRNG internals stop
outside the role-specific boundary.

Seven target memberships have globally shared native bodies. The generated
empty constructor shares `0x357920`; both pairs of ID and float selectors share
`0x3F0790` and `0x392D50`; the role constructor and integer Range also have
established aliases. Typed application canonicalizes six memberships whose
shared bodies already have exact ABI-compatible prototypes. Exact managed
identities remain distinct even where native code is folded.

## Registered-alignment projection

Both clue generators synchronously copy `Gameplay.CurrentCharacters` and
filter occurrences by alignment. The filter applies the live Unity-non-null
`registerAs` CharacterData alignment first and otherwise the Character's
stored alignment. It does not query apparent role, current role/dataRef,
runtime faction, corruption, truth appearance, status, reveal state, death,
visibility, or physical origin.

The resulting pools therefore mean **registered Good** and **registered Evil**,
not apparent Villager and apparent Evil. A corrupted Good remains in the Good
pool even though upstream truth routing makes that actor use `GetBluffInfo`.
An Evil displaying Empress remains registered Evil and uses the bluff body.
Identity-changing roles matter only insofar as they change `registerAs`, the
stored fallback alignment, `CurrentCharacters`, or upstream truth routing.

No Empress method caches a pool, result, or historical snapshot. Selection is
from the live Day call's `CurrentCharacters`; the returned `ActedInfo` then
becomes history outside this role boundary.

## Truth selection

Let `G` be the copied registered-Good pool after one `List.Remove(charRef)` and
let `E` be a separate copied registered-Evil pool. `Noble.GetInfo` performs:

1. `Random.Range(0, G)` and appends that Good occurrence;
2. removes the chosen occurrence from the Good pool;
3. `Random.Range(0, G - 1)` and appends a second, distinct Good occurrence;
4. `Random.Range(0, E)` and appends one Evil occurrence; and
5. sorts those three references for output as described below.

Thus every successful normal truth result has exactly two distinct registered-
Good references and one registered-Evil reference. Under the integer Range
contract, each unordered Good pair has equal support and each Evil occurrence
has equal support; for pool sizes `G >= 2` and `E >= 1`, a particular unordered
pair-plus-Evil combination has probability `1 / (C(G,2) * E)` conditional on
the call reaching this selection path.

The actor removal is exact but asymmetric in the raw callable boundary:
`charRef` is removed only from the Good pool, not from the separate Evil pool.
Normal shipped truth routing supplies a registered-Good Empress or Poet actor,
so the actor is excluded. A malformed direct invocation of `GetInfo` with a
registered-Evil `charRef` could select that actor as its Evil reference; this
is not normal real-information dispatch.

## Bluff selection and exact probability domain

`Noble.GetBluffInfo` copies the registered-Good pool, removes `charRef`, and
makes three integer Range draws. After each of the first two draws it removes
the selected occurrence before drawing again. It therefore selects exactly
three distinct registered-Good references and no registered-Evil reference.

For a post-removal pool of size `G >= 3`, every ordered draw sequence has
conditional probability `1 / (G * (G - 1) * (G - 2))`, and every unordered
three-occurrence set has conditional probability `1 / C(G,3)`. The role does
not choose an Evil and replace it, flip the truth result, use authored faction
counts, or inspect which Good roles were selected. Its lie is the literal
all-Good counterexample promised by the asset's if-lying hint.

These probability statements are bounded to the max-exclusive integer Range
contract and sequential without-replacement algorithm. Consecutive calls use
Unity's global PRNG state; this target does not reconstruct that state or claim
statistical independence between calls.

## Exact ordering, text, references, and RNG chronology

After either selection path, Noble applies `OrderBy(Character.id)` followed by
`ThenBy(UnityEngine.Random.value)` and materializes a new three-reference list.
The two generated ID selectors return the displayed integer ID; the two
secondary selectors ignore their Character argument and return one Unity
`Random.value`. Materialization evaluates one secondary key per selected
reference, so a successful clue consumes three value draws even though normal
board IDs are unique and the secondary keys cannot change their order.
Malformed duplicate IDs would be ordered by those float keys; an exact float
tie retains the stable source order.

The sorted IDs are passed to `ConjourInfo` in indices zero, one, and two. The
exact executable format, with no terminal punctuation, is:

```text
One is Evil:
#1, #2 or #3
```

where `1`, `2`, and `3` are the ascending displayed IDs. The same sorted list
object is passed to `ActedInfo..ctor`, so `ActedInfo.characters` contains
exactly three non-null, distinct references in the same order as the printed
IDs. Unlike Bishop, there is no independent type sequence or reference
shuffle.

The direct successful chronology is therefore:

```text
three integer Range draws
-> three Unity Random.value selector draws during materialization
-> text formatting
-> ActedInfo construction
```

Neither path writes `runtimeData`, changes registration, mutates
`CurrentCharacters`, or consumes a `System.Random` stream.

## Day dispatch, execution achievement, and lifecycle

`Noble.Act` and `Noble.BluffAct` each recognize two trigger values:

- at `Day == 30`, a non-null `onActed` callback causes one corresponding
  virtual `GetInfo` or `GetBluffInfo` call and one delivery of that exact
  result; a null callback does nothing and consumes no clue RNG; and
- at `OnExecuted == 40`, the supplied non-null Character's current registered
  alignment is checked. Exact registered Good (`10`) requests achievement key
  `Empress_Halloween_ACHIV_8451`; registered Evil does not.

The execution branch is identical in the truth and bluff action bodies and
contains no clue generation or RNG. A null executed Character follows the
native failure path. All other triggers are no-ops. There is no Start, Night,
Reveal, death, reset, picker, ability-use, status, resistance, or role-local
runtime-data branch.

The separately declared
`Noble.CheckAchievementsAndUnlockIfAble(Character)` ignores its argument and
unconditionally requests the same achievement key. A pinned executable scan
finds zero direct relative call or jump references to RVA `0x3E5500` and one
ordinary IL2CPP method-registration pointer. The reachable action bodies
inline their own registered-Good test and unlock request, so the standalone
helper is classified unreachable through the normal shipped call graph while
reflection or malformed external invocation is not absolutely ruled out.

## Poet provider parity

Managed `Gossip` constructs `Noble` as provider entry nine in its exact
12-provider list. A successful Poet provider draw forwards the original Poet
Character as `charRef` to virtual `Noble.GetInfo` or `Noble.GetBluffInfo` and
returns the resulting `ActedInfo` unchanged.

Consequently direct and Poet/Empress use the same registered-alignment pools,
three-reference text, output ordering, and zero-runtime-data behavior. Actor
removal is relative to the physical Poet seat. Poet consumes one uniform-index
provider `Random.Range` draw before Noble consumes its three selection Range
draws and three value keys. No synthetic Empress Character is added to the
board and constructing the provider object runs no Start action.

## Small and malformed pools

Neither clue generator guards the required pool sizes or emits a sentinel:

- normal truth requires at least two other registered-Good occurrences and at
  least one registered-Evil occurrence;
- bluff requires at least three other registered-Good occurrences; and
- a deficient pool reaches `Random.Range(0, 0)` or a later indexed-list access
  and follows the native failure path rather than returning a short clue.

For a normal registered-Good actor, both paths therefore need at least four
physical board occurrences, though alignment composition rather than total
card count is decisive. Selection without replacement prevents duplicate
references on every successful normal path. Extra dead, executed, hidden, or
unrevealed occurrences remain eligible whenever the global lifecycle retains
them in `CurrentCharacters`; Noble adds no local state filter.

## Corpus and compatibility implications

The checked corpus inventory is:

| Corpus | Files | Decks containing Empress | Direct Empress observations | Poet/Empress observations | Baker original Empress |
| --- | ---: | ---: | ---: | ---: | ---: |
| `tests/cases_v2` | 426 | 121 | 124 across 109 fixtures | 9 across 8 fixtures | 4 across 4 fixtures |
| `tests/cases` | 137 | 42 | 51 across 39 fixtures | 2 across 2 fixtures | 1 across 1 fixture |

Across the 186 direct-plus-Poet records, all target arrays are ascending,
contain no duplicate or actor position, and have blank `info_text`. Direct
payloads contain only `targets`; Poet payloads contain only `copied_role` plus
`targets`. None has current native provenance. Board sizes range from six
through ten.

Of those records, 185 contain three targets. The one exception is the
textless, unversioned Poet/Empress observation in `asc65_v4`, which contains
only two. That archived shape is incompatible with the current native callable
boundary and must remain a legacy compatibility record rather than evidence
for a current two-target branch. Baker-original records prove only that
Empress can appear as Baker identity history; they do not execute Noble's clue
body.

The archive protects broad historical one-Evil compatibility, actor exclusion,
Poet delegation, and ascending target order. Its blank text and missing marker
do not independently prove current exact punctuation, register-as chronology,
truth-versus-bluff origin, probability weights, live-Day timing, achievement
behavior, or insufficient-pool failure.

## Reconstruction boundary and required tests

This section defines a consuming solver/bridge contract; it is not additional
evidence about hidden native state.

- A current direct observation should carry a closed `public_current` Empress
  provenance marker, exactly three ascending distinct in-range targets, the
  exact matching sentence, and exactly the same three chronological acted
  references.
- A current Poet observation should carry the same closed provider-specific
  contract plus exact `copied_role: Empress` and current Poet provenance.
- Truth requires exactly two registered-Good and one registered-Evil target;
  bluff requires three registered-Good targets. Apparent role, corruption, and
  current role must not replace registered-alignment projection.
- The observation is a live Day snapshot. Identity movers and registration
  writers must be solved in lifecycle order before validating the target set.
- Unmarked archived observations should stay on an explicit legacy predicate;
  especially, `asc65_v4` must not make two targets legal for a current marker.

Minimum focused tests for any implementation are:

1. exact direct and Poet text, newline, commas, `or`, hashes, no terminal
   punctuation, and exact reference-to-printed-ID order;
2. truth with two Good plus one Evil and bluff with three Good, using
   registerAs-first projection rather than apparent/current role;
3. actor exclusion for direct and Poet, including a corrupted registered-Good
   actor routed to bluff;
4. rejection of duplicate, unsorted, out-of-range, two-target, four-target,
   extra-field, stale-reference, and whitespace-normalized current payloads;
5. no-Evil truth, fewer-than-two-Good truth, and fewer-than-three-Good bluff
   failure boundaries without inventing a sentinel;
6. Day-only clue RNG, three Range then three value draws, and no `System.Random`
   or runtime-data write;
7. OnExecuted registered-Good achievement success, registered-Evil no-op,
   identical truth/bluff behavior, and no clue RNG; and
8. preserved legacy compatibility for the 186 archived direct/Poet records,
   with the lone two-target record never promoted to current provenance.

## Reproduction and coverage

The clean-room checkpoint ran these serialized stages against the saved build
projects:

```powershell
python reverse_engineering/scripts/validate_ghidra_targets.py `
  --targets reverse_engineering/targets/gameplay_role_empress.json `
  --script-json <private-current-build-script.json>

powershell -ExecutionPolicy Bypass -File `
  reverse_engineering/scripts/invoke_ghidra.ps1 `
  -GameRoot <private-game-root> -Stage export-target `
  -TargetSet gameplay_role_empress

powershell -ExecutionPolicy Bypass -File `
  reverse_engineering/scripts/invoke_ghidra.ps1 `
  -GameRoot <private-game-root> -Stage build-types

powershell -ExecutionPolicy Bypass -File `
  reverse_engineering/scripts/invoke_ghidra.ps1 `
  -GameRoot <private-game-root> -Stage typed-refresh

powershell -ExecutionPolicy Bypass -File `
  reverse_engineering/scripts/invoke_ghidra.ps1 `
  -GameRoot <private-game-root> -Stage typed-export `
  -TargetSet gameplay_role_empress

python reverse_engineering/scripts/audit_ghidra_type_quality.py `
  --baseline <private-baseline-export> `
  --typed <private-typed-export> `
  --output reverse_engineering/reports/f530404b0f3f_807de4a83df4_typed_quality_gameplay_role_empress.json `
  --check

python reverse_engineering/scripts/build_method_coverage.py `
  --build-manifest reverse_engineering/manifests/builds/f530404b0f3f_807de4a83df4.json `
  --type-index reverse_engineering/symbols/f530404b0f3f_807de4a83df4/assembly_csharp_types.json `
  --dump-cs <private-current-build-dump.cs> `
  --script-json <private-current-build-script.json> `
  --game-assembly <private-current-build-GameAssembly.dll> `
  --check
```

Results are 19/19 baseline, 19/19 typed, 19/19 signature application, 19/19
read-only signature validation, six newly imported program datatypes, 55 ABI
parameter storages, and zero validation mutations. Adding Empress produces a
35-target-set union with 724 memberships, 468 distinct selected
FunctionDefinitions, 256 exact-definition overlap memberships, and 390 unique
native RVAs. Six Empress memberships are exact-definition overlaps; its new
declaration boundary contributes eight previously unselected native RVAs.

The 4,207-method Assembly-CSharp ledger retains its exact census while adding
terminal evidence for the 13 previously unclassified Noble/closure methods and
strengthening the already classified Noble constructor.

## Remaining uncertainty

- The role target proves which registered-alignment pools and RNG calls are
  requested, but it does not reconstruct Unity's global PRNG state or make
  consecutive draws statistically independent.
- The standalone achievement helper has no normal direct executable caller;
  reflection or malformed external invocation is not absolutely ruled out.
- Archived observations are textless and unversioned. They cannot prove the
  current exact clue or the truth/bluff route of any individual record.
- Upstream lifecycle code owns `CurrentCharacters`, registerAs writes, truth
  routing, callback history, and achievement-platform persistence. This role
  checkpoint consumes those surfaces but does not re-audit all of them.
- This is a build-specific Empress checkpoint, not evidence that every role or
  the whole game has been fully decompiled.
