# Bard / Acrobat2 current-build native contract

This note records the clean-room native checkpoint for the shipped public
**Bard** role in build `f530404b0f3f_807de4a83df4`. The public asset is
implemented by managed `Acrobat2` (`TypeDefIndex 5896`). This is a bounded role
checkpoint, not a claim that the whole game is decompiled.

## Public asset binding and managed identity

`Demon Bluff_Data/sharedassets0.assets` contains the shipped Bard
`CharacterData` at path ID `21612`, object offset `23616552`, size `5668`, and
object SHA-256
`9D4C928D00C12AC2D1A37EEA0D810BC55D3296648659FEE3041B047A93F7DBBA`.
The containing asset has SHA-256
`E0D239C03FA47EC138F9C2797779E83B65AEF41F30C5AD84D4CA76767A93E967`.
Its serialized identity is:

- public name `Bard`;
- character ID `Athlete_95133291`;
- managed role type string `Acrobat2`, at raw object offset `0x1600`;
- Good Villager (`type == 10`, `startingAlignment == 10`);
- `abilityUsage == 0`, the serialized `Once` category;
- bluffable, not usually disguised, and non-picking;
- an empty bundled-character collection;
- one skin reference, path ID `21645`, named
  `Bard_Halloween_Normandia`, whose skin object has SHA-256
  `98480488F7ED6DC5FE6FB2CEF44433232700FD1131BCCBF1D7A014F1BA218362`;
- empty achievement, additional-status, hint, and if-lying collections;
- one `Corrupt` tag (`10`); and
- three serialized `canAppearIf` references, in order: public Plague Doctor
  path ID `21606`, Poisoner `21597`, and Pooka `21593`.

The asset's authored public description is exactly:

```text
Learn how far I am from closest Corrupted character.
```

The managed getter independently returns the older wording, without terminal
punctuation and without `closest`:

```text
Learn how far I am from Poisoned character
```

The normal serialized `level0` candidate pool at path ID `139347` contains one
file-ID-`2` reference to this exact path ID, at object-local offset `440`. The
candidate-pool object has SHA-256
`FB9D821AE0A7E3655BEF4A3DD3E544E85B3109258A48DCF68FF0969ACED8D948`;
`level0` has SHA-256
`B509AC15904F0926419CCCC1D2E86508C69B0E3AF2235A08FF8E9FC4D170C7F1`.
Bard is therefore live as a direct Standard/Ascension card independently of
Poet. It has zero references in the 15-entry `startGameActOrder` object at path
ID `137026`, whose object SHA-256 is
`544328634CD77D551B5864CDC1B643029F3B30BFFC5BB4350DFCF83C66226BB0`.

The adjacent managed `Acrobat` (`TypeDefIndex 5895`) is a separate, unbound
older implementation. The current public asset binds `Acrobat2`; its internal
name must not be used to merge the two roles.

## Exact callable boundary and shared bodies

The exact `Acrobat2` declaration boundary contains nine
FunctionDefinitions, no fields, and no `Acrobat2`-owned compiler-generated
type:

| Managed identity | RVA | Purpose |
| --- | ---: | --- |
| `Acrobat2.get_Description` | `0x3D0000` | Managed description text |
| `Acrobat2.GetInfo` | `0x3CFF20` | Truth result and geometric references |
| `Acrobat2.Act` | `0x3B09F0` | Truth Day callback |
| `Acrobat2.BluffAct` | `0x3CFAF0` | Bluff Day callback and execution achievement |
| `Acrobat2.GetBluffInfo` | `0x3CFC90` | Fixed-domain false distance |
| `Acrobat2.GetClosestPoisonedCharacter` | `0x3CFD70` | Circular Corrupted-distance scan |
| `Acrobat2.ConjourInfo` | `0x3CFBE0` | Exact distance sentence |
| `Acrobat2.CheckAchievementsAndUnlockIfAble` | `0x3CFBA0` | Dormant unconditional achievement helper |
| `Acrobat2..ctor` | `0x3CFFF0` | Fieldless role and Poet-provider construction |

The target also pins six semantic callees: `ActedInfo..ctor`, direct status
membership, `Characters.GetCharactersAtRange`, the actor-first circular-list
helper, the remove-one-integer random helper, and integer
`UnityEngine.Random.Range`. Generic list, callback-sink, string-format,
platform-achievement, Unity-object, and PRNG internals stop outside the
role-specific boundary.

Three target memberships have globally shared native bodies. `Acrobat2.Act`
has 17 metadata aliases, the role constructor has 537, and integer Range has
two. Typed application canonicalizes the two role memberships whose
ABI-compatible shared prototypes are already pinned. Exact managed identities
remain distinct even where native code is folded.

Both reachable clue generators call `GetClosestPoisonedCharacter`,
`ConjourInfo`, and `GetCharactersAtRange`; the separately declared formatter
is therefore part of the live direct and Poet paths.

## Circular truth distance

`GetClosestPoisonedCharacter` copies live `Gameplay.CurrentCharacters`, rotates
that copy so the supplied physical actor is first, then removes the first
entry. It scans the remaining circle in both list directions. Every physical
seat increments distance. The first occurrence in each direction whose live
`CharacterStatuses` directly contains `Corrupted` (`10`) is a candidate, and
the method returns the smaller of those first-hit distances.

The scan does not filter by registered or runtime alignment, apparent or
current role, reveal state, visibility, death, execution, identity origin, or
any other status. A dead or hidden Corrupted Character still counts if it
remains in `CurrentCharacters`. The actor is removed before the scan, so its
own Corrupted status is deliberately ignored.

When no remaining Character is Corrupted, both directional scans retain the
zero sentinel and the method returns exactly `0`; it does not return `N - 1`,
`-1`, or a null result. Truth consumes no RNG and caches no board state.

Malformed null/list/status surfaces follow their native failure paths. If the
supplied actor is absent, behavior inherits the separately audited global
actor-first rotation helper rather than gaining a role-local validation or
sentinel.

## Exact reference geometry

After computing distance `d`, both clue paths call
`Characters.GetCharactersAtRange(d, actor)` and pass its list unchanged to
`ActedInfo`. The list describes the two geometric seats at the reported range,
not the set of actual Corrupted Characters:

1. the forward/current-list-direction endpoint at `d`; then
2. the reverse endpoint at `d`.

For positive `d <= N - 1`, both references are appended. On an even board at
`d == N / 2`, the same opposite Character is appended twice. A false distance
greater than the shortest circular distance still returns its two `d`-step
endpoints, which may be the nearer pair in reversed order. `d == 0` and
positive `d > N - 1` return an empty list. A negative distance reaches the
helper's malformed index path rather than being normalized.

Consequently, one or both references may be uncorrupted, and a consuming
reader must preserve duplicate references and exact ordering. The text's
integer and the result's geometry are coupled; there is no separately sampled
target list.

## Exact clue text

`ConjourInfo` has three exact output forms:

```text
There are no Corrupted characters
I am 1 card away from Corrupted character
I am {N} cards away from Corrupted character
```

The first form is used only for `0`, the singular form only for `1`, and every
other supplied integer, including a malformed negative, uses the plural
format. Capitalization and spaces are exact. None contains a newline or
terminal punctuation.

The corresponding string-literal RVAs are `0x26E52F8`, `0x271D8D8`, and
`0x271DD18`. The managed-description literal is at `0x26E0B98`; the achievement
key is at `0x26D05E0`.

## Bluff domain and RNG chronology

`GetBluffInfo` first computes the same actual truth distance `t`. It then calls
`Calculator.RemoveNumberAndGetRandomNumberFromList(t, 0, 4)`. The helper
constructs the exact integer occurrence domain `[0, 4) == {0, 1, 2, 3}` and
removes `t` if present before making one max-exclusive integer Range draw over
the retained list:

| Actual `t` | Uniform false-distance support |
| ---: | --- |
| `0` | `{1, 2, 3}` |
| `1` | `{0, 2, 3}` |
| `2` | `{0, 1, 3}` |
| `3` | `{0, 1, 2}` |
| `>= 4` | `{0, 1, 2, 3}` |

Every retained integer occurrence has probability `1 / 3` when `t` is in the
fixed domain and `1 / 4` otherwise. There is no retry, second draw,
`Random.value`, `System.Random`, board-size clamp, weighting, or runtime-data
write. The generic helper's empty-list fallback is unreachable because Bard
always retains at least three values.

The selected false integer is immediately used for both exact text and range
references. The fixed domain is independent of board geometry, so a small
board may produce a false positive distance beyond `N - 1` and therefore an
empty reference list. A false zero emits the no-Corrupted sentence and empty
references even when another Character really is Corrupted.

## Day dispatch, execution achievement, and lifecycle

`Acrobat2.Act` uses the folded shared truth dispatcher. Only `Day == 30`, with
a non-null `onActed` callback, invokes one virtual `GetInfo` and delivers that
one result once. A null callback and every other trigger are no-ops and consume
no clue RNG.

`Acrobat2.BluffAct` has two branches:

- on `Day == 30`, a non-null callback receives exactly one virtual
  `GetBluffInfo` result; and
- on `OnExecuted == 40`, a non-null supplied Character whose live status list
  does **not** contain `HealthyBluff` (`30`) causes one unlock request for exact
  key `Bard_Halloween_ACHIV_6761`. Presence of that status suppresses the
  unlock. This branch consumes no clue RNG.

Null required objects follow native failure/no-op boundaries rather than
inventing an achievement result. The truth `Act` body has no corresponding
execution branch. Which action body is selected is upstream truth-routing
state; this checkpoint does not reinterpret the branch as a universal Bard
death hook.

The separately declared `CheckAchievementsAndUnlockIfAble` unconditionally
requests the same key, but a read-only executable-reference scan finds zero
direct relative call/jump edges to RVA `0x3CFBA0` and exactly one ordinary
IL2CPP method-registration pointer at RVA `0x26A4D20`. Reachable `BluffAct`
inlines its own HealthyBluff test. The helper is therefore classified
unreachable in the audited executable call graph, while reflection or
malformed external invocation remains outside that claim.

There is no role-local Start, Night, Reveal, death, reset, picker,
ability-used, resistance, runtime-data, or cached-selection branch. The asset's
serialized `Once` usage and empty achievement collection are framework-facing
metadata; neither changes the exact local method boundary. The three
`canAppearIf` references establish authored availability dependencies, not a
role-local runtime dependency or selection rule.

## Poet provider parity

Managed `Gossip` constructs `Acrobat2` as provider entry twelve in its exact
12-provider list. A successful Poet provider draw forwards the original
physical Poet Character as `charRef` to virtual `Acrobat2.GetInfo` or
`Acrobat2.GetBluffInfo` and returns the resulting `ActedInfo` unchanged.

Direct and Poet/Bard therefore share the same circular board scan, exact
distance/text/reference coupling, fixed false domain, and absence of runtime
data. The actor-first rotation and actor-self Corruption exclusion are relative
to the physical Poet seat. Constructing the provider adds no synthetic board
Character and runs no Start action.

Truthful Poet/Bard consumes the provider-index Range draw and no Bard-internal
draw. Bluffing Poet/Bard consumes the provider-index Range draw first, then the
one retained-false-distance Range draw. Consecutive calls share Unity's global
PRNG state; this target does not reconstruct that state or claim statistical
independence.

## Small and malformed boards

For valid actor/list state, the exact geometry includes:

- `N == 1`: the actor is removed, truth reports zero, and every supported Bard
  distance produces no references;
- `N == 2`: distance one references the other Character twice, while distances
  two and three produce no references; and
- `N == 3`: distance one references the two neighbors, distance two references
  the same pair in reverse order, and distance three produces no references.

On larger even boards, the opposite seat is duplicated at half-circle range.
No-Corrupted truth is always distance zero with no references. The false
domain remains `{0,1,2,3}` minus truth even when an element has no geometric
endpoint on that board.

Null globals, a null actor, a missing actor, a null status list, or malformed
negative distance follow shared helper/runtime failure behavior. They do not
produce a repaired clue, an empty `ActedInfo`, or a role-local error sentinel.

## Corpus and compatibility implications

The checked archive inventory is:

| Corpus | Files | Decks containing Bard | Direct Bard clues | Poet/Bard clues | Baker original Bard |
| --- | ---: | ---: | ---: | ---: | ---: |
| `tests/cases_v2` | 426 | 116 | 117 across 106 fixtures | 6 across 6 fixtures | 6 |
| `tests/cases` | 137 | 38 | 39 across 34 fixtures | 1 | 2 |

The v2 direct-plus-Poet inventory is 123 records across 110 fixtures. The
legacy inventory is 40 records across 35 fixtures. All 163 records have blank
`info_text`, use `corruption_distance`, and span boards of size six through
ten. Poet records add `copied_role: Bard`; direct records have no additional
fields. The legacy corpus also contains one apparent Bard with an empty
payload (`asc10_g8`), which is not counted as a clue above.

Historical bridge/solver convention normalized the native zero/no-Corrupted
sentence to `corruption_distance: -1`. Across the 163 archived clues, raw
values are `-1` 45 times, `0` once, `1` 47 times, `2` 26 times, `3` 37 times,
and `4` seven times. The sole raw zero is historical compatibility data, not
evidence that current native `ConjourInfo(0)` prints a numeric distance.

The archive broadly protects scalar distance compatibility and Poet
delegation. Its blank text, absent acted references, mixed zero normalization,
and missing current-build marker do not independently prove exact text,
reference geometry, duplicate opposite-seat references, actual Corruption
state, truth-versus-bluff origin, fixed-domain probabilities, or Day/execution
timing. Baker-original records prove identity history rather than an
`Acrobat2` action.

## Reconstruction boundary and required tests

This section defines a consuming bridge/solver contract; it is not additional
evidence about hidden native state.

- A current direct observation should carry a closed `public_current` Bard
  marker, one scalar corruption distance, the exact corresponding sentence,
  and the exact newest acted-reference list.
- A current Poet observation should carry the same closed provider-specific
  contract plus `copied_role: Bard` and current Poet provenance.
- Native distance zero may be represented internally as the established `-1`
  solver sentinel, but that normalization must occur only after authenticating
  the exact zero sentence and empty references. Native truth remains `0`.
- Truth distance is the minimum circular distance to any other live-list
  Character with direct Corrupted status. The actor's own status is ignored;
  apparent/current role, alignment, death, and visibility are not filters.
- A lying claim is supported exactly by `{0,1,2,3} - {truth}` when truth is in
  that domain, otherwise all four values. Board geometry must not shrink it.
- References are deterministic endpoints of the printed distance, including
  duplicates and empty out-of-geometry false results. They are not witnesses
  that those endpoints are Corrupted.
- Unmarked archive observations should remain on an explicit compatibility
  predicate; textless historical `-1` and `0` records must not authenticate
  current exact text or reference history.

Minimum focused tests for any implementation are:

1. all three exact sentence forms, capitalization, singular/plural, no newline,
   and no terminal punctuation;
2. current native zero authenticated by the exact no-Corrupted sentence and
   empty references, with explicit post-authentication `0 -> -1` normalization
   if the internal schema retains the historical sentinel;
3. truthful circular scans in both directions, actor-self Corruption ignored,
   dead/hidden Corrupted occurrences retained, and no target returning zero;
4. exact forward-then-reverse references, duplicate even-board opposite seats,
   empty zero and oversized references, and small-board cases for `N == 1..3`;
5. every bluff support set and probability above, including `t >= 4`, exactly
   one Range draw, no retry or board clamp, and false zero/oversized geometry;
6. Day-only clue dispatch, null-callback no-op, no Start/Night/reset state, and
   execution achievement only through bluff dispatch when HealthyBluff is
   absent;
7. direct and Poet actor-relative parity, with Poet's provider draw preceding
   any Bard false-distance draw and no nested truth draw;
8. current-schema rejection of booleans, zero-as-serialized-current payload,
   negative values other than the explicit internal sentinel, wrong text,
   stale/misordered references, extra fields, and variant confusion; and
9. compatibility for all 163 archived clue records while keeping the empty
   apparent Bard and eight Baker identity records outside current clue
   evidence.

## Reproduction, typed quality, and coverage

The clean-room checkpoint ran these serialized stages against the saved build
projects:

```powershell
python reverse_engineering/scripts/validate_ghidra_targets.py `
  --targets reverse_engineering/targets/gameplay_role_bard.json `
  --script-json <private-current-build-script.json>

powershell -ExecutionPolicy Bypass -File `
  reverse_engineering/scripts/invoke_ghidra.ps1 `
  -GameRoot <private-game-root> -Stage export-target `
  -TargetSet gameplay_role_bard

powershell -ExecutionPolicy Bypass -File `
  reverse_engineering/scripts/invoke_ghidra.ps1 `
  -GameRoot <private-game-root> -Stage build-types

powershell -ExecutionPolicy Bypass -File `
  reverse_engineering/scripts/invoke_ghidra.ps1 `
  -GameRoot <private-game-root> -Stage typed-refresh

powershell -ExecutionPolicy Bypass -File `
  reverse_engineering/scripts/invoke_ghidra.ps1 `
  -GameRoot <private-game-root> -Stage typed-export `
  -TargetSet gameplay_role_bard

python reverse_engineering/scripts/audit_ghidra_type_quality.py `
  --baseline <private-baseline-export> `
  --typed <private-typed-export> `
  --output reverse_engineering/reports/f530404b0f3f_807de4a83df4_typed_quality_gameplay_role_bard.json `
  --check

python reverse_engineering/scripts/build_method_coverage.py `
  --build-manifest reverse_engineering/manifests/builds/f530404b0f3f_807de4a83df4.json `
  --type-index reverse_engineering/symbols/f530404b0f3f_807de4a83df4/assembly_csharp_types.json `
  --dump-cs <private-current-build-dump.cs> `
  --script-json <private-current-build-script.json> `
  --game-assembly <private-current-build-GameAssembly.dll> `
  --check
```

Results are 15/15 baseline, 15/15 typed, 15/15 signature application, and
15/15 read-only signature validation. The 37-set type build selects 482 exact
FunctionDefinitions, applies 5,830 inheritance rewrites and 6,159 alignment
records, and validates a 151,612-datatype archive. Bard application imports six
additional reachable datatypes, canonicalizes two shared role bodies, and the
read-only pass validates 47 ABI parameter storages with zero program mutations.

The typed-quality check passes. Placeholder-parameter tokens fall from 91 to
zero, raw-field-offset accesses from 47 to 25, raw-integer-type tokens from 40
to 18, unresolved-type tokens from 62 to 18, and indirect-call patterns from
four to zero. Signature-parameter-name tokens rise from 27 to 80 and typed
IL2CPP-type tokens from 28 to 108. Decompiler error and warning markers remain
unchanged at three and 16. The nongating raw-pointer-cast count rises from 58
to 82; the report records no policy regression.

Adding Bard produces a 37-target-set union with 750 memberships, 482 distinct
selected FunctionDefinitions, 268 exact-definition overlap memberships, and
401 unique native RVAs. Seven Bard memberships are exact definition overlaps:
the constructor plus six shared semantic helpers. Its eight newly selected
definitions add seven new native RVAs because `Acrobat2.Act` uses a body
already present in the union.

Across all 37 read-only target validations, all 750 memberships and 2,198 ABI
parameter storages validate with zero program mutations. The rebuilt
Assembly-CSharp ledger retains its 4,207-method census, 3,066 unique native
bodies, and 107 shared-body groups while adding terminal evidence for the
eight previously unclassified `Acrobat2` methods and strengthening the
constructor's existing Poet-provider classification. The checked totals are
481 classifications and 243 evidence records.

## Remaining uncertainty

- The target proves the candidate occurrence domain and requested Unity RNG
  operation, but does not reconstruct global PRNG state or assert independence
  between calls.
- The executable-reference scan bounds ordinary direct call/jump edges to the
  standalone achievement helper; it cannot rule out reflection or malformed
  external invocation.
- Serialized `canAppearIf` references prove authored dependencies but not the
  wider deck-builder's boolean interpretation, which is outside this role
  target.
- The textless archive cannot prove current exact text, acted references,
  Corruption chronology, or the truth/bluff route of any individual record.
- Upstream systems own CurrentCharacters order/membership, Corrupted and
  HealthyBluff status mutation, truth routing, callback history, and framework
  interpretation of `Once`. This checkpoint consumes those surfaces but does
  not re-audit all of them.
- Null and structurally corrupted global-state paths are statically bounded
  here and were not forced in a live game.
- This is a build-specific Bard checkpoint, not evidence that every role or
  the whole game has been fully decompiled.
