# Gameplay role: Bishop

Build: `f530404b0f3f_807de4a83df4`

Evidence status: **metadata** for the public `CharacterData` binding, normal
candidate-pool membership, and Start-list absence; **native-static** for every
method declared by managed `Bishop` and its compiler-generated helper types,
the live truth pools, authored bluff domain, exact clue construction, reference
shape, RNG chronology, Day dispatch, and shared shuffle boundary; and
**behavioral** for the archived direct, Poet, and Baker-original corpus. Native
bodies and decompiler output remain outside the repository.

The checked target set is
[`reverse_engineering/targets/gameplay_role_bishop.json`](../../targets/gameplay_role_bishop.json).
It selects 25 exact managed FunctionDefinitions at 20 distinct target-local
native RVAs. Its read-only baseline and typed exports each complete at 25/25
functions with no failures. Typed application imports 38 newly reachable
datatypes, and post-save ABI validation records 67 parameter storages with zero
program mutations. The rebuilt GDT contains 151,585 datatypes and 455 function
definitions.

The body-free
[`quality report`](../../reports/f530404b0f3f_807de4a83df4_typed_quality_gameplay_role_bishop.json)
passes its regression check. Aggregate unresolved-type tokens fall from 187 to
98, raw field-offset accesses from 251 to 156, raw integer type tokens from 164
to 84, placeholder parameter tokens from 110 to zero, and indirect-call
patterns from six to zero. Both exports retain six decompiler-error and 46
warning markers.

## Public asset binding and managed identity

The shipped `sharedassets0.assets` `CharacterData` at path ID `21613` is named
`Bishop`, has `characterId` `Bishop_58855542`, and binds directly to managed
`Bishop` at TypeDefIndex `5888` in `Assembly-CSharp`. The object is 10,108 bytes
at file offset `23,622,224` and has SHA-256
`EAD498F47258EB9A5398F0E70D99BB180726DC3DAD10B6AE8A35D8943A0E45A4`.
The containing asset has SHA-256
`E0D239C03FA47EC138F9C2797779E83B65AEF41F30C5AD84D4CA76767A93E967`.

Bishop is a Good Villager (`characterType == 10`,
`startingAlignment == 10`). It uses passive Once behavior
(`abilityUsage == 0`), is bluffable, is not usually disguised, and has
`picking == false`. Its authored public description is exactly:

```text
Learn up to 3 characters.
Among them are 1 Villager, 1 Outcast and 1 Evil role if possible.
```

The same asset's exact if-lying hint is:

```text
All characters in my info are Villagers
```

The managed description getter separately returns the older exact text:

```text
Learn 3 players. They each are Outsider, Town and Minion. Can add +1 Outsider?
```

The serialized managed role type string `Bishop` occurs at raw object offset
`0x275C`. The bundled-character, skin-list, achievement, additional-status,
tag, and `canAppearIf` collections are empty. The asset has no authored hint
outside the if-lying field.

The normal serialized `level0` candidate pool at path ID `139347` contains one
reference from file ID `2` to this exact CharacterData at path ID `21613`.
`level0` has SHA-256
`B509AC15904F0926419CCCC1D2E86508C69B0E3AF2235A08FF8E9FC4D170C7F1`.
Bishop is therefore live as a direct Standard/Ascension card independently of
Poet. It is absent from the 15-entry `startGameActOrder` object at path ID
`137026`, and neither action body has a Start branch.

## Audited boundary and shared bodies

The managed declaration boundary contains 17 FunctionDefinitions:

| Managed identity | RVA | Purpose |
| --- | ---: | --- |
| `Bishop.<>c..cctor` | `0x3F07F0` | Allocate the singleton closure holder |
| `Bishop.<>c..ctor` | `0x357920` | Fieldless closure-holder construction |
| `Bishop.<>c.<GetInfo>b__2_0` | `0x3F0790` | Truth ID sort key, `Character.id` |
| `Bishop.<>c.<GetInfo>b__2_1` | `0x392D50` | Truth secondary key, `UnityEngine.Random.value` |
| `Bishop.<>c.<GetBluffInfo>b__5_0` | `0x3F0790` | Bluff ID sort key, `Character.id` |
| `Bishop.<>c.<GetBluffInfo>b__5_1` | `0x392D50` | Bluff secondary key, `UnityEngine.Random.value` |
| `Bishop.<>c__DisplayClass2_0..ctor` | `0x357920` | Truth random-key closure construction |
| `Bishop.<>c__DisplayClass2_0.<GetInfo>b__2` | `0x3F07C0` | Captured parameterless `System.Random.Next()` |
| `Bishop.<>c__DisplayClass5_0..ctor` | `0x357920` | Bluff random-key closure construction |
| `Bishop.<>c__DisplayClass5_0.<GetBluffInfo>b__2` | `0x3F07C0` | Captured parameterless `System.Random.Next()` |
| `Bishop.get_Description` | `0x3D5E80` | Older managed description text |
| `Bishop.GetInfo` | `0x3D5320` | Truth selection, ordering, text, and references |
| `Bishop.Act` | `0x3B09F0` | Day-only truthful callback |
| `Bishop.BluffAct` | `0x3B33E0` | Day-only bluff callback |
| `Bishop.GetBluffInfo` | `0x3D4730` | Villager-only bluff references and fabricated types |
| `Bishop.ConjourInfo` | `0x3D4220` | Exact one-, two-, and three-entry text |
| `Bishop..ctor` | `0x3CFFF0` | Fieldless role and Poet-provider construction |

The target also pins eight semantic callees: `ActedInfo..ctor`,
`Character.GetCharacterData`, the Character overload of
`Characters.FilterCharacterType`, `ListHelper.ShuffleList<object>`, the
parameterless `System.Random` constructor and `Next`, integer
`UnityEngine.Random.Range`, and `UnityEngine.Random.get_value`. Generic LINQ,
delegate, list, enum-to-string, and string-concatenation internals stop outside
the role-specific boundary.

Thirteen target memberships have shared native bodies. The generated empty
constructors share `0x357920`; ID selectors share `0x3F0790`; float selectors
share `0x392D50`; the two captured-`Next` selectors share `0x3F07C0`; Day
dispatchers and the role constructor use their established folded bodies.
Typed application canonicalizes 11 of these shared memberships. The two
`0x3F07C0` Bishop methods have different unused second-argument metadata types
(`Character` versus `ECharacterType`) but the same observable body: read the
captured `System.Random` at closure offset `0x10` and call its parameterless
virtual `Next` slot.

## Truth selection and live type projection

`GetInfo` ignores its `charRef` argument. It synchronously reads the current
board and builds a selected list in this order:

1. Copy `Gameplay.CurrentCharacters`, keep exact projected Outcasts, and if the
   pool is nonempty uniformly select one with integer `Random.Range`.
2. Make a fresh copy of `CurrentCharacters`, keep exact projected Villagers,
   and if nonempty uniformly select one.
3. Make another fresh copy, keep exact projected Minions, and select one if
   nonempty.
4. Only when the Minion pool is empty, make another fresh copy, keep exact
   projected Demons, and unconditionally attempt to select one.

The Evil slot therefore prefers every available Minion over every Demon. With
`M > 0` projected Minion occurrences, each Minion has probability `1/M` and
every Demon has probability zero. Only at `M == 0` does each of `D > 0` Demons
have probability `1/D`. Missing Outcast or Villager pools are skipped; missing
both Minion and Demon pools is not skipped and fails at the empty Demon index.

Both filtering and final type projection use the same exact data rule:

```text
registerAs is not Unity-null ? registerAs : dataRef
```

The `type` field of that returned `CharacterData` is compared or emitted.
Unity-null semantics matter: this is `UnityEngine.Object` equality rather than
only a raw pointer test. Wretch's registered surface can consequently count as
Minion; other registration or current-data changes can move an occurrence
between pools. Corruption, current alignment, hidden identity, dead/executed
state, reveal state, visibility, statuses, and physical origin are not filters.
Every occurrence in `CurrentCharacters`, including the Bishop actor, is
eligible when its projected type matches.

There is no Bishop start snapshot or stored selection. The category copies and
all final projections occur synchronously when the Day clue is generated.
Afterward the resulting `ActedInfo` persists through the ordinary Character
history surface, but Bishop itself writes no runtime data.

## Truth ordering, reference shape, and RNG

After category sampling, truth first constructs one parameterless
`System.Random` and stores it in the truth display closure. It then builds three
related but distinct surfaces:

1. It sorts the selected references by ascending displayed `Character.id`,
   with `UnityEngine.Random.value` installed as a secondary key, and copies the
   resulting IDs into the printed-ID list.
2. It sorts that ID-sorted reference list by one `Next()` key per occurrence
   from the already-constructed first `System.Random`.
3. It calls `ListHelper.ShuffleList<Character>`, which creates a second
   parameterless `System.Random` and performs another stable `OrderBy` using
   one `Next()` key per occurrence.

The final list from step 3 is copied into `ActedInfo.characters`. Truth maps
that same final list through live `GetCharacterData().type`, so its printed
type sequence and returned-reference sequence agree with each other. Neither
sequence is positionally aligned with the ascending ID sequence in the text;
the clue is a multiset statement about the displayed IDs.

The full normal truth RNG chronology is therefore category `Range` draws,
construction of the first parameterless `System.Random`, secondary
`Random.value` keys during ID ordering, `Next()` keys from the first random,
construction of the second random inside `ShuffleList`, and its `Next()` keys.
Parameterless `System.Random.Next()` returns a large discrete integer key, and
stable ordering preserves input order on equal keys. All permutations have
support, but exact permutation probabilities are not asserted to be perfectly
uniform because key collisions have stable-sort bias; separate construction
also does not by itself prove statistical independence between the two random
instances. This ordering bias has no clue-semantic effect because IDs and
categories are interpreted as multisets.

## Bluff selection, authored domain, and RNG

`GetBluffInfo` also ignores `charRef`, but its reference pool and fabricated
type domain come from different state surfaces.

First it makes one copy of `Gameplay.CurrentCharacters`, filters that list to
exact live projected Villagers, and samples without replacement:

1. one Villager by `Random.Range(0, V)`;
2. remove that chosen object, then choose one from the remaining list; and
3. only when `Gameplay.CurrentScript.outs > 0`, remove the second choice and
   choose a third from what remains.

Thus, with ordinary unique board objects, bluff returns exactly two distinct
Villager references when the authored Outcast count is zero and three when it
is positive. An ordered `k`-tuple from `V` Villagers has the ordinary
without-replacement selection probability `1/(V * (V-1) * ... * (V-k+1))`
before later display-only reordering. It requires at least two current
projected Villagers, or at least three when the authored script has an Outcast.
The actor is not excluded and can be sampled.

Separately, the fabricated type list is inserted in this order:

1. `Minion` when `CurrentScript.minion > 0`, otherwise `Demon`;
2. `Outcast` when `CurrentScript.outs > 0`; and
3. `Villager` when `CurrentScript.town > 0`.

The Evil type depends only on whether the authored script has any Minion. It
does not inspect the sampled references, live Minion/Demon pools, or the
authored Demon count. The if-lying asset text is exact for ordinary states:
every returned reference is a projected Villager, while the fabricated list
always includes either Minion or Demon.

Bluff then:

1. constructs one parameterless `System.Random`;
2. sorts sampled IDs ascending with the same `Random.value` secondary key;
3. uses the already-constructed first `System.Random` to random-key-sort the
   fabricated type list;
4. separately calls `ListHelper.ShuffleList` on the ID-sorted reference
   list, creating a second `System.Random`; and
5. returns that separately shuffled reference list.

Thus its RNG chronology is the two or three without-replacement `Range` draws,
construction of the first parameterless `System.Random`, `Random.value` keys
during ID ordering, `Next()` keys from the first random during type ordering,
construction of the second random inside `ShuffleList`, and its `Next()` keys
during reference ordering. The two objects are separately constructed; this
checkpoint does not assert statistical independence between their seeds.

Consequently bluff has no positional ID-to-type or reference-to-type mapping.
The IDs are a sorted displayed set, the types are a separately ordered
fabricated set, and `ActedInfo.characters` is a separately ordered set of the
all-Villager references. Solvers should validate the two multisets and the
native holder/provenance rules, not zip the arrays.

`CurrentScript` is the authored count record, while `CurrentCharacters` and
`GetCharacterData` are live. Runtime transformations can therefore change the
available bluff Villagers without changing whether a third reference or an
Outcast/Villager type token is authored.

## Exact clue text

`ConjourInfo` uses `ECharacterType.ToString()` and has these exact successful
forms, with no terminal punctuation:

```text
#{id} is a {Type}
```

```text
Between
#{id1}, #{id2}
there is:
{Type1} and {Type2}
```

```text
Between
#{id1}, #{id2}, #{id3}
there is:
{Type1}, {Type2} and {Type3}
```

The executable says `Between`, not the asset description's `Among`. IDs are
ascending on shipped boards. Supported emitted enum names are `Villager`,
`Outcast`, `Minion`, and `Demon`.

The helper branches independently on ID count and type count. An ID count of
one directly indexes type zero. Otherwise it begins `Between\n`, inserts IDs
only for counts two or three, adds `\nthere is:\n`, and inserts types only for
counts two or three. Shipped truth and bluff states normally keep both counts
equal, but malformed authored counts can expose the partial header behavior.

## Day dispatch and state lifecycle

`Bishop.Act` uses the folded shared truthful Day body at `0x3B09F0`;
`Bishop.BluffAct` uses the folded shared bluff Day body at `0x3B33E0`. Each:

- recognizes only `ETriggerPhase.Day == 30`;
- requires a non-null `onActed` callback;
- makes one virtual `GetInfo` or `GetBluffInfo` call;
- sends its one returned `ActedInfo` to the callback exactly once; and
- is a no-op for null callbacks and every other trigger.

There is no Start, Night, reset, execution, death, achievement, picker, or
activated-ability branch. The role constructor and helper constructors add no
role state. Repeated direct calls would resample from current state; the normal
Once lifecycle controls ordinary invocation outside these bodies.

## Poet provider parity

Managed `Gossip`, the shipped public Poet implementation, constructs a fresh
ordered provider list containing `Bishop` as one-based entry 10 of 12. Its
truth and bluff paths pass the original Poet Character as `charRef` and forward
the provider's returned `ActedInfo` unchanged.

Bishop ignores that actor parameter and writes no runtime data. Once Bishop is
selected, direct Bishop and Poet/Bishop therefore have identical board pools,
authored-count inputs, text schema, and reference shape. Poet's provider-index
draw occurs before Bishop's category/reference and ordering draws. A Poet actor
can appear among the Villager samples just as a direct Bishop actor can.

## Small and malformed boards

Truth has these exact edge outcomes for ordinary unique occurrences:

- empty board: Outcast/Villager are skipped, then empty Minion falls back to an
  empty Demon pool and fails;
- one projected Minion or Demon: a successful one-reference clue naming that
  one category, even when it is the actor;
- one Villager or Outcast with no projected Evil category: failure;
- Villager plus Demon, or Outcast plus Demon: a successful two-entry clue;
- Minion plus Demon: only the Minion is selected, producing one entry;
- Villager plus Outcast without Minion/Demon: failure; and
- normal available Outcast, Villager, and Minion-or-fallback-Demon pools: three
  entries, one from each.

Bluff always attempts two projected Villager selections and attempts a third
when `CurrentScript.outs > 0`. It therefore fails below those live pool sizes.
If `CurrentScript.town <= 0`, the fabricated type count can be smaller than the
reference/ID count; `ConjourInfo` then follows its independent-count partial
format rather than repairing the mismatch. If `CurrentScript` is null, bluff
fails before domain construction.

No native deduplication exists beyond `List.Remove` after each bluff draw.
Normal `CurrentCharacters` holds unique objects. A malformed list containing
the same Character object more than once could retain another equal occurrence
after a single removal and expose duplicate identity in the returned list.

## Cross-role and solver boundaries

Native facts that downstream clean-room implementations must keep separate:

- truth uses live current/register-as type data at Day, not alignment and not a
  stored game-start type snapshot;
- bluff references use live projected Villagers, while bluff cardinality and
  printed category availability use authored `CurrentScript` counts;
- Wretch or any other `registerAs` mutation can change truth pools and live
  bluff eligibility without rewriting authored counts;
- dead, hidden, executed, corrupted, or transformed occurrences remain
  eligible while they remain in `CurrentCharacters`;
- Shaman-copied Bishop and Poet/Bishop use the same global logic because the
  acting Character is ignored; and
- Bishop creates no runtime-data dependency for Baker, Poet, or later roles.

The pre-checkpoint Rust validator described a game-start snapshot and tried
both pre/post-Chancellor category views. That compatibility logic is a solver
approximation, not the native role contract. The executable samples and
projects live Day state, then the produced `ActedInfo` becomes historical.
Correcting that timestamp, adding current exact-text/provenance enforcement,
and preserving authored-count versus live-reference separation are solver work
outside this native checkpoint.

The existing direct automation extracts legal type names as a multiset. The
current Poet path additionally checks the exact one-, two-, or three-entry text
and compares returned references order-insensitively, which matches the native
ID/reference separation. Archived unversioned fixtures remain compatibility
inputs and must not be retroactively treated as exact current-build evidence.

## Corpus and compatibility implications

The checked corpus inventory is:

| Corpus | Files | Decks containing Bishop | Direct Bishop observations | Poet/Bishop observations | Baker original Bishop |
| --- | ---: | ---: | ---: | ---: | ---: |
| `tests/cases_v2` | 426 | 108 | 114 across 101 fixtures | 16 across 16 fixtures | 3 across 2 fixtures |
| `tests/cases` | 137 | 54 | 54 across 50 fixtures | 2 across 2 fixtures | 0 |

All 186 direct-plus-Poet observations have both `targets` and `types`, legal
category names, equal lengths, no duplicate target positions, blank
`info_text`, and no current native schema/provenance marker. Of those, 177 have
three targets/types and nine have two: six direct plus one Poet in v2, and two
direct in the legacy corpus. Forty direct observations and two Poet observations
include their own actor position, behavior allowed by both native paths.

The archived records protect the historical multiset compatibility path and
self-selection support. Their blank text and absent provenance do not prove the
current exact string, truth-versus-bluff origin, RNG chronology, live Day
timestamp, authored-count split, or small-board failure behavior. The three
Baker-original records prove only that Bishop can occur as Baker's reported
original identity; they do not execute Bishop's selection body.

## Reproduction and coverage

The clean-room checkpoint ran these serialized stages against the saved build
projects:

```powershell
python reverse_engineering/scripts/validate_ghidra_targets.py `
  --targets reverse_engineering/targets/gameplay_role_bishop.json `
  --script-json <private-current-build-script.json>

powershell -ExecutionPolicy Bypass -File `
  reverse_engineering/scripts/invoke_ghidra.ps1 `
  -GameRoot <private-game-root> -Stage export-target `
  -TargetSet gameplay_role_bishop

powershell -ExecutionPolicy Bypass -File `
  reverse_engineering/scripts/invoke_ghidra.ps1 `
  -GameRoot <private-game-root> -Stage build-types

powershell -ExecutionPolicy Bypass -File `
  reverse_engineering/scripts/invoke_ghidra.ps1 `
  -GameRoot <private-game-root> -Stage typed-refresh

powershell -ExecutionPolicy Bypass -File `
  reverse_engineering/scripts/invoke_ghidra.ps1 `
  -GameRoot <private-game-root> -Stage typed-export `
  -TargetSet gameplay_role_bishop

python reverse_engineering/scripts/audit_ghidra_type_quality.py `
  --baseline <private-baseline-export> `
  --typed <private-typed-export> `
  --output reverse_engineering/reports/f530404b0f3f_807de4a83df4_typed_quality_gameplay_role_bishop.json `
  --check

python reverse_engineering/scripts/build_method_coverage.py `
  --build-manifest reverse_engineering/manifests/builds/f530404b0f3f_807de4a83df4.json `
  --type-index reverse_engineering/symbols/f530404b0f3f_807de4a83df4/assembly_csharp_types.json `
  --dump-cs <private-current-build-dump.cs> `
  --script-json <private-current-build-script.json> `
  --game-assembly <private-current-build-GameAssembly.dll> `
  --check
```

Results are 25/25 baseline, 25/25 typed, 25/25 signature application, 25/25
read-only signature validation, 38 newly imported program datatypes, 67 ABI
parameter storages, and zero validation mutations. Adding Bishop produces a
34-target-set union with 705 memberships, 455 distinct selected
FunctionDefinitions, 250 exact-definition overlap memberships, and 382 unique
native RVAs. The 4,207-method Assembly-CSharp coverage ledger retains its
audited census while adding terminal evidence for the 17 declared Bishop
methods and the generic shuffle helper boundary.
