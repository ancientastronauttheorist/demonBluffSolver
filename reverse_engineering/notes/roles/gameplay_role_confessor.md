# Confessor current-build native contract

This note records the clean-room native checkpoint for the shipped public and
managed **Confessor** role in build `f530404b0f3f_807de4a83df4`
(`TypeDefIndex 5894`). It closes the role's exact nine-method declaration
boundary and the five shared callees needed to interpret it. It is not a claim
that the whole game is decompiled.

Evidence status is **native-static** for all 14 selected functions and
**metadata** for the serialized asset binding. The baseline and typed Ghidra
exports are private; this note, the target manifest, aggregate typed-quality
report, and normalized coverage records contain no decompiled bodies or private
filesystem paths.

## Public asset binding and managed identity

`Demon Bluff_Data/sharedassets0.assets` contains the shipped Confessor
`CharacterData` at path ID `21614`, absolute file offset `23632336`, size
`4636`, and object SHA-256
`23FD1C1465EFF58AE47A028191BEB328B09B43D856969A3ACE53E47ABDF86E40`.
The containing asset has SHA-256
`E0D239C03FA47EC138F9C2797779E83B65AEF41F30C5AD84D4CA76767A93E967`.
Its serialized identity is:

- public name `Confessor`;
- character ID `Confessor_18741708`;
- managed role type string `Confessor`, at raw object offset `0x11F8`;
- Good Villager (`type == 10`, `startingAlignment == 10`);
- `abilityUsage == 0`, the serialized `Once` category;
- bluffable, not usually disguised, and non-picking;
- an empty bundled-character collection;
- no role achievements, additional statuses, tags, or `canAppearIf` records;
- the exact hint `I can not Lie.` and an empty `ifLies` field; and
- one skin reference, path ID `21646`.

The skin object is at absolute file offset `23753712`, has size `296`, and has
SHA-256
`0FB8AD397B835CF5F13F9634E2D25EB4D1CAA676639E01D81D70C953552CEC80`.
Its exact name is `Confessor_Christmass_Normandia`, its skin ID is
`Confessor_Christmass_Normandia_SKIN_1147`, its artist is `normandia`, and its
unlock category is `UnlockWithAchievement`. That skin metadata does not create
a Confessor role-code achievement hook: the CharacterData achievement list is
empty and none of the nine managed methods has an achievement branch.

The authored public description is exactly:

```text
If I am Evil or Corrupted:
"I am dizzy"
```

The managed `get_Description` body independently returns the older sentence,
without terminal punctuation:

```text
Can not lie, even if I am Evil
```

The normal serialized candidate pool in `Demon Bluff_Data/level0`, path ID
`139347`, contains exactly one file-ID-`2` reference to Confessor path ID
`21614`, at pool-object-local offset `464`. The pool object is at file offset
`17578592`, size `1080`, and has SHA-256
`FB9D821AE0A7E3655BEF4A3DD3E544E85B3109258A48DCF68FF0969ACED8D948`.
The containing `level0` file has SHA-256
`B509AC15904F0926419CCCC1D2E86508C69B0E3AF2235A08FF8E9FC4D170C7F1`.
Confessor has zero references in the 15-entry ordered Start object at path ID
`137026`, file offset `17310672`, size `332`, and object SHA-256
`544328634CD77D551B5864CDC1B643029F3B30BFFC5BB4350DFCF83C66226BB0`.
It is therefore a direct normal-pool role whose setup is the Reveal-time
`Init` hook, not an ordered-`Start` participant.

## Exact callable boundary and shared bodies

Managed `Confessor` declares no fields, exactly nine FunctionDefinitions, and
no Confessor-owned compiler-generated type:

| Managed identity | RVA | Role-local purpose |
| --- | ---: | --- |
| `Confessor.get_Description` | `0x3D6A90` | Older managed description |
| `Confessor.GetInfo` | `0x3D68A0` | Truth clue, null references, and art side effect |
| `Confessor.OnInit` | `0x3D6A50` | Apparent-truth status attempt |
| `Confessor.Act` | `0x3D65D0` | Truth Init and Day dispatch |
| `Confessor.BluffAct` | `0x3D6650` | Bluff Init and Day dispatch |
| `Confessor.GetBluffInfo` | `0x3D6720` | Truth-identical bluff clue |
| `Confessor.ManageArt` | `0x3D6A20` | Declared but uncalled conditional-art helper |
| `Confessor.ConjourInfo` | `0x3D66D0` | Declared but uncalled two-string selector |
| `Confessor..ctor` | `0x3CFFF0` | Fieldless role construction |

The target also pins five semantic callees:

- `ActedInfo..ctor` at `0x35D5D0`;
- `CharacterStatuses.AddStatus` at `0x363AA0`;
- `Character.GetRegisterAlignment` at `0x365030`;
- `Character.ShowAnimatedArt` at `0x368B40`; and
- direct generic `List<Int32Enum>.Contains` at `0xB45070`.

The constructor's native body has 537 managed aliases. The target preserves
the exact Confessor metadata identity and applies the already established
ABI-compatible constructor prototype. No other Confessor declaration shares a
native RVA in this target. Generic callback, Unity-object, animated-art,
string-literal, allocator, and IL2CPP runtime internals stop outside the
role-specific boundary.

## Truth and bluff dizziness

`GetInfo` and `GetBluffInfo` are separate native bodies with identical gameplay
logic. For supplied physical actor `c`, define:

```text
C = c.statuses.statuses directly contains Corrupted (10)
E = c.GetRegisterAlignment() == Evil (20)
S = c.dataRef.role is Spy or a runtime-derived Spy type

dizzy = !S && (C || E)
```

The evaluation order is direct generic status membership, registered-alignment
lookup, then the current real `dataRef.role` runtime-type test. There is no
random draw, retry, pool, sorting step, board access, adjacency, distance,
character-ID read, callback-dependent variant, or runtime-data access.

`GetRegisterAlignment` returns `registerAs.startingAlignment` when the
`registerAs` record is Unity-live; otherwise it returns the Character's stored
alignment field. Confessor therefore uses registered/apparent alignment rather
than unconditionally using the current `dataRef` alignment. It does not inspect
the role name carried by `registerAs`.

The `Spy` exception inspects the actor's current real `dataRef.role`, not its
displayed Confessor role, bluff role, `registerAs` role, identity origin, or
serialized deck entry. A current Spy or derived-Spy record always forces
`dizzy == false`, even when the actor is Corrupted or registers as Evil. A null
role pointer is simply not Spy and still uses `C || E`; a null `dataRef` follows
the native failure path.

Consequently, an ordinary uncorrupted registered-Good Confessor says Good; an
ordinary Corrupted Confessor or registered-Evil actor says dizzy; and a Spy
using Confessor as a bluff says Good. `GetBluffInfo` does not invert, reroll, or
replace this result. Its output support and probabilities are exactly the same
deterministic singleton as `GetInfo` for the same live actor state.

## Exact text, references, and art side effect

The two exact result strings are:

```text
I am Good
I am dizzy
```

Neither has a newline or terminal punctuation. Their string-literal RVAs are
`0x271D960` and `0x271DAF8`, respectively. The managed description literal is
at `0x26D8328`.

Both clue generators allocate a fresh `ActedInfo` and pass the chosen string
plus a null `List<Character>` argument. `ActedInfo..ctor` first creates an empty
list and then overwrites that field with the supplied argument. The final
Confessor result therefore has a **null** `characters` pointer, not an allocated
empty list and not a self-reference. Confessor neither reads nor writes
`runtimeData` and does not cache its result.

Only the dizzy path calls `Character.ShowAnimatedArt`, exactly once and before
result allocation. The Good path has no art call. `ShowAnimatedArt` behaves as
follows:

1. if both `dataRef` and `bluff` are Unity-null, return without changing art;
2. otherwise call `GetCharacterBluffIfAble` and obtain its animated-art sprite;
3. call `GetCharacterBluffIfAble` again and obtain that record's art type; and
4. pass the sprite and art type to `Character.SetupArt`.

The separately audited display helper chooses a live unrevealed bluff only
while its state and flags allow it, otherwise `dataRef`. Thus the art side
effect follows the currently eligible display record, while the dizziness
predicate still uses the current real `dataRef.role` for its Spy exception.
This is a UI mutation only: it does not change clue text, alignment, statuses,
references, runtime data, ability usage, or RNG state.

## Initialization and apparent truth

`OnInit` calls:

```text
charRef.statuses.AddStatus(
    AppearTruthfull /* 25 */,
    sourceRef = charRef,
    targetRef = null)
```

The shipped enum spelling is exactly `AppearTruthfull`. `AddStatus` first
checks the exact status in the resistance list. Resistance makes the complete
attempt a no-op. Otherwise it unique-adds the status when absent and always
replaces the one shared `targetCharacter` pointer with null, including on a
non-resisted duplicate insertion. `sourceRef` is not retained or otherwise
used by this helper.

The previously audited appearance predicate is:

```text
AppearLying ||
(!AppearTruthfull &&
 (Corrupted || (!HealthyBluff && (live bluff || Evil alignment))))
```

`AppearTruthfull` therefore changes perceived truth, including the condition
queried by Judge, but it does not change actual `Character.Act` dispatch.
Corruption or Evil alignment can still send a displayed Confessor through a
lying action path even while the card appears truthful.

Internal `Character.Reveal` always dispatches `Init` after any HealthyBluff
`Start` dispatch. Both a real Confessor action path and a bluff-Confessor action
path virtually call this same `OnInit`, so any currently displayed Confessor
identity attempts the apparent-truth status. Repeated internal Reveals can
repeat the attempt; Confessor declares no local removal or reset path.

## Trigger dispatch and lifecycle

`Confessor.Act` and `Confessor.BluffAct` recognize exactly two trigger values:

- on `ETriggerPhase.Init == 3`, each virtually calls `OnInit(charRef)` and
  returns; this branch does not require `onActed`;
- on `ETriggerPhase.Day == 30`, each checks `Role.onActed`; when non-null,
  `Act` virtually calls `GetInfo` and `BluffAct` virtually calls
  `GetBluffInfo`, then invokes the callback exactly once with that result; and
- a null Day callback and every other trigger are clean no-ops.

There is no Confessor-local `Start`, `AfterRoundStart`, `Night`, `OnReveal`,
`OnExecuted`, `OnDied`, `OnProtected`, or `OnPicked` branch. In particular,
execution does not unlock an achievement, alter the clue, clear
`AppearTruthfull`, restore art, or write runtime data. The role has no fields,
picker, cached Character, use counter, reset callback, or role-specific death
state. The serialized `Once` category is framework-facing asset metadata, not
a native Confessor picker or one-shot role field.

The Day callback is downstream of clue construction. A null callback therefore
suppresses both result construction and the dizzy art side effect. The Init
branch remains active regardless of callback state.

## Dormant declared helpers

`Confessor.ManageArt(charRef, dizzy)` returns immediately when `dizzy` is
false, without dereferencing `charRef`. When true it calls
`Character.ShowAnimatedArt` once; a null Character then follows the native
failure path. `Confessor.ConjourInfo(dizzy)` independently returns `I am Good`
for false and `I am dizzy` for true, with no state, validation, art effect, or
RNG.

A full executable `.text` direct-call scan finds zero relative call or jump
references to either declaration. Each has exactly one ordinary IL2CPP
method-registration pointer, at RVAs `0x26A4C88` for `ManageArt` and
`0x26A4C90` for `ConjourInfo`. The reachable `GetInfo` and `GetBluffInfo` bodies
inline the Boolean, text, and art decisions rather than calling these helpers.
Their bodies are fully understood, but they are not part of the shipped direct
execution graph.

## Poet absence and cross-role inputs

Confessor is absent from the exact twelve-entry managed `Gossip` / public Poet
provider list. That list is, in order: Lover, Scout, Oracle, Bounty Hunter,
Medium, Knitter, Hunter, Enlightened, Empress, Bishop, Gemcrafter, and Bard.
Poet therefore cannot natively select or construct Confessor, and there is no
Confessor Poet-provider RNG chronology, acted-reference parity, or provider
slot. A historical `copied_role: Confessor` Poet payload would not be current
native evidence.

The direct Confessor result nevertheless depends on cross-role state already
stored on its physical actor:

- Pooka, Poisoner, Plague Doctor, or another producer can supply direct
  `Corrupted` status;
- another role can supply a Unity-live `registerAs` record and thereby change
  the registered alignment tested by Confessor;
- the current `dataRef.role` controls the exact Spy exception, including after
  data movement or replacement; and
- real/bluff action dispatch and internal Reveal determine when the displayed
  Confessor's `OnInit` runs.

No identity-origin filter is present. Doppelganger, Shaman, Drunk, Puppet,
ordinary evil bluff, and direct-role cases use the same actor-relative native
predicate once Confessor's method receives the physical Character.

## Malformed and edge surfaces

Ordinary board size is irrelevant: Confessor never reads the board and behaves
the same on one through ten cards. There is no small-pool or small-circle
special case.

The native failure and no-op boundary is exact:

- `GetInfo` and `GetBluffInfo` require a non-null Character, status container,
  inner active-status list, and `dataRef`; missing any of those reaches the
  native failure path;
- a null `dataRef.role` is allowed and behaves as non-Spy;
- `OnInit` requires a Character and status container, then inherits
  `AddStatus`'s list requirements and resistance no-op;
- `ManageArt(false)` is null-safe, while `ManageArt(true)` requires a Character;
- `ShowAnimatedArt` returns when both display records are Unity-null but fails
  if its later eligible-record lookup unexpectedly returns null; and
- Day with a null callback is a clean no-op, while Init does not consult the
  callback.

No path retries, substitutes a default Character, fabricates a reference,
normalizes a missing payload, or consumes RNG.

## Corpus and compatibility implications

The 426 checked-in active-v2 fixtures contain Confessor in 119 deck pools.
They contain 128 apparent-Confessor records across 109 fixtures, including 17
fixtures with multiple apparent Confessors and board sizes six through ten.
Of those records, 127 carry the historical Boolean payload: 56 `dizzy: true`
and 71 `dizzy: false`. `asc35_v2` position 8 is the sole empty payload. Every
one of the 128 `info_text` fields is blank. The active corpus also contains nine
Baker-original Confessor records and one Medium clue naming Confessor, but zero
Poet records with `copied_role: Confessor`.

The 137 legacy fixtures add 58 Confessor deck pools and 64 apparent-Confessor
records across 52 fixtures, including 12 multi-Confessor fixtures. All 64 have
a Boolean payload: 39 true and 25 false; all text fields are blank and board
sizes again span six through ten. The legacy corpus adds three Baker-original
and two Medium-name records, and again has zero Poet/Confessor records.

These 191 parsed Boolean observations protect the established reader/solver
schema, but they do not independently prove current exact text, null
`ActedInfo.characters`, the Spy override, registered-alignment precedence,
art mutation, status-resistance behavior, no-RNG chronology, or trigger
lifecycle. Multiple apparent Confessors are compatible with bluffing, Shaman,
Drunk, and other identity mechanics and must not be interpreted as multiple
physical copies of the direct role.

## Solver and reader boundary

The closed current-build observation boundary is one Boolean field, `dizzy`,
whose exact native support is deterministic for a proposed world:

```text
expected_dizzy =
    current_real_role_is_not_Spy_or_derived &&
    (direct_Corrupted || registered_alignment_is_Evil)
```

The public speech must be exactly `I am dizzy` when true and `I am Good` when
false. It carries no Character references and no runtime data. Reader code
should preserve the null native reference pointer long enough to authenticate
current provenance; the normalized solver payload carries no references and
must not invent self or another witness. Solver logic must retain the Spy
exception and registered-alignment semantics; replacing either with ordinary
real alignment changes native support.

The initial `AppearTruthfull` status is a separate appearance fact, not a
reason to force the clue through truthful dispatch or to ignore Corruption.
Likewise, the animated-art call is an observable presentation side effect, not
evidence of another status or a second clue result.

Focused regression coverage should include:

1. uncorrupted registered-Good -> exact Good string, `dizzy == false`;
2. direct Corrupted -> exact dizzy string and one art call;
3. registered Evil without Corruption -> dizzy;
4. current Spy with Corruption and/or registered Evil -> Good;
5. `registerAs` alignment taking precedence over the stored alignment;
6. `GetInfo`/`GetBluffInfo` parity and zero random draws;
7. final null native reference storage and zero runtime data;
8. Init status insertion, exact-resistance blocking, and duplicate targeting;
9. Init versus Day callback behavior and all unrelated trigger no-ops;
10. no execution/achievement/reset branch and no Poet provider path; and
11. compatibility for all 191 archived Boolean payloads while keeping the one
    empty apparent record and identity-only Baker/Medium records outside exact
    current-clue evidence.

## Reproduction, typed quality, and coverage

The clean-room checkpoint ran these serialized stages against the saved build
projects:

```powershell
python reverse_engineering/scripts/validate_ghidra_targets.py `
  --targets reverse_engineering/targets/gameplay_role_confessor.json `
  --script-json <private-current-build-script.json>

powershell -ExecutionPolicy Bypass -File `
  reverse_engineering/scripts/invoke_ghidra.ps1 `
  -GameRoot <private-game-root> -Stage export-target `
  -TargetSet gameplay_role_confessor

powershell -ExecutionPolicy Bypass -File `
  reverse_engineering/scripts/invoke_ghidra.ps1 `
  -GameRoot <private-game-root> -Stage build-types

powershell -ExecutionPolicy Bypass -File `
  reverse_engineering/scripts/invoke_ghidra.ps1 `
  -GameRoot <private-game-root> -Stage typed-refresh

powershell -ExecutionPolicy Bypass -File `
  reverse_engineering/scripts/invoke_ghidra.ps1 `
  -GameRoot <private-game-root> -Stage typed-export `
  -TargetSet gameplay_role_confessor

python reverse_engineering/scripts/audit_ghidra_type_quality.py `
  --baseline <private-baseline-export> `
  --typed <private-typed-export> `
  --output reverse_engineering/reports/f530404b0f3f_807de4a83df4_typed_quality_gameplay_role_confessor.json `
  --check

python reverse_engineering/scripts/build_method_coverage.py `
  --build-manifest reverse_engineering/manifests/builds/f530404b0f3f_807de4a83df4.json `
  --type-index reverse_engineering/symbols/f530404b0f3f_807de4a83df4/assembly_csharp_types.json `
  --dump-cs <private-current-build-dump.cs> `
  --script-json <private-current-build-script.json> `
  --game-assembly <private-current-build-GameAssembly.dll> `
  --check
```

Target validation reports 14 functions and one shared native body. Baseline
and typed exports each complete 14/14 with zero failures. Typed signature
application and read-only validation each complete 14/14; application imports
three additional reachable datatypes and canonicalizes one shared body, while
read-only validation checks 44 ABI parameter storages with zero program
mutations.

The 38-set type build selects 492 exact FunctionDefinitions, applies 5,830
inheritance rewrites and 6,159 alignment records, and validates a
151,623-datatype archive.

The typed-quality check passes. Placeholder-parameter tokens fall from 113 to
zero, raw-field-offset accesses from 55 to three, raw-integer-type tokens from
62 to two, unresolved-type tokens from 51 to four, indirect-call patterns from
six to zero, and raw-pointer casts from 77 to 57. Signature-parameter-name
tokens rise from 28 to 102 and typed IL2CPP-type tokens from 28 to 124.
Decompiler error and warning markers remain unchanged at four and 15. The
report records no policy regression.

Adding Confessor produces a 38-target-set union with 764 memberships, 492
distinct selected FunctionDefinitions, 272 exact-definition overlap
memberships, and 410 unique native RVAs. Four Confessor memberships are exact
definition overlaps: `OnInit`, `ActedInfo..ctor`,
`CharacterStatuses.AddStatus`, and `Character.GetRegisterAlignment`. The ten
newly selected definitions add nine native RVAs because the new Confessor
constructor metadata identity uses a constructor body already present in the
union.

Across all 38 read-only target validations, all 764 memberships and 2,242 ABI
parameter storages validate with zero program mutations. The rebuilt
Assembly-CSharp ledger retains its 4,207-method census, 3,066 unique native
bodies, and 107 shared-body groups while adding terminal classifications for
the eight previously unclassified Confessor declarations and reusing the
existing native status-system evidence for `OnInit`.

The checked overlay totals are 489 terminal classifications and 251 evidence
records. All nine Confessor declarations are terminal: seven are understood,
the two zero-caller declared helpers are unreachable, and no Confessor method
remains unclassified.

## Implementation regression gates

The matching reader, bridge, and solver checkpoint passed these focused and
aggregate gates:

- `python -m py_compile memory_reader.py game_loop.py
  tests/test_confessor_native.py tests/test_poet_native.py` completed cleanly.
  The combined Confessor/Poet focused suite passed 41/41 tests, and the full
  Python discovery suite passed 628/628.
- `cargo test -p solver-core current_confessor --lib -- --nocapture` passed
  7/7 focused tests. They cover exact schema/text/provenance, native-null
  references at the bridge, Corrupted and registered-alignment results, the
  current-Spy override, Puppet/Drunk/Doppelganger/Wretch worlds, Twin/Shaman
  current data, Baker-to-Spy clearing, proven raw-callback overwrite order,
  global raw/register-as identity joins, current Poet rejection, unresolved
  Start identities, and unmarked archive compatibility.
- `cargo test --release -p solver-core --lib` passed 374/374. Both
  `cargo check --all-targets` and `cargo build --release` completed cleanly.
- `cargo test --release --test simulation -- --nocapture` passed 31/31 in
  829.08 seconds. Its 426 active-v2 fixtures produced 303 wins, seven expected
  losses, 21 expected constraint issues, zero unexpected constraint failures,
  six known unexpected simulation losses, 15 hidden-Outcast truth gaps, and 74
  fixtures awaiting ordered Twin traces. Those aggregate counts are unchanged
  from the preceding Bard checkpoint.

The reader now preserves native null acted references distinctly from an
allocated empty list, authenticates only the newest exact Confessor event, and
leaves hidden status/alignment fields out of public ingestion. Manual entries
canonicalize explicit operator tokens into exact current text, while auto-card
may replace only an empty same-role placeholder. The Rust validator separately
reconstructs current data, registered alignment, raw bluff/register-as labels,
and Baker chronology. When a later raw callback is proven, it must also be
Confessor because any other provider would overwrite the observed event;
merely possible raw-pointer presence remains conservative.

## Remaining uncertainty

- The scenario model does not expose Confessor's resisted `AppearTruthfull`
  insertion. Judge/Rambler appearance logic therefore still assumes the status
  lands on a displayed Confessor; the actual Confessor clue predicate does not
  make that assumption.
- Dynamic animated-art frame timing and generic skin-unlock machinery were not
  live-forced. They do not feed the clue, references, runtime data, or RNG.
- When raw-bluff presence is only `Possible`, the hidden-state model cannot
  distinguish an absent pointer from an unobserved one. Proven pointers are
  constrained exactly and Baker's synchronous clear remains authoritative.
- The archived observations are textless and unversioned. They protect their
  historical Boolean predicate but cannot prove current text, null references,
  callback provenance, or the current-Spy exception.
- This is a build-specific Confessor checkpoint, not evidence that every role
  or the whole game has been fully decompiled.
