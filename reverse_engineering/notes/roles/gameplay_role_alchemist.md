# Alchemist native gameplay checkpoint

This note freezes the shipped public **Alchemist** boundary for build
`f530404b0f3f_807de4a83df4`. Public Alchemist binds managed `Alchemist`
(`TypeDefIndex 5873`), with `AlchemistRuntimeData` (`5495`) and
`AlchemistLoc` (`5678`) as its two owned support types.

The checkpoint is clean-room and build-specific. The checked-in target,
body-free typed-quality report, and normalized coverage records contain no
decompiled function bodies or private filesystem paths. Short exact strings
are retained only where they are required to reconstruct the public rule.

## Build and evidence identity

- `GameAssembly.dll` SHA-256:
  `F530404B0F3F28479A7CE21D5738C4E36C2A0A03E1B5520092975B4150D819EC`
- `global-metadata.dat` SHA-256:
  `807DE4A83DF41DEA29F31E98308131DBA93008B340746B8D6BE85CA5EA373713`
- `sharedassets0.assets` SHA-256:
  `E0D239C03FA47EC138F9C2797779E83B65AEF41F30C5AD84D4CA76767A93E967`
- `level0` SHA-256:
  `B509AC15904F0926419CCCC1D2E86508C69B0E3AF2235A08FF8E9FC4D170C7F1`

The deterministic 37-function target is
[`gameplay_role_alchemist.json`](../../targets/gameplay_role_alchemist.json).
Its aggregate body-free type-quality record is
[`f530404b0f3f_807de4a83df4_typed_quality_gameplay_role_alchemist.json`](../../reports/f530404b0f3f_807de4a83df4_typed_quality_gameplay_role_alchemist.json).

## Public asset binding and serialized placement

The public CharacterData object is `sharedassets0.assets` path ID `21609`, at
file offset `23599528`, with serialized size `3648` and object SHA-256
`A69F297F30136870821AD040AE16ED34890CA7B3ECF3049E15C6D8D5DB78818A`.
It has these exact identity and rule fields:

- object name `Alchemist`, character ID `Alchemist_94446803`, managed role
  `Alchemist` (`TypeDefIndex 5873`);
- Good Villager (`characterType = 10`, `startingAlignment = 10`);
- `abilityUsage = Once` (`0`), `bluffable = true`,
  `usuallyDisguised = false`, and `picking = false`;
- roguelike point value `10`, multiplier `1`, income `0`;
- no bundled characters, additional statuses, or achievements;
- one Corrupt tag (`10`); and
- appearance conditions Plague Doctor (`21606`), Poisoner (`21597`), and
  Pooka (`21593`).

The exact authored public description is:

```text
Villagers within [Range 2] from me are Cured from Corruption.
Learn how many Corruptions were around me.

I can't be Corrupted.
```

The exact hint is:

```text
Range 2:
2 Villagers to the left of me and 2 Villagers to the right of me are affected.
```

`ifLies` and notes are empty. The flavor text is `“He doesn’t know what’s in
the vial. But it seems to work.”`

The one skin reference is path ID `21642`, file offset `23752560`, serialized
size `296`, object SHA-256
`F8A347D46810BA2EBCCDB5EA3C4AEE41AC592720AE233422EB1E424035A15D8A`.
It names `Alchemist_Christmass_Normandia` /
`Alchemist_Christmass_Normandia_SKIN_0203`, credits `normandia`, and uses
`UnlockWithAchievement`.

The normal candidate-pool object is `level0` path ID `139347`, file offset
`17578592`, size `1080`, SHA-256
`FB9D821AE0A7E3655BEF4A3DD3E544E85B3109258A48DCF68FF0969ACED8D948`.
It references Alchemist once, at object-local offset `404` (path ID at `408`).

The ordered-Start object is `level0` path ID `137026`, file offset `17310672`,
size `332`, SHA-256
`544328634CD77D551B5864CDC1B643029F3B30BFFC5BB4350DFCF83C66226BB0`.
Its exact path-ID order is:

```text
21594, 21593, 21597, 21605, 21602, 21595, 21599, 21590,
21606, 21600, 21609, 21634, 21598, 21607, 21591
```

Alchemist is zero-based slot `10`: after Shaman and before Bounty Hunter.

## Exact owned callable boundary and shared bodies

The three owned types declare exactly 13 methods:

| Managed definition | RVA | Role in the boundary |
|---|---:|---|
| `AlchemistRuntimeData..ctor(int)` | `0x357700` | stores `cures` at `+0x10` |
| `AlchemistLoc.LocPL(List<object>)` | `0x395810` | alternate singular/plural formatter |
| `AlchemistLoc..ctor()` | `0x357920` | fieldless localization constructor |
| `Alchemist.get_Description()` | `0x3B0630` | stale managed description |
| `Alchemist.OnInit(Character)` | `0x3B05F0` | Corrupted resistance |
| `Alchemist.GetInfo(Character)` | `0x3B02A0` | truthful Day record |
| `Alchemist.CurePoisons(Character)` | `0x3B0040` | snapshot, count, and cure loop |
| `Alchemist.GetPoisonedCharactersAroundMe(Character)` | `0x3B03B0` | live Start scan |
| `Alchemist.Act(trigger, Character)` | `0x3AFDD0` | truthful trigger dispatch |
| `Alchemist.BluffAct(trigger, Character)` | `0x3AFE90` | lying trigger dispatch |
| `Alchemist.GetBluffInfo(Character)` | `0x3B0170` | false Day count and record |
| `Alchemist.ConjourInfo(int)` | `0x3AFF90` | exact English count text |
| `Alchemist..ctor()` | `0x357920` | fresh role-instance construction |

The managed `Alchemist` role has one instance field, `int corruptions` at
`+0x48`. `AlchemistRuntimeData` has one `int cures` field at `+0x10`.
`AlchemistLoc` has no fields.

The managed description getter retains the stale exact sentence
`2 characters to the left nad right of me are cured from Poison. Learn how
many Poisoning I cured.` Its string literal is at `0x270A270`. It does not
override the shipped public asset text or the executable rule.

`AlchemistLoc.LocPL` expects a boxed integer at `args[0]`. For exactly one it
returns `Uleczyłem {0}\nCorruption`; otherwise it returns
`Uleczyłem {0}\nCorruptions`. Those mixed-language strings are the exact
current literals. Invalid or null arguments follow ordinary managed failure
paths and are not a gameplay fallback.

The target also includes 24 minimum shared helpers: character setup, no-reset
identity replacement, delayed-Reveal role cloning, truth routing, role callback
scheduling and completion, Night refresh, status and resistance mutation, cure
authorization, circular-range and actor-first list construction, `ActedInfo`,
the false-number helper, integer `Random.Range`, and Gossip construction for
the Poet exclusion proof. In particular, `Character.InitWithNoReset`
(`0x365720`) and `Character.<DelayReveal>d__84.MoveNext` (`0x3756B0`) anchor the
identity-mover and fresh-clone behavior used below.

Six target memberships use already-shared native bodies while preserving
their exact managed prototypes. The three fieldless constructors resolve to
`0x357920`; `AlchemistRuntimeData..ctor`,
`Role.CheckIfCanRemoveStatus`, and integer `Random.Range` also resolve to
corpus-shared implementations. A shared native body is not shared managed
state.

## Trigger routing and ordered Start chronology

`Alchemist.Act` handles only these relevant triggers:

- `Init` (`3`): call the virtual `OnInit` hook;
- `Start` (`5`): call `CurePoisons` unless the physical Character has
  `BrokenAbility` (`35`); and
- `Day` (`30`): when `onActed` is non-null, construct truthful info and invoke
  the callback.

`Alchemist.BluffAct` differs:

- it has no `Init` branch;
- at `Start`, `WorkingAbility` (`38`) permits a cure; with or without that
  cure, the handled branch then replaces `charRef.runtimeData` with a fresh
  `AlchemistRuntimeData(0)`; and
- at `Day`, a non-null callback receives `GetBluffInfo`.

All other triggers are role-local no-ops. There is no Alchemist death,
execution, achievement, picker, cancel, or target-click branch.

`Characters.ManageCharacters` initializes every physical Character before
any configured Start action. It then walks the serialized roles in order.
Alchemist, Poisoner, and Plague Doctor are explicit multi-occurrence
exceptions: every exact matching physical card acts instead of stopping after
the first. Duplicate Alchemists act synchronously from highest displayed ID
to lowest.

Thus Plague Doctor's earlier corruption is visible when the first Alchemist
scans. A later Alchemist scans the live status state left by earlier
Alchemists, rather than a board-wide immutable snapshot.

## Per-character cloned role state

Every physical Character initialization clones the CharacterData role into a
live role object. Two cards backed by the same serialized Alchemist asset do
**not** execute through the asset's one role template object. Each physical
Alchemist owns a distinct live `Alchemist` clone and therefore a distinct
`corruptions` field, initially zero.

Within one live clone, `CurePoisons` increments that instance field and does
not locally reset it. Reinitializing or replacing the Character constructs a
fresh role clone with its own zero counter. The counter is not static, not in
Gameplay, not in CharacterData, not stored in the asset template as a mutable
run accumulator, and not shared across duplicate Alchemists.

Duplicate Alchemists consequently have only one cross-card dependency:
sequential status visibility. For example, if the highest-ID Alchemist cures
the only nearby Corrupted status, the next Alchemist may count zero because
its later live scan no longer finds that status. It never inherits the first
Alchemist's numeric count. Each later Day clue reports that physical clone's
own Start snapshot/attempt count.

`AlchemistRuntimeData.cures` is separate per-Character runtime storage, but
neither truthful `GetInfo` nor bluff `GetBluffInfo` reads it. Truthful Start
does not create Alchemist runtime data; bluff Start overwrites it with zero.
It is therefore not an alternative shared or per-round accumulator.

### State-ownership contradiction audit

The target, typed-quality report, this note, and all Alchemist coverage claims
were checked together after the clone correction. Every numeric accumulation
claim is scoped to one live cloned `Alchemist` instance. No current claim
attributes a run counter to the serialized role template, CharacterData,
Gameplay/global state, runtime data, or the set of duplicate Alchemists.
Mentions of duplicate sequencing mean only that later clones observe live
status removals made by earlier clones.

## Corruption immunity and cure authorization

`OnInit` unique-adds resistance to exact Corrupted status (`10`) on the
physical Character. `CharacterStatuses.AddStatus` checks the physical
resistance list first. A resisted insertion has no status or target-pointer
side effect. A non-resisted insertion unique-adds the status and updates the
status container's shared target Character even if the status was already
present; the source does not affect Alchemist's cure rule.

`CheckIfCanCurePoisonAndCure` consults that shared target Character when it is
Unity-live. It calls the target's current real `dataRef.role` virtual
`CheckIfCanRemoveStatus(Corrupted)`. This is not a `registerAs`, raw-bluff,
runtime CharacterData, apparent role, or alignment projection.

Base `Role.CheckIfCanRemoveStatus` permits removal. `Drunk` vetoes only
Corrupted removal. A null or destroyed stored target bypasses the virtual
veto. Once permitted, the helper removes Corrupted and returns true even if a
prior overlapping attempt already removed it. Alchemist ignores the returned
Boolean.

This makes Drunk a countable but non-curable occurrence. Ordinary Wretch,
Spy, Doppelganger, identity registration, dead state, and hidden state do not
create a cure veto; only the current real role virtual matters.

## Start scan, overlap, and small-board geometry

`GetPoisonedCharactersAroundMe` rotates `Gameplay.CurrentCharacters` so the
physical actor is first, removes that actor, and inspects the remaining list
in current-list/forward order. If the post-self list is indexed `0..m-1`, it
checks:

1. index `0`, then index `1`, when present;
2. index `m-1`, then index `m-2`, while stopping before index `0`.

It retains an occurrence exactly when that Character's raw active status list
currently contains Corrupted. It does not filter self by role identity beyond
removing the exact actor object, and it does not filter dead, killed, hidden,
unrevealed, type, alignment, `registerAs`, resistance, or cure eligibility.
It does not deduplicate overlap.

For a well-formed board of `N` physical cards, the selected occurrence shape
is:

| `N` | inspected neighbor occurrences |
|---:|---|
| 1 | none |
| 2 | the sole neighbor once |
| 3 | first neighbor once, other neighbor twice |
| 4 | first neighbor once, opposite neighbor twice, last neighbor once |
| 5+ | first two and last two, all distinct |

Only occurrences active-Corrupted at snapshot construction enter the returned
list. On three- and four-card boards, the overlapping opposite occurrence can
therefore be present twice.

## Truth count is pre-cure attempt multiplicity

`CurePoisons` snapshots the complete returned occurrence list before making
any cure call. For every list occurrence, in order, it:

1. increments this live Alchemist clone's `corruptions` field; then
2. calls the occurrence's `CheckIfCanCurePoisonAndCure` and ignores its result.

The truthful number is therefore the number of selected Corrupted
**occurrences before this Alchemist's cure loop**, including scan overlap. It
is not a post-cure count, successful-removal count, distinct-character count,
or global accumulation. The second visit to a small-board overlap still
counts after the first visit removed the status. A Drunk veto still counts.

Because every duplicate Alchemist owns a separate cloned role field, a later
physical Alchemist starts its own counter at zero and builds a new live
snapshot after earlier cures. There is no numeric aggregation across cards.

## Exact truthful text and ActedInfo references

`GetInfo` reads only the caller live role clone's `corruptions` field. Its
exact `ConjourInfo` mapping is:

| Count | Exact text |
|---:|---|
| 0 | `NO one was Corrupted around me` |
| 1 | `There was\n1 Corruption\naround me` |
| any other integer | `There were\nN Corruptions\naround me` |

The underlying templates are `There was\n{0} Corruption\naround me` and
`There were\n{0} Corruptions\naround me`. There is no terminal punctuation.
Their string-literal RVAs are `0x26EFCB0`, `0x26E63F8`, and `0x26E6728` for
zero, singular, and plural respectively.

The fresh `ActedInfo` reference list is geometric, not the Start snapshot and
not the set actually cured. It appends
`GetCharactersAtRange(2, actor)`, then
`GetCharactersAtRange(1, actor)`.

For positive distance `d <= N-1`, the shared range helper appends the forward
seat at distance `d` and then the reverse seat at distance `d`. Distance zero
or `d > N-1` is empty. It preserves coincident occurrences:

| `N` | truthful/bluff `ActedInfo.characters` order |
|---:|---|
| 1 | empty allocated list |
| 2 | sole neighbor twice (range 2 empty, range 1 forward/reverse) |
| 3 | farther neighbor, nearer neighbor, nearer neighbor, farther neighbor |
| 4 | opposite seat twice, then the two adjacent seats |
| 5+ | distance-2 forward, distance-2 reverse, distance-1 forward, distance-1 reverse |

IDs are not stored in place of references, references are not sorted, and
dead/hidden/registered identity does not change the geometry.

## Bluff domain, exclusion, probability, and RNG

`GetBluffInfo` first reruns the same live active-Corrupted occurrence scan at
Day. Let its current list count be `L`; this may include small-board overlap.
It calls
`Calculator.RemoveNumberAndGetRandomNumberFromList(L, 0, 2)`.

The helper constructs ordered integer candidates `[0, 1]`, removes one exact
`L` only when present, and performs exactly one integer
`UnityEngine.Random.Range(0, retainedCount)` call. Hence:

| Live `L` | retained candidate(s) | emitted false support |
|---:|---|---|
| 0 | `[1]` | `1` deterministically |
| 1 | `[0]` | `0` deterministically |
| 2 or more | `[0, 1]` | uniform `0` or `1`, probability `1/2` each |

Even the one-candidate cases invoke `Range(0, 1)`. The empty-retained-list
fallback is unreachable for this `[0,1]` construction. There is no clamp to
the four-neighbor maximum, no retry, no role weighting, no status-removal
side effect, and no other Alchemist-local random draw.

Immediately after that one draw, bluff uses the same exact text mapping and
the same fresh geometric reference ordering as truth. Truth consumes zero
role-local RNG.

## Result callback, use, history, and reset behavior

At Day, a non-null role callback schedules the resulting `ActedInfo` through
the shared delayed-display path. For a nonempty description, the completion
path runs the global `onAboutToAct` mutation hook, appends the resulting record
to `actedInfos` chronologically, decrements `pickableUses`, updates the global
event/display surfaces, and stores the final visible text in `savedAct`.

Alchemist is authored `Once`, not `ResetAfterNight`, and has no picker.
`Character.RefreshCharacter` therefore does not restore its use at Night and
does not clear prior `actedInfos`. A fresh `Character.Init` clears active
statuses, runtime data, and history, resets the use count, and installs a
fresh live role clone; physical resistances are preserved by that generic
initialization path.

When a physical Character has both a real and raw role, outer dispatch is
real first and raw second. If both Alchemist paths emit, two records append in
that order and each completed callback decrements the shared physical use, so
the ordinary `1` can become `0` and then `-1`. The later raw record wins
`savedAct`. A later result is not merged into the earlier count or references.

The global `onAboutToAct` hook can replace an imminent truthful clue, notably
the current Rambler shut-up rule. That mutation occurs after Alchemist has
computed its own count and references but before history append; it does not
rewind the cure or the per-clone counter.

## Identity movers and cross-role state

Alchemist's scan and acted references use physical board geometry and raw
active status only. They never inspect `registerAs`, Wretch/Spy registration,
apparent identity, alignment, or runtime CharacterData. Cure authorization
uses the current real `dataRef.role`, not registration or raw bluff data.
Truth-versus-bluff dispatch itself remains owned by the generic Character
truth router.

Important current-build interactions follow from the separation between
physical Character state and cloned role state:

- Universal Init installs immunity before ordered Start. Chancellor then moves
  or replaces CharacterData while physical resistances stay on their original
  cards. An Alchemist identity can move onto a card without immunity, while a
  former Alchemist card can retain immunity after losing that identity.
- Shaman runs before the ordered Alchemist slot. `InitWithNoReset` preserves
  destination status, resistance, and runtime data but installs a fresh copied
  role clone with a zero counter, then immediately invokes copied Start.
  Truthful copied Alchemist can cure; a lying copy reaches bluff Start, which
  cures only with WorkingAbility and overwrites runtime data with zero.
- A Puppet/raw copied Alchemist is created after universal Init. BrokenAbility
  blocks truthful raw Alchemist cure at Puppet Start, and the copied role did
  not receive the original universal Alchemist `OnInit`.
- A clean Doppelganger copied as Alchemist can run a late copied Start before
  its internal Reveal Init. A corrupted Doppelganger omits that copied Start;
  lying copied Init reaches `BluffAct`, which has no immunity branch.
- Baker, Twin, Shaman, and other CharacterData replacement paths create or
  select role objects at their own timing. Alchemist never reads Baker runtime
  data, and any newly cloned Alchemist counter starts at zero.
- Wretch and Spy registration do not alter range, count, references, or cure
  permission. Ordinary Doppelganger likewise supplies no special cure veto.
  A current real Drunk role is the one shipped role-level Corrupted-removal
  exception.

These interactions do not turn the per-instance field into a template or
global accumulator. Duplicate Alchemist Start numeric/cure dependency flows
only through live status removal. Identity movers can separately preserve
physical resistances and runtime data, but neither surface transfers a cloned
Alchemist role counter.

## Poet absence

Managed `Gossip` / public Poet constructs an exact twelve-role provider list:

```text
Lover, Scout, Oracle, Bounty Hunter, Medium, Knitter,
Hunter, Enlightened, Empress, Bishop, Gemcrafter, Bard
```

Their TypeDefIndices are `5863, 5854, 5893, 5871, 5881, 5855, 5891, 5890,
5883, 5888, 5884, 5896`. `Alchemist` (`5873`) is absent. There is no Poet
Alchemist slot, provider draw, delegated actor path, reference rewrite,
runtime-data parity, or provider-specific RNG chronology in this build.

## Corpus and compatibility implications

The 426 active-v2 fixtures contain Alchemist in 118 deck pools and contain
121 apparent Alchemist records across 106 files. Of those records:

- 120 have a count and one is empty (`asc27_v2`);
- 115 use legacy `cured_count`, while five use current `corrupted_count`;
- normalized values are `0` in 46 records, `1` in 34, and `2` in 40;
- all 121 `info_text` fields are blank;
- board sizes span six through ten; and
- 14 fixtures have multiple apparent Alchemists, including three in
  `asc36_v6` and two in each of the other 13.

The current wording is witnessed in notes for `asc82_v2`, `asc82_v3`,
`asc83_v4`, `asc84_v1`, and `asc84_v7`. V2 also contains five
Baker-original-Alchemist observations and three Medium-name-Alchemist
observations. It contains no Poet/Alchemist record.

The 137 legacy fixtures add 39 Alchemist deck pools and 42 apparent records
across 35 files. All 42 are complete legacy `cured_count` payloads: 18 zero,
13 one, and 11 two. All text is blank; five files have duplicate apparent
Alchemists. They add three Medium-name observations and no Baker-original or
Poet/Alchemist record.

Combined, the archive has 163 apparent records, 162 with a normalized count:
64 zero, 47 one, and 51 two. It does not preserve exact native references,
truth/bluff provenance, Start snapshots, cure success, clone identity, RNG,
or current clue text. In particular, duplicate apparent cards in a fixture do
not establish shared numeric state and must not be interpreted as an
aggregate counter.

## Reconstruction and regression requirements

A current-build reconstruction should pin at least:

- exact asset binding, Villager alignment, Once/no-picker authoring, ordered
  Start slot, and immunity initialization;
- a separate cloned live role and zero counter per physical Alchemist,
  including duplicate cards with sequential status visibility but no numeric
  aggregation;
- pre-cure occurrence counting, overlap preservation, Drunk veto counting,
  and ignored cure return values;
- Start scan eligibility and the complete one- through four-card geometry;
- exact zero/singular/plural strings, newlines, case, and no punctuation;
- geometric distance-2-then-distance-1 references, including coincident seats
  and empty allocated versus null distinctions;
- live-Day bluff rescanning, exact support for `L=0`, `L=1`, and `L>=2`, and
  one integer draw with no truth RNG;
- truthful and bluff Start gates, runtime-data overwrite, Broken/Working
  statuses, real-then-raw callback order, use decrement, history, and Night
  non-reset;
- physical resistance versus moved CharacterData for Chancellor, Shaman,
  Puppet, Doppelganger, Baker, Wretch, and Spy surfaces; and
- explicit Poet absence.

## Reproduction, typed quality, and coverage

The clean-room checkpoint ran these serialized stages against the saved build
projects:

```powershell
python reverse_engineering/scripts/validate_ghidra_targets.py `
  --targets reverse_engineering/targets/gameplay_role_alchemist.json `
  --script-json <private-current-build-script.json>

powershell -ExecutionPolicy Bypass -File `
  reverse_engineering/scripts/invoke_ghidra.ps1 `
  -GameRoot <private-game-root> -Stage export-target `
  -TargetSet gameplay_role_alchemist

powershell -ExecutionPolicy Bypass -File `
  reverse_engineering/scripts/invoke_ghidra.ps1 `
  -GameRoot <private-game-root> -Stage build-types

powershell -ExecutionPolicy Bypass -File `
  reverse_engineering/scripts/invoke_ghidra.ps1 `
  -GameRoot <private-game-root> -Stage typed-refresh

powershell -ExecutionPolicy Bypass -File `
  reverse_engineering/scripts/invoke_ghidra.ps1 `
  -GameRoot <private-game-root> -Stage typed-export `
  -TargetSet gameplay_role_alchemist

python reverse_engineering/scripts/audit_ghidra_type_quality.py `
  --baseline <private-baseline-export> `
  --typed <private-typed-export> `
  --output reverse_engineering/reports/f530404b0f3f_807de4a83df4_typed_quality_gameplay_role_alchemist.json `
  --check

python reverse_engineering/scripts/build_method_coverage.py `
  --build-manifest reverse_engineering/manifests/builds/f530404b0f3f_807de4a83df4.json `
  --type-index reverse_engineering/symbols/f530404b0f3f_807de4a83df4/assembly_csharp_types.json `
  --dump-cs <private-current-build-dump.cs> `
  --script-json <private-current-build-script.json> `
  --game-assembly <private-current-build-GameAssembly.dll> `
  --check
```

Target validation reports 37 functions and six target memberships backed by
shared native bodies. Baseline and typed exports each complete 37/37 with zero
failures. Typed signature application and read-only validation each complete
37/37. This revised application imports zero additional reachable datatypes and
canonicalizes five unique shared-body prototypes; read-only validation checks
116 ABI parameter storages with zero program mutations.

The typed-quality check passes. Placeholder-parameter tokens fall from 387 to
zero, raw-field-offset accesses from 420 to 68, raw-integer-type tokens from
275 to 61, unresolved-type tokens from 258 to 45, indirect-call patterns from
16 to one, and raw-pointer casts from 514 to 403. Signature-parameter-name
tokens rise from 70 to 332 and typed IL2CPP-type tokens from 66 to 464.
Decompiler errors remain two; warnings rise from 47 to 48, but the report's
policy metrics record no regression.

Adding Alchemist produces a 41-target-set union with 869 memberships, 536
distinct selected FunctionDefinitions, 333 exact-definition overlap
memberships, and 439 unique native RVAs. Alchemist contributes 37
memberships: 29 are exact-definition overlaps and eight are newly selected
definitions. Those eight definitions add five new native RVAs because the
three new constructor definitions use native bodies already in the union.
Across all 41 read-only target validations, all 869 memberships and 2,549
membership-level ABI parameter storages validate with zero program mutations.

The rebuilt Assembly-CSharp ledger retains 4,207 method definitions, 3,066
unique native bodies, and 107 shared-body groups. This slice adds terminal
classifications for the eight previously unclassified owned definitions and
eight Alchemist evidence records. The checked overlay totals are 529
classification records and 275 evidence records.

## Residual uncertainty and solver boundaries

- The target proves the specified PRNG call, candidate support, and call
  chronology, not Unity's hidden PRNG state or independence from other roles.
- Native geometry assumes a well-formed `CurrentCharacters` list containing
  the actor once. Malformed duplicate object entries and null global
  singletons follow native failure or multiplicity paths not forced in play.
- The asset claims Villagers in range, but the executable status scan does not
  filter Character type. The executable behavior is authoritative for this
  build.
- The archive has no exact text or references and cannot distinguish truth,
  bluff, successful cure, overlap, or per-clone provenance.
- Upstream roles own CharacterData movement, raw/register-as pointers,
  corruption insertion, truth routing, and physical resistance retention.
  This checkpoint audits Alchemist's exact reads and writes at those surfaces,
  not every producer's complete behavior.
- Animated skin behavior and generic unlock machinery do not feed count,
  text, references, status removal, runtime data, RNG, or lifecycle and were
  not dynamically forced.
- This is a build-specific Alchemist checkpoint, not evidence that every role
  or the whole game has been fully decompiled.
