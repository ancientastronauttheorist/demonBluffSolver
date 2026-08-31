# Gameplay role: Knitter

Build: `f530404b0f3f_807de4a83df4`

Evidence status: **metadata** for the public `CharacterData` binding and normal
roster membership, and **native-static** for every method declared by managed
`Knitter`, circular registered-Evil pair counting, exact truth and bluff result
construction, Day dispatch, and integer false-value selection. The retained
Poet-provider path and the archived fixture corpus provide additional
behavioral compatibility evidence. Native bodies and decompiler output remain
outside the repository.

The checked target set is
[`reverse_engineering/targets/gameplay_role_knitter.json`](../../targets/gameplay_role_knitter.json).
It selects 12 managed FunctionDefinitions at 12 distinct target-local native
RVAs. Its read-only baseline and typed exports each complete at 12/12 functions
with no failures. Post-save ABI validation records 36 parameter storages and
six imported datatypes; the rebuilt GDT contains 151,551 datatypes.

The body-free
[`quality report`](../../reports/f530404b0f3f_807de4a83df4_typed_quality_gameplay_role_knitter.json)
passes its regression check: unresolved-type tokens fall from 55 to nine, raw
field-offset accesses from 43 to 26, raw integer type tokens from 30 to 20,
placeholder parameter tokens from 67 to zero, and indirect-call patterns from
four to zero. Both exports retain three decompiler-error and 13 warning
markers.

## Public asset binding and managed identity

The shipped `sharedassets0.assets` `CharacterData` at path ID `21625` is named
`Knitter`, has `characterId` `Knitter_32352172`, and binds to managed
`Knitter` at TypeDefIndex `5855` in `Assembly-CSharp`. The object is 5,900
bytes at file offset `23,689,736` and has SHA-256
`D42FD202F887A090409C9C3D3C9FEEAF62600B2F154BB15E9D6CDB5688DEF875`.
The containing asset has SHA-256
`E0D239C03FA47EC138F9C2797779E83B65AEF41F30C5AD84D4CA76767A93E967`.

Knitter is a Good Villager (`characterType == 10`,
`startingAlignment == 10`). It uses passive Once behavior
(`abilityUsage == 0`), is bluffable, is not usually disguised, and has
`picking == false`. Its authored public description is:

```text
Learn how many Evils are adjacent to each other
```

The managed description getter separately returns:

```text
You start knowing how many pairs of evil players there are
```

The asset's bundled-character, skin, achievement, additional-status, tag,
`canAppearIf`, hint, and if-lies collections are all empty.

The normal serialized level object `level0` at path ID `139347`, SHA-256
`B509AC15904F0926419CCCC1D2E86508C69B0E3AF2235A08FF8E9FC4D170C7F1`,
contains one normal candidate-pool reference to this exact CharacterData at
path ID `21625`. Knitter is therefore live as a direct Standard/Ascension card,
independently of Poet. It is absent from the 15-entry
`startGameActOrder`, and its managed action bodies contain no Start branch.

## Audited boundary and shared bodies

| Group | Methods | Observable purpose |
| --- | ---: | --- |
| `Knitter` | 8 | Description, circular count, truth/bluff clue construction, Day dispatch, and construction |
| `ActedInfo` | 1 | Exact text with a null/zero-reference result list |
| `Character` | 1 | Register-as-first alignment projection |
| `Calculator` | 1 | Authored integer-domain construction, removal, and random selection |
| `UnityEngine.Random` | 1 | One integer-index draw over the retained domain |

All 12 target memberships are:

| Managed identity | RVA | Boundary purpose |
| --- | ---: | --- |
| `Knitter.get_Description` | `0x3BCC60` | Managed description text |
| `Knitter.ConjourInfo` | `0x3BC7D0` | Exact count-to-text construction |
| `Knitter.GetInfo` | `0x3BC960` | Truth count and zero-reference result construction |
| `Knitter.Act` | `0x3B09F0` | Day-only truthful callback |
| `Knitter.BluffAct` | `0x3B33E0` | Day-only bluff callback |
| `Knitter.GetBluffInfo` | `0x3BC880` | False-count selection and zero-reference result construction |
| `Knitter.GetPairCount` | `0x3BC9E0` | Circular registered-Evil adjacency count |
| `Knitter..ctor` | `0x357920` | Fieldless role construction and Poet-provider identity |
| `ActedInfo..ctor` | `0x35D5D0` | Exact text and null Character-reference-list storage |
| `Character.GetRegisterAlignment` | `0x365030` | Register-as-first alignment projection |
| `Calculator.RemoveNumberAndGetRandomNumberFromList` | `0x396490` | Half-open domain, exact-value removal, and one retained-index draw |
| `UnityEngine.Random.Range(Int32, Int32)` | `0x1C86600` | Inclusive-lower, exclusive-upper index selection |

`Knitter.Act` and `Knitter.BluffAct` use broad Day action bodies already
present under other managed role identities. `Knitter..ctor` uses the broad
fieldless-role constructor body, and the integer `Random.Range` overload is
canonicalized separately from its float sibling. Applied ABI-compatible
prototypes do not change the exact managed identities selected by the target.
The typed summary canonicalizes three shared native bodies.

The target intentionally stops at the role-specific callable boundary and its
semantic helpers. Generic list operations, string formatting, Unity
object-liveness helpers, and the global gameplay callback sink are shared
engine surfaces rather than additional Knitter memberships.

## Circular registered-Evil pair count

Each `Knitter.GetPairCount` call copies the complete physical
`Gameplay.CurrentCharacters` list, appends the copy's original first element,
then examines each consecutive pair. An edge contributes exactly one when both
endpoint occurrences report registered alignment Evil through:

```text
live registerAs != null ? registerAs.startingAlignment : Character.alignment
```

The operation preserves physical list order and occurrence multiplicity. It
does not filter the actor, corruption, current/displayed role, death, reveal,
visibility, status, or physical origin. In particular, this is registered
alignment rather than merely runtime faction: a runtime-Good Wretch can
register Evil, while a runtime-Evil Spy can register Good.

For a board list `C[0..N)`, the successful count is therefore:

```text
sum(i = 0..N-1, registeredEvil(C[i]) && registeredEvil(C[(i + 1) mod N]))
```

The construction has exact small-board consequences:

- `N == 0`: the native path attempts to obtain element zero before appending
  it, so it fails rather than returning a count or sentinel;
- `N == 1`: the single occurrence is compared with itself and contributes one
  when it registers Evil;
- `N == 2`: the two physical neighbors are compared in both directions, so an
  all-registered-Evil pair contributes two; and
- `N >= 3`: every directed step around the stored circle contributes once,
  which is the ordinary circular undirected-edge count for distinct physical
  occurrences.

The maximum successful count is `N`, including on the singleton and two-card
special cases. Removing dead or executed characters from this geometry would
be a reconstruction bug: they remain physical entries in
`CurrentCharacters`.

This projection is evaluated afresh at each clue; it is not a Start-time
snapshot. Cross-role chronology therefore matters. A stable hidden Spy can be
a Baker conversion target while its cached Villager `registerAs` makes it
registered Good. The converting Baker's synchronous `InitWithNoReset` changes
the Spy body's current data to Baker and clears its raw bluff immediately while
preserving the stale Good register-as. The copied role's delayed Reveal later
clears that register-as, after which the unchanged runtime-Evil body registers
Evil. The observable state therefore has three ordered phases:

1. before conversion: current data remains Spy and the cached register-as is
   Good;
2. converted, pending reset: current data is Baker and the raw bluff is null,
   but the stale register-as is still Good; and
3. reset: current data is Baker, the raw bluff remains null, and registered
   alignment falls back to runtime Evil.

Two truthful Knitter observations on opposite sides of the delayed reset can
consequently report different pair counts without any physical seat moving.

The converted stable Spy can also retain its earlier `bluffRole` pointer while
the Baker overwrite clears the raw bluff. Native Day routing can first produce
the saved real Baker result and then let that stale bluff role replace the
visible speech bubble. A final visible Baker sentence is therefore not, by
itself, a complete timestamp for the register-as reset; verified reveal order
and the monotonic delayed-Reveal boundary remain the authoritative chronology.

## Truth result construction

`Knitter.GetInfo` ignores its supplied `charRef`, calls `GetPairCount` once,
formats that exact count through `ConjourInfo`, and constructs one `ActedInfo`.
It performs no candidate selection and consumes no RNG. The result contains no
Character references: the constructor receives a null reference-list surface,
which is observed by the bridge as exactly zero references.

The exact strings are:

| Count | Exact text |
| ---: | --- |
| `0` | `Evils are not adjacent to eachother` |
| `1` | `There is only 1 pair of Evil` |
| `N >= 2` | `There are N pairs of Evil` |

These strings contain no newline or terminal punctuation. `eachother` is the
shipped joined spelling. Case, spacing, singular/plural wording, and canonical
decimal formatting are observable contract, not presentation normalization.

## Bluff domain and RNG behavior

`Knitter.GetBluffInfo` first computes the actual circular pair count. It then
reads the sizes of the current authored Demon and Minion lists:

```text
S = CurrentScript.demon.Count + CurrentScript.minion.Count
M = max(S, 2)
```

It calls `Calculator.RemoveNumberAndGetRandomNumberFromList(actual, 0, M)`.
That helper constructs the integer occurrence domain `[0, M)`, removes the
actual count if it occurs in that domain, and makes exactly one integer
`Random.Range(0, retainedCount)` draw to select a retained value. Selection is
uniform over the retained integer occurrences.

Consequently:

- `S == 0` or `S == 1` still uses domain `{0, 1}`;
- when `0 <= actual < M`, the actual value is absent and every other value in
  `[0, M)` remains once;
- when `actual >= M`, no list entry is removed and every value in `[0, M)`
  remains once; and
- because `M >= 2`, the retained domain is never empty and the selected count
  always differs from the actual count.

The selected value is formatted by the same exact text helper as truth and is
stored in an `ActedInfo` with zero references. The supplied `charRef` does not
affect the count, domain, removal, result text, or references. Native bluff
behavior is therefore stronger than a generic `claimed != actual` predicate:
the false claim must also belong to `[0, max(S, 2))`.

## Day dispatch

`Knitter.Act` recognizes only `ETriggerPhase.Day == 30`. With that trigger and
a live inherited acted-result callback, it calls virtual `GetInfo(charRef)`
once and invokes the captured callback once with that exact result. Other
triggers, or a missing callback, do not generate a clue and consume no RNG.

`Knitter.BluffAct` has the symmetric Day-only behavior through virtual
`GetBluffInfo(charRef)`. A successful bluff dispatch therefore performs the
one RNG draw described above; other triggers and missing callbacks do not.
Neither action body has Start, Executed, death, reset, or achievement-helper
behavior. Generic Character truth routing decides which action body receives
the trigger; Knitter does not recompute lying state inside either method.

## Poet provider and roster reachability

Managed Poet (`Gossip`) retains Knitter at slot 6 in its exact one-based
12-provider order:

```text
Lover, Scout, Oracle, Bounty Hunter, Medium, Knitter,
Hunter, Enlightened, Empress, Bishop, Gemcrafter, Bard
```

A successful Poet provider draw forwards the physical Poet Character as
`charRef` to virtual `Knitter.GetInfo` or `Knitter.GetBluffInfo` and returns the
resulting `ActedInfo` unchanged. Knitter ignores that reference, so direct and
Poet clues have identical circular geometry, text, bluff domain, and zero
result references.

The direct public Knitter asset is independently present in the normal roster,
so direct Knitter and Poet/Knitter are both shipped reachable clue surfaces.
Constructing the provider object does not add another board occurrence and
does not run a Start action.

## Current bridge and Scenario limitations

Current direct Knitter observations use exact provenance:

```json
{
  "evil_pairs": 2,
  "knitter_variant": "public_current"
}
```

Poet/Knitter uses the same count plus provider provenance:

```json
{
  "copied_role": "Knitter",
  "evil_pairs": 2,
  "poet_variant": "public_current"
}
```

The exact sentence remains in `info_text`, outside `info_parsed`. Direct and
Poet live ingestion require an in-board actor, a canonical integer count in
`0..=n_cards`, one exact full-string match, the newest coherent acted event,
and exactly zero references. Case, whitespace, punctuation, pluralization,
missing text, leading-zero numeric forms, stale events, extra references,
malformed provenance, and extra payload keys fail closed. Direct public entry
also requires canonical apparent role `Knitter`; Poet requires canonical
`Poet` plus copied role `Knitter`. Unity object-name normalization accepts the
exact shipped `Knitter` identity and numeric-suffixed instance names.

Unmarked observations remain on the compatibility path. Builders without an
explicit current marker continue to emit the historical textless schema, so
the checked-in corpus is not silently reinterpreted as build-pinned evidence.

The current Rust validator reconstructs native circular counts from registered
alignment supports. It models known runtime Evil bodies, Wretch's Evil
register-as identity, Spy's Good register-as identity, current-data mover
traces, singleton self-edges, and two-card double edges. When an unexposed
natural Wretch can occupy more than one ordinary-Good position, it enumerates
the legal placements and carries required/forbidden placement supports.

One scenario-wide consistency search now joins all marker-gated observations
from the seven audited passive-provider families: Lover, Scout, Oracle, Hunter,
Medium, Knitter, and Bounty Hunter (currently reachable through Poet). They
must share one compatible anonymous-Wretch placement, Wretch register-as
label, represented register-as identity, Medium raw-bluff identity, and one
complete Baker/Spy timeline. This prevents two individually valid clues from
choosing contradictory hidden worlds.

For an exact current Baker history in which a stable physical Spy finishes as
Baker, the solver records both the synchronous conversion event and one
monotonic delayed-Reveal clear boundary in verified reveal order. Supports for
all seven provider families project current data, raw bluff, Spy register-as,
and registered alignment at that provider observation's phase. They therefore
agree on Spy data before conversion, Baker data plus cleared raw bluff and
stale Good register-as while reset is pending, and Baker data plus runtime-Evil
alignment after reset. Every support carried into the global consistency
search must use the same complete three-phase timeline.

Truthful current observations require the claimed count to equal one supported
native count. Lying observations require a different supported count and the
claim to fall in the reconstructed authored domain. The solver derives that
domain from the HUD objective count minus a represented generated Puppet:

```text
approximate S = state.n_evil - has_generated_puppet
```

This is an explicit representation boundary. `GameState` does not retain the
native `CurrentScript.demon` and `CurrentScript.minion` lists, so unusual states
whose authored list sizes cannot be recovered from that projection may remain
conservative or fail closed. Deck role pools are deliberately not substituted:
they can include undealt Evil identities and are not native current-list
counts.

Other remaining Scenario limits include the absence of a general first-class
per-position `registerAs` CharacterData pointer and complete ordered identity
history for every mover/generator composition. The stable-Spy Baker conversion
is now represented across all seven audited provider families through one
shared three-phase timeline. Known Wretch, Spy, Twin, and Shaman surfaces are
also represented, but a future runtime trace should make arbitrary register-as
identity and current authored Demon/Minion counts explicit for every
observation type.

## Corpus and compatibility implications

A deterministic scan of the 426 checked-in `tests/cases_v2` fixtures finds:

- Knitter appears in a deck pool in 108 fixtures;
- 114 direct apparent Knitter observations occur, and 12 Poet/Knitter
  observations occur, for 126 observations across 106 fixtures;
- direct payloads use exactly `evil_pairs`, while Poet payloads use exactly
  `copied_role` plus `evil_pairs`;
- all 126 observations are textless, unversioned, reference-free, in-board,
  and carry an integer count from zero through three that does not exceed the
  fixture board size;
- the count distribution is 46 zeroes, 53 ones, 25 twos, and two threes;
- no source is night-killed at the observation surface;
- 51 sources are later executed: 43 known true-Evil, seven known
  Good-corrupted, and one other known Good; 49 are direct and two are Poet;
- two direct true-Evil sources are also Slayer targets (`asc54_v2` position 3
  and `asc58_v4` position 4);
- one direct true-Evil source is not recorded as executed (`asc29_v6`
  position 6); and
- every direct observation has Knitter in its deck pool, while 10 of the 12
  Poet/Knitter observations come from fixtures without Knitter in the pool.

The legacy `tests/cases` corpus contains 137 fixtures. It has 49 Knitter
observations across 39 fixtures: 45 direct and four Poet. Knitter appears in 43
deck pools. Its count distribution is 17 zeroes, 24 ones, seven twos, and one
three. All 49 records are likewise textless, unmarked, reference-free, and use
the same direct/Poet key shapes; every count is in-board. No source is recorded
as night-killed or Slayer-killed. Twelve sources are recorded as executed:
nine known true-Evil, one known Good-corrupted, and two other known Good;
eleven are direct and one is Poet.

These archived records exercise the broad historical truth/lie relationship
and retain useful scenario coverage. They do not independently prove exact
text, circular occurrence geometry on boards of size one or two,
registerAs-first alignment, the authored bluff domain, zero-reference acted
history, empty-board failure, or current fail-closed provenance. Current bridge
and Rust tests cover those audited surfaces without bulk-stamping historical
fixtures.

Reconstruction and tooling should therefore:

- bind public Knitter to managed `Knitter` and preserve direct roster use;
- count the complete physical list in stored circular order through registered
  alignment, retaining singleton and two-card duplicate-edge behavior;
- preserve dead, executed, hidden, and corrupted physical occurrences;
- emit only the three exact count text forms with zero references;
- derive bluff candidates from `[0, max(demon + minion, 2))`, remove the actual
  value, and make exactly one retained-index draw;
- treat an empty board as native failure rather than count zero; and
- preserve unmarked observations on their legacy compatibility path.

## Target union accounting

Five Knitter target memberships are exact managed-definition overlaps with the
previous 31 target sets:

- `Knitter..ctor`;
- `ActedInfo..ctor`;
- `Character.GetRegisterAlignment`;
- `Calculator.RemoveNumberAndGetRandomNumberFromList`; and
- `UnityEngine.Random.Range(Int32, Int32)`.

The seven other Knitter identities are new selected FunctionDefinitions.
`Knitter.Act` and `Knitter.BluffAct` add exact managed definitions on native
bodies already present in the union, so the target adds five role-specific
RVAs. This target advances the deterministic union to 32 checked sets, 666
memberships, 426 distinct selected FunctionDefinitions, and 365 unique native
RVAs. The rebuilt GDT has 151,551 datatypes, 426 function definitions, and
6,159 recorded alignments.

## Validation

The build-pinned native pipeline completed the following gates:

- target validation: 12/12 selected functions;
- read-only baseline export: 12/12;
- typed application: 12/12 with six imported datatypes;
- typed post-save read-only validation: 12/12 with 36 parameter storages;
- typed export: 12/12; and
- typed-quality regression check: passed with the metric changes recorded at
  the top of this note.

The current clean-room bridge and solver gates were:

```text
python -m py_compile game_loop.py
python -m unittest -v tests.test_knitter_native tests.test_poet_native
python -m unittest discover
cargo test -p solver-core --release --lib current_knitter
cargo test -p solver-core --release --lib
cargo build --release
cargo test --release --test simulation
```

They passed respectively as compile success, 38/38 focused Python tests,
520/520 full Python tests, 7/7 focused Rust tests, 337/337 full `solver-core`
library tests, a warning-clean release build, and 31/31 full simulation tests
covering all 426 v2 fixtures in 773.56 seconds.

The three added temporal regressions verify that:

- the synchronous conversion event exposes Baker current data and a cleared
  raw bluff while the Spy's Good register-as remains pending reset;
- Medium cannot select the Baker-cleared Spy raw bluff; and
- Lover and Poet/Bounty Hunter accept only one shared monotonic reset history,
  rejecting an Evil-after-reset observation followed by a stale-Good one while
  accepting the forward stale-Good-to-Evil order.

## Remaining uncertainty

- Unity's global PRNG implementation and statistical quality are outside this
  target; the boundary proves one integer-index request over the retained
  domain.
- The native empty-board failure path is identified, but its exact managed
  exception class, UI presentation, and recovery behavior are not reconstructed.
- `Scenario` does not yet store the native current authored Demon and Minion
  lists or a general per-position register-as pointer outside the exact
  Baker/stable-Spy timeline and other explicitly represented identity traces.
- The archived corpus carries no exact current text/reference history and no
  singleton or two-card fixture, so those surfaces depend on build-pinned
  native evidence and clean-room regressions.

This checkpoint closes the shipped Knitter clue-generation boundary for both
direct roster use and retained Poet-provider use. It does not claim that every
remaining role or every runtime identity-mover composition is fully
reconstructed.
