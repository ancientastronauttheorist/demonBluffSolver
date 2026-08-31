# Gameplay role: Enlightened

Build: `f530404b0f3f_807de4a83df4`

Evidence status: **metadata** for the public `CharacterData` binding and normal
roster membership, and **native-static** for every method declared by managed
`Shugenja`, circular nearest registered-Evil direction, exact truth and bluff
result construction, claimed-direction runtime data, Day dispatch, float RNG,
and the dormant marking helper. The retained Poet-provider path and archived
fixture corpus provide additional behavioral compatibility evidence. Native
bodies and decompiler output remain outside the repository.

The checked target set is
[`reverse_engineering/targets/gameplay_role_enlightened.json`](../../targets/gameplay_role_enlightened.json).
It selects 14 managed FunctionDefinitions at 14 distinct target-local native
RVAs. Its read-only baseline and typed exports each complete at 14/14 functions
with no failures. Typed application imports 12 newly reachable datatypes, and
post-save ABI validation records 43 parameter storages with zero program
mutations. The rebuilt GDT contains 151,561 datatypes and 435 function
definitions.

The body-free
[`quality report`](../../reports/f530404b0f3f_807de4a83df4_typed_quality_gameplay_role_enlightened.json)
passes its regression check: unresolved-type tokens fall from 97 to 24, raw
field-offset accesses from 60 to 38, raw integer type tokens from 44 to 23,
placeholder parameter tokens from 87 to zero, and indirect-call patterns from
four to zero. Both exports retain three decompiler-error and 20 warning
markers.

## Public asset binding and managed identity

The shipped `sharedassets0.assets` `CharacterData` at path ID `21618` is named
`Enlightened`, has `characterId` `Enlightened_62576217`, and binds to managed
`Shugenja` at TypeDefIndex `5890` in `Assembly-CSharp`. The object is 4,240
bytes at file offset `23,660,056` and has SHA-256
`BF64FF898064315E1BD3B2F61C53B9F21347B649037977573744B2905B4A6177`.
The containing asset has SHA-256
`E0D239C03FA47EC138F9C2797779E83B65AEF41F30C5AD84D4CA76767A93E967`.

Enlightened is a Good Villager (`characterType == 10`,
`startingAlignment == 10`). It uses passive Once behavior
(`abilityUsage == 0`), is bluffable, is not usually disguised, and has
`picking == false`. Its authored public description is:

```text
Learn if closest Evil to me is Clockwise or Counter-Clockwise.

Learn 'Equidistant' if Evils are at the same distance from me.
```

The managed description getter separately returns:

```text
Learn if closest Evil is clockwise or counter-clockwise. Learn 'either' if equidistant.
```

The serialized role type string occurs in the object at raw offset `0x1070`.
The asset's bundled-character, skin, achievement, additional-status, tag,
`canAppearIf`, hint, and if-lies collections are empty.

The normal serialized level object `level0` at path ID `139347`, SHA-256
`B509AC15904F0926419CCCC1D2E86508C69B0E3AF2235A08FF8E9FC4D170C7F1`,
contains one normal candidate-pool reference to this exact CharacterData at
path ID `21618`. Enlightened is therefore live as a direct Standard/Ascension
card, independently of Poet. It is absent from the 15-entry
`startGameActOrder`, and its managed action bodies contain no Start branch.

## Audited boundary and shared bodies

| Group | Methods | Observable purpose |
| --- | ---: | --- |
| `Shugenja` | 9 | Description, truth/bluff clue generation, circular direction, dormant marking, Day dispatch, and construction |
| `EnlightenedRuntimeData` | 1 | Claimed direction stored on the acting Character |
| `ActedInfo` | 1 | Exact text with a null/zero-reference result list |
| `CharactersHelper` | 1 | Actor-first physical-list rotation used only by dormant marking |
| `Character` | 1 | Register-as-first alignment projection |
| `UnityEngine.Random` | 1 | One float draw over `[0,1]` for bluff selection |

All 14 target memberships are:

| Managed identity | RVA | Boundary purpose |
| --- | ---: | --- |
| `Shugenja.get_Description` | `0x3EBAC0` | Managed description text |
| `Shugenja.GetInfo` | `0x3EB650` | Truth direction, runtime data, and zero-reference result |
| `Shugenja.Act` | `0x3B09F0` | Day-only truthful callback |
| `Shugenja.BluffAct` | `0x3B33E0` | Day-only bluff callback |
| `Shugenja.GetBluffInfo` | `0x3EAF50` | False direction, runtime data, and zero-reference result |
| `Shugenja.GetDirectionToEvil` | `0x3EB090` | Nearest registered-Evil direction on the full physical circle |
| `Shugenja.GetMarkedCharacters` | `0x3EB720` | Dormant positional marking helper |
| `Shugenja.ConjourInfo` | `0x3EAEA0` | Exact direction-to-text construction |
| `Shugenja..ctor` | `0x3CFFF0` | Fieldless role construction and Poet-provider identity |
| `EnlightenedRuntimeData..ctor` | `0x357700` | Direction enum storage at instance offset `0x10` |
| `ActedInfo..ctor` | `0x35D5D0` | Exact text and null Character-reference-list storage |
| `CharactersHelper.GetSortedListWithCharacterFirst` | `0x398AF0` | Global-board rotation for dormant marking |
| `Character.GetRegisterAlignment` | `0x365030` | Register-as-first alignment projection |
| `UnityEngine.Random.Range(Single, Single)` | `0x1C86640` | Inclusive float-range request used by bluff |

`Shugenja.Act`, `Shugenja.BluffAct`, and `Shugenja..ctor` use broad shared
native bodies and explicit ABI-compatible canonical prototypes already present
in the typed project. `EnlightenedRuntimeData..ctor` has 185 native aliases but
retains its exact managed signature and distinct FunctionDefinition. Target
validation reports four shared native bodies; signature application
canonicalizes the three bodies that reuse pre-existing target prototypes.
This does not erase any exact managed identity.

The target intentionally stops at the role-specific callable boundary and its
semantic helpers. Generic list operations, string concatenation, Unity object
liveness, the callback sink, and Unity's PRNG internals are shared engine
surfaces rather than additional Enlightened memberships.

## Circular nearest registered-Evil direction

For a normal non-null actor in `Gameplay.CurrentCharacters`,
`Shugenja.GetDirectionToEvil` constructs two occurrence lists:

1. stored forward `CurrentCharacters` order starting immediately after the
   actor, wrapping at the end, with the actor omitted; and
2. an exact reversed copy of that first list.

Each scan starts at physical distance one, stops at its first occurrence whose
`Character.GetRegisterAlignment()` is Evil (`20`), and otherwise exhausts the
whole list. The two physical step counts are then compared:

| Comparison | Internal enum | Public result |
| --- | ---: | --- |
| forward count `<` reverse count | `Counterclockwise == 20` | Counter-clockwise |
| reverse count `<` forward count | `Clockwise == 10` | Clockwise |
| counts equal | `Either == 0` | Equidistant |

This orientation is easy to invert accidentally. Stored `CurrentCharacters`
and solver/public position IDs increase in the native forward direction, which
the role names **Counterclockwise**. Clockwise therefore moves through
decreasing public IDs with wraparound. The clean historical observation
`asc27_v1` corroborates the mapping: truthful Enlightened #7 reports Clockwise
toward nearest Evil #6. A generic circle helper that assumes increasing IDs
are clockwise must swap its directions before it can model this role.

Registered alignment is exactly:

```text
live registerAs != null ? registerAs.startingAlignment : Character.alignment
```

The scan preserves the complete physical board order and occurrence
multiplicity. It does not filter corruption, current or displayed role, dead
or executed state, reveal or visibility, status, runtime-data class, or
physical origin. The actor itself is never a candidate, even when it registers
Evil. A malformed actor not found in the list leaves the full board as the
forward scan rather than producing a sentinel; shipped direct and Poet calls
pass their physical acting Character.

The equality rule has exact edge consequences:

- no registered Evil other than the actor: both scans exhaust, so Equidistant;
- `N == 0`: both existing empty lists exhaust at zero, so Equidistant;
- `N == 1`: the actor is removed and both scans exhaust, so Equidistant;
- `N == 2`: the one other occurrence appears at distance one in both lists,
  so every result is Equidistant whether or not that occurrence is Evil;
- equal-distance nearest Evils on opposite sides: Equidistant; and
- an even-board opposite nearest Evil: it occurs at the same distance in both
  lists, so Equidistant.

The result is recomputed at each Day action. It is not a Start-time snapshot
and does not read a previously stored `EnlightenedRuntimeData` value.

## Truth result, runtime data, and zero references

`Shugenja.GetInfo` computes `GetDirectionToEvil(charRef)` exactly once, calls
`ConjourInfo` with that enum, constructs a fresh `EnlightenedRuntimeData`, and
replaces `charRef.runtimeData` with it. `EnlightenedRuntimeData.direction` is
stored at object offset `0x10`. Truth consumes no clue RNG.

The exact public strings are:

| Enum | Exact text |
| --- | --- |
| `Clockwise == 10` | `Closest Evil is:\nClockwise` |
| `Counterclockwise == 20` | `Closest Evil is:\nCounter-clockwise` |
| `Either == 0` | `Closest Evil is equidistant` |

There is no colon in the Equidistant form, no newline in that form, and no
terminal punctuation in any form. `Counter-clockwise` has the shipped hyphen
and lower-case `c` after it. Case, spacing, and newline placement are part of
the observation schema.

The returned `ActedInfo` receives a null `List<Character>` argument. The live
bridge therefore observes exactly zero acted references, not an omitted
unknown list and not a list of marked or candidate Evil cards. A null
`charRef` reaches native failure after direction, text, and runtime-object
construction; normal dispatch always supplies a Character.

The fresh runtime object records the **emitted** direction. For truth that is
also the actual direction. The runtime field exists to support display/hint
state and observation provenance; it is not an independent hidden-truth
oracle.

## Bluff domain and RNG behavior

`Shugenja.GetBluffInfo` first computes the same actual direction. It then makes
exactly one `UnityEngine.Random.Range(0.0f, 1.0f)` call and selects from the two
false enum values with these exact comparisons:

| Actual | Draw branch | Emitted | Uniform branch mass |
| --- | --- | --- | ---: |
| Clockwise | `u < 0.2` | Equidistant | 20% |
| Clockwise | `u >= 0.2` | Counter-clockwise | 80% |
| Counter-clockwise | `u < 0.2` | Equidistant | 20% |
| Counter-clockwise | `u >= 0.2` | Clockwise | 80% |
| Equidistant | `u < 0.5` | Clockwise | 50% |
| Equidistant | `u >= 0.5` | Counter-clockwise | 50% |

The role never emits the actual enum on a valid path. There is no retry loop,
integer-domain construction, board-size dependency, authored Demon/Minion
count dependency, or second draw. Unity's float overload is requested with
the inclusive documented endpoint form; the branch comparisons above are the
native contract, while engine-level PRNG implementation and statistical
quality remain outside this target.

Bluff uses the same exact `ConjourInfo` strings and null/zero-reference
`ActedInfo` shape as truth. It constructs a fresh `EnlightenedRuntimeData` and
stores the **claimed false direction**, not the actual direction, on the
acting Character. A readable runtime value therefore authenticates the newest
claimed Enlightened sentence; it must not be treated as an actual-Evil-side
leak.

## Day dispatch

`Shugenja.Act` and `Shugenja.BluffAct` are separately folded shared bodies.
Each recognizes only `Day == 30`. With a non-null callback, truth calls the
virtual `GetInfo` once and delivers that result once; bluff analogously calls
`GetBluffInfo` once and delivers it once. A null callback or any other trigger
is a no-op and consumes no clue RNG.

Neither body contains Start, Night, Reveal, Executed, death, reset, picker,
achievement, status, resistance, or active-ability behavior. Upstream generic
truth routing decides whether the real or bluff body runs, so a corrupted Good
Enlightened uses the false-direction generator without changing the role's
geometry.

## Dormant marking helper

`Shugenja.GetMarkedCharacters` is declared and fully implemented, but it is
not called by `GetInfo`, `GetBluffInfo`, `ConjourInfo`, or the live Enlightened
hint path. A pinned executable-reference scan finds zero direct relative-call
or jump references to RVA `0x3EB720` and exactly one ordinary non-executable
IL2CPP method-registration pointer. The method is therefore classified
unreachable from shipped executable call sites, while remaining documented as
part of the declared managed surface.

Its direct-call behavior is notably asymmetric and must not be substituted for
the clue's nearest-Evil scan. With a normal actor on a board of size `N`, it
rotates the global list so the actor is first, removes that actor, and calls the
remaining ordered list `R` of length `m = N - 1`. Define:

```text
a = ceil(m / 2)
b = a                 when m is odd
    a - 1             when m is even
```

Using zero-based indices into `R`:

- Counterclockwise (`20`) returns every occurrence with `i <= a`;
- Clockwise (`10`) returns every occurrence with `i >= b`; and
- Either (`0`) returns an empty list.

For odd `N >= 3`, both directional results contain `(N + 1) / 2` occurrences
and overlap in two middle occurrences. For even `N`, Counterclockwise contains
`N/2 + 1`, Clockwise contains `N/2 - 1`, and they overlap in one occurrence.
At `N == 2`, Counterclockwise returns the sole other card and Clockwise returns
none. At `N == 1`, both return empty. An empty input fails at `RemoveAt(0)`.
No alignment or role test occurs in this helper.

These odd counts are dormant implementation facts, not the acted-reference
schema: actual Enlightened clues always return zero references.

## Poet provider parity

Public Poet is managed `Gossip`. Its constructor creates an exact ordered list
of twelve provider roles; `Shugenja`/Enlightened is provider entry 8 in the
one-based authored order. The provider remains present even when Enlightened
is absent from the current deck pool.

Once Poet selects Shugenja, it delegates with the original Poet Character as
`charRef`. Direct and Poet observations therefore share the same direction
algorithm, exact text, zero-reference shape, and runtime-data class, with two
important ownership consequences:

- circle geometry is anchored on the Poet's physical seat, not a temporary
  provider object or an Enlightened card elsewhere; and
- the fresh `EnlightenedRuntimeData` is stored on the Poet Character.

Truthful Poet consumes the provider-index selection draw and then Shugenja
truth consumes no additional clue draw. Lying Poet consumes the provider-index
draw and, after selecting Shugenja, exactly one Shugenja float draw. The
provider object itself is fieldless and carries no persistent actor geometry.

## Current state and cross-role chronology

Enlightened's only hidden-state dependency is the registered-alignment
projection at the instant its Day clue is generated. Several other roles can
change or expose that projection:

- a natural Wretch is runtime Good but can hold an Evil `registerAs`, so it is
  an ordinary nearest-Evil candidate;
- a stable Spy is runtime Evil but can retain a Good `registerAs`, so it is not
  a candidate while that surface remains live;
- Baker conversion can synchronously replace a stable Spy's current data while
  preserving the stale Good register-as until the delayed Reveal reset, after
  which the same physical runtime-Evil body falls back to Evil alignment; and
- any observation on opposite sides of that delayed reset can legitimately
  change direction without a physical seat moving.

Current-data changes do not move the anchor. If Shaman copies Enlightened onto
a destination Villager, the copied role has no Start action; its later Day clue
is anchored on the destination body and overwrites any runtime object preserved
by `InitWithNoReset`. Twin/current-data movement and other role replacement
likewise use the physical Character passed to the eventual Shugenja action.

Conversely, later Baker or other current-data replacement may preserve an old
`EnlightenedRuntimeData` on a Character that no longer emitted the newest
visible event as Enlightened. A memory bridge must authenticate the exact
current speech/event role and zero-reference shape before using the runtime
direction. Runtime data alone is insufficient and can be stale.

Corruption and Evil alignment affect upstream truth-versus-bluff dispatch, not
the circular candidate predicate. Dead, executed, hidden, blocked, or
unrevealed physical cards remain candidates if they register Evil. The clue
does not depend on current authored Demon/Minion counts, actual CharacterData
role names, displayed disguises, or the dormant marked list.

## Corpus and compatibility implications

A deterministic scan of the 426 checked-in `tests/cases_v2` fixtures finds:

- Enlightened appears in a deck pool in 103 fixtures;
- 112 direct apparent Enlightened observations occur across 93 fixtures;
- direct values are 42 Clockwise, 53 Counterclockwise, and 17 Equidistant;
- ten Poet/Enlightened observations occur across ten fixtures: three
  Clockwise, four Counterclockwise, and three Equidistant;
- three fixtures contain both a direct and Poet observation, for 122 total
  observations across 100 fixtures;
- every direct record has Enlightened in its deck pool, while seven of the ten
  Poet records come from fixtures without Enlightened in the pool;
- direct board sizes range from seven through ten, so no archived record
  exercises one- or two-card truth geometry; and
- seven Baker observations separately name Enlightened as an original role,
  which is cross-role identity history rather than an Enlightened direction
  clue.

All 122 v2 observations are textless, unversioned, and reference-free. Direct
payloads use exactly `direction`; Poet payloads use exactly `copied_role` plus
`direction`. They therefore remain historical compatibility records rather
than current exact-schema evidence.

The legacy `tests/cases` corpus contains 137 fixtures. Enlightened appears in
33 deck pools and has 33 direct observations across 29 fixtures plus three
Poet observations. Direct values are 17 Clockwise, 11 Counterclockwise, four
Equidistant, and one legacy `equal` spelling; Poet values are one Clockwise and
two Counterclockwise. All three Poet records occur in fixtures that also have
a direct observation. These 36 records are likewise unmarked, textless, and
reference-free, and two Baker records name Enlightened as an original role.

Together the two corpora contain 158 archived direct-or-Poet direction
observations. They protect broad legacy direction compatibility, ordinary
wraparound, Evil disguises, corrupted false directions, and Poet delegation.
They do not independently prove exact current text, runtime-data provenance,
zero-reference chronology, the 80/20 or 50/50 bluff branches,
registerAs-first timing, no-Evil behavior, or tiny-board ties. Current bridge
and solver tests must cover those native-static surfaces without bulk-stamping
historical fixtures.

The corpus also supplies a useful orientation regression: in `asc27_v1`,
clean Good Enlightened #7 reports Clockwise with true Evils #5 and #6. The
nearest registered Evil is #6, one decreasing-ID step, independently guarding
against a clockwise/counterclockwise inversion.

## Clean-room solver and bridge boundary

The native facts above define the following current-observation contract. This
section is a reconstruction boundary, not an additional claim about hidden
native state.

- Current direct observations should carry a closed `public_current` variant,
  one canonical `direction` value (`CW`, `CCW`, or `Equidistant`), the exact
  matching sentence, and zero acted references.
- Current Poet observations should carry the same closed direction/text
  contract plus exact copied-role provenance, anchored on the Poet position.
- A readable newest runtime object must be `EnlightenedRuntimeData` with the
  same claimed enum. An unreadable runtime object may leave provenance
  unknown; a readable mismatch must fail closed.
- Truth accepts only the actual registered-Evil direction. Bluff accepts the
  two and only two directions different from actual; both have positive native
  support, while their unequal probability weights need not become logical
  impossibility weights.
- Position arithmetic must implement increasing IDs as Counterclockwise and
  decreasing IDs as Clockwise.
- No-Evil, singleton, two-card, equal-nearest, and even-opposite worlds all
  support only Equidistant truth.
- Anonymous natural-Wretch assignments and Baker/Spy register-as reset timing
  must be solved jointly with other current observations rather than chosen
  independently per clue.
- The actor is excluded, but every other lifecycle seat remains in the circle
  regardless of death, execution, reveal, visibility, or corruption.
- The dormant `GetMarkedCharacters` output must never be used as acted
  references or as a substitute truth predicate.
- Unmarked archived observations should remain on their explicit legacy
  compatibility predicate; the current marker must fail closed on malformed or
  mixed provenance.

Minimum focused regressions for any consuming implementation are:

1. all three exact direct and Poet strings, including newline, hyphenation,
   capitalization, zero references, and readable runtime-data agreement;
2. rejection of whitespace-normalized, case-normalized, extra-field,
   out-of-board, wrong-role, wrong-runtime, and stale-runtime current records;
3. the `asc27_v1` #7-to-#6 decreasing-ID Clockwise orientation;
4. no Evil, actor-only Evil, `N == 1`, both `N == 2` cases, even opposite, and
   equal nearest Evils;
5. retained dead/executed candidates and actor exclusion;
6. natural-Wretch register-as Evil and stable-Spy register-as Good;
7. all three Baker/Spy phases around the delayed register-as reset;
8. truth versus both supported bluff outputs for each actual direction;
9. direct versus Poet anchor and runtime-data ownership parity; and
10. unmarked legacy preservation without accepting legacy spellings under a
    current marker.

## Target union accounting

Five Enlightened target memberships are exact managed-definition overlaps with
the previous 32 target sets:

- `Shugenja..ctor`;
- `ActedInfo..ctor`;
- `CharactersHelper.GetSortedListWithCharacterFirst`;
- `Character.GetRegisterAlignment`; and
- `UnityEngine.Random.Range(Single, Single)`.

The other nine memberships are new selected FunctionDefinitions. The folded
`Shugenja.Act` and `Shugenja.BluffAct` definitions use native bodies already in
the union; the remaining seven new definitions add seven new RVAs. This target
advances the deterministic union to 33 checked sets, 680 memberships, 435
distinct selected FunctionDefinitions, and 372 unique native RVAs. There are
245 membership overlaps and a 63-definition folded/shared-body gap.

The rebuilt GDT has 151,561 datatypes, 435 function definitions, and 6,159
recorded alignments. Across this target, application imports 12 datatypes,
applies and validates all 14 memberships, and canonicalizes three shared
bodies. The separate read-only pass validates all 14 memberships and 43
membership-level parameter-storage locations with zero program mutations.

## Validation

The build-pinned native pipeline completed the following gates:

- target validation: 14/14 selected functions, with four shared native bodies;
- read-only baseline export: 14/14, zero failures;
- deterministic type build: 5,830 inheritance rewrites, 6,159 alignments, 435
  function definitions, and 151,561 GDT datatypes;
- typed application: 14/14, with 12 imported datatypes;
- typed post-save read-only validation: 14/14, 43 parameter storages, zero
  mutations;
- typed export: 14/14, zero failures;
- typed-quality regression check: passed; and
- method-ledger write plus byte-for-byte check: 4,207 methods, 3,066 unique
  native bodies, 435 classifications, and 201 evidence records.

The serialized Ghidra commands were the existing `export-target`,
`build-types`, `typed-refresh`, and `typed-export` stages. Baseline and typed
exports remain in the private build artifact root; no native body, raw bytes,
decompiler excerpt, or private export path is checked into the repository.

## Remaining uncertainty

- Unity's global PRNG implementation, seed lifecycle, and exact finite-sample
  statistical quality are outside this target. The role boundary proves the
  one float request and exact comparison thresholds.
- Current archived fixtures do not store exact speech text, acted-reference
  history, or readable runtime-data provenance. A paired live capture would add
  behavioral corroboration beyond native-static proof and synthetic tests.
- `GetMarkedCharacters` has no executable direct caller, but this audit does
  not claim that arbitrary reflection or a future build could never invoke the
  public method.
- Native null/global-list failure paths are bounded, but exact managed exception
  classes, UI presentation, and recovery are not reconstructed.
- Arbitrary compositions that move current data, register-as identity, and
  preserved runtime objects still require one coherent chronological world in
  the solver; the native role itself does not serialize that history.

This checkpoint closes the shipped public Enlightened/Shugenja clue-generation
boundary and its declared helper surface. It does not claim that every Poet
provider, every role, or the whole game is fully decompiled.
