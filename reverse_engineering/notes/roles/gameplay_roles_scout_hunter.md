# Gameplay roles: Scout and Hunter (managed `Tracker`)

Build: `f530404b0f3f_807de4a83df4`

Evidence status: **metadata** for both public `CharacterData` bindings and
**native-static** for every method declared by managed `Scout` and `Tracker`,
their candidate, circular-distance, formatting, acted-reference, and
random-number helpers, and the real/bluff action bodies needed to emit their
Day clues. Native bodies and decompiler output remain outside the repository.

The checked target set is
[`reverse_engineering/targets/gameplay_roles_scout_hunter.json`](../../targets/gameplay_roles_scout_hunter.json).
Its read-only baseline and typed Ghidra exports each complete at 26/26
functions with no failures. The body-free
[`quality report`](../../reports/f530404b0f3f_807de4a83df4_typed_quality_gameplay_roles_scout_hunter.json)
passes its regression check. The exact aggregate deltas are recorded in that
report rather than reproducing private decompiler output here.

## Public asset bindings and managed identities

The shipped `sharedassets0.assets` `CharacterData` at path ID `21631` is named
`Scout`, has `characterId` `Scout_88081716`, and binds exact managed `Scout` at
TypeDefIndex `5854` in `Assembly-CSharp`. Its raw object is 11,252 bytes at
file offset `23,723,368` and has SHA-256
`2189096AC9CCFC39B4FE245805CE1B963C3D135534CC05A845A0953FD2521ACC`.
Its authored description and hint are:

```text
Learn how far a specific Evil is to another closest Evil

Tells you distance from 1 random Evil to its nearest Evil
```

The managed description instead says `Learn how far is a specific Evil to
another Evil`. Scout is a Good Villager (`characterType == 10`,
`startingAlignment == 10`), is bluffable, is not usually disguised, and has
`picking == false` and `abilityUsage == 0`.

The shipped `CharacterData` at path ID `21621` is publicly named `Hunter`, has
`characterId` `Hunter_93427887`, and binds exact managed `Tracker` at
TypeDefIndex `5891`; no managed `Hunter` class exists in the Assembly-CSharp
type index. Its raw object is 3,668 bytes at file offset `23,673,944` and has
SHA-256
`E180A1B912A9F604F15F8206AF64705E4BAD7FF6857C753EB8C49AD89A613419`.
Its authored description is `Learn how far from me is the nearest Evil.`;
`Tracker.get_Description` returns the same text without the period. Hunter has
the same Good-Villager, ability, bluff, disguise, and picker flags as Scout.
The containing asset has build-manifest SHA-256
`E0D239C03FA47EC138F9C2797779E83B65AEF41F30C5AD84D4CA76767A93E967`.

## Audited boundary and shared bodies

| Group | Methods | Observable purpose |
| --- | ---: | --- |
| `Scout` | 9 | Description, Dreamer fallback, real/bluff clue generation, circular scan, formatting, dispatch, construction |
| `Tracker` | 8 | Description, real/bluff clue generation, actor-relative scan, formatting, dispatch, construction |
| `Characters` | 4 | Runtime-alignment filter, attempted generic removal, circle rotation, exact range references |
| `Character` | 2 | Register-as identity and registered alignment |
| `Calculator` | 1 | Remove-one integer candidate draw |
| `UnityEngine.Random` | 2 | Float and integer overload boundaries |

The 26 memberships select 24 distinct native RVAs. Scout and Tracker share the
real `Act` body and the `BluffAct` body; every other membership in this target
has a distinct RVA. Some constructors and selected helper bodies are folded
with additional metadata identities elsewhere in the full union. Folded
canonical names in typed output do not replace the exact managed identities
preserved by the target manifest.

## Scout candidate pool

Both Scout paths begin from a copy of `Gameplay.CurrentCharacters` and call
`FilterRealAlignmentCharacters(..., Evil)`. That helper preserves board order
and occurrence multiplicity and retains exactly entries whose physical runtime
`Character.alignment` is Evil. It does not consult `dataRef`, `registerAs`,
display bluff, statuses, corruption, reveal/death state, or the acting
`charRef`, so the actor is eligible when it is itself runtime Evil.

The next call appears to remove Wretch through
`RemoveCharacterType<Recluse>`, but the shipped generic instantiation tests
whether each `Character.dataRef` object is a managed `Recluse`. `dataRef` is a
`CharacterData`; `Recluse` derives from `Role`. The type test cannot succeed in
a type-correct shipped state, so this call copies the pool without removing a
Wretch role. Ordinary Wretch is already absent because its physical alignment
is Good. A moved runtime-Evil body carrying Wretch data would remain eligible.

Scout makes one integer `Random.Range(0, count)` draw and selects that exact
occurrence. There is no distinct-role deduplication, retry, or self-removal.
An empty runtime-Evil pool reaches indexed-list failure; it does not emit the
one-Evil sentence.

## Scout truthful distance and text

Truthful `GetInfo` labels the selected body with
`picked.GetRegisterAs().characterName`: a live `registerAs` record takes
precedence, otherwise current real `dataRef` is used. The label is therefore
not necessarily the body's stable authored Evil origin after a current-data
move. The clue contains no selected-character reference, so two selectable
occurrences with the same emitted name remain observationally ambiguous.

`GetClosestEvilToEvil` rotates the complete global board so the selected body
is first, removes it, and scans the remaining circle in both directions. Each
scan stops at the first card whose `GetRegisterAlignment() == Evil`, where a
live `registerAs.startingAlignment` takes precedence over physical runtime
alignment. The internal result counts intervening cards: an adjacent Evil is
zero. The public formatter adds one, yielding the ordinary circular edge
distance.

If neither scan finds another registered Evil, the helper returns internal
sentinel `100`. `ConjourInfo` emits:

```text
{characterName} is 1 card away from closest Evil
{characterName} is N cards away from closest Evil
There is only 1 Evil
```

The singular branch is internal zero, the plural branch is internal 1 through
20, and every internal value at least 21 reaches the one-Evil sentence. Newline
layout in the native format strings is whitespace in the saved public text.
Every result is an `ActedInfo` with a null character-reference list.

## Scout bluff support

`GetBluffInfo` first consumes and discards one float
`Random.Range(0.0f, 1.0f)` result. It then selects a runtime-Evil occurrence by
the same integer draw as the truthful path. Its label is the selected body's
direct `dataRef.characterName`, not `GetRegisterAs`; a lie fabricates only the
number and still names a selected runtime-Evil occurrence.

The true internal gap is passed to
`Calculator.RemoveNumberAndGetRandomNumberFromList(actual, 0, 3)`. That helper
builds `[0, 1, 2]`, removes `actual` when present, and makes one uniform integer
index draw from the remainder. After public `+1` formatting, a Scout lie is
always distance 1, 2, or 3 and differs from the true public distance whenever
truth is also in that range. A large/no-other-Evil truth leaves all three lie
values available. Bluff never emits `There is only 1 Evil`.

The Calculator helper's generic empty fallback draws from
`[0, maxExclusive)` without honoring `minInclusive`, but Scout's concrete
`(0, 3)` pool cannot become empty. The observed Scout RNG order is therefore
one discarded float draw, one candidate-index integer draw, then one
false-gap integer draw.

## Hunter truthful distance and text

`Tracker.GetDistanceToEvil` rotates a copy of the global board around the
actor, excludes the actor, and scans forward and backward. Each direction
counts every traversed card starting at one and stops at the first registered
Evil. It returns the smaller count. It does not filter dead, revealed, hidden,
corrupted, or current-data roles beyond the registered-alignment query.

With no registered Evil other than the actor, or none anywhere, both scans
exhaust and return `N - 1`. This is a concrete numeric native behavior, not an
unconstrained/no-info branch. A one-card board consequently returns zero.
Hunter formats exactly:

```text
I am 1 card away from closest Evil
I am N cards away from closest Evil
```

Every integer other than one, including zero, uses the plural form.

Truthful `GetInfo` attaches `Characters.GetCharactersAtRange(distance,
actor)` as its acted-reference list.

## Hunter bluff domain

`Tracker.GetBluffInfo` first computes the truthful distance `t`. It builds the
distinct integers `1..=floor(N / 2)`, removes `t` if present, makes exactly one
uniform integer index draw, formats the selected value, and attaches the same
range helper's references. It does not choose a fake character or require the
two referenced characters to be Good.

The candidate list is empty, and native indexing fails, for `N == 1`, for
`N == 2`, and for a three-card board whose truthful distance is one. On a
three-card board with no other registered Evil, truth is two and lie one
remains available. These are failure edges, not fallback clue values.

## Exact acted references

`Characters.GetCharactersAtRange(d, actor)` rotates the board with the actor
first and removes it. Range zero returns an empty list. For positive
`d <= N - 1`, it appends the forward/current-list-direction seat at distance
`d`, followed by the reverse-direction seat at distance `d`. Order and
multiplicity are preserved. On an even board at `d == N / 2`, both indices are
the same opposite card and that reference is appended twice.

A positive range greater than `N - 1` returns an empty list. A negative range
reaches a negative index failure; an empty board fails during unconditional
actor removal. A missing actor is not validated: full rotation completes,
board index zero is dropped, and lookup continues relative to that malformed
boundary.

This two-reference shape is part of every ordinary native Hunter event,
including two adjacent references for the no-other-Evil `N - 1` result. Treating
Hunter as a zero-reference provider rejects genuine direct and Poet/Hunter
events.

## Decompiler and typing caveats

Typed and baseline role bodies agree on the algorithms above. Several local
artifacts required instruction-level cross-checks:

- the first Scout bluff `Random.Range` was previously conflated with the
  integer overload; machine code and the exact `1.0f` constant establish the
  float overload;
- the baseline Calculator call was typed as returning `void`, while metadata
  and the caller consume its integer return in `eax`;
- generic fully-shared list/type names do not change the concrete
  `<Recluse>` MethodInfo at Scout's call site; and
- folded `Act`, `BluffAct`, and constructor names are ABI-compatible canonical
  labels, not evidence that Scout or Tracker has another managed identity.

## Corpus and reconstruction consequences

The active v2 corpus contains 141 direct Scout observations: 138 numeric
role/distance payloads, including the archived `distance == 0` one-Evil
encoding in `asc41_v4`; two empty observations; and one Rambler shut-up
replacement. It contains 127 direct Hunter observations, all with positive
distances. The corpus also
contains 12 Poet/Scout and nine Poet/Hunter observations, all legacy and
unmarked; it does not independently exercise the new exact public provenance.

Reconstruction, solver, and live ingestion should therefore:

- bind public Hunter to managed `Tracker`, while keeping public Bounty Hunter
  distinct;
- preserve archived unmarked Scout/Hunter fixture behavior, including the
  sole legacy Scout zero sentinel, but provenance-mark current native payloads;
- validate the complete Scout name/distance statement against every matching
  selectable occurrence rather than only the first duplicate role;
- require even a lying Scout to name a selectable runtime-Evil current identity
  and restrict its fabricated public distance to 1 through 3;
- represent the current Scout one-Evil sentence explicitly rather than as
  numeric zero;
- resolve an unrevealed natural Wretch through the ordinary-Outcast multiset
  and trusted HUD budget before accepting or rejecting a registered-Evil
  distance, and keep one stored register-as Minion choice for every explicit
  Wretch body across current Scout observations;
- model Hunter truth as nearest registered Evil or exactly `N - 1`, and model
  its lie as a different member of `1..=floor(N / 2)`; and
- verify Hunter's exact ordered two-reference event, retaining the duplicated
  opposite seat on even boards.

The solver deliberately keeps anonymous natural Good identities grouped inside
each generated scenario. It proves each current Scout/Hunter observation
against an exact required/forbidden hidden-Wretch placement using the shared
Outcast multiset and HUD allocator, and it joins register-as labels whenever
the Wretch's current-data position is explicit. It does not persist one
anonymous hidden-Wretch seat jointly across several separate passive
observations, so that narrow cross-observation surface remains conservative
rather than inventing a hidden identity.

Unity's global PRNG algorithm and statistical quality, exact managed exception
classes on malformed/empty pools, destroyed-object behavior beyond ordinary
Unity-null checks, and a full model of every role-specific live register-as
identity remain outside this boundary.
