# Gameplay role: Medium (managed `Lookout`)

Build: `f530404b0f3f_807de4a83df4`

Evidence status: **metadata** for the public `CharacterData` binding and normal
roster membership, and **native-static** for every method declared by managed
`Lookout`, its truth and bluff selection surfaces, exact result construction,
Day dispatch, and execution achievement. The retained Poet-provider path and
the archived 426-fixture corpus provide additional behavioral compatibility
evidence. Native bodies and decompiler output remain outside the repository.

The checked target set is
[`reverse_engineering/targets/gameplay_role_medium.json`](../../targets/gameplay_role_medium.json).
It selects 14 managed FunctionDefinitions at 14 distinct target-local native
RVAs. Its read-only baseline and typed exports each complete at 14/14 functions
with no failures. Post-save ABI validation records 42 parameter storages and
six imported datatypes; the rebuilt GDT contains 151,544 datatypes.

The body-free
[`quality report`](../../reports/f530404b0f3f_807de4a83df4_typed_quality_gameplay_role_medium.json)
passes its regression check: unresolved-type tokens fall from 87 to 30, raw
field-offset accesses from 72 to 18, raw integer type tokens from 51 to eight,
placeholder parameter tokens from 79 to zero, and indirect-call patterns from
four to zero. Both exports retain three decompiler-error and 21 warning
markers.

## Public asset binding and managed identity

The shipped `sharedassets0.assets` `CharacterData` at path ID `21627` is named
`Medium`, has `characterId` `Lookout_41018246`, and binds its managed role to
exact `Lookout` at TypeDefIndex `5881` in `Assembly-CSharp`. The object is
10,280 bytes at file offset `23,701,456` and has SHA-256
`EADF3FDFBFE26A013803DFCFBBB0B1AF94AFB7762095E36B5E322410BA8A9650`.
The containing asset has SHA-256
`E0D239C03FA47EC138F9C2797779E83B65AEF41F30C5AD84D4CA76767A93E967`.

Medium is a Good Villager (`characterType == 10`,
`startingAlignment == 10`). It is bluffable, is not usually disguised, has
`picking == false`, and uses passive `abilityUsage == 0`. Its authored
description is:

```text
Learn a good character and its role.
```

The authored if-lies text is:

```text
My info includes Disguised character
```

The managed description getter retains older wording:

```text
Learn that a character is a particular Villager
```

The executable identity is therefore managed `Lookout`, not a guessed class
named after the public card. This is also distinct from similarly phrased
selection roles: Medium chooses one Character and reports a role identity,
while Bounty Hunter reports only an Evil ID and Hunter reports a distance.

The normal serialized level object, SHA-256
`B509AC15904F0926419CCCC1D2E86508C69B0E3AF2235A08FF8E9FC4D170C7F1`,
contains one normal candidate-pool reference to this exact CharacterData. The
role is therefore live as a direct Standard/Ascension card, unlike the dormant
physical Bounty Hunter asset. Medium is absent from `startGameActOrder`, and
its managed action bodies contain no Start branch.

## Audited boundary and shared bodies

| Group | Methods | Observable purpose |
| --- | ---: | --- |
| `Lookout` | 8 | Description, truth/bluff generation, exact text, Day and execution dispatch, achievement helper, and construction |
| `ActedInfo` | 1 | Exact text plus one Character-reference list |
| `Characters` | 1 | Registered-alignment filtering over a supplied Character list |
| `Character` | 2 | Register-as-first alignment and CharacterData projections |
| `CharacterStatuses` | 1 | `HealthyBluff` presence test |
| `UnityEngine.Random` | 1 | One integer-index draw over a retained occurrence pool |

All 14 target memberships are:

| Managed identity | RVA | Boundary purpose |
| --- | ---: | --- |
| `Lookout.get_Description` | `0x3E4060` | Managed description text |
| `Lookout.GetInfo` | `0x3E3DE0` | Registered-Good truth selection and result construction |
| `Lookout.Act` | `0x3B09F0` | Day-only truthful callback |
| `Lookout.BluffAct` | `0x3E3710` | Day bluff callback and Executed achievement branch |
| `Lookout.GetBluffInfo` | `0x3E3920` | Raw-bluff-holder selection and false-role result |
| `Lookout.ConjourInfo` | `0x3E3800` | Exact `real`/`actually` text construction |
| `Lookout.CheckAchievementsAndUnlockIfAble` | `0x3E37C0` | Declared but normally unreachable unconditional unlock helper |
| `Lookout..ctor` | `0x3CFFF0` | Fieldless role construction and Poet-provider identity |
| `ActedInfo..ctor` | `0x35D5D0` | Exact text and one-reference result storage |
| `Characters.FilterAlignmentCharacters(Character)` | `0x36A030` | Ordered registered-alignment occurrence filter |
| `Character.GetRegisterAlignment` | `0x365030` | Register-as-first alignment projection |
| `Character.GetCharacterData` | `0x364D60` | Register-as-first role-label projection |
| `CharacterStatuses.Contains` | `0x363C40` | `HealthyBluff` status query |
| `UnityEngine.Random.Range(Int32, Int32)` | `0x1C86600` | Inclusive-lower, exclusive-upper index selection |

`Lookout.Act` uses the broad Day real-action body already selected under other
managed role identities. `Lookout..ctor` uses the broad fieldless-role
constructor body, and the integer `Random.Range` overload is canonicalized
separately from its float sibling. Applied ABI-compatible prototypes do not
change the exact managed identities selected by this target.

The target intentionally stops at the role-specific callable boundary and its
semantic helpers. Generic list operations, Unity object-liveness helpers, and
the global achievement sink are shared engine surfaces rather than additional
Medium memberships. Bluff acquisition is an upstream Character lifecycle
dependency documented below, not a call made by `Lookout.GetBluffInfo`.

## Truthful candidate pool and role source

Each `Lookout.GetInfo` call starts from a fresh copy of the complete physical
`Gameplay.CurrentCharacters` list. It filters that snapshot to occurrences
whose current registered alignment is Good (`10`) through
`Characters.FilterAlignmentCharacters(Character)`. The filter preserves list
order and occurrence multiplicity and uses:

```text
live registerAs != null ? registerAs.startingAlignment : Character.alignment
```

No additional filter checks current role type, current `dataRef`, displayed
bluff, corruption, status, death, reveal, visibility, or physical origin.
Dead, hidden, and revealed registered-Good Characters remain candidates.

After filtering, truth removes one occurrence Unity-equal to the supplied
`charRef` only when the filtered list contains more than one occurrence. It
does not unconditionally exclude the actor. Consequently:

- when another registered-Good occurrence exists, the actor is removed once;
- when the actor is the only registered-Good occurrence, self remains and is
  the only legal result;
- when the actor is not registered Good, removal changes nothing; and
- a malformed duplicated actor reference can leave another equal occurrence.

Truth then makes exactly one integer `Random.Range(0, Count)` draw and indexes
that ordered pool. There is no shuffle, retry, cached target, weighted role
draw, or separate identity draw.

The selected role label comes from `Character.GetCharacterData`:

```text
live registerAs != null ? registerAs : dataRef
```

This label surface is deliberately parallel to, but distinct from, the
registered-alignment filter. Consequences include:

- a Spy's live Villager `registerAs` can make a physically runtime-Evil body a
  registered-Good candidate and is also the role Medium reports;
- a Wretch's live Evil register-as excludes it from truth even on a
  runtime-Good body;
- ordinary Drunk and Doppelganger do not gain a special register-as from their
  bluff, so Medium reports their current data rather than displayed bluff;
- a runtime-Good body holding moved Twin Minion data remains registered Good
  when it has no register-as and truthfully reports that current Twin data;
  and
- a runtime-Evil Twin origin holding moved Good data remains registered Evil
  and is excluded unless another live Good register-as overrides it.

Truth constructs a fresh reference list containing exactly the one selected
physical Character, calls `ConjourInfo(selected.id, selectedData)`, and returns
`ActedInfo(exactText, [selected])`.

An empty registered-Good pool is not a supported sentinel. Native reaches the
zero-width integer draw and then indexed-list failure. Null board/list entries
or a null selected CharacterData likewise follow failure paths rather than
yielding a no-information result. A null `charRef` does not itself create a
sentinel: removal simply cannot exclude a board Character.

## Bluff-holder pool and reveal history

`Lookout.GetBluffInfo` also starts from a fresh copy of the complete physical
`CurrentCharacters` list, but it does not filter by alignment or role. It
builds a fresh first-pass pool from every occurrence satisfying both:

```text
Character.bluff is Unity-live
Character is not Unity-equal to charRef
```

If that non-self pool is empty, it performs a second full-board pass and adds
every occurrence with a live raw `Character.bluff`, including the actor. Thus
self is legal only when no other raw-bluff holder exists. The fallback is based
on holder existence, not on whether another Character has the same bluff role.

Bluff makes one integer `Random.Range(0, Count)` draw, appends exactly the
selected physical Character to a fresh reference list, and passes the selected
Character's raw `bluff` CharacterData to `ConjourInfo`. It does not use the
target's `GetCharacterData`, current role, apparent role, register-as identity,
runtime alignment, or registered alignment. The exact false-role output is:

```text
ActedInfo(ConjourInfo(selected.id, selected.bluff), [selected])
```

The pool keeps full board order and occurrence multiplicity and does not filter
death, reveal state, visibility, corruption, status, or faction. If no raw
bluff holder exists after fallback, the zero-width draw/index path fails;
there is no “no disguised character” sentence.

Raw bluff availability is produced upstream by the Character lifecycle.
Internal `Character.Reveal` resolves register-as first and, only while the raw
`bluff` pointer is Unity-null, asks the then-current `dataRef.role` for
`GetBluffIfAble(charRef)` and stores the returned CharacterData through
`GiveBluff`. A live first result persists, so later delayed-Reveal
continuations do not redraw it. A null result leaves the pointer null and can
be retried by a later continuation.

This persistence makes holder identity a history surface, not a simple final
alignment predicate:

- ordinary runtime-Evil Minions and Demons acquire Good bluffs;
- Spy, Mutant, Drunk, Doppelganger, and generated Puppet have their own
  acquisition paths;
- Twin inherits Minion's selector, so a runtime-Good body holding current Twin
  data can acquire and retain a raw bluff;
- Twin's additional endpoint continuations make first-success ordering
  observable; and
- later Twin, Shaman, Chancellor, Baker, or other data movement can leave a
  raw bluff whose identity no longer matches final current data.

Therefore “runtime Evil plus Drunk, Doppelganger, and Puppet” is not a complete
bluff-holder representation. Conversely, an ordinary Good body whose known
current role uses base `Role.GetBluffIfAble == null` and has no prior mover or
bluff history can be proved not to occur in the bluff pool.

## Exact text and result references

`Lookout.ConjourInfo` receives a displayed ID and a CharacterData role source.
It checks whether that data's managed role is `Drunk` or assignable to Drunk.
The two exact public formats are:

```text
ordinary data:  #{id} is a real
                {characterName}

Drunk data:     #{id} is actually a
                {characterName}
```

The embedded newline before `characterName`, lower-case `real` and `actually`,
article placement, spaces, and lack of terminal punctuation are exact. For
ordinary shipped Drunk data, `characterName` is `Drunk`, yielding:

```text
#{id} is actually a
Drunk
```

The branch tests the supplied CharacterData role, not the target's physical
origin or apparent card. Truth supplies registerAs-or-dataRef; bluff supplies
the raw bluff CharacterData. A null CharacterData fails. A non-null data object
with a null role takes the ordinary branch, while its stored `characterName`
remains the appended label surface.

Both truth and bluff results carry exactly one Character reference: the same
physical occurrence whose displayed ID appears in the sentence. This is not a
null reference list, an empty list, or a reference to the reported
CharacterData object.

## Day dispatch and execution achievement

`Lookout.Act` recognizes only `ETriggerPhase.Day == 30`. On Day it captures the
inherited acted-result callback. With a non-null callback it invokes virtual
`GetInfo(charRef)` once and invokes the captured callback once with that exact
result. With a null callback, or on any other trigger, it returns without
selection or RNG.

`Lookout.BluffAct` has two observable trigger branches:

- on Day (`30`), a non-null callback causes one `GetBluffInfo` call and one
  callback; a null callback causes no generation; and
- on Executed (`40`), it checks the supplied Character's status collection for
  `HealthyBluff` (`30`). When `HealthyBluff` is absent it requests exact
  achievement key `Medium_Halloween_ACHIV_6997`; when present it does not.

The execution branch has no clue callback or RNG. Null Character/status
dependencies follow native failure paths. The generic Character truth-routing
boundary determines whether `Lookout.Act` or `Lookout.BluffAct` receives a
trigger; Medium itself does not recompute lying state inside these methods. A
truthful Medium therefore has no Medium-specific execution unlock path, while
a lying Medium or Medium bluff can reach the `BluffAct` branch.

`Lookout.CheckAchievementsAndUnlockIfAble` is a separate declared method that
unconditionally requests the same achievement key. A pinned executable-xref
scan finds no executable direct relative call or jump to it and exactly one
ordinary IL2CPP method-registration pointer. Reachable `BluffAct` contains its
own inlined HealthyBluff test and unlock request. The helper is consequently
classified unreachable through the normal shipped call graph, while
reflection or malformed external invocation is not absolutely ruled out.

`Lookout.ConjourInfo` is not dormant: the same scan finds one direct executable
call from `GetInfo` and one from `GetBluffInfo`.

## Poet provider and roster reachability

Managed `Gossip` constructs `Lookout` as provider slot five in its exact
12-provider order. A successful Poet provider draw forwards the original Poet
Character as `charRef` to virtual `Lookout.GetInfo` or
`Lookout.GetBluffInfo`, and returns the resulting `ActedInfo` unchanged.
Consequently the actor-exclusion and self-fallback rules are relative to the
physical Poet, not to a synthetic Medium Character. The exact one-reference
result is also preserved through Poet.

The direct public Medium asset is independently present in the normal roster,
so both direct Medium and Poet/Medium are shipped reachable clue surfaces.
Constructing a provider object does not add another board card and does not run
a Start action. Medium has no Start branch in either role action body.

## Current bridge and Scenario limitations

Current direct Medium observations use exact provenance:

```json
{
  "good_position": 4,
  "good_role": "Scout",
  "medium_variant": "public_current"
}
```

Poet/Medium uses the same role payload plus exact provider provenance:

```json
{
  "copied_role": "Medium",
  "good_position": 4,
  "good_role": "Scout",
  "poet_variant": "public_current"
}
```

Live ingestion requires an in-board actor and target, the exact canonical
two-line sentence, and the newest acted event with exactly the same one target
reference. It distinguishes the Drunk `actually a` form from every `real`
form. Extra text, same-line approximations, stale or mismatched references,
unknown roles, and malformed provenance fail closed. Unmarked fixture builders
remain on the legacy path.

The current Rust validator treats each exact observation as a set of compatible
supports rather than pretending every native runtime field is serialized. It
tracks required and forbidden anonymous Wretch placements, a consistent Spy
register-as identity, and a consistent raw-bluff role per physical position
across multiple Medium observations. It also enforces native self-selection:
truthful self requires every other board occurrence to register Evil, while
bluff self requires every other occurrence to be proved raw-bluff-null.

When the current live chronology marker certifies reveal order, a non-self
bluff holder must precede the Medium actor in that order. Archived or manual
orders remain conservative because they cannot prove when the upstream raw
bluff was acquired.

Several exact native state surfaces remain absent from `Scenario`:

- there is no general per-position `registerAs` CharacterData identity;
- there is no stored raw `Character.bluff` pointer or bluff CharacterData;
- delayed-Reveal continuation creation and resume order are not fully traced;
- current-data traces do not by themselves prove which selector first supplied
  a persistent bluff;
- Spy's cached Villager identity is represented as a consistency candidate,
  not a first-class scenario field; and
- mover and Baker histories can require conservative role or holder support
  even when final current data is known.

These limits are why the exact validator must admit mover-history branches
rather than deriving bluff eligibility only from final runtime alignment or
`current_data_role_at`. A future full ordered lifecycle trace should promote
register-as and raw bluff data into one authority and then narrow these
possibilities.

## Corpus and compatibility implications

A deterministic scan of the 426 checked-in `tests/cases_v2` fixtures finds:

- Medium appears in a deck pool in 128 fixtures;
- 121 direct apparent Medium observations occur across 106 fixtures;
- all 121 direct observations have `good_position` plus `good_role`;
- eight Poet/Medium observations occur across eight fixtures and also have
  complete target-plus-role payloads;
- all 129 observations have blank `info_text` and no current provenance marker;
- no historical direct or Poet observation selects its own actor;
- 35 direct sources and two Poet sources are recorded at known true-Evil
  physical positions; and
- 51 direct targets and two Poet targets are recorded at known true-Evil
  physical positions.

These archived records exercise the legacy broad truth/lie relationship and a
large role-label surface. They do not independently prove exact newline text,
registerAs-first labels, raw-bluff identity, one-reference acted history,
actor fallback, empty-pool failure, delayed-Reveal chronology, or the current
fail-closed schema. Current bridge and Rust tests cover those audited surfaces
without retroactively reinterpreting the archived data.

Reconstruction and tooling should therefore:

- bind public Medium to managed `Lookout`;
- preserve the full physical board, list order, and occurrence multiplicity;
- filter truth by registered Good and label through registerAs-or-dataRef;
- remove the actor from truth only when the pool contains more than one entry;
- build bluff from live raw bluff pointers, non-self first, with self only as
  the empty-first-pool fallback;
- preserve exact `real` versus Drunk `actually` text and one reference;
- make exactly one integer-index draw per successful result;
- treat empty candidate pools as failure rather than a sentinel; and
- preserve unmarked observations on their compatibility path.

## Target union accounting

Seven Medium target memberships are exact managed-definition overlaps with the
previous 30 target sets:

- `Lookout..ctor`;
- `ActedInfo..ctor`;
- `Characters.FilterAlignmentCharacters(Character)`;
- `Character.GetRegisterAlignment`;
- `Character.GetCharacterData`;
- `CharacterStatuses.Contains`; and
- `UnityEngine.Random.Range(Int32, Int32)`.

The other seven Lookout identities are new selected FunctionDefinitions.
`Lookout.Act` uses a native body already present in the union, so those seven
definitions add only six unique native RVAs. This target therefore advances
the deterministic union to 31 checked sets, 654 memberships, 419 distinct
selected FunctionDefinitions, and 360 unique native RVAs. The rebuilt GDT has
151,544 datatypes.

## Remaining uncertainty

- Unity's global PRNG implementation and statistical quality are outside this
  target; the boundary proves one integer-index request over the retained pool.
- Native empty/null failure paths are identified, but their exact managed
  exception class, UI presentation, and recovery behavior are not reconstructed.
- The native lifecycle proves first-live-bluff persistence, but sibling
  delayed-Reveal continuation resume order is dynamically unresolved.
- The executable-reference scan rules out ordinary direct calls to the
  standalone achievement helper, not reflection or malformed invocation.
- The checked corpus carries no exact current text/reference history or self
  result, so those surfaces depend on build-pinned native and clean-room tests.
- Current scenario support is conservative around raw bluff and register-as
  histories; it does not yet reconstruct native probability weights over those
  hidden histories.

This checkpoint closes the shipped Medium clue-generation and execution-side
effect boundary for both direct roster use and retained Poet-provider use. It
does not claim that every remaining role or every delayed-Reveal composition is
fully reconstructed.
