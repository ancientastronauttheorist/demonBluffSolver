# Gameplay role: Bounty Hunter (managed `BountyHunter`)

Build: `f530404b0f3f_807de4a83df4`

Evidence status: **metadata** for the public `CharacterData` binding and
serialized Start placement, **native-static** for every method declared by
managed `BountyHunter`, the registered-alignment filter, integer selection,
exact clue construction, and Day callback boundary, and **behavioral** for the
archived Poet-provider corpus and current bridge contract. Native bodies and
decompiler output remain outside the repository.

The checked target set is
[`reverse_engineering/targets/gameplay_role_bounty_hunter.json`](../../targets/gameplay_role_bounty_hunter.json).
It selects 12 managed FunctionDefinitions at 12 distinct target-local native
RVAs. Its read-only baseline and typed Ghidra exports each complete at 12/12
functions with no failures. The body-free
[`quality report`](../../reports/f530404b0f3f_807de4a83df4_typed_quality_gameplay_role_bounty_hunter.json)
passes its regression check: unresolved-type tokens fall from 68 to 32, raw
field-offset accesses from 48 to 22, raw integer type tokens from 32 to eight,
placeholder parameter tokens from 54 to zero, and indirect-call patterns from
four to zero. Both exports retain three decompiler-error and 18 warning
markers.

## Public asset binding and dormant direct identity

The shipped `sharedassets0.assets` `CharacterData` at path ID `21634` is named
`Bounty Hunter`, has `characterId` `Bounty Hunter_39284184`, and binds its
managed-reference role to exact `BountyHunter` at TypeDefIndex `5871` in
`Assembly-CSharp`. The raw object is 484 bytes at file offset `23,741,552` and
has SHA-256
`A912DB6B49377C8067D28210249E7D4085FD10312CABDEFE412A6B2D847BB5C0`.
The containing asset has SHA-256
`E0D239C03FA47EC138F9C2797779E83B65AEF41F30C5AD84D4CA76767A93E967`.

Its authored public text is:

```text
Game Start:
1 Villager becomes Evil

Learn 1 Evil character.
```

The managed description getter independently returns older wording:

```text
[1 Villager becomes Evil]. Learn which character is Evil.
```

The public asset appears at zero-based index 11 in the serialized ordered
Start list, after Alchemist and before Puppet:

```text
... -> Shaman -> Alchemist -> Bounty Hunter -> Puppet -> Rambler -> Lilis
```

The same asset is absent from the normal shipped Standard and Ascension
candidate rosters. Its physical-card Start behavior is therefore dormant in
those ordinary modes even though the asset remains in `startGameActOrder`.
This is separate from Poet: managed `Gossip` always constructs a fresh
`BountyHunter` role object as provider slot four and calls its information
methods, but does not run that provider object's Start action. The executable
identity is also distinct from public Hunter, which binds managed `Tracker`.

The asset and normal-roster result prove the current shipped configuration.
They do not prove that no other mode, future roster, addressable, or malformed
runtime list can instantiate a physical Bounty Hunter card.

## Audited boundary and shared bodies

| Group | Methods | Observable purpose |
| --- | ---: | --- |
| `BountyHunter` | 8 | Description, Start mutation, truth/bluff generation, exact text, Day dispatch, and construction |
| `ActedInfo` | 1 | Result construction with a null Character-reference list |
| `Characters` | 1 | Registered-alignment filtering over a supplied Character list |
| `Character` | 1 | Register-as-first alignment projection |
| `UnityEngine.Random` | 1 | One integer-index draw over the retained occurrence pool |

All 12 target memberships are:

| Managed identity | RVA | Boundary purpose |
| --- | ---: | --- |
| `BountyHunter.get_Description` | `0x3B3930` | Older managed description text |
| `BountyHunter.GetInfo` | `0x3B3780` | Registered-Evil draw and truthful result |
| `BountyHunter.Act` | `0x3B3270` | Start mutation and Day truthful callback |
| `BountyHunter.CreateNewEvil` | `0x3B34B0` | Declared but normally unreachable mutation helper |
| `BountyHunter.BluffAct` | `0x3B33E0` | Day-only bluff callback |
| `BountyHunter.GetBluffInfo` | `0x3B35D0` | Registered-Good draw and false result |
| `BountyHunter.ConjourInfo` | `0x3B3440` | Declared but normally unreachable text helper |
| `BountyHunter..ctor` | `0x357920` | Fieldless role construction and Poet-provider identity |
| `ActedInfo..ctor` | `0x35D5D0` | Exact text plus null reference-list storage |
| `Characters.FilterAlignmentCharacters(Character)` | `0x36A030` | Ordered registered-alignment occurrence filter |
| `Character.GetRegisterAlignment` | `0x365030` | Live register-as or runtime alignment |
| `UnityEngine.Random.Range(Int32, Int32)` | `0x1C86600` | Inclusive-lower, exclusive-upper index selection |

`BountyHunter.BluffAct` uses the broad action body already exported under the
ABI-compatible Witness prototype. `BountyHunter..ctor` uses the broad
fieldless-role constructor body already exported under Dreamer. The integer
`Random.Range` overload is canonicalized separately from the float overload.
Those shared bodies do not change the exact managed identities selected by the
target.

There is no Bounty Hunter achievement helper, unlock call, subscription, or
other achievement side effect in the declared role boundary. The clue's
selected Character is likewise not stored as an `ActedInfo` reference.

## Registered alignment is the selection surface

Every Start, truth, and bluff candidate filter in this role uses
`Character.GetRegisterAlignment`:

```text
live registerAs != null ? registerAs.startingAlignment : Character.alignment
```

The filter does not use current `dataRef`, public apparent role, bluff role,
corruption, status, reveal state, death state, Character type, or the role's
authored faction. It preserves the input list's order and occurrence
multiplicity and does not deduplicate physical references.

Consequences include:

- an ordinary runtime-Evil Character with no register-as is registered Evil;
- a natural Wretch with live Evil register-as is registered Evil even while
  its physical runtime alignment remains Good;
- a runtime-Evil body with live Good register-as remains registered Good;
- moved current Evil data on a runtime-Good body remains registered Good when
  there is no Evil register-as;
- dead, hidden, and revealed Characters remain candidates; and
- the acting Character is not excluded from any pool.

This same projection is used at each call time. Bounty Hunter does not cache a
candidate set or an alignment result between Start and Day.

## Start mutation

`BountyHunter.Act` recognizes `ETriggerPhase.Start == 5`. Its reachable Start
branch contains the complete mutation sequence directly rather than calling
the declared `CreateNewEvil` method:

1. make a fresh copy of the complete physical
   `Gameplay.CurrentCharacters` list;
2. filter that copy to occurrences whose registered alignment is Good (`10`);
3. call integer `Random.Range(0, Count)` exactly once;
4. index the retained ordered list with that result; and
5. assign the selected physical Character's runtime `alignment` field to Evil
   (`20`).

Despite the authored “1 Villager” sentence, the executable filter is not a
Villager-type or current-role filter. Any registered-Good board occurrence is
eligible. The actor itself is eligible, so an ordinary physical Bounty Hunter
can select and convert itself.

The write changes only physical runtime alignment. It does not reinitialize the
Character or replace its current data, bluff, register-as, state, statuses,
resistance, runtime data, acted history, or presentation. If the selected body
has a live Good `registerAs`, the runtime field becomes Evil while subsequent
registered-alignment queries can still return Good. If it was already runtime
Evil but registered Good, the write is a behavioral no-op.

There is no retry, fallback, or sentinel. An empty registered-Good list still
reaches the integer draw and then indexed-list failure. Null global/list or
selected-Character dependencies likewise follow native failure paths rather
than yielding a clue or skipping the mutation.

## Truthful Day result

`BountyHunter.GetInfo` ignores its `charRef` argument. Each invocation:

1. copies the complete current physical Character list;
2. filters it to registered-Evil (`20`) occurrences;
3. makes exactly one integer `Random.Range(0, Count)` draw;
4. retrieves that occurrence;
5. formats the selected Character's displayed numeric `id`; and
6. constructs a fresh `ActedInfo` with the exact text and a null Character
   reference list.

The exact public text is:

```text
#{id}
is Evil
```

There is no punctuation, leading space, or reference to the selected Character
inside `ActedInfo.characters`. Native stores null, not a one-element list and
not an empty list object. Live ingestion observes that surface as exactly zero
references.

The selected Character need not be the body converted at Start. Truth samples
uniformly by retained list occurrence from every Character registered Evil at
the time of the result. Self, dead, hidden, and revealed targets remain legal.
An empty registered-Evil pool has the same draw-then-index failure shape as the
empty Start pool; there is no no-Evils sentence.

## Bluff Day result

`BountyHunter.GetBluffInfo` has the same sequence and output shape, except that
it filters to registered-Good (`10`) occurrences. The selected target is
therefore false on the role's native registered-alignment surface when named
Evil. It still need not be physically runtime Good if a live Good register-as
overrides an Evil runtime alignment.

The bluff path makes one fresh draw per invocation, preserves candidate order
and multiplicity, allows self/dead/hidden/revealed targets, ignores `charRef`,
and constructs `ActedInfo(exactText, null)`. It does not reuse or remove the
truth draw, remember the Start target, or attach a Character reference. An
empty registered-Good pool reaches native failure rather than a sentinel.

## Trigger dispatch and callbacks

`BountyHunter.Act` has two observable trigger branches:

- on Start (`5`), it performs the alignment mutation above without requiring
  an acted-result callback; and
- on Day (`30`), it first tests `onActed`, then makes one virtual `GetInfo`
  call and invokes the non-null callback once with that exact result.

If the Day callback is null, `Act` returns without generating information or
consuming RNG. Other triggers are no-ops.

`BountyHunter.BluffAct` handles only Day. With a non-null callback it makes one
virtual `GetBluffInfo` call and invokes the callback once; with a null callback
or any other trigger it returns without selection. It never runs the Start
mutation.

The separately audited generic `Character.Act`/`Character.RoleAct` boundary
installs the callback and chooses the concrete role's `Act` or `BluffAct`
according to the Character's truth/lying appearance. For Poet, the outer
`Gossip` role instead chooses its own real or bluff provider path and directly
invokes the selected provider's virtual `GetInfo` or `GetBluffInfo`. Because
Bounty Hunter ignores the forwarded Poet `charRef`, a Poet/Bounty result still
samples the whole board and can name the Poet itself.

## Declared helper reachability

`BountyHunter.CreateNewEvil` contains a standalone copy of the registered-Good
selection and runtime-alignment mutation. `BountyHunter.ConjourInfo` contains a
standalone copy of the exact text formatting for a non-null Character. Neither
is called by the reachable role methods: `Act`, `GetInfo`, and `GetBluffInfo`
carry independent inline copies of the corresponding behavior.

A pinned executable-reference audit finds zero executable direct relative-call
or jump references to either helper and exactly one ordinary, non-executable
IL2CPP method-registration pointer for each. They are retained in the target
because they are declared concrete methods and corroborate the behavior, but
they are classified as unreachable through the normal shipped native call
graph. Reflection, tooling, or malformed external invocation is not absolutely
ruled out.

This helper result is separate from the dormant physical role result. The
physical Start branch is executable inside `BountyHunter.Act`; normal
Standard/Ascension simply does not place the public asset in its candidate
rosters. Poet reaches the provider's clue generators but not its Start branch.

## Current bridge and Rust schema

`memory_reader.py` maps managed `BountyHunter` to public `Bounty Hunter` and
keeps it distinct from managed `Tracker`, which maps to public `Hunter`.
Current live ingestion supports Bounty Hunter as the retained Poet provider.
A captured current observation is stored as an apparent Poet with exact text
and payload:

```json
{
  "evil_position": 4,
  "copied_role": "Bounty Hunter",
  "poet_variant": "public_current"
}
```

The bridge accepts only the exact case-sensitive two-line sentence, a positive
in-board displayed ID without leading zeroes, an in-board Poet actor, and the
newest acted event carrying that exact sentence and exactly zero references.
Stale events, absent acted history, a selected-Character reference, altered
case/spacing/newline/punctuation, or an out-of-board ID fail closed. Manual
`card poet <position> bounty_hunter <target>` entry stamps the same exact text
and provenance after board-bound validation.

The Rust current schema requires:

- apparent role exactly Poet;
- exactly the three payload fields shown above;
- exact `poet_variant: "public_current"` and
  `copied_role: "Bounty Hunter"` strings;
- in-board actor and target positions; and
- `CardInfo.info_text` exactly equal to `#{target}\nis Evil`.

Extra, missing, mixed, future, wrong-typed, or role-inconsistent current fields
fail closed. A current-marked card whose apparent role is direct Bounty Hunter
also fails closed: no direct current schema or Start trace is claimed while the
physical role remains absent from normal candidate rosters.

For a truthful current Poet source, the named target must be reachable as
registered Evil. For a lying source, it must be reachable as registered Good.
Rust uses the same stable runtime/current-data separation and registered-
alignment projection used by other current native validators. Explicit Wretch
data registers Evil; an anonymous natural Wretch candidate is admitted only
when one compatible hidden Outcast assignment can place or exclude Wretch at
the named target. Multiple current Poet/Bounty observations also share a joint
required/forbidden anonymous-Wretch consistency check, preventing one hidden
Wretch from satisfying incompatible clues independently.

The validator imposes no self, death, reveal, or visibility restriction on the
target. Generated Puppet and other modeled runtime-Evil positions can satisfy
a truthful result. The check is reachability, not a probability weight or a
reconstruction of which registered-Evil occurrence the native RNG would have
selected.

Unmarked historical Poet/Bounty observations retain the legacy scalar
predicate: a present `evil_position` must agree with the source's broad
truth/lie status, while a missing position remains non-constraining. They are
not retroactively subjected to exact text, reference, provenance, or current
registered-alignment schema checks.

The solver does not model the dormant direct role's Start mutation. If a future
shipped roster enables physical Bounty Hunter, support requires an ordered
Start outcome in scenario generation plus a direct provenance schema; merely
reusing the Poet clue validator would miss the extra runtime-Evil branch.

## Corpus and compatibility implications

A deterministic scan of the 426 checked-in `tests/cases_v2` fixtures finds:

- 18 apparent Poet/Bounty Hunter observations across 18 fixtures;
- all 18 contain complete `copied_role` plus `evil_position` payloads;
- all 18 have blank `info_text` and no `poet_variant` marker;
- three name the Poet actor itself;
- six sources are recorded at known true-Evil physical positions, while four
  named targets are recorded as known true Evil;
- board-size distribution of four seven-card, five eight-card, six nine-card,
  and three ten-card observations;
- zero direct apparent Bounty Hunter cards; and
- zero Bounty Hunter entries in the checked fixture deck pools.

The fixtures do not store acted-reference arrays, so they cannot independently
prove the native null-reference result. They protect the legacy scalar clue
predicate, self-target acceptance, and broad Good/Evil disguises, but do not
exercise exact current text, zero-reference event ingestion, fail-closed
provenance, empty-pool failure, direct Start selection, or the distinction
between runtime and registered alignment after the dormant mutation. Those
surfaces are covered by build-pinned bridge and Rust regressions derived from
the native boundary.

Reconstruction, solver, and live tooling should therefore:

- bind public Bounty Hunter to managed `BountyHunter`, never public Hunter's
  managed `Tracker` identity;
- treat Poet's Bounty Hunter provider as clue-only and never apply its Start
  mutation merely because `Gossip` constructed the provider object;
- select Start and result candidates by registered alignment, not current role
  type or authored “Villager” wording;
- preserve the full board, actor/self eligibility, dead/hidden/revealed
  eligibility, list order, and occurrence multiplicity;
- mutate only physical runtime alignment at Start and preserve register-as and
  current data;
- make one fresh integer-index draw per successful Start or clue generation;
- preserve exact two-line text and native null references;
- treat empty candidate pools as native failure rather than inventing a
  sentinel; and
- keep every unmarked historical observation on its compatibility path.

## Target union accounting

Five Bounty Hunter target memberships are exact managed-definition overlaps
with the previous 29 target sets:

- `BountyHunter..ctor`;
- `ActedInfo..ctor`;
- `Characters.FilterAlignmentCharacters(Character)`;
- `Character.GetRegisterAlignment`; and
- `UnityEngine.Random.Range(Int32, Int32)`.

`BountyHunter.BluffAct` adds a new exact FunctionDefinition on a native body
already present in the union. The remaining six newly selected definitions add
six new native RVAs. This target therefore adds seven distinct selected
FunctionDefinitions and six unique native RVAs.

The deterministic 30-set union contains 640 memberships, 412 distinct selected
FunctionDefinitions, and 354 unique native RVAs. The rebuilt GDT contains
151,537 datatypes.

## Remaining uncertainty

- Unity's global PRNG implementation and statistical quality are outside this
  target; the boundary proves one integer-index request over each retained
  ordered pool.
- Native empty/null failure paths are identified, but their exact managed
  exception class, UI presentation, and recovery behavior are not
  reconstructed.
- The normal Standard/Ascension roster absence is build-pinned. Other modes,
  future assets, runtime list replacement, or external invocation could make a
  physical Bounty Hunter reachable.
- The executable-reference scan rules out ordinary direct relative calls to
  `CreateNewEvil` and `ConjourInfo`, not every reflective or malformed call
  mechanism.
- The current solver models Poet/Bounty clue reachability but not RNG weights
  or the dormant direct Start mutation.
- The checked corpus has no direct Bounty Hunter card, current provenance,
  exact text, acted-reference capture, or empty candidate pool. A paired live
  direct-role capture would require a mode or build that actually exposes the
  asset.

This checkpoint closes the shipped Bounty Hunter Start and clue-generation
boundary while preserving the distinction between its dormant direct asset
and its active Poet-provider use. It does not claim that every remaining Poet
provider or every gameplay role is decompiled.
