# Gameplay role: Baker (managed `Baker`)

Build: `f530404b0f3f_807de4a83df4`

Evidence status: **native-static** for all methods declared by the shipped
`Baker` role, its runtime-data record, Baker achievement bookkeeping, and the
click, reveal, dispatch, filtering, replacement, and acted-history helpers
needed to close the public behavior. Serialized asset evidence binds the
public Baker to this managed type and fixes the authored contract. Native
bodies and decompiler output remain outside the repository.

The checked target set is
[`reverse_engineering/targets/gameplay_role_baker.json`](../../targets/gameplay_role_baker.json).
Its read-only baseline and typed Ghidra exports each complete at 36/36
functions with no failures. The body-free
[`quality report`](../../reports/f530404b0f3f_807de4a83df4_typed_quality_gameplay_role_baker.json)
passes its regression check: unresolved-type tokens fall from 261 to 71, raw
field-offset accesses from 396 to 85, placeholder parameter tokens from 360 to
zero, and indirect-call patterns from 24 to four. Both error and warning marker
counts remain unchanged. The target includes `Character.OnClick` because the
synchronous click-to-Day edge is part of the observable Baker rule, rather
than assuming that a user flip eventually invokes the role.

## Public asset binding and authored contract

The shipped `sharedassets0.assets` `CharacterData` at path ID `21611` is named
`Baker`, has `characterId` `Baker_22847064`, and binds its SerializeReference
role to managed `Baker` in `Assembly-CSharp`. Its raw object SHA-256 is
`4CB21DD7D4DDBEA914D8013E507F7C4BE320EED2EB634C5CE8A5F8152DB07137`.
It is a Good Villager (`characterType == 10`, `startingAlignment == 10`), has
`abilityUsage == 0` (`Once`), is bluffable, is not usually disguised, and has
no picker. Its roguelike values are 10 points, a 1.0 multiplier, and zero
income. Its bundled-character, flavor-hint, additional-status, tag, and
conditional-appearance lists are empty, as is its achievement list.

The exact public description is:

```text
<b>Reveal:</b>
1 random Unrevealed Good Villager becomes a Baker.

Learn which Villager I was.
```

The exact authored lying variant is:

```text
I say that I was a random Villager role.
```

The flavor text is:

```text
"No one knows how it works.
They just wake up baking."
```

The managed `get_Description` string is the older single-line text
`On Reveal: 1 random unrevealed Villager becomes baker.`. The serialized
public description above is the player-facing current contract.

`Baker` is TypeDefIndex `5872`, has no fields, and declares exactly 11
methods. `BakerRuntimeData` is TypeDefIndex `5496`; its one field is the
managed string `charName` at instance offset `0x10`. Its constructor first
initializes the field to the empty string and then assigns its supplied value.

## Audited boundary

| Group | Methods | Observable purpose |
| --- | ---: | --- |
| `Baker` | 11 | Complete role, real/lying clues, conversion, and role-level achievement hook |
| `BakerRuntimeData` | 1 | Preserved previous-role name |
| `BakersAchivementHelper` | 3 | Append-only counter, reset, and allocation |
| `Character` and its coroutines | 10 | User click, state transition, dispatch, replacement, internal reveal, and history append |
| `RevealCard` | 3 | Synchronous Day/OnReveal ordering and visual completion closures |
| `Gameplay` | 2 | Reveal accounting and script Villager pool |
| Status, filters, lookup, and acted record | 6 | Raw status gates, ordered conversion pool, Baker-data lookup, and output shape |

The target keeps exact metadata identities, signatures, and RVAs for all 36
entries. `Baker..ctor` is an ABI-compatible empty constructor at `0x357920`
whose native body is shared by 3,052 managed identities; the target therefore
applies the canonical `Dreamer..ctor` prototype. Every other target entry has
its own distinct RVA, and all 36 target RVAs are distinct from one another.

## Synchronous user-reveal chronology

An ordinary allowed click is the only normal public path that starts a Baker
conversion. The relevant calls and callbacks are synchronous:

1. `Character.OnClick` verifies the reveal quota and exact Hidden state.
2. It writes `prevState = Hidden` and `state = Alive` directly.
3. It invokes the card's state-change delegate before returning.
4. The registered `RevealCard.Reveal` callback calls
   `Character.Act(Day)` (`ETriggerPhase == 30`).
5. Only after that action returns does it call `Character.OnReveal`.
6. It then starts the 0.2-second reveal tween; the two captured closures only
   advance the presentation from Revealing to Revealed and complete the
   animation.

Thus the entire Baker clue, target draw, runtime-data write, and replacement
occur before `Character.OnReveal`, before the animation delay, and before a
subsequent serialized automation click. `Gameplay.OnCharacterReveal` performs
reveal accounting after the Baker action. The source is already Alive while
its Day action runs.

This also settles the historical reveal-order ambiguity. Baker is not seeded
at board Start. Old observations recorded before the verified-first-click fix
could store click attempts rather than successful state transitions. In the
known `asc77_v6` shape, an actual order such as
`[2,3,4,5,6,7,8,9,10,1]` permits original Baker #6 to replace hidden #9 and
then #9 to replace the still-hidden flaked #1, with #1's retry last. A stale
recorded `[1..10]` does not imply a pre-Start Baker chain.

`Character.InitWithNoReset` also schedules an internal reveal for a newly
converted target, but that is a different path. `DelayReveal.MoveNext` clones
the new Baker role before its first 0.3-second yield; after resumption,
`Character.Reveal` computes setup state and dispatches Init (`3`) and
AfterRoundStart (`7`). Baker ignores both phases. This internal reveal does
not run `OnClick`, change public Hidden state to Alive, call `OnReveal`, or add
a reveal-order entry. The converted hidden target extends the chain only when
the user later reveals it.

## Real Day action and exact prior-role clue

`Baker.Act` is a clean no-op for every trigger except Day. At Day it performs
these steps in order:

1. append the physical actor to the Baker achievement helper;
2. derive and emit its previous-role clue; and
3. unless raw `BrokenAbility` (`35`) is present, run `CreateNewBaker`.

The previous-role derivation is exact:

- when `charRef.runtimeData` is non-null, it must cast to
  `BakerRuntimeData`, and `charName` is used as supplied;
- when runtime data is null and raw `AlteredCharacter` (`70`) is present, the
  synthetic previous name is `Baker`;
- when runtime data is null and status 70 is absent, the previous name is
  empty.

An incompatible non-null `RuntimeCharacterData` subtype follows the IL2CPP
invalid-cast failure path. The real action has already appended its achievement
entry, but it emits no `ActedInfo` and never reaches conversion. A null runtime
record is not an error.

`ConjourInfo` produces exactly one of three forms:

```text
I am the original Baker
I was an <name>
I was a <name>
```

The article test examines only the first UTF-16 character and recognizes
exactly uppercase `A`, `E`, `I`, `O`, or `U`. It is case-sensitive and does not
apply a linguistic or whitespace normalization. Empty or null selects the
original-Baker text. The emitted record is a fresh
`ActedInfo(description, null)` with no referenced characters.

`ShowMyPreviousRole` duplicates this clue path but does not perform the
achievement or conversion steps. `GetInfo` returns a fresh empty-description,
null-reference record. They are retained because they are declared public role
surface even though the ordinary Day body does not call them as separate
managed methods.

## Lying Day action and random false history

`Baker.BluffAct` is also Day-only. It calls
`ShowMyPreviousRoleLying` first, then appends the achievement entry, and then
calls `CreateNewBaker` only if raw `WorkingAbility` (`38`) is present. This
ordering differs from the real action: a failing lying clue prevents both the
achievement append and conversion.

The lying clue first scans `Gameplay.CurrentCharacters` in physical order. It
skips only entries whose exact current state is Hidden, asks every other entry
for `GetCharacterBluffIfAble()`, and records only whether at least one returned
apparent role has exact runtime type `Baker`. It does not retain that card or
use it as a random candidate. On an ordinary user reveal, the acting Baker
surface is already Alive and sees itself as an apparent Baker, so this Boolean
is true.

When that Boolean is true **or** the actor's runtime data is null, the method:

1. copies the current script `CharacterData` list for type Villager, preserving
   asset order and one occurrence per list entry;
2. if runtime data is non-null, casts it to `BakerRuntimeData`, finds the first
   Villager asset whose `characterName` equals stored `charName`, and removes
   that exact first asset when found;
3. if runtime data is null, removes nothing;
4. draws one uniform `Random.Range(0, count)` index; and
5. formats that chosen asset name with the same `ConjourInfo` article rule.

There is no comparison against the actor's real physical `dataRef`, no demand
that the false name differ from that real role, no board-presence requirement,
no uniqueness normalization, and no retry. The only ordinary exclusion is the
first script Villager asset whose name matches a non-null saved Baker history.
If the saved name is an Outcast, Minion, Demon, absent asset, or mismatched
string, nothing is removed. If a matching removal empties the pool, the later
random/index operation fails rather than returning a clean no-info result.

The unusual branch in which no non-Hidden apparent Baker exists while runtime
data is non-null skips both the cast and random pool and formats the empty name,
thereby saying `I am the original Baker`. This is a native off-surface edge;
the already-Alive ordinary user-reveal actor makes it unreachable in the
normal Day path. An incompatible non-null runtime record on the ordinary path
fails at the cast before output and, because of BluffAct ordering, before its
achievement append.

`GetBluffInfo` is the same empty-description, null-reference shell as
`GetInfo`. The exact random rule lives in `ShowMyPreviousRoleLying`, not in
that shell.

## Conversion candidate pool and draw

`CreateNewBaker` copies `Gameplay.CurrentCharacters` and applies these filters
in this order, with no hidden predicates between them:

1. apparent type is Villager;
2. apparent alignment is Good;
3. exact current state is Hidden; and
4. remove the first exact occurrence Unity-equal to the source actor.

The apparent-type filter reads a Unity-live `registerAs` first and otherwise
falls back to real `dataRef`. The alignment filter likewise reads
`registerAs.startingAlignment` when live and otherwise the physical runtime
alignment. Neither filter consults the displayed `bluff`, `bluffRole`,
corruption, truth, status source, resistance, runtime-data class, actual role
object, acted history, killed flags, or public reveal Boolean.

This distinction closes several tempting false inclusions in current setup:

| Hidden physical identity displaying a Villager | Relevant stored surface | Baker candidate result |
| --- | --- | --- |
| Puppeteer-created Puppet | saved Villager is `bluff`; `registerAs` is null | Excluded by real Puppet Minion/Evil fallback |
| Clean or Corrupted Doppelganger | copied role is `bluff`; base `registerAs` is null | Excluded by real Outcast type fallback |
| Drunk | disguise is `bluff`; base `registerAs` is null | Excluded by real Outcast type fallback |
| Ordinary Minion/Demon with Villager bluff | disguise is `bluff`; base `registerAs` is null | Excluded by real non-Villager/Evil fallback |
| Special card with live Villager/Good `registerAs` | explicit register-as record | Included if exact state is Hidden |

The filtered lists are fresh lists that preserve source order, duplicate
occurrences, and malformed-entry failure behavior. Removing the source happens
after all three filters and removes at most its first exact occurrence. On an
ordinary reveal the source is already Alive and therefore absent before that
removal; the step matters only on programmatic/off-surface calls.

An empty pool is a clean no-op. Otherwise Baker draws exactly one uniform
`Random.Range(0, count)` index with no reroll. There is no role-identity
deduplication, so duplicated physical list entries would weight the draw and
same-role Villager identities remain legal.

## Replacement data and delayed setup

After selecting a physical target, Baker resolves the Baker `CharacterData`
by creating a `Baker` role object and asking
`CharactersHelper.GetCharacterDataOfRole` for the first current/all-character
record whose embedded role has exact runtime type `Baker`. The public Baker
asset supplies that record. A missing match is not converted into a clean
no-op; subsequent use reaches the native failure path.

The mutation order is critical:

1. read the target's **real current `dataRef.characterName`**, not its
   `registerAs`, displayed bluff, or selected apparent identity;
2. construct `BakerRuntimeData(realName)`;
3. assign it to `target.runtimeData`; and
4. call `target.InitWithNoReset(bakerData, -100)`.

`InitWithNoReset` hides acted presentation, clears native acted history and the
Start latch, destroys a dead prefab when present, clears the current bluff and
revealed flag, assigns Baker data, preserves the physical ID for the `-100`
sentinel, snapshots the old state into `prevState`, writes state Hidden, invokes
the state callback, refreshes the view, and starts delayed internal Reveal.

It does not clear active statuses, resistances, runtime alignment,
`runtimeData`, `bluffRole`, or the current `registerAs` field. Therefore the
newly written `BakerRuntimeData` survives replacement, as do status gates and
alignment. The later internal `Character.Reveal` recomputes `registerAs` and
ordinary bluff presentation; Baker itself has no register-as override.

## Truth dispatch, status composition, and chain reachability

The surrounding `Character.Act` dispatch matrix is:

- a truthful physical actor calls real-role `Act` and bluff-role `Act`;
- a lying non-Evil actor calls real-role `BluffAct` and bluff-role `BluffAct`;
- a lying runtime-Evil actor with a non-null bluff calls real-role `Act` and
  bluff-role `BluffAct`.

Applied to reachable Baker surfaces:

| Baker surface | Day clue and conversion |
| --- | --- |
| Clean real public or converted Good Baker | Real clue; converts unless Broken |
| Corrupted real/descendant Good Baker | Random lying clue; no conversion without WorkingAbility |
| Drunk or corrupted Doppelganger with Baker bluff | Random lying clue; no conversion without WorkingAbility |
| Ordinary Evil with Baker bluff | Bluff Baker lies; no conversion without WorkingAbility |
| Clean HealthyBluff Doppelganger with Baker bluff | Real Baker path; null runtime says original; can convert |
| Puppeteer-created Puppet whose saved bluff is Baker | Real Baker path; null runtime plus AlteredCharacter says previous Baker; BrokenAbility blocks conversion |
| Shaman-copied Baker on a clean, null-runtime Good destination | Real clue says original; can convert |
| Shaman-copied Baker preserving incompatible non-null runtime data | Normal Day cast fails before clue/conversion |

The current core CharacterData assets author no additional status 38, and the
audited current setup/status producers do not add `WorkingAbility`. Therefore
none of the ordinary corrupted natural/descendant Baker, Drunk, corrupted
Doppelganger, ordinary Evil disguise, or Shaman destination created by current
setup passes BluffAct's conversion gate. `WorkingAbility` remains a real native
gate: a future, debug, or externally injected status 38 would enable that
lying actor to convert. Solver models should preserve this conservative escape
hatch rather than erase the gate from the rule.

`BrokenAbility` is produced by Puppet Start and Puppeteer conversion.
`AlteredCharacter` is added by Puppeteer conversion, alongside HealthyBluff,
BrokenAbility, and MessedUpByEvil; ordinary Shaman does not add it. Shaman adds
only MessedUpByEvil around its role overwrite and `InitWithNoReset` itself adds
no status. A Shaman-copied Baker thus normally says original when its preserved
runtime data is null. If its prior data is already `BakerRuntimeData`, that
history works; preserved `AlchemistRuntimeData`, `EnlightenedRuntimeData`, or
another incompatible subtype fails the cast.

Puppeteer can convert a real Baker neighbour into Puppet and preserve the old
Baker as its displayed bluff. This is how an AlteredCharacter Baker surface is
reachable; Baker conversion itself does not select the hidden Puppet through
its mere bluff. The resulting truthful bluff action synthesizes previous name
`Baker` from status 70 and BrokenAbility stops another descendant.

## Small boards, duplicates, and deterministic order

- With one physical card, an ordinary source reveal leaves no Hidden target,
  so conversion is a clean no-op.
- If an off-surface one-card source were still Hidden when conversion ran, the
  post-filter source removal also leaves an empty pool.
- With two or more cards, every independently eligible exact list occurrence
  participates; there is no adjacency rule.
- Physical list order is preserved through filtering and only the final random
  index chooses among entries.
- Several physical Baker cards may each act and convert. There is no global
  once-per-asset latch.
- A converted target cannot extend the chain during internal setup. Actual
  successful user-reveal order, including retried swallowed clicks, determines
  the chronology.
- The lying role-name pool is the script Villager **asset** list, not the
  physical board. Board duplicates neither add nor weight false role names.

## Achievement bookkeeping

`Baker.CheckAchievementsAndUnlockIfAble` reaches the global
`BakersAchivementHelper.AddCharacter`. The helper appends the physical
`Character` reference without deduplication. Whenever the accumulated count is
at least three, it requests unlock key
`Baker_Halloween_ACHIV_8526`. The same actor can therefore contribute more
than once if its Day action is invoked repeatedly, and bluff/copy Baker
surfaces contribute when their action reaches the hook.

Ordering matters for failures: real `Act` appends before its runtime-data cast,
whereas `BluffAct` appends only after the lying clue succeeds. BrokenAbility
and WorkingAbility affect conversion after the append; neither suppresses a
successful clue or achievement count. `Reset` clears the helper list and its
constructor creates a fresh empty list. The helper has no gameplay mutation
beyond list bookkeeping and achievement unlock requests.

## Solver and live-observation consequences

- A real clue with a saved name identifies the target's real `dataRef` at the
  moment that physical card was converted. It does not identify a displayed
  bluff or register-as role.
- `I am the original Baker` means no usable saved name on the reached clue
  path; it is also the off-surface lying fallback when no visible apparent
  Baker exists. Normal lying user reveals instead randomize from the script
  Villager pool.
- A lying clue is not constrained to differ from the actor's real role. With
  non-null Baker history, only the first matching saved Villager asset is
  excluded; with null history, nothing is excluded.
- Normal public Baker chaining is reveal-time and synchronous. Do not preseed
  Baker descendants at Start, and do not infer chronology from pre-fix click-
  attempt logs without accounting for failed first clicks.
- Corrupted Baker surfaces do not currently extend the chain. Keep an explicit
  WorkingAbility override in any general native model, because the role checks
  the raw status even though current audited setup has no producer.
- Native `actedInfos` stores a fresh record with null character references for
  each successful clue. `savedAct` should be corroborated against the latest
  current-initialization history; replacement clears acted history.

## Bounded unknowns

The role fixes per-click synchronous ordering and uniform index selection, but
Unity's random seed/state is outside this slice, so a particular eligible
target or false Villager name cannot be predicted statically. The global
absence of a current WorkingAbility producer is supported by the audited core
asset/status setup boundary, not by a semantic promise in Baker itself; status
38 must remain modeled if another system or later build can inject it.
