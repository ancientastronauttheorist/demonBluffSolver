# Gameplay role: Bombardier

Build: `f530404b0f3f_807de4a83df4`

Evidence status: **native-static** for all five methods declared by the shipped
managed `Saint` role and the exact dispatch, death, bookkeeping, and terminal
helpers needed to close Bombardier's loss rule. Serialized asset evidence fixes
the public binding and authored contract. Native bodies and decompiler output
remain outside the repository.

The checked target set is
[`reverse_engineering/targets/gameplay_role_bombardier.json`](../../targets/gameplay_role_bombardier.json).
Its read-only baseline and typed Ghidra exports each complete at 23/23 functions
with no failures. The body-free
[`quality report`](../../reports/f530404b0f3f_807de4a83df4_typed_quality_gameplay_role_bombardier.json)
passes its regression check: unresolved-type tokens fall from 134 to 39, raw
field-offset accesses from 245 to 98, placeholder parameter tokens from 172 to
zero, and indirect-call patterns from 27 to 11. Both exports retain three
decompiler-error markers and 21 warning markers.

## Public asset binding and authored contract

The shipped `sharedassets0.assets` `CharacterData` at path ID `21603` is named
`Bombardier`, has `characterId` `Bombardier_79093372`, and binds its
SerializeReference role to managed `Saint` at TypeDefIndex `5906` in
`Assembly-CSharp`. Its raw object SHA-256 is
`7904D8F6DF56CA5FA9F2A22CF4B7C042DC94957B1D2F56C8A7C252B30A025545`.

The card is a Good Outcast (`characterType == 20`,
`startingAlignment == 10`), is bluffable, is not usually disguised, and has
`picking == false`. Its `abilityUsage` is enum value zero (`Once`), but the
non-picking card has no active picker or target. It has no serialized statuses,
tags, appearance conditions, or achievements. The exact public description is:

```text
Lose if you Execute me.
```

The managed description getter instead returns:

```text
Lose if you Kill me
```

The managed wording is the accurate native boundary: any qualifying non-Demon
death is terminal, not only an ordinary player execution.

A separate shipped CharacterData at path ID `21630` is named `Saint` and embeds
managed `SaintVillager`, not `Saint`. That Villager is not the public
Bombardier binding and does not satisfy the exact Bombardier terminal-type
test. Public display names therefore cannot substitute for managed role
identity here.

## Audited boundary and shared bodies

| Group | Methods | Observable purpose |
| --- | ---: | --- |
| `Saint` | 5 | Complete passive surface, no-op action, and construction |
| `Role` / `ActedInfo` | 4 | Bluff dispatch, default killability/damage, and empty result records |
| `Character` / `CharacterHelper` | 9 | Truth dispatch plus ordinary, forced, and Demon death routes |
| `Gameplay` | 1 | Dead-list publication and terminal-check request |
| `WinConditions` | 4 | Delayed loss predicate, automatic panel, and loss completion |

The 23 target memberships select 23 distinct managed FunctionDefinitions and
23 distinct native RVAs. Five selected identities have native bodies shared
with methods outside this target:

- `Saint.Act` at `0x33ED50` is an all-trigger no-op body with 3,540 aliases and
  uses the established canonical `Recluse.Act` prototype;
- `Saint..ctor` at `0x3CFFF0` is a base-only constructor body with 537 aliases
  and uses the canonical Slayer constructor prototype;
- `Role.GetDamageToYou` at `0x3C4CD0` has 11 aliases;
- `Role.BluffAct` at `0x3C4CA0` has two aliases; and
- `Role.CheckIfCanBeKilled` at `0x3B24C0` has 865 aliases.

`Role.ActOnDied` is inherited by Bombardier and is the ordinary empty base
hook. Its shared empty native body is also folded across managed methods with
different arities, so it is deliberately excluded from this target rather
than applying an ABI-incompatible canonical prototype. The no-op inherited
hook is corroborating managed context; it is not one of this checkpoint's 23
native-static selections.

## Passive surface and inherited dispatch

`Saint.GetInfo` and `Saint.GetBluffInfo` each ignore the actor argument and
return a fresh `ActedInfo` whose description is the empty string and whose
character-reference list is null. Bombardier therefore contributes no passive
clue, picker result, retained reference, local runtime field, reset history, or
achievement state.

`Saint.Act` is an all-trigger no-op. `Role.BluffAct` simply invokes the apparent
role's virtual `Act`, so a corrupted real Bombardier or an Evil character
displaying Bombardier also reaches the same no-op action. Actual truth dispatch
does not manufacture a special lying result.

The inherited defaults return `true` from `Role.CheckIfCanBeKilled` and five
from `Role.GetDamageToYou`. An ordinary real Bombardier can consequently be
killed, and an ordinary mistaken execution follows the usual five-damage
event path before terminal evaluation. `NoDamage` or another current-data
override can change the resource result without changing Bombardier's terminal
identity predicate.

## Ordinary and forced non-Demon deaths

The ordinary execution route begins in `Character.OnClick`, consults the
current real role's virtual killability, changes the card to Dead on success,
and runs the execution/death callbacks and resource-damage path. The inherited
Bombardier death hook has no extra effect.

`Character.ExecuteAndReveal` and `Character.KillAndReveal` are forced routes:
they do not ask `CheckIfCanBeKilled` before killing. Slayer reaches
`KillAndReveal`. `Character.Kill` is likewise a non-Demon death route and does
not set `killedByDemon`. Every successful route publishes the exact physical
card to `Gameplay.ManageKilledCharacter`, which appends it to the dead list and
requests `WinConditions.CheckEndGameConditions`.

Thus all of these successful deaths can trigger Bombardier's automatic loss:

- ordinary player execution;
- forced `ExecuteAndReveal`;
- Slayer's `KillAndReveal`; and
- an ordinary `Character.Kill` call from another source.

The trigger source, apparent role, reveal state, current alignment, corruption,
and HP damage do not narrow the later type test. Duplicate dead-list entries
can cause repeated terminal requests, because condition checks are independent
coroutines rather than a coalesced singleton.

## Demon-kill exemption

`Character.KillByDemon` and its delayed iterator ask the current real role's
killability before a successful night death. On success the native path sets
`killedByDemon = true` before changing state to Dead, publishing the death, and
requesting the terminal check. A protected or already-Dead target does not
complete that death path.

The delayed terminal predicate expressly requires a dead Bombardier whose
`killedByDemon` field is false. A Lilis/Demon night kill is therefore exempt
because of this stored death-path flag, not because of the killer's public
role, the target's alignment, or any corruption/status check. A non-Demon
route which kills the same real Bombardier remains fatal.

## Terminal identity uses current dataRef

After its delay, `WinConditions.DelayedCheckCondition.MoveNext` scans dead-list
entries and follows each dead `Character`'s **current `dataRef`**, then that
CharacterData's SerializeReference `role`. It tests the exact managed runtime
type `Saint` and separately requires `killedByDemon == false`.

This is not a stable/original-card identity test. It does not inspect the live
cloned `Character.role` field, `bluff`, `bluffRole`, `registerAs`, displayed
name, role name, starting/runtime alignment, or status set. The consequences
are exact:

- a real `Saint`/Bombardier hidden under an ordinary bluff remains fatal;
- an Evil card merely **displaying** Bombardier as its bluff is not fatal;
- Drunk and Doppelganger display/register-as copies do not replace `dataRef`
  and therefore do not become terminal Bombardiers;
- Chancellor generation or another genuine replacement whose resulting
  `dataRef` is the public Bombardier asset is fatal after a non-Demon death;
- a Shaman `InitWithNoReset` destination whose current `dataRef` genuinely
  becomes Bombardier is fatal even when the destination preserves runtime Evil
  alignment or prior status/runtime data; and
- the separate `SaintVillager` role fails the exact `Saint` test despite its
  public card being named Saint.

Normal Shaman selection uses an apparent-Villager source, while public
Bombardier is an Outcast, so the ordinary authored board does not normally
offer Bombardier as the copied Shaman source. That eligibility fact does not
change the terminal identity rule if some supported or future composition does
produce a current Bombardier `dataRef`.

The distinction also applies in the other direction: a physical card that
started as Bombardier but whose current `dataRef` was genuinely replaced with a
different CharacterData no longer satisfies the exact test. Physical origin
alone is not retained by this predicate.

## Terminal precedence and output

`CheckEndGameConditions` starts a new delayed coroutine on every request. After
the scaled delay, its first terminal branch is any qualifying non-Demon dead
`Saint`. It calls `AutoLose`, which activates the automatic-loss panel, and
then `Lose`. Only if no such card exists does the predicate test HP at or below
zero and then the raw dead-Evil/current-Evil count comparison.

Bombardier loss therefore has precedence over simultaneous zero HP or an
otherwise complete Evil execution. The usual five damage may have occurred,
but panel selection is still the automatic Bombardier loss. `Lose` reveals all
cards, updates scores, schedules Summary, and finally invokes `OnDied`; it does
not reinterpret the Bombardier identity.

## Typed-union accounting

Seventeen target memberships are exact managed-identity overlaps with the
previous 21 target sets. The target adds all five `Saint` methods and
`WinConditions.AutoLose`, for six new selected FunctionDefinitions. The
`Saint.Act` and constructor bodies already exist under canonical folded-body
identities, so only four native RVAs are new.

The deterministic 22-set union contains 502 memberships, 330 distinct selected
FunctionDefinitions, and 303 unique native RVAs. Its 172 exact membership
overlaps and 27 folded-body differences remain explicit rather than collapsing
managed identities.

## Corpus and reconstruction implications

A deterministic scan of the 426 checked-in `tests/cases_v2` fixtures finds:

- 167 Bombardier deck entries across 167 cases;
- 165 apparent Bombardier cards across 145 cases; and
- 20 notes mentioning Bombardier, four of them explicitly recording a loss.

Reconstruction, solver, and live validation should therefore:

- bind public Bombardier to exact managed `Saint`, never `SaintVillager`;
- emit no role-local clue or active result;
- treat any successful non-Demon death of the current `dataRef.role == Saint`
  identity as an automatic loss, including forced and ordinary kill routes;
- exempt only deaths whose stored `killedByDemon` flag is true;
- evaluate genuine current-data replacements, including Shaman/Chancellor
  composition, separately from ordinary bluff and Drunk/Doppel display copies;
  and
- preserve Bombardier terminal precedence over HP and Evil-count outcomes.

The bounded native checkpoint proves the identity and control-flow rules. It
does not claim that every hypothetical current-data replacement is reachable
from the authored Standard deck; reachability remains the responsibility of
the audited role-selection and deck-construction boundaries that create it.
