# Gameplay roles: Lilis (managed `Striga`) and Knight (managed `Immortal`)

Build: `f530404b0f3f_807de4a83df4`

Evidence status: **native-static** for every method declared by the shipped
Lilis and Knight role classes and for the ordered Start, Night-rule, selection,
delayed-kill, protection, ordinary-execution, Slayer, health, status, death,
and reset helpers needed to close their observable boundary. Serialized asset
evidence establishes both public bindings and Lilis's exact Start-order slot.
Native bodies and decompiler output remain outside the repository.

The checked target set is
[`reverse_engineering/targets/gameplay_roles_lilis_knight.json`](../../targets/gameplay_roles_lilis_knight.json).
Its read-only baseline and typed Ghidra exports each complete at 54/54
functions with no failures. The body-free
[`quality report`](../../reports/f530404b0f3f_807de4a83df4_typed_quality_gameplay_roles_lilis_knight.json)
passes its regression check. The exact aggregate deltas are recorded in that
report rather than reproducing private decompiler output here.

## Public asset bindings and authored text

The shipped `sharedassets0.assets` `CharacterData` at path ID `21591` is named
`Lilis`, has `characterId` `Lillith_90453844`, and binds its SerializeReference
role to managed `Striga` in `Assembly-CSharp`. Its raw object SHA-256 is
`924EC7C600C232F33B2A3B83D34B18ACFF367C155F2E8B75AD998F568AB26721`.
It is an Evil Demon (`characterType == 100`, `startingAlignment == 20`), has
`abilityUsage == 0` (`Once`), and is not bluffable, usually disguised, or
picking. Its roguelike values are 10 points, a 1.0 multiplier, and one income.
Its additional-status, tag, and conditional-appearance lists are empty.
`Striga` is TypeDefIndex `5923`, declares no fields, and declares exactly three
methods.

Lilis's exact authored description is:

```text
<b>At Night:</b>
Kill a random unrevealed character.
Deal 2 damage to you.

I Lie and Disguise.
```

Its exact authored hint is:

```text
Will prioritize killing Good characters first.

Will not kill herself when she is the only card left.
```

The hint is accurate for the normal shipped setup, but both clauses are
implemented by concrete mechanics below: Good is an exclusive first-pass
candidate pool, and self-exclusion comes from an active status rather than an
identity comparison.

The shipped public Knight is `sharedassets0.assets` path ID `21624`, is named
`Knight`, has `characterId` `Knight_47970624`, and binds managed `Immortal` in
`Assembly-CSharp`. Its raw object SHA-256 is
`32B3DF20DBA8C95FDB9F11674AD87AEC6159DDA7B06E00C2716AA0EC49D02896`.
It is a Good Villager (`characterType == 10`, `startingAlignment == 10`), has
`abilityUsage == 0` (`Once`), is bluffable, and is not usually disguised or
picking. Its roguelike values are 10 points, a 1.0 multiplier, and zero income;
the same three authored lists are empty. `Immortal` is TypeDefIndex `5869`,
declares no fields, and declares exactly ten methods.

Knight's exact public description is:

```text
I can't die.
```

Its exact authored hint is:

```text
If I am Executed while Corrupted:
I deal 4 additional damage to you
```

The managed `Immortal.get_Description` and parameterless `GetInfo` surfaces
both omit the public period and return `I can't die`.

The `level0` `Characters` component at path ID `137026` references Lilis path
ID `21591` at zero-based index 14, the final entry in `startGameActOrder`.
Knight is absent from that order. Thus all ordinary Init work and every other
serialized Start role run before Lilis's self-protection Start action.

## Target boundary

### Role-declared methods

| Managed identity | RVA | Boundary role |
| --- | ---: | --- |
| `Striga.GetRules` | `0x3EE7B0` | Four-reveal Night-rule construction |
| `Striga.Act` | `0x3EE6F0` | Start protection and Night kill/damage |
| `Striga.ctor` | `0x3CFFF0` | Folded base-role construction |
| `Immortal.get_Description` | `0x3BC7A0` | Managed description literal |
| `Immortal.GetInfo(Character)` | `0x3BC600` | Empty truthful passive result |
| `Immortal.GetBluffInfo(Character)` | `0x3BC5A0` | Empty bluff passive result |
| `Immortal.Act` | `0x3BC2A0` | Protection bookkeeping and executed damage |
| `Immortal.BluffAct` | `0x3BC3A0` | Identical bluff-side consequences |
| `Immortal.GetInfo()` | `0x3BC660` | Parameterless managed text literal |
| `Immortal.CheckIfCanBeKilled` | `0x3BC4A0` | Exact status/alignment precedence |
| `Immortal.ProtectedKnight` | `0x3BC690` | Protected-Knight achievement bookkeeping |
| `Immortal.CorruptedKnightKilled` | `0x3BC540` | Corrupted-Knight achievement bookkeeping |
| `Immortal.ctor` | `0x357920` | Folded base-role construction |

### Dispatch, Night, selection, and consequence helpers

| Managed identity or group | RVA(s) | Boundary role |
| --- | ---: | --- |
| `Characters.ManageCharacters` | `0x36CE30` | Ordered Start and ordinary duplicate policy |
| `Gameplay.UpdateRules`, `Gameplay.ChangeGameplayState` | `0x3814F0`, `0x37B620` | Rule aggregation and same-state transition guard |
| `Character.Act`, `Character.RoleAct`, `CharacterHelper.CheckLying`, `Role.BluffAct` | `0x3645C0`, `0x368790`, `0x397750`, `0x3C4CA0` | Truth-aware concrete action dispatch |
| `Character.Init` | `0x365A20` | Active-status and per-role-history reset boundary |
| `NightModeRule.ctor`, `Init`, `Remove`, `Revealed` | `0x3E5370`, `0x3E4DE0`, `0x3E5080`, `0x3E52A0` | Reveal counter and Night request lifecycle |
| `NightModeRule.ManageKill`, `ManageKill.MoveNext` | `0x3E5000`, `0x3F0640` | Kill subscriber and its 100 ms no-op completion |
| `NightPhase.ManagePhase`, `StartPhase.MoveNext`, `ReorderList` | `0x383540`, `0x3927C0`, `0x3838D0` | Night coroutine and ordered actor enumeration |
| `Demon.KillHidden`, `Demon.KillRandom` | `0x3D6C30`, `0x3D6E70` | Good-first and fallback victim passes |
| `Characters.FilterAliveCharacters`, `FilterAlignmentCharacters(Character)` | `0x36A240`, `0x36A030` | Non-Dead and registered-alignment filters |
| `Characters.FilterHiddenCharacters`, `FilterCharacterMissingStatus` | `0x36B020`, `0x36A8C0` | Exact Hidden and active-status-60 filters |
| `Characters.GetRandomAliveCharacter` | `0x36C740` | Uniform guarded index draw |
| `CharacterStatuses.AddStatus`, `Contains` | `0x363AA0`, `0x363C40` | Resistance-aware insertion and exact membership |
| `Character.KillByDemon`, `DelayedDemonKill.MoveNext` | `0x365FB0`, `0x3757F0` | Delayed protection check and demon death |
| `Character.OnClick`, `KillProtected`, `Kill` | `0x366270`, `0x366080`, `0x366130` | Ordinary execution/protection/death paths |
| `Character.RevealAllReal`, `KillAndReveal` | `0x367E80`, `0x365F10` | Slayer forced reveal and immunity bypass |
| `Character.GetRegisterAlignment`, `Slayer.CharacterPicked` | `0x365030`, `0x3EC9C0` | Slayer's registered-alignment decision |
| `Role.CheckIfCanBeKilled`, `Doppleganger.CheckIfCanBeKilled` | `0x3B24C0`, `0x3D7090` | Base and displayed-Knight killability composition |
| `PlayerController.ManageResources`, `Role.GetDamageToYou`, `Drunk.GetDamageToYou` | `0x384C20`, `0x3C4CD0`, `0x3DA560` | Base wrong-execution damage and Drunk override |
| `Health.Damage` | `0x382100` | Fixed Lilis/Knight health mutations |

The two role constructors and several base helpers share native bodies with
other managed identities. The manifest retains the exact role-facing metadata
definitions while applying the established canonical prototypes where folded
bodies require one.

## Lilis ordered Start and self-protection

`Characters.ManageCharacters` initializes and publishes every physical card,
runs every Init action, then walks the serialized Start list. Lilis is the last
entry. As an ordinary entry, its scan visits physical cards from highest
displayed ID to lowest, calls Start on the first exact `CharacterData` match,
and stops. Lilis is not one of the three duplicate exceptions.

An ordinary Evil Lilis satisfies `CharacterHelper.CheckLying`. Because Striga
does not override `BluffAct`, the inherited `Role.BluffAct` virtually forwards
back to the same Striga object's concrete `Act`. Truthful dispatch would reach
that body directly, so either path has the same role effect.

On Start (`5`), `Striga.Act` attempts to add `UnkillableByDemon` (`60`) to its
own physical card, using self as source and a null shared target. `AddStatus`
first checks exact resistance 60, then unique-adds the status. The current
shipped role set produces no exact-60 resistance, but the general resistance
edge is real: if such resistance already existed on that physical card, the
status would be absent and no separate self predicate would replace it.

The Start latch is consumed before dispatch. Repeated Start requests on the
same initialized card do not add another status. Same-asset duplicates receive
only the one ordinary ordered Start described above; the other physical Lilis
cards therefore lack normal self-protection.

## Night-rule trigger, repetition, and death persistence

Every `Striga.GetRules` call creates a new one-entry list containing a fresh
`NightModeRule` configured for four reveals. `Gameplay.UpdateRules` removes old
rules, clears the aggregate, and iterates every roster `CharacterData`
occurrence. For each nonempty role result it calls `GetRules` a second time and
adds that second fresh list. Consequently one shipped Lilis occurrence creates
one active rule, and same-data roster duplicates create one independent rule
per occurrence. There is no deduplication by data, role, or native type.

Each rule subscribes to killed-character and revealed-character events. The
reveal handler ignores Summary and an event card whose state is exactly Dead;
all other reveal events increment `currentStep`. At four it invokes the
optional Night callback, requests global Night, then resets its counter to
zero. `Gameplay.ChangeGameplayState` suppresses a request equal to the current
state. Thus duplicate rules all reach four on the same event, but only the
first Day-to-Night request emits the state-change event and launches a Night
coroutine; later same-event Night-to-Night requests are no-ops. The kill
subscriber only awaits 100 ms and completes. It does not increment, reset, or
otherwise mutate the rule.

The rule is tied to the roster, not to a live Striga character. Killing Lilis
does not remove it. It continues requesting Night after each four eligible
reveals until the rules are rebuilt; dead Lilis actors simply do no work when
that Night arrives. `Striga` has no fields, use counter, acted information, or
role-local history. Each later Night repeats the same action while the
physical Lilis remains non-Dead.

`NightPhase.ReorderList` includes every physical card whose current `dataRef`
occurs in the serialized Night order, without a liveness filter. Same-data
Lilis duplicates each occur once, in stable global-list order; Dead duplicates
are included in the actor list but rejected by `Striga.Act`. Each non-Dead
duplicate therefore acts exactly once per actual Night coroutine even though
only one duplicate received ordered Start protection.

Same-data actors are consecutive and the Night coroutine waits 0.4 seconds
before each action. Demon death resolves after 0.45 seconds. Two live duplicate
Lilis cards can therefore select the same still-Hidden victim: the first
delayed callback kills it, the later callback finds it already Dead and aborts,
while both actors' immediate 2 HP charges remain.

## Exact victim pools: hard Good priority, not weighting

On Night (`20`), `Striga.Act` first requires its own state not be Dead. It then
calls `Demon.KillHidden` and immediately deals a fixed 2 HP to the player. The
kill helper starts a delayed coroutine; the health mutation therefore precedes
victim resolution and is not contingent on a successful death.

`Demon.KillHidden` builds the first-pass pool through this exact pipeline:

1. retain state not equal to Dead;
2. retain alignment Good (`10`);
3. retain state exactly Hidden (`5`); and
4. retain cards whose active status list lacks `UnkillableByDemon` (`60`).

The alignment filter uses `registerAs.startingAlignment` when the live
`registerAs` Unity reference is non-null; otherwise it uses the physical
character's runtime alignment. It never reads the displayed `bluff`. This is
best described as **registered/apparent alignment**, not displayed-role
alignment and not necessarily real runtime alignment.

If that Good pool is nonempty, exactly one list index is drawn uniformly and
the Evil/fallback pool has zero probability. “Prioritize Good” is therefore a
hard two-pass rule, not a weight. If the Good pool is empty, `KillRandom`
rebuilds state-not-Dead, exact-Hidden, missing-status-60 filters without any
alignment condition, then uniformly selects exactly one occurrence. The
filters allocate new lists, preserve source order and multiplicity, and do not
exclude the demon reference or any role identity.

Consequences follow directly:

- revealed/Alive and Dead cards are ineligible in both passes;
- a normally initialized lone Lilis is excluded by its status 60, so no victim
  is selected, but the player still loses 2 HP;
- a Lilis whose status insertion was resisted or never received Start can
  select herself in the fallback; duplicate unprotected Lilis cards can select
  themselves or each other;
- an eligible Good card prevents every Evil-aligned fallback selection;
- when no candidate exists at all, the helper returns cleanly and the fixed 2
  HP still applies; and
- target count is at most one per live Lilis action, with no reroll after the
  delayed path rejects the selected target.

The v2 corpus is an observation log rather than complete frame-by-frame state.
For example, a recorded Evil night kill beside a seemingly available Good card
can reflect earlier reveals, a repeated Night, registration state, or lossy
capture. Such a fixture cannot turn the native hard first pass into a weight.

## Delayed killability and successful Lilis death

`Character.KillByDemon` immediately emits the picked-character visual event
and starts a 0.45-second victim coroutine. It does not check killability or
mutate state. After the delay, `DelayedDemonKill.MoveNext`:

1. aborts if the selected card is already Dead;
2. asks the selected card's **current real `dataRef.role`** whether it can be
   killed and aborts when false;
3. preserves the old state, sets `killedByDemon`, and changes state to Dead;
4. attempts `MessedUpByEvil` (`50`) and `KilledByEvil` (`55`), sourced from the
   captured Lilis card;
5. runs Died action and exact real-role `ActOnDied` callbacks; and
6. publishes the normal killed-character event and final UI update.

A protected victim produces no death, no status 50/55, no Died action, no
`OnProtected` action, and no reroll. The earlier Lilis 2 HP is retained. Exact
resistance 50 or 55 can independently reject the corresponding post-death
status, although no current shipped role installs either resistance. Ordinary
Alchemist resistance 10 is irrelevant to these two insertions and to status
60 selection.

Successful demon death does not call `Character.Kill` and never runs
OnExecuted. `killedByDemon` also suppresses the normal wrong-execution resource
charge. A corrupted Knight killed by Lilis therefore does **not** deal Knight's
additional 4 HP; only Lilis's own fixed 2 HP has already been charged.

## Registered alignment compositions

The native alignment source matters for several shipped identities:

| Physical/current identity | Lilis Good pass | Delayed killability |
| --- | --- | --- |
| ordinary Good Knight | Included while Hidden | Immune when clean; killable when Corrupted |
| Drunk displaying Knight | Included from runtime Good; bluff ignored | Real Drunk is killable; no Knight OnExecuted damage |
| Puppet displaying its former Villager | Excluded from Good pass; runtime Puppet is Evil | Base Puppet role is killable despite HealthyBluff |
| Wretch registered as a Minion | Excluded from Good pass by live Evil `registerAs` | Real Wretch/base role is killable |
| Shaman-copied Knight on a Good destination | Included after register-as refresh | Depends on preserved Corrupted/HealthyBluff statuses |
| Shaman-copied Knight on an Evil-aligned destination | Fallback unless a live Good register-as overrides it | Killable from preserved runtime Evil alignment |

Puppet's displayed saved Villager and Drunk's Knight bluff are not
`registerAs`, so neither changes Lilis's alignment filter. Corruption also does
not change selection alignment; it matters later when the selected real role
checks killability.

Shaman's `InitWithNoReset` installs copied Knight data but preserves the
destination's runtime alignment, active statuses, resistance, and runtime
data. A copied Knight on a Corrupted Good destination is therefore killable
and has the normal corrupted-Good execution consequence. A clean copied Knight
on a normal Good destination is immune. A copied Knight on a preserved Evil
destination is killable but does not apply Good-only execution damage. The
MessedUpByEvil marker Shaman later attempts has no Knight killability effect.

## Knight passive surfaces and killability precedence

`Immortal.GetInfo(Character)` and `GetBluffInfo(Character)` each return a fresh
`ActedInfo` with an empty description and null character list. Knight creates
no picker target, clue reference, saved target, or role-local history. Its two
action methods are semantically identical and react only to OnProtected (`60`)
and OnExecuted (`40`). The private helpers update achievements; they do not
change card state, status, protection, alignment, or health beyond the explicit
4 HP action below.

`Immortal.CheckIfCanBeKilled` has this strict precedence:

1. active `HealthyBluff` (`30`) -> `false`;
2. otherwise active `Corrupted` (`10`) -> `true`;
3. otherwise runtime alignment Evil (`20`) -> `true`; and
4. otherwise -> `false`.

It reads runtime alignment, not live `registerAs` and not displayed bluff.
HealthyBluff therefore has absolute precedence over both corruption and Evil
alignment when those states coexist. That coexistence is an edge, not the
normal corrupted-Doppelganger setup: corrupted Doppelganger does not add
HealthyBluff, and `Doppleganger.CheckIfCanBeKilled` returns true immediately
when HealthyBluff is absent.

A normal clean Doppelganger does have HealthyBluff. When it displays Knight,
its real killability method delegates to the displayed Knight role, whose own
HealthyBluff-first check returns false. A normal corrupted Doppelganger lacks
HealthyBluff, so an apparent Knight bluff does not protect it.

## Ordinary execution and exact HP sequencing

The ordinary execution branch in `Character.OnClick` asks the exact real
`dataRef.role.CheckIfCanBeKilled`. A false result performs only the Protected
action and restores the prior gameplay state. It does not reveal the real
identity, change card state, or damage HP. A real truthful Knight reaches
`Immortal.Act(OnProtected)`, whose only effect is achievement bookkeeping.

When killability is true, the order is:

1. force-display the real role;
2. `Character.Kill` changes state to Dead and emits its killed event;
3. `PlayerController.ManageResources`, subscribed to that event, applies base
   wrong-execution damage before later role actions;
4. invoke the reveal callback;
5. run OnExecuted; and
6. run Died.

The base `Role.GetDamageToYou` in this build returns 5. Drunk's real override
returns 2. Resource damage is skipped for active NoDamage, runtime Evil, or
`killedByDemon`. OnExecuted `Immortal.Act`/`BluffAct` separately checks runtime
non-Evil plus active Corrupted, performs the achievement helper, and calls
`Health.Damage(4)` directly. It does not consult NoDamage or a configurable
wrong-execution-cost field.

| Executed physical card | Base event damage | Knight OnExecuted | Total |
| --- | ---: | ---: | ---: |
| clean Good real Knight | protected; no kill | none | 0 |
| Corrupted Good real Knight | 5 | 4 | **9** |
| Corrupted Good Knight with NoDamage | 0 | 4 | **4** |
| runtime-Evil real Knight, clean or Corrupted | 0 | 0 | **0** |
| real Drunk displaying Knight | 2 | `Immortal.BluffAct` adds 4 | **6** |

Thus the authored phrase “4 additional damage” is exact. The 4 does not
replace the base 5 and is not combined through `wrong_exec_cost`; it is a
second fixed mutation after the base event. This native result closes the old
TODO hypothesis that a corrupted Knight should cost only 4 total.

Lilis death is not ordinary execution and never runs the +4 hook. A successful
Slayer `KillAndReveal` also omits OnExecuted and therefore omits the +4 hook.

## Slayer bypass and reveal consequence

`Slayer.CharacterPicked` decides its truth branch with
`Character.GetRegisterAlignment`, which prefers live
`registerAs.startingAlignment` over runtime alignment. A normal or merely
Corrupted Knight still registers Good, so Slayer reports the Good branch and
does not attempt a kill.

When the selected card registers Evil and is not already Dead, Slayer calls
`Character.KillAndReveal`. That helper force-displays the real identity, calls
`Character.Kill`, invokes the reveal callback, runs Died, and restores the
prior gameplay state. It never calls `CheckIfCanBeKilled`, never runs
OnProtected, and never runs OnExecuted. Consequently a real clean Knight with
an Evil live register-as can be killed by Slayer despite Knight immunity. If
its preserved runtime alignment is Good, the kill event can still apply the
base 5 HP; if runtime Evil, resource damage is skipped. Neither case applies
Knight's additional 4.

This is not a contradiction with the normal public card: a normal Knight has
no Evil register-as and never enters Slayer's kill branch. It is a composition
edge for transformed/stale registration states and demonstrates why Slayer
cannot be modeled as an ordinary execution.

## Death, status, and reset persistence

Neither Striga nor Immortal overrides `ActOnDied` or owns mutable fields. An
ordinary death does not clear active statuses. A dead Lilis can retain status
60 and a dead Knight can retain Corrupted/HealthyBluff in the physical status
list, but Dead state prevents Lilis action and ordinary victim eligibility.
The roster's Night rule remains active after Lilis death as described above.

`Character.Init` clears the physical card's active statuses, bluff,
register-as, runtime/acted information, saved action surface, and Start latch
for a new assignment, while physical resistance storage is preserved by that
method. Lilis's status60 and Knight's corruption are active statuses, so they
do not persist across full reinitialization. Shaman's distinct
`InitWithNoReset` is the intentional exception that preserves them. Neither
role has an ability-use reset hook or a history list to clear after Night.

## Active-corpus surface and limitations

The 426 active `tests/cases_v2` fixtures contain 122 Lilis deck entries across
122 cases. Those cases record 136 night-kill positions: 100 contain one, 18
contain two, and four contain none. The repeated entries establish that the
observable corpus is not a one-Night-per-village schema. The same corpus has
115 Knight deck entries across 115 cases and 121 apparent-Knight card records;
none of those apparent records is night-killed. Recorded corrupted-Good Knight
executions include the native 9 HP consequence in `asc51_v1` and `asc77_v2`.

These counts validate reachability but are not proof of exact target pools,
registered identity at selection time, failed protected selections, or event
ordering. Fixtures generally record successful deaths, not every Night attempt;
a protected Knight selection and a true no-candidate Night are both represented
as no new `night_kills` entry. Native control flow is authoritative for those
distinctions.

## Implementation consequences

- Model Lilis as a hard Good-first candidate pass followed only when empty by
  an unaligned fallback. Do not assign merely larger weight to Good targets.
- Use current registered alignment for selection, displayed bluff for neither
  pass, and current real role for delayed killability.
- Charge fixed 2 for every non-Dead Lilis Night action before knowing whether
  a victim dies. Preserve no-kill outcomes caused by empty pools, already-Dead
  delayed targets, and protected real roles.
- Treat clean Good real Knight as immune to ordinary execution and Lilis, but
  not as a universal guard around Slayer's forced `KillAndReveal` helper.
- A corrupted Good Knight ordinary execution costs 9 in this fingerprint:
  base 5 then fixed additional 4. Drunk displaying Knight costs 6.
- Do not apply Knight's +4 to Lilis or Slayer deaths; neither path runs
  OnExecuted.
- Preserve the duplicate asymmetry: one same-asset Lilis receives Start
  protection, every live duplicate acts once per actual Night, and delayed
  target collisions do not reroll.

## Bounded unknowns

The boundary establishes the exact candidate construction and uniform integer
index draw but does not recover or predict Unity's session RNG state. It also
does not assert behavior for modded assets, runtime replacement of the
serialized orders, or externally injected resistance 50/55/60. The public
asset and current native fingerprint contain no such replacement or shipped
resistance producer. Those are explicit out-of-build conditions, not ambiguity
in the current Lilis/Knight rules.
