# Gameplay Execution And Resolution Boundary

Build: `f530404b0f3f_807de4a83df4`

Evidence status: **native-static**. All 30 selected methods have native control
flow confirmation. No statement here is based on live dynamic observation.

The checked target set is
[`reverse_engineering/targets/gameplay_execution_resolution.json`](../../targets/gameplay_execution_resolution.json).
It is the next compact boundary around execution dispatch, player-health
mutation, kill protection, terminal-condition dispatch, and Striga's night
rule. Its baseline Ghidra export completed read-only at 30/30 functions with
zero failures. Four entries use expected shared constant/helper bodies, while
their requested managed identities remain preserved in the export headers.
Decompiled native bodies remain outside the repository.

## Typed-analysis corroboration

The fully analyzed isolated typed project also validated and exported all 30
methods. Post-save read-only ABI checks confirmed 88 parameter-storage
locations without mutating the program. Relative to the baseline export,
unresolved-type tokens fell from 160 to 96 and raw field-offset accesses fell
from 243 to 120; the four pre-existing decompiler error markers did not
increase. Aggregate, body-free metrics are recorded in the
[`typed quality report`](../../reports/f530404b0f3f_807de4a83df4_typed_quality_gameplay_execution_resolution.json).

## Boundary groups

### Action and lying dispatch

| Method | RVA |
| --- | ---: |
| `Character.Act` | `0x3645C0` |
| `Character.RoleAct` | `0x368790` |
| `CharacterHelper.CheckLying` | `0x397750` |
| `CharacterStatuses.AddStatus` | `0x363AA0` |

Native-static audit confirms that `CharacterHelper.CheckLying` returns true when
the character is Corrupted, or when it is not HealthyBluff and either its
runtime alignment is Evil or its bluff data is Unity-non-null. Corruption has
precedence over HealthyBluff; otherwise HealthyBluff forces a truthful result.
The helper does not consult the separate Lying, AppearTruthfull, or AppearLying
statuses, and it uses the runtime alignment and bluff reference rather than the
starting alignment or cloned role fields.

`Character.Act` has one special early return: while the gameplay state is
Summary, a Day trigger is suppressed when the real data or non-null bluff data
is still marked as picking. Otherwise it logs and invokes `onTrigger` before
role dispatch. Start is one-shot only after that callback: the first Start marks
`characterStartActed` before dispatch, and later Start calls return at that
point.

Role dispatch uses this matrix:

| Lying result | Runtime alignment | Real role | Bluff role, when present |
| --- | --- | --- | --- |
| false | any | `Act` | `Act` |
| true | non-Evil | `BluffAct` | `BluffAct` |
| true | Evil | `Act` | `BluffAct` |

The Evil exception therefore applies even when corruption caused the lying
result. `Character.RoleAct` replaces the selected role's `onActed` callback,
then calls `Act` for case zero and `BluffAct` for every nonzero case. The
callback starts `ShowActedDelayed(0.0, info, trigger)`; creation of the acted
information remains the role implementation's responsibility. The real role is
required, while a missing bluff role is a clean no-op after real dispatch.

`CharacterStatuses.AddStatus` first checks exact enum membership in the
resistance list. A resisted value returns without touching any other field.
Otherwise the method appends only when the status is not already present, then
overwrites the single shared `targetCharacter` field even for a duplicate
status. `sourceRef` is unused, `targetRef` may be null, and there is no event,
role notification, resistance consumption, or character-state gate. Thus most
two-argument calls, whose optional target is null, clear the shared target after
an accepted status application.

### Wrong-execution damage and value reduction

| Method | RVA |
| --- | ---: |
| `Role.GetDamageToYou` | `0x3C4CD0` |
| `Drunk.GetDamageToYou` | `0x3DA560` |
| `CurrentMaxValue.Reduce` | `0x379610` |
| `CurrentMaxValue.Add` | `0x3795D0` |

Native-static audit confirms that the default wrong-execution cost is 5 HP and
Drunk overrides it with 2 HP. `CurrentMaxValue.Reduce` performs signed
subtraction, clamps only the lower bound to zero, stores the result, and invokes
`onValueChanged` exactly once even when the value was already zero or clamps to
zero. A negative input therefore increases the current value without applying
the maximum bound.

`CurrentMaxValue.Add` is the signed mirror with only an upper clamp: it stores
the unchecked 32-bit sum unless that sum is greater than `max`, in which case it
stores `max`. Negative inputs can reduce below zero, and overflow occurs before
the signed comparison. As with `Reduce`, the value-change callback fires once
after the store even when the effective value did not change.

### Terminal result selection

| Method | RVA |
| --- | ---: |
| `WinConditions.CheckEndGameConditions` | `0x3ADEA0` |
| `WinConditions.DelayedCheckCondition.MoveNext` | `0x3A9480` |
| `WinConditions.Win` | `0x3AF320` |
| `WinConditions.Lose` | `0x3ADFC0` |
| `WinConditions.ChangeGameplayState.MoveNext` | `0x3A9100` |

`CheckEndGameConditions` starts an independent coroutine every time it is
called; it does not coalesce requests or guard against an existing terminal
state. After a scaled 0.1-second delay, the iterator applies this precedence:

1. if any dead real-role Saint was not killed by a demon, activate the automatic
   loss panel and lose;
2. otherwise, if signed player HP is at most zero, activate the ordinary loss
   panel and lose;
3. otherwise, if the number of Evil-aligned dead-list entries is at least the
   number of Evil-aligned current-character entries, win; and
4. otherwise, do nothing.

The two Evil counts are independent raw-list scans with no identity matching or
deduplication. Saint detection ignores demon-killed Saints and checks the real
role object, not displayed alignment or statuses. As a result, duplicate dead
entries can satisfy the win comparison early, and empty current/dead lists win
when HP is positive and no qualifying Saint exists.

`Win` reveals all characters first. Its final-day Roguelike branch activates
the last-day panel, updates scores, and schedules Summary without invoking the
ordinary round-win event. The ordinary branch invokes `OnRoundWon`, optionally
signals ascension completion after rereading the mode, activates the win panel,
and schedules Summary. `Lose` reveals all, updates scores, schedules Summary,
and only then invokes `OnDied`; panel selection belongs to the delayed
predicate. The state-change iterator waits a scaled 0.03 seconds before asking
Gameplay to enter Summary. Repeated condition checks can therefore schedule
duplicate terminal work.

### Knight and Doppelganger protection

| Method | RVA |
| --- | ---: |
| `Role.CheckIfCanBeKilled` | `0x3B24C0` |
| `Immortal.CheckIfCanBeKilled` | `0x3BC4A0` |
| `Doppleganger.CheckIfCanBeKilled` | `0x3D7090` |
| `Immortal.Act` | `0x3BC2A0` |
| `Immortal.BluffAct` | `0x3BC3A0` |

`Role.CheckIfCanBeKilled` always returns true. `Immortal`, the managed
implementation associated with Knight behavior, applies this native predicate
in order: HealthyBluff is protected; otherwise Corrupted can be killed;
otherwise Evil alignment can be killed; otherwise the character is protected.
HealthyBluff therefore has absolute precedence over corruption and alignment.

`Doppleganger.CheckIfCanBeKilled` returns true whenever HealthyBluff is absent.
With HealthyBluff it delegates virtually to the role stored in the character's
bluff data; missing bluff data or its role is not handled as a protected
fallback. `Doppleganger` is the spelling present in current metadata.

`Immortal.Act` and `Immortal.BluffAct` are semantically identical. A Protected
trigger performs only unique-Knight achievement bookkeeping. An Executed
trigger deals an additional 4 HP only when the runtime alignment is non-Evil
and the character is Corrupted. Joined to the previously audited execution
path, a normally executed corrupted good Knight incurs the ordinary 5 HP first
and this separate 4 HP hit afterward, for 9 total. NoDamage suppresses only the
ordinary hit; an Evil Knight receives neither hit. Each applied hit performs
its own value mutation and callback.

### Night-rule lifecycle and target selection

| Method | RVA |
| --- | ---: |
| `Striga.GetRules` | `0x3EE7B0` |
| `NightModeRule.Init` | `0x3E4DE0` |
| `NightModeRule.Remove` | `0x3E5080` |
| `NightModeRule.Revealed` | `0x3E52A0` |
| `Striga.Act` | `0x3EE6F0` |
| `Demon.KillHidden` | `0x3D6C30` |
| `Demon.KillRandom` | `0x3D6E70` |
| `Characters.FilterAliveCharacters` | `0x36A240` |
| `Characters.FilterAlignmentCharacters.Character` | `0x36A030` |
| `Characters.FilterHiddenCharacters` | `0x36B020` |
| `Characters.FilterCharacterMissingStatus` | `0x36A8C0` |
| `Characters.GetRandomAliveCharacter` | `0x36C740` |

`Striga.GetRules` returns a new list containing exactly one new `NightModeRule`
whose reveal threshold is 4. Every call produces distinct list and rule objects
with a zero counter and null callbacks; it does not initialize or register the
rule. Rule initialization subscribes bound handlers to the global
killed-character event first and revealed-character event second; it does not
reset the counter. Repeated initialization appends duplicate subscriptions.
Removal performs the corresponding two delegate removals in the same order,
removing only one matching occurrence from each event.

The reveal handler ignores events while Gameplay is in Summary and ignores a
character whose state is exactly Dead. Every other event increments
`currentStep`, invokes `onStepIncrease`, and then rereads the counter and
threshold. At or above the threshold it invokes `onNightStart`, asks Gameplay
to enter Night, and resets the counter to zero, in that order. Callback changes
can therefore affect the comparison, and callback exceptions can prevent the
later transition or reset. Gameplay state-change callbacks observe the
pre-reset counter. Duplicate subscriptions can count the same reveal more than
once and, because Night is not excluded, a later duplicate can start a fresh
cycle after an earlier copy crossed the threshold and reset it.

On Start, `Striga.Act` applies `UnkillableByDemon` to its own character with a
null target reference. On Night, a non-Dead Striga first dispatches
`Demon.KillHidden` and then deals 2 HP to the player. The kill call only starts
the previously audited delayed demon-kill path, so the player damage happens
before the victim's delayed death resolves. It still occurs when both victim
passes find no candidate, and it is not rolled back if the delayed kill later
finds the target dead or protected.

`Demon.KillHidden` filters the global current-character list through: state is
not Dead, apparent alignment is Good, state is exactly Hidden, and the active
status list lacks `UnkillableByDemon`. A nonempty result produces one random
target and one `KillByDemon` dispatch. An empty result falls back to
`KillRandom`, which rebuilds the pipeline without the alignment filter. The
fallback can therefore select any alignment, returns cleanly when empty, and
never rerolls after a delayed kill is rejected.

The selected helpers all allocate new lists, preserve source order and duplicate
occurrences, and do not mutate their inputs. `FilterAliveCharacters` means only
`state != Dead`; `FilterHiddenCharacters` requires exactly Hidden; and the
missing-status helper consults only `statuses.statuses`, not resistances or the
shared target. The `List<Character>` alignment overload uses live
`registerAs.startingAlignment` when `registerAs` is Unity-non-null, otherwise it
uses the character's runtime alignment; it does not read `bluff`.
`GetRandomAliveCharacter` performs no alive check of its own: it draws one
integer index in `[0, Count)` and returns that item. Callers guard the empty
case, while a direct empty-list call fails at lookup. Preserved duplicates
therefore bias selection probability.

The `Characters$$FilterAlignmentCharacters` metadata name is overloaded; this
boundary deliberately selects the `List<Character>` overload at `0x36A030`,
not the `List<CharacterData>` overload at `0x369EB0`.

## Audited target-set overlaps

All 30 metadata identity/signature/RVA tuples are new relative to the checked
`gameplay_core`, `gameplay_lifecycle`, and `gameplay_roster_helpers` target
sets. The following already-audited methods are call-chain context and are not
duplicated here:

- core: `Gameplay.StartKill`, `Gameplay.StartNight`,
  `Gameplay.ChangeGameplayState`, `PlayerController.ManageResources`, the four
  `Health` mutations, and `CurrentMaxValue.GetValue`;
- lifecycle: `Gameplay.UpdateRules`, `Character.OnClick`,
  `Character.RevealAllReal`, `Character.KillProtected`, `Character.Kill`,
  `Character.KillByDemon`, `Character.DelayedDemonKill.MoveNext`,
  `Gameplay.ManageKilledCharacter`, and the three `NightPhase` methods; and
- roster helpers: no direct overlap or required duplicate.

The audited joined call chains are:

1. execution request: existing `Gameplay.StartKill` and `Character.OnClick`
   dispatch into the selected protection and `Character.Act` methods;
2. resource consequence: existing kill/resource handlers dispatch into the
   selected damage override and `CurrentMaxValue` methods;
3. terminal consequence: existing dead-character bookkeeping dispatches into
   the selected `WinConditions` methods; and
4. night consequence: existing rule initialization and Night phase dispatch
   join the selected NightModeRule, Striga, Demon, and collection methods.

## Deliberate exclusions

- `NightModeRule.ManageKill` and
  `NightModeRule.<ManageKill>d__8.MoveNext` are omitted. Managed reconstruction
  presents this subscriber as a short delay with no clear rule mutation, but
  that interpretation has not been established natively.
- `CurrentMaxValue.Reset`, `CharacterStatuses.Contains`, and
  `NightModeRule.ctor` are small supporting utilities that do not add a new
  rule edge to this compact boundary.
- `AddHealthCard` belongs to the reward/card subsystem. No direct managed call
  to `Health.Heal` was found in this boundary; the already-targeted wrapper and
  selected `CurrentMaxValue.Add` cover the health-addition primitive.
- Score-detail calculation, summary-panel selection, restart behavior,
  unrelated role actions, and general status resistance/removal remain later
  boundaries.

## Native-audited boundary

All 30 methods now have native-static coverage: action and lying dispatch,
status insertion, wrong-execution damage, bounded-value mutation, Knight and
Doppelganger protection, terminal resolution, Night-rule lifecycle, Striga's
action, Demon victim selection, and the supporting collection filters.

## Metadata alias cautions

- `Role.CheckIfCanBeKilled` uses a heavily shared trivial-body RVA, so exports
  must validate the requested metadata identity and signature rather than a
  primary symbol label.
- `Role.GetDamageToYou` and `Drunk.GetDamageToYou` also use shared constant-body
  RVAs.
- `CurrentMaxValue.Reduce` shares its RVA with
  `CurrentStartingValue.Reduce`.
- `Characters.FilterAlignmentCharacters.Character` must be selected by both
  RVA and exact signature because its metadata name is overloaded.
