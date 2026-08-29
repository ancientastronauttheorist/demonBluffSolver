# Gameplay Execution And Resolution Boundary

Build: `f530404b0f3f_807de4a83df4`

Evidence status: **metadata plus managed-reconstruction hypothesis only**. None
of the 30 methods in this boundary has been promoted to native-static evidence
by this note.

The checked target set is
[`reverse_engineering/targets/gameplay_execution_resolution.json`](../../targets/gameplay_execution_resolution.json).
It is the next compact boundary around execution dispatch, player-health
mutation, kill protection, terminal-condition dispatch, and Striga's night
rule. Its baseline Ghidra export completed read-only at 30/30 functions with
zero failures. Four entries use expected shared constant/helper bodies, while
their requested managed identities remain preserved in the export headers.
Decompiled native bodies remain outside the repository.

## Boundary groups

### Trigger dispatch and status application

| Method | RVA |
| --- | ---: |
| `Character.Act` | `0x3645C0` |
| `Character.RoleAct` | `0x368790` |
| `CharacterHelper.CheckLying` | `0x397750` |
| `CharacterStatuses.AddStatus` | `0x363AA0` |

The managed reconstruction suggests that `Character.Act` gates repeated Start
actions, decides whether the real and bluff roles use `Act` or `BluffAct`, and
routes role-produced information through `Character.RoleAct`. Native audit must
confirm the exact lying/bluff decision table, callback assignment, null paths,
and ordering.

### Wrong-execution damage, healing primitive, and terminal result

| Method | RVA |
| --- | ---: |
| `Role.GetDamageToYou` | `0x3C4CD0` |
| `Drunk.GetDamageToYou` | `0x3DA560` |
| `CurrentMaxValue.Reduce` | `0x379610` |
| `CurrentMaxValue.Add` | `0x3795D0` |
| `WinConditions.CheckEndGameConditions` | `0x3ADEA0` |
| `WinConditions.DelayedCheckCondition.MoveNext` | `0x3A9480` |
| `WinConditions.Win` | `0x3AF320` |
| `WinConditions.Lose` | `0x3ADFC0` |
| `WinConditions.ChangeGameplayState.MoveNext` | `0x3A9100` |

The existing core target already contains `PlayerController.ManageResources`,
`Health.Damage`, `Health.Heal`, `Health.AddMaxHp`, `Health.ResetHp`, and
`CurrentMaxValue.GetValue`. Managed output suggests that the ordinary damage
override returns 5, Drunk returns 2, `Reduce` clamps toward zero, and `Add`
clamps toward the maximum. These values and clamp branches are hypotheses until
the selected native methods are audited.

The delayed win-condition iterator is included because its managed output is
substantially malformed around evil counts, player health, Saint handling, and
the win/lose branches. The wrapper and result methods close the intended
call-chain boundary without importing score-panel and restart UI internals.

### Execution protection

| Method | RVA |
| --- | ---: |
| `Role.CheckIfCanBeKilled` | `0x3B24C0` |
| `Immortal.CheckIfCanBeKilled` | `0x3BC4A0` |
| `Doppleganger.CheckIfCanBeKilled` | `0x3D7090` |
| `Immortal.Act` | `0x3BC2A0` |
| `Immortal.BluffAct` | `0x3BC3A0` |

`Immortal` is the managed implementation associated with Knight behavior. The
managed reconstruction suggests that alignment, Corrupted, and HealthyBluff
status participate in protection; it also suggests that Doppelganger can
delegate the check to its bluff role and that the protected/executed triggers
have additional effects. Native audit must establish the exact predicates,
damage amount, trigger ordering, and exceptional paths. `Doppleganger` is the
spelling present in current metadata.

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

The managed hypothesis is that `Striga.GetRules` introduces a `NightModeRule`,
the rule subscribes to reveal/kill events, and its reveal counter transitions
the round into Night. The existing lifecycle target then dispatches Night
actions through `Character.Act`. Managed output further suggests that
`Striga.Act` applies a demon-kill immunity status during Start, selects a night
victim through `Demon.KillHidden`, and damages the player during Night.

The two Demon methods and five collection helpers are kept together so native
audit can determine the exact alive/alignment/visibility/status filters,
fallback relationship, empty-list behavior, and random-selection semantics.
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

The intended joined hypotheses are:

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

## Recommended first native slice

Audit these 13 methods first:

1. `Character.Act` (`0x3645C0`)
2. `Character.RoleAct` (`0x368790`)
3. `CharacterHelper.CheckLying` (`0x397750`)
4. `Role.GetDamageToYou` (`0x3C4CD0`)
5. `Drunk.GetDamageToYou` (`0x3DA560`)
6. `CurrentMaxValue.Reduce` (`0x379610`)
7. `Role.CheckIfCanBeKilled` (`0x3B24C0`)
8. `Immortal.CheckIfCanBeKilled` (`0x3BC4A0`)
9. `Doppleganger.CheckIfCanBeKilled` (`0x3D7090`)
10. `Immortal.Act` (`0x3BC2A0`)
11. `Immortal.BluffAct` (`0x3BC3A0`)
12. `WinConditions.CheckEndGameConditions` (`0x3ADEA0`)
13. `WinConditions.DelayedCheckCondition.MoveNext` (`0x3A9480`)

This slice is intended to close the execution path first: real/bluff action
dispatch, ordinary and Drunk HP costs, lower-bound clamping, Knight and
Doppelganger protection, and terminal-condition selection. It prioritizes the
managed bodies with malformed branches or indirect virtual dispatch before the
larger night-selection slice.

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
