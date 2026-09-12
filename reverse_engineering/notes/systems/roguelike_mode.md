# RoguelikeMode remaining declared surface

Pinned build `f530404b0f3f_807de4a83df4`. The
[audit](../../scripts/audit_roguelike_mode.py) covers seventeen remaining
RoguelikeMode declarations. Together with its two current/previous profile
selectors in [ascension setup](ascension_setup.md), all nineteen declarations
have native evidence. Inherited GameMode behavior and runtime services remain
separate boundaries; this is not a claim that every gameplay effect is closed.

## Lifecycle

Init first initializes GameData as needed and sets GameData.CurrentVillage to
zero. It then subscribes OnFailed to GameplayEvents.OnDied `+0xB0`, followed
by OnRestartLevel to OnRestartGame `+0x38`. Both use Delegate.Combine with the
receiver and exact method metadata. DeInit removes those same callbacks in
the same order. There is no kill-handler subscription in these bodies.

The early CurrentVillage write is retained if a later delegate operation or
cast fails. A failing GameData class initialization occurs before that write.
Earlier successful event writes survive later subscription/removal failures.
Neither method changes score, maxLevel or timesDied. Delegate internals remain
gateways here; the [native multicast audit](roguelike_delegate_score.md) covers
the underlying list mechanics for its explicitly described fixtures.

LoadGame tailcalls SavesGame.get_RoguelikeMode and returns that result unchanged,
including null. Its implementation is already covered in
[the saves surface](saves_game_surface.md): the key is `SavedRoguelike`, with
fresh allocation for empty text and exact JSON parsing otherwise. This audit
tests the caller without duplicating those service internals.

## Failure, restart and score writes

OnFailed only resets Gameplay.Instance.currentDay `+0x7C` to zero. It does not
increment timesDied or directly reset score. A missing Gameplay instance fails;
class initialization precedes the instance access.

OnRestartLevel inspects timesDied:

- Zero returns immediately without state transitions.
- Positive values are reset to zero.
- Negative values remain unchanged.

Every nonzero input then calls Gameplay.ChangeGameplayState with Init `1`,
followed by Map `70`. Positive-counter reset precedes class initialization
and both transitions; failures do not undo it. The state transition service
itself is a gateway, so callbacks and side effects caused by those states are
outside this audit.

UpdateScore compares the incoming score against the stored score using a
strict signed greater-than condition. Equal/lower values neither update fields
nor save. Improvement stores score `+0x10` and the supplied level in maxLevel
`+0x14`, then tailcalls SavesGame.set_RoguelikeMode. The level is not range
checked. A failing save leaves both field updates in place. The separately
audited setter serializes/writes `SavedRoguelike` without an explicit flush.

## Locking and presentation

IsLocked returns false immediately when GameData.DebugMode is enabled, skipping
save loading. Otherwise it loads the saved **RoguelikeStandard** mode and returns
whether that object's bestAscension is less than two, using a signed comparison.
A null loaded result fails; this is not a check of RoguelikeMode.maxLevel.

GetScores invokes the receiver's virtual GetScore slot (`class +0x258`), boxes
the resulting int32, and formats the highest-score label. Native tests install
both the actual GetScore body and a distinct override result to prove the
virtual call is respected. The exact template is preserved in the
[report](../../reports/f530404b0f3f_807de4a83df4_roguelike_mode.json).

| Remaining simple declaration | Verified behavior |
| --- | --- |
| GetGameMode | Returns `10`. |
| GetMaxLevel / GetCurrentLevel | Return maxLevel `+0x14`. |
| GetScore | Returns score `+0x10`. |
| GetStartingLevel / GetResetLevel | Return zero. |
| CanResetLevel | Returns false. |
| OnStageCompleted | Shared no-op body; no field stores. |
| Constructor | Calls the shared no-op base body; no explicit field initializers. |

The void shared body preserves return-register contents; the audit assigns no
meaningful return value to it. Default field values come from runtime allocation.

## Validation and limits

The report records all seventeen exact metadata declarations, fifteen native
instruction relationships and 155 fixture cases. Normal returns verify stack
and preserved registers. Cases exercise lifecycle ordering and failures,
signed score/level bounds, restart counter signs, state-service failures,
nullable Gameplay/save references, debug bypass, virtual formatting, and shared
getter/constructor bodies. All seven requested private baseline exports
completed; native bytes and bodies remain private.

Allocation, delegate construction/combination/removal, class initialization,
SavesGame properties, gameplay-state changes, boxing and formatting use explicit
gateways. Metadata is warmed. Exceptions stop at the gateway without emulating
managed unwinding. No typed target, shared coverage membership, solver, save,
or live game state is changed.

```powershell
python reverse_engineering/scripts/audit_roguelike_mode.py `
  'B:\SteamLibrary\steamapps\common\Demon Bluff Playtest' `
  'B:\CodexTools\DemonBluffReverseEngineering\artifacts\f530404b0f3f_807de4a83df4\il2cppdumper-v6.7.46' `
  --output reverse_engineering/reports/f530404b0f3f_807de4a83df4_roguelike_mode.json
```
