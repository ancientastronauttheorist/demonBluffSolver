# RoguelikeStandard lifecycle and saved-mode loading

Pinned build `f530404b0f3f_807de4a83df4`. The
[native audit](../../scripts/audit_roguelike_standard_lifecycle.py) passes 36
cases for five exact declarations. This closes the high-ascension mode's
Init/LoadGame/DeInit caller boundary following
[GameData mode publication](game_data_lifecycle.md) and the
[StandardMode lifecycle](game_mode_lifecycle.md).

## Event lifecycle discrepancy

Both Init and DeInit process events in this exact order, constructing delegates
for the current instance and storing each operation's result before proceeding:

| GameplayEvents event | Callback | Init | DeInit |
| --- | --- | --- | --- |
| OnCharacterKilled `+0x48` | RoguelikeStandard.OnCharacterKilled | Combine | **Combine** |
| OnRoundWon `+0x20` | Virtual OnStageCompleted, metadata from class `+0x240` | Combine | Remove |
| OnDied `+0xB0` | RoguelikeStandard.OnFailed | Combine | Remove |

The DeInit CharacterKilled call is the actual native call at `0x3E9963` to
Delegate.Combine `0x116BCC0`. The next two calls at `0x3E9A35` and `0x3E9AF5`
resolve to Delegate.Remove `0x116E070`. All six Init/DeInit operation calls
are independently asserted after decoding from their method entries. This
is not inferred from a similarly named method or decompiler alias.

DeInit therefore passes the kill handler to Combine instead of Remove. Under
normal multicast delegate behavior this can retain/add a subscription when
changing modes, potentially duplicating later kill callbacks. That impact is
an inference; this audit does not execute the native Delegate.Combine body,
invoke kill callbacks, or establish actual live subscription counts. The
caller mismatch is verified regardless of that remaining boundary.
The [follow-up audit](roguelike_delegate_score.md) now executes the delegate
list and actual multicast invocation through the native scoring callback,
confirming accumulation under the recovered operation sequence. It still
does not claim observed live subscription counts.

Both methods type-check nonnull operation results and publish null results as
null. Earlier event writes remain when a later operation/cast fails. The native
methods do not reset roundScore, set GameData.CurrentVillage, alter instance
progression fields, load a profile, or clear a profile/script cache. The harness
checks the instance field block and GameData static canary block remain
unchanged across normal and failure cases. This differs from StandardMode.Init,
which performs later score and village writes.

## Saved-mode loader

LoadGame at `0x3EA1B0` tailcalls SavesGame.get_RoguelikeStandard at `0x387A20`.
The getter reads `SavedRoguelikeStandard` from PlayerPrefs. Empty/null text
allocates a fresh RoguelikeStandard and executes its shared constructor.
Nonempty text causes a second GetString call, followed by the exact
`FromJson<RoguelikeStandard>` metadata instantiation. The second result is used
when the reads differ; the JSON result, including null, is returned unchanged.
No preference writes or JSON serialization occur here.

RoguelikeStandard's constructor aliases GameMode's constructor at `0x357920`,
which tailcalls the shared no-op body `0x33ED50`. There are no explicit
instance-field initializers. Zero/default allocation behavior belongs to the
runtime allocation boundary. The constructor target retains its exact managed
signature while reusing the compatible canonical applied prototype already
assigned to that native RVA.

## Evidence and limits

The [manifest](../../targets/roguelike_standard_lifecycle.json) includes Init,
LoadGame, DeInit, constructor and save getter. All five baseline native exports
completed; private decompilations are not checked in. The
[report](../../reports/f530404b0f3f_807de4a83df4_roguelike_standard_lifecycle.json)
contains nine native relationship checks and 36 cases, covering four level
canaries, null delegate results, each operation failure/cast boundary, seven
save-loading paths, and the no-op constructor. Normal returns verify preserved
registers and stack. Exact DLL, Dumper script and declaration fingerprints are
required; metadata initialization is supplied warmed.

Delegate allocation/construction, Combine/Remove, runtime type services,
PlayerPrefs and JSON are explicit gateways. Failure cases stop at the gateway
rather than emulate managed exception unwinding. Complete delegate-list
semantics, callback progression/scoring, save writes and all-mode lifecycle
behavior remain separate boundaries. Existing current/previous ascension
selectors remain in [ascension setup](ascension_setup.md); they are not
duplicated here.

```powershell
python reverse_engineering/scripts/audit_roguelike_standard_lifecycle.py `
  'B:\SteamLibrary\steamapps\common\Demon Bluff Playtest' `
  'B:\CodexTools\DemonBluffReverseEngineering\artifacts\f530404b0f3f_807de4a83df4\il2cppdumper-v6.7.46' `
  --output reverse_engineering/reports/f530404b0f3f_807de4a83df4_roguelike_standard_lifecycle.json
```

No live game/save mutation, solver change, or shared coverage edit is included.
