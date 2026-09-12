# StandardMode lifecycle and saved-mode loading

Pinned build `f530404b0f3f_807de4a83df4`. The new
[audit](../../scripts/audit_game_mode_lifecycle.py) passes 41 native cases
across six selected declarations plus exact metadata checks of two inherited
base callbacks. This follows the
[GameData mode-publication boundary](game_data_lifecycle.md).

## Initialization and teardown

StandardMode.Init subscribes instance callbacks in this order:

| GameplayEvents field | Callback source |
| --- | --- |
| OnRoundWon `+0x20` | Virtual OnStageCompleted, method metadata from instance class `+0x240` |
| OnDied `+0xB0` | Exact StandardMode.OnFailed metadata |
| OnCharacterKilled `+0x48` | Exact StandardMode.OnCharacterKilled metadata |

Each step allocates and constructs a delegate, calls Delegate.Combine, validates
the result's type, stores the result, and invokes a write barrier. Only after
all three subscriptions does Init read currentLevel `+0x14`, clear roundScore
`+0x28`, ensure GameData class initialization, and publish that captured level
to GameData.CurrentVillage. Negative and large signed int32 levels are not
clamped. There is no duplicate-subscription guard in this method; the internals
of Delegate.Combine remain a service boundary.

DeInit removes the callbacks in a different order: OnCharacterKilled,
OnRoundWon, then OnDied. It constructs fresh delegates for Delegate.Remove;
it does not clear all event subscribers. Its body makes no score, current
village, save, profile, or cache writes. Callback invocation bodies and multicast
delegate internals are outside this audit.

Failures do not roll back earlier event stores. The harness tests failures at
each Combine/Remove call and invalid results at each corresponding cast. For
Init, these failures precede score reset and village publication. A failing
GameData class-initialization gateway occurs later: all subscriptions and the
roundScore reset have already happened, while CurrentVillage retains its prior
value. The harness stops at explicit failure gateways and does not emulate
managed exception unwinding. Metadata initialization is supplied warmed;
GameData class initialization additionally has explicit cold success/failure
cases.

## Loading and base declarations

StandardMode.LoadGame does not return its receiver. It tailcalls
SavesGame.get_StandardMode. That getter reads PlayerPrefs key `SavedStandard`
and calls String.IsNullOrEmpty. Empty/null text causes a fresh StandardMode
allocation and its constructor; nonempty text causes a second PlayerPrefs read
and FromJson with the exact `StandardMode` generic method metadata. The second
read's value is passed to JSON even if it differs from the first. The JSON
result, including null, is returned unchanged. PlayerPrefs reads and JSON
failures remain explicit gateways; no host save is read or modified.

StandardMode's constructor writes only failScoreDecrease `+0x38 = 20` before
its shared no-op base call. Remaining default fields depend on runtime
allocation initialization, not explicit constructor assignments. GameMode's
constructor tailcalls the same no-op body. GameMode.Init and LoadGame are
abstract declarations without native bodies. Its OnLoadGame and DeInit have
exact metadata declarations at shared RVA `0x33ED50`; StandardMode inherits
OnLoadGame and overrides DeInit. Aliased Ghidra names on that shared body do
not identify a different lifecycle operation.

Neither Init nor LoadGame selects or copies a current AscensionsData profile.
Init sets the village index; profile lookup and temporary-copy setup remain
in the separate [ascension setup](ascension_setup.md) boundary. Combined with
GameData.ChangeGameMode, Standard loading therefore precedes old-mode teardown;
Standard initialization then subscribes callbacks and publishes its level
before GameData publishes the loaded mode reference.

## Validation and artifact scope

The [target manifest](../../targets/game_mode_lifecycle.json) selects Standard
Init/LoadGame/DeInit/constructor, GameMode constructor, and the SavedStandard
getter. All six private baseline exports completed. Shared base OnLoadGame and
DeInit are supplemental exact metadata checks in the
[report](../../reports/f530404b0f3f_807de4a83df4_game_mode_lifecycle.json), rather
than new typed target memberships: their two-parameter declarations alias a
body whose existing canonical applied prototype has four parameters, which
current inventory validation correctly rejects. The GameMode constructor
reuses the existing compatible canonical applied prototype.

The harness executes native caller instructions in Unicorn 2.1.4, checks exact
DLL and Dumper fingerprints and declarations, verifies seven selected native
relationships, and validates preserved registers and stack on normal returns.
Delegate construction, allocation, combine/remove, runtime casts, class init,
PlayerPrefs, and generic JSON use explicit gateways. Tests include null delegate
results, cast/service failures, four level values, class-init ordering, load
read counts and second-read identity, and base/Standard constructors. This is
not a full save serializer, all-mode lifecycle, or event callback audit.

```powershell
python reverse_engineering/scripts/audit_game_mode_lifecycle.py `
  'B:\SteamLibrary\steamapps\common\Demon Bluff Playtest' `
  'B:\CodexTools\DemonBluffReverseEngineering\artifacts\f530404b0f3f_807de4a83df4\il2cppdumper-v6.7.46' `
  --output reverse_engineering/reports/f530404b0f3f_807de4a83df4_game_mode_lifecycle.json
```

Native code remains in private exports. No live game state, save data, solver,
or shared coverage inventory is changed by this checkpoint.
