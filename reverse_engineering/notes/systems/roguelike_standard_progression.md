# RoguelikeStandard progression, reset and saving

This audit executes the pinned `RoguelikeStandard` progression bodies, their
actual nested reset/save calls and the immediate `SavedRoguelikeStandard`
setter. It additionally checks ten small getter/no-op declarations. All
seventeen declarations are exact supplemental metadata checks; no new typed
target membership is introduced.

## Stage and ascension completion

`OnStageCompleted` (`0x3ea320`) increments `currentVillage` using int32 wrap.
It adds `roundScore + 50 + currentAscension * 50` to `ascensionScore`, with
int32 arithmetic, then clears `roundScore`. It updates `bestVillage` only when
both `currentAscension >= bestAscension` and the incremented
`currentVillage >= bestVillage`. It raises `bestScore` only when the resulting
signed `ascensionScore` is greater. There is no last-village or completion test
in this body: after these field writes it initializes GameData if necessary,
calls `GameData.IncreaseVillage`, then saves.

`AscensionComplete` (`0x3e97b0`) independently increments `currentAscension`,
clears `bestVillage`, raises `bestAscension` if the incremented signed value is
larger, and captures the wrapping sum `ascensionScore + roundScore` into
`prevAscensionScore`. It then initializes GameData if necessary, sets static
`CurrentVillage` to zero, calls the actual `ResetScores` body, and saves again.
Since `ResetScores` itself saves, successful ascension completion performs
**two JSON/SetString pairs**. Both observe the reset field state.

These methods do not clamp arithmetic overflow. Incrementing an ascension of
INT_MAX produces INT_MIN before the signed best-ascension comparison.

## Failure, reset and abandonment

`OnFailed` (`0x3ea2e0`) first increments `currentDiedTimes`. It calls virtual
slot 15 (`AbandonRun`) only when the incremented signed count is at least four,
then saves. The fixtures install the actual RoguelikeStandard override in that
slot, including its method metadata argument. Thus threshold failures run the
native reset and save, followed by the caller's second save. Below the
threshold, neither `roundScore` nor `ascensionScore` is deducted or reset.
INT_MAX deaths wrap to INT_MIN and bypass abandonment; no clamp is implied.

`AbandonRun` (`0x3e97a0`) tailcalls `ResetScores` (`0x3ea3b0`). Reset first
initializes GameData if needed, sets its static `CurrentVillage` to zero,
clears only these mode fields, and saves:

- `currentDiedTimes` at `+0x18`
- `currentVillage` at `+0x20`
- `roundScore` and `ascensionScore` together at `+0x28/+0x2c`

It preserves current/best ascension, best village/score, previous ascension
score and `showedNewCharacters`. There is no DeInit or UI-event invocation in
these reset/abandon bodies. `Save` (`0x3ea450`) forwards to the immediate setter
at `0x387eb0`, which serializes its receiver and calls `PlayerPrefs.SetString`
with the exact key `SavedRoguelikeStandard`.

## Partial failures and small accessors

Native writes persist when a later gateway fails. Stage completion can leave
its village/score updates before class initialization or village advancement.
Ascension completion can leave its new ascension, cleared best village and
previous-score capture before class initialization. Reset's first class-init
failure happens before its reset writes. A failure at the first or second JSON
or SetString call retains the reset state; failure at the first save prevents
the caller's second save. An injected virtual-dispatch failure in OnFailed
retains the incremented death count before reset. Failure fixtures stop at the
gateway and do not emulate managed exception unwinding.

The supplemental getters are verified independently:

| Declarations | Native effect |
| --- | --- |
| GetCurrentLevel, GetStartingLevel, GetResetLevel | Return `currentVillage` at `+0x20`. |
| GetCurrentRunScore | Return wrapping `ascensionScore + roundScore`; do not include previous score. |
| CanResetLevel | Return true when current village or death count is strictly positive. |
| GetGameMode, GetMaxLevel, GetScore | Return int32 zero from their shared folded body. |
| IsLocked | Clear AL only, returning false through the Boolean ABI. |
| UpdateScore | Shared `ret 0` no-op; supplied score/level do not update fields. |

## Verification and remaining services

`reverse_engineering/scripts/audit_roguelike_standard_progression.py` pins
GameAssembly and Dumper inputs, verifies exact metadata signatures and mode
field declarations, checks complete native instruction boundaries and eighteen
selected instruction relationships, then executes 1,498 cases in Unicorn 2.1.4.
Every case compares all 128 synthetic mode-object bytes, static village state,
event order, and each save snapshot. Successful returns verify preserved
integer registers and the stack. Cases include repeated-save failure indices,
cold class initialization, deaths around the threshold, INT_MIN/INT_MAX fields,
300 seeded random int32 states through each core progression operation, and
Boolean return tests with nonzero upper register bits.

The 144 executed instructions cover every instruction outside the explicitly
listed eighteen warmed-metadata instructions and two redundant class-init call
sites skipped after the first successful initialization. The report separates
these exclusions instead of claiming full native instruction coverage.

Class initialization, `GameData.IncreaseVillage`, JSON serialization and
PlayerPrefs are explicit gateways. In particular, the village-advancement
helper's own side effects are not projected; mode `currentVillage` writes and
static reset writes are native. No live save/state is changed. This does not
audit showcase decisions, current/previous ascension selectors, character-kill
callbacks, all subclass overrides, or full serializer behavior.

The authored report is
`reverse_engineering/reports/f530404b0f3f_807de4a83df4_roguelike_standard_progression.json`.
With the pinned local reverse-engineering dependencies on `PYTHONPATH`:

```powershell
python reverse_engineering/scripts/audit_roguelike_standard_progression.py 'B:\SteamLibrary\steamapps\common\Demon Bluff Playtest' 'B:\CodexTools\DemonBluffReverseEngineering\artifacts\f530404b0f3f_807de4a83df4\il2cppdumper-v6.7.46' --output reverse_engineering/reports/f530404b0f3f_807de4a83df4_roguelike_standard_progression.json
python -m py_compile reverse_engineering/scripts/audit_roguelike_standard_progression.py
```

## Offline Rust replay

`crates/solver-core/src/roguelike_progression.rs` implements the versioned
`roguelike_progression_native_v1` contract. Its five focused tests compare all
1,138 applicable native progression fixtures, retaining wrapped fields, indexed
service failures, class-initialization state and both save attempts. Callers
must explicitly supply initialized metadata, the audited abandonment override
and preserving-service provenance. The village helper remains a gateway here;
the separate [village bridge](roguelike_village_bridge.md) opts into its native
side effects without weakening this original contract.
