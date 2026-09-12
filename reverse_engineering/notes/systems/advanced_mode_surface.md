# AdvancedMode declared native surface

Pinned build: `f530404b0f3f_807de4a83df4`; GameAssembly SHA-256
`F530404B0F3F28479A7CE21D5738C4E36C2A0A03E1B5520092975B4150D819EC`.

`audit_advanced_mode_surface.py` verifies 21 exact Dumper declarations for
`AdvancedMode` (tdi5935), excluding the two ascension getters already covered by
`ascension_setup.md`. The report contains their exact names, RVAs, and
signatures. Together these audits cover the class's 23 declarations at explicit
service boundaries. No typed-target manifest was changed.

Validation: **295 native Unicorn 2.1.4 cases**, nine instruction/call-sequence
assertions, 515 visited instructions. Inputs include signed integer boundaries,
wrapped additions, delegate null/cast/failure results, null character/data,
class initialization failure, persistence failure, achievement failure, UI
failure, and formatter failure. All inputs are authored fixtures; the pinned
private PE supplies executable instructions at runtime. No binary bytes or
native/decompiled bodies are stored in the repository.

## Lifecycle and loading

`Init` 0x3D1960 combines delegates into GameplayEvents in this order:
character-killed (+0x48), stage-completed (+0x20), failed (+0xB0). The stage
callback comes from the mode's virtual method slot; the other two use their
exact method metadata. After all three assignments, Init reads **currentScore**
and publishes it to GameData.CurrentVillage (+0x18 of static storage), after
GameData class initialization if needed. Init does not reset any mode fields.

`DeInit` 0x3D1420 visits the same event order. Its first operation is **Combine**
for character-killed, followed by Remove for stage-completed and failed. This is
the shipped call sequence, independently asserted against instructions and
executed fixtures; it must not be normalized to three removals. The resulting
invocation-list contents remain the delegate service's responsibility. DeInit
does not modify the mode fields or CurrentVillage.

Delegate-service failure or invalid cast retains prior event assignments. A
null result is a legal event assignment. Null `this` reaches the first delegate
construction and event assignment, then fails before construction of the
virtual stage callback. A failing GameData initializer during Init occurs after
all event assignments and before CurrentVillage publication.

`LoadGame` 0x3D1D70 forwards to SavesGame.get_AdvancedMode with null MethodInfo;
it returns that service's result, including null. Existing SavesGame evidence
covers the SavedAdvanced key, default allocation, and JSON service boundary.

`OnLoadGame` 0x3D1E40 changes state only when highscoresReset is false and
currentSavedVillage is signed-greater than allSavedVillagesEver. It then sets
highscoresReset true and copies currentSavedVillage into allSavedVillagesEver.
A false flag remains false when the comparison fails. There is no save here.

## Progression, achievements, and scoring

Field layout: currentScore +0x10, bestScore +0x14, roundScore +0x18,
currentSavedVillage +0x1C, currentSavedVillageHighscoreTracker +0x20,
allSavedVillagesEver +0x24, bestOverallSavedVillage +0x28, highscoresReset +0x2C,
showedNewCharacters +0x2D.

`OnStageCompleted` 0x3D1E60 wraps each of currentSavedVillage,
allSavedVillagesEver, and currentSavedVillageHighscoreTracker upward by one,
and clears roundScore. If the updated currentSavedVillage is at least ten
(signed comparison), it requests achievement `Lilis_ACHIV_5030`. It then sets:

- allSavedVillagesEver = signed max(currentSavedVillage, allSavedVillagesEver).
- bestScore = signed max(currentScore, bestScore).
- bestOverallSavedVillage = signed max(currentSavedVillageHighscoreTracker,
  bestOverallSavedVillage).

Finally it forwards this mode to SavesGame.set_AdvancedMode. Achievement
failure retains increments and round-score reset but prevents all three maxima
and saving. Save failure occurs after those updates. Wrapped INT_MAX becomes
INT_MIN before the achievement threshold and maximum comparisons.

`CheckAchievements` 0x3D13D0 requests the same achievement when
currentSavedVillage >= 10, without updating fields or saving. Achievement
unlock internals remain the existing ProjectContext service.

`OnFailed` 0x3D1E20 clears currentScore, roundScore, currentSavedVillage, and
currentSavedVillageHighscoreTracker, then saves. It preserves bestScore,
allSavedVillagesEver, bestOverallSavedVillage, and both flags. A failing save
retains all four resets. `Save` 0x3D1EE0 only forwards this mode to that setter.

`OnCharacterKilled` 0x3D1D80 requires a nonnull character and Character.data
(+0x50). Only CharacterData.type (+0x130) values Minion=30 and Demon=100 award
points. It queries GetUnrevealedCharactersCount and adds the signed 32-bit
wrapped result of `10 * (count + 5)` to both roundScore and currentScore. It
then invokes UIEvents.OnUIUpdate if present. A count-service failure leaves
scores unchanged; UI failure follows both score writes. Other types do not
query the count or notify. There is no save in this callback. Selection and
counting internals are explicit services, not reconstructed by these fixtures.

## Accessors, locking, and presentation

`GetGameMode`, `GetMaxLevel`, `GetCurrentLevel`, and **GetScore** share 0x3712B0
and return integer zero. GetScore does not expose the currentScore field.
`GetResetLevel` and `GetStartingLevel` share 0x379600 and return
currentSavedVillage. `CanResetLevel` 0x3BCC90 clears AL, returning false while
preserving the rest of RAX; callers must use the Boolean ABI. `UpdateScore`
0x33ED50 is inert, and the constructor 0x357920 forwards to that inert base body.
The constructor performs no field stores; a fresh allocation's zero state
belongs to managed allocation, not an explicit field initializer.

`IsLocked` 0x3D1CF0 ensures GameData is initialized. DebugMode immediately
returns false without loading a saved mode. Otherwise it loads
SavesGame.RoguelikeStandard, rejects null, and returns the signed comparison
`bestAscension < 3` (that class's +0x10 field). This is bestAscension, not current
village or current ascension.

`GetScores` 0x3D17E0 boxes bestScore, bestOverallSavedVillage, and
allSavedVillagesEver in that order, formats the three pinned rich-text strings,
and concatenates an initial newline plus those three results. The labels are
Highest Score, Saves in a row, and Saved villages. Each format string ends in a
newline. `GetSummaryScores` 0x3D18F0 boxes currentScore and formats the pinned
Score text followed by a blank line and `Score resets on round loss.` The audit
asserts the entire format inputs including existing tags and newlines; no
markup normalization is implied. Formatting failures stop subsequent boxes,
formats, and concatenation. Fields remain unchanged.

## Boundaries and reproduction

External services: delegate allocation/construction/combine/remove and runtime
casts; GameData initialization; SavesGame getters/setter and underlying JSON;
achievement unlock; unrevealed-character counting; UI callback; boxing and
String.Format/Concat. Native code through each boundary is executed, with
controlled results and failures. This does not claim serializer, UI, runtime
allocation, or achievement implementation closure. Engine JSON internals remain
bounded by the separate gateway/constructor audits, with core serialization
explicitly outside scope.

```powershell
$env:PYTHONPATH='B:\CodexTools\DemonBluffReverseEngineering\python-emulation'
python reverse_engineering/scripts/audit_advanced_mode_surface.py --game-root 'B:\SteamLibrary\steamapps\common\Demon Bluff Playtest' --dumper-root 'B:\CodexTools\DemonBluffReverseEngineering\artifacts\f530404b0f3f_807de4a83df4\il2cppdumper-v6.7.46' --output reverse_engineering/reports/f530404b0f3f_807de4a83df4_advanced_mode_surface_audit.json
```
