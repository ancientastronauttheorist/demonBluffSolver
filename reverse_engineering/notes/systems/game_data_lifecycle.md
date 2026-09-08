# GameData lifecycle and catalogue boundary

Pinned build `f530404b0f3f_807de4a83df4`. These twelve targets complete the
23-method GameData declaration together with the eleven methods in
[ascension setup](ascension_setup.md). The
[native audit](../../reports/f530404b0f3f_807de4a83df4_game_data_lifecycle_audit.json)
passes 189 cases with explicit service gateways.

## Mode and state publication

ChangeGameMode first calls the supplied mode's virtual LoadGame. When the old
global mode exists, it then calls old.DeInit and loaded.Init, in that order.
The new Init still sees the old global reference. Only afterward does the
method publish the loaded result and invoke optional OnGameModeChanged, then
OnGameInit. A null input fails before LoadGame. With an existing old mode, a
null loaded result fails after DeInit and before publication.

The old-null branch skips both DeInit and Init and can publish a null loaded
result. Callback failures before publication retain the old reference; event
failures after publication retain the new one. The harness stops at an explicit
failing gateway; it does not emulate managed exception unwinding.

ChangeGameState always moves the old state to PrevState, stores the requested
state, then invokes optional OnGameStateChange. Equal old/new values do not
suppress the event. UpdateCurrentVillage directly stores the supplied int32,
without range checks.

IncreaseVillage recognizes StandardMode and RoguelikeStandard by runtime type
assignability. StandardMode increments only below standardAscensions.Length - 1;
RoguelikeStandard increments only below its MaxLevel result. Other modes and a
null mode do not increment. Negative values are not clamped. Required missing
Standard configuration fails before an increment.

## Initialization and references

Init publishes the instance localization reference to GameData and the
character localization provider before loading save data. It reads the key
from the existing save object, calls PlayerPrefs.GetString and tests for an
empty result. A nonempty result causes a second GetString call and
FromJson<SavedGameInfo>; an empty result allocates a new SavedGameInfo and
calls its constructor. It stores the resulting save reference, then calls
CharacterData.LoadPreferences for every catalogue occurrence in order.

Earlier localization/save writes survive a later null catalogue or null entry
failure. Duplicate entries cause repeated preference loads. This body does not
copy or clear ascension pools. JSON, PlayerPrefs and individual preference
loading remain service boundaries; this is not a claim about complete startup.

GetCharacterDataOfId and GetCharacterDataOfName scan allCharacterData, using
static String.Equals against each asset's ID or characterName respectively.
They return the first match and null if absent. Null query strings are valid
arguments to String.Equals; a null asset fails when reached. GetAllRelics
returns the exact stored list reference, including null.

The instance constructor allocates catalogue and relic lists and calls the
ScriptableObject base constructor. This constructor is checked statically.
The native static constructor allocates StandardMode, sets its failScoreDecrease
to 20, publishes it, sets CurrentVillage to zero and clears the three debug /
trailer flags. It does not explicitly reset GameState or PrevState; runtime
zero initialization is a separate mechanism.

## Achievement store

GetUnlockedAchieves allocates a fresh list, scans allAchievements in array
order and retains every occurrence whose ID is in SavesGame's unlocked
achievement list. It retrieves that store for each entry. This achievement
store is separate from SavedGameInfo's character unlock IDs.

UnlockAchievement scans the whole achievement array and calls
AchievementData.DebugUnlockAchiv for every matching ID, rather than stopping
at the first. A later null asset fails after earlier callbacks or result-list
additions. Achievement mutation itself is an explicit harness gateway: this
offline audit does not write game saves or unlock achievements.

## Scope and reproduction

The [harness](../../scripts/audit_game_data_lifecycle.py) pins the DLL and exact
Dumper signatures, executes native caller instructions in Unicorn 2.1.4 and
checks preserved registers and stack on normal returns. Allocation, collection,
string, mode virtual, delegate, JSON, preferences and achievement services are
explicit gateways. The repository contains authored findings and metadata;
native bodies and instruction bytes remain private.

The cases cover 32 mode combinations, five callback failure boundaries,
18 state transitions, five direct village stores, 32 catalogue lookups, ten
initialization paths, 72 mode-specific increments, one static constructor,
two relic-reference reads and twelve achievement paths.

The 47-set typed inventory now contains 151,760 datatypes, 968 memberships,
615 exact definitions and 509 native RVAs. The read-only pass validated 2,804
parameter locations with zero program mutations. The coverage overlay checks
1,216 classifications and 287 evidence records; all 32 reverse-engineering
tests passed.

All twelve baseline and typed exports completed and the quality check passed.
Placeholder parameter tokens fell from 39 to zero; two error markers and
17 warning markers remain in both export sets. Typed signatures improve
readability without making the decompiled C universally complete.
