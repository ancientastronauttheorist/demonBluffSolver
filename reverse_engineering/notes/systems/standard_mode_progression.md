# Standard mode progression, scoring and persistence

This pinned-build audit covers the remaining StandardMode methods after the
[lifecycle](game_mode_lifecycle.md) and [ascension selectors](ascension_setup.md).
The [harness](../../scripts/audit_standard_mode_progression.py) executes native
caller bodies, including three SavesGame helpers, with explicit services. It
does not read or modify live saves.

## Score and completion writes

OnCharacterKilled reads `Character.dataRef.type`, not alignment, bluff identity
or registration. Only Minion (30) and Demon (100) records qualify. It requests
the current unrevealed count, then adds `(count + 5) * 10` to both roundScore
and currentScore before invoking optional UIEvents.OnUIUpdate. It does not
save. Null Character/dataRef fail before scoring; count-service failure leaves
score unchanged, while UI failure preserves both additions. Opposing alignment
and apparent-type fixtures independently verify the real-type criterion.

OnFailed increments currentDiedTimes, subtracts failScoreDecrease plus the
existing roundScore from currentScore, then clears roundScore. Before the first
completed run it copies the new death count into bestDiedTimes; afterward it
leaves that field alone. It finally saves. Signed int32 overflow wraps in these
native operations; there is no zero floor on the score. Save failures preserve
the earlier fields.

OnStageCompleted first increments savedVillages. It then obtains the Standard
ascension-array length through ProjectContext and compares the mode's own
currentLevel against `length - 1`. Missing project/game/array fails after the
village-total increment but before roundScore is cleared. After a valid lookup:

- Before the last level it clears roundScore, copies currentScore to bestScore
  only if completed is false, and calls GameData.IncreaseVillage.
- At or beyond the last level it clears roundScore, sets currentCompleted and
  completed, and retains the larger current/best values for both deaths and
  score. It does not call IncreaseVillage on this branch.

Both successful branches save. The terminal death comparison is **maximum**,
despite GetScores displaying the label `Lowest Deaths`. This audit preserves
that discrepancy. Before terminal completion, the unconditional bestScore copy
can also lower the previous best. Failure cases retain exactly the writes made
before class initialization, village progression or persistence failed.

## UpdateScore and the two level sources

UpdateScore first tests the old mode currentLevel against the configured final
index. If below that index, it replaces currentLevel with the supplied level,
without clamping or requiring an increase. It raises score only when the supplied
score is greater. It then compares that resulting currentLevel to the saved
maximum. If greater, it asks SavesGame to store **GameData.CurrentVillage**,
which can differ from both the old and supplied mode levels. It finally saves
the mode. Thus currentLevel, the global village and the saved maximum must not
be collapsed into one variable.

SavesGame's maximum property uses PlayerPrefs integer key `SavedStandardMode`,
with default zero. Its setter reads that key again and writes only if the
requested value is greater than this second result. A changed second read can
therefore suppress a write that the first comparison admitted. Negative and
extreme signed values retain their native comparisons. The separate complete
mode JSON uses string key `SavedStandard`: compact ToJson followed by SetString.
Neither helper calls PlayerPrefs.Save. Persistence and JSON internals remain
services, so the audit does not claim disk durability or serializer fidelity.

## Reset and display surfaces

AbandonRun first invokes virtual slot 8, DeInit. After that returns, it clears
score, currentLevel, roundScore, currentScore, currentDiedTimes and
currentCompleted, then saves and invokes optional OnUIUpdate. It preserves
completed, savedVillages, bestDiedTimes, bestScore and failScoreDecrease.
DeInit failure leaves those reset fields untouched; later failures preserve
their reset. The native virtual call remains a gateway in this audit.

GetStartingLevel and GetResetLevel have equivalent behavior: return zero when
Unity's object-equality service considers ProjectContext.Instance null,
otherwise return currentLevel. They do not read the saved maximum or validate
the level. CanResetLevel tests `currentLevel > 0 || currentDiedTimes > 0`.

MaxLevel returns the configured array length minus one; GetMaxLevel returns the
saved maximum. CheckIfLastLevel compares currentLevel to the wrapping int32 sum
`array length - 1 + mod`. CheckIfCompleted reads currentCompleted, distinct from
completed. GetCurrentLevel and GetScore return their own fields. GetGameMode
returns zero; IsLocked clears only the Boolean return byte and returns false.
Tests seed unrelated return-register bits to avoid inventing a full-width
Boolean ABI guarantee.

GetScores formats bestScore, bestDiedTimes and virtual GetMaxLevel plus one,
then concatenates the three resulting strings in that order. The exact markup
is retained in the report's metadata literals. Boxing, formatting and the
virtual getter are explicit gateways; arbitrary locale formatting is not
reimplemented here.

## Evidence and limits

The [report](../../reports/f530404b0f3f_807de4a83df4_standard_mode_progression.json)
records native cases for normal paths, extrema/overflow, signed comparisons,
missing references, mismatched level sources, changing preference reads,
callback failures and partial writes. Every case compares the complete 48-byte
mode-field region, including untouched bytes; normal returns verify the stack
and all eight Windows nonvolatile integer registers. Seventeen exact targets
have completed private baseline exports. Four folded simple declarations have
supplemental metadata and native execution evidence rather than guessed types
from shared Ghidra names.

Metadata is warmed; class initialization is an explicit gateway with tested
cold/failing cases. IncreaseVillage, Unity object equality, unrevealed counting,
virtual/delegate callbacks, PlayerPrefs, JSON and string services remain bounded.
Failure cases stop at the gateway and do not emulate managed exception unwinding.
This completes the declared StandardMode caller surface, not every callee or
the complete game lifecycle.

## Offline Rust replay

`solver_core::standard_progression` replays the five mutating caller actions:
Failed, CharacterKilled, StageCompleted, UpdateScore and AbandonRun. The
versioned `standard_progression_native_v1` context requires initialized
metadata and explicit provenance that services preserve modeled mode fields.
Global village and preference results are separate inputs. Null references
and a requested first service failure return the exact partial mode state and
event trace; unsupported provenance rejects the whole contract without
modifying the input. The replay records attempted saved-maximum writes, not
disk persistence.

Three Rust tests compare every output field, gateway event, save request and
failure against all 304 applicable native fixtures, reject unsupported
contracts, and preserve noncanonical nonzero completion bytes. All 679 Rust
library tests passed. No live solver strategy or gameplay automation uses
this offline replay yet.
