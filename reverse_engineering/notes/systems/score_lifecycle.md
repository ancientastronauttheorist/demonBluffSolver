# Native Score lifecycle and Gameplay composition

Pinned build `f530404b0f3f_807de4a83df4`. `scripts/audit_score_lifecycle.py` verifies GameAssembly, Dumper script and dump hashes, all 16 Score/ScoreNew/ScoreOld declarations and exact signatures, declared virtual slots, field offsets, complete next-managed-entry decoding, and key arithmetic instructions/constants. The 4,703 passing cases execute 803 distinct native instruction addresses across all 14 distinct Score bodies and Gameplay.UpdateScore. The report is `reports/f530404b0f3f_807de4a83df4_score_lifecycle.json`.

## Base and New Score

Score and ScoreNew have distinct entries for GetBaseScore, GetMultiplier, AddPointsOnEvilKill and UpdateFullScore with the same tested behavior. GetFullPoints and the constructor are folded aliases. Score is abstract in metadata; its body fixtures are controlled receivers, not a claim it is instantiated by the game.

GetBaseScore starts at zero, obtains Gameplay.Instance.GetAllCurrentCharacters(), sums each CharacterData.roguelikeInfo.point, then sums CurrentRelics' roguelikeData.point. Every addition wraps int32. GetMultiplier starts at float32 1, multiplies the same two ordered sources' pointsMult fields, then calls CharactersHelper.GetUnrevealedCharactersCount. The result is `(float32(min_signed(unrevealed,3)) * 1.25f) * accumulated_product`, with rounding after each operation. There is no lower clamp: negative counts yield a negative factor, zero yields zero, and zero times infinity yields NaN. This includes character factors and the unrevealed factor; Gameplay.GetScoreMultiplayer only multiplies relic factors.

AddPointsOnEvilKill dispatches actual slot-4 GetBaseScore, then slot-5 GetMultiplier. It converts the base integer to float32, multiplies, and uses x86 CVTTSS2SI truncation. NaN, infinity and out-of-range results yield the int32 indefinite value (-2147483648) with the tested masked-exception MXCSR. It wrapping-adds the result to roundPoints before testing the Character argument or Character.icon, resolving transform/position, or calling VfxController.SpawnFloatingScore. Every later null/provider/VFX failure preserves the score write. The report contains exact binary32 inputs/returns rather than JSON NaN numbers.

UpdateFullScore requires Gameplay.Instance, writes completedStages = currentLevel+1, wrapping-adds roundPoints into overallPoints, writes completedDays = argument+1, and clears roundPoints. It then initializes GameData if needed, captures current GameMode, executes actual slot-8 GetFullPoints, and dispatches mode slot-17 UpdateScore(score,currentLevel+1). GetFullPoints is `completedDays * pointsForCompleting + overallPoints` with int32 wrap. Failures after the flush retain all flushed fields. Gameplay.UpdateScore is executed as a composed entry: it uses its receiver currentDay and the static Score's concrete updater; that updater separately reads Gameplay.Instance for currentLevel. The fixture supplies the same Gameplay identity for these roles and does not infer identity equality in the game.

## Old Score

ScoreOld inherits base GetBaseScore/GetMultiplier, but its kill method does not call them. It adds the unrevealed count to tempUnrevealedCards, increments tempKilledEvils, and displays `unrevealed * pointsPerUnrevealed + pointPerKill`, all with int32 wrap. These counter writes occur before Character/icon/transform/VFX checks.

Its full update writes completedStages and completedDays, adds tempKilledEvils to killedEvils and tempUnrevealedCards to unrevealedCards, and clears those two temporary counters. It leaves roundPoints and overallPoints unchanged. Its GetFullPoints is `pointPerKill*killedEvils + pointsPerUnrevealed*unrevealedCards + completedDays*pointsForCompleting`, with int32 wrap. It then follows the same mode slot-17 dispatch and partial-failure ordering.

Base/New constructors only write pointsForCompleting=100 before the object-constructor gateway. Old additionally writes pointPerKill=50 and pointsPerUnrevealed=10. Nonzero sentinel fixtures verify all other state bytes are preserved; fresh-allocation zeroing is a separate runtime responsibility.

## Fixture boundary and report format

Metadata is warmed. Runtime-class cold paths are tested through explicit preserving initializers. Gameplay.GetAllCurrentCharacters, list enumeration/disposal, unrevealed-count lookup, engine transform/position, floating-score VFX, and mode.UpdateScore are supplied gateways. No callback overrides or provider mutations occur. Exact event snapshots and argument order are compared before report serialization; repeated snapshots are stored once in event_state_table and event.state_index references that table. Each case retains full initial/final score fields. This compact format retains partial-failure evidence without repeating the same state at every enumeration step.

The suite covers signed extrema, overflowing sums/products/counters, float32 integer rounding, negative and zero factors, gradual-underflow subnormals, infinity, signaling/quiet NaNs, ordered multiplication, null sources/elements/info objects, explicit getter/provider/VFX/mode failures, and successful/failing runtime initialization. All fixture object bytes are compared against the authored write set. The numeric environment is MXCSR 0x1F80; other rounding/flush modes and native managed exception unwinding are excluded. Collection implementation/version checking, live HUD behavior, real character selection, mode score persistence, startup choice of Score implementation, and service-driven global replacement remain separate.

Reproduce with the script's game-root and Dumper-root arguments plus `--output`, using Unicorn 2.1.4 from the private python-emulation PYTHONPATH. Python compilation passes. No new typed target, native bytes or live save state is included.

## Offline Rust replay

`crates/solver-core/src/score_replay.rs` matches all 4,703 native fixtures.
Its context requires native MXCSR controls 0x1F80 and preserving-service,
audited-slot and composed-receiver provenance. A read-only host check accepts
only x86-64 with matching rounding, exception-mask and gradual-underflow
controls; sticky exception-status flags are ignored. Production does not
change the host environment. Five focused tests cover the native corpus,
partial failures, bounded inputs and thread-scoped unsupported-control checks.
