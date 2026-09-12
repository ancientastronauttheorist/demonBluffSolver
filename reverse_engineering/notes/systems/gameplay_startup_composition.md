# Native startup and saved-roster composition

Build `f530404b0f3f_807de4a83df4`. `scripts/audit_gameplay_startup_composition.py` composes the actual Gameplay.Init (37DEF0) caller with ResetSavedCharacters (37FDE0), and executes RestartGame (37FFC0) as a contrasting entry. It pins GameAssembly and Dumper inputs, validates exact signatures and complete entry decoding, and carries the previous bounded ScoreOld allocation-reference checks. The report contains 117 passing native cases and 399 executed instruction addresses; Python compilation passes.

Init first executes the complete saved-roster reset caller. After the Unity null check, ResetSavedCharacters rereads ProjectContext.Instance and captures its gameData. In order, typed-pool requests 10, 20, 30 and 100 feed new list-copy allocations, which replace savedTownsfolks (+48), savedOutsiders (+50), savedMinions (+58) and savedDemons (+60). Returning from this body, Init requests gameplay state 1, clears its deck and static relic list, then makes four further copies of those saved lists into current lists +28/+30/+38/+40. Saved lists and current lists retain distinct allocation identities while containing the same supplied elements. It next constructs and publishes ScoreOld with defaults100/50/10, queries two independent starting levels, conditionally loads characters and schedules InitCoroutine.

RestartGame does not call ResetSavedCharacters. Its immediate ResetPlayerInfo body has no saved-roster-reset call either; that player reset remains a gateway here. RestartGame copies the existing saved list identities, obtains a reset level, and only then constructs/publishes ScoreOld. A Rust replay must preserve this difference rather than implicitly refresh a restart roster.

The composed fixtures include failure at every reached gateway occurrence, all four null typed-pool returns, score allocation and constructor failures, null project/game/deck/relic/mode references, and signed selector extrema. The successful outcome assertions verify exact saved/current content and identity relationships and publication order. Failure runs match the corresponding successful event/state prefix. Earlier saved-list publications survive later reset failure. Reset failure prevents current-list and Score replacement; failure after reset may leave all four saved lists refreshed while later state is only partly updated. A mode-null failure in Init occurs after Score publication; a mode-null failure in RestartGame occurs before it.

A true Unity-null result skips saved replacement and Init continues using old saved lists, even when the supplied managed project pointer is null. Conversely, a false Unity-null result followed by ProjectContext replacement with null fails at the subsequent reread. Replacement after the first typed-pool call does not change the captured gameData used for the remaining three requests, and the full initialization can complete with the ProjectContext global now null. These are explicit controlled provider mutations, not live race or engine scheduling claims.

The snapshot table records references, counters, logical list contents and headers, copy sources, project presence, current Score defaults and allocated-but-unpublished Score defaults. A list's empty logical contents follow its native size write before the array-clear gateway; the fixture also supplies that gateway's backing-array clear. Metadata is warmed, while optional runtime class initialization remains explicit.

Allocator/list-copy/barrier operations, typed GameData pool bodies, Unity equality, class initializers, ResetPlayerInfo, health reset, ChangeGameplayState, mode selectors, LoadCharacters and coroutine scheduling remain supplied gateways. In particular ChangeGameplayState is preserving in this fixture, so the composed model does not claim its state/event effects. List elements are synthetic IDs and no solver truth or live save content is involved. No complete startup ordering, native allocation failure unwinding, or all-indirect-writer absence claim is made. Reproduce with game-root/Dumper-root positional arguments and `--output`, with Unicorn2.1.4 on the private emulation PYTHONPATH.

## Offline Rust replay

`crates/solver-core/src/gameplay_startup.rs` exposes the versioned startup replay
with explicit list and score identities, captured provider behavior, copy
lineage and partial publication snapshots. It rejects overlapping list/score
identity registries and unsupported service provenance. Three tests replay all
117 native fixtures and check strict input rejection. The complete release
library suite passed 735 tests. This model is offline and does not supply engine
scheduling or callback internals.


## Shared list identities

Six additional fixtures rewrite real list pointers before the initial snapshot.
When deck and relics share one list, Init clears that object twice: the first
clear empties it and wraps version `0xFFFFFFFF` to zero, and the second increments
the version to one without another array-clear call. A failure at the first
array-clear gateway retains the empty logical list/version zero and prevents
current-list and Score replacement.

When savedTownsfolks shares the deck, Init with a supplied true Unity-null result
keeps that saved reference, then clears the deck before copying current rosters.
Its new current Townsfolk list is consequently empty. Restart copies the same
saved/deck object without clearing it and retains its original elements.
When savedTownsfolks shares the old current Outsiders object, both entries copy
that original object's elements; replacing the current Outsiders field later
does not redirect the saved reference or mutate the original list.

These are pointer-identity fixtures, not merely lists with equal content. All
117 cases pass native execution and the existing Rust differential fixture test
now consumes the expanded corpus; Rust validation is coordinated separately.
