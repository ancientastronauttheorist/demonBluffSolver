# Native mode-transition composition

Build: `f530404b0f3f_807de4a83df4`. Reproduction: `scripts/audit_mode_transition_composition.py GAME_ROOT DUMPER_ROOT --output reports/f530404b0f3f_807de4a83df4_mode_transition_composition.json`, with Unicorn 2.1.4 available through the private emulation PYTHONPATH. The fixture pins GameAssembly and Dumper script hashes and verifies the existing three lifecycle manifests' exact declarations. No new typed targets are introduced.

The 161 passing cases join actual `GameData.ChangeGameMode` (3DBE10) to actual StandardMode and RoguelikeStandard LoadGame, their SavedStandard/SavedRoguelikeStandard getters, and both concrete Init/DeInit bodies. Thus these lifecycle virtual calls are no longer service mocks in this composed fixture. Existing independent audits retain the details of constructor/empty-save branches and native delegate implementation.

## Identity and publication

The supplied input object's LoadGame executes first, but its JSON result determines the initialized and published object. A distinct loaded object is preserved as distinct from the input. If the old global mode is null, ChangeGameMode skips both DeInit and Init: even a nonnull loaded StandardMode retains its score and does not publish its level to CurrentVillage. A null loaded result can be published on this branch.

With a nonnull old global mode, old.DeInit precedes loaded.Init. Both execute while the global still points at old. Standard Init stores its event handlers, resets roundScore and publishes its level to CurrentVillage before ChangeGameMode publishes the loaded mode. RoguelikeStandard Init makes no village update. The subsequent OnGameModeChanged and OnGameInit gateways both see the published mode. Same-object replacement has no identity short circuit: the same instance still executes DeInit and Init. A controlled cross-class JSON return follows the returned object's runtime vtable; this is an adversarial gateway fixture, not a claim about JsonUtility producing cross-class objects.

## Event consequences

Fixtures begin with one handler for each old-mode event. Old Standard teardown removes CharacterKilled, RoundWon and Died in that order; new Standard initialization appends RoundWon, Died and CharacterKilled. RoguelikeStandard initialization uses CharacterKilled, RoundWon, Died. Its teardown also uses that order but combines CharacterKilled instead of removing it.

Consequently, a successful transition away from an old RoguelikeStandard leaves two old kill callbacks and adds one loaded-mode kill callback, for three total. Same-instance replacement has three equal identity entries; distinct replacement has two old entries and one loaded entry. RoundWon and Died each finish with one loaded handler. Transitions away from Standard finish with one handler per event. These are controlled initial-list consequences, not reconstructed live event history. Identity/list operations are explicit gateways; the separate `roguelike_delegate_score` audit executes native Combine/Remove and invocation behavior.

## Failure order

The matrix crosses both old/requested classes, distinct/same/old-null/loaded-null identities, and failure at JSON, each of six delegate operations, or either notification. Every successful event-store prefix is asserted. JSON failure leaves lifecycle state untouched. Delegate failure preserves prior event stores while retaining the old global. A null loaded result with a nonnull old mode fails after all old teardown effects, including RoguelikeStandard's added kill handler. Notification failure occurs after publication; OnGameModeChanged failure prevents OnGameInit. A null input fails before LoadGame and leaves the snapshot unchanged.

## Boundary

Preferences return nonempty strings; JSON results/failures, delegate allocation/construction/list operations, casts, GC barriers and notification invocation remain explicit gateways. Runtime classes are initialized. The test does not execute notification bodies, kill callbacks, empty-save constructors, class initializers or managed exception unwinding. It proves native transition ordering and identity composition for the supplied fixture states, not a complete whole-game event history. No native code, live process state or save data is included in the report.

## Offline reconstruction

`crates/solver-core/src/mode_transition.rs` provides the opt-in
`mode_transition_native_v1` replay. Object IDs keep input, old and loaded
instances distinct, and ordered event lists preserve exact handler identity.
Native failures return completed state prefixes; unsupported provenance and
capacity overflow reject the request atomically. Nonempty preferences, warmed
metadata/classes, audited callbacks and stable external services are explicit
requirements. Optional notifications are represented without running bodies.

All three focused tests pass, including all 161 native fixtures, optional
notifications, last-matching handler removal and capacity/contract rejection.
