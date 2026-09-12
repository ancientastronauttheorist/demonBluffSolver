# Mode-selection UI callers

Build `f530404b0f3f_807de4a83df4`. The authored harness `scripts/audit_mode_selection_ui.py` pins GameAssembly and Dumper script/dump hashes. Its report `reports/f530404b0f3f_807de4a83df4_mode_selection_ui.json` records 70 passing native fixtures and all seven exact declarations of ChangeGameModeButton and GameModeCard. Six distinct native bodies have completed private baseline exports. This is supplemental metadata evidence; no new typed target set is introduced.

## Selection

ChangeGameModeButton.OnClick (397440) reads its mode reference at +20, initializes GameData if required, and forwards that captured reference to ChangeGameMode. It does not independently call IsLocked, inspect a button enabled flag, or reject a null mode. A cold class-init failure prevents the call. ChangeGameMode remains a service here; its separate native lifecycle and composition audits establish downstream null and publication behavior.

## Card refresh

GameModeCard.UpdateView (3A1300) reads gameMode at +20. An initially null reference takes the locked visual branch directly. Otherwise it calls virtual LoadGame (slot6), stores that returned reference back to +20 with a GC barrier, then calls returned-mode OnLoadGame (slot7), GetScores (slot22), and the text object's setter (+558). It uses the returned identity even when distinct from the original card reference. A null returned mode fails after the reference has already been replaced with null.

After text update, the native type hierarchy test accepts StandardMode or a derived runtime type. If that mode's currentCompleted byte (+34) is nonzero, UpdateView calls virtual AbandonRun (slot15/+228). It then calls IsLocked (slot21) and updates three UI services in order:

1. locked GameObject.SetActive with the locked result;
2. buttonController Behaviour.enabled with its inverse;
3. eventTrigger Behaviour.enabled with its inverse.

Thus refreshing a completed Standard card can abandon/reset the loaded run through its virtual callback. The text was obtained and assigned before that callback. This audit executes the call site and argument/identity flow; AbandonRun's internals remain a gateway here and are separately audited in Standard progression. It does not claim that every actual refresh reaches the completed branch or reconstruct persistence timing across real UI events.

GameModeCard.Lock (3A1060) executes the same three locked-state UI calls without loading or querying a mode. Native null checks occur immediately before each UI service. Missing text fails after GetScores; a missing locked object, button or trigger preserves earlier calls. The fixture respects Boolean byte arguments rather than requiring unrelated upper register bits to be zero.

## Subscription lifecycle

OnEnable (3A11D0) constructs the instance UpdateView delegate, combines it into UIEvents.OnUIUpdate, checks the returned Action type, stores it and executes the GC barrier, then directly calls native UpdateView. A refresh failure therefore leaves the subscription installed. Repeated enable operations can add equal entries; there is no caller-level duplicate guard.

OnDisable (3A10B0) constructs an equal UpdateView delegate, removes one matching occurrence and stores the result, including null. It does not call UpdateView. Tests preserve unrelated subscribers, exercise duplicate entries, null removal results, and cast/service failure before the store. Delegate list behavior is an explicit identity gateway supported by the separate native delegate audit, not an assertion of observed live subscriber counts.

Both declared constructors share 33E820, which tailcalls UnityEngine.Behaviour's constructor. The shared native alias name is not used as a substitute for either exact managed declaration.

## Validation and boundary

The matrix covers ordinary and noncanonical nonzero completed bytes, Standard/non-Standard runtime types, locked/unlocked results, distinct/same loaded identities, initial/returned null modes, missing UI references, each ordered mode/UI gateway failure, cold/failing GameData class initialization, subscription failures and constructors. Normal returns verify the stack and all eight Windows nonvolatile integer registers. Exact ranges retain internal failure traps while excluding trailing alignment padding.

Mode virtual callbacks, UI setters, class initialization, delegate allocation/construction/list operations and the Behaviour constructor are gateways. Those gateways preserve the card's references; arbitrary callback mutation/reentrancy, actual scene wiring and live UI services are not reconstructed. There is no save or process interaction. Reproduce with `python scripts/audit_mode_selection_ui.py GAME_ROOT DUMPER_ROOT --output REPORT`, using Unicorn 2.1.4 from the private emulation PYTHONPATH.
