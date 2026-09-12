# Standard run reset during card refresh

Build `f530404b0f3f_807de4a83df4`. `scripts/audit_card_standard_reset.py` and `reports/f530404b0f3f_807de4a83df4_card_standard_reset.json` extend the mode-selection caller audit with **19 native composition fixtures**. GameAssembly and Dumper script/dump hashes are pinned. Existing lifecycle/progression manifests verify nested declarations; no new typed targets or Ghidra exports are required.

## Concrete native path

The fixture executes GameModeCard.UpdateView, StandardMode.LoadGame and its SavedStandard getter, inherited GameMode.OnLoadGame (`ret 0`, preserving RAX), StandardMode.GetScores and the saved maximum getter, StandardMode.AbandonRun and its JSON save setter, and StandardMode.IsLocked. The synthetic runtime class points virtual slots at these actual native bodies. LoadGame returns a controlled distinct loaded object and the native card stores it.

When loaded currentCompleted is nonzero (including byte7), UpdateView obtains and assigns score text **before** entering AbandonRun. That method first calls virtual DeInit. After it returns, the actual native instructions clear score, currentLevel, currentDiedTimes, roundScore, currentScore and currentCompleted. They preserve savedVillages, completed, bestDiedTimes, bestScore and failScoreDecrease. The native save caller then invokes compact JSON serialization followed by PlayerPrefs.SetString. The card eventually reaches native IsLocked=false and the three unlocked-state UI calls.

This joins the former UpdateView-to-AbandonRun boundary without claiming real serializer durability. Virtual DeInit remains an explicit gateway here; its actual Standard lifecycle is independently audited, and its failure is exercised before reset writes.

## Bounded callback reentry

The UIEvents.OnUIUpdate fixture holds a delegate with invoke pointer equal to native GameModeCard.UpdateView, target equal to the card and the observed method argument slot. AbandonRun's **native indirect delegate call** invokes that pointer; the harness does not replace the notification with a simulated refresh call.

Under the current-state JSON gateway, the nested native LoadGame returns the already reset loaded object. The second UpdateView sees currentCompleted=0 and does not abandon again. The fixture completes with two native refresh entries, one native reset/save, nested unlocked UI calls, and then the outer unlocked UI calls. Without an installed callback there is one refresh. Initially uncompleted runs never enter reset/save or callback reentry.

A separate adversarial JSON gateway restores currentCompleted before every return. Native execution then reenters refresh repeatedly. The audit stops at the fourth refresh entry after three reset/save cycles and labels the result `reentry_bound`. This proves the conditional dependency on returned state; it does **not** establish infinite recursion in the real game or assert that PlayerPrefs/JSON returns stale completed saves.

## Failure chronology

Injected failures at preference read, JSON load, saved-maximum read, score concatenation, text assignment or DeInit leave the mode's run fields intact. JSON-save and SetString failure occur after reset and before notification. UI setter failures retain reset state and whichever earlier nested calls completed. A null JSON result fails after the card has stored null. The report records full modeled state at every gateway, making publication and reset order inspectable.

Successful fixtures verify the stack and all eight Windows nonvolatile integer registers. The actual method ranges include complete final instructions and internal failure traps but exclude trailing padding. Preferences/JSON, boxing/formatting, UI setters, and DeInit remain explicit gateways. This is a synthetic single-delegate topology, not the complete UI event history, Unity scene wiring, multicast reentry, or managed exception unwinding.

Reproduce with `python scripts/audit_card_standard_reset.py GAME_ROOT DUMPER_ROOT --output REPORT`, making private Unicorn 2.1.4 available through PYTHONPATH.
