# Reveal view tail

Build `f530404b0f3f_807de4a83df4`; native-static audit and bounded offline
projection in `bluff::reveal_view`. This closes three previously unclassified
method bodies, rather than claiming a complete renderer or Unity scene model.

## Native boundary

| Method | Method ID | RVA |
| --- | --- | --- |
| Character.RefreshView | tdi5487.m0063 | 0x367B60 |
| Character.UpdateView | tdi5487.m0064 | 0x3695A0 |
| Character.UpdateViewReal | tdi5487.m0065 | 0x3694D0 |

Both game-input SHA-256 fingerprints matched the checked-in build manifest
before extraction. The existing typed Ghidra project was opened read-only with
analysis disabled. `ExportFunctionDecompilations.java` exported all three
metadata-selected bodies without failure or cancellation. Their new method
prototypes were not installed: UpdateView propagated Character fields, while
UpdateViewReal and RefreshView retained raw field accesses. Those offsets were
cross-checked against TypeDefIndex 5487 in dump.cs. This is not a new typed-quality
report or expansion of the 41-set typed target union.

The public audit report records exact signatures and RVAs. Private inputs and
exports are under the build-keyed artifact root in `reveal-view-audit-targets.json`
and `reveal-view-audit/`. Reproduce with the existing exporter using its output
directory and the report's `functions` list as target JSON, against the saved
typed project with `-readOnly -noanalysis`. No proprietary body is checked in.

## Identity and calls

UpdateView chooses background and border colors from the existing
GetCharacterBluffIfAble surface: current data when state is Dead 20, Revealed 30,
the separate revealed flag is true, or raw bluff is Unity-null; otherwise raw
bluff. Register-as and the copied action-role pointer do not participate.
UpdateViewReal always uses current data colors. Both finish with RefreshView.

Character.Reveal (`0x368410`) selects its name/art path using raw bluff alone.
The null/destroyed path calls RevealReal, which calls UpdateViewReal, and then
the common final UpdateView. The live path calls UpdateView, an explicit
RefreshView, and the common final UpdateView. Therefore the successful paths
perform two or three RefreshView calls respectively. A live bluff on a dead or
revealed body can supply name/art while colors use current data. The projection
records the source identities without synthesizing pixels or localized assets.

## Refresh writes

RefreshView first hides the pickable object when pickableUses (offset 0xDC) is
less than one. Positive uses do not force it active. It does not alter the count.

When state (0xE4) is Dead and createdDeadPrefab (0x98) is Unity-null, it creates
the deadPrefab under the Character, stores that reference, positions it at the
icon, resets its orientation, and activates ripView (0x78). Later refreshes
reuse the stored object. An already-existing death presentation does not force
ripView active; nondead state does not destroy an existing death presentation.

Finally, an absent/destroyed disguiseIcon (0x88) skips the icon update. A true
killedByDemon byte (0xED) also skips it, preserving the prior active state.
Otherwise the icon is active exactly when state is Dead or Revealed and raw
bluff is live. The separate revealed flag does not enter this icon predicate.
The routine neither changes Character.state/revealed nor dispatches role acts.

## Bounded implementation and limits

`reveal_view_native_v1` accepts explicit body and UI state after gameplay
callbacks. `created_dead_presentation` represents Unity liveness, and the
optional disguise icon represents an absent/destroyed object as None. It emits
ordered color-source and refresh-write traces, preserving the input body except
for creation of the retained death presentation. Eight tests exercise repeated
calls, object liveness, color/icon predicate differences, Demon-kill preservation,
exhausted controls and schema rejection.

Required UI objects/assets must be valid and Unity SetActive/Instantiate and
other presentation lifecycle callbacks must not mutate modeled state. Native
null failures, transforms, localization/art getters, SetupArt, pixels, and
arbitrary scene callbacks are outside this projection. RefreshCharacter is a
different routine and can reset pickableUses for a supported active-ability
surface; it is not called by these three methods. The standalone audit does not
remove presentation-side-effect restrictions. The versioned integration below
adds only the audited effects; coroutine readiness still needs provenance.

The coverage ledger adds three `understood` classifications and one native-static
evidence record: 532 classified methods and 276 evidence records, with the
4,207-method / 3,066-native-body denominator unchanged.

## Ordered replay integration

The separate `ordered_reveal_writer_view_native_v2` contract now joins the
bounded view model to `replay_reveal_writers`. V2 requires exactly one UI record
per physical body, including explicit pickable/rip active flags and an explicit
nullable disguise-icon state. Missing and extra records, or an omitted icon
field, reject the context. V1 still rejects populated UI state and omits all new
UI fields from serialized output. The original standalone view API is unchanged.

Every event first performs register-as/acquisition and optional Start, then
Init/AfterRoundStart against the resulting role identities. The view tail uses
the final raw bluff and body state. Its updated retained-death-object flag is
written into the board before the next resume, while active UI states are kept
in a separate per-body map. Later refreshes therefore reuse a death presentation
created by an earlier event. Per-event traces retain visual identity sources and
the ordered tail writes without embedding duplicate full board snapshots.

Twin replacements also affect UI before the final tail, including on the other
endpoint that has not resumed. Native InitWithNoReset (`0x365720`) hides RIP
only when it destroys a live createdDeadPrefab. Subsequent replacements on the
same body see the cleared pointer. After state becomes Hidden and killedByDemon
is cleared, RefreshView hides an existing disguise icon. Pickable activity is
preserved because uses is reset to one. RefreshCharacter (`0x367970`) cannot
activate that control on a Hidden body; its possible count reset remains one.
This allows the modeled net UI changes to be applied in replacement order from
the Start trace, before the final Reveal tail, without running scene callbacks.

`replacement_views` records the aggregate RIP/disguise writes per replacement,
not every duplicate RefreshView call. Identity colors, transforms and other
unmodeled controls on a replaced non-resuming endpoint are not invented. A
missing optional icon remains absent, and a RIP object can remain active when
there was no live death presentation to destroy. Self-swaps hide RIP only on the
first replacement while retaining both continuation registrations.

Six new tests cover both endpoints, self-swap destruction, death-object reuse,
new-data reacquisition changing the view path, absent-icon/RIP preservation,
strict UI provenance and v1 output compatibility. The same whole-branch failure
and retained-state budgets apply; UI snapshots and traces count toward capacity.
The integration still requires valid required UI objects and inert Unity/asset
callbacks, and does not infer coroutine readiness or arbitrary intervening events.

Integration validation passed 594 Rust library tests, 778 Python tests, 13
reverse-engineering tests, release build, formatting and coverage integrity.
Native coverage stays at 532 classifications and 276 evidence records. The
long simulation suite was not rerun for this offline-only extension; no live
solver or scenario-generation caller changed.

Validation passed 588 Rust library tests, 778 Python tests, 13 reverse-engineering
tests, release build, formatting and coverage integrity. The long simulation
suite was not rerun because this adds an offline module without live/scenario
callers. AGENTS.md also records the PowerShell ripgrep glob form after an
unexpanded wildcard path caused a read-only search failure during this audit.
