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
surface; it is not called by these three methods. This audit does not remove
the existing writer's presentation-side-effect restrictions or automatically
join the tail into ordered Reveal replay. That integration and provenance for
coroutine readiness remain subsequent work.

The coverage ledger adds three `understood` classifications and one native-static
evidence record: 532 classified methods and 276 evidence records, with the
4,207-method / 3,066-native-body denominator unchanged.

Validation passed 588 Rust library tests, 778 Python tests, 13 reverse-engineering
tests, release build, formatting and coverage integrity. The long simulation
suite was not rerun because this adds an offline module without live/scenario
callers. AGENTS.md also records the PowerShell ripgrep glob form after an
unexpanded wildcard path caused a read-only search failure during this audit.
