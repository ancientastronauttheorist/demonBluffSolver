# Full Reconstruction Roadmap

## Continuation checkpoint: 2026-10-01 native storage and refresh

Branch `codex/full-decompile` now includes the clocked Reveal adapter and
additional native timing audits. The overlay contains 1,572 classifications and 414 evidence records
against 4,207 managed definitions; these include explicit unresolved states
and are not a claim that the game is fully reconstructed. The typed union has
50 sets, 996 memberships, 643 exact definitions and 535 native RVAs, with
2,861 parameter locations validated read-only and zero program mutations.

Latest validation: 907 Rust library tests, 36 reverse-engineering tests and the release build; prior full regression: 34 simulation tests over 426 fixtures,
778 Python tests, 32 reverse-engineering tests and the release build passed.
The latest full simulation took 1,953.46 seconds while native audits ran concurrently.
New mode, roster, score and startup reconstructions are offline, with explicit service contracts.

New caller/source/init joins now include 140 complete ManageCharacters caller fixtures,
516 unique-source composition fixtures and 475 initialization/first-yield fixtures.
Bounded Rust replays compare their supported native traces. The pool-to-selector
bridge preserves joint probabilities and construction failure mass under explicit
identity, stable-script and ordered-selector provenance. See [setup caller](notes/systems/manage_setup_caller.md),
[unique sources](notes/systems/unique_source_composition.md),
[initialization](notes/systems/character_initialization.md), and
[pool ledger bridge](notes/systems/pool_ledger_bridge.md). The complete standalone
[RefreshCharacter caller](notes/systems/character_refresh.md) now executes 2,094
fixtures, ten callback mutations and twenty controlled stops. Its guarded Rust
replay matches 1,975 supported normal fixtures and preserves physical data/UI
aliases. The existing initializer join remains scoped to Hidden state; arbitrary
post-initializer states and scheduler admission still require composition evidence.
The standalone [RefreshView caller](notes/systems/character_refresh_view.md)
adds 1,645 fixtures, eight callback mutations, three retained sequences and 29
controlled stops, preserving exact transform bits and partial pointer/UI writes.
Unity liveness, transforms, Instantiate, metadata and renderer behavior remain
explicit supplied boundaries.
Its separate guarded Rust caller replay compares 1,639 normal native fixtures
and three retained sequences, preserving UI/Transform aliases, exact vector
bits and consumed fresh-allocation plans. Seven focused tests pass.
The [Character field surface](notes/systems/character_fields.md) executes
thirteen exact getters/setters in 146 cases, five stopped barriers and four
retained sequences. These leaves preserve raw references and enum widths;
`CreateRuntimeData` stores its argument without allocation or initialization.
The complete [Character constructor](notes/systems/character_constructor.md)
adds 26 fixtures, six callback probes and eleven stops, executing all 46 native
instructions. It publishes two distinct List allocations, one-use/act defaults
and the empty saved string before its supplied MonoBehaviour base gateway.
Scene ordering and serialized component production remain open. The separate
[constructor-to-initializer join](notes/systems/character_constructor_init.md)
executes 148 fixtures, three reuse sequences, four callback probes and 119 exact
stops. Constructor-produced physical Lists/defaults feed actual initialization,
Hidden refresh and first yield without replacement; scene loading and real
scheduler readiness remain supplied boundaries.
Its guarded Rust producer replay compares 153 normal profiles in five tests,
retaining complete intermediate/final Actor and physical storage/UI state.
Constructor and refresh projections remain separate; runtime traces and real
readiness are not inferred.

The actual [Manage pool composition](notes/systems/manage_pool_composition.md)
now joins both builders and their sources in 760 native fixtures. A second
[ledger bridge](notes/systems/manage_pool_ledger_bridge.md) consumes those combined
histories once and uses final current roster identities. The
[initialization producer](notes/systems/setup_initialization_batch.md) retains
physical aliases and distinct continuations, backed by twelve native repeated
initializer calls. [Action dispatch](notes/systems/character_action_setup.md)
adds 370 native Init/Start caller fixtures and a supported producer-to-writer
bridge. These remain separate guarded compositions; complete setup-to-scheduler
execution and unsupported writers are still open.
The [role callback publication join](notes/systems/character_role_callback.md)
executes RoleAct, its installed callback, delayed-result construction and an
explicit first MoveNext in 185 cases and 47 stopped prefixes. Retained callbacks
preserve their captured triggers after the role field is overwritten. Concrete
role effects, second resume/history/UI and real scheduling remain open.
The separate [history and speech publication join](notes/systems/character_role_publication.md)
adds 158 cases, four chronology/alias sequences and 196 exact stops through
explicit result and speech resumes. It preserves pointer history, wrapping Day
uses, trailer speech overrides and savedAct publication order. Registration and
resumes are supplied schedules; elapsed time and real readiness remain open.
Its guarded Rust replay compares 78 supported normal fixtures/baselines and two
distinct-record chronology sequences, retaining physical history/string aliases,
exact UTF-16 units and service order. Five focused tests pass; growth, mutations,
failures and cross-kind interleaving remain outside the accepted contract.
The [factory and direct speech entry audit](notes/systems/character_publication_entries.md)
adds six complete callers in 112 cases, four reuse sequences, one supplied UI
callback and 25 exact stops. Native captures preserve float/trigger bits; direct
speech activates UI before reloading its component. Registration remains supplied
and does not imply generator execution or readiness.
The [history/type caller audit](notes/systems/character_history_entries.md) adds
four complete callers in 116 cases, two retained alias sequences, one supplied
callback and 12 exact stops. Append and last-removal version/barrier order differ;
current-info lookup on an empty List reaches an index guard. Register-as type
selection preserves native field reload and DWORD return width.
Its guarded Rust caller compares 93 normal profiles and two retained alias
sequences in six tests, preserving complete storage/Actor and barrier-time
snapshots. Declared capacity bounds all accesses, including diagnostic tail
retention; growth, invalid storage, mutation and failure remain rejected.

The [tutorial handler publication audit](notes/systems/tutorial_handler_publication.md)
adds four complete handlers in 55 fixtures and 42 exact stopped prefixes.
Start and kill handlers retain ordered routine captures, including reversed
poison fields. Level handling rereads gameplay after publication callbacks;
runtime services and generated routine execution remain explicit boundaries.

The [Character presentation helper audit](notes/systems/character_presentation_helpers.md)
adds six complete callers in 311 cases, two retained sequences and 15 stopped
prefixes. Exact byte gates, field reloads and transform bits retain partial
UI writes. CharacterView/CardHighlight bodies and Unity rendering are supplied
boundaries rather than implementations established by these callers.

The [Character tutorial generator join](notes/systems/tutorial_character_generators.md)
adds five game-owned definitions in 47 fixtures and 206 exact stopped prefixes.
Explicit native resumes retain wait bits and the same restricted queue through
callback unrestriction, close/hide, queue processing and two saves. Real elapsed
time, engine scheduling and event admission remain outside this composition.

The guarded Rust factory/direct speech replay compares 116 supported normal
profiles and retained sequences in seven tests. Complete Actor/iterator/UI
snapshots preserve float/trigger bits, null/self captures and physical aliases.
Supplied registration identities do not infer a void caller return; aggregate
future snapshot budgets and typed identities validate before cloning.

The [CharacterInfo tutorial join](notes/systems/tutorial_character_info_join.md)
adds actual publication and generator execution in 15 fixtures and 164 stops.
The generator uses Character.acteds rather than its icon, then joins Show30,
native queue unrestriction, close/hide and the later Show80/save. Explicit
resumes and supplied services keep engine readiness outside the claim.

Its guarded Rust publication replay compares all 48 supported normal native
fixtures in five tests, preserving capture/barrier order, physical routine
identity and raw metadata/class widths. Retained unconsumed byte ranges exclude
consumed gameplay fields; future allocations and whole-state snapshots reserve
capacity before cloning. Callbacks, failures and routine scheduling reject.

The [CharacterView presentation join](notes/systems/character_view_presentation.md)
adds four native bodies in 193 standalone cases, 22 disguise joins, two retained
sequences and 92 stopped prefixes. Native colour, sprite and activation requests
preserve physical aliases and partial writes; DOTween, renderer, text and data
art services remain supplied. Concurrent array-size races remain unclaimed.

The [delayed Demon-kill join](notes/systems/character_delayed_demon_kill.md)
executes 1,728 normal combinations, additional policy/alias/callback profiles
and 149 stopped prefixes through actual death/status/action callers. The evil
capture is a source argument; accepted statuses store the separately supplied
null target. Concrete role/HP/subscriber policies and real timing remain open.

The [death tutorial generator join](notes/systems/tutorial_death_generators.md)
adds both native MoveNext callers in 64 fixtures and 220 stops. Summary gates
bypass null captures; Poison tests Corrupted rather than a death-reason field.
Both explicitly supplied resume orders join queued presentation and two saves,
with runtime/class/List/Unity services and real scheduling still separate.

The [hover/description-hide audit](notes/systems/character_description_hide.md)
adds two complete callers in 52 cases, two retained sequences and nine stops.
It preserves the captured left Acted, later nullable speech reload and physical
Action dispatch order. Actor memory diagnostics do not infer valid sentinel
objects; game-owned presentation and Unity/runtime services remain supplied.

The guarded Rust CharacterView replay compares 166 normal native profiles in six
tests, including every service-entry snapshot and final modeled state. All View
fields and physical UI aliases retain exact colour/float/byte/DWORD semantics;
future snapshot work is bounded before cloning. Disguise joins, mutations,
failures, renderer behavior and scheduling remain outside the contract.

The guarded Rust hover/description-hide replay compares 42 normal native contexts and two retained
sequences in five tests. It preserves the full supplied logical Actor, raw byte
gates, nullable speech and physical selected Action identity/target/MethodInfo.
Service snapshots compare the modeled projection; callable pointers, runtime
internals and diagnostic sentinel references remain supplied provenance.

The guarded Rust [death-generator replay](notes/systems/tutorial_death_show_requests.md) compares 48 native profiles in seven
tests, including full represented physical bytes, wait/state/current captures
and observed Show arguments. Runtime, List, transform and Show acceptance are
explicit supplied outcomes. Queue/save behavior and real scheduling are outside
this caller contract; aggregate future snapshot work validates before cloning.

The [Oracle presentation audit](notes/systems/character_oracle_presentation.md)
adds two complete native callers in 129 cases, two retained sequences and 26
exact stopped prefixes. Captured Acted and saved-colour recipients remain
distinct from later field reloads; Color return-buffer and raw byte/DWORD
gates are exact. Named game/Unity services and actual rendering remain open.

The [RevealOrder presentation audit](notes/systems/reveal_order_presentation.md)
adds two native callers in 50 cases, two retained sequences and six exact
stopped prefixes. Init captures the order DWORD, activates the GameObject,
then captures text before formatting and reloads its virtual class afterward.
Unity/formatting/TMP implementations remain supplied; the separate actual
Oracle join below now verifies the game-owned composition.

The guarded Rust [RevealOrder replay](notes/systems/reveal_order_presentation.md)
compares 23 normal native profiles and two retained sequences in 4 tests.
Physical records, supplied effects, order bits and captured text retain exact
service-entry chronology. Future snapshot work validates before cloning;
formatting, Unity/TMP implementation and actual Oracle composition remain open.

The [RevealOrder constructor audit](notes/systems/reveal_order_constructor.md)
adds the exact folded wrapper in 54 contexts, six full stopped prefixes and 11
retained constructor/presentation sequences. It forwards the owner and zeroed
MethodInfo to a supplied MonoBehaviour base without initializing custom text.
Shared aliases are not promoted; Unity construction and serialization remain open.

The actual [Oracle-to-RevealOrder join](notes/systems/character_oracle_reveal_join.md)
executes both native caller/callee families in one physical state across 182
cases, eight retained sequences and 88 full stopped prefixes. Order/text capture,
TMP class reload and shared GameObject aliases retain exact chronology; other
game-owned, runtime, formatting and Unity services remain supplied.

The [CardTokens audit](notes/systems/card_tokens.md) executes both native
callers in 302 cases, two retained sequences and 73 full stopped prefixes.
Five independent key branches preserve ordered physical tag effects, pointer
reloads and exact byte/register widths. Input collection and Unity tag effects
remain explicit supplied services; no live keyboard or rendering is inferred.

The [DeckView visibility audit](notes/systems/deck_view_visibility.md) executes
four native bodies in 142 cases, two retained sequences and 81 full stopped
prefixes. Update tail-calls the actual Close body; canvas/animation captures and
later reloads retain callback timing and exact register widths. Input, Unity,
formatting and tween implementations remain explicit supplied services.

The [pin-button audit](notes/systems/pin_deck_view_button.md) adds four native
callers in 213 contexts, 12 retained sequences and 22 full stopped prefixes.
Setting DWORD gates, registration source capture and later static/callback
reloads preserve exact native chronology. Settings, delegate and UI effects
remain supplied; twelve trap/post-store stub instructions are explicitly unexecuted.

The guarded Rust [CardTokens replay](notes/systems/card_tokens.md) compares
267 supported normal native profiles and two retained sequences in five tests.
Complete represented physical storage and supplied ledgers retain service-entry
chronology, ordered tag aliases and exact byte/register widths. Future snapshot
and log work validates before cloning; engine effects and failure paths are excluded.

The Oracle-to-View report now losslessly interns repeated complete snapshots.
Its 16.1 MB encoding expands to exactly the prior 61.9 MB report, including
all 168 complete stopped prefixes. Hash verification and independent mutable
expansion are covered by four codec tests; all 36 reverse-engineering tests pass.
This storage change adds no method coverage.

The actual [Oracle-to-CharacterView join](notes/systems/character_oracle_view_join.md)
executes both native families across 253 cases, six retained sequences and 168
full stopped prefixes. Captured animation/data/text, later reloads and shared
GameObjects preserve callback chronology and exact warm/cold register values.
RevealOrder, Acted and engine/data services remain explicit supplied boundaries.

The [InGameSettings audit](notes/systems/in_game_settings.md) executes three
native menu callers in 151 contexts, four retained sequences and 18 full stopped
prefixes. Active-state capture and field reload, Escape low-byte gating and exact
setter register widths preserve callback timing. Engine input and GameObject
effects remain supplied; the shared NightStep alias is not promoted.

The actual [Oracle-to-Acted join](notes/systems/character_oracle_acted_join.md)
executes 331 cases, two retained sequences and 85 complete stopped prefixes.
Captured Acted/layout receivers and array elements survive later field changes,
while current lengths and parent fields are reloaded in native order. ActedVersion
Show, layout engine and the separate View/RevealOrder services remain supplied.

The guarded Rust [settings replay](notes/systems/in_game_settings.md) compares
92 normal native profiles and two retained seven-call sequences in five tests.
Complete physical storage, request ledgers and service-entry snapshots preserve
Escape gating, active state and full method-specific setter registers. Future
state/log work validates before cloning; engine effects and failures are excluded.

The [DeckCharacter surface audit](notes/systems/deck_character_surface.md)
executes five callers in 83 cases, two retained sequences and 50 full stopped
prefixes. Hover executes the actual HintInfo constructor, retaining the callback
captured before allocation and reloading Character/pivot after construction.
List membership, callback effects and the whole RevealNoAct callee remain supplied.

The [CharacterData consumer audit](notes/systems/character_data_consumers.md)
executes eight getters and skin selectors in 354 profiles, four retained sequences
and 56 complete stopped prefixes. It preserves nullable outputs, raw enum widths,
art_cute defaults and captured versus reloaded skin/literal references. Byte retention
allows only completed writes; Unity liveness and larger provider bodies remain supplied.

The guarded Rust [CharacterData replay](notes/systems/character_data_consumers.md) compares
146 normal profiles, eight inert baselines and three retained sequences in five
tests. All represented byte storage, flags and comparison histories remain in
service-entry snapshots. Nullable returns, raw enum DWORDs and independent
comparison AL match native behavior; future clone work validates before replay.

The Character art report now losslessly pools full snapshots as well as bytes.
Its 15.0 MB encoding expands to exactly the prior 43.0 MB report, including all
110 complete stopped prefixes and raw service arguments. Both successful producers
match; this storage change adds no native method coverage.

The [Character art preferences audit](notes/systems/character_art_preferences.md)
executes four complete bodies in 518 cases, five retained sequences and 110
full stopped prefixes. Native SetupArt preserves captured sprites, exact type
DWORDs and later Image reloads, including aliased GameObjects. Appearance/data
selection, Unity effects and preference subscription remain supplied boundaries.

The [DeckCharacter registration audit](notes/systems/deck_character_registration.md)
executes Init and OnDisable in 96 cases, two retained sequences and 72 complete
stopped prefixes. It preserves captured first-channel operands, reloaded second
interaction and the original data argument through later stores and InitReward.
Action/Delegate services and event admission remain explicitly supplied.

The complete [HintInfo constructor audit](notes/systems/hint_info_constructor.md)
executes 288 cases, two retained sequences and 25 full stopped prefixes. An
independent byte-level model checks raw arguments and every partial snapshot,
including early register captures, late stack title/flavor/Color loads and exact
fault sites. This closes the prior single hover-argument constructor limitation.

The actual [View-to-CharacterData join](notes/systems/character_view_data_join.md)
executes 221 profiles, four retained sequences and 173 full stopped prefixes.
Original data and produced sprites survive later field changes while actual
getters reload current skin; full raw arguments and native phases remain in
service snapshots. Runtime/Unity/TMP and animation remain separate boundaries.

The guarded Rust [HintInfo constructor replay](notes/systems/hint_info_constructor.md)
compares 262 inert native call inputs in five tests, preserving complete physical
records, argument slots, prior history and full service-entry raw arguments.
Nullable aliases and exact Color bytes survive reuse; nominal storage and future
clone/history budgets validate before replay. GC and callback effects stay excluded.

The complete [Character.InitReward caller](notes/systems/character_init_reward.md)
executes 72 cases, three retained sequences and 28 exact stopped prefixes.
An independent model compares every full snapshot and raw service call, including
captured Acted/input identities and late alignment/state reads. RevealReal and
Unity/Action services remain explicit boundaries.

The reward and bluff presentation reports now losslessly pool raw memory
as well as full snapshots. Together they shrink from 50.3 MB to 13.8 MB and
expand to exactly every prior field, byte and stopped prefix. Two independent
native producers per family match; this storage change adds no method coverage.

The [reward presentation audit](notes/systems/character_reward_presentation.md)
executes SetupObject and RevealReal in 341 cases, three retained sequences and
61 full stopped prefixes. Every event and final state matches an independent
model. Captured name, sprite and background identities remain distinct from
later Data/component reloads; art/View/engine bodies stay supplied here.

The [Character art-to-Data join](notes/systems/character_art_data_join.md) executes
535 profiles, six retained sequences and 216 exact stopped prefixes through
seven actual bodies. The first sprite survives a later independent appearance
selection; current skin reloads and separate type reads match complete physical
state. Appearance and Unity/runtime services remain explicit boundaries.

The [RevealBluff audit](notes/systems/character_bluff_presentation.md) executes
144 cases, three retained sequences and 70 full stopped prefixes. A null uppercase
result reaches TMP directly, and the physical TMP class remains in R9. Every
full event and final state matches an independent model; supplied UpdateView
precedes RefreshView without another bluff read.

The guarded Rust [reward initialization replay](notes/systems/character_init_reward.md)
compares 34 normal native contexts and two complete retained three-call sequences
in five tests. Full physical storage, phase/history logs, raw service registers,
exact caller returns and cumulative ordinals match native snapshots. Nominal
storage and future work validate before cloning; service bodies remain supplied.

The complete [Character event lifecycle](notes/systems/character_event_lifecycle.md)
executes 282 cases, four retained sequences and 252 full stopped prefixes.
Seven channels preserve captured old Actions and exact instance/static destination
reloads. Every full snapshot, raw call and cumulative ordinal matches an independent
model; CLR delegates, engine lifecycle and subscriber bodies remain supplied.

The [Character picker/details callers](notes/systems/character_pick_details.md)
execute 420 profiles, four retained sequences and 56 full stopped prefixes.
Array capture follows membership; later iterations reload length and slots.
Details gates preserve byte versus DWORD behavior and physical delegate identity.
Every complete event and final state matches an independent ordered model.

The six [CharacterData text callers](notes/systems/character_data_text_consumers.md)
execute 159 cases, four retained sequences and 16 exact stopped prefixes.
Flavor keeps its captured array across RNG callbacks, translations use the
original-owner Unity-name fallback, and name writes precede their barrier.
Every full snapshot/raw call/final byte matches an independent model.

The [reward initialization join](notes/systems/character_reward_init_join.md)
executes actual side selection, reward initialization and real presentation
in one retained graph. Its 144 cases, ten sequences and 141 exact stopped
prefixes preserve captured inputs and late callback changes across the chain.

The [CardInteraction setup audit](notes/systems/card_interaction_awake.md)
executes Awake and both hover gates. Its 67 cases, four retained sequences
and 21 exact stopped prefixes pin Character and animation-ID stores,
Int32 boxing width, literal reload timing and one-byte hover writes.

The [reward art join](notes/systems/character_reward_art_join.md)
now executes six actual bodies through Data art selection and SetupArt.
Its 394 cases, 14 retained sequences and 337 exact stopped prefixes
verify captured Sprite versus reloaded type and Image receivers across the chain.

The [CharacterData description audit](notes/systems/character_data_description.md)
executes 298 cases, nine retained sequences and 27 exact stopped prefixes.
It pins the current language reloads and repeated conversion of the same
description field, including callback changes and partial recovery.

The [Character description caller](notes/systems/character_show_description.md)
executes 310 cases, four retained sequences and 262 full stopped prefixes.
Its complete native branches preserve captured speech targets, savedAct
publication, history highlighting and the full hint/delegate call ABI.

The [CardInteraction lifecycle audit](notes/systems/card_interaction_lifecycle.md)
executes both complete registration callers across 130 cases, four retained
sequences and 61 exact stopped prefixes. It pins captured animation and
Character receivers, delegate operands and the native click-field store.

The guarded Rust [description getter replay](notes/systems/character_data_description.md)
compares 145 normal native contexts and three complete retained sequences in
five tests. Full storage, raw service arguments, exact callers, histories and
return identities match; nominal input and future snapshot work are bounded.

The [card audio callers](notes/systems/card_interaction_audio.md) execute
82 cases and 24 full stopped prefixes. Both consume a float RNG draw before
reloading the audio callback, then dispatch their exact sound identifier.

The [CharacterData constructor](notes/systems/character_data_constructor.md)
executes 148 cases and 205 full stopped prefixes. It captures six allocated
lists across supplied constructor calls, publishes them with reference barriers,
and sets the native bluffable and picking bytes before the base tail call.

The [reward color join](notes/systems/character_reward_color_join.md)
executes seven actual native bodies, 475 cases and 534 complete stopped prefixes.
UpdateViewReal captures the border array but reloads character data for each
border color, then tail-calls a supplied RefreshView.

The guarded Rust [CharacterData text replay](notes/systems/character_data_text_consumers.md)
compares 100 normal native fixtures across six methods and two retained
continuation calls. Five tests verify full storage, ordered service requests,
raw call ABI, exact callers, nullable results and the name store before its barrier.

Resume with these boundaries in view:

1. The explicit `bluff::clock` transitions now feed the weighted scheduler via
   `bluff::clocked_reveal`, with callback clock stability required as provenance.
   Clock update/reset/calibration, default-loop placement, four public setters,
   normalization/refresh and shipped TimeManager fields are audited. Provider/pause
   callbacks, optional setter notifications, runtime configuration loading order
   and modified runtime loops remain open.
   Phase bit 8 and complete delayed-Reveal interleaving are still unresolved.
   The additional mask-16 dispatch at `0x59F692` now has a complete enclosing
   method audit and three direct caller gates. Their public lifecycle identity
   remains open. `0x59F67B` is only an unwind chunk start. Normalization at
   `0x5520E0` is distinct from reciprocal refresh and leaves its caches untouched.
   A bounded phase-eight inventory checks 23 global loads and 158 immediate
   slot branches without establishing another dispatcher. Continue from its
   explicit candidate/exclusion list rather than repeating the same search.
   The [phase-eight handoff](notes/systems/unity_wait_phase8_handoff.md) expands
   eighteen chained-unwind families and establishes entries for three raw
   forwarding candidates. Their receiver aliases and caller masks remain open.
2. Continue the ascension-to-acquisition bridge: weighted cached script selection
   is reconstructed, and all 23 GameData methods are audited. Engine JSON copy
   semantics and remaining mode/save callees are explicit boundaries. Both
   fully-shared generic copy alternatives now have native caller/boxing evidence;
   their actual invoker and engine-serialization services remain open.
   Engine JSON registration/fallback and ToJson/FromJson gateways are now
   audited, including conditional constructor-error reporting and allocation
   retention. [Core parser and parsed-tree rendering](notes/systems/unity_json_parser.md)
   now execute 89 natural inputs and eighteen controlled diagnostics. Actual
   [Metadata adapter traversal](notes/systems/unity_json_fields.md) now executes
   166 fixtures for direction-9 cache selection, cold initialization, descriptor
   order and retained failures. Individual field conversion bodies and metadata
   discovery remain supplied; writer-side traversal and managed references remain
   next boundaries.
   [Native reader registry construction](notes/systems/unity_json_registry.md)
   now executes 32 fixtures, recovering the actual handler pointers in its 33
   ordinary records and optional extension. Class discovery remains supplied.
   [Numeric field application](notes/systems/unity_json_primitives.md) now executes
   nine actual readers in 333 scalar fixtures and sixteen adapter batches over
   authored descriptors. Field inclusion, descriptor construction and the remaining
   conversion families remain open.
   [Numeric descriptor construction](notes/systems/unity_json_descriptors.md)
   adds 105 construction fixtures and 36 copies joining actual factory output to
   numeric application. Metadata exports remain services; native field eligibility
   and enumeration are next boundaries.
   [Native numeric metadata construction](notes/systems/unity_json_metadata.md)
   now executes 226 fixtures for field eligibility/enumeration, inherited ordering
   and joined copy over supplied metadata. Real managed discovery, compound field
   families and writer conversion remain open.
   [Numeric save/load composition](notes/systems/unity_json_numeric_roundtrip.md)
   now executes 110 round trips and 32 controlled writer stops, joining actual
   native writing/rendering to numeric reload over supplied metadata. Compound
   serialization, reference processing and real runtime discovery remain open.
   [String save/load composition](notes/systems/unity_json_strings.md) adds 38
   round trips, 13 reader cases, two mixed inherited objects and 67 service stops.
   Native UTF-16 conversion executes with explicit thread-local initialization;
   nullness and embedded NULs are observably lost. Runtime string creation,
   compound graphs and real metadata discovery remain open.
   [Scalar/string array composition](notes/systems/unity_json_arrays.md) adds 72
   numeric round trips, 44 reader cases, two string-array cases, two shared-array
   cases and 152 service stops. Native collection processors and element handlers
   execute, preserving reuse/resizing and separate alias allocation requests.
   Lists, compound elements and runtime allocation/discovery remain open.
   [Scalar/string List composition](notes/systems/unity_json_lists.md) adds 86
   normal cases and 373 service stops. Native backing-field discovery, collection
   traversal and constructor wrapper execute, including missing-member allocation
   for a null List. Runtime classification/construction, enabled lookup caches
   and compound elements remain open.
   [Native List classification](notes/systems/unity_json_list_classifier.md) adds
   eighteen exact name/image cases and six joined copies. Classification now
   executes over supplied class metadata; managed construction/discovery and
   compound elements remain open.
   [SavedGameInfo field composition](notes/systems/saved_game_info_json.md) joins
   all three pinned public fields in 21 normal cases and 528 service stops.
   [SavedGameInfo construction and mutation](notes/systems/saved_game_info_methods.md)
   now execute all five callers in 132 cases, 20 controlled service stops and ten
   value-level JSON joins. [SavedGameData persistence callers](notes/systems/saved_game_data.md)
   add all four methods in 44 cases, 41 service stops and six native JSON joins.
   Actual preference storage, generic gateway internals and runtime discovery
   remain explicit boundaries.
   [Native generic FromJson](notes/systems/saved_game_generic_json.md) now runs
   inside Load in 48 cases, 64 service stops and four reader joins. Runtime
   type/context/class/cast helpers and the non-generic gateway remain services.
   [Native preference wrappers](notes/systems/saved_game_preferences.md) add 52
   cases, 28 service stops and six JSON joins. Resolver/backend storage and
   exception construction/throw remain explicit services.
   The [preference registration join](notes/systems/saved_game_preference_lookup.md)
   verifies both exact requests against shipped bare names and ten native lookup
   cases. Next engine entries are `0xF3150` (GetString) and `0xF22B0` (TrySetSetString);
   their native entry execution follows below; registry processing and platform
   storage remain open.
   [Engine preference entries](notes/systems/unity_preferences_entries.md) now
   execute their native conversion and chained cleanup in 380 cases and 42
   service stops. Backend registry processing and other allocator modes remain
   open. Entry interfaces preserve explicit lengths, including embedded NULs.
   [Native registry setter](notes/systems/unity_preferences_setter.md) adds 132
   cases and 41 service stops. Actual signed-byte hashing, argument formatting
   and setter cleanup run through the Windows API boundary. The getter's type,
   legacy fallback and C-string policies follow below; provider acquisition remains open.
   [Native registry getter](notes/systems/unity_preferences_getter.md) adds 398
   cases and 65 service stops, including independent size/data lookup fallback,
   binary versus ASCII-string policy, NUL truncation and authored race responses.
   Runtime/API outcomes and provider acquisition remain explicit services.
   [Native provider acquisition](notes/systems/unity_preferences_provider.md)
   adds 163 cases, 14 entry joins and 152 stops for cached configuration, both
   handles, path conversion and invalidation/retry. [Cold token discovery](notes/systems/unity_preferences_token.md)
   adds 87 cases, five cache sequences and 17 stops. [Joined cold discovery](notes/systems/unity_preferences_cold_provider.md)
   adds 25 provider cases, four entry joins and 99 stops with actual predicate,
   acquisition and storage-entry execution in one emulator. Runtime configuration
   loading and actual OS outcomes remain separate boundaries.
   [Save/storage composition](notes/systems/saved_game_storage_join.md) joins
   native save/mutation callers, actual JSON execution and native provider/getter/
   setter bodies in 43 cases and 22 outer service stops. Runtime construction,
   object identity, private List versions and actual OS outcomes remain separate
   boundaries; the adapters transfer only public values.
   [Exported runtime string construction](notes/systems/il2cpp_string_creation.md)
   now executes native validation/conversion, temporary wide-string callers and
   managed UTF-16 construction in 720 cases and seven controlled stops. Malformed
   UTF-8 returns cached empty rather than replacement characters. Runtime class
   discovery, GC, arbitrary pointers and overflow remain explicit boundaries.
   The [runtime string/storage join](notes/systems/saved_game_runtime_strings.md)
   adds 24 direct runtime cases, 48 getters, twenty Loads, two Save/Load round
   trips and ten controlled stops. Malformed binary UTF-8 returns cached empty
   before JSON and reaches native fresh-save construction. Cross-emulator text
   is transferred without runtime identity or allocation ownership.
   The [tutorial-reset caller join](notes/systems/reset_tutorial_button_join.md)
   adds 26 cases and ten controlled stops, following the exact ProjectContext
   chain into native reset, JSON and storage. Persistence failures retain the
   already-cleared tutorial List. Singleton production and Unity Button routing
   remain explicit boundaries.
   The [guarded SavedGameInfo replay](notes/systems/saved_game_info_replay.md)
   compares all 134 normal native method/JSON-caller fixtures, including service
   entry snapshots, wrapping versions, retained backing slots and ordered fresh
   List publication. Growth and runtime services remain explicit; aggregate slot
   and text budgets are checked before snapshot cloning. Six focused tests pass.
   The [tutorial presentation/persistence join](notes/systems/tutorial_persistence_join.md)
   adds seven previously unclassified methods in 474 cases and 64 exact stops.
   Presentation reset is distinct from persisted completion reset; native showing
   joins AddTutorial/Save and four storage reloads. Controller type/delegate
   publication can follow a note's early return. Unity/event/timing services and
   remaining tutorial methods remain explicit boundaries.
   The [tutorial registration audit](notes/systems/tutorial_event_wiring.md)
   adds both lifecycle subscription callers in 228 retained cases and 100 exact
   stops. Seven event slots retain physical delegate identity/order, including
   a generic second-cast failure after pointer storage and before its barrier.
   Combine/Remove/casts are supplied; automatic handler dispatch remains open.
   Its guarded Rust caller compares 226 normal profiles and four cast stops in
   five tests, retaining all 29 fields and exact physical/header/pointer order.
   Supplied delegate outcomes do not implement CLR multicast or event dispatch.
   The [tutorial close/reveal join](notes/systems/tutorial_close_reveal_join.md)
   adds five new caller definitions in 445 cases and 179 exact stops. Native
   hide callbacks process queues by the count of restricted records, then show
   before removal. Save failure retains earlier note/completion/queue effects;
   coroutine publication and explicit callback invocation do not prove readiness.
   All declarations of GameMode, StandardMode, RoguelikeStandard, AdvancedMode,
   RoguelikeMode and SavesGame now have caller evidence. Standard and roguelike
   progression have versioned Rust replays. Native delegate composition confirms
   the teardown Combine discrepancy. Composed mode transitions, village
   advancement and the weighted starting-character sequence now have native
   fixtures and bounded Rust replays. Completed Standard reset during UI refresh
   now has bounded native reentry evidence. All 55 Gameplay declarations have
   scoped caller evidence; roster filtering, score arithmetic, delayed deck
   intro, character filters, saved/current startup copies and lazy starting-pool
   composition now have native fixtures and offline Rust replays. Preserve each
   caller's faction-read order and explicit service contracts. All 45 Characters
   declarations now have scoped native evidence. Duplicate selection has a
   weighted replay, and its actual candidate filter composition has 124 native
   cases plus an exact bounded Rust trace/weighted bridge. Unique-pool ordering
   is audited in 108 cases; clear precedes predicate construction and removal.
   Its weighted Rust replay now matches all 108 native traces, 18 initial and
   nine fallback paths. All Acted and ActedVersion declarations now have scoped
   native caller evidence, including delayed speech and text/scale animation.
   Rotation/highlight replay and full diagnostic Reveal-wrapper callers are
   audited, with transform, formatting and UI callback internals still explicit.
   All 77 Character declarations now have scoped evidence. CharacterView,
   hover/description, Oracle and RevealOrder have scoped caller audits and
   guarded normal replay contracts. Separate actual Oracle joins now execute
   RevealOrder, CharacterView and Acted; next compose all three over one retained
   physical state. Those separate joins do not establish the combined chain.
   Likewise connect delayed-kill callbacks to tutorial publication and
   event dispatch before inferring engine readiness or subscriber lifetimes.
   Continue from [the pool-to-acquisition handoff](notes/systems/round_pool_acquisition_frontier.md):
   extend the actual ManageCharacters/both-builders prefix through native Init
   and writer dispatch, then join the exact actor/pool/continuation state required
   by the existing acquisition and scheduled replays.
3. Preserve the corrected individually aligned serialized Boolean fields:
   15 of 46 core roles are usuallyDisguised. Public Dreamer's script-priority
   support now uses those flags. The old all-false result was a parser error.
   Keep public Mutant/Skinwalker separate from the unbound managed Mutant class.
4. Expand the remaining Assembly-CSharp ledger from native evidence without
   equating a shared RVA or typed prototype with every method being recovered.

Start with the linked clock, ascension setup/helpers and GameData lifecycle
notes in README. Proprietary exports remain in the private artifact workspace.
No live game or automation loop was used for this reconstruction session.

## Definition of complete

For this project, “fully decompiled” means:

1. Every type in the game-owned `Assembly-CSharp.dll` range is inventoried.
2. Every nontrivial native method is classified as reconstructed, understood
   boilerplate/generated code, unreachable, or explicitly unresolved.
3. Gameplay-critical methods have readable authored pseudocode or clean-room
   implementations, call relationships, field layouts, and validation evidence.
4. Deck construction, board lifecycle, statuses, corruption, clues, active
   abilities, execution/damage, night resolution, scoring, ascension rules, and
   every role have differential tests against observed behavior.
5. A new game build can be fingerprinted, dumped, diffed, and triaged by the
   checked-in scripts without relying on undocumented local steps.

This does not claim recovery of original variable names, comments, project
layout, or byte-for-byte C# source. Those do not survive IL2CPP compilation.

## Milestones

- [x] Create and push a dedicated branch.
- [x] Fingerprint the current game and metadata.
- [x] Produce a current-build Il2CppDumper extraction.
- [x] Commit the reproducible foundation and build manifest.
- [x] Generate the first complete `Assembly-CSharp` type inventory.
- [x] Establish the complete 4,207-method coverage and evidence ledger.
- [x] Produce Cpp2IL managed-IL recovery and an explicit quality baseline.
- [x] Import `GameAssembly.dll` into Ghidra and apply IL2CPP method, metadata,
  and string symbols.
- [x] Import IL2CPP headers, selected prototypes, and reachable field layouts
  into an isolated typed Ghidra project; complete full auto-analysis and
  read-only post-save signature/ABI validation.
- [x] Export and confirm the first gameplay-core native target set.
- [x] Recover and native-audit the first roster-selection helper boundary.
- [x] Map and baseline-export the 28-method gameplay-lifecycle boundary, then
  native-audit its first 11-method setup, board, reveal, and click/kill slice.
- [x] Native-audit the remaining 17 initialization, reveal/kill-helper,
  bookkeeping, and Night-flow methods; close the lifecycle boundary.
- [x] Expand the isolated typed project to all 28 lifecycle methods and pass
  post-save ABI validation plus baseline-versus-typed quality checks.
- [x] Map and baseline-export the 30-method execution-resolution boundary.
- [x] Native-audit the first 16-method execution, damage, protection, and
  terminal-result slice.
- [x] Native-audit the remaining 14 status-insertion, Night-rule, Striga,
  Demon-selection, and collection-helper methods; close the boundary.
- [x] Expand the isolated typed project to all 77 methods across the four
  reviewed target sets; pass post-save ABI validation and body-free quality
  checks for every set.
- [x] Map and baseline-export the 40-method status, corruption, truth/lie, and
  bluff-orchestration boundary, including explicit C prototype aliases for
  overloaded managed methods.
- [x] Native-audit the first 16 status-storage, cure-gating, selection, and
  truth/appearance methods in that boundary.
- [x] Native-audit the next 11 Pooka, Poisoner, Puzzlemaster/Plague Doctor,
  Drunk, and Alchemist status-lifecycle methods in that boundary.
- [x] Native-audit the final 13 bluff storage, Puppet/Puppeteer,
  Doppelganger, Confessor, Reveal, and shared orchestration methods; close the
  40-method status/corruption/truth boundary.
- [x] Map and native-audit the 20-method bluff-acquisition boundary, including
  common assignment, Demon/Minion/Spy/Mutant selectors, pool mutations,
  shared-body identity, and stale-role lifecycle reachability.
- [x] Add bluff acquisition to deterministic checked-target discovery and the
  six-set typed-header/refresh union with exact overload aliases.
- [x] Refresh the preserved typed project and publish the bluff-acquisition
  baseline-versus-typed quality report.
- [x] Map, baseline-export, and native-audit the complete Slayer and Wretch
  role implementations; join registered alignment to kill-and-reveal behavior
  and fix the live Wretch bookkeeping regression.
- [x] Expand the deterministic typed union to eight target sets and 154 target
  memberships; support folded per-role native bodies in apply/validation and
  publish both role quality reports.
- [x] Map, baseline-export, type, and native-audit all 12 methods in the
  internal `Dreamer2` boundary, including its randomized type-exclusion clue
  and the complete `GetDreamerClue` provider set.
- [x] Asset-bind the public Dreamer card to managed `Dreamer`; prove that
  `Dreamer2` and `DreamerOld` are unbound in the current gameplay assets.
- [x] Map, baseline-export, type, and native-audit the complete public
  `Dreamer` boundary: all 11 role methods plus five compiler-generated helpers,
  including its Cabbage branch and exact current-build role-pair weighting.
- [x] Implement and regression-test the public Dreamer parser, native-support
  validator, and weighted role-pair recommendation model.
- [x] Asset-bind public Baa to managed `Imp`; native-audit its complete role
  class plus deck-view add/remove helpers, and remove the false board-reveal
  inference from the live wrapper.
- [x] Asset-bind public Shaman to managed `Illuzionist`; native-audit its four
  role methods plus seven selection, status, and lifecycle helpers, expand the
  typed union to twelve target sets and 198 memberships, and publish its
  baseline-versus-typed quality report.
- [x] Asset-bind public Plague Doctor to managed `Puzzlemaster`; native-audit
  all 11 role methods plus 12 dispatch, click, picker, status, and filter
  helpers, close truthful/bluff Day output and Drunk status handling, expand
  the typed union to thirteen target sets and 221 memberships, and publish its
  baseline-versus-typed quality report.
- [x] Asset-bind public Judge to managed `Judge2`; native-audit all ten role
  methods plus eight dispatch, truth-appearance, click, and picker helpers,
  close unrestricted target legality, deterministic corrupted-actor inversion,
  exact one-reference output, and ResetAfterNight history, expand the typed
  union to fourteen target sets and 239 memberships, and publish its
  baseline-versus-typed quality report.
- [x] Asset-bind public Witch to managed `Cipher`; native-audit all five role
  methods plus 14 ordered-Start, inherited-dispatch, global-value, hidden-count,
  click, reset, and ordinary/night-death helpers, close the exact last-card
  predicate, lack of blocked identity, killed-hidden membership, self-block,
  stacking/duplicate behavior, and death cleanup, expand the typed union to 15
  target sets and 258 memberships, and publish its baseline-versus-typed
  quality report.
- [x] Asset-bind public Chancellor to managed `Baron`; native-audit all five
  role methods, all eight Witness methods, and 18 ordered-Start, selection,
  status, identity-mutation, and death helpers; close anywhere-Villager
  eligibility, exact anchor/neighbour order, `c/v/o/f/a` identity equations,
  duplicate and resistance behavior, current-status Witness truth/bluff
  semantics, and death persistence; expand the typed union to 16 target sets
  and 289 memberships and publish its baseline-versus-typed quality report.
- [x] Asset-bind public Lilis and Knight to managed `Striga` and `Immortal`;
  native-audit all 13 role methods plus 41 ordered-Start, Night-rule,
  selection, delayed-kill, protection, ordinary-execution, Slayer, HP, status,
  and reset helpers; close hard registered-Good priority, protected no-kill and
  duplicate-Night behavior, exact Knight killability precedence, and the
  additional-four/total-nine corrupted-Good execution result; expand the typed
  union to 17 target sets and 343 memberships and publish the combined quality
  report.
- [x] Asset-bind public Rambler to managed `Rambler2`; native-audit all 14 role
  methods, both compiler-generated closure methods, and 20 setup, dispatch,
  adjacency, interference, acted-history, and reveal helpers; close pre-flip
  AfterRoundStart installation, actual-source versus apparent-target truth,
  hidden callback persistence and last-writer behavior, duplicate/small-board
  adjacency, immediate versus pre-append history, and constraint-free Day
  quotes with exact references; expand the typed union to 18 target sets and
  379 memberships and publish its baseline-versus-typed quality report.
- [x] Asset-bind public Baker to managed `Baker`; native-audit all 11 role
  methods, Baker runtime history, all three achievement-helper methods, and 21
  click, reveal, dispatch, filter, replacement, lookup, and acted-history
  helpers; close synchronous Day-only chain timing, exact real/lying role-name
  pools, runtime-cast and status composition, registered candidate eligibility,
  physical order/duplicates, small boards, and achievement ordering; expand
  the typed union to 19 target sets and 415 memberships and publish its
  baseline-versus-typed quality report.
- [x] Asset-bind public Doppelganger and Drunk to managed `Doppleganger` and
  `Drunk`; native-audit all 17 role methods plus 22 setup, delayed-reveal,
  source-filter, unique-pool, registration, status, and execution helpers;
  close ordered Start/Puppeteer conversion before disguise selection,
  erased-Villager exclusion, clean/corrupted source pools, state and duplicate
  weighting, Drunk's two-draw must-include priority and bounded not-in-play
  guarantee, failure mutations, and register-as/HUD separation; expand the
  typed union to 20 target sets and 454 memberships and publish its combined
  baseline-versus-typed quality report.
- [x] Asset-bind public Fortune Teller to managed `FortuneTeller`;
  native-audit all 11 role methods, all six compiler-generated helpers, and
  eight dispatch, registered-alignment, click, picker, and acted-record
  helpers; close unrestricted two-target legality, exact-reference toggle and
  `OnPicked` ordering, truthful registered-Evil OR and deterministic lying
  complement, discarded RNG consumption, ascending-ID speech/reference shape,
  cancellation, ResetAfterNight history, and the both-Evil achievement; expand
  the typed union to 21 target sets and 479 memberships and publish its
  baseline-versus-typed quality report.
- [x] Asset-bind public Bombardier to exact managed `Saint`; native-audit all
  five role methods plus 18 dispatch, ordinary/forced/Demon death,
  bookkeeping, and terminal helpers; close the broader non-Demon-death rule,
  current-`dataRef` managed-type identity, Shaman/Chancellor replacement
  composition, ordinary-bluff and Drunk/Doppel non-composition,
  `SaintVillager` exclusion, and automatic-loss precedence; expand the typed
  union to 22 target sets and 502 memberships and publish its
  baseline-versus-typed quality report.
- [x] Asset-bind public Pooka to exact managed `Pooka`; native-audit all five
  declared role methods plus four status, real-type, adjacency, and rotation
  helpers; close Start-only Evil dispatch, deterministic two-neighbour current-
  real-Villager eligibility, independent Corrupted/MessedUpByEvil attempts,
  ordinary duplicate and small-board behavior, and the native-xref proof that
  private random-one-neighbour `PoisonClosestNeighbours` is unreachable in the
  shipped flow; expand the typed union to 23 target sets and 511 memberships
  and publish its baseline-versus-typed quality report.
- [x] Asset-bind public Poisoner to exact managed `Poisoner`; native-audit all
  four declared role methods plus 13 ordered-Start, dispatch, output,
  adjacency, real-type, status, resistance, and integer-RNG helpers; close
  previous-then-next eligibility, live Corrupted exclusion, independent marker
  resistance, all-match high-ID-first duplicates, dead and small-board
  behavior, and the stale managed-description/nonexistent-dormant-helper
  distinction; expand the typed union to 24 target sets and 528 memberships and
  publish its baseline-versus-typed quality report.
- [x] Asset-bind public Twin Minion to exact managed `Marionette`; native-audit
  all five declared role methods plus 15 ordered-Start, dispatch, Demon-filter,
  alive-adjacency, current-data replacement, delayed-reveal, bluff, and integer-
  RNG helpers; close the two-draw current-`CharacterData` swap, physical-state
  preservation, duplicate/small-board behavior, pending-coroutine multiplicity,
  and dormant-helper reachability; expand the typed union to 25 target sets and
  548 memberships and publish its baseline-versus-typed quality report.
- [x] Asset-bind public Poet to exact managed `Gossip`; native-audit all six
  declared role methods, the twelve exact provider constructors, and generic
  Character real/bluff dispatch; close the ordered provider pool, fresh
  per-invocation real/bluff draw, Day-only callback routing, and strict current
  provenance schema while preserving unmarked legacy fixtures; expand the typed
  union to 26 target sets and 568 memberships and publish its
  baseline-versus-typed quality report.
- [x] Asset-bind public Scout to managed `Scout` and public Hunter to managed
  `Tracker`; native-audit all 17 declared role methods plus nine exact runtime-
  alignment, registration, circular-distance, range-reference, calculator,
  and RNG helpers; close Scout's occurrence-weighted target identity,
  one-Evil sentinel, strict 1-through-3 bluff domain, Hunter's exact `N - 1`
  exhaustion value and half-circle bluff domain, and ordered duplicate-preserving
  acted references; expand the typed union to 27 target sets and 594
  memberships and publish its baseline-versus-typed quality report.
- [x] Asset-bind public Oracle to managed `Investigator`; native-audit all seven
  declared role methods, all six generated comparer methods, and six exact
  Character, script-pool, all-ascension-pool, and registration helpers; close
  the independent truthful Minion/Good draws, moved-Twin duplicate reference,
  exact no-Minions sentinel, distinct-Good bluff pair, and fallback Minion
  label pool for both direct and Poet observations; expand the typed union to
  28 target sets and 613 memberships and publish its baseline-versus-typed
  quality report.
- [x] Asset-bind public Lover to managed `Empath`; native-audit all nine role
  methods, exact circular-adjacency and registered-alignment helpers, and all
  four achievement-helper methods; close registered-Evil occurrence counting,
  duplicate small-board references, exact truth text, the authored
  Minion-plus-Demon bluff domain, and truth-only achievement subscriptions for
  both direct and Poet observations; expand the typed union to 29 target sets
  and 628 memberships and publish its baseline-versus-typed quality report.
- [x] Asset-bind public Bounty Hunter to managed `BountyHunter`; native-audit
  all eight declared methods plus registered-alignment, board-filter, acted-
  record, and integer-RNG helpers; close its dormant direct Start mutation,
  active Poet truth/bluff pools, exact zero-reference clue, and joint anonymous-
  Wretch constraints; expand the typed union to 30 target sets and 640
  memberships and publish its baseline-versus-typed quality report.
- [x] Asset-bind public Medium to managed `Lookout`; native-audit all eight
  declared methods plus registered-alignment, live-identity, raw-status,
  acted-record, and integer-RNG helpers; close its actor-sensitive truthful
  pool, raw-bluff-holder fallback, exact one-reference two-line clue, and
  conditional execution achievement; expand the typed union to 31 target sets
  and 654 memberships and publish its baseline-versus-typed quality report.
- [x] Asset-bind public Knitter to managed `Knitter`; native-audit all eight
  declared methods plus registered-alignment, acted-record, count-removal, and
  integer-RNG helpers; close circular registered-Evil pair counting, exact
  small-board occurrence geometry, truth text, the authored false-count domain,
  direct/Poet parity, and Baker-to-Spy registration chronology; expand the
  typed union to 32 target sets, 666 memberships, 426 selected managed methods,
  and 365 unique native RVAs, then publish its typed-quality report.
- [x] Asset-bind public Enlightened to managed `Shugenja`; native-audit all
  nine declared methods plus registered-alignment, acted-record, physical-list,
  runtime-data, and float-RNG helpers; close exact direction text, public circle
  orientation, no-Evil and small-board ties, always-false bluff support,
  direct/Poet parity, joint anonymous-Wretch worlds, and Baker-to-Spy
  registration chronology; expand the typed union to 33 target sets, 680
  memberships, 435 selected managed methods, and 372 unique native RVAs, then
  publish its typed-quality report.
- [x] Asset-bind public Bishop to managed `Bishop`; native-audit all 17 declared
  role/compiler-generated methods plus registered-data, type-filter, acted-
  record, list-shuffle, and RNG helpers; close exact truth category precedence,
  authored-count bluff construction, live register-as-first references,
  separate ID/type/reference ordering, direct/Poet parity, joint anonymous-
  Wretch worlds, identity movers, and Baker-to-Spy chronology; expand the typed
  union to 34 target sets, 705 memberships, 455 selected managed methods, and
  382 unique native RVAs, then publish its typed-quality report.
- [ ] Preserve removed executed-evil role-to-position assignments during
  scenario construction, branch them before Start, and replay their complete
  ordered mutation histories so `Unknown` seats can be resolved without
  validator-local faction guesses.
  - [x] Stable-origin checkpoint: branch each untyped dead Evil over the exact
    authored multiset before construction; retain role-to-seat identity in the
    scenario; enforce trusted Minion/Demon quotas, exact trusted HUD Evil totals
    with a provenance-gated archival Puppet-count ambiguity, and identity-aware
    native Puppeteer/Puppet conversion in stable worlds, including an explicit
    stable-Twin/current-Puppet body overlay and conservative projection of its
    real Villager source through Start, while retaining conservative mixed-
    writer branches; and re-enable card and historical validators only for the
    resulting exact supported worlds.
  - [ ] Complete the remaining general ordered replay beyond the exact gated
    Twin/Puppeteer, Puppeteer/Shaman, and Twin/Shaman slices: broader mixed
    writer pools, duplicate mutators, split Twin presentation/action provenance,
    and probability-exact occurrence weighting. Strict current observations
    remain fail-closed for inferred incomplete writers.
    - [x] Implement the pure post-Twin Puppeteer boundary: select the first
      current actor, preserve physical previous/next occurrences, filter exact
      real Villagers, remove only the first Saint occurrence, make nonempty
      conversion mandatory, and retain the erased Villager role in an exact
      serializable replay trace.
    - [x] Integrate an atomic exact Twin-to-Puppeteer scenario slice for trusted
      no-Outcast boards with exactly the selected Twin/Puppeteer Minions and
      supported identity-stable writers: enumerate the complete pre-Twin
      Villager occurrence map, replay current-data relocation before selecting
      the Puppeteer actor and target, preserve erased-role provenance, validate
      exact current/public evidence, and fall back wholesale on unsupported or
      capped inputs without resurrecting exact contradictions.
    - [x] Integrate an atomic exact Puppeteer-to-Shaman scenario slice for
      trusted no-Outcast boards with exactly those two Minions, fully dealt
      Lilis Demons, and one deterministic non-Saint Villager neighbour for
      Puppeteer: enumerate the complete initial Villager occurrence map, replace
      the selected identity with Puppet before constructing Shaman's ordered
      Villager pairs, preserve both writer traces and all three marker attempts,
      prevent the erased Puppet identity from re-entering Shaman provenance,
      validate exact final current/public evidence, and fall back wholesale on
      ambiguity, preserved-state hazards, or caps.
    - [x] Cross the existing ordered Shaman trace with every exact Twin outcome
      when all possible Twin endpoints are proven structural non-Villagers, so
      the live Shaman candidate pool is invariant even if Shaman data relocates
      onto the Twin body. Preserve both trace identities and the complete
      Cartesian product, admit the no-Demon path, reject copied Bounty Hunter,
      and fall back atomically if any Twin branch touches a Villager or unknown
      endpoint.
    - [x] Integrate the first candidate-changing Twin-to-Shaman role-flow slice
      for trusted no-Outcast Scout/Witness boards with exactly Twin, Shaman,
      and fully dealt Lilis Demons: enumerate the complete Villager occurrence
      map, replay every Twin occurrence before rebuilding Shaman's live ordered
      pair pool, preserve both trace identities and duplicate-role RNG weight,
      validate the complete baseline and both native traces independently, and
      distinguish exact contradiction from cap/incomplete fallback. Distinct
      swaps with any captured reveal/action history fall back wholesale until
      runtime alignment, dispatched-role truth, and delayed Minion bluff
      presentation have separate provenance.
    - [x] Replay every selected current Plague Doctor at the global Start slot
      in descending displayed-ID order, rebuilding the live eligible pool for
      each actor and retaining ordered target/no-op history through Alchemist
      convergence. Preserve exact uniform target mass only for the singular,
      one-actor Start kernel; grouped Chancellor/Shaman/Poisoner/Twin/
      Puppeteer roots keep equal logical-world semantics.
    - [x] Implement the pure latent Shaman-copied Plague Doctor callback for a
      caller-proven ordinary runtime-Good/no-stale-bluff destination: rebuild
      its live apparent-Villager pool after global PD, preserve separate copied
      target/no-op provenance through Alchemist convergence, and prove that the
      overwritten destination drops when pre-Reveal `registerAs` is null.
      Keep it outside normal scenario generation because shipped initial Start
      has no live source bluff and therefore cannot copy Outcast PD data.
    - [x] Derive the settled physical `AppearTruthfull` status for both exact
      Shaman-copied Confessor endpoints from the existing ordered trace, retain
      it through later Baker no-reset presentation changes, and project it into
      Judge and shipped Rambler appearance checks without changing actual truth
      dispatch. Keep grouped erased-prior Confessor candidates fail-closed.
- [x] Asset-bind public Empress to managed `Noble`; native-audit all 14 declared
  role/compiler-generated methods plus registered-alignment, acted-record,
  pool-filter, and RNG helpers; close its direct/Poet three-reference schema,
  exact truth/bluff registered-alignment pools, lifecycle eligibility, actor-
  self parity, identity-mover behavior, text/reference ordering, anonymous-
  Wretch and Baker-to-Spy worlds, and RNG chronology; expand the typed union to
  35 target sets, 724 memberships, 468 selected managed definitions, and 390
  unique native RVAs, then publish its typed-quality report.
- [x] Asset-bind public Gemcrafter to managed `Archivist`; native-audit all
  seven declared methods plus registered-alignment, acted-record, pool-filter,
  and integer-RNG helpers; close exact direct/Poet text and reference parity,
  registered-Good truth and registered-Evil bluff pools, conditional actor
  removal and sole-pool self support, full lifecycle eligibility, managed-name
  ingestion, identity movers, anonymous-Wretch and Baker-to-Spy worlds, and
  RNG chronology; expand the typed union to 36 target sets, 735 memberships,
  474 selected managed definitions, and 394 unique native RVAs, then publish
  its typed-quality report.
- [x] Asset-bind public Bard to managed `Acrobat2`; native-audit all nine
  declared methods plus acted-record, circular-order, range-reference, false-
  number, and integer-RNG helpers; close exact direct/Poet text and ordered
  reference geometry, actor-self exclusion, full-lifecycle Corruption scans,
  fixed non-board-clamped bluff domain, managed-name ingestion, native real/bluff
  callback ordering, raw-bluff identity, identity movers, Baker-to-Spy
  chronology, and archive compatibility; expand the typed union to 37 target
  sets, 750 memberships, 482 selected managed definitions, and 401 unique
  native RVAs, then publish its typed-quality report.
- [x] Asset-bind public Confessor to managed `Confessor`; native-audit all nine
  declared methods plus acted-record, status, registered-alignment, animated-
  art, and exact-membership helpers; close exact direct text and native-null
  reference provenance, truth-identical Corrupted/registered-Evil behavior,
  current-Spy override, Poet absence, real/raw callback ordering, raw-bluff and
  register-as identity, identity movers, Baker-to-Spy chronology, and archive
  compatibility; expand the typed union to 38 target sets, 764 memberships,
  492 selected managed definitions, and 410 unique native RVAs, then publish
  its typed-quality report.
- [x] Asset-bind public Druid to managed `Librarian`; native-audit all ten
  declared role methods, all six compiler-generated ordering helpers, and 20
  picker, acted-record, registered-data, pool-filter, lifecycle, and RNG
  helpers; close exact three-target selection, click-order references versus
  sorted display IDs, registered-Outcast truth, the complementary authored
  false-role ladder, full lifecycle eligibility, ResetAfterNight history,
  managed-name ingestion, Poet absence, identity movers, anonymous Outcast and
  Wretch worlds, raw callback ordering, Baker-to-Spy chronology, and archive
  compatibility; expand the typed union to 39 target sets, 800 memberships,
  512 selected managed definitions, and 424 unique native RVAs, then publish
  its typed-quality report.
- [x] Close the six-method managed Spy boundary and three semantic callees;
  verify inert real/copied Start, cached Villager identity and empty clue
  construction, preserve the unestablished shipped-asset binding, and join
  Spy Start to standalone/ordered/queue-driven Reveal. Extend the typed union
  to 42 sets, 891 memberships, 548 definitions and 448 native RVAs.
- [x] Audit all 175 direct void definitions sharing the immediate-return body
  against the pinned native image and regenerated denominator, including
  delayed Reveal Dispose. Preserve six existing classifications and add 169
  individually identified methods, with 64 isolated execution cases.
- [x] Verify 503 direct shared constructor/getter definitions with individual
  metadata field identities and 384 native execution cases. Preserve 49 existing
  classifications, add 454, and exclude generic-shared and Unity constructor
  paths from the closed boundary.
- [x] Close all six managed Mutant declarations separately from the public
  Mutant/Skinwalker binding; model occurrence-weighted exact-field acquisition
  with retained Mad on indexed-draw failure, and compare 16 native caller cases.
  Extend the typed union to 43 sets, 899 memberships and 553 definitions.
- [x] Correct serialized Boolean alignment across all 46 core CharacterData
  prefixes and validate following role-reference IDs. Restore Dreamer's native
  script-priority selection, correct Baa/Dreamer2 guidance and eight stale asset
  flag facts, and compare the Rust flag table against all audited records.
- [x] Replace the solver's generic Shaman duplicate allowance with an ordered
  source/target/copied trace plus a viable overwritten-identity class,
  native-timed status effects, and copied-Alchemist Start regressions.
- [ ] Add a versioned offset registry and migrate `memory_reader.py` to it.
- [ ] Live-validate HP, gameplay-state, and board-count pointer chains.
- [x] Recover the gameplay lifecycle and its call graph.
- [ ] Recover deck/board construction and ascension rules.
- [x] Recover status, corruption, truth/lie, and bluff-acquisition pipelines.
- [ ] Recover clue-generation pipelines.
- [x] Recover execution, damage, protection, and night-resolution pipelines.
- [ ] Recover and validate every role implementation.
- [ ] Extract an authored clean-room behavioral core with differential tests.
  - [x] Add the offline Demon/ordinary-Minion/Drunk selector ledger with
    occurrence-preserving pool mutations, exact rational path probabilities,
    script registration, Drunk corruption-attempt effects, and conditional
    equivalence tests against the one-Lilis prefix. Keep unsupported Reveal
    hooks, dispatch, scheduler order, and intervening writers outside this API.
  - [x] Compose the selector ledger with the bounded Lilis/Twin/Drunk Reveal
    callback projection: constant-null register-as, live-bluff guard,
    GiveBluff, repeated continuations, separate real/copied Init/AfterRoundStart
    dispatch, and Scout/Witness/Confessor callbacks with exact status targets.
    Require explicit resume/acquisition provenance and exclude subscriptions,
    HealthyBluff re-entry, view epilogues, and intervening writers.
  - [x] Add the versioned Spy register-as override with explicit data-role cache
    identity, shared/distinct-object provenance, script-occurrence weighting,
    cache reuse after script growth, and register-as updates despite live bluff.
    Preserve the v1 schema and exclude unsupported callback identities.
  - [x] Add versioned HealthyBluff Start latch provenance, status-only Drunk/Lilis
    callbacks, frozen per-trigger dispatch, resistance and repeated-Reveal
    regressions. Preserve v1/v2 serialized shapes and reject reached Twin/Spy
    Start callbacks atomically.
  - [x] Add an isolated Twin Start writer kernel with occurrence-weighted Demon
    and alive-neighbor selection, ordered InitWithNoReset effects, immediate
    action-role clones, preserved stale register-as/copied-role storage and
    explicit new continuation counts. Test self-swaps and a moved Drunk resume.
  - [x] Compose the Twin writer with one explicit Character.Act(Start), its
    one-shot guard, frozen truth decision, current copied-role reread, and
    optional second Twin swap. Preserve reset latches and unconditional mass.
  - [x] Join Character.Act(Start) composition to Reveal acquisition and later
    Init/AfterRoundStart under explicit resume/acquisition provenance. Transport
    writer-created continuation counts and validate repeated new-data resumes.
  - [ ] Establish scheduler provenance and enumerate justified interleavings,
    including branch-dependent acquisition events and omitted view epilogues.
  - [x] Explore all orders of a caller-sealed ready batch, with distinct
    coroutine IDs, branch-local acquisition decisions, conditional RNG weights,
    deferred writer-created continuations and whole-exploration failure caps.
  - [x] Carry a complete logical continuation registry across explicit batches,
    remove consumed identities, allocate writer-created instances in trace order,
    validate per-body counts and reject replayed IDs or allocation overflow.
  - [x] Fingerprint shipped UnityPlayer and audit the diagnostic-linked
    WaitForSeconds record producer and deadline-tree insertion with reproducible
    native semantic checks. Keep unidentified engine fields/order unresolved.
  - [x] Trace the engine deadline-queue consumer, phase/generation/signed-counter
    eligibility gates, successor traversal and one-shot dispatch/release slots.
    Pin the constructor/vtable binding and record the distinct producer/consumer
    time fields without inferring their public identities.
  - [x] Recover the engine's 3,447-pair internal-call registration loop and
    independently bind StartCoroutineManaged2 and selected Time getters. Identify
    the consumer clock and public frame-count backing without narrowing its
    native signed 64-bit gate.
  - [x] Trace normal valid-owner coroutine creation through immediate
    SetupCoroutine.InvokeMoveNext, current-yield dispatch, WaitForSeconds
    registration and the later callback into the same managed-step dispatcher.
    Pin the engine/CoreModule bridge without claiming all lifetime branches.
  - [x] Identify the retained frame clock versus selected public Time.time,
    fixed-step equality boundary and special first fixed step; bind fixed delta,
    time scale and inFixedTimeStep and verify full-width frame-counter updates.
  - [x] Verify finite equal-deadline occurrence order through native tree
    insertion, balancing and arbitrary erasure with a differential emulator
    corpus and payload/link/red-black invariant checks after every operation.
  - [x] Bind five default PlayerLoop nodes to native wait-dispatch masks through
    qualified managed type-cache lookups, callback installation and isolated
    native construction of the 131-node loop; preserve phase-8 uncertainty.
  - [x] Exercise one-shot consumer traversal with synthetic callbacks that insert
    and cancel native records; verify saved-successor updates, retained clock
    samples, timing gates, owner-failure removal and exact release conditions.
  - [x] Audit finite clock-update arithmetic, clamp/capture/skip precedence and
    public snapshot order in 1,437 native cases plus 180 fixed selections;
    add the exact offline Rust projection and round-trip JSON float parsing.
  - [x] Trace QPC timestamp conversion, process baseline, clock constructor and
    reset plus frequency calibration in 62 native cases; distinguish caller
    suppression from updater skip.
  - [x] Join the clock caller to TimeUpdate/WaitForLastPresentationAndUpdateTime
    at native default-loop node 2, preserving all five audited wait bindings.
  - [x] Audit four public timing setters in 72 native cases, including fixed-step
    clamp propagation, timeScale rejection and admitted nonfinite values.
  - [ ] Resolve provider/pause callbacks, remaining
    configuration initialization and optional setter notification effects,
    remaining phase provenance,
    repeating/reentrant drains, release-body mutation and remaining coroutine
    lifetime/cancellation branches.
  - [x] Bind handle, IEnumerator and StopAll cancellation entry points; audit
    owner/callback/payload matching, marked links, cached/GC enumerator identity,
    empty-list guards and bounded unlink behavior in 26 native-emulated cases.
  - [x] Trace reference-release and managed Coroutine finalizer ownership,
    signature-fallback internal-call resolution, handle clearing and allocation
    release in both invocation orders; verify 14 bounded native lifetime cases,
    including auxiliary cleanup while another reference retains the object and
    the AsyncOperation type/cache/callback binding for that auxiliary pointer.
  - [x] Execute managed-step dispatch through native invocation-frame setup and
    actual queue/reference release; verify survival guards, error-out timing,
    native stops/releases during invocation and StopAll with real sibling
    destruction and saved-cursor cancellation in 848 isolated cases.
  - [x] Project finite WaitForSeconds production and local consumer eligibility
    with separate clock snapshots, promoted float duration, signed 64-bit frame
    gates, wrapping generations and explicit traversal-stop versus skip results.
    Keep owner resolution, queue mutation and automatic registry admission open.
  - [x] Project finite one-shot queue traversal and supplied callback mutations,
    with stable equal deadlines, saved-successor cancellation, monotonic labels,
    exact release conditions and atomic failure bounds; compare complete results
    against 23 isolated native consumer cases in proprietary-input-free CI.
  - [x] Join the one-shot queue kernel to weighted Reveal/continuation replay
    for a complete DelayReveal-only queue with explicit matching-owner and
    producer-clock provenance. Allocate 0.3f waits in actual writer order,
    preserve branch-local identities and defer new waits by native generation.
    Keep mixed queues, lifetime-driven mutation and live admission unsupported.
  - [x] Audit UpdateView/UpdateViewReal/RefreshView and add a bounded view-tail
    projection for identity sources, retained death presentation, exhausted
    pickable controls and conditional disguise-icon writes.
  - [x] Join audited view-tail effects to ordered Reveal replay with explicit
    UI-state provenance, replacement-side UI changes and retained death objects.
  - [ ] Close remaining asset/scene callback assumptions and establish native
    coroutine readiness/order provenance for automatic interleaving support.
  - [ ] Extend callback replay to remaining register-as overrides, additional roles,
    subscriptions, mixed wait kinds, and remaining identity writers; establish
    scheduler order provenance or justified interleaving support before
    scenario integration.
- [ ] Run and publish the final method-classification coverage audit.

## Method classification

The coverage ledger will use these terminal states:

- `reconstructed`: readable behavior and tests exist.
- `understood`: behavior is documented; standalone reconstruction is unnecessary.
- `generated`: compiler/Unity boilerplate with its origin identified.
- `unreachable`: not reachable in the shipped Standard/Ascension game surface.
- `unresolved`: work remains, with the exact blocker recorded.

No nontrivial method may disappear from the denominator.

## Validation gates

- Verify both `GameAssembly.dll` and `global-metadata.dat` SHA-256 values before
  extraction or memory-layout use.
- Reject RVAs outside valid PE sections and offsets not tied to a declaring type.
- Pair live memory observations with screenshots; the screen remains UI truth.
- Record controlled one-event before/after traces for state transitions.
- Keep CI independent of proprietary inputs through manifests and synthetic
  fixtures.
- Run Python tests, `cargo build --release`, and
  `cargo test --release --test simulation` for gameplay-facing changes.
- Commit and push each discrete subsystem or role milestone.

- [x] Audit public Mutant/Skinwalker and the full Demon declaration, including
  empty special-rule/clue behavior; extend the typed union to 44 sets, 913
  memberships, 561 exact definitions and 462 native RVAs.
- [x] Decode the complete ProjectContext -> GameData -> 46 ascension / 12
  custom-script configuration graph, and correct Compendium catalogue provenance.
  Keep mode choice, runtime cloning/writers, and global reachability open.

- [x] Audit complete AscensionsData/ScriptInfo declarations and selected mode /
  GameData callers, 33 methods with 412 native cases. Model weighted cached
  script selection in Rust, preserving superseded draws and partial failures.
  Extend the typed union to 45 sets, 946 memberships, 594 definitions and 488 RVAs.

- [x] Audit reference-type ascension copy helpers, managed JSON wrappers and
  distinct lock/unlock pool sources: 140 native cases, 46 typed sets and 956
  memberships. Retain unresolved generic alternates and engine JSON internals.

- [x] Complete the 23-method GameData declaration with a twelve-target lifecycle
  audit and 189 native cases. Retain mode virtual methods, save serialization,
  preference loading and achievement effects as explicit callee boundaries.

The [ManageCharacters prefix audit](notes/systems/manage_pool_prefix.md) adds 44 native cases for the pool-builder handoff and first Init arguments. Builders remain supplied services; this prefix stops before Init or empty-board publication. Full startup-to-acquisition composition remains open.
