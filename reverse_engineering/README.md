# Demon Bluff Reverse Engineering

This directory tracks the reproducible reconstruction of Demon Bluff's Unity
IL2CPP gameplay code. The objective is not to recover the developer's exact
original C# formatting or comments—IL2CPP compilation discards those—but to
account for every game-owned type and method and reconstruct behavior precisely
enough to validate it against the installed game.

## Public-repository boundary

This repository is public. Commit only our own tooling, manifests, normalized
symbol indexes, offset evidence, behavioral notes, clean-room reconstruction,
and synthetic fixtures.

Do **not** commit the game binaries, Unity assets, bulk dumper output, dummy
assemblies, native-analysis databases, memory dumps, or extracted media. These
are ignored under `work/`, `generated/`, and `private/`. Raw artifacts should be
kept on a separate backed-up private store and keyed by the hashes in the public
build manifest.

## Current build

- Steam app: `3749960` (`Demon Bluff Playtest`)
- Steam build: `23084916`
- Unity: `2022.3.10f1`
- Architecture: Windows x86-64
- IL2CPP metadata: version 29
- Build ID: `f530404b0f3f_807de4a83df4`

The immutable input fingerprints are in
[`manifests/builds/`](manifests/builds/). Tool releases and archive checksums are
in [`toolchain/toolchain.lock.json`](toolchain/toolchain.lock.json).

## Reproduce the metadata dump

From PowerShell at the repository root:

```powershell
python reverse_engineering/scripts/build_manifest.py `
  --game-root 'B:\SteamLibrary\steamapps\common\Demon Bluff Playtest' `
  --steam-manifest 'B:\SteamLibrary\steamapps\appmanifest_3749960.acf'

powershell -ExecutionPolicy Bypass -File `
  reverse_engineering/scripts/invoke_il2cppdumper.ps1 `
  -GameRoot 'B:\SteamLibrary\steamapps\common\Demon Bluff Playtest'
```

The dumper command downloads the pinned release, verifies its checksum, and
writes raw output outside the repository by default:

```text
B:\CodexTools\DemonBluffReverseEngineering\artifacts\
  f530404b0f3f_807de4a83df4\il2cppdumper-v6.7.46\
```

Index the result without committing the raw files:

```powershell
python reverse_engineering/scripts/index_il2cpp_dump.py `
  --dump-cs '<artifact-dir>\dump.cs' `
  --script-json '<artifact-dir>\script.json' `
  --output-dir reverse_engineering/generated/current
```

Recover native methods into managed IL with the pinned Cpp2IL development
commit, then render the recovered assembly as local C# with ILSpyCmd:

```powershell
powershell -ExecutionPolicy Bypass -File `
  reverse_engineering/scripts/invoke_cpp2il.ps1 `
  -GameRoot 'B:\SteamLibrary\steamapps\common\Demon Bluff Playtest'
```

Cpp2IL's “success” count means it emitted IL for a method, not that the output
is source-accurate. Complex methods contain explicit recovery warnings and must
be checked against Ghidra/native instructions and live behavior. The checked-in
quality report keeps those warnings visible.

Create a symbolized Ghidra project and export the first native target set in
three stages:

```powershell
powershell -ExecutionPolicy Bypass -File `
  reverse_engineering/scripts/invoke_ghidra.ps1 `
  -GameRoot 'B:\SteamLibrary\steamapps\common\Demon Bluff Playtest' `
  -Stage import

powershell -ExecutionPolicy Bypass -File `
  reverse_engineering/scripts/invoke_ghidra.ps1 `
  -GameRoot 'B:\SteamLibrary\steamapps\common\Demon Bluff Playtest' `
  -Stage analyze

powershell -ExecutionPolicy Bypass -File `
  reverse_engineering/scripts/invoke_ghidra.ps1 `
  -GameRoot 'B:\SteamLibrary\steamapps\common\Demon Bluff Playtest' `
  -Stage export-core
```

Export any other checked-in target set from the completed baseline project with
the generic, read-only target exporter:

```powershell
powershell -ExecutionPolicy Bypass -File `
  reverse_engineering/scripts/invoke_ghidra.ps1 `
  -GameRoot 'B:\SteamLibrary\steamapps\common\Demon Bluff Playtest' `
  -Stage export-target `
  -TargetSet gameplay_lifecycle
```

The import and analysis stages write explicit completion summaries; the wrapper
rejects cancelled imports, analysis timeouts, missing target symbols, partial
exports, stale files, and filename collisions. Raw Ghidra state remains outside
the public repository under the build-keyed artifact directory. The current
`gameplay-core` export is expected to produce 13 of 13 functions. Folded native
bodies retain every requested managed-method identity in their export headers.
The generic exporter applies the same build, RVA, signature, filename, and
count checks without mutating or reanalyzing the saved project; the current
`gameplay_lifecycle` boundary exports 28 of 28 functions. All 28 entries now
carry native-static behavioral coverage in the
[`gameplay_lifecycle` audit](notes/systems/gameplay_lifecycle.md).
The subsequent `gameplay_execution_resolution` boundary baseline-exports 30 of
30 functions, including four explicitly tracked shared-body identities. Its
first 16-method slice now has native-static coverage for action/lying dispatch,
wrong-execution damage, Knight and Doppelganger protection, and terminal result
selection. The remaining 14-method status-insertion and Striga night slice is
also native-audited, closing all 30 selected methods. The next
`gameplay_status_corruption_truth` boundary maps 40 methods and
baseline-exports all 40 without failure. Eight entries deliberately reuse exact
identities from earlier boundaries. Its `Characters.FilterRealCharacterType`
overload uses the optional `prototype_name` field to give Ghidra's C datatype
parser a unique definition name while preserving the exact metadata signature
in `signature`; target validation still checks that unmodified signature and
RVA against `script.json`. All 40 status-storage, selection, truth/appearance,
corruption-producer, Alchemist, and bluff-orchestration methods now have
native-static coverage in the
[`status/corruption/truth audit`](notes/systems/gameplay_status_corruption_truth.md).
The subsequent `gameplay_bluff_acquisition` boundary adds 31 checked methods
covering common assignment, Demon/Minion/Spy/Mutant selection, bluff pools,
script registration, fresh-card creation, delayed-Reveal registration and
resume, first-bluff assignment, and the GameAssembly-to-Unity coroutine/RNG
handoff. Thirteen entries are exact overlaps with earlier sets; CharacterData
overloads receive explicit typed prototype aliases, and the Helpers/Calculator
`RollDice`, `Random.Range`/`RandomRangeInt`, and
`StartCoroutine(IEnumerator)`/`StartCoroutine_Auto` shared native bodies retain
their exact managed identities. All 31 are documented in the
[`bluff-acquisition audit`](notes/systems/gameplay_bluff_acquisition.md).
Its offline clean-room selector ledger now composes Demon, ordinary Minion,
and Drunk acquisitions with exact rational path mass, occurrence-sensitive
must-include consumption, script registration, and Drunk corruption-attempt
effects. It remains separate from full Reveal scheduling and live solver input;
see the [ledger boundary](notes/systems/gameplay_bluff_acquisition.md#composable-offline-selector-ledger).
The subsequent [bounded Reveal callback projection](notes/systems/gameplay_bluff_acquisition.md#bounded-reveal-callback-composition)
adds register-as reset, first-bluff installation, repeated continuations,
real/copied Init and AfterRoundStart dispatch, and Confessor status effects for
Lilis/Twin/Drunk bodies with supported bluff assets. It requires explicit resume
provenance and excludes HealthyBluff, subscribers, and intervening writers.
Its [Spy v2 extension](notes/systems/gameplay_bluff_acquisition.md#spy-register-as-and-role-cache-extension)
adds explicit role-cache identity, script-occurrence-weighted register-as
selection, shared-cache reuse, and live-bluff register-as updates while
preserving the original v1 serialized shape.
The v3 extension adds explicit `characterStartActed` provenance and bounded
HealthyBluff Start replay, including Drunk/Lilis status effects and the
per-trigger truth decision. Active Twin/Spy Start callbacks, subscriptions,
and writer-created continuations remain outside this offline API.
An isolated [Twin Start writer kernel](notes/roles/gameplay_role_twin_minion.md#offline-start-writer-kernel)
now reconstructs the weighted swap, ordered field resets and immediate role
clones, including two new continuations on self-swap. A separate
`character_start_native_v1` projection now joins it to Character.Act's guard,
frozen dispatch and subsequent copied callback, including two Twin swaps in
one call. The `ordered_reveal_writer_native_v1` projection now composes
acquisition, conditional Start and current-role Init/AfterRoundStart across
explicitly ordered resumes, carrying newly created continuation counts.
An offline [sealed ready-batch explorer](notes/systems/gameplay_ready_batch.md)
(`sealed_ready_batch_native_v1`) now enumerates every order
of up to six caller-proven ready continuations. Each order has a separate RNG
distribution, with no scheduler probability assigned. Native readiness capture
and admission of later-ready continuations remain pending.
A logical continuation registry now carries complete pending identities between
explicit batches, with consumed-ID removal and ordered writer-created labels.
Those labels are simulation-local and do not establish native readiness.
The [UnityPlayer wait-boundary audit](notes/systems/unity_wait_boundary.md)
now fingerprints the engine and verifies its diagnostic-linked wait producer,
deadline insertion, consumer eligibility gates and one-shot callback protocol.
The [internal-call registration audit](notes/systems/unity_icall_bindings.md)
validates all 3,447 registered pairs and independently binds StartCoroutineManaged2
and the public time/frame getters, including frame/fixed clock selection.
The [PlayerLoop phase audit](notes/systems/unity_wait_phases.md) binds five
default-loop nodes to wait-dispatch masks, including both mask-2 dynamic-frame
callbacks. Full clock policy, phase bit 8 and callback-mutated dispatch remain
unresolved.
The [native tree differential audit](notes/systems/unity_wait_tree.md) establishes
stable finite equal-deadline occurrence order through insertion, balancing and
removal in the pinned engine, with synthetic records executed in an emulator.
The [one-shot consumer audit](notes/systems/unity_wait_consumer.md) additionally
checks saved-successor behavior under callback insertions/cancellations,
generation exclusion and retained clock samples in 23 isolated native cases.
Full lifetime, reentrant drains and release-body mutation remain separate.
A bounded offline `wait_eligibility` module now projects finite wait arithmetic
and timing gates from explicit producer/consumer snapshots, preserving native
float promotion, signed frame counters and wrapping dispatch generations.
The [one-shot queue projection](notes/systems/unity_wait_queue_projection.md)
adds stable queue traversal, supplied callback insertions/cancellations and
exact release conditions. Its Rust regression compares 23 native-emulated
synthetic cases; owner/lifetime provenance and registry admission remain explicit.
The [scheduled Reveal adapter](notes/systems/scheduled_reveal.md) connects that
same kernel to weighted Reveal and logical continuations for a complete
DelayReveal-only queue, using explicit matching-owner and producer snapshots.
Writer-created 0.3f waits retain branch-local labels and native generation gates.
The [native coroutine bridge](notes/systems/unity_coroutine_bridge.md) now links
valid-owner creation to the immediate managed MoveNext call, WaitForSeconds
registration and the later callback into that same dispatcher.
The [coroutine cancellation audit](notes/systems/unity_coroutine_cancellation.md)
binds handle, IEnumerator and StopAll entry points and verifies 26 native cases
for queue matching, cursor updates and bounded owner-list unlinking.
The [reference-release/finalizer audit](notes/systems/unity_coroutine_release.md)
connects native reference cleanup to managed Coroutine cleanup and verifies
both invocation orders and retained auxiliary cleanup in 14 native cases,
retaining explicit final-reference destructor and lifetime-graph limits.
The [completion/ownership audit](notes/systems/unity_coroutine_completion.md)
executes the dispatcher together with queue release and in-invocation native
stop/release calls in 848 cases, including StopAll retiring a queued sibling.
The [Reveal view-tail audit](notes/systems/gameplay_reveal_view.md) now covers
UpdateView, UpdateViewReal and RefreshView, including death-presentation
creation, preserved icon state and a bounded offline presentation projection.
The `ordered_reveal_writer_view_native_v2` extension carries explicit per-body
UI snapshots through Twin replacements and each Reveal tail, preserving newly
created death presentations across later resumes.
The first per-role boundaries add all ten Slayer methods and all seven Wretch
methods (the latter is managed internally as `Recluse`). Their paired
[`Slayer`](notes/roles/gameplay_role_slayer.md) and
[`Wretch`](notes/roles/gameplay_role_wretch.md) audits join registered
alignment, picker dispatch, the folded Wretch action body, and Slayer's
kill-and-reveal behavior to the live Wretch regression.
The next role-class boundary adds all ten `Dreamer2` methods plus the two
`GetDreamerClue` virtual providers. Its
[`Dreamer2`](notes/roles/gameplay_role_dreamer2.md) audit reconstructs the
two-character randomized type-exclusion result. Serialized asset evidence
binds the public card to managed `Dreamer`, however, and finds no current
gameplay binding for `Dreamer2`; the alternate class therefore does not replace
the live role-pair contract. The subsequent
[`public Dreamer`](notes/roles/gameplay_role_dreamer.md) boundary adds all 11
role methods plus five compiler-generated helpers. It reconstructs the exact
weighted role-pair and Cabbage paths. The corrected
[aligned asset audit](notes/systems/character_asset_flags.md) identifies 15
usually-disguised assets among 46 core records, restoring the script-priority
pool ahead of board helpers. When that pool is empty, the board-entry fallback
can truthfully emit both selected roles, while a selected
bluff can make a lying clue collide with the other target's real role.
The following [`Baa`](notes/roles/gameplay_role_baa.md) boundary asset-binds
the public card to managed `Imp` and adds all three role methods plus the two
deck-view helpers carrying its visible effect. It proves that Baa obscures one
existing Outcast identity at Start and removes that exact entry on any Baa
death; no gameplay role is added and no board card is flipped.
The next [`Shaman`](notes/roles/gameplay_role_shaman.md) boundary asset-binds
the public card to managed `Illuzionist` and distinguishes it from the public
Witch's managed `Cipher`. Its four role methods plus seven exact selection,
status, and lifecycle helpers show an ordered apparent-Villager source and
destination clone: the source stays in place, the destination is overwritten,
and the destination's truth/status and runtime state survive the replacement.
The following
[`Plague Doctor`](notes/roles/gameplay_role_plague_doctor.md) boundary
asset-binds the public card to managed `Puzzlemaster`. All 11 role methods plus
12 exact dispatch, click, picker, status, and filter helpers establish the
Start corruption pool, truthful and lying Day branches, apparent-alignment
random result pools, two-entry acted-information shape, self override, and raw
Corrupted-status handling. In particular, Drunk has no native blanket-clean
exception: the asc84_v2 generated Drunk was compatible with retained
Alchemist resistance blocking its self-corruption.
The next [`Judge`](notes/roles/gameplay_role_judge.md) boundary asset-binds the
public card to managed `Judge2` and proves that the similar `Arbiter` class is
unbound. All ten Judge2 methods plus eight exact dispatch, truth-appearance,
click, and picker helpers establish its unrestricted one-target selection,
exact one-reference result, deterministic normal/bluff inversion, and
`ResetAfterNight` history. A corrupted Good Judge takes `BluffAct` and
deterministically negates target lying appearance; it is not an unconstrained
result.
The following [`Witch`](notes/roles/gameplay_role_witch.md) boundary
asset-binds the public card to managed `Cipher`. All five Cipher methods plus
14 exact Start, inherited-dispatch, player-value, click, reset, and ordinary/
night-death helpers establish a global scalar reveal quota rather than a stored
target: the ordinary transition is allowed exactly while block count is less
than the number of current Hidden states, killed-hidden cards are excluded once
Dead, picker and execution paths bypass the gate, and exact real Witch death
reduces the quota. Inherited `Role.BluffAct` forwards the Evil Witch's Start
back to concrete `Cipher.Act`, closing the apparent lying-dispatch ambiguity.
The next
[`Chancellor/Baron and Witness`](notes/roles/gameplay_role_chancellor.md)
boundary adds all five managed Baron methods, all eight Witness methods, and
18 exact ordered-Start, selection, status, mutation, and roster helpers. It
proves that Chancellor first replaces an anywhere non-Dead real Villager with
an added Outcast role, then independently marks an apparent-Outcast anchor and
swaps Chancellor role data with one alive circular neighbour. The resulting
`c/v/o/f/a` equations distinguish the first target, marker anchor, final
Chancellor, and generated-Outcast home. Witness reads only surviving current
physical `MessedUpByEvil` status: the first Villager is not marked merely by
replacement, dead markers remain eligible, truthful NO requires zero markers,
and lying NO requires every physical card to be marked.
The combined
[`Lilis/Knight`](notes/roles/gameplay_roles_lilis_knight.md) boundary
asset-binds the public cards to managed `Striga` and `Immortal`. It includes
all three Lilis methods, all ten Knight methods, and 41 exact ordered-Start,
Night-rule, victim-filter, delayed-kill, ordinary-execution, Slayer, HP,
status, and reset helpers. The native audit proves a hard registered-Good
first pass rather than weighted selection, fixed two-HP cost on every live
Lilis Night attempt, no reroll after protected or colliding delayed targets,
and per-physical-duplicate Night actions despite one same-asset Start actor.
Knight protection follows HealthyBluff, Corrupted, then runtime-Evil
precedence. A corrupted runtime-Good Knight costs the ordinary five HP plus a
fixed additional four, for nine total; Lilis and Slayer never run that
OnExecuted hook.
The following [`Rambler`](notes/roles/gameplay_role_rambler.md) boundary
asset-binds the public card to managed `Rambler2` and covers all 14 role
methods, both compiler-generated closure methods, and 20 exact setup,
truth-dispatch, adjacency, reveal, interference, and acted-history helpers.
It proves that interference installs during each physical card's internal
pre-flip AfterRoundStart reveal. Clean real and HealthyBluff fake Rambler
surfaces target appearance-truthful neighbours; corrupted real and ordinary
lying fake surfaces target appearance-lying neighbours. Hidden targets retain
persistent callbacks which replace the imminent acted record with exactly one
Rambler reference, while already non-Hidden targets receive a separate history
entry. User-reveal quotes are constraint-free but carry exact circular
predecessor/successor references, including duplicate small-board entries.
The next [`Baker`](notes/roles/gameplay_role_baker.md) boundary asset-binds the
public Good Villager to fieldless managed `Baker` and covers all 11 role
methods, its runtime-data constructor, all three Baker achievement-helper
methods, and 21 exact click, reveal, dispatch, filtering, replacement, lookup,
and acted-history helpers. An allowed click writes Hidden to Alive and
synchronously completes Baker's Day action before OnReveal or the tween. The
conversion uniformly selects an exact Hidden registered-or-real Good Villager,
stores that target's real current name before `InitWithNoReset`, and extends
only on the descendant's later user reveal. Real and lying prior-role clues,
runtime cast failures, Broken/Working/Altered status gates, Shaman composition,
small boards, physical multiplicity, and achievement ordering are closed.
The combined
[`Doppelganger/Drunk`](notes/roles/gameplay_roles_doppelganger_drunk.md)
boundary asset-binds both public Good Outcasts to managed `Doppleganger` and
`Drunk`, covers all 17 declared role methods plus 22 exact setup, reveal,
filter, pool, registration, status, and execution helpers, and closes their
complete disguise lifecycle. Drunk runs before Puppeteer, while both disguise
selectors run only in delayed Reveal after the synchronous ordered Start pass;
a converted former Villager is real non-bluffable Puppet and cannot be copied
through its saved display bluff. The audit also fixes clean/corrupted
Doppelganger source weighting and state-sensitive returned identity, Drunk's
two-draw must-include priority, its bounded not-in-play guarantee, duplicate
pool behavior, failure mutations, and the separation between display bluff,
register-as data, script-role registration, and upstream HUD counts.
The following
[`Fortune Teller`](notes/roles/gameplay_role_fortune_teller.md) boundary
asset-binds the public Good Villager to managed `FortuneTeller`, covers all 11
role methods, all six compiler-generated ordering helpers, and eight exact
dispatch, registered-alignment, click, picker, and acted-record helpers. The
native audit closes unrestricted two-target legality, exact-reference toggling
and `OnPicked` chronology, registered-alignment OR truth, its deterministic
lying complement, the discarded bluff-path random draw, ascending-ID speech
and reference order, exact output strings, `ResetAfterNight` history, and the
truthful both-Evil achievement.
The next
[`Bombardier`](notes/roles/gameplay_role_bombardier.md) boundary asset-binds
the public Good Outcast to exact managed `Saint`, covers all five declared role
methods plus 18 dispatch, death, bookkeeping, and terminal helpers, and closes
the actual broader non-Demon-death loss rule. The terminal predicate follows a
dead card's current `dataRef.role`, not physical origin, display bluff,
register-as data, alignment, or status: a genuine current-data replacement to
Bombardier is fatal even with preserved Evil alignment, while ordinary bluff
and Drunk/Doppel display copies are not. Exact managed `SaintVillager` is also
distinct. Successful forced kills and ordinary `Character.Kill` qualify;
Demon deaths are exempt only through the stored `killedByDemon` flag.
The following [`Pooka`](notes/roles/gameplay_role_pooka.md) boundary
asset-binds the public Evil Demon to exact managed `Pooka`, covers all five
declared role methods plus four status and ordering helpers, and distinguishes
the shipped deterministic Start path from a dormant older helper. The active
path visits both circular neighbours, qualifies each by current real Villager
type, and independently attempts Corrupted then MessedUpByEvil. A native xref
scan finds no executable caller for private `PoisonClosestNeighbours`; its
random-one-neighbour, Corrupted-only body has only the ordinary IL2CPP method
registration pointer. Ordinary duplicate Pookas run only the highest-ID match,
and the role owns no clue, picker, reset history, or achievement action.
The next [`Poisoner`](notes/roles/gameplay_role_poisoner.md) boundary
asset-binds the public Evil Minion to exact managed `Poisoner`, covers all four
declared role methods plus 13 dispatch, lifecycle, adjacency, filtering,
resistance, status, output, and integer-RNG helpers, and closes its live
ordered-Start behavior. Every exact-data duplicate acts high-ID first after
Pooka and before Drunk. Each action filters the previous-then-next pair to
current real Villagers missing both Corrupted and Corrupted resistance, draws
one occurrence, then independently attempts Corrupted and MessedUpByEvil.
Dead cards remain eligible, the two-card pair repeats its sole neighbour, and
the one-card self pair filters to an empty no-op. The managed class has no
dormant alternate helper; only its older `good`/`Poisoned` description is
legacy text.
The following [`Twin Minion`](notes/roles/gameplay_role_twin_minion.md)
boundary asset-binds the public Evil Minion to exact managed `Marionette`,
covers all five declared role methods plus 15 ordered-Start, dispatch, Demon-
filter, alive-adjacency, current-data replacement, delayed-reveal, bluff, and
integer-RNG helpers, and closes its shipped two-draw identity mutation. It
swaps current `CharacterData` with one alive neighbour of a selected current
Demon while preserving physical alignment, status, resistance, runtime data,
and ID. Existing reveal coroutines are not cancelled, a same-card branch still
performs both reinitializations, and the private duplicate helper has no
executable caller. This disproves stable Twin/Demon adjacency and exposes an
explicit solver/live identity-trace parity gap.
The following [`Poet`](notes/roles/gameplay_role_poet.md) boundary asset-binds the
public Good Villager to exact managed `Gossip`, covers all six declared Gossip
methods, the twelve exact provider constructors, and generic Character action
dispatch, and closes the shipped selector. Every real or bluff result makes
one fresh `Random.Range(0, Count)` provider draw and delegates to that
provider's corresponding virtual information method. The constructor pool is
exactly Lover, Scout, Oracle, Bounty Hunter, Medium, Knitter, Hunter,
Enlightened, Empress, Bishop, Gemcrafter, and Bard in that order. Current live
payloads now carry a strict provenance marker while unmarked historical
fixtures retain legacy compatibility.
The latest combined
[`Scout/Hunter`](notes/roles/gameplay_roles_scout_hunter.md) boundary closes
two of those provider bodies and their direct public roles. It covers every
method declared by managed `Scout` and `Tracker` plus nine exact selection,
registration, distance, range-reference, calculator, and RNG helpers. Scout
selects a runtime-Evil occurrence, truthfully measures its nearest other
registered Evil, uses an explicit one-Evil sentence, and lies only with
distance 1 through 3 while retaining a selected candidate name. Public Hunter
binds managed `Tracker`, truthfully returns the nearest registered Evil or
exactly `N - 1`, and lies with a different member of
`1..=floor(N / 2)`. Its acted record stores forward then reverse range
references, including a duplicated opposite card on even boards.
The newest [`Oracle`](notes/roles/gameplay_role_oracle.md) boundary asset-binds
the public role to managed `Investigator`, covers all seven declared role
methods, all six generated comparer methods, and six exact Character, pool,
and fallback helpers. Truth independently draws one current registered Minion
and one current registered-Good occurrence, preserves a possible moved-Twin
duplicate reference, and emits exact `There are no minions` text when its
Minion pool is empty. Bluff draws two distinct registered-Good Characters and
uses a script Minion label, falling back to the all-ascension Minion pool.
Direct and Poet observations now share one strict current payload and validator.
The newest [`Lover`](notes/roles/gameplay_role_lover.md) boundary asset-binds
the public role to managed `Empath`, covers all nine declared role methods,
the exact circular-adjacency and registered-alignment helpers, and all four
achievement-helper methods. Truth counts registered-Evil previous/next
occurrences without deduplication and stores those exact references. Bluff
removes truth from the authored Minion-plus-Demon count domain before one
integer-index draw. Direct and Poet/Lover observations now enforce the same
exact text, reference shape, and current provenance schema while preserving
unmarked historical fixtures.
The newest
[`Bounty Hunter`](notes/roles/gameplay_role_bounty_hunter.md) boundary covers
all eight methods declared by managed `BountyHunter` plus exact board,
registered-alignment, acted-record, and integer-RNG helpers. Its dormant direct
Start path uniformly chooses registered Good and changes only physical runtime
alignment. The active Poet provider truthfully chooses registered Evil, bluffs
from registered Good, and emits exact two-line text with no acted references.
Current solver observations enforce one joint anonymous-Wretch assignment;
the two duplicate declared helpers are proven unreachable from executable code.
The newest [`Medium`](notes/roles/gameplay_role_medium.md) boundary asset-binds
the public role to managed `Lookout`, covers all eight declared methods plus
registered-alignment, live-identity, raw-status, acted-record, and integer-RNG
helpers. Truth samples the complete registered-Good board and excludes the
actor only when another candidate exists. Bluff prefers non-actor characters
with a persisted raw bluff and falls back to self only when none exist. Both
paths preserve one selected reference and exact two-line `real`/Drunk
`actually` wording; direct and Poet observations share the strict current
schema while unmarked historical fixtures retain their legacy path.
The newest [`Knitter`](notes/roles/gameplay_role_knitter.md) boundary asset-binds
the public role to managed `Knitter`, covers all eight declared methods plus
registered-alignment, acted-record, count-removal, and integer-RNG helpers, and
closes both direct and Poet use. Truth counts circular physical neighbour pairs
through register-as-first alignment, retaining the singleton self-edge and both
directional edges on two-card boards. Bluff removes truth from
`[0, max(authored Demons + Minions, 2))` before one retained-index draw. Exact
current observations use one shared hidden-state search, including delayed
Baker-to-Spy registration chronology, while unmarked fixtures retain their
legacy path.
The newest
[`Enlightened`](notes/roles/gameplay_role_enlightened.md) boundary asset-binds
the public role to managed `Shugenja`, covers all nine declared role methods
plus exact registered-alignment, acted-record, circle-rotation, runtime-data,
and float-RNG helpers, and closes both direct and Poet use. Truth scans the
complete physical circle for the nearest registered Evil, with increasing
public IDs named Counter-clockwise and decreasing IDs named Clockwise; ties,
double exhaustion, and every two-card board are Equidistant. Bluff makes one
float draw and emits one of the two false directions. Current observations
enforce exact text, zero references, runtime-data agreement, joint anonymous-
Wretch assignments, and delayed Baker-to-Spy registration chronology while
unmarked fixtures retain their legacy path.
The newest [`Bishop`](notes/roles/gameplay_role_bishop.md) boundary asset-binds
the public role to managed `Bishop`, covers all 17 declared role/compiler-
generated methods plus registered-data, character-type, acted-record, list-
shuffle, and integer/float RNG helpers, and closes direct and Poet use. Truth
samples live register-as-first Outcast and Villager pools when present, then a
Minion or Demon with exact Minion precedence. Bluff samples only live projected
Villagers while its two- or three-entry type multiset follows authored
town/outcast/minion counts. IDs, types, and acted references are separately
ordered; strict current observations join them to anonymous Wretch,
identity-mover, and delayed Baker-to-Spy worlds while unmarked fixtures retain
their legacy path.
The newest [`Empress`](notes/roles/gameplay_role_empress.md) boundary asset-binds
the public role to managed `Noble`, covers all 14 declared role/compiler-
generated methods plus registered-alignment, acted-record, pool-filter, and RNG
helpers, and closes direct and Poet use. Truth samples two distinct live
registered-Good occurrences after removing the actor only from that pool, then
one live registered-Evil occurrence; bluff samples three distinct registered-
Good occurrences after actor removal. Both paths make three integer selection
draws, sort three references by displayed ID with float secondary keys, and
emit exact `One is Evil:` text whose references match that order. Strict
current observations join this contract to anonymous-Wretch and Baker-to-Spy
registration worlds while unmarked fixtures retain their legacy path.
The newest
[`Gemcrafter`](notes/roles/gameplay_role_gemcrafter.md) boundary asset-binds the
public role to managed `Archivist`, covers all seven declared methods plus
registered-alignment, acted-record, pool-filter, and integer-RNG helpers, and
closes direct and Poet use. Truth samples one live registered-Good occurrence;
bluff samples one live registered-Evil occurrence. Both inspect the original
pool and remove the actor only when it contains more than one member, so a sole
eligible actor remains selectable. Each path makes one integer draw and emits
exact `#X is Good` text with the same single acted reference. Strict current
observations join this contract to anonymous-Wretch and Baker-to-Spy worlds,
while unmarked clues and Rambler interruptions retain their legacy paths.
The newest [`Bard`](notes/roles/gameplay_role_bard.md) boundary asset-binds the
public role to managed `Acrobat2`, covers all nine declared methods plus acted-
record, circular-order, range-reference, false-number, and integer-RNG helpers,
and closes direct and Poet use. Truth scans the physical circle for the nearest
other direct-Corrupted status and returns zero when none exists. Bluff draws
one retained value from fixed domain `{0,1,2,3}` after removing truth when it
is in-domain, without clamping to board geometry. Both paths emit exact text
and forward-then-reverse range endpoints, preserving duplicate opposite seats
and empty oversized ranges. Strict current observations also preserve native
real-role/bluff-role callback order and join raw-bluff identity plus Baker/Spy
chronology globally, while unmarked fixtures retain their legacy path.
The newest
[`Confessor`](notes/roles/gameplay_role_confessor.md) boundary asset-binds the
public role to managed `Confessor`, covers all nine declared methods plus acted-
record, status, registered-alignment, animated-art, and exact-membership
helpers, and closes direct use while proving current Poet absence. Truth and
bluff deterministically emit the same exact Good/dizzy result from direct
Corruption or registered-Evil alignment, except current Spy data always forces
Good. The result has a native-null reference list, no runtime data, and zero
RNG. Strict observations preserve that null provenance and join current data,
raw bluff/register-as identity, callback order, anonymous Wretch, and
Baker/Spy chronology globally; unmarked fixtures retain their legacy path.
The newest [`Druid`](notes/roles/gameplay_role_druid.md) boundary asset-binds
the public role to managed `Librarian`, covers all ten declared role methods,
all six compiler-generated ordering helpers, and 20 picker, acted-record,
registered-data, pool-filter, lifecycle, and RNG helpers. Its resettable Day
picker accepts any three distinct physical Characters and retains click-order
references while sorting only displayed IDs. Truth uniformly samples selected
registered-Outcast occurrences; Wretch and stable Spy are excluded while
ordinary Doppelganger and Drunk remain eligible. Bluff is the exact complement
and uses the authored non-bluffable Outcast ladder for false positives. Strict
direct observations join current data, raw-bluff identity, anonymous Outcasts,
and Baker/Spy chronology globally; current Poet excludes Druid and unmarked
fixtures retain their legacy path.

Build the deterministic IL2CPP datatype archive, create the isolated typed
project, analyze it, and export any checked-in target set with:

```powershell
powershell -ExecutionPolicy Bypass -File `
  reverse_engineering/scripts/invoke_ghidra.ps1 `
  -GameRoot 'B:\SteamLibrary\steamapps\common\Demon Bluff Playtest' `
  -Stage build-types

powershell -ExecutionPolicy Bypass -File `
  reverse_engineering/scripts/invoke_ghidra.ps1 `
  -GameRoot 'B:\SteamLibrary\steamapps\common\Demon Bluff Playtest' `
  -Stage typed-import

powershell -ExecutionPolicy Bypass -File `
  reverse_engineering/scripts/invoke_ghidra.ps1 `
  -GameRoot 'B:\SteamLibrary\steamapps\common\Demon Bluff Playtest' `
  -Stage typed-analyze

# Reapply the current canonical signatures to an already analyzed project,
# with analysis disabled, then validate every checked target set.
powershell -ExecutionPolicy Bypass -File `
  reverse_engineering/scripts/invoke_ghidra.ps1 `
  -GameRoot 'B:\SteamLibrary\steamapps\common\Demon Bluff Playtest' `
  -Stage typed-refresh

# Repeat the post-save, read-only signature/ABI validation without reanalysis.
powershell -ExecutionPolicy Bypass -File `
  reverse_engineering/scripts/invoke_ghidra.ps1 `
  -GameRoot 'B:\SteamLibrary\steamapps\common\Demon Bluff Playtest' `
  -Stage typed-validate

powershell -ExecutionPolicy Bypass -File `
  reverse_engineering/scripts/invoke_ghidra.ps1 `
  -GameRoot 'B:\SteamLibrary\steamapps\common\Demon Bluff Playtest' `
  -Stage typed-export `
  -TargetSet gameplay_role_druid
```

`build-types` normalizes the private `il2cpp.h`, validates 5,830 inheritance
rewrites and 6,159 explicit alignments, and builds one deterministic GDT from
the union of every checked target set. The current archive contains 151,788
datatypes. Its fifty-set inventory contains 996 target memberships, 643
distinct selected FunctionDefinitions, and 535 unique native RVAs. The typed
project is
separate from the baseline project. It applies only datatype graphs reachable
from the checked-in function signatures and validates exact entry points,
labels, prototypes, dynamic Windows x64 storage, and transaction completion
before writing success summaries. `typed-analyze` and `typed-all` also reopen
the saved program read-only and repeat those checks; `typed-export` requires
their fresh success summaries and opens the project read-only. `typed-refresh`
is the bounded post-analysis path for a changed canonical signature: it reopens
the preserved project with analysis disabled, reapplies all target sets, saves,
and then performs the same exact validations in a separate read-only headless
pass. A single all-target invocation can exceed Windows' command-line limit
before Ghidra launches. `typed-refresh` and `typed-validate` therefore split
the deterministic target inventory into serialized batches of at most eight
sets; the current fifty-set run used seven batches for each phase. Ghidra
commands still must not overlap on the saved project.

The preserved fully analyzed typed project now covers all fifty target
sets after a no-analysis refresh. Three hundred fifty-three memberships are
exact FunctionDefinition overlaps between boundaries. Folded/shared bodies make
the 643 selected definitions exceed the 535 unique native RVAs by 108;
each canonical native prototype is explicit while all exact managed
definitions remain in the GDT. The original full
import added 2,032 reachable datatypes and completed its analysis pass in 2,781
seconds without a timeout. Subsequent refreshes imported 121 additional
reachable datatypes, including 40 for the public Dreamer boundary, six for
Baa, eight for Plague Doctor, six for Judge, and 12 for Witch; the latest
Chancellor/Witness refresh imported 12 more, and the first Lilis/Knight refresh
imported 212 more. The Rambler refresh imported 36 more, and the first Baker
refresh imported 18 more. The first Doppelganger/Drunk refresh required no new
reachable datatypes. The Fortune Teller refresh imported 26 additional
reachable datatypes. The Bombardier refresh imported six additional reachable
datatypes. The Pooka refresh required no new reachable datatype import. The
Poisoner refresh also required no new reachable datatype import. The Twin
Minion refresh imported six additional reachable datatypes. The Poet refresh
imported 17 additional reachable datatypes. The Scout/Hunter refresh imported
five additional reachable datatypes. The Oracle refresh imported six additional
reachable datatypes. The Lover refresh imported 12 additional reachable
datatypes. The first Bounty Hunter application imported six additional
reachable datatypes. The first Medium application imported six additional
reachable datatypes. The Knitter refresh imported six additional datatypes,
the Enlightened refresh imported 12, the Bishop refresh imported 38, and the
Empress refresh imported six additional reachable datatypes, the Gemcrafter
refresh imported six more, the Bard refresh imported six more, and the
Confessor refresh imported three more. The Druid refresh imported 157 more and
canonicalized six shared bodies. The bluff-acquisition pool refresh imported
six additional reachable datatypes. Its scheduler-handoff expansion added two
FunctionDefinitions to the rebuilt GDT and required no additional reachable
datatype imports during application. The Spy boundary adds five FunctionDefinitions
without additional reachable datatype imports. Managed Mutant adds five more
FunctionDefinitions without additional reachable datatype imports. The refresh
reapplied and validated all 996 memberships without rerunning auto-analysis. The final
read-only pass validated all 996 memberships (643 exact definitions) and 2,861
membership-level parameter-storage locations with zero program mutations.

The three new mode target sets completed baseline and typed exports at 6/6,
17/17 and 5/5. Their quality reports remove all 120 placeholder parameter tokens;
raw field-offset accesses fall from 141 to 82 with no added decompiler warnings.

The signature-application ABI check now derives each of the first four Win64
register families from the parameter datatype: integer and pointer parameters
use `RCX`/`RDX`/`R8`/`R9` at their ordinal, while `float` and `double` use the
corresponding `XMM0`-`XMM3` family. It also recognizes Ghidra's width-specific
names such as `XMM2_Da` as members of `XMM2`. Rambler's mixed float-argument
`Character.InterfereActed` and `Character.InterfereDelay` signatures exercise
this path. This is an ABI-validator correction, not a gameplay inference.

Compare private baseline and typed exports without putting decompiled bodies or
private paths in the public report:

```powershell
python reverse_engineering/scripts/audit_ghidra_type_quality.py `
  --baseline '<baseline-export-dir>' `
  --typed '<typed-export-dir>' `
  --output reverse_engineering/reports/<report-name>.json `
  --check
```

For the current exports, unresolved-type tokens fell from 78 to 43 in
`gameplay_core`, from 160 to 96 in `gameplay_execution_resolution`, from 370
to 140 in `gameplay_lifecycle`, from 38 to 34 in
`gameplay_roster_helpers`, from 281 to 145 in
`gameplay_status_corruption_truth`, and from 244 to 74 in
`gameplay_bluff_acquisition`. The role boundaries fell from 70 to 51 for
Slayer, from 14 to 5 for Wretch, from 105 to 80 for Dreamer2, and from 190 to
92 for the public Dreamer, from 25 to 17 for Baa, from 95 to 18 for Shaman,
from 190 to 123 for Plague Doctor, from 146 to 80 for Judge, from 119 to 30 for
Witch, from 237 to 105 for Chancellor/Witness, and from 387 to 154 for the
combined Lilis/Knight boundary. Rambler fell from 405 to 103, and Baker fell
from 261 to 71. The combined Doppelganger/Drunk boundary fell from 216 to 72.
Fortune Teller fell from 171 to 67, Bombardier fell from 134 to 39, and Pooka
fell from 53 to 31. Poisoner fell from 104 to 46, Twin Minion fell from 168 to
45, Poet fell from 74 to 9, Scout/Hunter fell from 148 to 56, Oracle fell
from 116 to 78, Lover fell from 92 to 38, Bounty Hunter fell from 68 to 32,
Medium fell from 87 to 30, Knitter fell from 55 to nine, and Enlightened fell
from 97 to 24. Bishop fell from 187 to 98, Empress fell from 102 to 45,
Gemcrafter fell from 67 to 27, Bard fell from 62 to 18, and Confessor fell
from 51 to four. Druid fell from 262 to 101.
Raw field-offset accesses fell from 237 to 144, from
243 to 120, from 678 to 289, from 76 to 41, from 421 to 148, and from 361 to 88
for the six subsystem boundaries, then from 97 to 83 for Slayer, from 20 to 8
for Wretch, from 167 to 156 for Dreamer2, from 370 to 186 for the public
Dreamer, from 33 to 28 for Baa, from 144 to 21 for Shaman, from 329 to 223
for Plague Doctor, from 268 to 175 for Judge, and from 241 to 89 for Witch.
The Chancellor/Witness boundary fell from 294 raw field-offset accesses to
102, Lilis/Knight fell from 581 to 203, Rambler fell from 699 to 95, and Baker
fell from 396 to 85. Doppelganger/Drunk fell from 266 to 54.
Fortune Teller fell from 286 to 194, Bombardier fell from 245 to 98, and Pooka
fell from 42 to 28. Poisoner fell from 105 to 26, Twin Minion fell from 211 to
39, Poet fell from 144 to 21, Scout/Hunter fell from 98 to 55, Oracle fell
from 139 to 101, Lover fell from 59 to 24, Bounty Hunter fell from 48 to 22,
Medium fell from 72 to 18, Knitter fell from 43 to 26, and Enlightened fell
from 60 to 38. Bishop fell from 251 to 156, Empress fell from 122 to 92,
Gemcrafter fell from 53 to 14, Bard fell from 47 to 25, and Confessor fell
from 55 to three. Druid fell from 424 to 261.
Error-marker counts did not increase;
lifecycle and the status boundary each gained one nonfatal decompiler warning,
and the expanded bluff-acquisition boundary retained three error markers,
gained one nonfatal warning, reduced placeholder parameters from 320 to zero,
and reduced indirect-call patterns from ten to zero. Eight role reports
retained their baseline warning counts, and Witch and the Chancellor/Witness
boundary each gained one nonfatal warning marker. The
Lilis/Knight boundary retained four error markers and gained one nonfatal
warning; placeholder parameters fell from 451 to zero and indirect-call
patterns from 44 to 12. Rambler retained zero error markers, gained one
nonfatal warning, reduced placeholder parameters from 436 to zero, and reduced
indirect-call patterns from 16 to three. Baker retained its two error and 50
warning markers, reduced placeholder parameters from 360 to zero, and reduced
indirect-call patterns from 24 to four. Doppelganger/Drunk retained one error
marker, gained one nonfatal warning, reduced placeholder parameters from 291
to zero, and reduced indirect-call patterns from nine to zero. Fortune Teller
retained zero error markers and 43 warning markers, reduced placeholder
parameters from 163 to zero, and reduced indirect-call patterns from 15 to
five. Bombardier retained three error markers and 21 warning markers, reduced
placeholder parameters from 172 to zero, and reduced indirect-call patterns
from 27 to 11. Pooka retained zero error markers and 11 warning markers,
reduced placeholder parameters from 49 to zero, and retained zero
indirect-call patterns. Poisoner retained two error markers, gained one
nonfatal warning marker, reduced placeholder parameters from 141 to zero, and
reduced indirect-call patterns from five to zero. Twin Minion retained two
error markers, gained one nonfatal warning marker, reduced placeholder
parameters from 218 to zero, and reduced indirect-call patterns from 11 to
zero. Poet retained three error markers and twelve warning markers, reduced
placeholder parameters from 98 to zero, and reduced indirect-call patterns
from eight to one. Scout/Hunter retained six error and 35 warning markers,
reduced placeholder parameters from 142 to zero, and reduced indirect-call
patterns from eight to zero. Oracle retained two error and 27 warning markers,
reduced placeholder parameters from 69 to zero, and reduced indirect-call
patterns from four to zero. Lover retained two error and 23 warning markers,
reduced placeholder parameters from 82 to zero, and reduced indirect-call
patterns from four to zero. Bounty Hunter retained three error and 18 warning
markers, reduced placeholder parameters from 54 to zero, and reduced indirect-
call patterns from four to zero. Medium retained three error and 21 warning
markers, reduced placeholder parameters from 79 to zero, and reduced indirect-
call patterns from four to zero. Knitter retained three error and 13 warning
markers, reduced placeholder parameters from 67 to zero, and reduced indirect-
call patterns from four to zero. Enlightened retained three error and 20
warning markers, reduced placeholder parameters from 87 to zero, and reduced
indirect-call patterns from four to zero. Bishop retained six error and 46
warning markers, reduced placeholder parameters from 110 to zero, and reduced
indirect-call patterns from six to zero. Empress retained four error and 21
warning markers, reduced placeholder parameters from 80 to zero, and reduced
indirect-call patterns from four to zero. Gemcrafter retained three error and
15 warning markers, reduced placeholder parameters from 58 to zero, and
reduced indirect-call patterns from four to zero. Bard retained three error
and 16 warning markers, reduced placeholder parameters from 91 to zero, and
reduced indirect-call patterns from four to zero. Confessor retained four
error and 15 warning markers, reduced placeholder parameters from 113 to zero,
and reduced indirect-call patterns from six to zero. Druid retained two error
and 75 warning markers, reduced placeholder parameters from 245 to zero, and
reduced indirect-call patterns from 17 to six. The original typed import
is recorded in
[`reports/f530404b0f3f_807de4a83df4_typed_import.json`](reports/f530404b0f3f_807de4a83df4_typed_import.json),
with the new role comparisons in the
[`Slayer typed-quality report`](reports/f530404b0f3f_807de4a83df4_typed_quality_gameplay_role_slayer.json)
and
[`Wretch typed-quality report`](reports/f530404b0f3f_807de4a83df4_typed_quality_gameplay_role_wretch.json),
plus the
[`Dreamer2 typed-quality report`](reports/f530404b0f3f_807de4a83df4_typed_quality_gameplay_role_dreamer2.json)
and
[`public Dreamer typed-quality report`](reports/f530404b0f3f_807de4a83df4_typed_quality_gameplay_role_dreamer.json).
The Baa comparison is in the
[`Baa typed-quality report`](reports/f530404b0f3f_807de4a83df4_typed_quality_gameplay_role_baa.json),
and the Shaman comparison is in the
[`Shaman typed-quality report`](reports/f530404b0f3f_807de4a83df4_typed_quality_gameplay_role_shaman.json).
The Plague Doctor comparison is in the
[`Plague Doctor typed-quality report`](reports/f530404b0f3f_807de4a83df4_typed_quality_gameplay_role_plague_doctor.json).
The Judge comparison is in the
[`Judge typed-quality report`](reports/f530404b0f3f_807de4a83df4_typed_quality_gameplay_role_judge.json).
The Witch comparison is in the
[`Witch typed-quality report`](reports/f530404b0f3f_807de4a83df4_typed_quality_gameplay_role_witch.json).
The Chancellor/Witness comparison is in the
[`Chancellor/Witness typed-quality report`](reports/f530404b0f3f_807de4a83df4_typed_quality_gameplay_role_chancellor.json).
The combined boundary comparison is in the
[`Lilis/Knight typed-quality report`](reports/f530404b0f3f_807de4a83df4_typed_quality_gameplay_roles_lilis_knight.json).
The current Rambler comparison is in the
[`Rambler typed-quality report`](reports/f530404b0f3f_807de4a83df4_typed_quality_gameplay_role_rambler.json).
The current Baker comparison is in the
[`Baker typed-quality report`](reports/f530404b0f3f_807de4a83df4_typed_quality_gameplay_role_baker.json).
The current combined comparison is in the
[`Doppelganger/Drunk typed-quality report`](reports/f530404b0f3f_807de4a83df4_typed_quality_gameplay_roles_doppelganger_drunk.json).
The current Fortune Teller comparison is in the
[`Fortune Teller typed-quality report`](reports/f530404b0f3f_807de4a83df4_typed_quality_gameplay_role_fortune_teller.json).
The current Bombardier comparison is in the
[`Bombardier typed-quality report`](reports/f530404b0f3f_807de4a83df4_typed_quality_gameplay_role_bombardier.json).
The current Pooka comparison is in the
[`Pooka typed-quality report`](reports/f530404b0f3f_807de4a83df4_typed_quality_gameplay_role_pooka.json).
The current Poisoner comparison is in the
[`Poisoner typed-quality report`](reports/f530404b0f3f_807de4a83df4_typed_quality_gameplay_role_poisoner.json).
The current Twin Minion comparison is in the
[`Twin Minion typed-quality report`](reports/f530404b0f3f_807de4a83df4_typed_quality_gameplay_role_twin_minion.json).
The current Poet comparison is in the
[`Poet typed-quality report`](reports/f530404b0f3f_807de4a83df4_typed_quality_gameplay_role_poet.json).
The current combined Scout/Hunter comparison is in the
[`Scout/Hunter typed-quality report`](reports/f530404b0f3f_807de4a83df4_typed_quality_gameplay_roles_scout_hunter.json).
The current Oracle comparison is in the
[`Oracle typed-quality report`](reports/f530404b0f3f_807de4a83df4_typed_quality_gameplay_role_oracle.json).
The current Lover comparison is in the
[`Lover typed-quality report`](reports/f530404b0f3f_807de4a83df4_typed_quality_gameplay_role_lover.json).
The current Bounty Hunter comparison is in the
[`Bounty Hunter typed-quality report`](reports/f530404b0f3f_807de4a83df4_typed_quality_gameplay_role_bounty_hunter.json).
The current Medium comparison is in the
[`Medium typed-quality report`](reports/f530404b0f3f_807de4a83df4_typed_quality_gameplay_role_medium.json).
The current Knitter comparison is in the
[`Knitter typed-quality report`](reports/f530404b0f3f_807de4a83df4_typed_quality_gameplay_role_knitter.json).
The current Enlightened comparison is in the
[`Enlightened typed-quality report`](reports/f530404b0f3f_807de4a83df4_typed_quality_gameplay_role_enlightened.json).
The current Bishop comparison is in the
[`Bishop typed-quality report`](reports/f530404b0f3f_807de4a83df4_typed_quality_gameplay_role_bishop.json).
The current Empress comparison is in the
[`Empress typed-quality report`](reports/f530404b0f3f_807de4a83df4_typed_quality_gameplay_role_empress.json).
The current Gemcrafter comparison is in the
[`Gemcrafter typed-quality report`](reports/f530404b0f3f_807de4a83df4_typed_quality_gameplay_role_gemcrafter.json).
The current Bard comparison is in the
[`Bard typed-quality report`](reports/f530404b0f3f_807de4a83df4_typed_quality_gameplay_role_bard.json).
The current Confessor comparison is in the
[`Confessor typed-quality report`](reports/f530404b0f3f_807de4a83df4_typed_quality_gameplay_role_confessor.json).
The current Druid comparison is in the
[`Druid typed-quality report`](reports/f530404b0f3f_807de4a83df4_typed_quality_gameplay_role_druid.json).
The current bluff-acquisition comparison is in the
[`bluff-acquisition typed-quality report`](reports/f530404b0f3f_807de4a83df4_typed_quality_gameplay_bluff_acquisition.json).

The [Spy managed boundary](notes/roles/gameplay_role_spy.md) closes all six
Spy declarations and three inherited/shared callees. Its inert real/copied
Start dispatch now composes with cache-aware Reveal and the native queue;
no current shipped Spy asset binding is assumed. The native audit verifies
17 instruction/literal relationships and four pinned asset-name observations.
Its [typed-quality report](reports/f530404b0f3f_807de4a83df4_typed_quality_gameplay_role_spy.json)
compares nine completed exports in each project and passes all quality gates.

The [shared void return audit](notes/systems/folded_void_return.md) verifies
175 direct managed definitions against the regenerated denominator and 64
isolated native executions. It adds 169 explicit method classifications,
including delayed Reveal's empty Dispose, without inferring caller behavior.

The [shared scaffolding audit](notes/systems/shared_scaffolding.md) verifies
503 direct constructor/getter definitions and their individual metadata fields
with 384 isolated native cases. It adds 454 classifications while preserving
49 existing records, including DelayReveal's state constructor and Current.

The [managed Mutant boundary](notes/roles/gameplay_role_managed_mutant.md)
closes all six declarations separately from public Mutant/Skinwalker. Its
offline selector preserves Mad across empty-draw failure and compares against
16 native caller cases with explicit service gateways.

## Method coverage

[`coverage/`](coverage/) contains the deterministic 4,207-method denominator,
sparse authored classifications, and reusable evidence. Missing classifications
resolve to `unresolved/not-reviewed`; shared native RVAs never collapse managed
method identities. The current overlay contains 1,579 classifications backed by
421 evidence records. See [`coverage/README.md`](coverage/README.md) for the
generation and byte-for-byte check command.

## Evidence levels

Every behavioral or layout claim should carry one of these labels:

- `metadata`: directly present in IL2CPP metadata, such as a field offset.
- `native-static`: recovered from disassembly/decompilation.
- `native-emulated`: pinned native routines executed in an isolated emulator
  against authored synthetic inputs, with explicit environment and scope.
- `live-validated`: confirmed against paired UI and process-memory observations.
- `behavioral`: confirmed through controlled gameplay or regression traces.
- `hypothesis`: plausible but not yet validated.

Memory remains validation-only during solver play, per `AGENTS.md`.

## Directory map

```text
manifests/builds/       Immutable hashes and build identity
toolchain/              Pinned tool versions and configuration
scripts/                Reproducible extraction and indexing tools
offsets/                Versioned field/RVA evidence
symbols/                Small normalized, reviewable symbol indexes
coverage/               Complete method denominator and sparse evidence
notes/systems/           Authored subsystem reconstruction notes
notes/roles/             Authored per-role reconstruction notes
reports/                 Coverage and build-diff summaries
fixtures/synthetic/      Redistributable validation fixtures
work/                    Ignored raw inputs and local analysis projects
generated/               Ignored reproducible bulk output
private/                 Ignored non-redistributable material
```

See [`ROADMAP.md`](ROADMAP.md) for the completion definition and milestones.

Public Mutant is now audited as [Skinwalker](notes/roles/gameplay_role_public_mutant.md),
separate from managed Mutant. Its 14-target boundary completes the declared
Skinwalker and Demon surfaces. The [ascension asset graph](notes/systems/ascension_asset_graph.md)
fully parses 61 configuration objects and corrects older Compendium-as-pool
provenance claims; mode selection and temporary-profile writers remain open.

The [ascension setup boundary](notes/systems/ascension_setup.md) now covers
33 methods and 412 native cases, including mode selection, cached script RNG,
shared-reference copying and partial failures. An offline weighted Rust kernel
replays script selection and retains superseded draws. All 662 Rust unit tests
and the release build passed. Generic copy internals and lock policy are the
next audited callees; the complete mode-to-Reveal lifecycle remains open.

The [ascension helper boundary](notes/systems/ascension_helpers.md) verifies
140 native cases for JSON-copy callers and the distinct lock/unlock selectors.
The [fully-shared follow-up](notes/systems/classconv_shared_copy.md) adds 132
native cases and closes both generic definitions' alternate-body gap. Managed
JSON wrappers are traced to their engine
internal-call boundary; UnityPlayer serialization remains open.

The [engine JSON gateway](notes/systems/unity_json_gateway.md) now binds those
signature-bearing requests through the actual fallback lookup to a separate
two-entry registration module. Its 31 native cases cover ToJson marshaling and
inline results. The [FromJson gateway](notes/systems/unity_fromjson_gateway.md)
adds 180 native cases for rank rejection, create/overwrite class selection,
allocation ordering and partial failures. Core parsing and field application
remain explicit services. The [native parser and tree renderer](notes/systems/unity_json_parser.md)
adds 89 natural input cases and eighteen controlled diagnostics, preserving
duplicate keys, byte strings and exact numeric payloads. Metadata-directed
managed field application remains open. The
[metadata adapter audit](notes/systems/unity_json_fields.md) adds 166 fixtures for
cache selection and native descriptor traversal while retaining individual field
processors and metadata discovery as explicit services.
The [native reader registry](notes/systems/unity_json_registry.md) adds 32
fixtures that construct all 33 handler records, the optional 34th extension,
storage resets and retained failure prefixes. Class discovery and field
conversion remain separate boundaries.
The [numeric reader follow-up](notes/systems/unity_json_primitives.md) executes
nine actual field handlers in 333 scalar fixtures and sixteen composed adapter
batches. Lookup, conversion and field writes are native; descriptor construction,
managed type discovery and the remaining field families stay open.
The [numeric descriptor factory](notes/systems/unity_json_descriptors.md) now
executes 105 construction fixtures and 36 factory-to-adapter numeric copies.
Registry lookup and descriptor writes are native; runtime metadata exports,
eligibility and enumeration remain separate boundaries.
The [native metadata builder](notes/systems/unity_json_metadata.md) adds 226
fixtures for enumeration, eligibility, parent ordering and joined numeric copy.
It builds descriptors natively over supplied runtime metadata; compound types,
real metadata discovery and writer conversion stay open.
The [numeric serializer/reader join](notes/systems/unity_json_numeric_roundtrip.md)
now executes 110 compact/pretty round trips and 32 writer-service stops. It joins
native writer registry, metadata construction, field conversion and rendering to
native reload, retaining observed boolean and floating-point normalization. Runtime
metadata discovery and compound/reference serialization remain open.
The [string serializer/reader join](notes/systems/unity_json_strings.md) adds 38
save/load cases, 13 reader cases, two mixed inherited objects and 67 controlled
service stops. Actual UTF-16 conversion and field handlers retain the observed
null-to-empty and embedded-NUL truncation behavior. Runtime string creation and
thread-local initialization remain explicit services; compound graphs stay open.
The [one-dimensional array join](notes/systems/unity_json_arrays.md) adds 72 numeric
save/load cases, 44 reader cases, two string-array cases, two shared-array cases
and 152 service stops. Native array traversal/conversion preserves same-length
reuse and makes separate allocation requests for copied aliases. Runtime array
services remain explicit; Lists and compound graphs stay open.
The [List serializer/reader join](notes/systems/unity_json_lists.md) adds 86 normal
cases and 373 service stops. Actual backing-field discovery and traversal separate
logical count from capacity; a null List can be constructed even for a missing
member. Runtime classification and managed construction remain explicit services.
The [native List classifier](notes/systems/unity_json_list_classifier.md) replaces
the classification service in 18 direct cases and six joined copies. It compares
the exact class name and corlib image; runtime metadata and construction stay open.
The [SavedGameInfo field join](notes/systems/saved_game_info_json.md) now executes
its complete pinned three-field inventory in 21 normal cases and 528 service
stops. Native string/List copying composes over actual field names and offsets;
the [native constructor and mutation audit](notes/systems/saved_game_info_methods.md)
adds all five callers, 132 cases, 20 service stops and ten value-level JSON joins.
The [SavedGameData persistence callers](notes/systems/saved_game_data.md) add all
four native methods in 44 cases, 41 service stops and six JSON joins. Preference
storage, generic gateway internals and runtime discovery remain services.
The [native generic FromJson wrapper](notes/systems/saved_game_generic_json.md)
now executes inside Load in 48 cases, 64 service stops and four field-reader
joins. Type/context/class/cast helpers and the non-generic gateway remain services.
The [native preference wrappers](notes/systems/saved_game_preferences.md) add
52 cases, 28 service stops and six JSON joins inside those callers, retaining
cached lookup, empty defaults and failed-write exception ordering. Internal-call
resolution, platform storage and exception construction/throw remain services.
The [preference lookup join](notes/systems/saved_game_preference_lookup.md) now
verifies both shipped registration pairs and ten native fallback/precedence
cases. Platform storage remains open; engine entry execution follows below.
The [engine preference entries](notes/systems/unity_preferences_entries.md) now
execute native conversion and chained cleanup in 380 cases and 42 service stops.
Length-aware entry interfaces preserve embedded NULs; backend storage and other
allocator modes remain separate boundaries.
The [registry setter](notes/systems/unity_preferences_setter.md) adds 132 cases
and 41 service stops with actual signed-byte hashing and key formatting. Values
use an explicit binary count; Windows key names truncate at embedded NULs.
Registry APIs and provider acquisition remain supplied services.
The [registry getter](notes/systems/unity_preferences_getter.md) adds 398 cases
and 65 service stops for hashed/legacy lookup, type checks and data-read races.
Accepted stored values truncate at NUL; legacy string values require every
returned byte to be ASCII. Windows outcomes and provider acquisition remain services.
The [native preference provider](notes/systems/unity_preferences_provider.md)
adds 163 standalone cases, 14 entry joins and 152 service stops for path changes,
read/write acquisition and cache recovery. [Cold token discovery](notes/systems/unity_preferences_token.md)
adds 87 cases, five cache sequences and 17 stops. Actual OS services and runtime
configuration initialization remain explicit boundaries.
The [cold-provider composition](notes/systems/unity_preferences_cold_provider.md)
adds 25 provider cases, four entry joins and 99 stops in one emulator, retaining
token-cache behavior through native handle recovery. Windows outcomes remain supplied.
The [save/storage value composition](notes/systems/saved_game_storage_join.md)
adds 43 cases and 22 outer service stops, joining native save/mutation callers,
the JSON field pipeline and actual preference provider/getter/setter execution.
Registry contents and cross-emulator values remain explicit authored boundaries.

The [exported IL2CPP string constructors](notes/systems/il2cpp_string_creation.md)
execute native UTF-8 validation/conversion and managed UTF-16 construction in
720 cases and seven controlled stops. Malformed UTF-8 returns cached empty,
discarding a valid prefix. Explicit-length construction retains embedded NULs;
the C-string wrappers truncate first. GC, class globals and allocator services
remain supplied, and this runtime audit adds no Assembly-CSharp classification.
The [runtime string/storage join](notes/systems/saved_game_runtime_strings.md)
adds 24 direct constructors, 48 getters, twenty Loads, two round trips and ten
stops. Malformed binary UTF-8 takes the native fresh-save branch before JSON;
the getter's earlier NUL truncation and REG_SZ ASCII policy remain distinct.
Adapters transfer text only, preserving separate object and allocator identities.
The [tutorial-reset caller join](notes/systems/reset_tutorial_button_join.md)
adds 26 cases and ten controlled stops, following ProjectContext through the
native reset, JSON and storage pipeline. Failed persistence retains the earlier
clear; unlocked-character values and versions remain unchanged. Singleton
production and Unity Button event routing remain supplied.
The [guarded SavedGameInfo Rust replay](notes/systems/saved_game_info_replay.md)
compares 134 normal native method/JSON-caller fixtures and service-entry List
snapshots in six focused tests. It retains versions and backing slots, requires
supplied growth/allocation outcomes and rejects aggregate capacity overflow.
All 917 Rust library tests and the release build pass.
The [tutorial presentation/persistence join](notes/systems/tutorial_persistence_join.md)
adds seven complete caller methods in 474 cases and 64 exact stopped prefixes,
including native AddTutorial/Save and four storage reloads. It separates
presentation reset from persisted completion reset, retains post-save-failure
mutations and checks controller publication after a note's early return.
Unity/event/coroutine/timing effects remain explicit services.
The [tutorial registration audit](notes/systems/tutorial_event_wiring.md)
adds OnEnable/OnDisable in 228 retained cases and 100 exact stopped prefixes.
Seven event fields preserve ordered delegate tokens; generic second-cast
failure retains the new field pointer before its barrier. Combine/Remove and
casts remain supplied services; actual event dispatch remains separate.
Its guarded Rust caller compares 226 normal profiles and four cast stops in
five tests, retaining all 29 fields and exact physical/header/pointer order.
Supplied delegate outcomes do not implement CLR multicast or event dispatch.
The [tutorial close/reveal join](notes/systems/tutorial_close_reveal_join.md)
adds five new caller definitions in 445 cases and 179 exact stops. Native
hide callbacks process queues by the count of restricted records, then show
before removal. Save failure retains earlier note/completion/queue effects;
coroutine publication and explicit callback invocation do not prove readiness.
The [tutorial handler publication audit](notes/systems/tutorial_handler_publication.md)
adds four complete handlers in 55 fixtures and 42 exact stopped prefixes.
Start and kill handlers retain ordered routine captures, including reversed
poison fields. Level handling rereads gameplay after publication callbacks;
runtime services and generated routine execution remain explicit boundaries.

Its guarded Rust publication replay compares all 48 supported normal native
fixtures in five tests, preserving capture/barrier order, physical routine
identity and raw metadata/class widths. Retained unconsumed byte ranges exclude
consumed gameplay fields; future allocations and whole-state snapshots reserve
capacity before cloning. Callbacks, failures and routine scheduling reject.

The [Character tutorial generator join](notes/systems/tutorial_character_generators.md)
adds five game-owned definitions in 47 fixtures and 206 exact stopped prefixes.
Explicit native resumes retain wait bits and the same restricted queue through
callback unrestriction, close/hide, queue processing and two saves. Real elapsed
time, engine scheduling and event admission remain outside this composition.

The [CharacterInfo tutorial join](notes/systems/tutorial_character_info_join.md)
adds actual publication and generator execution in 15 fixtures and 164 stops.
The generator uses Character.acteds rather than its icon, then joins Show30,
native queue unrestriction, close/hide and the later Show80/save. Explicit
resumes and supplied services keep engine readiness outside the claim.

The [death tutorial generator join](notes/systems/tutorial_death_generators.md)
adds both native MoveNext callers in 64 fixtures and 220 stops. Summary gates
bypass null captures; Poison tests Corrupted rather than a death-reason field.
Both explicitly supplied resume orders join queued presentation and two saves,
with runtime/class/List/Unity services and real scheduling still separate.


The [complete RefreshView caller](notes/systems/character_refresh_view.md)
executes 1,645 fixtures, eight callback mutations, three retained sequences and
29 controlled stops. It verifies death-object publication before the barrier,
separate transform queries, exact Vector3 bits and disguise-icon predicates.
Unity APIs and rendering remain explicit services.
The separate guarded Rust caller replay compares 1,639 supported native
fixtures and three retained sequences in seven focused tests, preserving
physical aliases and rejecting inconsistent liveness or unsupported provenance.
The [Character field surface](notes/systems/character_fields.md) adds thirteen
complete field leaves in 146 fixtures, five barrier stops and four retained
setter/getter sequences. Raw pointers, low-DWORD enum widths and unrelated actor
bytes are checked; runtime creation and delegate invocation are not performed
by these methods.
The [Character constructor audit](notes/systems/character_constructor.md)
adds 26 fixtures, six callback probes and eleven stopped prefixes. It verifies
physical List allocation/publication, nonzero defaults, the empty saved string
and restored-stack base tailcall. Scene ordering remains separate.
The [constructor-to-initializer join](notes/systems/character_constructor_init.md)
now carries those produced Lists/defaults through actual initialization, Hidden
refresh and first yield in 148 fixtures, three reuse sequences, four callback
probes and 119 stops. Native actor-byte retention and physical List version
changes are checked; components and scheduler handoff remain supplied.
Its guarded Rust producer replay compares 153 normal profiles in five tests,
retaining complete intermediate/final Actor and physical storage/UI state.
Constructor and refresh projections remain separate; runtime traces and real
readiness are not inferred.
The [role callback publication join](notes/systems/character_role_callback.md)
adds 185 cases and 47 exact stopped prefixes, executing the freshly installed
closure through delayed-result construction and one explicit first yield.
Old and new delegates retain separate triggers; repeated invocations allocate
distinct iterators and waits. Real role effects and scheduler admission remain
supplied, and second resume/history/UI require separate composition evidence.
The [history and speech publication join](notes/systems/character_role_publication.md)
now supplies that separate evidence in 158 cases, four alias/order sequences and
196 stopped prefixes. Native history append and Day decrement precede speech
registration; explicit later speech resumes preserve text/saved-pointer ordering
and the distinct trailer override. Real coroutine readiness remains unclaimed.
The separate guarded Rust replay compares 78 supported normal fixtures/baselines
and two distinct-record chronology sequences in five tests. Physical aliases,
UTF-16 units, wrapping uses and UI order are retained; growth, mutations, failures
and unsupported resume schedules reject atomically.
The [factory and direct speech entry audit](notes/systems/character_publication_entries.md)
adds six complete callers in 112 cases, four reuse sequences, one supplied UI
callback and 25 exact stopped prefixes. It preserves native iterator captures,
float/trigger bits and immediate speech/UI order; registration alone does not
establish generator execution or readiness.
The [Character presentation helper audit](notes/systems/character_presentation_helpers.md)
adds six complete callers in 311 cases, two retained sequences and 15 stopped
prefixes. Exact byte gates, field reloads and transform bits retain partial
UI writes. CharacterView/CardHighlight bodies and Unity rendering are supplied
boundaries rather than implementations established by these callers.

The guarded Rust factory/direct speech replay compares 116 supported normal
profiles and retained sequences in seven tests. Complete Actor/iterator/UI
snapshots preserve float/trigger bits, null/self captures and physical aliases.
Supplied registration identities do not infer a void caller return; aggregate
future snapshot budgets and typed identities validate before cloning.

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

The [MouseExit caller](notes/systems/card_interaction_mouse_exit.md)
executes 58 cases and 67 full stopped prefixes. Its native hover callback,
captured highlight loop, movement and scale requests retain exact raw ABI
and reload animation IDs independently of the earlier Kill capture.

The [skin lookup callers](notes/systems/character_data_skin_lookup.md)
execute 352 cases and 151 full stopped prefixes. CheckIfSkinUnlocked uses the
first matching skin, while LoadSkin keeps scanning and stores every match.
Separate cleanup-frame probes preserve their explicit exception boundary.

The [CharacterLoc text getters](notes/systems/character_loc_text.md)
execute 134 cases and ten full stopped prefixes. They capture the locale record,
branch on the supplied emptiness result low byte, then reload its current text.
The locale search and string implementation remain explicit service boundaries.

The guarded Rust [CharacterLoc text replay](notes/systems/character_loc_text.md)
compares 102 normal native fixtures, three full retained sequences and three
resumed suffixes. Five tests verify full storage and histories, all seven
volatile integer and six XMM registers, exact callers and nullable results.

The [skin change composition](notes/systems/character_data_change_skin_join.md)
executes ChangeSkin and its actual unlock lookup together across 260 cases and
161 exact stopped prefixes. The caller stores its original skin even when
unlock lookup selects another skin with the same ID.

The guarded Rust [CharacterData constructor replay](notes/systems/character_data_constructor.md)
compares 66 normal native fixtures and three resumed recovery calls. Five tests
verify all 37 physical records, complete histories and volatile registers,
exact generic metadata, six captured list stores and the base-constructor tail.

The actual [locale search/getter join](notes/systems/character_loc_search_join.md)
compares 504 cases and 166 full stopped prefixes. Both text getters consume
the first matching captured locale record from actual FindLocaleLoc, with
one shared stack graph and explicit synthetic cleanup evidence.

The [history/type caller audit](notes/systems/character_history_entries.md) adds
four complete callers in 116 cases, two retained alias sequences, one supplied
callback and 12 exact stops. Append and last-removal version/barrier order differ;
current-info lookup on an empty List reaches an index guard. Register-as type
selection preserves native field reload and DWORD return width.
Its guarded Rust caller compares 93 normal profiles and two retained alias
sequences in six tests, preserving complete storage/Actor and barrier-time
snapshots. Declared capacity bounds all accesses, including diagnostic tail
retention; growth, invalid storage, mutation and failure remain rejected.

The [GameData lifecycle audit](notes/systems/game_data_lifecycle.md) completes
all 23 methods in that declaration. Its twelve-target extension checks 189
native cases for mode publication, initialization, state changes, catalogue
lookups and achievement callers. Service internals remain explicit boundaries.

The [base GameMode audit](notes/systems/game_mode_base_surface.md) covers all
21 declarations. [Standard lifecycle](notes/systems/game_mode_lifecycle.md)
and [progression](notes/systems/standard_mode_progression.md) complete its
24-method caller surface, including native save/score failure ordering.
The offline Rust progression replay matches all 304 applicable native fixtures;
all 747 Rust library tests passed.
[Roguelike lifecycle](notes/systems/roguelike_standard_lifecycle.md) identifies
the kill-handler Combine during teardown. The
[delegate-to-score follow-up](notes/systems/roguelike_delegate_score.md)
executes accumulated handlers through the actual multicast invocation path;
live subscription counts remain unobserved. The [progression audit](notes/systems/roguelike_standard_progression.md)
adds 1,498 native cases and a Rust replay checked against 1,138 applicable fixtures.
[Presentation helpers](notes/systems/roguelike_presentation.md),
[AdvancedMode](notes/systems/advanced_mode_surface.md),
[RoguelikeMode](notes/systems/roguelike_mode.md), and
[SavesGame](notes/systems/saves_game_surface.md) complete their declared caller
surfaces. The [mode-transition composition](notes/systems/mode_transition_composition.md)
executes concrete teardown/load/init ordering in 161 native cases; the
[village bridge](notes/systems/roguelike_village_bridge.md) checks 660 cases with
separate caller and globally selected mode identities. UI and persistence bodies
remain explicitly scoped services. [Mode-selection UI callers](notes/systems/mode_selection_ui.md)
add 70 cases, including the completed-Standard reset triggered during card refresh.
The opt-in Rust village bridge matches all 660 native fixtures.

The [starting-character sequence](notes/systems/ascension_starting_sequence.md)
executes both lazy concatenators and the stored-array helper in 965 native cases.
Its weighted Rust replay matches 964 existing-profile cases and all 46 native
choice paths, preserving repeated draws after null payloads and partial outputs.

The [Unity clock audit](notes/systems/unity_clock.md) checks 1,437 native frame
updates and 180 fixed selections. Its offline Rust projection preserves exact
snapshot ordering, partial early-return writes, float rounding and full-width
counters. JSON float parsing now preserves native timestamp bits. Timestamp
production, initialization, setters and complete PlayerLoop composition remain
explicit next boundaries.

The [clock source extension](notes/systems/unity_clock_source.md) adds 62 native
cases for QPC conversion, construction, reset and baseline initialization. It
verifies double forwarding into the updater and preserves separate caller-level
suppression. Provider/pause callbacks and remaining configuration
writers are still open.

The [clock phase join](notes/systems/unity_clock_phase.md) binds its native
caller to TimeUpdate/WaitForLastPresentationAndUpdateTime at node 2 of the
131-node default loop. Nine static relationships and native loop construction
verify the join while preserving the existing five wait-node bindings.

The [timing setter audit](notes/systems/unity_clock_setters.md) checks 72 native
cases for fixedDeltaTime, maximumDeltaTime, timeScale and captureDeltaTime,
including distinct clamp floors, rejection paths and admitted nonfinite values.

The [clock normalization audit](notes/systems/unity_clock_normalization.md)
adds 14,673 native cases for the separate normalization and reciprocal-refresh
virtual operations. [Shipped clock settings](notes/systems/unity_clock_assets.md)
recover the complete TimeManager payload and bind all four named fields to
native clock offsets. Serialized defaults do not establish runtime load order.
The [particle timing extension](notes/systems/unity_particle_timing.md) verifies
8,438 native writer invocations. Raising fixed delta leaves particle delta
unchanged; separate normalization restores its floor. The particle setter's
unordered comparison preserves supplied NaN bits.

The [mask-16 lifecycle audit](notes/systems/unity_wait_dispatch16.md) executes
the complete enclosing routine in 32 cases and checks three direct caller
sites. Its reset-like final clock writes are recovered; public lifecycle names
and phase bit 8 remain unresolved.
The [bounded phase-eight inventory](notes/systems/unity_wait_phase8_inventory.md)
records 23 verified global loads and 158 verified immediate slot branches,
including the five known dispatch instructions. No additional dispatcher is
established; computed masks, aliases and unverified candidates remain open.
The [phase-eight ownership handoff](notes/systems/unity_wait_phase8_handoff.md)
adds eighteen chained-unwind families and three pointer-backed forwarding leaves;
it does not establish their receiver aliases or phase masks.

The [clocked Reveal adapter](notes/systems/clocked_reveal.md) now derives wait
consumer/producer timestamps from explicit audited clock transitions. Seven
tests cover weighted replay, chained drains, fixed selection, counter rollover
and atomic fallback. All 676 Rust library tests passed.

The [roster composition audit](notes/systems/gameplay_roster_composition.md)
adds 113 native cases across seven methods, preserving per-caller faction order,
initial-snapshot filtering and input-list alias clearing.
[Score/resource callers](notes/systems/gameplay_score_resources.md) add 335 cases
for ordered float multiplication and mode/count dispatch.
[Iterator factories](notes/systems/gameplay_iterator_factories.md) add 64 cases
for the delayed deck intro, receiver capture and Reset failures.
[Characters lifecycle](notes/systems/characters_lifecycle.md) adds 264 cases for
singleton publication, ordered pool hiding and four-list construction.
[Card reset composition](notes/systems/card_standard_reset.md) executes the
Standard reset and a bounded native UI callback reentry in 19 cases.

The [score lifecycle replay](notes/systems/score_lifecycle.md) matches 4,703
native fixtures, including exact float bits, partial writes and the explicit
MXCSR arithmetic contract. [Character filters](notes/systems/characters_filter_tail.md)
retain distinct managed-Contains and Unity-equality query traces across 83 cases.
[Saved-roster reset](notes/systems/gameplay_roster_reset.md),
[startup callers](notes/systems/gameplay_score_startup.md),
[relic and generic rule lookup](notes/systems/gameplay_relic_rules.md), and
[Oracle-eye events](notes/systems/gameplay_oracle_eye.md) complete scoped native
evidence for all 55 top-level Gameplay declarations. This is caller coverage,
with engine and callback bodies still explicit boundaries.

The [starting-pool bridge](notes/systems/roster_starting_bridge.md) composes lazy
selection with roster removal across 2,122 native fixtures and 46 weighted paths.
The [startup composition](notes/systems/gameplay_startup_composition.md) joins
Init with saved-roster reset, preserving distinct saved/current allocations and
contrasting RestartGame across 117 native fixtures. Both have bounded Rust
replays; the complete release library suite passes 747 tests.

All 45 top-level Characters declarations now have scoped native evidence.
[Duplicate selection](notes/systems/round_duplicates.md) adds 52 cases and a
weighted Rust replay; the [candidate composition](notes/systems/round_candidate_composition.md)
executes the actual concatenator and three filters in 124 native cases.
[Rotation and highlighting](notes/systems/characters_layout_highlight.md)
have a Rust replay matched against 96 cases, while the
[generic real-role filter](notes/systems/character_role_filter.md) adds 40 cases.
[Reveal wrappers](notes/systems/characters_reveal_entries.md) add 386 cases for
diagnostic selection/format arguments, post-callback state swaps and delegate
rereads. Scoped caller coverage still leaves engine and callback bodies open.

The [candidate Rust composition](notes/systems/round_candidate_replay.md) now
compares all 124 native event/snapshot traces and the 18 weighted paths, retaining
pre-draw failures and failed-path mass. The [unique-pool follow-up](notes/systems/round_bluffs.md)
adds 108 native cases and corrects clear-before-predicate/removal failure ordering.
[CardHighlight](notes/systems/card_highlight.md) adds 76 cases for animation IDs,
Kill flags, coroutine state/Current writes and exact timing arguments, with engine
scheduling and tween effects still explicit service boundaries.

The [unique-pool Rust replay](notes/systems/round_bluffs_replay.md) compares all
108 native traces plus 18 initial and nine fallback paths. The complete release
library suite passes 747 tests. [Immediate Acted](notes/systems/acted_surface.md)
and [delayed Acted](notes/systems/acted_delayed.md) complete scoped evidence for
all nine declarations with 68 and 124 cases. [ActedVersion helpers](notes/systems/acted_version.md)
and [animation callers](notes/systems/acted_version_animation.md) add 30 and 72
cases, completing all seven declarations while retaining explicit tween and
framework services. The [pool-to-acquisition frontier](notes/systems/round_pool_acquisition_frontier.md)
records the remaining setup, identity, RNG and continuation-state joins.

The [ManageCharacters prefix audit](notes/systems/manage_pool_prefix.md) adds 44 native cases for the pool-builder handoff and first Init arguments. Builders remain supplied services; this prefix stops before Init or empty-board publication. Full startup-to-acquisition composition remains open.

The September 30 checkpoint adds [complete setup caller orchestration](notes/systems/manage_setup_caller.md), [exact unique-source composition](notes/systems/unique_source_composition.md), [card initialization and first-yield publication](notes/systems/character_initialization.md), and a bounded [pool-to-selector bridge](notes/systems/pool_ledger_bridge.md). The subsequent [actual shared pool prefix](notes/systems/manage_pool_composition.md), [pool-history ledger bridge](notes/systems/manage_pool_ledger_bridge.md), [initialization producer](notes/systems/setup_initialization_batch.md), and [setup action bridge](notes/systems/character_action_setup.md) preserve their distinct provenance requirements. The October 1 [standalone RefreshCharacter audit](notes/systems/character_refresh.md) adds 2,094 fixtures, ten callback mutations and twenty controlled stops; its bounded Rust replay compares 1,975 supported fixtures while retaining physical data/UI aliases and rejecting unsupported services. All 804 Rust library tests and the release build pass; full setup/writer/scheduler composition remains open.
