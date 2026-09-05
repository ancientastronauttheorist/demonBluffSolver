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

- Steam app: `3749680` (`Demon Bluff Playtest`)
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
  --steam-manifest 'B:\SteamLibrary\steamapps\appmanifest_3749680.acf'

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
weighted role-pair and Cabbage paths and proves that all 46 shipped core
CharacterData assets currently have `usuallyDisguised == false`. The resulting
board-entry fallback can truthfully emit both selected roles, while a selected
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
the union of every checked target set. The current archive contains 151,680
datatypes. Its forty-one-set inventory contains 882 target memberships, 543
distinct selected FunctionDefinitions, and 445 unique native RVAs. The typed
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
sets; the current forty-one-set run used six batches for each phase. Ghidra
commands still must not overlap on the saved project.

The preserved fully analyzed typed project now covers all forty-one target
sets after a no-analysis refresh. Three hundred thirty-nine memberships are
exact FunctionDefinition overlaps between boundaries. Folded/shared bodies make
the 543 selected definitions exceed the 445 unique native RVAs by ninety-eight;
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
datatype imports during application. The six-batch refresh reapplied and
validated all 882 memberships without rerunning auto-analysis. The final
read-only pass validated all 882 memberships (543 exact definitions) and 2,585
membership-level parameter-storage locations with zero program mutations.

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

## Method coverage

[`coverage/`](coverage/) contains the deterministic 4,207-method denominator,
sparse authored classifications, and reusable evidence. Missing classifications
resolve to `unresolved/not-reviewed`; shared native RVAs never collapse managed
method identities. The current overlay contains 532 classifications backed by
276 evidence records. See [`coverage/README.md`](coverage/README.md) for the
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
