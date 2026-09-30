# ManageCharacters through both actual pool builders

Pinned build `f530404b0f3f_807de4a83df4`. This composition executes the actual
ManageCharacters36CE30 prefix, both actual pool builders and their actual
source/filter methods in one native transaction. It stops immediately before
the first Character.Init365A20 call, or before empty-board publication at
36D01E. It does not execute Init, later Act passes, ordered Start, onSetup or
ShuffleDeck. The earlier complete-caller replay retains those boundaries.

The authored `audit_manage_pool_composition.py` report contains 760 cases and
878 executed instruction addresses. It pins GameAssembly, script.json and
dump.cs hashes; validates eleven exact method signatures and entry ends, two
gateway signatures, twenty instruction assertions and exact class fields.
The verified ManageCharacters entry ends at36D396 before alignment padding.
UpdateCharacterPositions36E4E0 is a supplied service. After it returns, the
native caller directly invokes PickRoundBluffs36D3A0, then
PickRoundDuplicates36D720. The builders themselves are not supplied gateways.

Unique construction executes the starting getter37C3F0 and actual lazy typed
getter3B1E10, preserving100/20/30/10 order and every profile graph reread/cache
store. It executes the script concatenator37DC00, native capture predicate377170
and Bluffable/RealType/Alignment filters. Its fallback37C1A0 preserves
10/20/30/10 again, including duplicate Townsfolk source weight. This is the same
source boundary established by the separate unique-source audit.

After unique construction returns, duplicate construction executes a second
GetScriptCharacters call and allocates a distinct snapshot. It clears only its
own pool and executes Bluffable, real-Villager, real-Outcast and discarded-Good
filters. The discarded result still contributes allocation, enumeration,
append and failure events. The two returned script lists and all filter lists
remain distinct; shared CharacterData references and repeated occurrences are
retained. Supplied current roster providers may alias each other. A fixture
with two roster fields pointing to one `[1,7]` provider appends `[1,7]` twice
and preserves all twenty-four duplicate occurrence paths.

The two pool identities and versions are independent. Ordinary unique and
Outcast duplicate Add services can fail before appending. A duplicate Villager
append instead publishes size, version and reference in native instructions
before invoking its GC barrier; a failure there retains the new occurrence
and prevents subsequent local removal. An empty duplicate Villager candidate
list still attempts Range with width zero and then fails at indexed access.
Stable append services make its later empty-pool fallback unreachable; that
unreached branch is not inferred into the bounded replay.

Profile cache and RNG history are shared across the whole prefix. Mixed direct
sources have eight combined occurrence paths. Fallback has four; aliased
rosters have twenty-four. Two inline script choices and nullable inline
reselection are also enumerated together with both builders' later draws.
Each complete support family's unconditional probability sums to one.
Unique failure stops before the duplicate builder; duplicate failure preserves
the already-completed unique pool. There is no renormalization or restarting
an earlier source selection.

Controlled AddRange callbacks demonstrate temporal separation between the
script copies. Mutating current roster0 to `[7,7]` after unique's script copy
leaves its captured `[1,7,5,3,6]` unchanged, while duplicate's later copy becomes
`[7,7,5,3,6]`. Replacing that provider with null fails the later concatenation
before duplicate-pool clear, retaining the old pool. Replacing its contents
with a physical-null data occurrence completes concatenation and clears the
pool, then fails inside actual Bluffable enumeration. A disappeared Gameplay
singleton after unique script capture can still finish unique sampling, but
prevents duplicate script acquisition. Disappearance after duplicate's final
script append permits both builders to finish. Snapshot state includes the
current roster source identities, so these writes remain visible.

Class state is also shared. A cold Gameplay class initializes during unique
construction; the duplicate builder sees the resulting initialized state.
System.Math initialization is reached only afterward on the nonempty board
path. A null board fails after both builders; an empty board disposes its
enumerator and stops before publication. The caller rereads the current board
for the first displayed ID after MoveNext. Thus a supplied enumerator can retain
the original first actor while a controlled board replacement changes the
count used for that actor's ID. Roster indexing happens before the actor null
check: an empty input roster reports bounds even when the yielded actor is
null. A null CharacterData value itself reaches the Init boundary unchanged.

Every reached service occurrence in direct, fallback, cached and inline
baselines is independently failed. Exact event prefixes, both pool states,
cache writes, classes, source aliases and intermediate collection contents must
match the successful prefix. The offline `bluff/manage_pool_composition.rs`
replay follows the same transaction and retains one bounded support/trace
budget. Its tests compare the complete native corpus and five weighted
families; setup board arguments and callback state remain explicit inputs.

Positions/layout, collection allocation/construction, AddRange/Add/Remove,
Contains, stable enumerators/disposal, ToArray snapshots, class initialization,
GC and uniform supplied RNG indices remain services. RemoveAll invokes the
actual predicate per occurrence but commits only after all supplied membership
results return; native List compaction, backing-slot retention and exception
unwinding are not claimed. Source arrays have inert logical version-zero views.
CustomScriptData is explicitly empty in this bounded composition. General
scene callbacks, native positions/layout, arbitrary collection aliasing,
Character.Init and later setup writers, Unity RNG state and engine scheduling
remain open. This closes the actual shared pool-builder prefix, not full setup
or a live solver integration.

Run `audit_manage_pool_composition.py GAME_ROOT DUMPER_ROOT --output REPORT`
with private Unicorn2.1.4 dependencies. Frozen earlier native corpora and
replays are unchanged. No proprietary native bytes or decompiler bodies are
stored in this report.
