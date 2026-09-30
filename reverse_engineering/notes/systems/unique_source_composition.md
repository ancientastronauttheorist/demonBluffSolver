# Unique pool through exact native sources

Pinned build `f530404b0f3f_807de4a83df4`. The authored
`audit_unique_source_composition.py` executes nine actual methods in one native
fixture: `PickRoundBluffs`36D3A0, the capture predicate377170,
`GetAscensionAllStartingCharacters`37C3F0,
`GetAllAscensionCharacters`37C1A0, `GetScriptCharacters`37DC00,
`GetStartingtCharactersOfType`3B1E10, and the Bluffable36A550,
RealType36B9C0 and Alignment369EB0 filters. The report records 516 cases and
671 executed instruction addresses. GameAssembly, script.json and dump.cs hashes,
nine exact signatures and entry ends, twelve instruction assertions and exact
class field declarations are checked before execution.

The starting source calls the actual lazy typed getter in Demon100,
Outcast20, Minion30, Villager10 order. It appends each returned array before
rereading `ProjectContext.Instance.gameData.currentTemporaryAscension` for the
next faction. Four independent profile replacements after these appends verify
that a later request can use another profile. Null project, GameData or profile
after appends one through three stops at the next graph read; the same change
after append four leaves this getter complete. The replacement does not rewrite
already-appended occurrences. Direct arrays, preselected ScriptInfo lists and
inline lazy selection execute through the actual typed body. `ToArray` uses a
distinct supplied snapshot. Every native cache store remains observable before
its GC service can fail.

The fallback independently allocates its list and concatenates the profile's
arrays at +68, +70, +78 and **+68 again**: Villagers10, Outcasts20, Minions30,
Villagers10. It rereads the same graph between appends. The final Villager array
can therefore come from a replacement profile rather than the first one.
Demons100 are omitted. In the fixture, `[1,7]`, `[5]`, `[3]`, `[6]` produces
`[1,7,5,3,1,7]`; actual Bluffable, Good and real-Villager filters reduce this to
`[1,7,1,7]`. Four occurrence indices remain equally likely. Script membership
is not subtracted again, so ID1 can be returned even when present in the script.

The script concatenator executes against current roster fields +28/+30/+38/+40.
Its completion and capture/barrier precede unique-pool clear. The RemoveAll
service then invokes the actual native capture predicate once per starting
occurrence, including repeated references and nulls. The predicate forwards the
captured list and candidate to the supplied reference-membership Contains
service. Under this fixture's explicit deferred-commit contract, all accepted
predicate results are collected before removing script members. A failing
Contains preserves the full starting list; this is a service contract, not a
claim about the managed List implementation's compaction or exception state.

Actual filters allocate before enumeration, preserve occurrence order, reject
physical null data and append selected references one at a time. Their real
type and starting alignment tests remain distinct. The ordinary unique path
filters bluffability and real type, with no Good filter; fallback additionally
filters Good. Sampling draws up to four Villagers and one Outcast, removes the
first equal local occurrence after each append, and falls back when unique
count is at most one. An empty fallback still attempts zero-width Range and
then fails at indexed access. No failure is converted to successful empty output.

The direct mixed fixture exhausts all eighteen pool paths. The two inline
script fixture exhausts nineteen combined source/pool paths; the nullable
inline fixture exhausts twenty-six, retaining every consumed null draw before
later faction reselection. Each family's unconditional probability sums to
one. Each reached service occurrence in direct, fallback, cached and inline
successful baselines is failed separately and must preserve its exact event
and state prefix. These cases include source allocation/constructor, cache
barrier, ToArray, AddRange, predicate, filter/enumerator, pool append and removal
failures. Null direct or roster collections fail before unique clear; null
fallback collections fail after it. Singleton disappearance after starting
prevents script acquisition, whereas disappearance after script only matters
when the later fallback is reached.

The offline `bluff/unique_source_composition.rs` replay follows this same bounded
contract, with source profile identities, cache state, all intermediate logical
collections, per-service snapshots and one combined RNG chronology. It requires
explicit metadata, distinct-capacity, stable-reference-service, deferred-commit
and uniform-occurrence flags. Weighted replay bounds combined retained trace
state and path count, retaining failure mass. CustomScriptData selection is
explicitly empty here; its general behavior remains in the separate starting
audit and is not silently inferred into this composition.

Allocation, List construction/AddRange/Add/Remove, stable enumerators/disposal,
Contains, ToArray, GC, class initialization and supplied uniform RNG indices
remain services. Source arrays have logical collection views with inert version
zero in the report. No List backing-slot retention, comparer internals, managed
unwind recovery, general callback/reentrancy, Unity RNG state or live setup
transaction is claimed. In particular, this join does not yet execute
ManageCharacters, duplicate pool construction, ordered Start writers or delayed
Reveal in the same transaction. It supplies a stronger unique-source boundary
for those future joins, rather than a completed solver or live bridge.

Run with the private Unicorn2.1.4 dependency path:
`audit_unique_source_composition.py GAME_ROOT DUMPER_ROOT --output REPORT`.
The native report contains authored inputs, observations and signatures, not
proprietary instruction bytes or decompiler bodies.
