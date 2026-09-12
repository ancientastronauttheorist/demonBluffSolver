# Script-to-duplicate candidate composition

Pinned build `f530404b0f3f_807de4a83df4`. The new audit executes the actual
`Characters.PickRoundDuplicates` caller together with `Gameplay.GetScriptCharacters`
(37DC00), `FilterBluffableCharacters` (36A550), `FilterRealCharacterType` (36B9C0,
twice), and `FilterAlignmentCharacters` (369EB0). The report contains 124 native
cases, including all 18 weighted occurrence paths and 330 executed instruction
addresses. GameAssembly, script.json and dump.cs hashes, five exact signatures,
complete entry ends, three predicate operands and field declarations are checked.

The concatenator allocates one distinct script list and appends current roster
fields +28, +30, +38, +40 in that order. Its completion precedes clearing the
old duplicate pool. Each filter then allocates a distinct list before testing
its input, enumerates in occurrence order, rejects physical null elements, and
preserves duplicates. Bluffable tests byte +13C for nonzero (including 255);
real type compares the complete signed-width field +130 for equality; alignment
compares field +134. The composition calls real-type 10 and 20 against the same
bluffable list, then alignment 10 against that list. It discards only the last
returned reference: the Good filter's allocations, reads and possible failure
still happen before the first RNG request.

The mixed fixture concatenates `[0,0,1,4]`, `[2,2,5]`, `[3]`, `[6]`. ID4 is
not bluffable; the resulting bluffable list is `[0,0,1,2,2,5,3,6]`. Real-type
Villagers are `[0,0,1]`; Outcasts are `[2,2,5]`. The discarded Good result is
`[1,5,3]`. IDs0 and2 have starting-Evil alignment and remain fully eligible for
the duplicate pool. ID6's bluffable byte255 is accepted, but its Demon type
excludes it from both sampled candidate lists. Good Minion ID3 likewise does
not become a Villager or Outcast.

The 18 consumed paths have widths3,2,1,3 and probability1/18 each. An independent
occurrence model checks ordered output and first-equal removal from both local
candidate lists. A six-Villager input verifies the four-draw cap. Empty candidates,
including a nonempty script with no eligible Villager, reach the unguarded
zero-width RNG/indexed-access bounds failure before any Outcast draw or fallback.

Every reached service occurrence in a successful cold-class baseline is then
failed independently. Attempt events and complete modeled list snapshots must
match that successful prefix exactly. In particular, a null roster collection
fails script concatenation while preserving the old duplicate pool. A null data
occurrence anywhere in the concatenated script reaches the native Bluffable
null check after the pool was cleared; earlier filter additions remain, and no
RNG request occurs. Failure in the discarded Good filter retains completed
Villager/Outcast candidate lists but stops sampling. Inline Villager append and
barrier failure retain the native published occurrence before any removal.

Allocation, empty-list construction, stable enumeration/disposal, AddRange,
filter/Outcast Add, indexed-access bounds, first-reference-equal Remove, GC,
class initialization and supplied RNG indices remain explicit services. Allocated
list identities are distinct, capacity is sufficient, and services preserve all
inputs except their declared destination mutations. Metadata is warmed. The
fixture tracks logical list contents and versions; it does not reconstruct
backing-slot retention, collection comparer internals, exception unwinding,
Unity PRNG state or live scene behavior. Existing duplicate replay evidence
continues to govern growth and standalone null candidate-provider results.

This closes the native candidate-source bridge needed before replacing the
three filter gateways in the offline duplicate replay. Existing Status/Revealed/
Unique filter kernels are different methods and cannot substitute for these
predicates. A future Rust composition should preserve distinct intermediate
lists, all pre-draw failures, and the discarded Good call while feeding the
proven real-type candidates into the existing weighted sampler.

Run `audit_round_candidate_composition.py GAME_ROOT DUMPER_ROOT --output REPORT`
with the private Unicorn2.1.4 PYTHONPATH. No proprietary bytes, decompiler body,
live state, Rust code or shared target/coverage changes are included.
