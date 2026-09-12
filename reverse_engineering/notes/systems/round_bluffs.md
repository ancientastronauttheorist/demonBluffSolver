# Round unique-bluff pool and captured membership predicate

Pinned build `f530404b0f3f_807de4a83df4`. The audit executes the complete native
`Characters.PickRoundBluffs` (36D3A0) and, separately, its generated predicate
(377170). It contains 108 passing cases and 204 executed instruction addresses.
GameAssembly/script hashes, both exact signatures, complete decode ranges and
eight instruction relationships are asserted. The three filter bodies themselves
remain supplied services here; their direct native evidence is in the companion
`round_candidate_composition.md` audit.

## Ordering correction

The actual caller allocates its closure and invokes its base constructor, ensures
Gameplay initialization, calls GetAscensionAllStartingCharacters37C3F0, rereads the Gameplay
singleton, and gets the current script through 37DC00. It publishes that script
reference at closure+10 and performs the barrier. It then clears UniquePool (+40):
version increments and logical count becomes zero before the optional array-clear
service. **Only after clearing does it allocate/construct the predicate and call
RemoveAll on the all-character list.** Earlier acquisition prose placing RemoveAll
before clear must be corrected.

Failure at either getter leaves the old pool intact. Capture-barrier failure has
already published the script but still precedes clear. Predicate allocation,
predicate construction or RemoveAll failure leaves the pool cleared. Even a null
all-character getter result is checked only after current-script capture, clear,
and predicate construction. Native failure fixtures match the complete successful
attempt/state prefix at every baseline service occurrence.

The closure predicate loads captured+10, rejects a null captured list, and tail
forwards candidate identity to List<CharacterData>.Contains with the exact method
context. It does not dereference the candidate or perform Unity object equality.
Separate fixtures cover a null candidate, null capture, Contains failure, and
return-register values with noncanonical high bits; AL is preserved. Contains
comparison semantics are supplied, not inferred from this caller.

Within the enclosing fixtures, RemoveAll is an explicit first-reference-membership
filter that commits after all supplied membership outcomes. This is not execution
of the predicate inside native RemoveAll and does not establish collection-internal
rollback. A null captured list with an empty input requires no predicate call under
that service contract and can therefore continue; nonempty input fails when that
predicate is required. Physical null occurrences can be removed through supplied
null membership before the later native-equivalent filter service sees them.

## Initial selection and fallback

After RemoveAll, the caller obtains bluffable candidates and separate exact real
Villager10 and Outcast20 lists. The Villager loop checks emptiness before drawing,
appends at most four sampled occurrences, and removes one first-equal occurrence
from the local candidate list after each successful Add. It then appends at most
one sampled Outcast and removes that occurrence. All Adds are explicit before-effect
services in this caller; a failing Add leaves the prior prefix, while a failing
Remove retains the preceding append.

If UniquePool count is **zero or one**, the native caller performs one fallback:
GetAllAscensionCharacters37C1A0, Bluffable36A550, Good Alignment369EB0, and exact
real-Villager FilterRealCharacterType36B9C0. It calls Range/get_Item without checking
that fallback list is nonempty, then appends the chosen reference. It does not loop
until size two, remove the fallback occurrence, or exclude current-script identities
from that fallback. Thus an initially empty pool ends with one successful fallback
entry; a one-entry pool ends with two. Empty fallback produces a zero-width range
attempt and indexed-access bounds failure, retaining any earlier selected prefix.

The `[0,0,1]` Villager and `[2,2,5]` Outcast inputs enumerate all18 occurrence paths
with widths3,2,1,3 and probability1/18. Independent first-equal removal checks the
ordered results. Nine additional fallback paths cover initial zero, one Villager,
and one Outcast, each against fallback `[1,1,7]` with probability1/3. ID1 is also in
the supplied current script, proving fallback eligibility is not its exclusion
predicate. The fallback candidate list remains unchanged after sampling. Six
Villagers verify the four-selection cap; all matching script occurrences are
removed before the supplied candidate filters.

Controlled singleton replacement after the first getter fails at the second
getter's reread before clearing. Replacement after the script getter permits
normal selection to finish when count exceeds one; it fails only if fallback needs
the singleton again. Replacement after fallback's getter does not invalidate its
already returned list. These are authored preserving-service probes, not a live
race claim.

## Boundaries and reproduction

Allocator/base/predicate construction, class initialization, getters, filters,
Contains/RemoveAll, RNG/get_Item, Add/Remove, array clear and GC are explicit
services. List capacity is sufficient; identities and occurrences are synthetic.
The native caller and standalone predicate execute actual instructions, while
filter/list failures stop at their declared boundaries. No full sampler/filter
composition, List comparer/rollback, Unity PRNG, managed unwinding or live state is
claimed. Metadata is warmed. Successful returns validate the stack and all eight
Windows nonvolatile integer registers.

Run `audit_round_bluffs.py GAME_ROOT DUMPER_ROOT --output REPORT` with private
Unicorn2.1.4 on PYTHONPATH. Python compilation and scoped diff checks pass. No
proprietary native bytes/decompiler body, Rust changes or shared inventory edits
are included.
