# Gameplay roster composition

Pinned build `f530404b0f3f_807de4a83df4`; GameAssembly SHA-256
`F530404B0F3F28479A7CE21D5738C4E36C2A0A03E1B5520092975B4150D819EC`.

The supplemental audit executes all seven selected Gameplay declarations:

| Method | RVA |
| --- | --- |
| GetAllCurrentCharacters | 0x37C320 |
| CleanupCharactersList | 0x37B750 |
| FilterIfCanAppearCharacters | 0x37BE30 |
| GetNotInPlayCharacters | 0x37CC00 |
| GetNotInDeckCharacters | 0x37C8E0 |
| GetScriptCharactersOfAlignment | 0x37D910 |
| UpdateCurrentCharacters | 0x381260 |

`audit_gameplay_roster_composition.py` passes **113 native cases**, eight
instruction assertions, and 1,048 distinct executed instructions. Its report
contains exact Dumper method metadata, authored asset IDs, input/output lists,
partial allocations, versions, removal/addition order, typed-pool request order,
and RNG widths/indices. Seven private read-only Ghidra exports completed with
the exact requested entries. No typed manifest, shared guide, or Rust source
was changed for this audit.

## Aggregation and exclusion

The four persistent CharacterData rosters are currentTownsfolks +0x28,
currentOutsiders +0x30, currentMinions +0x38, and currentDemons +0x40.
GetAllCurrentCharacters allocates a new list and appends all four in that
order. It preserves duplicate and null occurrences under the collection
service contract. A later append failure leaves the earlier local prefix.

GetNotInPlayCharacters embeds that same ordered aggregation, initializes
Gameplay if needed, then enumerates static Gameplay.CurrentCharacters
(+0x18). For every physical Character it passes Character.data (+0x50) to
List.Remove on the aggregate. A null physical Character fails; a nonnull
Character with null data forwards null to the removal service. Each successful
Remove removes the first matching occurrence in the fixture comparator. Extra
occurrences can remain after one in-play reference has been processed.

GetNotInDeckCharacters instead obtains typed starting arrays from the current
temporary ascension in this order: **Demon 100, Outcast 20, Minion 30, Villager
10**. It resolves ProjectContext.GameData.currentTemporaryAscension again
before each getter, then appends immediately. It subsequently performs the same
CurrentCharacters/data removal loop. It does not read the receiver's roster
fields or roguelikeDeck; a null receiver works when the required global graph
is present. A missing global or later getter failure retains the completed
local prefix. The typed getter's lazy-selection internals are the existing
ascension audit service, so this order must not be replaced by 10,20,30,100
when composing RNG/cache chronology.

Both exclusion methods require the CurrentCharacters list even when the
aggregate is empty. Class-initializer failure occurs after aggregate creation;
GetNotInDeck has already requested/appended all four starting pools then.
Local prefixes have not been successfully returned when these methods fail.

## Alignment and roster replacement

GetScriptCharactersOfAlignment creates the same four-roster aggregate and a
separate output list. It enumerates every source occurrence, rejects null
CharacterData, and appends those whose **startingAlignment +0x134** equals the
requested integer. There is no faction conversion or deduplication; unsupported
alignment integers are ordinary equality comparisons. A later null or add
failure retains prior matching output.

UpdateCurrentCharacters first clears the persistent lists in reverse order:
Demon, Minion, Outcast, Villager. Native code increments each list version and
sets its size to zero before calling Array.Clear on a previously nonempty
backing range. Empty lists still have their versions incremented. A null later
list preserves earlier clears; Array.Clear failure occurs after that current
list's version/size changes, with backing cleanup delegated to the service.

Only after all four clears does it require and enumerate randomCharacters.
A null input therefore fails with all four lists cleared. An input aliasing any
one destination has become empty before its enumerator is created and produces
no additions. For each nonnull CharacterData, it checks type +0x130 against
100,30,20,10 and appends to the corresponding persistent list. Duplicate inputs
are preserved; unsupported types are ignored. A null entry or append failure
retains previous additions. The type checks are separate native reads; these
fixtures assume collection/GC services do not mutate CharacterData between
checks.

## canAppearIf filtering

FilterIfCanAppearCharacters(type,inData,outData) enumerates **inData**, not
outData. Every inData entry must be nonnull; entries of another type are skipped.
For a same-type entry it requires CharacterData.canAppearIf (+0x128), enumerates
all dependencies, and tests each with inData.Contains. Any present dependency
satisfies the condition; the loop does not stop at the first successful test.
An empty dependency list imposes no restriction.

When a nonempty canAppearIf list has no member in inData, the method removes
that entry once from the matching persistent roster and then once from outData.
For an unsupported requested type that matches an entry's type, it skips the
persistent-roster removal but still removes from outData. It returns the same
outData object. Thus it does **not** validate every candidate in outData: a
candidate absent from inData is never examined, even if its dependencies are
absent. Duplicate inData entries can cause repeated single removals.

Null inData fails immediately. Null outData can be returned successfully when
no entry requires removal; when removal is required, the persistent roster can
already have changed before the null outData check. A removal-service failure
at either position preserves its completed prefix. A null same-type dependency
list fails before those removals.

When outData aliases inData, a successful removal changes the active input
list's version. The next native MoveNext rejects that version mismatch. The
fixture executes actual enumerator creation and MoveNext, including this
failure gateway; it does not simulate the entire managed unwinder/finally path.

## Cleanup refill chronology

CleanupCharactersList allocates an unused empty list, obtains
GetAllCurrentCharacters, and clones that aggregate into a single input snapshot.
It reuses that **unchanged initial snapshot** for all four filter calls. The
phase order is **Outcast 20, Minion 30, Demon 100, Villager 10**.

For each phase it resolves the current temporary ascension, calls the typed
starting getter, copies the returned array into a candidate list, then runs the
actual native FilterIfCanAppearCharacters with the shared snapshot. It reads
Gameplay.CurrentScript and computes the signed maximum of the faction's normal
and disguised count, subtracting the persistent roster count after filtering.
CharactersCount normal fields are town +0x14, demon +0x18, outs +0x1C, minion
+0x20; disguised counterparts are +0x24,+0x28,+0x2C,+0x30. There is no sum of
normal and disguised targets. Integer subtraction follows native 32-bit
arithmetic.

For a positive deficit, while candidates remain, it draws an index in
[0,candidate.Count), gets that occurrence, appends it to the persistent roster,
and removes one matching occurrence from candidates. It stops at the initial
deficit or pool exhaustion. Even a zero deficit still reaches that phase's
getter, candidate constructor, and filter. Null candidate entries can be
selected and appended. Duplicate candidates retain occurrence-dependent draw
widths; selecting the later duplicate still removes the first equal occurrence
under the fixture comparator.

The snapshot is not recomputed after filtering or refill. A Minion whose
canAppearIf contains an Outcast removed earlier in the call can remain allowed
because that Outcast is still in the initial snapshot. Newly appended roles do
not enter that snapshot either. Tests exercise this case, exhausted candidate
pools, ordinary/disguised maxima, signed negative counts and wrapped deficits, duplicate picks,
null candidates, and failures after prior faction mutations.

## Runtime service limits

The seven managed bodies execute privately in Unicorn 2.1.4. Cleanup calls
actual native aggregation and filtering. Native generic GetEnumerator,
enumerator construction, MoveNext, version checks, folded Dispose, and the
capacity-sufficient List.Add body also execute. Fixture capacities avoid Add's
resize path; allocation/resizing internals are not inferred.

Allocation, collection constructors, AddRange, Contains, Remove, indexed
access, Array.Clear, starting typed getters, and RNG are explicit services.
Contains/Remove use reference identity in these authored fixtures, with Remove
operating on the first match. This audit establishes which values and order the
caller supplies; it does not close the runtime EqualityComparer or Unity
Object destroyed-object/null equivalence rules. RNG receives supplied valid
indices; no Unity random state is recovered. Inputs do not mutate through
collection callbacks except for the documented destination operations.

Injected service failures stop at their entry. Native pre-call writes and
completed earlier mutations are observed, but generic exception unwinding and
finally execution after those injected failures are outside scope. The audit
contains original assertions/results only; PE bytes and decompiled bodies
remain private. JSON/serializer internals remain their existing independent
service boundaries. No Rust or live solver path changed.

```powershell
$env:PYTHONPATH='B:\CodexTools\DemonBluffReverseEngineering\python-emulation'
python reverse_engineering/scripts/audit_gameplay_roster_composition.py --game-root 'B:\SteamLibrary\steamapps\common\Demon Bluff Playtest' --dumper-root 'B:\CodexTools\DemonBluffReverseEngineering\artifacts\f530404b0f3f_807de4a83df4\il2cppdumper-v6.7.46' --output reverse_engineering/reports/f530404b0f3f_807de4a83df4_gameplay_roster_composition_audit.json
```

The auditor passes and `python -m py_compile` succeeds.

## Offline logical-list replay

`crates/solver-core/src/bluff/roster.rs` adds opt-in
`gameplay_roster_native_v1` replay for these seven operations. Explicit list IDs
preserve input/output/roster aliasing. Each list supplies logical contents,
version, and capacity. Assets supply exact type, alignment, and canAppearIf
list identities; null references are concrete nulls. Current Characters and
starting-pool results are separate typed inputs.

The context requires reference equality, input-preserving callbacks, and
sufficient native Add capacity. Its service contract matches the native
fixtures: allocation, enumeration/disposal, Contains, and indexed access
return normally; declared injection points can fail range append, Remove, Add,
Array.Clear, typed-pool requests, or class initialization. Input preservation
excludes each explicit destination mutation, including a destination that
aliases an enumerated input. These are caller-established offline preconditions,
not facts inferred about arbitrary Unity objects or callbacks.

Outputs retain all list identities, logical contents and versions, allocated
local lists, returned list identity, additions/removals, typed-pool chronology,
and failures. Cleanup enumerates occurrence-sensitive index paths with reduced
unconditional rational probability. Failed paths retain their mass and completed
mutations; no successful subset is renormalized. Provenance/capacity/support
failure rejects the entire request without mutating the input context.

This model exposes GetNotInDeck's 100,20,30,10 typed-pool request order and
Cleanup's 20,30,100,10 order. It does not change or extend ascension_starting's
lazy-selection composition: the roster audit supplied those getters as services,
so their nested RNG/cache effects remain a separate composition boundary.

State is intentionally logical list state. A failing Array.Clear has already
changed native size/version, which the model preserves, but stale inaccessible
backing slots and native unwinding/finally are not modeled. Source identities
and stable service semantics must be established before using this offline
result. No live solver decision path consumes this module.

Rust validation: the parent broad library run passed all 712 tests after this
module was integrated. Seven roster tests cover all 113 native fixtures, six
weighted duplicate-pick paths, preserved failed mass, identity aliasing,
logical size/version changes, and strict provenance/support/capacity rejection.

The final 726-test release library run includes the roster capacity regression:
input list entries are counted once, including a 526,336-entry accepted context.
