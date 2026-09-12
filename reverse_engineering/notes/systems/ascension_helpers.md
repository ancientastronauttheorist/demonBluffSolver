# Ascension copy and unlock helpers

Pinned build `f530404b0f3f_807de4a83df4`. The ten-target boundary contains the
reference-type ClassConv copy implementations, two Compendium selectors and
six managed JsonUtility entry points. The [audit report](../../reports/f530404b0f3f_807de4a83df4_ascension_helpers_audit.json)
records 140 native caller cases and 22 instruction/metadata checks.

## Copy mechanism

ClassConv.CreateCopy&lt;object&gt; (`0x6033B0`) forwards the object to the compact
JsonUtility.ToJson overload, then forwards that JSON and its exact generic
FromJson method context to JsonUtility.FromJson&lt;T&gt;. The private call target
`0x3B590` used by CopyArrayIntoList has the same verified control-flow shape;
both bodies are executed by the harness.

CopyArrayIntoList&lt;object&gt; (`0x602EE0`) allocates an empty result list first,
then processes array occurrences in order. Every occurrence independently
passes through the JSON copy helper, and its returned value is appended,
including null. It does not deduplicate repeated source references or cache
earlier copies. A null source array fails after the list allocation. Before
copying an entry assignable to CharacterData, it logs that asset's characterName
field. This log branch is distinct from the serializer's supported-type rules.

These are JSON copies, not a binary serializer or a general object-graph clone.
In AscensionsData.CopyData, ScriptInfo, CharactersCount and CardAdditionPerDay
use these helpers; CharacterData collections instead use shared arrays or a
new list built from the original enumerable. The latter retains asset-reference
identity. UnityPlayer's JSON serializer and its reference-resolution rules
remain outside this boundary, so no complete deep-copy guarantee follows.

## JsonUtility wrapper boundary

The compact and explicit-pretty-print ToJson overloads return the pinned empty
string for null input. Non-null ordinary managed objects proceed to the cached
internal call. UnityEngine.Object-derived inputs must also be MonoBehaviour or
ScriptableObject instances to pass the wrapper's check; other such engine
objects take the argument-error route. The compact overload passes false for
prettyPrint.

FromJson(string, Type) returns null immediately for null or empty JSON, before
validating Type. For nonempty JSON it rejects a null Type, abstract types and
types whose IsSubclassOf(UnityEngine.Object) check succeeds. The exact base
UnityEngine.Object case should not be conflated with that subclass predicate.
On the supported path it forwards JSON, null object-to-overwrite, and the
requested Type to FromJsonInternal.

The generic FromJson wrapper resolves T's runtime Type, calls that overload,
then checks/casts any non-null result to T; a failed cast throws. Both internal
wrappers cache resolved internal-call pointers. Their requests include full
parameter signatures. This audit does not assert a request-to-UnityPlayer
registration match or execute the engine serialization implementation.

The native caller harness supplies explicit JSON-service results to check
argument forwarding, ordering and returned-value preservation. Its successful
CharacterData gateway case verifies the log and caller sequence; it does not
prove that FromJson can construct a CharacterData ScriptableObject.

## Distinct lock and unlock policies

These two static Compendium methods do not read the Compendium scene object
or its page list. They resolve ProjectContext.Instance.gameData.

**GetAllUnlockedCharacters** (`0x3992E0`) scans allCharacterData and retains
each occurrence present in allCharactersAscension.unlockedCharacters. It does
not directly consult SavedGameInfo.unlockedCharactersId. The current contents
of the referenced unlock list are inputs; this is not an immutability claim.

**GetAllLockedCharacters** (`0x399010`) scans the temporary profile's four
stored starting arrays through GetAllStartingtCharacters, then excludes each
asset present in allCharactersAscension.unlockedCharacters or whose `id` is
present in saveData.save.unlockedCharactersId. It does not use the catalogue
as its candidate source and does not trigger lazy script selection.

Both methods preserve order and duplicate occurrences. They are not set
complements: they scan different source lists and read different unlock state.
The 128 membership cases vary both unlock inputs independently and confirm the
source distinction. Four null-entry cases verify short-circuit behavior: a
null starting entry already present in the baseline unlock list is skipped;
otherwise the locked selector fails when reading its ID, preserving earlier
result-list additions. The unlocked selector can retain a null catalogue entry
when the baseline list contains null.

Consequently AscensionsData.GetOnlyUnlockedCharacters removes one occurrence
per entry of this **locked starting-pool result**, not a hypothetical global
locked catalogue. GameData.GetAllUnlockedCharactersOfType uses the other
selector before type filtering. Whether callers want these differing policies
is outside the recovered behavior; the audit does not normalize them.

## Evidence scope

The [harness](../../scripts/audit_ascension_helpers.py) executes native
Compendium loops and the ClassConv caller/private-helper chain in Unicorn
2.1.4. Enumeration, membership, allocation, JSON, logging and starting-array
concatenation are explicit gateways. Metadata slots use Dumper's exact names
and are checked before execution. Native bodies and instruction bytes remain
private; the repository contains authored assertions, metadata and results.

ClassConv's fully-shared generic alternates (`0x602C60`, `0x6032F0`) are now
covered by the [follow-up audit](classconv_shared_copy.md), including their
hidden-buffer ABI and runtime boxing branches. JsonUtility is
outside the Assembly-CSharp method denominator. Engine JSON internals, other
ClassConv helpers and global reachability remain open.

All ten baseline and typed exports completed and the quality check passed.
The 46-set GDT contains 151,749 datatypes and 604 exact definitions across
956 memberships and 498 native RVAs. Read-only validation checked 2,779
parameter locations with zero program mutations. All 32 reverse-engineering
tests passed. The existing 662 Rust unit tests and release build remain the
latest solver validation; this checkpoint changes offline audit artifacts.
