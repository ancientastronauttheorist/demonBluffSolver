# Ascension selection and temporary setup

Pinned build `f530404b0f3f_807de4a83df4`. The 33-target boundary includes all
12 AscensionsData declarations, both ScriptInfo declarations, 11 GameData
consumers, and current/previous profile selection for four mode classes.
The [serialized configuration graph](ascension_asset_graph.md) supplies the
asset identities; these native methods supply runtime selection and mutation.

## Mode selectors

| Mode | Current profile | Previous profile |
| --- | --- | --- |
| AdvancedMode | GameData.advancedAscension | Same reference |
| StandardMode | standardAscensions at CurrentVillage, upper-clamped to last | CurrentVillage minus one, without upper clamp |
| RoguelikeMode | roguelikeAscensions at CurrentVillage, upper-clamped to last | CurrentVillage minus one, without upper clamp |
| RoguelikeStandard | roguelikeStandardAscensions at currentAscension, then its ascensions at currentVillage; independently upper-clamp both | Upper-clamp outer currentAscension; use currentVillage minus one without inner clamp |

Negative indices fail native array checks; these methods do not lower-clamp to
zero. Empty arrays fail as well. CurrentVillage here is the GameData static
field; RoguelikeStandard instead reads its instance fields `+0x14` and `+0x20`.
Previous selection is not defined by simply taking the previous element of
the already clamped current result. At the start of an inner group it does not
cross to the preceding group. The GameData.GetPrevAscension wrapper returns
null when static CurrentVillage is zero, otherwise dispatching virtual slot 12.

GameData.GetStandardAscensionOfId is a separate policy: an index at or above
the array length returns allCharactersAscension, rather than the final Standard
profile. Negative indices still fail. GameData.GetCurrentAscension returns the
temporary profile directly; it does not repeat mode selection.

## Temporary profile copy

GameData.SetupCurrentAscension captures its existing temporary profile. With
DebugAscension set it copies the debug profile; otherwise it calls the current
GameMode's virtual GetCurrentAscension and passes that result to CopyData.
It does not allocate a replacement AscensionsData object. A missing temporary
profile fails, and a null source fails within CopyData.

CopyData writes fields in this exact order:

| Destination field | Source treatment |
| --- | --- |
| possibleScriptsData `+0x18` | Share array reference |
| possibleScripts `+0x20` | ClassConv.CopyArrayIntoList&lt;ScriptInfo&gt;, then List.ToArray |
| unlockedCharacters `+0x28` | New List&lt;CharacterData&gt; from source enumerable |
| mustInlcude `+0x30` | Share array reference |
| alwaysInDeck `+0x38` | New List&lt;CharacterData&gt; from source enumerable |
| four starting arrays `+0x40..+0x58` | Share each reference |
| currentPickedScript `+0x60` | ClassConv.CreateCopy&lt;ScriptInfo&gt; |
| four candidate arrays `+0x68..+0x80` | Share each reference |
| characterCounts `+0x88` | List.ToArray, then CopyArrayIntoList&lt;CharactersCount&gt; |
| cardAdditions `+0x90` | List.ToArray, then CopyArrayIntoList&lt;CardAdditionPerDay&gt; |

The helper names do not establish their internal deep-copy semantics. This
audit verifies the caller, exact generic instantiations, reference sharing,
and write order. The generic copy bodies remain a separate boundary. A null
characterCounts source fails after 14 fields have been written; null
cardAdditions fails after 15. Earlier writes are not rolled back. The native
CopyData body spans three adjacent unwind chunks, not just its first entry.

## Cached script selection and RNG chronology

SetupCharactersCount actually selects currentPickedScript. If that field is
already non-null, it returns immediately and reads neither source array.
Otherwise it:

1. Requires possibleScripts to be non-null. If nonempty, draws a uniform index
   and stores that inline ScriptInfo reference, including a null entry.
2. Requires possibleScriptsData to be non-null. If nonempty, draws another
   uniform index, requires that CustomScriptData record to be non-null, and
   replaces the cache with its scriptInfo reference, which may itself be null.
3. Leaves the cache null when both arrays are empty.

Both draws occur when both arrays are nonempty, even though the custom result
replaces the inline choice. Repeated entries remain distinct occurrences.
A null custom array or selected custom record fails after any inline write.
A null scriptInfo is a successful null cache, allowing selection again on the
next request. ClearCurrentPickedScript writes null and emits its GC barrier.

GetStartingtCharactersOfType embeds this same lazy selection before inspecting
the requested type. Thus even an unsupported type can consume those draws.
For types 10, 20, 30, 100, a selected script supplies its corresponding list
through ToArray; without a selected script the original starting array is
returned directly. Unsupported types return null. A null selected-script list
fails instead of falling back. GetCharactersCount itself performs no selection:
it returns selectedScript.characterCounts or the profile's characterCounts.

The offline Rust [kernel](../../../crates/solver-core/src/bluff/ascension_script.rs)
models SetupCharactersCount's weighted cache transitions, ordered occurrence
draws and partial failures. Seven tests cover discarded draws, duplicate
occurrences, null records versus null script payloads, caching, capacity limits,
and the native corpus. It is not wired into live solver decisions and does not
claim a complete mode-to-Reveal replay.

## Materialization, accumulation, and accessors

SetupStartingCharacters does no work when both possible-source arrays are
empty. Otherwise it requires an existing selected script and materializes, in
order, mustInclude and the four faction lists into the profile's arrays.
Each ToArray result is stored before proceeding to the next field. Later null
list failures preserve all prior writes. This method does not itself choose a
script.

GetAllStartingtCharacters (with the extra `t`) concatenates the four stored
starting arrays in type order 10, 20, 30, 100. GetAllStartingCharacters and
GameData.GetAllStartingCharactersFromAscension use the lazy typed getters in
that same order, so an initially null cache can be selected on the first call.
If the chosen payload remains null, a later typed getter can select again.
Neither helper deduplicates occurrences. GameData.GetAllCharactersData instead
concatenates the four temporary-profile candidate arrays; unlike the separate
Gameplay.GetAllAscensionCharacters bug, it includes the Demon array once.

GameData.GetCharactersOfType reads one of those candidate arrays. Its
GetAllCharactersOfType filters allCharacterData instead, while
GetAllUnlockedCharactersOfType filters Compendium.GetAllUnlockedCharacters.
AscensionsData.GetOnlyUnlockedCharacters mutates the supplied list by removing
one matching occurrence for each entry returned by Compendium.GetAllLockedCharacters;
it returns that same list. The lock-policy helper is a separate boundary.

UpdateScriptCharactersFromPreviousLevels traverses every Roguelike Standard
group and profile in order with one shared accumulating list. It appends each
profile's unlockedCharacters before calling AddCharactersToScript for that
profile. The accumulation does not restart between outer groups.
AddCharactersToScript modifies **possibleScripts[0]**, not currentPickedScript
or a randomly chosen custom script. For each input CharacterData it selects the
matching faction list using serialized type 100, 30, 20, or 10, and adds the
asset only if that list does not already contain it. Unsupported types are
ignored. A matching type requires that first inline ScriptInfo and its list;
there is no empty-array fallback.

AscensionsData.UpdateCharCounts sums town, demon, outs, and minion into
allCharCount for each entry in its own characterCounts list. ScriptInfo's
method does the same for its characterCounts list. Neither includes the four
disguised-count fields in that sum. Constructors allocate the declared list
fields before the base constructor; they do not initialize every array field.

## Evidence and limits

The [native auditor](../../scripts/audit_ascension_setup.py) executes 412 cases
covering mode indices, script caching, ordered draws, partial failures,
materialization and the actual temporary-setup-to-CopyData chain. Native code
runs only inside Unicorn 2.1.4. Metadata/class state is initialized explicitly.
RNG, GC barriers, allocator, collection conversion/constructor, and ClassConv
services are authored gateways. The report records those boundaries; it does
not claim native execution of their internals or a recovered Unity RNG state.
Other accessors and accumulation loops above are static native control-flow
findings, not additional emulated cases. Mode initialization/UI selection,
generic copy internals, lock policy and the complete generation lifecycle remain
open.

Validation: all 662 Rust unit tests and the release build passed. The new
module is an offline reconstruction kernel; no live decision path changed.

Both baseline and typed exports completed all 33 targets. The 45-set typed
union validates 946 memberships, 594 exact definitions, 488 native RVAs and
2,756 parameter locations with zero read-only mutations. Its GDT contains
151,736 datatypes. The body-free quality check passed.
