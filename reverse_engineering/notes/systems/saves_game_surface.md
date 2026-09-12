# SavesGame remaining JSON properties and character preferences

Pinned build `f530404b0f3f_807de4a83df4`. This complements the StandardMode,
maximum-ascension and RoguelikeStandard property audits. It covers the other
five JSON property pairs, UpdateCharacterPreference, the inert
GetCharacterPreference, the nested update predicate, and three Saved* constructors.

The [harness](../../scripts/audit_saves_game_surface.py) and
[report](../../reports/f530404b0f3f_807de4a83df4_saves_game_surface_audit.json)
record **151 native cases and 14 instruction assertions**. The report includes
all 16 exact Dumper method names, signatures and RVAs as supplemental metadata;
no shared typed target inventory is changed.

## JSON property behavior

| Property | PlayerPrefs key | Default object |
| --- | --- | --- |
| AdvancedMode | `SavedAdvanced` | new AdvancedMode |
| RoguelikeMode | `SavedRoguelike` | new RoguelikeMode |
| CharacterPreferences | `SAVED_CHARACTERS` | new SavedCharacters with empty prefs list |
| UnlockedSkins | `SAVED_SKINS` | new SavedSkins with empty ids list |
| UnlockedAchievements | `SAVED_ACHIEVEMENTS` | new SavedAchievements with empty ids list |

Every getter reads its key and tests that first result for null or empty.
On that branch it allocates a default object. AdvancedMode and RoguelikeMode
call the shared constructor at `0x357920`, which only forwards to the inert
Object constructor and performs no field stores. Their field defaults therefore
come from allocation zero initialization, not authored assignments in this
constructor. The Saved* paths allocate and construct a fresh list and store it
at the object's `+0x10` field. They do not return a shared cached collection.

On the nonempty branch, each getter reads PlayerPrefs **a second time** and
passes that second result to its exact FromJson<T> context. It returns the
parser's result, including null. A changed second read is not replaced with
the first value; parsing failure does not take the default-allocation branch.
Default construction does not immediately write a save.

Each setter calls compact ToJson on the supplied object, including null, and
forwards the returned string to PlayerPrefs.SetString using its own exact key.
There is no explicit PlayerPrefs.Save flush in these bodies. JSON and
PlayerPrefs are service boundaries; no real user preferences are read or written
by the harness. A serialization failure precedes the SetString call.

## Replacement rather than in-place preference update

UpdateCharacterPreference captures the CharacterData argument in a closure,
loads the SavedCharacters object using the same property logic, and invokes
List<CharacterPreference>.RemoveAll with the native nested predicate. The
predicate compares `CharacterPreference.chId` to `CharacterData.characterId`
using string equality. Null IDs can compare equal; null preference objects or
a null captured CharacterData fail when dereferenced.

All matching occurrences are removed, retaining the relative order of other
entries. The method then allocates a **new** CharacterPreference, copies the
characterId into chId, and chooses prefSkinId from currentSkin's identifier.
If currentSkin or that identifier is null, it uses the pinned empty string.
The new record is appended at the end, and the entire SavedCharacters object
is serialized and written to `SAVED_CHARACTERS`.

For example, updating ID A in `[B, A, A, C]` produces `[B, C, new A]`. The
method neither updates the first A in place nor leaves duplicate A records.
Appending exercises native inline capacity checks; the capacity-growth helper
is an explicit service. No unrelated unlock or character-preference key is
written.

GetCharacterPreference(CharacterData) at the folded `0x33ED50` address is an
inert return in this build. Its name does not imply that it loads, applies,
returns or updates any preference. Both null and non-null arguments return
without a read or write in the native fixtures.

## Partial failures and native list compaction

The harness executes List.RemoveAll at `0xB59980` and the actual predicate at
`0x393420`, rather than approximating removal as an atomic filter. A null entry
can therefore expose partial backing-array compaction. For `[A, B, null]` while
removing A, the method copies B into the first slot and then fails on null.
The backing array is `[B, B, null]`, but logical count and version remain their
old values because the final clear/count/version update was not reached.
No replacement record is appended and no save write occurs.

A null deserialized SavedCharacters object or null prefs list also fails before
persistence. A list with an empty logical count but null backing array reaches
the append branch's version increment and then fails. Serialization or SetString
service failures occur after successful removals and append; the modified local
object is not rolled back by these caller bodies. These observations describe
the state at the explicit failure boundary, not managed exception unwinding or
the eventual persistence behavior of a failing service.

## Scope

The cases cover missing/empty/present stores, changed second reads, null parser
results, JSON and write failures, all five keys and contexts, default constructor
collections, duplicate replacement, skin fallbacks, both append-capacity paths,
missing-store updates, null object/list/backing-array failures, null-ID equality,
and partial RemoveAll compaction. Successful returns verify stack restoration and
all eight nonvolatile general-purpose registers.

Native caller instructions, constructors, predicate, removal loop and inline
append run in Unicorn 2.1.4. Allocation, list construction/capacity growth,
delegate setup, string equality, array clearing, GC barriers, JSON and PlayerPrefs
are explicit gateways. This audit does not re-audit JSON internals or claim
complete persistent-storage reliability.
