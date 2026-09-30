# SavedGameData native persistence callers

All four methods of pinned `SavedGameData` now execute offline: `Load`, `Save`,
`ResetTutorials` and the constructor. The audit verifies the complete declared
field inventory (`SavedGameInfo save`, offset `0x18`), exact Dumper signatures,
three metadata slots, 28 instruction assertions and all three chained unwind
chunks of Load. It passes 44 cases, 41 controlled service stops and six native
JSON joins. SavedGameInfo construction and String.IsNullOrEmpty execute natively.

Load reads the current save's key and tests the first preference value. Null or
empty creates a new SavedGameInfo through its actual constructor, then publishes
it. A nonempty first value causes a second preference read, using the current
save object's key again. Only the second value reaches FromJson. It is not
checked for emptiness; the returned object, including null, is stored before the
write barrier. Callbacks can change the key/object between the two reads. A null
save after the first read reaches the native null-reference gateway.

Save serializes the current save object first, even when null. It rereads the
save reference afterward, requires it to be nonnull, and takes the preference
key from that current object. A supplied callback can therefore serialize one
object and persist under another object's key. ResetTutorials increments the
tutorial List version and sets size to zero before optional array clearing,
then follows the same serialization/key-reread path. Empty Lists still change
version; the unlocked-character List remains intact. Neither caller invokes
PlayerPrefs.Save in its audited body.

The constructor allocates and runs native SavedGameInfo construction, publishes
the new reference, calls its write barrier and then requests ScriptableObject
base construction. Warm/cold cases cover existing/null saves, null/empty Lists,
different first/second preference values, null JSON results and callbacks that
replace or clear the save reference. Normal returns retain the stack and all
eight nonvolatile registers. Controlled service stops retain exact snapshots
and complete event prefixes, including mutation before serialization failure.
They do not simulate native managed exception unwinding.

Preference getters/setters, the generic JSON gateway, ScriptableObject base
construction, allocation and remaining runtime helpers are explicit services.
Six joins transfer the exact three public values into actual native engine JSON
writing or reading. Read joins use an authored zero-initialized destination;
they do not establish runtime generic construction behavior. Object identities
and private List versions are not transferred between emulators. No live game
preferences are accessed or written.

```powershell
python reverse_engineering/scripts/audit_saved_game_data.py GAME_ROOT DUMPER_ROOT --output REPORT
```

Report: `f530404b0f3f_807de4a83df4_saved_game_data.json`. Private Unicorn 2.1.4
dependencies are required. Shared preparation/invocation helpers were extracted
from the SavedGameInfo audit; its complete report reruns identically.
