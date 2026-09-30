# SavedGameInfo native JSON composition

The pinned engine pipeline now composes all three public `SavedGameInfo` fields:
`key` (`string`, offset `0x10`), `completedTutorials` (`List<string>`, `0x18`) and
`unlockedCharactersId` (`List<string>`, `0x20`). GameAssembly, global metadata
and the exact Dumper output are fingerprinted; the complete field inventory must
match before fixtures execute. Runtime metadata services use those declarations.

Ten compact/pretty save/load cases, nine reader cases, two shared-List cases and
528 controlled service stops pass. Native metadata building, descriptor/registry
selection, exact List classification, string/List handlers, Unicode conversion,
JSON writing/rendering and reload all execute. Class name/image/corlib, managed
allocation and construction, array/GC/cache/allocator services remain explicit.

The public values retain ordinary text, ordering and duplicate List entries.
Unicode survives; null strings become empty strings and embedded NUL truncates.
Null Lists reload as empty Lists. Missing members preserve existing fields; on an
all-null authored destination, a missing key remains null while both missing List
fields are constructed as empty Lists. Differently cased and unknown names are
ignored. Duplicate names use the first value. Mixed JSON element types follow
the previously audited native string conversion rather than a Python conversion.

Two fields pointing to one supplied List save equal values and reload as distinct
Lists. This observes native traversal and separate construction requests, with
actual runtime object allocation still supplied. It does not establish arbitrary
graph copying. Writing preserves the complete authored object, both List/array
storage and every source UTF-16 buffer.

Every observed service occurrence is independently stopped in fresh reads,
existing-object reads and writing. Each failure must equal the complete baseline
event prefix; read failures also preserve its exact snapshot. The report stores
three baseline streams once and references their verified prefix lengths. This
captures writes already made to key/List fields before later failures, without
claiming native exception unwinding.

This is complete field composition over the pinned public inventory, not live
save-file access or full startup. The [native constructor and mutation audit](saved_game_info_methods.md)
now supplies defaults and value-level joins. GameData/PlayerPrefs callers, real
class discovery and runtime allocation remain separate boundaries. No game
preference is read or written.

```powershell
python reverse_engineering/scripts/audit_saved_game_info_json.py GAME_ROOT DUMPER_ROOT --output REPORT
```

Report: `f530404b0f3f_807de4a83df4_saved_game_info_json.json`. Private Unicorn
2.1.4 dependencies are required. No native bodies or bytes are retained. This
engine field-composition audit adds no Assembly-CSharp method classification.
