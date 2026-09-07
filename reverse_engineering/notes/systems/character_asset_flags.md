# Aligned core CharacterData fields

The pinned build has 46 core CharacterData records in `sharedassets0.assets`,
path IDs 21590 through 21635. **15 have usuallyDisguised set and seven have
picking set.** Earlier all-false observations read alignment padding instead
of the second and third serialized Boolean fields. This is an audit correction,
not a changed game build.

[`audit_character_assets.py`](../../scripts/audit_character_assets.py) verifies
the asset-file and Dumper fingerprints, the exact CharacterData field order,
the external MonoScript binding to CharacterData, each variable-length prefix,
and the independent alignment of all three Boolean fields. It resolves the
following role RID to exactly one compatible managed-reference registration.
The role-registration payload remains opaque; reading that header does not
claim complete deserialization of every managed-reference object.

The runtime IL2CPP fields are adjacent at `+0x13C`, `+0x13D`, `+0x13E`.
In these serialized objects each Boolean is followed by padding to the next
four-byte boundary. For example, Drunk's flags begin at serialized `0x3E4`:
bluffable is false, usuallyDisguised is true at `0x3E8`, and picking is false
at `0x3EC`. Its role RID begins at `0x3F0` and matches the later Drunk
registration exactly. Treating the runtime adjacency as file adjacency reads
the padding after bluffable and reports both later fields as false.

The same layout independently resolves all 46 role RIDs. The report records
each field offset, object hash, role identity and the erroneous packed-byte
interpretation for comparison. It contains names, scalar metadata and pointer
identifiers, without descriptions, textures, native bytes or decompiler bodies.

## Flagged assets

The usually-disguised assets are Baa, Lilis, Mutant, Pooka, Chancellor,
Marionette, Minion, Poisoner, Puppet, Puppeteer, Shaman, Twin Minion, Witch,
Doppelganger and Drunk. Both serialized Minion-family identities are retained;
this inventory does not infer which same-purpose asset a particular script uses.

The seven picking assets are identified individually in the
[report](../../reports/f530404b0f3f_807de4a83df4_character_assets_audit.json).
The public Mutant record is type Demon (`100`), starting alignment Good (`10`),
bluffable and usually disguised, and binds managed Skinwalker. Alignment, type,
serialized flags and managed role binding are distinct fields.

## Gameplay corrections

The audit verifies native reads of runtime field `+0x13D` in public Dreamer's
truth and lying paths and Baa's Start action. Their branches were already
recovered correctly; the erroneous asset values incorrectly made them latent.

- Baa first draws an Outcast, then replaces it with a draw from flagged
  Outcasts when that priority list is nonempty. The shipped priority assets
  are Drunk and Doppelganger. Both random calls remain part of the chronology.
- Truthful Dreamer tries the other target's distinct bluff, then flagged
  current-script assets excluding both selected real identities, then its
  board-entry fallback. A board result cannot bypass a nonempty priority pool.
- Lying Dreamer keeps initial target bluffs, fills from flagged script assets
  after its exclusions, then uses board helpers only for still-missing outputs.
  Script removals preserve occurrence semantics; repeated entries cannot be
  silently collapsed to invent a helper fallback.
- Unbound Dreamer2 prefers the registered types of flagged targets when any
  exist. Its alternate clue format remains unbound.

The Rust flag table is compared against every audited record. Current public
Dreamer validation uses the deck's current script role pool and preserves its
existing conservative treatment of unknown target/board identities. Unversioned
archived Dreamer observations retain their legacy predicate. The correction
does not use hidden live-memory facts or change execution choices directly.

```powershell
python reverse_engineering/scripts/audit_character_assets.py `
  'B:\SteamLibrary\steamapps\common\Demon Bluff Playtest' `
  --dumper-root 'B:\CodexTools\DemonBluffReverseEngineering\artifacts\f530404b0f3f_807de4a83df4\il2cppdumper-v6.7.46' `
  --output reverse_engineering/reports/f530404b0f3f_807de4a83df4_character_assets_audit.json
```

Validation: 655 Rust unit tests, 34 simulations covering 426 fixtures
(1173.15 seconds), the release build, 778 Python tests, and 32 reverse-engineering
tests passed. The asset audit verified all 46 records and all three native field
reads; the coverage checker accepted 1,163 classifications and 282 evidence records.
