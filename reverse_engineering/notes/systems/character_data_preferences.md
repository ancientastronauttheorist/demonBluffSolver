# CharacterData preference-loading caller

The pinned `f530404b0f3f_807de4a83df4` family executes actual
`CharacterData.LoadPreferences` only. Saved preference lookup, generic
enumeration, string equality and every `LoadSkin` request remain explicit whole
supplied services. No preference file, live game or engine save is changed.

Source: `reverse_engineering/scripts/audit_character_data_preferences.py`.
Report: `reverse_engineering/reports/f530404b0f3f_807de4a83df4_character_data_preferences.json`.

## Exact binding and complete body

The sole native declaration is `tdi5845.m0011`, exact
`void CharacterData__LoadPreferences(CharacterData_o*, const MethodInfo*)`,
Dumper type signature `vii`. Its exact public dump declaration and ordinal 11
within the 21-method CharacterData declaration are independently asserted.
The complete body is `[0x3B4DB0, 0x3B4F02)`, followed by the next managed entry
at `0x3B4F10`. It has 338 file-backed bytes, 79 instructions, one complete
EH/UH unwind range with flags 3 and handler `0x30CD28`, and fourteen following
`CC` alignment bytes. The final one-byte `int3` is pinned. Twenty selected
operand assertions and exact complete call-site lists verify the caller.
Repository evidence retains fingerprints and authored assertions; complete
native bytes and disassembly remain private.

Consumed fields are exact CharacterData TypeDefIndex 5845 `characterId +0x18`
and `currentSkin +0xC0`; SavedCharacters TypeDefIndex 5552 `prefs +0x10`;
CharacterPreference TypeDefIndex 5553 `chId +0x10` and `prefSkinId +0x18`.
`SavedCharacters` is separate from `SavedGameInfo`. The exact supplied getter
at `0x387870` is `SavesGame.get_CharacterPreferences`, returning SavedCharacters
with a full zero RCX MethodInfo argument.

## Ordered clear and preference scan

The caller captures its owner before metadata. A zero flag `0x288C4DD` initializes
four distinct CharacterPreference enumerator MethodInfo roots in order:
Dispose, MoveNext, get_Current, GetEnumerator. It writes flag 1 only after all
four services complete. Any nonzero flag skips them; the initialized get_Current
method is not called.

It then stores raw null to the captured owner's currentSkin and calls the
reference barrier before retrieving preferences. Earlier clear/barrier/metadata
effects survive a later failure. A null owner faults at the actual first store,
`0x3B4E0B`; a null returned SavedCharacters or null prefs List reaches the
same supplied guard at `0x3B4EFC` after that clear.

The returned SavedCharacters is consumed immediately for its current prefs
List. The hidden GetEnumerator output receives 24 bytes: list pointer, index
DWORD, version DWORD, current pointer. Its layout is pinned to the exact generic
declaration/header. Native MOVUPS/MOVSD copies it into active stack state,
clears the output/exception slot and stores the active enumerator address.
A supplied null RAX return does not replace this output storage.

Each MoveNext consumes only AL. After a nonzero byte, the caller captures its
current CharacterPreference in RDI; null reaches `0x3B4EF0`. String equality
receives that record's current chId in RCX, the captured owner's freshly
reloaded characterId in RDX and full zero R8. The owner ID is reloaded on every
iteration, so a callback can change later matching behavior.

After nonzero equality AL, it reloads prefSkinId from the already captured
preference, sets the original owner in RCX, and clears full R8 before supplied
LoadSkin at `0x3B4EAD`. A comparison callback may replace active current storage
without changing this captured recipient; changing the captured record's
prefSkinId does change the ensuing request. Nullable owner/preference IDs and
skin IDs are explicit service inputs.

The loop does not stop at the first match. Duplicate matching preference
occurrences cause repeated ordered LoadSkin requests. Inert supplied LoadSkin
does not invent a skin-loading result; authored completed callbacks can change
currentSkin or the owner's ID. Actual LoadSkin's scan and its full physical
composition require separate evidence. The frozen standalone skin caller
audit remains unchanged.

## Full state, registers and stopped effects

Thirty-one full diagnostic windows retain both owners, SavedCharacters,
Lists/arrays, three preferences, strings/skins, metadata records, exception,
aliased enumerator and a 64-byte scratch region. These are authored diagnostic
sizes rather than inferred managed object extents. Scratch covers frame
`+0x20..+0x60`; hidden output is `+0x28`, active state `+0x40`, with the
captured exception/enumerator slots retained. Unrepresented surrounding stack
bytes are not claimed as complete stack storage.

Native entries and every supplied service, including stopped entries, record
seven volatile integer registers and XMM0 through XMM5. Service events also
record full RCX/RDX/R8/R9, native caller and per-invocation ordinal. Explicit incoming seeds prevent
preceding fixtures from contributing accidental unused register bits.
Services poison volatile registers independently of their authored RAX.
Native vector copies also have their register effects independently predicted.
Every final volatile integer/XMM value is compared on returns, stops and faults.
The void method's RAX is only opaque service residue, not a preference result.
Normal returns preserve the stack, eight integer nonvolatiles and XMM6–15.

The ordered model uses only initial bytes, supplied options and incoming
registers. It predicts all events, complete storage/history/iterator state,
metadata roots/flag, dispositions and final registers before report pooling.
Only actual reached native stores and completed supplied writes grant mutation
permission. Skipped or failed callback plans have no effects. Per-invocation
service counts reset independently of the retained chronological history.

## Synthetic cleanup and verification

Separate direct probes enter cleanup at `0x3B4EC5` with an authored saved frame,
active or aliased enumerator and null/live exception. Dispose executes, then
the exception slot is reloaded: null follows the ordinary epilogue; nonnull
reaches supplied rethrow at `0x3B4EF6`. Dispose callbacks can introduce or clear
an exception. These probes do not execute managed exception dispatch, handler
`0x30CD28` or a Windows unwinder.

The corpus has 181 cases, five retained four-call sequences, eleven baselines
and 87 exact stopped prefixes. Recovery includes a failed fourth metadata
request before the warm flag is set, followed by complete cold retry and warm
reuse. All adjacent retained final/initial snapshots match. Every stopped event
list equals its entire baseline prefix and its final state equals the complete
pre-effect snapshot. Seventy-six of 79 instructions execute; the only three
excluded instructions are post-nonreturning guard NOP/INT3 and rethrow INT3.

Two separately syntax-preceded successful producers matched byte-for-byte.
Both lossless memory and complete-snapshot codecs assert full combined expansion
equals the original modeled report. Decode with
`audit_character_oracle_reveal_join.expand_memory(audit_report_snapshots.expand_snapshots(report))`.
The matching private peer is
`B:/CodexTools/DemonBluffReverseEngineering/artifacts/f530404b0f3f_807de4a83df4/character_data_preferences.peer.private.json`.
All 36 RE infrastructure tests pass. Actual preferences persistence, collection
and string implementations, skin loading, runtime admission and exception
machinery remain outside this caller contract.

## Guarded Rust normal caller replay

The guarded Rust [preference-loading replay](notes/systems/character_data_preferences.md)
compares 84 inert normal native cases and eight retained calls. Six tests check
complete physical state, chronological history, volatile registers, duplicate
skin requests, cold recovery and atomic storage/budget/schema guards.

The replay consumes explicit whole-service outputs for saved lookup, 24-byte
enumeration, MoveNext, equality and inert LoadSkin. Every future call is normal
and finite; historical native entries and service results retain the full prior
state. Per-invocation ordinals reset while the chronological history persists.
All nominal record ranges are checked against each other and native metadata
slots/flag; the output and active enumerators are the designated scratch
interiors. Complete future entry, history, iterator, trace and snapshot costs
are reserved before cloning. The fixed Call input reservation includes a
conservative 160-byte allowance. Cleanup, callback and stopped paths remain
native evidence rather than supported future Rust calls.

Source: [character_data_preferences.rs](../../../crates/solver-core/src/bluff/character_data_preferences.rs).
All 928 Rust library tests and the release build pass; all 36 RE infrastructure
tests passed at the native checkpoint. Independent read-only review found no
remaining blockers. Simulation and Python bridge suites were not rerun for
these offline caller modules.
