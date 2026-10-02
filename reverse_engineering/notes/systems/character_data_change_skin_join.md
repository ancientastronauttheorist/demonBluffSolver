# CharacterData ChangeSkin and unlock lookup composition

This family executes actual `CharacterData.ChangeSkin` calling actual
`CharacterData.CheckIfSkinUnlocked` in one physical offline graph. The caller's
captured skin and the inner lookup's selected unlock recipient are independent
identities, even when their skin IDs compare equal. The complete ordered model
predicts every event, physical snapshot, result and final volatile register
before lossless report pooling.

## Exact native bindings and boundaries

| Stable ID | Exact declaration | Type signature | Complete body | Next entry |
| --- | --- | --- | --- | --- |
| tdi5845.m0014 | public void ChangeSkin(SkinData skin) | viii | `[0x3B42C0, 0x3B43A8)` | `0x3B43B0` |
| tdi5845.m0013 | public bool CheckIfSkinUnlocked(string skinId) | iiii | `[0x3B43B0, 0x3B4511)` | `0x3B4520` |

ChangeSkin has 232 file-backed bytes, 61 instructions, one unwind range with
Flags 0 and eight following `CC` padding bytes. CheckIfSkinUnlocked has 353
bytes, 84 instructions, one complete unwind range with EH/UH flags 3, handler
`0x30CD28`, and 15 following `CC` bytes. Both final one-byte `int3` instructions
are explicitly pinned. The actual nested call is at `0x3B4372`; its return site
is `0x3B4377`. Full bounds, field declarations, generic enumerator layout and
metadata are independently hash-pinned from the current build. Repository
reports contain body byte lengths/counts/SHA-256, without complete native
bytes or disassembly. Thirty-one selected operand assertions and exact
call-site lists remain authored in source.

The exact owner is CharacterData TypeDefIndex 5845. Consumed storage is
`currentSkin +0xC0`, `skins +0xC8`, and SkinData TypeDefIndex 5945 `skinId +0x18`.
The standalone frozen skin lookup source class provides its pinned setup and
service shape; its report is not spliced into this execution, and its files
remain unchanged.

## Outer admission, capture and reload chronology

ChangeSkin captures its input skin in RBX and owner in RDI before either
metadata service. A zero method flag initializes the exact
`List<SkinData>.Contains()` MethodInfo slot and Unity Object TypeInfo slot,
then writes flag byte 1. Any nonzero byte skips that prefix.

The caller loads the current Object TypeInfo and tests its full DWORD at
`+0xE0`. A zero DWORD invokes supplied class initialization on that captured
class. Replacing the metadata root during initialization does not change that
call's receiver; the second Object-class load later observes the replacement.
Fixture class flags include values with only upper DWORD bits set, verifying
that this is a DWORD gate. Only completed initializer calls grant permission
for the supplied `+0xE0 = 1` write and reached callback writes.

The first supplied Unity Object.op_Inequality call receives captured skin,
null comparison target, and full zero MethodInfo. Only AL controls whether to
perform Contains. A nonzero low byte reloads the owner's current skins List
and current exact Contains MethodInfo. Contains receives that List, original
captured skin and exact generic MethodInfo; zero AL ends the method without
replacing currentSkin or updating preferences. A zero first comparison byte
bypasses Contains.

After passing or bypassing Contains, the caller reloads Object TypeInfo,
performs its independent class-initialization check and makes a second supplied
inequality request using the same captured skin. That second raw AL result is
independent of the first. A zero low byte bypasses unlock lookup and admits
the captured skin directly, including null under explicitly supplied
comparison profiles. A nonzero byte checks the captured pointer and reads its
current skinId after the second comparison callback, then calls the actual
CheckIfSkinUnlocked with captured owner RCX, that loaded String RDX and full
zero R8 MethodInfo.

These are supplied comparison outcomes. No real Unity liveness, destroyed-object
comparison policy or runtime null admission is inferred.

## Actual inner lookup and original skin store

The inner method captures its String query and owner, performs its own
independent metadata prefix when cold, then reloads the owner's skins List.
Thus Contains and GetEnumerator can consume different Lists after a callback.
The four exact enumerator MethodInfo slots remain distinct from Contains.
Whole supplied GetEnumerator writes 24 bytes through its hidden output pointer
in RCX, receiving the List in RDX and exact MethodInfo in R8. Actual native
MOVUPS/MOVSD copies that output into the active enumerator; its original
output/exception storage, active list/index/version/current and surrounding
scratch bytes remain fully represented.

MoveNext and String equality remain supplied services with explicit raw
outputs. The actual Check caller consumes only their AL bytes. It captures the
current skin after MoveNext callbacks, loads that skin's ID, and stops at the
first nonzero supplied equality byte. Changing the active current pointer in
the equality callback does not replace the skin already captured for unlock.
SkinData.CheckIfUnlocked remains supplied with the selected skin and full
zero MethodInfo.

The inner caller captures the raw unlocked AL byte with MOVZX EDI,AL, calls
supplied Dispose, then returns that byte zero-extended through MOVZX EAX,DIL.
Diagnostic `00`, `01`, `80` and `FF` unlock bytes remain exact. With no match,
Dispose's upper RAX bits survive while XOR AL,AL forces the low byte to zero.
The outer caller tests only AL, so those preserved upper bits cannot admit a
no-match result.

The inner method can select another skin with the same ID for the supplied
unlock request. Regardless of that selection or later callbacks, an admitted
outer call stores its original captured input skin in currentSkin, invokes
the reference barrier, then supplies exact
`SavesGame.UpdateCharacterPreference` at `0x3874B0` with owner RCX and full
zero RDX MethodInfo. A barrier callback can change currentSkin before the
preference service observes it. The preference implementation and persistence
policy do not execute here. The void outer return records only opaque supplied
RAX residues, not a produced skin or public Boolean result.

## One physical graph, frames and ABI

Let S be the outer entry stack pointer. The outer frame is `S-0x28`; the actual
Check entry is `S-0x30`; its frame is `S-0x98`. The hidden output occupies
Check frame `+0x28`, with the active enumerator at `+0x40`. Actual Check entry
records its native caller return `0x3B4377`, complete incoming integer volatile
and XMM registers, owner and String argument. Outer entry records the fixture
return sentinel and nominal SkinData argument. The actual nested stack pointer
is checked at entry.

Every supplied boundary records full RCX/RDX/R8/R9, all seven integer volatile
registers, XMM0 through XMM5, exact caller, native phase and full pre-effect
snapshot. Services poison volatile integer/XMM registers independently of
authored RAX outputs. The ordered model predicts initial argument clearing,
metadata/class residues, hidden-output copy register effects, callee entry,
inner return and outer continuation. All final volatile values are compared
on returns and stopped/faulted paths. Normal returns preserve stack discipline,
all eight integer nonvolatile registers and XMM6 through XMM15.

The graph retains both owners, Lists/arrays, skins, strings, all MethodInfo
records, both Object runtime-class records, opaque exception storage and full
64-byte nested scratch window. Window sizes are diagnostic contracts rather
than managed object extents. Only reached native stores and completed supplied
writes grant mutation permission. Both method flags and all six metadata roots
are checked separately. Requested callbacks in skipped phases have no effects
and grant no permission.

Four retained four-call sequences cover normal success, an exact stopped inner
equality prefix, recovery in the same graph and a later preference callback.
They include owner aliases and different same-ID inner recipients. Each prior
final snapshot equals the next initial snapshot, retaining histories, runtime
class flags, metadata roots and native scratch state.

## Exceptions, guards and evidence limits

Outer null-List and null captured-skin guards, inner null-List/current guards,
and null-owner native read/write faults preserve exact prefixes and storage.
Guard services are explicit nonreturning stops. Whole Unity comparisons,
Contains, class initialization, metadata, iterator/string/unlock methods,
reference barriers, UpdateCharacterPreference and guards remain supplied.

Of 145 decoded instructions, 132 execute. Thirteen exclusions are explicit:
the complete inner cleanup-only path and rethrow call, post-nonreturning
padding/traps, and Check's second null captured-skin guard that is unreachable
when services preserve RDI. This ordinary caller composition does not run
cleanup or managed exception dispatch/unwind. The standalone lookup family's
separate synthetic cleanup evidence remains separate and unchanged.

## Verification

The final corpus contains 260 cases, four retained four-call sequences, 14
baselines and 161 exact stopped prefixes. Each stopped event list equals the
entire baseline prefix and its final state equals that boundary's complete
pre-effect snapshot. Two syntax-preceded independent final producers yield
byte-identical reports; all 36 RE infrastructure tests pass.

Source: `reverse_engineering/scripts/audit_character_data_change_skin_join.py`.
Report: `reverse_engineering/reports/f530404b0f3f_807de4a83df4_character_data_change_skin_join.json`.
Private peer:
`B:/CodexTools/DemonBluffReverseEngineering/artifacts/f530404b0f3f_807de4a83df4/character_data_change_skin_join_peer.json`.

Actual collection membership, iterator/string implementation, unlock policy,
Unity admission, persistence and exception machinery require separate evidence.
