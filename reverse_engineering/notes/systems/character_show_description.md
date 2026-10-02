# Character.ShowDescription caller

Pinned build: `f530404b0f3f_807de4a83df4`.

Producer: `reverse_engineering/scripts/audit_character_show_description.py`.
Report: `reverse_engineering/reports/f530404b0f3f_807de4a83df4_character_show_description.json`.

## Exact scope

The sole target is `tdi5487.m0056`, the exact declaration
`Character::public void ShowDescription()`, with signature
`void Character__ShowDescription (Character_o* __this, const MethodInfo* method);`
and type signature `vii`. Its one complete native unwind covers
`0x368C20..0x369235` exclusive. The next managed entry is `0x369240`, preceded
by eleven `CC` padding bytes. There is one declaration at the entry.

The audit executes the complete caller with a valid authored Character graph.
It pins the private PE and Dumper hashes, full file-backed bounds, next entry,
padding, exact nominal fields, consumed operand widths, delegate fields,
virtual Value.GetValue slot, and complete HintInfo call ABI. The tracked report
contains a native body fingerprint and selected assertion counts. Complete
native bytes and disassembly remain private.

All helper implementations remain **supplied**: Acted.GetActed/Act/Hide,
CharacterData.GetDescription, String.IsNullOrEmpty, generic history get_Item,
Characters.HighlightCharacters, Character.GetHiddenCardsAmount, Unity object
and component operations, virtual Value.GetValue, allocation, HintInfo.ctor,
metadata resolution, class initialization, reference barriers, and UI actions.
GetDescription is the actual `0x3B4BF0` callee; this caller does not call GetHints.
No concrete provider implementation, generic shared-RVA alias, renderer,
localization rule, lifetime policy, scheduler, or exception unwinding is inferred.

## Description and prior speech

The normal branch captures `Character.bluff` at `+0x58` before a possible
Object class initialization, then supplies that captured pointer to Unity
Inequality with full zero `RDX/R8`. Only returned `AL` selects the bluff branch.
The caller reads the raw `leftAct` byte at `+0xB0` after this callback.

When leftAct is nonzero, the caller repeatedly reloads `acteds` at `+0xA8`.
It supplies Component.get_gameObject, guards a null returned GameObject, and
supplies GameObject.activeSelf. A nonzero raw `AL` enables GetActed followed by
String.IsNullOrEmpty. A nonempty supplied result enables **two more** GetActed
calls. The second produced string is stored natively into `savedAct` at
`+0x198` before its supplied reference barrier. The third string is passed to
the left Acted.Act receiver captured at `+0xB8` before that third GetActed call.
The three produced strings can differ or be null independently. The delayed
Act argument has exact float bits `0x3E4CCCCD` in XMM2 and full `R9=0` for
MethodInfo. The caller then reloads acteds and supplies Hide.

With no Unity-live bluff, the caller guards reloaded dataRef (`+0x50`) and
checks its Role pointer at `+0x140`. A null Role skips the data-hint stage,
including when showDisguise is set. With a live bluff, no corresponding Role
gate occurs. If the raw showDisguise byte (`+0x1A0`) is zero, the caller reloads
the relevant data/bluff pointer, supplies GetDescription, then supplies
String.IsNullOrEmpty. A nonempty result selects UIEvents.OnShowCharacterHint
at static `+0x28`. If showDisguise is nonzero, it selects
OnShowCharacterDataHint at `+0x20` and reloads the current bluff as the payload;
that payload can be null after a supplied callback.

Each UI action captures the delegate, loads `method` at `+0x28` into R9 and
`method_code` at `+0x40` into RCX, and dispatches via `invoke_impl` at `+0x18`.
The actor's current hintPivot (`+0x38`) is loaded into R8 for these hint calls.
The deliberately different delegate m_target at `+0x20` remains unused.
UIEvents class initialization is not requested by this caller.

## Repeated history reads

The caller guards `actedInfos` at `+0x148` and checks its signed count DWORD
at `+0x18`. If positive, it supplies the exact
List<ActedInfo>.get_Item MethodInfo at `0x270DFB8` to the shared `0xB22150`
gateway, using the reloaded count minus one as a full zero-extended EDX value.
The generic gateway is supplied, not classified as an actual ActedInfo list
implementation. The Count metadata slots are resolved by the cold prologue;
the caller consumes count fields directly rather than calling Count helpers.
It does not directly traverse a backing array in this scope.

The first returned ActedInfo is guarded, then its characters pointer (`+0x18`)
controls whether another history read occurs. The second returned ActedInfo
and characters pointer are guarded; a positive characters-list count enables
highlighting. The caller captures Characters.Instance at static `+0` before
a **third** get_Item, guards the third returned info and the captured instance,
and passes that third info's current characters pointer to HighlightCharacters.
The list checked for a positive count can differ from the list passed to the
supplied highlight service. A callback changing the Characters static instance
after capture does not replace the captured receiver.

After highlighting or skipping it, the caller reloads the history pointer and
count. A positive count captures the current bluff before a second possible
Object class initialization and supplies Unity Equality. It reloads
OnShowCustomHint at static `+0x90`; returned AL chooses the current dataRef or
current bluff payload. Callback changes to those fields affect this later
reload independently of the pointer used for Equality.

The supplied get_Item results are explicit inputs. One diagnostic callback
reduces a history count between reads and demonstrates the caller requesting
the raw index `0xFFFFFFFF`. The supplied gateway returns authored data for that
request. This establishes caller argument formation without claiming the
actual generic provider would admit that index or return normally.

## Hidden and killed-hidden hints

If the exact current-state DWORD is Hidden (`5`), the caller checks GameData
class initialization and reloads its metadata root after the supplied initializer.
Only the exact static GameState DWORD Gameplay (`30`) enables the remaining
path. It supplies GetHiddenCardsAmount, captures only the low EAX DWORD, and
walks PlayerController.PlayerInfo → PlayerInfo.blocks → Resource.value, guarding
each pointer. The virtual Value.GetValue call uses slot 7's function pointer at
class `+0x1A8` and full MethodInfo pointer at `+0x1B0`. Both result DWORDs are
compared as signed integers. If the hidden-card count is greater, the caller
returns; otherwise it considers a blocking hint.

If the current state is not Hidden but prevState is exactly Hidden and the raw
killedByDemon byte is nonzero, the caller considers a killed-hidden hint instead.
Both paths first capture OnShowHint at UIEvents static `+0x10`; a null delegate
suppresses allocation and construction. A supplied allocation returns an opaque
HintInfo record. The caller then loads the appropriate literal and the shared
empty literal from their slots, preserving effects of allocation callbacks:

| Slot | Exact value |
| --- | --- |
| `0x26D8428` | `Can not reveal, something is blocking me!` |
| `0x26DF2A0` | `Killed by the demon\ncan not be revealed` |
| `0x26DF1B8` | empty string |

The HintInfo constructor receives RCX=allocated record, RDX=text, R8=null image,
and R9=empty hints. The full stack arguments are empty flavor, empty title,
pointer to four zero color DWORDs, and zero MethodInfo. The audit records and
independently checks all of those arguments, the physical color bytes, and the
native return address. The constructor's modeled field writes are explicitly
supplied effects; the frozen actual constructor is not joined here.

The caller dispatches to the OnShowHint delegate captured before allocation,
even if a callback replaces that static field, and reloads the actor's hintPivot
after construction. Supplied allocation fixtures cover two opaque receivers;
retained fixtures explicitly reuse a record. No fresh-allocation or allocator
identity policy is inferred from that diagnostic driver behavior.

## Verification and retention

The independent model starts from every complete initial byte record, metadata
and literal slot, native flag, and logical service state. It predicts the entire
ordered event sequence, full RCX/RDX/R8/R9 and caller for every boundary including
native null guards, each service-entry snapshot, and complete final storage.
Serialized callback write plans are supplied inputs shared with the driver;
the branch, capture/reload, service-order, and ABI predictions are independent.

Only reached native savedAct/flag stores and completed initializer, constructor,
or callback writes receive retention permissions. Skipped phases and stopped
service entries grant no writes. Metadata/literal slots and native flag changes
are checked separately from the diagnostic object windows. Stops compare the
whole event prefix and full final state with the exact boundary-entry snapshot.
The normal return restores the stack, all eight integer nonvolatile registers,
and XMM6..15 despite volatile integer and XMM0..5 poisoning on supplied returns.

Diagnostic windows retain every authored byte but do not establish managed
object extents or meaningful types for unused sentinel fields.

The corpus executes 342 of 343 decoded instructions; only the terminal null-guard
int3 at `0x369234` remains unexecuted. There are 32 selected operand assertions.
The final corpus has 310 profiles, four retained three-call sequences, twelve
independent baselines, and 262 exact stopped-prefix probes. Both final producers
passed a preceding Python syntax check and produced byte-identical
26,865,483-byte reports. All 36 reverse-engineering infrastructure tests passed.
The lossless memory and snapshot codecs both round-trip to the complete
original report before serialization.

Report SHA-256:
`031eaae56d152126a402c737f4fe9d59d56b40591fea96d82fbc4bd4c3622aac`.
The exact independent private peer is
`B:/CodexTools/DemonBluffReverseEngineering/artifacts/f530404b0f3f_807de4a83df4/character_show_description_peer.json`.
