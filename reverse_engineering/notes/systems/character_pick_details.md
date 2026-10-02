# Character picker and details callers

Pinned build: `f530404b0f3f_807de4a83df4`.

Producer: `reverse_engineering/scripts/audit_character_pick_details.py`.
Report: `reverse_engineering/reports/f530404b0f3f_807de4a83df4_character_pick_details.json`.

## Exact native scope

| Symbol key | Exact declaration | Entry | Body end, exclusive | Next managed entry | Padding |
| --- | --- | --- | --- | --- | --- |
| `tdi5487.m0040` | `Character::private void PickCharacter()` | `0x367790` | `0x36788E` | `0x367890` | 2 `CC` bytes |
| `tdi5487.m0053` | `Character::private void Update()` | `0x369700` | `0x3697B7` | `0x3697C0` | 9 `CC` bytes |

Both declarations have exact Dumper signatures
`void Character__<name> (Character_o* __this, const MethodInfo* method);`
and type signature `vii`. Each entry has one declaration. The producer pins
the Dumper extraction hashes, exact signatures, complete native unwind bounds,
file-backed section extent, next managed entry, and alignment bytes. The report
retains per-body byte length, instruction count, and SHA-256; complete native
bytes and disassembly stay private. No shared-RVA alias is
promoted. `ShowDescription`, actual lifecycle dispatch, and actual picker,
engine-input, delegate, or UI implementation remain separate boundaries.

## PickCharacter

The cold metadata path resolves `CharacterPicker_TypeInfo` at `0x26DCB58` and
the exact `Method$System.Collections.Generic.List<Character>.Contains()` slot
at `0x27130A0`, then writes the native flag `0x288C177`. If the picker class
initialization DWORD at `+0xE0` is zero, the caller requests initialization.
It calls supplied `CharacterPicker.ClickedCharacter` (`0x378DB0`) with the
actor and full zero `RDX` for MethodInfo.

After that callback, the caller reloads the picker metadata root, its static
storage at `+0xB8`, and `PickedCharacters` at static `+0`. A null list reaches
the native null guard. The exact caller supplies its List<Character> MethodInfo
to the `0xB55950` generic Contains gateway. The Dumper declaration at that
gateway is List<object>.Contains; this audit supplies its behavior and does
not classify the generic body or any alias as the exact Character list.

The caller captures `Character.pickeds` at `+0x188` **after** Contains returns.
The raw return's `AL` selects the on/off loop; upper return bits are irrelevant.
Each loop re-reads the signed low DWORD of the captured array length at `+0x18`
and loads each GameObject pointer at `+0x20 + index*8`. The high length DWORD is
diagnostic storage. An array replacement before capture changes the consumed
receiver; replacement of the actor field after capture does not. A callback can
change the captured array's next slot or length before the next iteration.
Aliased slots cause repeated calls to the same supplied object.

The off branch supplies full `RDX=0`; the on branch writes only `DL=1`, retaining
the poisoned upper bytes of `RDX`. Both supply full `R8=0` for MethodInfo to
`GameObject.SetActive` (`0x1C7D810`). The supplied service records the full
register and consumes the low byte. No engine active-state or rendering claim
is made from these supplied calls.

The signed length check immediately precedes an unsigned bounds check with no
callback between them. The bounds guard call at `0x367882` is unreachable for
these bounded inputs without an artificial instruction-time memory race. It
and the two terminal `int3` instructions are the only decoded instructions not
executed. The actual null guard call and both membership loops execute.

## Update

The cold path resolves `Gameplay_TypeInfo` (`0x26F8140`) and
`UIEvents_TypeInfo` (`0x26E5580`), then sets flag `0x288C183`. A nonzero raw
`Character.hover` byte at `+0x190` enables the supplied
`Input.GetMouseButtonDown` (`0x1CD3DD0`) call with full `RCX=1` and `RDX=0`.
Only its raw `AL` controls the next gate. The actor's `killedByDemon` byte
`+0xED` and state DWORD `+0xE4` are checked after that callback. Any nonzero
killed byte or exact `ECharacterState.Hidden=5` suppresses details.

Gameplay initialization, if requested, precedes a metadata-root reload. The
caller reads static `GameplayState` at `+0x28` and suppresses details only for
the exact DWORD `EGameplayState.Night=20`; values with the same low byte are
not equivalent. It then loads `UIEvents.OnShowCharacterDetails` at static
`+0x30` without requesting UIEvents class initialization.

If the captured delegate is nonnull, the caller loads `method` at `+0x28`,
`method_code` at `+0x40`, and tail-jumps to `invoke_impl` at `+0x18` with the
actor in `RDX`. The deliberately different delegate `m_target` at `+0x20` is
unused. Null, actor, other-actor, and unrelated method-code receivers are covered.
The tail callback sees the outer fixture return sentinel, explicitly labelled
as such rather than represented as an invented native call-site return.

## Supplied services and independent comparison

Metadata resolution and class initialization are explicit supplied services.
Initialization writes only the reached class's `+0xE0..+0xE3` bytes after a
successful supplied return. ClickedCharacter is inert by default; an explicitly
selected profile toggles logical list membership. Contains uses that supplied
membership or a separately authored full raw return value. Membership records
are logical service inputs, not claims about physical native List internals.
Input, SetActive, and the details delegate likewise have explicit supplied
results and effects. Their object windows remain complete diagnostic byte
records rather than asserted serialized object extents.

An independent ordered Python model starts from each complete initial snapshot
and predicts every service-entry snapshot, full final storage, logical service
state, return/stop outcome, raw `RCX/RDX/R8/R9`, and caller residue. This includes
the no-argument null guard's residual registers. Every normal native return
checks restored stack, all eight integer nonvolatile registers, and XMM6..15;
supplied returns poison volatile integer registers and XMM0..5.

The producer grants physical write permissions only to completed class or
callback effects and separately verifies reached native metadata-flag writes
and completed metadata-root mutations. A configured phase that is skipped or
stopped has no write permission. Stops compare the entire event prefix and
complete final snapshot against the corresponding boundary-entry snapshot.
Retained sequences invoke both actual callers in one physical state, preserving
metadata flags, class words, actor fields, array aliases, and supplied histories.

The report uses the existing lossless memory and snapshot pools. Both codec
round trips must reproduce the complete unpooled report before serialization.
The final corpus contains 420 profiles, four retained four-call sequences,
nine independent baselines, and 56 exact stopped-prefix probes. There are 19
exact instruction assertions; 113 of 116 decoded instructions execute. Both
final producer processes passed a preceding Python syntax check and produced
byte-identical 4,069,528-byte reports. All 36 reverse-engineering infrastructure
tests passed.

Report SHA-256:
`7732b509bf70d7c358c93a30fa5091edd62cd9b69dc09120c27187485af28305`.
The independent private peer is
`B:/CodexTools/DemonBluffReverseEngineering/artifacts/f530404b0f3f_807de4a83df4/character_pick_details_peer.json`.
