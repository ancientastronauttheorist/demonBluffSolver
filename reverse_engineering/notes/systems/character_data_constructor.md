# CharacterData constructor caller

This audit executes the actual current-build `CharacterData` constructor offline.
Allocation, generic List constructors, metadata initialization, reference barriers,
and the ScriptableObject base constructor remain explicit supplied services. The
independent ordered model compares every complete event, physical snapshot and
final state with the native execution before report pooling.

## Exact declaration and complete bounds

The sole managed binding at `0x3B50A0` is `tdi5845.m0020`, with immutable symbol
key `CharacterData::public void .ctor()`. Dumper declares
`void CharacterData___ctor (CharacterData_o* __this, const MethodInfo* method);`
with type signature `vii`. The exact owner is `CharacterData : ScriptableObject,
ICharacterLocData, ICardData`, TypeDefIndex 5845; its declaration is bounded by
the closing brace. No other declaration is classified through a shared service
RVA.

The complete body occupies `[0x3B50A0, 0x3B5274)`, one matching unwind range,
468 file-backed bytes and 95 instructions. Its final instruction is the
five-byte tail jump at `0x3B526F` to `0x1C8A5C0`. The next managed entry is
`0x3B5280`; all 12 intervening bytes are `CC` padding. The repository report
stores the body byte length, instruction count and SHA-256, without copying
complete native bytes or disassembly. Fourteen selected operand assertions and
the complete call-site lists are authored in the executable source.

## Native initialization order

The caller captures its input owner before any supplied metadata service. A zero
metadata flag performs ten ordered initializations for the five exact
`List<T>` TypeInfo slots and five `List<T>..ctor()` MethodInfo slots. Only after
all ten services complete does native code store flag byte 1. Any nonzero flag,
including diagnostic `80` and `FF` bytes, skips that prefix.

Each list allocation loads the current matching TypeInfo slot. Its returned
record is captured across the subsequent List constructor service. That call
loads the current matching MethodInfo slot after allocation callbacks, then the
caller stores the captured record in the owner and invokes the reference barrier
with the exact field address and captured pointer. The six operations are:

| Order | Exact generic argument | Exact CharacterData field | Offset |
| --- | --- | --- | --- |
| 1 | CharacterData | bundledCharacters | `0x48` |
| 2 | SkinData | skins | `0xC8` |
| 3 | AchievementData | achievements | `0xD0` |
| 4 | ECharacterStatus | additionalStatuses | `0x118` |
| 5 | ECharacterTag | tags | `0x120` |
| 6 | CharacterData | canAppearIf | `0x128` |

The common List constructor gateway is supplied at `0xB02160`, pinned to its
exact canonical `List<object>` declaration. Each caller's generic identity is
bound through its own exact metadata slots. This establishes the six caller
operations, without executing or promoting the folded generic constructor body.
Allocation results may alias when their nominal generic kind agrees; the corpus
includes the final CharacterData list sharing the first allocated record.

After the six barriers complete, native code stores `bluffable` at `+0x13C` and
`picking` at `+0x13E` to byte 1. It then restores its frame and tail-calls the
exact supplied `UnityEngine.ScriptableObject` constructor at `0x1C8A5C0`, with
captured owner in RCX and full zero RDX MethodInfo. The tail boundary's caller is
qualified as the fixture return sentinel. Supplied base callbacks can change
those freshly written bytes. The native CharacterData body never initializes
`usuallyDisguised` at `+0x13D` or `currentSkin` at `+0xC0`; the diagnostic seed
and reached callbacks determine their observed contents.

## ABI, mutations and retention

Every supplied boundary, including a stopped entry, records raw RCX, RDX, R8,
R9, all seven integer volatile registers, XMM0 through XMM5, its exact caller,
phase and pre-effect full snapshot. Services poison the integer/XMM volatile
registers independently of their authored RAX outputs. The ordered model also
compares all final volatile integer and XMM values on both returns and stops.
Normal returns preserve Win64 stack discipline, all eight integer nonvolatile
registers and XMM6 through XMM15. The public return type is void; recorded RAX
bits are supplied base-constructor residues, not a produced CharacterData value.

The corpus covers zero and sentinel diagnostic storage, empty and prefilled
owners, two owner receivers, fresh and reused physical records, raw warm flag
bytes, legal list aliases and independent opaque service return patterns.
Callbacks can mutate metadata roots, List storage, owner fields and unrelated
records. In particular, allocation callbacks can replace the subsequently
reloaded constructor MethodInfo; constructor callbacks cannot replace the
already captured allocation result; barriers can overwrite a just-stored owner
field; later native Boolean stores overwrite earlier callbacks; and base
callbacks occur after both native Boolean writes.

Physical windows retain every byte, including padding and opaque service data.
Their sizes are diagnostic windows, not asserted managed object extents. Only
actual reached native writes and completed callback writes grant mutation
permission for that invocation. Metadata slots and flag transitions are checked
separately. A supplied failure stops before its state update or callback effects,
and every stopped case matches the entire baseline event prefix and exact
pre-effect snapshot. A null owner reaches the actual first field-store write
fault after allocation and the first List constructor; preceding metadata and
service effects remain preserved.

## Verification and limits

The final corpus comprises 148 cases, four retained four-call sequences, eight
baselines and 205 supplied-boundary stops. Retained sequences cover success,
partial barrier failure, recovery and a later base callback; each previous final
snapshot equals the next initial snapshot. All 95 decoded native instructions
execute. Two syntax-preceded independent producers emit byte-identical reports,
and all 36 reverse-engineering infrastructure tests pass.

Source: `reverse_engineering/scripts/audit_character_data_constructor.py`.
Report: `reverse_engineering/reports/f530404b0f3f_807de4a83df4_character_data_constructor.json`.
The peer report lives privately at
`B:/CodexTools/DemonBluffReverseEngineering/artifacts/f530404b0f3f_807de4a83df4/character_data_constructor_peer.json`.

No actual heap allocator, generic List internals, metadata resolver, collector,
base constructor or Unity initialization executes here. Their supplied effects
are explicit inputs. Skin loading, preferences, identity generation and engine
asset construction require separate evidence.
