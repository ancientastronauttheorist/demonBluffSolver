# Roguelike kill-handler accumulation and native score mutation

The [lifecycle audit](roguelike_standard_lifecycle.md) found that
RoguelikeStandard.DeInit passes its OnCharacterKilled callback to Combine.
This follow-up executes the native delegate constructor, Combine/Remove
wrappers, multicast list implementations, native equality, specialized clone,
Action Invoke and multicast invocation trampoline. The finding now has an
executed native list-to-score path, not just a mocked delegate result.

## Accumulation and invocation

Two distinct delegates with the same instance and method compare equal, but
Combine retains both entries. It does not deduplicate equal callbacks.
Appending singles and combining two multicast lists preserves their order.
The verified Init/DeInit kill-handler operation sequence, composed with the
native Combine implementation, produces these counts from an empty event:

| Operation | Kill handlers |
| --- | ---: |
| Init | 1 |
| DeInit | 2 |
| Init | 3 |
| DeInit | 4 |
| Init | 5 |

This composition uses native delegate operations, with method/target fixtures
matching the lifecycle call pattern; it does not invoke the complete lifecycle
methods again. Other code can clear/replace the actual global event, so these
counts are not asserted for an observed live run.

The shared Action constructor at `0x4D5B60` supplies multicast trampoline
`0x5C70` through delegate extra_arg `+0x38`. Combine's specialized clone helper
`0x25E600` creates a new delegate, carries the target/method, installs that
extra_arg as invoke_impl `+0x18`, and sets method_code `+0x40` to the new
receiver. This helper is not a generic byte-for-byte MemberwiseClone.

Action<object>.Invoke at `0x4A86F0` routes through invoke_impl. The native
multicast trampoline reads the invocation array and visits every entry in
order, passing the same character argument and each entry's saved target and
method. Both private helper boundaries are verified against their unwind
entries. Five accumulated handlers execute the native scoring callback five
times in the fixture.

Native Remove handles equal single delegates by returning null, and leaves a
single delegate when target identity differs. The fixture also exercises
multicast subsequence removal with native equality and array-copy gateways.
This is not exhaustive Remove behavior: the generic LastIndexOf route for
removing a single value from a longer multicast list is not executed. Removing
distinct, elementwise-equal two-entry arrays in this fixture yields a nonnull
clone with an empty invocation array; no wider normalization claim is made.

## Kill-score behavior

RoguelikeStandard.OnCharacterKilled reads Character.dataRef `+0x50`, then
CharacterData.type `+0x130`. Only Minion `30` and Demon `100` qualify. It does
not query displayed bluff, registerAs, runtime alignment or corruption here.
Null character/dataRef fails before score mutation.

For an eligible character it reads CharactersHelper.GetUnrevealedCharactersCount,
then updates roundScore `+0x28` by `10 * (count + 5)` with native int32 arithmetic.
The harness supplies the helper result explicitly; live board counting remains
outside this execution boundary. Eighty score cases cover all five character
factions, four count inputs (including a synthetic negative boundary), two
initial scores including overflow, and UI-event presence.

After the score write, the method invokes optional UIEvents.OnUIUpdate. A
failing UI gateway preserves the increment. Through the actual multicast
trampoline, that failure stops before later handlers: with count seven, the
first increment is 120. With no failure, five handlers yield 600. A second
fixture containing two old-instance handlers and one new-instance handler
updates the old instance by 240 and the new instance by 120. Retained delegate
targets therefore matter even after GameData publishes a replacement mode.

## Reproduction and remaining services

The [script](../../scripts/audit_roguelike_delegate_score.py) pins the DLL,
Dumper script and declaration fields, checks eleven native relationships,
validates preserved registers and stack on normal returns, and records its
executed native ranges in the
[report](../../reports/f530404b0f3f_807de4a83df4_roguelike_delegate_score.json).
No target manifest or shared coverage membership is added.

Runtime class/type hierarchy, method-info resolution and instance-method
classification, object/array allocation, GC reference stores, runtime casts,
and Array.Copy are explicit fixture gateways. Native clone field assignment,
list order/mutation, target/method equality control flow, constructor selection,
invocation traversal and score arithmetic execute from the pinned binary.
The reflection method provider returns fixture identities for exact method
matches; this does not cover virtual/reflection binding variants.

```powershell
python reverse_engineering/scripts/audit_roguelike_delegate_score.py `
  'B:\SteamLibrary\steamapps\common\Demon Bluff Playtest' `
  'B:\CodexTools\DemonBluffReverseEngineering\artifacts\f530404b0f3f_807de4a83df4\il2cppdumper-v6.7.46' `
  --output reverse_engineering/reports/f530404b0f3f_807de4a83df4_roguelike_delegate_score.json
```

No live play, event mutation, save changes or game patches occur. Whole-runtime
subscription history, event resets, method-binding variants, reentrant UI
mutation and complete progression/scoring services remain separate boundaries.
