# ManageCharacters complete caller orchestration

Pinned build `f530404b0f3f_807de4a83df4`. The distinct
`audit_manage_setup_caller.py` audit executes the actual native
`Characters.ManageCharacters` body at `0x36CE30` through normal return with
explicit supplied callees. It preserves the separate 44-case
[pre-Init prefix audit](manage_pool_prefix.md). The new report has 140 fixtures,
264 executed caller instruction addresses and 23 exact instruction assertions.
GameAssembly, script metadata and dump declarations are hash-pinned. Exact
Character/Characters/CharacterData/iterator fields and four generic collection
MethodInfo bindings are checked. Every successful fixture restores the stack
and all eight nonvolatile integer registers.

## Complete sequence and identity

The caller invokes position update, unique-pool construction and duplicate-pool
construction, then enumerates the current board. Each occurrence fetches the
same-index data-roster occurrence and calls `Character.Init` with wrapping
`abs(current board Count - index)`. All Init calls complete before publication
through `Gameplay.UpdateCharacters`, and publication completes before a new
board enumeration calls `Character.Act(Init=3)` on every occurrence. Physical
aliases are not deduplicated in either pass; data-roster aliases are preserved.
The supplied Init gateway writes only synthetic `dataRef` for subsequent caller
reads; this fixture does not execute actual Character.Init initialization.

The initial enumerator captures its board, but each Init iteration rereads the
owner's board field for Count. Replacing the board during first Init preserves
the original three-card enumeration while changing IDs from `[3,2,1]` to
`[3,1,0]`. Publication gets the replacement board; its later Init-action pass
also gets the replacement. Replacing the board during publication affects that
following pass. Replacing during the first Init action leaves that enumeration
on the old list but affects subsequent ordered Start scans.

## Ordered Start dispatch

After the complete Init-action pass, the caller captures `startGameActOrder`.
For each array occurrence, it freshly captures the owner's current board and
compares the ordered CharacterData with each physical card's current `dataRef`
using the supplied Unity equality gateway. A match invokes `Character.Act`
with Start=5 before inspecting the ordered data's current role field.

A null role or ordinary role stops at the first match. A role whose native
class hierarchy contains Alchemist, Poisoner or Puzzlemaster continues through
all matching board occurrences. Fixtures exercise each direct type and a
synthetic subclass, as well as null role, duplicate order entries and absent
identities. These are runtime class checks, not a test of a public role name.
Repeated ordered entries repeat requests; any Character.Act latch behavior is
callee scope and is not inferred here.

The role is reread after the Start gateway. Changing ordinary to Alchemist
there changes the same scan into an all-match scan; changing Alchemist to an
ordinary role stops it immediately. Later card identities are likewise read
when encountered. The owner array reference is captured once: replacing
`startGameActOrder` during Start does not replace the active traversal. Its
length is reread from the captured array, so controlled length reduction stops
later entries. Such array-length mutation is an adversarial raw fixture, not
an operation offered by managed arrays.

## Completion and failure boundaries

After ordered scans, `onSetup` is read from the owner, so earlier supplied Act
callbacks can install or clear it. The delegate invocation receives its method
code and method context from its own fields. Then the caller resolves/allocates
the ShuffleDeck iterator, invokes its folded no-op constructor, writes its
state field to zero and supplies it to StartCoroutine. This is registration
only: neither ShuffleDeck.MoveNext nor the Unity scheduler executes here.
The iterator has no captured owner field in the pinned declaration.

Warm and cold baseline fixtures inject a stop at every reached supplied gateway.
Every attempted event and synthetic state exactly matches that baseline's
failure prefix. Native null checks also cover null board, roster, order and
physical card; indexed roster exhaustion stops at the supplied bounds service.
An empty board/order returns despite a null roster. A null data identity can be
forwarded by Init and match a null order entry under the supplied equality;
Start is then requested before the caller's ordered-data null failure.
Completed effects are never rolled back by the harness. Exception unwinding
and real callee partial effects are not modeled.

Layout, both pool builders, Character.Init, publication, Character.Act,
delegate effects, allocation and coroutine registration remain supplied
boundaries. Collection enumeration/get_Item/disposal, metadata resolution,
class initialization and Unity equality are also supplied. Equality is pointer
identity for these fixtures; destroyed Unity objects are not modeled. Stable
list occurrence sequences are supplied even when owner fields change; managed
list version checks and mutation during actual enumerator execution are not
claimed. Callback mutations are explicit controlled writes, not complete
recursive ManageCharacters execution. This establishes native caller control
flow under those service contracts, not a complete game/engine reconstruction.

Run with the private Unicorn 2.1.4 runtime on PYTHONPATH:

```powershell
python reverse_engineering/scripts/audit_manage_setup_caller.py GAME_ROOT DUMPER_ROOT --output reverse_engineering/reports/f530404b0f3f_807de4a83df4_manage_setup_caller.json
```

Native fixtures and Python compilation pass. No Rust or shared coverage
inventory is changed, and no proprietary bytes or decompiler bodies are stored.
