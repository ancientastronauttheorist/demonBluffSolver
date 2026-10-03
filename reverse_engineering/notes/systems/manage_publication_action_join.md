# Retained initialization to publication and action dispatch

Build `f530404b0f3f_807de4a83df4`. The distinct
[audit](../../scripts/audit_manage_publication_action_join.py) continues the
[retained initializer join](manage_initialization_join.md) through actual
`Gameplay.UpdateCharacters`3811B0, `Character.Act`3645C0,
`Character.RoleAct`368790 and `CharacterHelper.CheckLying`397750. The
[report](../../reports/f530404b0f3f_807de4a83df4_manage_publication_action_join.json)
contains 161 development fixtures, 322 attempted initializers (321 complete),
266 attempted Act calls (154 complete), 44 selected instruction assertions and
1,284 distinct native instruction addresses. Four complete baselines, 156
stopped post-Init prefixes and one stopped second initializer are admitted.
The input binaries, dump, metadata and declarations are pinned. Native body
fingerprints contain no instruction bytes.

## Named blocker and exit test

Uncertain solver dependency: which shared actor, board and cloned action role
survive initialization to become the input to ordered setup dispatch? Separate
pool, Init and action fixtures did not establish that retained input. The
admitted scenario is an authored repeated-body board `[A,A]`, roster `[D0,D1]`
with numeric identities 7 and 1, descending IDs `[2,1]`, and two distinct
first-yield DelayReveal instances retaining A. This adversarial native setup
fixture is not proof that the original live generator can produce that board.

Exit test: execute pool construction, both real initializers, shallow-copy
publication and the actual generic Init/Start dispatch caller without resetting
the memory fixture. Preserve iterator bytes and compare every supplied-service
stop with its successful chronological prefix. This closes that dependency;
concrete role execution, clue production and engine queue admission remain open.

## Publication and board rereads

Both initializers finish before publication. A's current data and current clone
come from D1, while both independently allocated first-yield iterators still
retain A and their original current WaitForSeconds objects. Their bytes remain
unchanged across the second initializer, publication and all admitted actions.

UpdateCharacters allocates a List, invokes the verified generic copy-constructor
gateway with the board captured by its caller, writes that new identity into
static `Gameplay.CurrentCharacters` and performs the GC barrier. The supplied
constructor copies references and occurrence order; the native body establishes
the allocation, input, publication and barrier ordering. It does not implement
managed List internals. The published list is distinct from the owner's board
and retains `[A,A]`, including its alias, in the ordinary baseline.

Replacing the owner's board with `[B]` in the first Init callback preserves the
captured original enumeration: A still receives both Init calls with IDs
`[2,0]`, but publication and the following action pass use B. Replacing the
board inside the supplied copy-constructor gateway leaves the published copy
`[A,A]`; the following fresh enumeration calls Init action on B. B is an
authored existing ordinary role with retained corruption, so its Init dispatch
uses BluffAct. Neither replacement claims arbitrary managed reentrancy.

## Actual action routing and retained partial effects

The alias baseline calls Init action twice on A, then attempts Start twice
because the ordered input repeats D1. The actual Start latch permits only the
first Start role callback. An extra absent D0 order occurrence does not alter
that result: final actor data, rather than an erased roster occurrence, drives
the matching scan. Init leaves the latch clear. Both Init dispatches and the
first Start use the second clone through Act. Successful native Act calls
restore the stack and all eight nonvolatile integer registers.

RoleAct's real closure allocation, actor capture, delegate allocation, delegate
construction request, onActed assignment and barrier execute before each
supplied role callback. Status membership, liveness, delegate construction,
logging and virtual role bodies are explicit supplied services. Virtual bodies
are inert ordinary authored roles, with verified ABI inputs; they do not stand
for Striga, Twin, Drunk, Spy or any other concrete class. Runtime class metadata
is authored ordinary metadata, so all-match Alchemist/Poisoner/Puzzlemaster
dispatch remains in the separate generic caller audit rather than this corpus.

Every post-Init recorded event is stopped once in each complete baseline. Each
stopped attempted event list and retained actor/publication/continuation state
equals the corresponding successful prefix exactly. Failure after publication
assignment preserves the published identity; later allocation or dispatch
failure preserves completed allocations, onActed writes and Start-latch writes.
No exception unwinding or rollback is claimed. A separate second Init callback
stop preserves the earlier iterator, D1 overwrite and partial initializer
writes while preventing publication and actions entirely.

## Guarantees, exclusions and reproduction

The original pre-publication eight-case domain and default 760-case pool corpus
remain separate. The initializer's folded no-op correction now executes native
`ret 0`; the `365D2F` regression requires RAX to preserve the iterator identity.
Its report removes exactly 12 spurious disposal events while all actor,
continuation and pool projections remain identical to the earlier report.
The default pool report reproduces byte-for-byte.

This corpus uses supplied first-step scheduling, clone class/identity results,
occurrence RNG draws, layout, stable collections, UI and runtime services. It
stops before `onSetup` at `36D2DB`. It does not resume DelayReveal, execute
ShuffleDeck, recover deadlines or release order, establish queue completeness,
run concrete role bodies, construct legal player observations, compare possible
world sets or integrate a solver recommendation. It contains no independently
frozen held-out result, policy guarantee, weight model or win-rate claim.

With the pinned private Unicorn 2.1.4 runtime on PYTHONPATH:

```powershell
python -m py_compile reverse_engineering/scripts/audit_manage_publication_action_join.py reverse_engineering/scripts/audit_manage_initialization_join.py
python reverse_engineering/scripts/audit_manage_publication_action_join.py GAME_ROOT DUMPER_ROOT --output reverse_engineering/reports/f530404b0f3f_807de4a83df4_manage_publication_action_join.json
```

The report pools 103 exact snapshots; event snapshot indices preserve each
chronological occurrence. Its 161-case output reproduces byte-for-byte.
No raw native exports, proprietary bytes or
identifying local paths are included. Next: bind concrete supported action
classes and their writer effects to this retained transaction, then establish
the actual engine producer/frame/phase facts needed for queue and Reveal
admission. The strict player-history boundary remains a separate requirement.
