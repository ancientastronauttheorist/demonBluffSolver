# Actual Manage pool-to-Init join before publication

Build `f530404b0f3f_807de4a83df4`. The
[audit](../../scripts/audit_manage_initialization_join.py) extends the existing
[actual pool harness](../../scripts/audit_manage_pool_composition.py) through
actual Init occurrences without resetting memory or replaying a prior report.
Both pool builders, source getters, filters and predicate calls execute first.
The [report](../../reports/f530404b0f3f_807de4a83df4_manage_initialization_join.json)
records eight development fixtures, sixteen attempted Init calls, twelve
completed calls, 960 native instruction addresses and thirty selected
instruction assertions. Binaries/Dumper sources and fields are pinned;
initializer body hashes contain no native bytes. Completed Init calls restore
the entry stack and all eight nonvolatile integer registers.

The corrected harness executes the folded constructor's native `ret 0` instead
of falling through to the pool harness's supplied disposal gateway. Its caller
regression checks that RAX still contains the new iterator at `365D2F`; a
synthetic `ret(0)` fails that check. Report compatibility preserves all eight
families, actor/continuation projections and counters, but removes the spurious
pool disposal event from each completed iterator construction. This correction
does not change the default 760-case pool corpus or claim a new admitted domain.

## Retained aliases and callback prefixes

Each fixture captures board `[A,A]` and roster `[D0,D1]`. D0/D1 are authored
numeric asset identities 7 and 1, not public role names. Actual caller arguments
have displayed IDs `[2,1]`. Second Init overwrites A's current data to D1. Every
completed call registers a fresh iterator retaining A; the second call leaves
the first iterator's bytes unchanged. IDs are not physical or queue order.

In `registration_only`, supplied StartCoroutine returns without stepping:
iterators remain state zero with null current. In `first_yield`, the service
invokes actual first MoveNext synchronously with an authored Windows x64 frame.
Native code copies the current source role through a supplied clone service and
yields a separately allocated WaitForSeconds. Both iterators become state one;
A's shared role points to the second clone. Both stages execute actual Hidden
RefreshCharacter; RefreshView/presentation remain supplied inert services.

Replacing the owner's board with `[B]` in the first state callback preserves the
original enumerator. Second Init still targets A, while Count rereading changes
IDs to `[2,0]`. B stays unchanged. The audit stops before publication at
`36D01E`; it does not execute the replacement board's subsequent action pass.

Stopping the second callback occurs after data/ID/info/state writes and before
second status clear, refresh, clone or iterator publication. The first iterator
survives, no second is admitted, and first-yield A retains the first clone while
its data is already D1. Status version increments once, info version twice.
The full initializer event prefix and stopped actor/continuation state match
the corresponding successful fixture, including board replacement. No rollback
or managed exception unwinding is claimed.

## Scope and reproduction

The default pool corpus retains its pre-first-Init boundary. The join uses its
stable collection services, supplied occurrence RNG stream, layout, metadata,
allocation and GC services. Callback replacement is an authored field write,
not arbitrary managed mutation/reentrancy. Cloning, UI/logging, Unity liveness,
RefreshView and coroutine effects are supplied. Synchronous first-step behavior
has separate [native engine evidence](unity_coroutine_bridge.md); this fixture
does not execute the engine or recover deadlines/resume order.

The Rust initializer regression compares complete actor projections, callback
observations and retained continuations for the two successful first-yield
families (four Init calls). Registration-only and stopped partial bodies are
outside that replay's admitted domain and stay native-only evidence. Board
replacement is an external effect for the primitive, not newly supported batch
caller mutation. No publication, Act Init/Start, resumed Reveal, concrete clue,
player adapter or solver deduction runs here. These are development fixtures, not held-out
observations or actual generated villages. Scene dependencies are valid, raw
death/bluff references absent; broader null/liveness/failure domains remain in
the [initializer audit](character_initialization.md).

With the pinned private Unicorn 2.1.4 runtime on PYTHONPATH:

```powershell
python -m py_compile reverse_engineering/scripts/audit_manage_pool_composition.py reverse_engineering/scripts/audit_manage_initialization_join.py
python reverse_engineering/scripts/audit_manage_initialization_join.py GAME_ROOT DUMPER_ROOT --output reverse_engineering/reports/f530404b0f3f_807de4a83df4_manage_initialization_join.json
```

The distinct [publication/action join](manage_publication_action_join.md) now
retains this actor/pool/continuation state through publication and generic
Act Init/Start. Concrete role bodies, queue admission and resumed Reveal remain
open. The
[S0 contract](../../SOLVER_CONTRACT.md) distinguishes native dependency closure
from a solver-integrated supported domain.
