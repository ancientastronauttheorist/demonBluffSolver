# Gameplay iterator factories and delayed deck continuation

Build `f530404b0f3f_807de4a83df4`. The new supplemental auditor `scripts/audit_gameplay_iterator_factories.py` and report `reports/f530404b0f3f_807de4a83df4_gameplay_iterator_factories.json` cover seven previously unclassified declarations. All seven private baseline exports completed. GameAssembly and Dumper script hashes are pinned; native instructions remain private.

Coverage already records SetupDelay.MoveNext (390AB0) and InitCoroutine.MoveNext (38FD10) as understood through `gameplay_lifecycle.md`. Their generated constructor, Dispose and Current getters also have shared-body evidence. This audit adds the three public/private iterator factories, DelayedDeckIntro.MoveNext, and the three explicit Reset methods without recounting those existing boundaries.

## Factory capture

Gameplay.DelayedDeckIntro (37BDE0) and InitCoroutine (37DEA0) allocate their exact iterator types, execute the empty native base call, write state=0 and return. They neither capture nor dereference the Gameplay receiver. The metadata declarations remain instance methods, even though native code does not consume their receiver. A null receiver consequently works in these fixtures.

Gameplay.SetupDelay (380C70) allocates its own iterator, executes the same base stub, captures the supplied receiver at +20, writes state=0, emits the capture's GC barrier and returns. It also accepts a null receiver at this factory boundary; later MoveNext behavior remains the separately documented lifecycle. Current (+18) is zero in the managed-zeroed allocation fixture. Factories do not start a coroutine, run setup or acquire a scheduler entry by themselves.

## DelayedDeckIntro.MoveNext

The actual native body at 38F9C0 has three state cases:

- State0 writes -1, allocates WaitForSeconds, passes the exact float32 constant **1.0** to its constructor, stores the yielded object in Current (+18), emits its GC barrier, writes state1 and returns Boolean true.
- State1 writes -1 before reading Settings.BlindDeck. Only the integer value **1** suppresses the continuation. Every tested other signed value, including extrema, initializes Gameplay if required and calls ChangeGameplayState with numeric state **8**, then returns false.
- Any other state returns false without changing iterator state or Current and without calling those services.

The continuation reads BlindDeck on resumption, not at factory construction or first yield. It captures no Gameplay receiver, mode or phase. It does not locally check the current mode/day/state before issuing state8. ChangeGameplayState's own behavior remains a gateway here. The yielded object remains in Current after successful termination, after failed resume services, and after subsequent false-returning MoveNext calls.

The 1.0 value is a constructor argument, not a proven wall-clock interval. The existing coroutine scheduler/clock evidence controls when this continuation is resumed. These fixtures invoke MoveNext deterministically and do not simulate elapsed live time, scheduler acquisition, cancellation or nested waits.

## Failures and Reset

Allocation or wait-constructor failure leaves state=-1 and preserves the old Current because publication has not occurred. BlindDeck, class-init or state-change failure likewise leaves state=-1. Successful normal returns verify the stack and all eight Windows nonvolatile integer registers.

DelayedDeckIntro.Reset (38FA90), InitCoroutine.Reset (38FE50) and SetupDelay.Reset (391290) unconditionally resolve and allocate NotSupportedException, construct it, resolve their own exact Reset MethodInfo, and call the runtime throw helper. They do not reset iterator state or Current. Metadata resolution, exception construction and throwing are explicit gateways; no managed stack unwinding is claimed.

## Validation

**64 native runs pass**, covering all three factory receiver/null/allocation-failure cases, yield/resume/terminal sequences across signed BlindDeck inputs and cold/warm Gameplay, invalid states, before-publication failures and all three Reset methods. SHA-256 checks and exact metadata signatures bind the seven declarations; six native instruction relationships and the float constant are asserted. Range ends include complete final instructions and exclude trailing alignment padding.

Reproduce with `python scripts/audit_gameplay_iterator_factories.py GAME_ROOT DUMPER_ROOT --output REPORT`, using private Unicorn 2.1.4 through PYTHONPATH. Allocation, wait construction, Settings, class initialization, state transition, exception construction/throw, metadata resolution and barriers are bounded gateways. No live process or save data is used; no typed target set is added.

## Offline Rust replay

`solver_core::gameplay_iterator` exposes `replay_gameplay_iterator` under
`gameplay_iterator_native_v1`. Its strict context requires initialized metadata
and services that preserve iterator fields. Operations are the three factories,
DelayedDeckIntro MoveNext, and Reset for each generated iterator kind. Factory
results are separate from the supplied iterator snapshot; fresh opaque labels
represent allocations without inventing runtime pointers.

The result preserves Current on termination and service failure, records the
attempted state8 request even if its gateway fails, and distinguishes an absent
MoveNext return from false. A successful first yield exposes exact float32 bits
`0x3f800000`. It deliberately does not construct an engine wait-queue record:
producer clock, frame, owner and scheduler acquisition are outside this audit.
Reset returns the precise NotSupported outcome and retains iterator fields.
Unsupported guards or known allocation-reference aliases reject the request
atomically. The replay never mutates its input context.

The companion test module compares all 64 native report runs after normalizing
opaque identities, and checks strict fields/provenance, atomic rejection and
attempted state-change failure. This offline API is not connected to live play.
