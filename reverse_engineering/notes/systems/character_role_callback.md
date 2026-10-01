# Role callback to delayed-result first publication

Pinned build `f530404b0f3f_807de4a83df4`. This separate native composition executes
`Character.RoleAct` at `0x368790`, its installed closure callback at `0x377120`,
`ShowActedDelayed` at `0x368A50`, and the iterator's first `MoveNext` at
`0x375FE0`. The report has 185 cases, 47 exact stopped-service prefixes and 175
executed instruction/service addresses. It adds executable evidence for the
previously unclassified callback `tdi5481.m0001`; common coverage files are not
changed by this audit.

A separate process reproduced the report exactly. Python compilation and
`git diff --check` passed.

The [setup action audit](character_action_setup.md) supplies role virtual calls
without invoking `onActed`. This composition supplies a role gateway which
explicitly invokes the newly installed callback zero, one or two times. The
delegate constructor and invocation adapter remain authored services; the
callback body itself runs. No concrete role's clue generation is inferred from
that gateway.

`RoleAct` allocates a closure, publishes its actor reference before the barrier,
then stores the low-DWORD trigger. It allocates and constructs a delegate before
checking the role pointer. A nonnull role gets its `onActed` reference before
the barrier and virtual dispatch. A zero `eCase` selects Act; every nonzero
low-DWORD value selects BluffAct. Exact virtual function/context slots are
verified against the native call sites.

The callback reads its own captured actor and trigger. It checks the actor
before constructing an iterator, forwards the supplied info reference, and
sets delay to the exact positive-zero float bits by `xorps xmm1,xmm1`. It then
registers the returned iterator through `StartCoroutine`. It does not reread the
role's current `onActed` field. An explicit sequence installs an Init-trigger
delegate, overwrites the role field with a Start-trigger delegate, then invokes
both retained callbacks. Their constructed iterators retain triggers `[3,5]`
and distinct identities.

The actual factory writes actor/state before its first GC barrier, then writes
info and delay before its second barrier. The trigger write occurs after that
second barrier. Stopped snapshots therefore distinguish a fully published
capture from an iterator which still has its allocator-supplied zero trigger.
Repeated callback invocations retain the same closure/delegate while producing
new iterators and wait objects in invocation order.

The supplied registration service explicitly performs one immediate native
`MoveNext`. This is a chosen service contract, not a scheduler reconstruction.
The first step changes state to `-1` before wait allocation, forwards the captured
zero delay to the supplied WaitForSeconds constructor, publishes Current before
its barrier, then changes state to `1` and returns true. Each successful fixture
retains the same info/trigger/owner capture through this first yield. Null info
is accepted at this stage. Null actor fails in the callback before construction;
a supplied role gateway that does not invoke that callback can return with the
null capture retained. Null role fails after closure/delegate construction.

The two cold baselines stop at every reached metadata, allocation, constructor,
barrier, role gateway and registration service occurrence. Every attempted
event and visible managed-state snapshot equals its baseline prefix, including
second-callback failures after the first iterator has reached state one. Native
null guards stop at the runtime failure boundary. Effects are not rolled back;
native exception unwinding is not simulated.

GameAssembly, Dumper script and dump declarations are hash-pinned. Four exact
managed targets, complete unwind families, exact declaring-type fields and 18
instruction assertions are verified. Normal outer calls and retained-callback
invocations restore the stack, all eight general nonvolatile registers and all
128 bits of XMM6. The authored nested-call adapters provide Windows shadow space
and stack alignment, and check their return stack before restoring the supplied
gateway's frame. Actor and info fixture buffers remain unchanged.

Metadata, zeroed allocation, GC barriers, delegate construction/invocation, role
virtual effects, WaitForSeconds construction and coroutine registration/resume
remain explicit services. Physical Unity ownership, actual delegate internals,
role results, second resume/history publication, UI effects, cancellation,
wait scheduling and readiness/order provenance remain open. This includes a
zero-duration delayed-result wait; it does not claim the existing DelayReveal-only
queue replay supports mixed coroutine kinds.

```powershell
$env:PYTHONPATH='B:/CodexTools/DemonBluffReverseEngineering/python-emulation'
python reverse_engineering/scripts/audit_character_role_callback.py GAME_ROOT DUMPER_ROOT --output REPORT
```

Report: `f530404b0f3f_807de4a83df4_character_role_callback.json`. The report
contains synthetic fixtures and native evidence facts; native bodies/bytes stay
private. No Rust changes or live game access are required.
