# Acted delayed coroutine caller and iterator

Pinned build `f530404b0f3f_807de4a83df4`. The audit executes four exact declarations:
Act(string,float)35DDC0, ActDelayed35DC70, generated MoveNext374FC0 and Reset3750F0.
The report contains124 native cases,169 executed instruction addresses and11
instruction assertions. GameAssembly/Dumper hashes, signatures, iterator/receiver
fields and five immediate gateway identities are verified. Successful returns
check stack, all eight nonvolatile integer registers and all128 bits of XMM6.

## Construction and publication

Act(string,float) calls StopAllCoroutines before metadata resolution or allocation.
After allocation and the shared native empty constructor, it publishes receiver
at iterator+28, state0 at +10, then its first barrier. It publishes description at
+30 and its second barrier, then copies the raw delay bits from XMM2 to +20.
Finally it tail-forwards receiver/iterator to StartCoroutine. Failure at the first
barrier leaves receiver/state published; failure at the second leaves description
published too, while delay still has the allocator's zero value. A StartCoroutine
failure sees the fully constructed iterator. The ActDelayed factory performs the
same construction but neither stops nor starts coroutines; it returns the iterator.

Null receiver and description captures are forwarded under the explicit service
contract. This does not establish real engine acceptance of a null MonoBehaviour.
Signed zero, subnormal, negative, infinity, quiet-NaN payload and signaling-NaN
payload bits pass unchanged through construction. No time conversion or duration
clamp occurs in these callers.

## MoveNext states

At state0, MoveNext copies delay bits, writes state-1 before allocating WaitForSeconds,
and supplies those bits in XMM1 to its constructor. It stores the wait object in
Current+18 before the barrier, then changes state to1 and returns true only after
that barrier succeeds. Wait allocation/constructor failure therefore preserves old
Current with state-1; Current-barrier failure retains the new Current with state-1.
There is no receiver-null check during this first yield.

At state1, the iterator writes state-1 before checking its captured receiver and
receiver.acted+20. It calls ActedVersion.Show35D920 with captured description.
It then reads receiver.layoutsToRebuild+28 and loops its array in occurrence order,
calling LayoutRebuilder.ForceRebuildLayoutImmediate1EC1010 for every element. The
array itself is checked for null; its elements are forwarded without a local null
or Unity-object comparison. Duplicate elements cause duplicate calls. Null receiver
or acted prevents Show; null array fails after Show. A failing rebuild retains all
earlier successful Show/rebuild effects in the fixture.

The array reference is captured after Show: a controlled Show service replacement
is observed. Replacement of the receiver's array field during a rebuild does not
redirect the captured array. An empty array invokes no rebuild. Runtime class
initialization is performed only when an element is reached. Successful completion
returns false with state-1 and retains the wait object in Current. Further MoveNext
calls return false without effects. States other than0/1 preserve all iterator
fields and return false. A null receiver can yield successfully, then fail on the
next MoveNext while retaining Current and the state-1 write.

Factory-to-three-resume sequences execute the real caller instructions together.
They verify raw WaitForSeconds constructor bits, one true return followed by two
false returns, retained Current and the Show/rebuild order. These are supplied
resume calls; no elapsed delay or engine scheduler behavior is inferred.

## Reset and service scope

Generated Reset resolves/allocates NotSupportedException, invokes its constructor,
resolves its own exact MethodInfo and calls the runtime throw helper. It does not
reset state or clear Current. Failure fixtures preserve the iterator fields at
each metadata/allocation/constructor/throw boundary.

Metadata/class initialization, zeroed allocation, GC barriers, Stop/StartCoroutine,
WaitForSeconds constructor, ActedVersion.Show, layout rebuild, exception construction
and throwing remain explicit services. Attempt snapshots and successful effect
prefixes are checked independently at every reached cold-baseline failure point.
The shared empty constructor executes its native ret. This audit does not execute
the UI bodies, infer Unity object lifetime, recover exception unwinding, or prove
real coroutine cancellation, ownership or timing. No live state, proprietary native
bytes or decompiled bodies are included.

Run `audit_acted_delayed.py GAME_ROOT DUMPER_ROOT --output REPORT` with private
Unicorn2.1.4 on PYTHONPATH. Python compilation and scoped diff checks pass.
