# Captured constructor errors during FromJson allocation

Pinned build `f530404b0f3f_807de4a83df4`. This closes the local constructor-error
handling branch left at `0x75A3F6` by [FromJson gateway](unity_fromjson_gateway.md).
It does not reconstruct constructors, serializers or the final logging backend.

The [harness](../../scripts/audit_unity_constructor_reporting.py) and
[report](../../reports/f530404b0f3f_807de4a83df4_unity_constructor_reporting_audit.json)
record **192 native cases and 25 instruction assertions**. Both pinned binaries
are emulated; the GameAssembly constructor-selection body is called from the
actual UnityPlayer allocation helper through the runtime export pointer.

## Return and reporting policy

The allocation helper first publishes the result of `il2cpp_object_new` into
its caller's output slot. A null result skips initialization. A non-null result
is passed to `il2cpp_runtime_object_init_exception` with local exception storage.
If that storage contains a captured initialization error, the helper:

1. Calls `UnityEngine.Debug.CallOverridenDebugHandler(exception, null)`.
2. If that returns false, calls `UnityEngine.Debug.IsLoggingEnabled()`.
3. If logging is enabled, calls the native fallback reporting entry `0x75B4C0`,
   with the exception descriptor, zero context/auxiliary inputs and flag true.
4. After returning reporting services, returns its original output-slot address.

The **published allocation remains in that slot**, even after a captured
constructor error. The helper has no local rethrow, object reset or replacement
on these returning branches. A constructor may already have changed the new
object before recording an error; the fixture mutation remains present. This
means the previously audited FromJson dispatcher can continue to its field
application service with that allocation after local reporting completes.

This statement is conditional on invoked reporting services returning. A
failure or nonlocal exit inside a reporting callback/backend is not guaranteed
to return. The fixture separately stops at the fallback-report service and
verifies that publication has already occurred at that boundary.

## Debug dispatch evidence

The method identities are recovered from the native metadata-cache initializer
starting at `0x81A880`. It resolves Debug.CallOverridenDebugHandler into cache
slot `+0x2F8` and Debug.IsLoggingEnabled into `+0x300`. The native wrappers at
`0x820F70` and `0x821050` consume those exact slots.

The harness executes those wrappers, their call-frame builder `0x75D3A0`, and
the Boolean invocation bridge `0x7C17F0`. The override wrapper packs exactly
the exception object and a null context; the enabled wrapper has no arguments.
On a normal managed callback return, the bridge reads the returned boxed
Boolean's payload. On a callback error it conditionally dispatches reporting
through `0x75BE00` and returns false. Thus an override callback error can lead
to the enabled check and fallback reporting; an enabled callback error yields
false and suppresses the outer fallback after its own reporting returns.
The managed Debug implementations themselves remain invocation services.

## Constructor selection and invocation interface

The pinned GameAssembly export `il2cpp_runtime_object_init_exception` at
`0x281DD0` jumps to `0x2E5DF0`. Native instructions obtain the allocated object's
class and request `.ctor` with zero parameters through the method-lookup service.
They inspect the selected method's declaring class. For a value type, the
receiver becomes the boxed object's payload at `object + 16`; for a reference
type it remains the object pointer.

The selected method, receiver, null argument vector, and caller-supplied
exception-output pointer are forwarded to runtime invocation at `0x2E5310`.
The audit statically checks that this routine clears a non-null exception slot
before invoking its lower dispatch helper. The fixture supplies invocation and
exception capture as a service at that interface. It does not emulate C++
exception unwinding or the managed constructor body, and it does not infer
behavior for a missing `.ctor` lookup.

## Validation limits

Cases cross null/non-null allocation, value/reference receivers, captured/no
constructor error, handled/unhandled override results, enabled/disabled logging,
errors in either managed Debug callback, and a returning/stopped fallback
reporter. Successful paths verify the returned output address, published
allocation identity, retained fixture mutation, stack restoration, all eight
nonvolatile general-purpose registers and XMM6 preservation.

The recovered local policy is reporting followed by continued return of the
allocation when services return. Final logger behavior, callback implementation,
constructor field semantics, runtime exception capture and nonlocal unwinding
remain explicit boundaries. No additional managed method coverage is claimed.
