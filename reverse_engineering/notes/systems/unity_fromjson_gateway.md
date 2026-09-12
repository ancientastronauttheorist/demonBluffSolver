# Unity FromJson create and overwrite dispatch

Pinned build `f530404b0f3f_807de4a83df4`. The separate JSON module and
signature-fallback evidence are in [Unity JSON gateway](unity_json_gateway.md).
This audit follows its registered FromJsonInternal entry `0x192170`, the
immediate dispatcher `0xAACEE0`, and allocation/initialization helper
`0x75A3A0`. It adds no registration cases or Assembly-CSharp classifications.

The [harness](../../scripts/audit_unity_fromjson_gateway.py) and
[report](../../reports/f530404b0f3f_807de4a83df4_unity_fromjson_gateway_audit.json)
record **180 native cases**, 19 instruction assertions and five resolved
runtime-export identity checks. Parsing and field application remain services.

## Inputs and class selection

The engine entry takes JSON in RCX, an optional object to overwrite in RDX,
and a System.Type reference in R8. It records each argument into local
managed-reference slots using GC barriers. A non-null Type is converted with
the resolved `il2cpp_class_from_system_type` export. A null Type becomes a
null native class pointer; this wrapper does not itself reject it.

A non-null JSON reference is converted into native string storage. Null JSON
leaves that storage empty. The wrapper then calls the dispatcher with the
native text, overwrite object, requested class, output storage, and an error
descriptor. Direct native entry therefore has no public-wrapper-style
null/empty JSON return shortcut. The existing managed FromJson(string, Type)
shortcut remains valid and normally prevents those direct-native inputs.

The dispatcher first calls **il2cpp_class_get_rank on the requested class**.
Nonzero rank produces an array-specific error before parsing, allocation, or
overwrite-class selection. This is an array-rank test, not a general
UnityEngine.Object or abstract-type predicate. The error text says the return
type must represent an object type and identifies an array as the input.
The native entry passes null class to that runtime rank service when Type is
null; the fixture stops there rather than inventing the runtime's null policy.

If rank is zero and an overwrite object exists, the dispatcher obtains its
class through `il2cpp_object_get_class`. That **actual object class** is used
for field application, rather than the supplied requested class. The rank
test still occurs first on the supplied class. The audit varies these class
identities independently; it does not silently make them equal.

## Create, overwrite and error ordering

On the create path, the native dispatcher requests parsing at `0xAACC80`
before allocating any managed result. A null parse result without an error
message returns null. For a parse result, it calls `0x75A3A0` with the requested
class. That helper calls the verified `il2cpp_object_new` export, publishes its
returned pointer into the caller's output slot, and, if non-null, calls
`il2cpp_runtime_object_init_exception`. Successful initialization returns that
same pointer. The dispatcher then invokes field application at `0xA8E030` with
the parse result, new object, requested class and local auxiliary storage.

On the overwrite path, there is no managed allocation/initialization call.
Parsing precedes field application into the exact supplied object, using its
actual class. A null parse result without an error message preserves the
overwrite object as the result. Successful application is followed by parse
result destruction and freeing, then result publication.

The parser service can populate a native error string. The dispatcher checks
that error after its parse/create/overwrite branch. A nonempty error becomes
an exception descriptor and a null result slot; the outer entry cleans its
wrapper storage and forwards the descriptor to its raise service. Error
construction and raising are gateways, so the exact exception implementation
and unwinding are not inferred.

The constructor-error fixture verifies that the allocated pointer has already
been written into the helper's output slot before its exception-handling branch.
It stops at `0x75A3F6`: this audit does **not** establish whether that later
runtime reporting path throws, logs, or suppresses a constructor exception.
The [constructor-reporting follow-up](unity_constructor_reporting.md) now
executes that path: when its reporting services return, the allocation and
constructor mutations remain available for subsequent field application.
The field-application failure fixture writes a sentinel into the destination
and stops at the application service. At that boundary the mutation is present
and parse-result destruction has not run. This establishes ordering, not a
claim about managed exception cleanup or eventual rollback.

## Native execution and service limits

The 180 cases cross create/overwrite, null/empty/nonempty JSON references,
present/absent Type, ranks 0/1/2, and normal/no-parse-result/parser-error/
constructor-error/application-failure outcomes. Gateways assert exact object,
class, text, output-storage and error-storage arguments. Successful returns
verify result identity, stack restoration and all eight nonvolatile
general-purpose registers. Constructor and application failure cases check
the specific partial-write boundaries described above.

The native decode follows both methods beyond their initial unwind chunks:
FromJsonInternal's successful return is at `0x19259B`; the dispatcher's return
is at `0xAAD64A`. Fixtures execute inline string paths and the native diagnostic
cleanup path against explicit allocator services. They do not validate the
allocator's ownership implementation.

Managed/native string conversion and ownership, class queries, allocation,
runtime initialization, GC, parsing, field application, parse-result cleanup,
exception services and allocator policy remain boundaries. In particular,
supplied parser outcomes are fixtures, not predictions for the given JSON.
The audit does not recover field inclusion, nested construction, object-reference
resolution, arbitrary supported-class behavior or complete JSON semantics.
