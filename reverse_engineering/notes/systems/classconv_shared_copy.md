# ClassConv fully-shared copy alternates

Pinned build `f530404b0f3f_807de4a83df4`. This closes the caller-body gap left
by the [reference-type audit](ascension_helpers.md) for managed definitions
`tdi5681.m0000` (`CopyArrayIntoList`) and `tdi5681.m0001` (`CreateCopy`).
The new [harness](../../scripts/audit_classconv_shared_copy.py) and
[report](../../reports/f530404b0f3f_807de4a83df4_classconv_shared_copy_audit.json)
cover the distinct fully-shared bodies at `0x602C60` and `0x6032F0`.

## Actual calling convention

The Dumper signature for fully-shared CreateCopy describes a generic struct
return and two parameters. It is not a sufficient machine-level prototype.
Native instructions establish this ABI:

| Body | Native inputs | Result |
| --- | --- | --- |
| CopyArrayIntoList `0x602C60` | RCX source array; RDX MethodInfo | RAX result list |
| CreateCopy `0x6032F0` | RCX boxed input object; RDX caller result buffer; R8 MethodInfo | Exactly the generic size is copied into the caller buffer |

CreateCopy preserves the original RDX in R14 at `0x603304`, original RCX in
R15 at `0x60330B`, and original R8 in RDI at `0x603316`. It later copies the
temporary generic result into R14. A reference-width return is therefore also
written through a buffer in this body. RAX happens to retain the memory-copy
service return on the observed path; it is not the recovered managed value.
No typed manifest or Dumper signature has been rewritten based on this finding.

Both bodies use MethodInfo's `+0x38` generic context, initializing that context
when absent. Pinned Dumper header declarations independently identify the slots:

| Context | Slots in order |
| --- | --- |
| CopyArrayIntoList | List<T> class, list constructor MethodInfo, T[] class, T class, CreateCopy<T> MethodInfo, list Add MethodInfo |
| CreateCopy | JsonUtility.FromJson<T> MethodInfo, T class |

The array class slot exists in metadata but is not read by this body. The array's
own runtime class supplies its stride. Generic invocation forwards the selected
MethodInfo's code pointer at `+0x00` and calls its invoker at `+0x10`, retaining
that exact MethodInfo as context. The invoker gets RCX code, RDX MethodInfo, R8
target object (null for the static copies), R9 argument-vector address, and a
fifth stack argument. The copy/FromJson calls supply a generic result-buffer
pointer both in the second vector slot and as that fifth argument. Add supplies
its element representation in its vector and fifth argument. These observations
describe these call sites; they do not imply all IL2CPP invokers use identical
argument layouts.

## Generic storage and copy behavior

The copy size comes from T class `+0xFC` (Dumper `actualSize`). Array addressing
uses the separate array-class `+0x104` (`element_size`). Each temporary buffer
allocation rounds the copy size up to 16 bytes. CopyArrayIntoList reserves
three such buffers and initializes its per-element storage before allocating
and constructing the result list. It then checks the source array for null.
Empty arrays return the constructed empty list.

Each occurrence is copied from `array + 0x20 + index * stride` into local generic
storage using the generic size. That snapshot is boxed for a CharacterData
assignability check and possible characterName log. The same snapshot is boxed
again for CreateCopy. Consequently the log and JSON call receive separately
boxed values for non-null value types; reference boxing retains the reference.
The array caller does not pass a value buffer to the object-taking ToJson API.

CreateCopy calls compact ToJson with that boxed object, invokes the context's
FromJson<T> into temporary storage, and copies exactly the generic size into its
result destination. It performs no object-graph traversal or elementwise clone
of a struct's members itself. The array caller then selects Add's argument form
using bit 31 of T class `+0x28`: value types pass their result-storage address;
reference types pass the pointer loaded from that storage. Order, duplicates,
and null results are preserved. Every occurrence causes a separate round trip.

This supports the same managed reconstruction as the reference-type body:
create a List<T>, iterate the source, conditionally log CharacterData names,
and append `CreateCopy<T>(element)`, where CreateCopy is
`FromJson<T>(ToJson(objectInput))`. The generic runtime supplies boxing and
result storage; replacing these operations with pointer-sized loads is invalid
for the fully-shared body.

## Boxing branch evidence

The harness also executes the actual boxing trampoline `0x282580`, its runtime
body `0x2BFD80`, and the instance-size getter `0x2F1430`. Reference types return
the pointer loaded from the input storage. Ordinary value types allocate a box,
copy `instance_size - 16` payload bytes, and request a GC write barrier.

The native nullable branch requires non-null class `+0x60` and flag `0x08` at
class `+0x135`. A zero leading presence byte returns null. Otherwise boxing
uses the field-derived payload offset from `[class+0x80]+0x38`, adjusted by the
16-byte object header, and excludes that prefix from its payload copy. The
allocator's handling of the underlying nullable type remains a service boundary;
the audit does not substitute a claim that Nullable<T> boxes as a Nullable<T>
object. Synthetic nullable fixtures verify absent and present payload branches,
including the resulting null versus non-null JSON input.

## Validation and remaining boundary

The report records 132 native caller cases and 30 instruction assertions plus
two pinned context-layout checks. Cases cover reference storage, value widths
1/4/8/12/16/24/40/129, nullable shapes of widths 8 and 24, and array strides that
differ from generic copy size. They cover cold/warm contexts and list-class
initialization, empty/null arrays, repeated references, null elements/results,
CharacterData logging, direct output-buffer guard bytes, and JSON-service failure
after a previous list addition. Successful paths check stack restoration and all
eight nonvolatile general-purpose registers. Failure cases stop at the explicit
service boundary; exception unwinding is not claimed.

These are synthetic generic layouts, not a catalogue of real instantiated
classes. The harness executes both ClassConv bodies together, with a gateway
adapter that dispatches the copy invoker into the native hidden-buffer ABI.
Allocation, list methods, invoker adapters, initialization, memory primitives,
stack probing, GC barriers, and JSON services remain explicit boundaries.
It does not execute engine JSON parsing, supported-type policy, field inclusion,
reference resolution, or FromJson<T>'s type-specific invoker/unboxing behavior.
Arbitrary serializer-supported value types cannot be certified from caller
fixtures, nor can successful reconstruction of Unity assets be inferred.

With the prior reference-type evidence, the two managed definitions can honestly
move beyond **unresolved native body** to the repository's authored-source,
service-bounded reconstruction classification. This is not complete engine
serialization coverage, real-instantiation ABI certification, or a general
deep-clone guarantee. The explicit UnityPlayer JSON boundary must remain.
