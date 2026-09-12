# Unity JSON registration and ToJson engine boundary

Pinned build `f530404b0f3f_807de4a83df4`. This follows the managed JsonUtility
wrappers already described in [ascension helpers](ascension_helpers.md) into
the previously open registration and engine entry boundary. It does not
re-audit the managed type policy or reconstruct the serializer.

The [native harness](../../scripts/audit_unity_json_gateway.py) and
[report](../../reports/f530404b0f3f_807de4a83df4_unity_json_gateway_audit.json)
pin both GameAssembly and UnityPlayer. They record 31 native cases and 20
instruction assertions, with additional request, table and export checks.

## Exact request and distinct module registration

GameAssembly's ToJsonInternal wrapper requests
`UnityEngine.JsonUtility::ToJsonInternal(System.Object,System.Boolean)`.
Its resolver at `0x2B7DF0` calls the native internal-call map lookup at
`0x265BC0`. The map first seeks the complete request. When that is absent,
it finds the first opening parenthesis, makes the prefix before it, and
looks up that name. An existing complete-signature registration wins over
the shorter form. Failure to find either returns null from this lookup;
the resolver's subsequent error machinery remains outside execution scope.

JSON is registered by a separate two-entry module at UnityPlayer `0xFA1850`.
It is not in the previously audited 3,447-entry registration table. The new
module pairs function pointers at `0x18DEDD8` with names at `0x18DEDE8`:

| Registration name | UnityPlayer RVA |
| --- | --- |
| UnityEngine.JsonUtility::ToJsonInternal | `0x191D80` |
| UnityEngine.JsonUtility::FromJsonInternal | `0x192170` |

The module calls the same resolved `il2cpp_add_internal_call` sink used by
the earlier registration audit. The harness executes this module's two
registrations with an empty optional callback list. It also executes the
actual GameAssembly tree comparisons and signature fallback with synthetic
registrations: bare-name resolution, signature fallback, exact precedence,
and three missing-name variants. String construction, comparison, search,
and freeing are explicit services; the map traversal and fallback decisions
run as native code. This establishes the request-to-registration relation
without falsely requiring the signature-bearing request to equal the table
name byte for byte.

## ToJson native entry

The registered entry at `0x191D80` takes the object in RCX and the pretty-print
byte in DL. It first copies the object into two local managed-reference
slots using `il2cpp_gc_wbarrier_set_field`. The export identity is verified
from UnityPlayer's initialization site, not inferred from the calling shape.
The entry then checks the second local reference for null. A null reference
takes the exception-construction/raise route carrying parameter name `obj`.
That direct engine behavior differs from the public managed ToJson wrapper,
which returns the empty string before making the internal call for null.

For non-null input, the entry normalizes the pretty byte by testing it for
zero, initializes temporary native string storage, and calls `0xAACA50` with
the object, destination string storage, and normalized pretty flag. This
call is the serializer service boundary. There is no additional role-specific
or CharacterData-specific branch before it in this entry; that does not
establish which types the serializer supports.

After the service returns, the entry selects the native result bytes and
length from its string representation. On the inline representation, the
length is `24 - signed_tail_byte`; on the heap representation it loads the
stored pointer and 32-bit length. It calls the resolved
`il2cpp_string_new_len` export with those bytes and length, then writes the
returned managed reference through another GC barrier. Inline storage can
return immediately after register/stack restoration. Heap storage enters a
substantial allocator-dependent cleanup region before returning.

The initial unwind chunk ends at `0x191DE5`, in the middle of this method.
The audited decode continues from the verified entry through `0x192163`;
`0x192170` is the independently registered next JSON entry. Treating the first
unwind chunk's end as the method end would omit serialization, result creation,
cleanup and the null-input exception path.

## Executed scope

The 24 engine fixtures cross pretty bytes 0, 1, 2 and 255 with six outcomes:
empty inline string, short inline string, full 24-byte inline payload, heap
payload, serializer-service failure, and direct null input. The noncanonical
pretty bytes are synthetic ABI probes; normal managed bool callers supply
canonical values. Inline returns verify the managed result pointer, three
reference-store arguments, stack restoration and all eight nonvolatile
general-purpose registers. Heap fixtures verify pointer/length conversion and
managed result publication, then deliberately stop at `0x191E52` before
allocator cleanup. Failure fixtures stop at the explicit serializer boundary;
exception unwinding is not executed.

String results are supplied by the fixture service. These cases demonstrate
the caller's treatment of those results, not that the serializer emits those
bytes for a particular object. GC behavior, exception construction/raising,
native string allocation policy, field inclusion, supported-type checks,
reference resolution and actual JSON serialization remain open. FromJson's
engine body is only identified in the registration pair, not executed here.
No additional Assembly-CSharp managed method classification follows from
auditing these runtime/engine helpers.
