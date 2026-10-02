# DeckCharacter hover-channel registration

Build: `f530404b0f3f_807de4a83df4`.

Producer: `reverse_engineering/scripts/audit_deck_character_registration.py`.
Report: `reverse_engineering/reports/f530404b0f3f_807de4a83df4_deck_character_registration.json`.

This audit executes complete DeckCharacter.Init and OnDisable caller bodies.
Metadata, Action allocation and construction, Delegate.Combine/Remove,
reference barriers, and the whole downstream Character.InitReward remain
explicit supplied services. It reuses the frozen surface audit's immutable
build/declaration pins, complete native decoder, Unicorn mapping and volatile
register poisoning; it owns an independent diagnostic graph and corpus.

The report uses compact JSON and the lossless
`sha256-full-authored-snapshot-v1` encoding from
`reverse_engineering/scripts/audit_report_snapshots.py`. Its `pool_snapshots`
interns each full initial/final/service-entry snapshot by SHA-256;
`expand_snapshots` checks every blob hash and reconstructs independent full
snapshot values. The producer asserts exact expanded-value equality after all
native, semantic, retention and stopped-prefix checks. No fields, cases,
prefixes or diagnostic bytes are omitted. The `audit()` function continues to
return the full expanded report for independent in-memory verification.

## Exact family and bounds

| Stable method | Exact managed signature | RVA | Exclusive end | Bytes | Instructions |
| --- | --- | --- | --- | --- | --- |
| `tdi5514.m0001` | `public void Init(CharacterData data)` | `36FA20` | `36FBF8` | 472 | 123 |
| `tdi5514.m0002` | `private void OnDisable()` | `36FC00` | `36FDB5` | 437 | 110 |

Each method has one matching unwind range. The next verified managed entries
are `36FC00` and `36FDC0`; only verified `CC` alignment follows each body.
Pinned Dumper metadata gives Win64 signatures `viii` and `vii`, respectively.
Full PE raw backing, input hashes, exact signatures and complete decode lengths
are asserted. No shared constructor alias is promoted.

DeckCharacter is TypeDefIndex 5514: instance onClick `+20`, Character `+28`,
RevealCard `+30`, CardInteraction `+38`, and private CharacterData data `+40`.
CardInteraction is 5468: Action onHover `+60`, Action onHoverExit `+68`.
The surrounding CardInteraction fields are unrelated retained diagnostics;
neither method accesses an onClick event channel there.

System.Action is the exact sealed class 153, derived from MulticastDelegate
440. Its nominal TypeInfo comes from the decoded metadata slot. Delegate 419
declares invoke_impl `+18`, managed m_target `+20`, IntPtr method `+28`, and
IntPtr method_code `+40`; MulticastDelegate declares Delegate[] delegates `+78`.
Only the nominal class header is consumed by these caller cast checks.

Three metadata requests bind System.Action_TypeInfo and the exact
`Method$DeckCharacter.OnHover()` / `Method$DeckCharacter.OnHoverExit()` slots.
Their metadata MethodAddress values are independently checked against
`36FE10` / `36FDC0`. The per-method cold flags are `288C1E2` / `288C1E3`.
Cold requests set only the relevant flag after all three supplied metadata
calls finish; a failure at any request leaves it zero. Nonzero `FE` diagnostic
flags remain unchanged and skip requests.

## Capture, native stores and reload chronology

Both methods load current owner.interaction and guard null. They capture that
interaction identity and its current onHover Action before allocating a fresh
Action. Constructor entry receives the allocated identity in RCX, owner in RDX,
the OnHover metadata token in R8, and exact zero MethodInfo in R9. The metadata
token is the constructor's declared intptr_t method argument; it is not
substituted with a native code address. The supplied constructor's void return
poisons RAX; native RBX retains the allocation identity.

Init calls supplied Delegate.Combine; OnDisable calls supplied Delegate.Remove.
Both pass captured old channel in RCX, captured newly allocated Action in RDX,
and full-width zero MethodInfo in R8. R9 is not an argument and retains the
explicit volatile poison. Null results are stored as null. Nonnull results
must have the exact nominal System.Action class header; a foreign header
reaches the native cast-failure service before any channel store.

The accepted result is written to the first captured interaction at `+60`,
then passed to the reference barrier. A second exact-class check follows the
nonnull native store. There is no supplied callback or runtime call between
the first successful cast, store and second check, so the second failure branch
is excluded by the fixture's stable metadata and object-header contract.

After the first barrier, both bodies reload current owner.interaction rather
than reusing the prior captured identity. They guard that new pointer, capture
its current onHoverExit Action, allocate and construct the OnHoverExit Action,
and Combine/Remove it. The accepted result is stored into that second captured
interaction at `+68`, with the same exact-class checks. Allocation,
constructor and combination mutations demonstrate that later owner or channel
changes do not change either captured destination or captured old operand.

Init then writes the original captured input CharacterData into owner `+40`
and reaches its barrier. Nullable input data is passed through. It reloads
current Character `+28`, guards null, and tail calls whole supplied
Character.InitReward at `365640` with that current Character in RCX, the original
input data in RDX, and exact zero MethodInfo in R8. R9 is recorded only as
diagnostic bits. If Character is null, both channel operations and the owner
data store have already completed. A data-barrier mutation can replace current
owner data without changing the captured input subsequently passed to InitReward.

OnDisable ends by restoring the Win64 nonvolatile registers and tail calling
its second reference barrier. It leaves owner data and Character unchanged
except for explicitly authored service mutations.

## Honest supplied-service boundary

Supplied entries and exact signatures are pinned:

| RVA | Service | Consumed ABI |
| --- | --- | --- |
| `4D5170` | `System.Action..ctor` | allocated Action, managed owner, intptr_t metadata method token, zero MethodInfo |
| `116BCC0` | `System.Delegate.Combine` | old Delegate, new Delegate, zero MethodInfo; nullable pointer result |
| `116E070` | `System.Delegate.Remove` | old Delegate, new Delegate, zero MethodInfo; nullable pointer result |
| `365640` | `Character.InitReward` | current Character, captured input CharacterData, zero MethodInfo |

The allocation service authors a fresh cleared diagnostic buffer with a nominal
class header. The supplied constructor authors owner/method identity fields
and records an ordered invocation identity. Supplied Combine appends ordered
identities; supplied Remove removes the last matching sequence. Alternate
outputs independently supply null, old source, new Action, other Action, or
foreign nominal class. These algorithms and tokens are service contracts for
the caller audit, not evidence of the actual CLR constructor or delegate
implementation. Raw headers and buffers do not prove CLR admission, multicast
array shape, allocation size, event delivery or scene lifecycle validity.
No registered callback body is executed by this family.

Every supplied entry captures its full diagnostic state before any service
effect or authored mutation. Mutation timing is part of the stated synthetic
service contract. Callers execute their actual stores and reloads around those
boundaries; native callee behavior is not inferred from synthetic results.

## Corpus and retention

The corpus has 96 individual cases: 86 normal returns, six reached native null
guards and four reached nominal cast failures. It covers warm/cold metadata,
empty/null/prior/owned/duplicate/mixed invocation identities, nullable
interaction/Character/input data, aliased channel objects, and alternate
results for both operation channels. Duplicate identities exercise the supplied
last-match Remove contract without claiming a native delegate algorithm.

Authored mutations replace or clear owner interaction, Character, data or the
current interaction's channel. They occur during metadata, allocation,
construction, combination, barriers and InitReward. The allocation-entry oracle
independently captures current interaction and old channel before mutations;
actual constructor, combination and barrier arguments must match those
identities. Mutation at the first barrier demonstrates the required second
interaction reload. Clearing it stops before the second allocation.

Every native field-store PC is asserted and grants only its exact eight-byte
field range after being reached. Only reached authored mutations grant their
exact field ranges; supplied fresh-buffer construction grants that new buffer.
All other physical owner, interaction, Character, data, RevealCard, instance
Action, existing delegate, class and metadata-token bytes are retained exactly.
Metadata pointer slots are unchanged; the exact per-method flag expectation is
checked independently from completed initialization services.

Two retained six-call sequences execute Init, Init, OnDisable, OnDisable, Init,
OnDisable on one graph, with separate alias input variants. Allocations,
channels, invocation identities, metadata warming and request logs persist
between calls. Per-call channel ordinals remain independent of cumulative
event ordinals.

Six successful baselines produce 72 stopped service prefixes. They include
allocation-time and first-barrier interaction replacement for both methods.
Every stop occurs before the chosen supplied effect or mutation, preserves all
actual native stores already reached, and exactly equals both the successful
event prefix and corresponding full service-entry snapshot. No managed unwind,
continuation or post-failure service effect is invented.

Normal returns verify the stack, all eight integer nonvolatile registers and
all ten Win64 nonvolatile XMM registers. Supplied returns poison all volatile
integer and XMM registers. Full pointer, MethodInfo and nominal class widths
are asserted at each consumed boundary.

There are 31 selected exact instruction assertions plus complete decoded
call-site counts and the exact missing-instruction set. Across the two bodies,
213 of 233 instructions execute, across 222 total native/service addresses.
The remaining 20 instructions are explicitly listed in the report: ten terminal
`int3` traps after stopped native guards and ten setup/call instructions for
the four redundant post-store cast-failure stubs. The latter remain uncovered
under stable nominal metadata; the caller bodies are completely decoded and
those paths are not silently discarded.

## Remaining work

The shared DeckCharacter constructor remains unpromoted. Action construction,
Combine/Remove, Character.InitReward and runtime metadata/GC remain supplied
whole services. The surface audit covers the callback bodies independently;
this pair does not prove event registration admission, actual callback dispatch,
Unity rendering, scheduling or managed exception behavior.
