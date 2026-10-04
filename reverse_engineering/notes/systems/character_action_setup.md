# Character setup action dispatch

Pinned build `f530404b0f3f_807de4a83df4`. The executable audit
`audit_character_action_setup.py` joins actual `Character.Act`3645C0,
`Character.RoleAct`368790 and `CharacterHelper.CheckLying`397750 for
Init3 and Start5. Its 370 fixtures execute 218 instruction addresses, check
11 exact instructions, pin metadata and field declarations, and check stack
plus all eight nonvolatile integer registers on successful returns.

Init does not inspect or change the Start latch. Start logs and invokes the
current onTrigger subscriber before inspecting that latch. A subscriber can
clear an already-set latch to permit dispatch or set a clear latch to suppress
it. Successful unlatched Start sets the latch before computing truth. Failures
later in truth calculation or role dispatch retain that write.

CheckLying executes its native precedence: live raw bluff or runtime Evil
alignment20 establishes lying, HealthyBluff30 clears that decision, and
Corrupted10 overrides it. The supplied Unity-liveness and status-membership
services are independent. Appearance statuses and data-role identities are not
used by this decision.

Act computes truth once for both action slots. A truthful real callback which
adds corruption still dispatches the copied callback through Act in the same
invocation. A lying real callback which clears corruption/raw bluff and changes
alignment still sends the copied callback through BluffAct. A runtime Evil
actor with a non-null copied-role pointer sends its real role through Act even
when lying; the copied role still uses the frozen lying route. These are raw
copied-role pointer checks and do not require live raw bluff data.

The real role is captured for its call, while the copied pointer is reloaded
after that call. Supplied callbacks can replace or clear that second target.
A real/copied alias is invoked twice, with its onActed delegate overwritten
for each invocation. RoleAct allocates its closure, captures actor and trigger,
allocates and constructs the delegate, then checks the role reference. Thus a
null real role fails only after those allocations. A non-null role receives the
new onActed reference before the GC barrier and before exact virtual Act or
BluffAct dispatch with the class's MethodInfo. Barrier failure preserves that
reference write. No onActed callback is invoked by the supplied role gateways.

The cross-product fixtures distinguish Init/Start, clear/set latch, ordinary,
Evil and other raw alignment values, raw-bluff liveness, independent Corrupted
and HealthyBluff membership, absent/copied/aliased role slots. Every reached
cold baseline gateway failure retains the same attempted event and state
prefix. Subscriber and first-role mutations are explicit supplied effects.

Role virtual bodies, allocation/delegate construction, runtime metadata/class
initialization, logging, status membership and Unity liveness remain gateways.
The native Character caller, dispatch adapter, truth predicate and folded
constructor return execute. This is neither a concrete role audit nor a claim
about managed exception unwinding, clue production, delegate invocation,
Unity scheduling or callback recursion. The following setup bridge reuses
previously audited concrete role behavior under its separate supported-class
contract rather than attributing that behavior to these role gateways.

## Supported producer-to-action bridge

`bluff::setup_action_bridge` consumes the explicit initialization producer and
projects its completed physical actors into the existing writer kernels. The
source asset identity, source-role class/cache, newly published action clone,
and retained copied-role pointer remain separate inputs. The exact managed
classes are Striga, Marionette, Drunk and Spy for data, with Scout, Witness and
Confessor additionally supported as stale copied action roles. One actual asset
per DataRole is required; Spy includes its explicit cache key. No public-name
substitution stands in for clone/source metadata.

All successful initializers finish before the Init-action enumeration. It
preserves repeated physical occurrences. Init applies supported Confessor
appearance status attempts separately from ordered Start and leaves the Start
latch unchanged. Current classes have otherwise inert Init bodies. Copied
roles remain callable even though ordinary initialization cleared raw bluff.

For each ordered asset, the bridge searches the then-current board identities.
The first match consumes that order entry even if its latch suppresses Start.
Twin writes can move an asset before a later ordered scan, reset a latch and
create additional continuations. Each replacement maps the existing writer
kernel's DataRole back to the unique actual asset identity. All-match ordered
classes (Alchemist, Poisoner and Puzzlemaster) are outside this bridge's supported
actor classes; their generic native caller behavior remains in the separate
ManageCharacters evidence.

The output is a continuation-registry state with no resume or readiness event.
Existing first-yield continuations retain their IDs, and Twin-created instances
receive new logical labels plus exact physical positions and creation trace.
These labels do not pretend to be newly recovered native handles. Explicit
post-initialization UI snapshots are carried and the existing bounded Twin
replacement UI writes are applied. Pool contents and Spy caches are supplied
verified state, not produced by this bridge.

Contexts require complete successful initialization, stable physical lists,
absent onTrigger/state subscribers, inert remaining setup services and verified
source/action classes, caches and UI. Null clones, unsupported classes, stale
or ambiguous asset mappings, non-first-yield continuations and missing state
reject the entire replay. Capacity checks bound retained joint output to 256
paths and 1,048,576 entries; invoked writer kernels retain their own working
budgets. This does not extend managed collection mutation, arbitrary callbacks,
concrete unsupported Start writers, RoleAct delegate identity persistence or
Unity scheduler behavior. The projected roles use only callbacks that do not
invoke onActed at these phases; native delegate writes remain native-audit
observations rather than reconstructed persistent delegate objects.

## Original N5 setup-only domain

`setup_action_bridge_native_v2` adds exact original data/class bindings:

| Public DataRole | Managed source/action class |
| --- | --- |
| Minion | Minion |
| Confessor | Confessor |
| Lover | Empath |
| Hunter | Tracker |
| Enlightened | Shugenja |

Its output embeds `bounded_setup_callbacks_native_v4`. This version accepts
canonical retained pools and explicit Start latches, and rejects every nonempty
resume vector. Legacy Reveal versions reject the new data and callback types,
including a copied new callback on an otherwise supported data actor. Shared
board/registry validation accepts the setup-only state; concrete Twin Start
still requires V3. No new acquisition, Day or mixed Twin-copying domain is
established by these enum additions.

`replay_init_prefix` projects the completed initializer state, runs the Init
action pass and validates retained continuation identities, then returns before
ordered Start. The existing full `replay` continues from that prefix through its
ordered Start scan. Thus an Init-only comparison needs no manufactured empty
Start array. Generic setup truth uses actual lying, with Confessor appearance
status kept separate. The [concrete role dependency](first_village_role_setup.md)
verifies Confessor's unique/resisted status application and the other selected
roles' no-effect Init/Start branches. Separate Start probes preserve status and
exercise the one-shot latch; they do not claim original ordered scheduling.

This semantic output does not carry native status-list versions, onActed
delegate/closure identities or physical publication-list storage. Explicit UI
snapshots and native-slot labels remain adapter inputs, not rendered captures.
The [retained publication/Init witness](first_village_publication_init.md)
executes the same generated five-actor caller and all 69 postpublication
service-prefix stops. Its report SHA-256 is
`31929f003a87c6e0ae4242ea9dc3eb090fba00bb641eedad0526c1afc651b48e`.
The [independent Python projection](../../scripts/project_first_village_publication_init.py)
produces [one semantic checkpoint](../../fixtures/synthetic/first_village_publication_init_v1.json),
SHA-256 `d6c639806e6d2afda5e22853ca6906eeb067dca7a82e414fe6c9df844b2f0326`.
It imports no Rust; another reviewer reproduced the fixture bytes exactly.

The Rust test compares all five represented actors, bodies and Init call traces,
current data/order, canonical pools and five pending identities. Confessor's
native active-list version advances from Init-produced 1 to 2; the other four
remain at 1. A separate check derives that delta from initializer version plus
inserted status effects, rather than claiming the semantic Path stores physical
versions. All ten setup bridge tests pass, including legacy/direct/copied type
guards, absent latches, noncanonical pools, acquisition resumes, and separate
Start/repeated-Start behavior. This evidence cannot promote legal player
observations, acquisition chronology or world sets by itself.

Final validation passed all 970 library tests (16.44s) and the release build.
Changing only Confessor's Init insertion from status 25 to 26 made the native
semantic checkpoint test fail with test exit 101. The mutation helper restored
the exact original source bytes before the final library run. This establishes
sensitivity to that represented status rule; it does not validate unrepresented
delegate identities, physical publication or acquisition scheduling.

## Original ordered Start and first-wait admission

The [retained Start/queue audit](first_village_start_queue.md) continues the same
original generated N5 caller through the actual serialized 15-entry Start
array. All 75 pointer comparisons are false for these five assets, so no Start
body executes. Twenty exact data bindings and 22 class aliases remain distinct
from the five supported semantic actor roles. This witness does not establish
concrete Start support for the other 15 assets.

Before each original Init returns, the audit executes the actual engine first
step and registers that actor's original state-zero DelayReveal iterator. Five
distinct supplied native owners share one queue. It retains the actor, payload,
GC target, iterator and live tree-node identity, rather than restarting an
already-yielded iterator. The native report SHA-256 is
`7439fd1370909aa79bf2575b101ff8e943941fa148bb75e15254f822a14b3a60`.

The [independent projector](../../scripts/project_first_village_start_queue.py)
produces [the semantic checkpoint](../../fixtures/synthetic/first_village_start_queue_v1.json),
SHA-256 `8c30de632c902477f14d28d84640903143ff0ee3a1f69d8015309d8c977324e9`.
It imports no Rust. The Rust comparison covers represented actors, bodies,
Init calls, data/order, pools, pending identities and all 75 ordered comparisons.
Status-list version checks derive only insertion deltas, as above. Logical
queue labels are adapter identities; physical owners, tree storage, callback
addresses, native CPU and GC references remain native validation evidence.

For each of five admissions, Rust derives the native timing from the explicit
producer: float32 duration bits `3E99999A`, time 1.0, signed frame 7, phase mask
`A` and generation 0 produce deadline `1.30000001192092896` and frame threshold
8. These match the decoded live native records. Changing the Rust increment
from 1 to 0 caused the checkpoint to fail with frame 7 versus 8 and test exit
101; the helper restored the exact source bytes afterward.

A synthetic predeadline scheduling probe preserves the five pending entries.
An eligible probe supplies all five matching callback boundaries and still
rejects the unsupported V4 acquisition resume. These probes establish adapter
compatibility and its guard, not a native drain history. The native stop is
before the runtime `onSetup` read at `36D2DB`; its subscription, Shuffle,
acquisition, public observations and PlayerHistory remain unresolved. The
generic caller's supplied later callback/Shuffle bookkeeping does not certify
that the original runtime subscriber is absent.

All eleven focused bridge tests passed. After adding the direct version-delta
comparison and restoring the frame mutation, all 971 library tests passed
(51.66s). Independent native reproduction matched the complete report bytes;
projection, source hashes, retained-prefix checks, links and privacy checks also
passed. Production deduction/action rules were unchanged in this integration.
