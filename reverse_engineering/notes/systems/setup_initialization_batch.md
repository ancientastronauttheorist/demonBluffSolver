# Setup caller to actor and continuation production

`bluff::setup_initialization_batch` consumes the successful Init prefix of the
complete ManageCharacters caller and executes the bounded initializer replay for
each occurrence. The version is `setup_initialization_batch_native_v1`.
Character aliases, physical positions, CharacterData identities, source Role
classes and clone/wait/iterator allocation bindings are explicit inputs.
Displayed card IDs are not physical positions or continuation order.

Repeated occurrences of one physical actor produce distinct retained first-yield
continuations. The last initializer overwrites that actor's shared data and role
clone; earlier iterator objects remain unchanged. Source roles, action clones,
retained copied roles and raw bluff liveness are represented separately. Existing
destroyed raw pointers are distinct from absent pointers until ordinary Init
clears raw bluff. Null clone results remain supported in this generic producer,
but the typed action bridge rejects them.

Only successfully completed caller Init gateways produce initializer effects.
An injected failed attempt produces none in this composition: native initializer
failure prefixes remain the separate audit's evidence. Later caller failures can
retain the completed producer prefix. `initialization_complete` means the caller
reached publication, not that publication or any action pass succeeded.

Allocation identities must be fresh against all represented retained objects.
Existing pending iterators cannot alias known actors, data, source roles or
incompatible wait objects, including in a zero-Init prefix. Valid shared prior
wait references and external owners remain allowed. Actor/position mappings are
bijective, and each successful occurrence has exactly one explicit allocation
binding. Unsupported callbacks, null dependencies and ambiguous provenance
reject the entire join. Registration is not readiness or resume order.

Nine batch tests cover alias retention, caller failure prefixes, allocation and
type collisions, explicit clone provenance and capacity guards. Six initializer
tests validate the underlying primitive, including retained identity checks.
The separate native sequence audit executes four two-call sequences and four
second-body calls: Init/Init, Init/NoReset, NoReset/Init and NoReset/NoReset. Actor
memory survives between calls, earlier iterator bytes remain unchanged, and a
different physical body's bytes are preserved. Its twelve real initializer
calls execute 324 instruction addresses.

Evidence: [initializer and first yield](character_initialization.md),
[complete supplied setup caller](manage_setup_caller.md),
`f530404b0f3f_807de4a83df4_character_init_sequence.json`, and the authored Rust
producer tests. [Setup action dispatch](character_action_setup.md) consumes the
supported completed producer state. Actual engine queue registration, scheduler
admission, subsequent Reveal and general scene callbacks remain open.

## Original generated five-actor prefix comparison

The [retained original N5 audit](first_village_initialization.md) executes actual
generation row zero and pool row zero, five native constructors, and five Init
returns in one live Manage caller. Native report SHA-256 is
`855f8976081da9541161bcdf15e0a130cbd6f0115f2351c027708b6c3fb08eb1`.
Its source-owned expectations are projected by
[the independent Python adapter](../../scripts/project_first_village_initialization.py)
to [five offline contexts](../../fixtures/synthetic/first_village_initialization_v1.json).
The adapter imports no Rust. Fixture SHA-256 is
`0efc7e616901c64415101721266e9c0e471c946aa5bd10676806b1d82ad2d223`.

The Rust comparison checks all 25 complete represented Actor snapshots and 15
retained Continuation snapshots across the five successful return prefixes.
Initial actor:0 retains its supplied Minion data through native construction;
the other four data references are null before Init. Exact generated data,
source/clone identity, status/history versions, one use, Hidden state, descending
IDs and earlier waits survive the comparison. Positions are explicit native-slot
labels, not certified UI positions or displayed IDs.

Synthetic Init occurrence failures delimit the first four completed prefixes.
A synthetic Publish failure selects the fifth, before any publication effect.
Those selectors do not represent original native exceptions or replay failed
service partial mutations. The Rust caller reaches its modeled Publish gateway;
the native caller stops earlier at instruction `0x36D01E` before board reload.
Only Actor/Continuation projections are compared, not full caller equivalence.

Physical list identities/backings, constructor-produced act/text state, wait
duration, nonvolatile CPU/stack state and retained pools remain native-only
assertions. This adapter does not execute constructors, collection providers,
engine scheduling, role Init, pixels or public-history admission. The separate
[concrete role audit](first_village_role_setup.md) supplies post-Init state and
verifies Confessor's status-25 effect; it is not a continuation of this witness.
