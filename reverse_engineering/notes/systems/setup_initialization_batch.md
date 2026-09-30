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

Eight batch tests cover alias retention, caller failure prefixes, allocation and
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
