# Guarded SavedGameInfo Rust replay

`bluff::saved_game_info` reconstructs the five pinned mutation and construction
callers independently of persistence. It consumes versioned
`saved_game_info_native_v1` contexts and retains the key, physical List records,
logical counts, complete backing slots and wrapping 32-bit versions.

The native source is the [five-method executable audit](saved_game_info_methods.md).
The Rust comparison uses all 124 normal standalone fixtures and ten native
caller/JSON joins. It compares complete List state at each reached non-metadata
service entry and at return, rather than only the final public values. Metadata
initialization events remain supplied and are not replayed.

For an add, Contains runs before mutation. A duplicate retains every field.
A new value increments the version before a supplied resize or an inline
append. Inline append writes the element and count before its barrier. A full
List requires an explicit verified capacity outcome; no growth algorithm is
inferred. Clear increments the version even when empty and sets count to zero
before clearing the old live slots. Unused backing slots are retained.

Construction publishes `Tutorials` before its first barrier, then allocates,
constructs and publishes two fresh Lists in order. Service-entry snapshots
distinguish allocated uninitialized Lists from constructed and published ones.
Previously allocated records remain retained, including those whose owner
fields were replaced. Ordered allocation identities are supplied explicitly.

The context requires verified runtime/metadata, inert initialization and
callbacks, canonical nullable-text membership, unaliased backing storage,
zeroing Array.Clear, empty-List constructor behavior and normal completion.
Physical List fields may reference the same retained List record; backing-array
alias effects and arbitrary managed String identity are outside the contract.
Text is bounded valid Rust UTF-8 with at most 4,096 bytes per value; capacity
and retained List records are limited to 256.
Aggregate input is also capped at 4,096 backing slots and 1 MiB of text,
including the key and appended argument, before snapshots clone that state.
Construction reserves the fixed replacement key's bytes before cloning too.
Supplied growth must fit the aggregate slot budget.
Null required receivers, invalid counts/identities, missing or incompatible service outcomes, unsupported
provenance and unknown serialized fields fail closed.

Focused tests additionally check nullable duplicates, version wrap, supplied
growth distinct from the authored native fixture's growth, retained unused
backing slots, constructor publication order and rejected provenance/shapes.
Native service failures and exception unwinding are not replayed. This API has
no live solver, persistence or scheduler caller, and does not transplant
private List versions or object identities across the JSON/storage emulators.
