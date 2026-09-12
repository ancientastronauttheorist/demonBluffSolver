# Round-pool to acquisition frontier

Build `f530404b0f3f_807de4a83df4`. This is an integration handoff, not evidence that
startup, pool construction and delayed Reveal form one completed replay.

## Existing boundaries and read orders

| Boundary | Established order and contract | Remaining join |
|---|---|---|
| [Startup](gameplay_startup_composition.md), `gameplay_startup.rs` | Init37DEF0 calls ResetSavedCharacters37FDE0: typed10/20/30/100 results become saved copies, then current copies; Restart37FFC0 copies existing saved lists. Real alias fixtures preserve clear/copy identity and failure prefixes. | Typed GameData providers, LoadCharacters37E240, state/event callbacks and scheduling remain supplied. LoadCharacters can append unlocked occurrences after the initial copies; these copies alone are not the final round script. |
| [Starting selection](ascension_starting_sequence.md), `bluff/ascension_starting.rs` | AllLazy reads10/20/30/100; a null custom payload overwrites an inline choice and permits later faction reselection. Both consumed draws remain in chronology. | Carry actual profile identity and cache writes across callers; do not assume every getter shares one frozen result. |
| [Starting/roster bridge](roster_starting_bridge.md), `bluff/roster_starting_bridge.rs` | GetNotInDeck37C8E0 invokes actual AscensionsData typed getter3B1E10 in100/20/30/10 order, appending between calls, then removes one first-equal data occurrence per physical board entry. The profile graph is reread per faction; the kernel requires it to remain stable. | This is not the unique-pool source getter. Board exclusion is different from removing every current-script match. Preserve nullable assets, physical-null failures and the existing board snapshot. |
| [Unique pool](round_bluffs.md), `bluff/round_bluffs.rs` | PickRoundBluffs36D3A0 gets starting occurrences through GetAscensionAllStartingCharacters37C3F0, then script through GetScriptCharacters37DC00. Capture/barrier precedes UniquePool clear; predicate construction/RemoveAll follow clear. Samples up to4 Villagers and1 Outcast, then one fallback when count<=1. | The getters, filters and RemoveAll remain explicit services in this kernel. The separately executed predicate377170 forwards Contains; it does not recover comparer/RemoveAll internals. |
| [Duplicate composition](round_candidate_composition.md), [Rust contract](round_candidate_replay.md) | PickRoundDuplicates36D720 executes actual GetScriptCharacters37DC00 and Bluffable36A550, RealType36B9C0 twice, Alignment369EB0. Script order is10/20/30/100. The discarded Good filter still executes before sampling. | This composition starts from supplied current rosters; it does not execute ManageCharacters or share state with the unique replay automatically. |

The unique first getter's recorded source order is100/20/30/10. Its fallback
GetAllAscensionCharacters37C1A0 instead concatenates10/20/30/**10 again**, omitting
Demons; it then filters bluffable, Good, real-Villager. Duplicate Townsfolk source
weight must survive that join. The fallback does not subtract script again and
can reintroduce an in-script identity. These getter details are documented in
[acquisition evidence](gameplay_bluff_acquisition.md); execute the exact getters
in a future unique composition rather than substituting AllLazy or GetNotInDeck.

## Missing setup-to-pool transaction

[Lifecycle evidence](gameplay_lifecycle.md) records SetupDelay.MoveNext390AB0
clearing the picked-script cache, current rosters and BluffMustInclude, then
running count/script selection, relic callbacks, starting and random roster
selection, possible trailer replacement and later Characters.Init. Consequently
startup's saved/current copies cannot simply be piped into pool builders without
these intervening writes and callbacks. A bounded composition needs explicit
mode/profile stability, the cache-reset point, current-roster mutations, callback
results and every stopped partial prefix.

ManageCharacters36CE30 records: update positions; PickRoundBluffs; PickRoundDuplicates;
ordinary per-card Character.Init with descending displayed IDs; publish the shallow
board copy; per-card Act(Init); configured ordered Act(Start); onSetup; shuffle.
Ordinary ordered-Start roles stop after their first current-data match; Alchemist,
Poisoner and Puzzlemaster/Plague Doctor scan all matching occurrences. Preserve
that exception when producing writer/continuation state. Pool construction
therefore precedes these per-card initializations and ordered writers. Its call order must be executed or supplied with provenance, including
failure of the unique builder preventing the duplicate builder and later setup.
Shared pool/script identities must be mapped explicitly: the current kernels
construct distinct intermediate lists and do not prove arbitrary alias/reentrancy.

The immediate next native join is ManageCharacters through the two builders with
preserving creation/layout/callback services, followed by actual unique source
getters and its filter/predicate service transition. Keep existing duplicate
composition as the tested subordinate boundary. The next Rust join should retain
one combined RNG/callback/state history and a combined support/retention budget;
never recompute a prior lazy selection or rebuild a round pool after acquisition.

## Pool identity to ledger and Reveal

The new pool kernels use nullable u16 asset identities and expose failure paths.
[Selector ledger](../../../crates/solver-core/src/bluff/ledger.rs) instead accepts
canonical role strings representing one live current-build asset per role,
ordered unique/duplicate/must-include pools, faction script lists and independently
ordered successful selector events. A bridge must verify the identity mapping,
reject null/unsupported or distinct same-name assets, preserve repeated occurrences,
and supply the actual BluffMustInclude state. Neither builder initializes that
ledger state by itself. A native setup failure cannot be silently discarded to
feed the successful-selector kernel; retain its unconditional failure mass or
reject the full unsupported composition.

Pool-construction RNG is separate in time from later acquisition RNG. Preserve
inline/custom script draws, both builders' occurrence draws and fallback draws,
then actual selector draws in resume order. Minion's branch roll and discarded
must-include probe also consume RNG. Ledger probabilities marginalize some probe
outcomes and are not a seed/state transcript. Uniform weighted support can compose
only under the declared service contracts; it does not recover Unity's shared RNG
state, intervening unrelated draws or a prior-history probability.

[Reveal](../../../crates/solver-core/src/bluff/reveal.rs) additionally needs physical
actor state, current data/cloned/copied roles, statuses, raw bluff liveness,
register-as/cache provenance and explicit resume/acquisition ordinals. Ordered
Start writers can transform cards and register sibling continuations after pool
construction. A live earlier bluff suppresses a later sibling selector, but
register-as still runs first. Board order is not an acquisition schedule.

The existing [continuation registry](../../../crates/solver-core/src/bluff/continuation_registry.rs),
[scheduled replay](../../../crates/solver-core/src/bluff/scheduled_reveal.rs) and
[clocked replay](../../../crates/solver-core/src/bluff/clocked_reveal.rs) already join
supported writers, complete supplied DelayReveal-only queues and explicit clock
transitions. Their requirements remain: matching live owners, complete pending
records, supported release/callback behavior, actual producer/frame/phase facts
and stable callback clocks. The [native coroutine bridge](unity_coroutine_bridge.md)
proves the synchronous first step; older purely static scheduling uncertainty is
not a reason to undo that evidence. What is missing is production of this exact
initial actor/continuation/queue state from ManageCharacters and all intervening
supported writers. Do not replace it with board order, equal-delay registration
order, public reveal order or inferred OS time. Engine allocation/lifetime, scene
callbacks, unsupported writers, general PlayerLoop/queue contents, List internals
and unsupplied RNG state remain explicit boundaries.

The [ManageCharacters prefix audit](manage_pool_prefix.md) now verifies caller handoff and first Init arguments across 44 native cases. Its pool builders remain supplied gateways, and it stops before Init or empty-board publication; the complete composition above remains open.
