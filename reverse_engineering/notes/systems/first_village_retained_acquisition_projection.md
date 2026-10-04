# Original N5 acquisition: independent Rust comparison

Build `f530404b0f3f_807de4a83df4`, Steam build `23084916`.

The decision blocker was whether the guarded Rust acquisition boundary could
consume the retained original five-card transaction after animation, while
preserving the complete queue and source-role identity semantics. The
[native witness](first_village_retained_acquisition.md) and
[guarded Rust support](non_day_setup_reveal.md) are separate prior checkpoints.
This comparison joins their semantic and queue boundaries without promoting
the native object graph to planner input.

## Evidence and reproduction

The independent [projector](../../scripts/project_first_village_retained_acquisition.py)
does not import or invoke Rust. It checks the frozen native source/report and
asset report hashes, decodes declared lossless snapshot pooling, and derives
the [synthetic fixture](../../fixtures/synthetic/first_village_retained_acquisition_v1.json)
from original actor/status/pool/iterator/queue fields and concrete call traces.
Constructor/Init projection helpers retain their existing scopes.

Frozen SHA-256 identities:

| Artifact | SHA-256 |
| --- | --- |
| Native source | `cdcc5ef15c2d36a1b961263c483b27c264e520a495f97985ad640d68df4c68cd` |
| Native report | `f74f10f6f9eccd5ae1af512871dc01e5c173e590290a414b59a11b74122ce9f3` |
| Projector | `da5c819004588eb2fce396602eca152e3703cf4d7201e8cd49f13465a32131eb` |
| Fixture | `e35322e0e415469613910ab464551b9ab5de7a3eec7b07476bf333d271c60e40` |

```powershell
python reverse_engineering/scripts/project_first_village_retained_acquisition.py `
  reverse_engineering/reports/f530404b0f3f_807de4a83df4_first_village_retained_acquisition.json `
  --assets reverse_engineering/reports/f530404b0f3f_807de4a83df4_character_assets_audit.json `
  --native-source reverse_engineering/scripts/audit_first_village_retained_acquisition.py `
  --output $env:DEMON_BLUFF_PROJECTION_OUTPUT
```

The configured output parent must already exist. Independent producer and
reviewer reproductions match the assigned 128,002-byte fixture exactly,
including physical Windows newline translation. All 18 native source and
11 consumed-report hashes remain unchanged. Reproduction establishes the
serialized checkpoint; it is not an independent gameplay probability model.

## Compared boundary

The regression first replays the existing setup-action V2 context and compares
complete original setup actors, bodies, pools, current data/order and pending
continuations. Acquisition begins with those same semantic actors and UI
contract, changing the explicit callback version from setup-only V4 to V5.
The fixture binds each native wait label `0..4` to its original iterator,
physical actor and modeled position. Initialization publications independently
check the display IDs `5..1`. This offline physical ordering is not a reviewed
public seat/orientation mapping.

The existing wait-queue kernel separately compares the three rejecting gate
drains and five animation drains, with actual input clocks/frames, before/after
queues, visits, callbacks, insertions and releases. Four re-yields allocate
labels `8..11` using exact promoted `.05f` duration, producer frame plus one and
current generation. Newly inserted earlier nodes are not visited in that drain.
All original `.3f` card waits and Audio/Shuffle waits remain retained.

The actual consumer release path establishes dispatch result 1 on each
animation callback. This is separate from managed MoveNext results
`[1,1,1,1,0]`. On re-yield the registered payload survives; a queue release
event does not assert physical payload deallocation. The Rust queue replay
receives these native-derived callback effects as explicit supplied responses;
it does not execute the complete animation/tween body.

The last animation output must equal the acquisition input: generation 8,
cursor 12, five card waits plus Audio 5 and Shuffle 7. One frame-13 acquisition
drain then produces generation 9 with cursor 12 and only waits `[5,7]`.
Complete final actors/bodies/pools/order, supplied UI flags, empty card registry,
deferred identities, callback order, visits, erasures and releases are compared.
Every callback has no Start and creates no replacement continuation.

All six positive conditional Minion outcomes remain in Rust before selecting
the recorded Confessor branch for native comparison. Under the supplied
uniform selector service, each of four duplicate outcomes has weight `1/10`
and each of two unique outcomes `3/10`. These are not original PRNG or joint
generation probabilities. The selected branch checks the duplicate occurrence
0, two selector draws and no script addition. Four Good source selectors retain
acquisition provenance but produce no selector draw or copied role.

Real/copied callback role, slot, dispatch and trigger order match concrete native
entries. The original Confessor source clone is distinct from the true actor's
runtime clone. Minion remains actually Evil while receiving appearance 25;
true Confessor's existing 25 is not inserted again. Status-version baselines
are bound to modeled initialization plus setup insertions, then acquisition
insertions are compared: Minion `1→2`, true Confessor `2→2`. Shared targets are
null. Hidden state, uses 1 and empty information history remain unchanged.

## Verification and limits

The focused retained-native comparison passed after a fresh release compile.
All 987 release library tests passed in 29.54 seconds. Source syntax, scoped
Rust formatting, differential/source review, exact reproduction and privacy
checks passed. The prior release build and all 34 simulation tests remain the
verification for unchanged production behavior; this tranche adds a fixture
and comparison test rather than a production rule change. It adds no new
mutation or outcome campaign beyond the prior copied-Alchemist mutation.

UI booleans are an explicit adapter service contract. Native name/sprite/color
routing, pixel visibility, complete animation execution, original lifecycle,
global Gameplay phase and input readiness are outside this Rust comparison.
Role triggers Init3/AfterRoundStart7, Hidden body5 and engine queue phase2 are
distinct. Audio/Shuffle remain unresumed. Original RNG, roles, corruption and
native storage are privileged oracle evidence, never player-history fields.

This closes one original-derived setup/acquisition differential dependency.
It does not certify S1 legal observations, an independent held-out corpus,
complete possible worlds, clue likelihoods, policy optimality or ascension
outcomes. All eight full-plan gates remain open. The next bounded dependency is
original deck/readiness callback provenance before one legal Hunter reveal and
its separate mixed-N5 player-history adapter.

## Independent six-outcome conditional support reference

The [Rust reference](../../../crates/solver-core/src/bluff/conditional_row_zero_setup_reference_tests.rs)
now independently derives all six outcomes for this one fixed input. Its input
reader omits the fixture's expected-output fields. Authored transitions build
initialization publications, setup state, acquisition callback states/traces,
final actor/body/pool/continuation state and complete queue chronology before
comparing them with the production replays. No production transition helper
constructs the reference expectations.

The input is explicitly the five physical rows above, with the four ordered
duplicate entries and two ordered unique entries. It preserves the actual
15-entry nonmatching Start array, source/clone classes, empty fresh resistance,
null target, status-version baseline and the supplied seven-entry acquisition
queue. The six canonical support keys retain complete modeled state and
current data. Status versions are projected from initialization and modeled
insertion effects; RevealActor does not store native list versions. Missing
bindings, inconsistent late queue/callback fields, unsupported selector entries
and an eligible deferred Audio or Shuffle callback reject without input mutation.

Confessor remains the only retained native full-history anchor. The other five
outcomes are independently authored conditional rule support, not five new
original histories. The comparison excludes conditional probabilities, native
generation priors, arbitrary placements, the full fallback catalogue, Unity
publication, global Day, legal player history and policy. Six equal modeled
support states do not establish six complete public belief worlds.

Both focused release tests pass, including the atomic rejection cases; all
1,002 release library tests pass in 18.93 seconds. The fixture is unchanged at
the hash above. These are development regressions, not held-out gameplay trials.
