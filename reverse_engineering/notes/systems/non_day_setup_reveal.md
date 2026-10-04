# Bounded non-Day setup acquisition in Rust

Build: `f530404b0f3f_807de4a83df4`, Steam build `23084916`.

## Decision blocker and admitted boundary

The retained original first-village board has ordinary Minion, Confessor,
Lover, Hunter and Enlightened data. The setup-only Rust callback version V4
rejects every continuation resume. The older acquisition versions admit only
Scout, Witness and Confessor bluff assets, while the original Minion pools have
four duplicate occurrences (Confessor, Lover, Hunter, Enlightened) and two unique
occurrences (Gemcrafter, Alchemist). Dropping those positive branches would make
the modeled support incomplete.

`bluff::reveal::bounded_setup_reveal_callbacks_native_v5` reconstructs the
caller-proven acquisition and Init3/AfterRoundStart7 callbacks for these five
data classes. V4 remains setup-only. V5 requires exact data-to-real-action class
matching, runtime Evil for Minion alone, explicit Start-latch provenance, no
HealthyBluff30, no trigger subscribers, and no copied role on the four Good
classes. A live Minion bluff must have the matching copied callback class.
Destroyed/stale copied-role topologies and other source roles are rejected.

"Non-Day" names the dispatched role triggers Init3 and AfterRoundStart7,
rather than the Day30 role action. It does not certify the global Gameplay
phase. Hidden body state5 and the supplied queue dispatch phase mask are
distinct from that global phase; the original outer setup caller remains
outside this callback checkpoint.

V5 accepts every candidate in the six-role pool domain; any unsupported pool
entry rejects the whole invocation. Repeated occurrences and order remain
significant. Selector pools are not reconstructed from hidden state or guessed
from an apparent deck. The caller supplies their independently established
chronological provenance.

## Reused original evidence

The [executed original role setup](first_village_role_setup.md) and
[publication/Init](first_village_publication_init.md) retain the original five
source classes and their actual concrete setup callbacks. The
[bluff acquisition audit](gameplay_bluff_acquisition.md) establishes the ordinary
Minion selector, folded null selectors, GiveBluff and dispatch order. Additional
bounded role evidence is:

| Callback class | Relevant original behavior |
| --- | --- |
| Ordinary Minion | Real `Act 33ED50` returns without a gameplay writer. |
| [Confessor](../roles/gameplay_role_confessor.md) | Act `3D65D0` and BluffAct `3D6650` handle Init3 through OnInit `3D6A50`, attempting appearance25; trigger7 is a no-op. |
| [Lover](../roles/gameplay_role_lover.md), [Hunter](../roles/gameplay_roles_scout_hunter.md), [Enlightened](../roles/gameplay_role_enlightened.md), [Gemcrafter](../roles/gameplay_role_gemcrafter.md) | Shared Act `3B09F0` and BluffAct `3B33E0` handle Day30 only; Init3/AfterRoundStart7 return without gameplay effects. |
| [Alchemist](../roles/gameplay_role_alchemist.md) | BluffAct `3AFE90` has no Init branch and does not handle trigger7. Its real Act `3AFDD0` Init resistance hook is outside the admitted Minion copied-role dispatch. |

These are reused native execution and reviewed native-static dependencies.
The new authored Rust tests are not a newly executed retained native acquisition
corpus. In particular, this checkpoint does not assert a new retained execution
for copied Gemcrafter or Alchemist.

GiveBluff clones the selected asset's original source role, not another board
actor's real clone. Rust represents the admitted semantic class rather than
native object pointers or every clone field. Four Good source classes enter
the folded null selector: they clear registerAs, consume the explicit
continuation/acquisition ordinal, draw no RNG and install no copied role.

Actual lying and appearance remain separate. A copied Confessor adds25 to the
Evil Minion but does not make it truthful. An accepted repeated Confessor25
does not insert a second status, yet still clears the shared status target.
Copied Alchemist uses BluffAct because the Minion remains lying; neither Init3
nor AfterRoundStart7 adds Corrupted resistance, cures or runtime data.

## Complete mixed queue, with deferred callbacks guarded

`bluff::scheduled_reveal::scheduled_setup_reveal_native_v2` composes the same
logical-continuation registry and native one-shot queue kernel. Its complete
queue includes the card continuations plus exactly one explicitly typed Audio
and one Shuffle wait. The deferred IDs must be distinct from card continuation
IDs; the queue must be their exact union with a shared allocation cursor. Every
record retains mask0xA and a present release slot. The board must be Hidden5 and
unrevealed, and use V5 callbacks.

Queue timing, generation, signed frame thresholds and saved-successor traversal
use the [reviewed wait kernel](unity_wait_queue_projection.md). Caller-supplied
clock/phase snapshots, live-owner provenance and callback results remain
explicit. An entered card callback must return the witnessed release result1.
The queue and registry versions, typed deferred map, chronology and storage
bounds survive every output branch.

If either deferred wait becomes eligible, the whole drain is unsupported.
This includes a drain that already completed earlier card callbacks. The
adapter does not drop Audio/Shuffle, silently assume their effects are inert,
return successful siblings, or infer an original engine schedule. The original
DelayReveal-only scheduled V1 rejects deferred waits and the new V5 domain;
its empty deferred-map serialization remains unchanged.

## Probability, information and exclusions

The six output weights are conditional on the existing selector's supplied
uniform integer-service model: roll outcomes1..4 choose duplicate,5..10 choose
unique. For the original four/two pools with empty must-include, each duplicate
outcome has mass1/10 and each unique outcome3/10; each consumes two service draws.
The unique source list is preserved; only must-include selection removes the
first matching occurrence. Script registration is performed on the unique
branch and suppressed when already present.

Those fractions are not established original PRNG/seed, full-generation or
prior-history probabilities. This offline boundary receives privileged oracle
state. No live GameState/planner caller, legal observation, Day clue, readiness,
input gate, fair-history adapter, world-set agreement, held-out policy outcome
or S1 completion is promoted here. Animation resumes and native lifetime/object
graphs require their separate retained witness. Start/HealthyBluff, arbitrary
copied-role truth dispatch, trailer mode and other role classes remain excluded.

The exit checks are complete six-outcome support; null-selector chronology;
truth versus appearance; repeated-status target reset; copied Alchemist's
non-effects; preserved Audio/Shuffle queue records; and atomic rejection when
either deferred callback becomes eligible. Native retained differential
comparison is the next checkpoint, followed by actual Day/readiness and legal
public-observation admission.

## Verification checkpoint

Thirteen new authored regressions cover these boundaries. The focused bluff
suite passed422tests; the restored full library suite passed986tests in14.79s.
A deliberate mutation incorrectly routed copied Alchemist Init through the
Confessor status hook. Its focused regression failed with exit101. Restoration
was byte-exact (SHA-256
`e61171fb0652d7eabbd8c60005a3adae82a93d9d82acd418289e848396e694c5`), followed
by a confirmed fresh Cargo compile and the passing full suite. A backup copy's
older Windows timestamp had initially reused the mutant test binary; that
stale run is excluded from successful verification. The guide now requires
both restored bytes and fresh compilation.

The release build and all34release simulation tests passed; the simulation
suite took1392.17s. Local evidence links, scoped Rust formatting, privacy,
diff checks and the prior installed-subscriber script/note/report hashes passed.
These engineering results do not promote an independent held-out corpus,
original path weights, new native retained execution or legal player history.

The subsequent [retained-native comparison](first_village_retained_acquisition_projection.md)
matches the recorded original Confessor path and actual animation/acquisition
queue chronology through an independent fixture. Its separate scope preserves
the six positive Rust outcomes and leaves legal observations/readiness open.
