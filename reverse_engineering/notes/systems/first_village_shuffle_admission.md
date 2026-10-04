# Original N5 Manage return and Shuffle queue admission

Build: `f530404b0f3f_807de4a83df4`.

The decision is whether the same original row-zero generation/pool witness
can return from Manage with its five initialized actors unchanged and its
sixth, manager-owned Shuffle wait admitted to the retained engine queue.
The missing runtime rule is the `Characters.onSetup` binding. This audit
supplies that field as null explicitly; it does not establish that the
original scene's delegate was absent at runtime.

The [audit](../../scripts/audit_first_village_shuffle_admission.py) extends
the frozen [Start/queue join](first_village_start_queue.md) through normal
Manage return. Its assigned
[report](../../reports/f530404b0f3f_807de4a83df4_first_village_shuffle_admission.json)
uses schema `first_village_shuffle_admission_v1`. Complete native bytes and
exports remain private. CLI serialization uses the existing lossless snapshot
codec only after expanded assertions and JSON roundtrip checks; `audit()`
returns the expanded report.

## Conditional original witness

The same original generation row 0 and pool row 0 return, in physical scene
order, Minion 21596, Confessor 21614, Lover 21626, Hunter 21621 and Enlightened
21618. Actual constructors, Manage/pool builders, five primitive Init calls,
publication, five concrete Init dispatches and the original 15-entry no-match
Start scan execute. Display IDs remain 5 through 1. Source assets, distinct
runtime clones, statuses, histories, published shallow-copy board and the five
original DelayReveal state-1 waits are retained; actors are not prepared again
after Init or after the ordered scan.

The original Characters object is fully reparsed and matched to the preceding
report, including its exact MonoScript binding and the serialized Start array.
An ordinary managed `Action` field is not present in those serialized custom
fields. Reviewed [Characters lifecycle](characters_lifecycle.md) bodies do not
install it: Awake publishes the singleton, OnEnable/OnDisable hide pools and
the constructor leaves onSetup unchanged. A separate binding investigation
identifies `CharacterShuffleAnimation.OnEnable` (`363500`) and `OnDisable`
(`363160`) as the installer/clearer for its `Animates` callback. Those lifecycle
invocations and that subscriber are not executed in this audit. The null
binding is therefore conditional despite the original scene component.
The new adapter writes owner `+0x58 = 0` as an explicit
pre-entry conditional provider and records that contract in every snapshot.

Scene object/status/UI hydration, CLR allocation and collections, role cloning,
managed interface lookup and native coroutine-record creation remain supplied
providers inherited from the prior join. The six physical native owners and
their keys are supplied bindings, not live Unity identities or pixels.

## Retained native tail

The same live Manage invocation executes the onSetup load at `36D2DB`. The
null branch avoids the delegate call at `36D2ED`. That call is pinned as an
excluded path rather than serviced by a synthetic subscriber. Manage then
allocates the exact Shuffle iterator, calls the folded native no-op at
`33ED50`, writes state 0 at `36D325`, and passes the original Characters
receiver to StartCoroutine at `36D335`.

The iterator metadata declaration names constructor `357700`; Manage's decoded
callsite instead uses `33ED50` and writes its state itself. These are separate
facts. `Characters.<ShuffleDeck>d__16` has only state `+0x10` and current
`+0x18`, with no captured owner. The physical coroutine owner comes from the
actual StartCoroutine receiver.

The supplied synchronous bridge preserves the whole recorded outer CPU and
executes actual `SetupCoroutine.InvokeMoveNext` inside actual engine dispatch.
The interface slot provider selects actual Shuffle MoveNext `376B00` for this
new iterator; it never selects one of the five already-yielded acquisition
iterators. State 0 creates a native WaitForSeconds request using the verified
literal at `1F34B14`, loaded at `376B68`: float bits `3F000000`, exactly `0.5`.
Its constructor and current-field write barrier are supplied services with
exact arguments. Native MoveNext stores state 1 at `376B94` and returns true.

`shuffle_first_yield.entry_cpu` records the suspended outer caller before
seeding the bridge's stack and arguments; `return_cpu` records the nested
bridge return. Raw service-entry CPU belongs to the separately labelled
paused prefix records. These fields do not name a direct MoveNext entry ABI.

The current-yield provider mirrors that actual retained wait into the engine.
Reading iterator current `+0x18` and constructing the mirror remain supplied;
the folded managed Current getter at `353580` is not executed by this adapter.
Actual type dispatch, wait-record production, red-black tree insertion and
native reference release execute. The manager has a sixth distinct native
MonoBehaviour owner, reciprocal list head and key 106; the five character
owners retain keys 101 through 105 and their original payloads. Creation's
temporary reference is released, leaving every queued native record linked
to its owner with reference count 1, nonzero GC handle and the exact retained
managed iterator in its cache.

## Heterogeneous queue and return comparison

The supplied common producer snapshot is time 1.0, full signed frame 7 and
generation 0. The five actual acquisition waits retain float bits `3E99999A`,
promoted exactly to `0.30000001192092896`, and deadline
`1.30000001192092896`. The Shuffle wait has deadline 1.5. Each live native
record is compared byte-for-byte with its insertion record and checked for
frame threshold 8, phase mask `0xA`, generation 0, the respective owner key,
payload, callback `778B30` and release `778BD0`. The actual tree in-order walk
retains the five equal-deadline occurrences followed by Shuffle.

Before executing onSetup, the audit captures allowlisted raw actor, clone,
callback, list/status, iterator, wait, owner and publication storage. Every
later service preserves those bytes. Existing engine checks preserve the five
original payloads and physical owner objects; tree links may change during
the sixth insertion, while live wait-record contents remain fixed. Semantic
actors, continuations, statuses, runtime-role saved fields, publication and
source/pool/cache graph must match before and after the tail.

The exit is the actual Manage return at `36D356`, followed by the original root
sentinel. Original stack pointer, eight integer nonvolatiles and XMM6 through
XMM15 are checked before the return and after nested stepping; each emulator's
CPU context is restored exactly across runtime-invoke handoffs. Entry and
return CPU records distinguish this normal return from the preceding audit's
intentional pre-onSetup stop.

## Prefix scope and exclusions

Fresh normal and fresh paused runs must have exactly equal chronological
services, raw arguments/callers, semantic/raw-storage snapshots, admissions
and final managed/engine CPU. Every actual invocation consumes one retry token.
Unicorn pre-instruction pauses verify CPU, stack bytes and snapshot equality
before retry at the same instruction. Python-direct record creation pauses
are labelled separately and certify pre-effect provider reentry only.

Independent fresh restart aborts cover one reached service per materially new
tail family. Their service ledgers and stopped snapshots must equal the normal
prefix exactly. Paused/reentered prefixes are not counted as failures; the
previous Start/queue abort corpus remains a frozen separate dependency.
No managed exception unwinding, authored failed-API effects or equivalence of
all independently aborted continuations is claimed.

All six continuations remain pending. Shuffle state 1, Gameplay.ShuffleDeck,
later event delegates, queue drains, acquisition callbacks, initial-Day clues
and rendered/player observations are excluded. The sixth wait cannot be
silently passed to a five-acquisition-only scheduled-Reveal consumer.

## Validation

Source and consumed prior-report hashes are checked unchanged at exit.
Original asset reports are reparsed against the pinned game inputs. Python
syntax validation precedes the report producer.

```powershell
$env:PYTHONPATH='B:/CodexTools/DemonBluffReverseEngineering/python-emulation'
python reverse_engineering/scripts/audit_first_village_shuffle_admission.py `
  --game-root 'B:/SteamLibrary/steamapps/common/Demon Bluff Playtest' `
  --dumper-root 'B:/CodexTools/DemonBluffReverseEngineering/artifacts/f530404b0f3f_807de4a83df4/il2cppdumper-v6.7.46' `
  --output 'B:/CodexTools/DemonBluffReverseEngineering/artifacts/f530404b0f3f_807de4a83df4/worker_first_village_shuffle_admission_v1.json'
```

Python syntax validation and private producer session 34423 completed with
exit 0. Expanded JSON/codec equality and source/prior-report hash checks passed.
The assigned report is a byte-exact copy of that successful private output.

Derived counters are five constructors, five primitive Init calls, five
concrete Act(Init) calls, five acquisition first yields and one Shuffle first
yield. Six records remain in the queue under six distinct supplied owners.
The original 15-entry array makes 75 comparisons and zero Start calls; the
conditional null branch makes zero onSetup calls. There are 935 chronological
services, 935 paused/reentered prefixes and 935 consumed retry tokens, including
22 tail services. Fifteen independent fresh aborts cover the reached tail
service families. The report records 2,532 managed and 394 engine instruction
addresses, 63 body fingerprints, 12 selected instruction assertions, 16 Python
source hashes and 393 losslessly pooled snapshots. These counts describe one
conditional original-asset witness and its instrumentation checks.
