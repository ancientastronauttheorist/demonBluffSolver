# Character initialization and first-yield publication

Build `f530404b0f3f_807de4a83df4`. The executable audit runs 475 authored
cases through 486 distinct native instructions. It pins GameAssembly, Dumper
metadata and exact field declarations, checks 20 instruction relationships,
and validates the stack and all eight nonvolatile integer registers on normal
returns. No live game memory or proprietary executable bytes enter the report.

The native boundary is `Character.Init` (`0x365a20`),
`InitWithNoReset` (`0x365720`), and the joined Hidden-state portion of
`RefreshCharacter` (`0x367970`) followed by the first invocation of
`Character.<DelayReveal>d__84.MoveNext` (`0x3756b0`). The shared call at
`0x33ed50` executes its exact folded `ret 0` instruction; the audit does not
misidentify that no-op as the arbitrary managed alias printed by a decompiler.

## Ordered caller writes

Both entry points first hide the Acted GameObject and clear the existing
`actedInfos` list. Clear increments its version even when empty, zeroes its
count, and calls the array-clear service only for a positive prior count.
Neither entry point replaces the list, clears `savedAct`, clears `bluffRole`,
or dispatches a role action.

Ordinary Init additionally clears `trailerInfo` before the Acted operations.
After the info clear it clears `runtimeData` and the Start guard. No-reset
clears only the Start guard at that point.

Both test `createdDeadPrefab` through Unity liveness. A live object causes
RIP hiding, Destroy, and then a raw pointer clear. An absent or destroyed object
skips all three operations. A destroyed, non-null reference therefore remains
stored. This differs from treating Unity-null as a zero pointer.

Both clear raw `bluff`, store the new `dataRef`, and log its name. No-reset
clears `revealed` before the data store and uses the context-bearing Debug.Log
overload after obtaining the Character GameObject. Ordinary Init clears
`registerAs` and `revealed` after its plain Debug.Log call, then copies the
incoming data's `startingAlignment` to the actor. No-reset preserves
`registerAs`, `trailerInfo`, `runtimeData`, and runtime alignment.

Both reset `killedByDemon` to false and `pickableUses` to one. `killedHidden`
is preserved. An ID other than `-100` is boxed and formatted using the pinned
`# {0}` literal, sent to the number component's virtual text setter, and only
then stored to `id`. Exactly `-100` skips both the UI setter and stored-ID
write, even when the number component is null.

Both copy current `state` to `prevState`, store Hidden (5), and then invoke
the current `onStateChange` delegate if present. This callback observes the
old active statuses. Ordinary Init reloads `statuses` **after** the callback,
then increments that active list's version and zeroes its count. A callback
which replaces the statuses component therefore changes which list is
cleared; the corpus also exercises replacement of an initially null component.
The integer status backing array is not cleared, while its logical list is
empty. No-reset does not read or mutate either status list.

Neither entry point clears resistance entries or the shared status target.
These are fields on `CharacterStatuses`, separate from its active-list field.
The audit checks all other actor bytes against authored sentinels, so the
preservation claims do not depend on fresh-allocation zeroes.

## Refresh and first yield

After callback/status handling, both callers invoke RefreshCharacter, then
RefreshView, then allocate and publish a new DelayReveal iterator. They retain
no Coroutine handle and cancel no earlier iterator.

Joined fixtures execute RefreshCharacter's actual native body with Hidden
state, zero/two picked controls, previous gameplay state zero/Night (20), and
ability usage zero/ResetAfterNight (10). The method first hides each picked
control. It reads `Gameplay.PrevState` at static offset `0x2c`, not current
GameplayState at `0x28`. Its possible use-count reset leaves the caller's one
unchanged; Hidden prevents pickable activation. Other body states are outside
this joined sub-boundary. RefreshView remains an explicit inert actor-state
service linked to the [view audit](gameplay_reveal_view.md); pixels and other
presentation side effects are not reconstructed here.

The caller directly sets iterator state zero and stores the physical Character
reference. In joined mode, the scheduler service explicitly invokes native
MoveNext to its first yield. MoveNext changes iterator state to -1, reloads the
actor's current `dataRef.role`, calls the clone service, and stores that result
to the actor's shared `role` field before its barrier. It then allocates and
constructs a WaitForSeconds, stores iterator current, and changes state to one.
The literal is float32 bits `0x3e99999a` (about 0.3 seconds). A service-supplied
null clone result is stored and still reaches the yield; the caller has no
post-clone null check. This does not assert that every clone implementation can
produce null for an arbitrary input.

The role pointer belongs to the actor, not the iterator. A later initializer
can publish another clone on that same actor without cancelling the earlier
continuation. Actual scheduler registration/resume order and post-yield Reveal
are covered by the separate [bluff acquisition evidence](gameplay_bluff_acquisition.md).

## Failure prefixes and scope

The corpus stops at every reached metadata, barrier, allocation, log, UI,
callback, refresh and scheduler boundary on representative cold paths. Each
stop must reproduce the full native actor snapshot at the corresponding
successful prefix. Null Acted/list/RIP/data/number/status dependencies have
dedicated cases. No-reset succeeds with null status components and lists;
ordinary Init fails after the Hidden callback unless that callback supplied a
valid replacement. Null RIP is irrelevant when no live death object exists.

In particular, failure during formatting leaves the stored ID unchanged;
callback failure leaves Hidden state but old active statuses; failure after a
role store retains the new role even when wait allocation or construction
does not finish. No rollback or managed exception unwinding is synthesized.

Metadata resolution, Unity liveness and object operations, diagnostic
formatting, role cloning, wait construction, and scheduling are authored
services with exact caller argument checks. Joined mode exercises native
first-yield publication, not UnityPlayer's scheduler implementation. Other
callbacks are inert except the two explicit state/status replacement probes.
Report event snapshots use cumulative `actor_changes`; each case's `final`
is a complete semantic snapshot. Addresses in those snapshots are authored
fixture identities, not live process addresses.

## Bounded offline replay

`bluff::character_initialization` provides the versioned
`character_initialization_native_v1` replay. It requires valid objects/lists,
inert callbacks/UI, a verified explicit clone result, and independently verified
synchronous first-yield handoff. It returns the actor, inert callback
observation, semantic write order and retained/new continuations. Existing
continuations, resistance, target, copied bluff role and saved speech survive.
It represents live and destroyed retained death objects separately, preserves
the `-100` ID sentinel, wraps native list versions, and permits a null clone
result. It never executes role actions or infers readiness.

The Rust boundary deliberately excludes the native corpus's injected failures,
null dependencies and mutating callbacks rather than pretending to reconstruct
those services. Six isolated tests include comparison of 162 successful joined
native cases, callback timing, no-reset preservation, continuation retention,
overflow, boundary rejection and capacity rejection. Module declaration and
shared-build validation are coordinated by the parent task. The subsequent
[setup initialization batch](setup_initialization_batch.md) joins successful
caller occurrences to explicit actor/allocation state and adds a native repeated
initializer sequence audit.

Reproduce with Unicorn 2.1.4 on the private emulation PYTHONPATH:

```powershell
python reverse_engineering/scripts/audit_character_init.py GAME_ROOT DUMPER_ROOT --output reverse_engineering/reports/f530404b0f3f_807de4a83df4_character_init.json
```
