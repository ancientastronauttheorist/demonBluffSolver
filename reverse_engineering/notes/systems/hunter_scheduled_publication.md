# Hunter publication through native scheduled resumes

Evidence: `native-static` and `native-emulated`, conditional on supplied finished
acquisition, engine/runtime and UI services. The producer is
[audit_hunter_scheduled_publication.py](../../scripts/audit_hunter_scheduled_publication.py);
its normalized corpus is
[hunter_scheduled_publication.json](../../reports/f530404b0f3f_807de4a83df4_hunter_scheduled_publication.json).
This is a new composition, preserving the original
[Hunter role publication](hunter_role_publication.md) and
[coroutine completion](unity_coroutine_completion.md) reports.

## Decision blocker and admitted domain

The earlier Hunter publication corpus supplied an authored resume schedule and
queued speech at state zero. That left open whether synchronous speech startup
could change retained text/history before the result callback registered its
next native wait. This composition executes the actual managed producers and
consumers through native engine dispatch, wait production and queue drains.

The board is still a supplied four-character finished setup with three repeated
Hunter current-data occurrences and one Baa. Actors and their role objects are
distinct; the native seat-to-display-ID map is `[4,3,2,1]`. Clear statuses,
null `registerAs`, a retained `prior_info` stress record and runtime uses zero
are supplied. Asset `abilityUsage=0`/`picking=false` establishes the selected
Hunter identity, not a runtime InitializePickable certificate. Day wraps the
supplied uses zero to `0xFFFFFFFF`; two requests wrap it to `0xFFFFFFFE`.
These bookkeeping changes do not make the passive Hunter an active ability.

The single-request family starts all board characters Hidden (`5`). It supplies
an installed `Character.onClick` delegate targeting `RevealCard.Reveal`, global
gameplay Day (`10`), a non-killed actor, `initReveal=false`, backside
`activeSelf=true`, mana one and blocks zero. Valid `Gameplay.Instance/settings`
objects with `readRealRole=false` avoid the optional debug logging branch; false
is a fixture choice, not a legality requirement. Resource values are supplied
through their verified virtual getter slot. Mana is distinct from health.

Actual `Character.OnClick` (`0x366270`) invokes `onClick +0x100` before its later
quota/state branch. Actual `RevealCard.Reveal` (`0x386A60`) and
`CheckIfCanRevealCard` (`0x385FF0`) enforce the local backside, reveal, hidden
count, block and mana predicates, then invoke `Character.Act(Day=30)` while the
actor is Hidden. `Character.OnReveal` (`0x367500`) calls the installed
`Gameplay.OnCharacterReveal` (`0x37ED60`), increments reveal order, and stores
it on the actor. After the callback returns, original OnClick stores
`prevState +0xE0=5` and `state +0xE4=10`. Native result-wait registration occurs
before this alive-state change; queued resumes occur afterward.

Every retained snapshot separately records `reveal_card_init_reveal_byte`
(`RevealCard +0x50`), `reveal_card_state_raw` (`+0x54`) and
`gameplay_current_reveal` (Gameplay static `+0x38`). Initial UI state zero is
supplied zero storage, not an executed UI Init. At the first result-wait
insertion these fields are `1/0/0`; after the click returns and through the
scheduled drains they are `1/20/1`. Native writes are initReveal byte one at
`0x386B6A`, raw reveal state `0x14` at `0x386BCB`, and the global counter
increment at `0x37EDA6`. Actor Alive (`10`) and raw UI reveal state (`20`) are
distinct. The supplied tween completion closure is not invoked, so the corpus
does not claim the eventual UI state or rendering completion.

Delegate installation is supplied. The separately reviewed native OnEnable
binding uses `RevealCard.Reveal` on `onClick +0x100`, with RemoveBackside on
`onReveal +0x108`; it is not an onStateChange binding. Reveal tween creation,
configuration and completion registration, UI callbacks and text/layout effects
are explicit supplied services. No rendered capture is established.

This closes local request predicates under those responses. The N4 Hunter/Baa
setup, retained prior history and supplied use count do **not** establish a
reachable generated deck or a complete initial-Day PlayerHistory. The separate
two-request family invokes Day directly on the same retained actor to stress
captured callback identity; it is not asserted to be legal player chronology.
DelayReveal is acquisition and does not cause a Hunter Day clue.
This is a positive local-predicate fixture, excluding negative UI-gate rejection
certification: OnClick can still change an actor to Alive when its reveal
callback exits without a Day action.

## Synchronous producer and queue join

StartCoroutine is a supplied native-record creation gateway. It authors the
normal retained `0x88`-byte payload and owner links, then invokes actual engine
dispatcher `0x778D90`. This does not execute or certify full record-creation
body `0x77BC80`. The cached method, parameter-count/runtime-invoke exports,
GC handle services, owner context/lookup and type queries are supplied.

Runtime invocation calls actual GameAssembly
`SetupCoroutine.InvokeMoveNext` (`0x1C8A780`). Its folded pointer equality and
identity helpers execute, and the supplied IEnumerator slot-zero adapter
tail-dispatches the actual result (`0x375FE0`) or speech (`0x376240`) MoveNext.
The actual managed bridge writes its Boolean result through the actual native
dispatcher frame's byte address. Native dispatch/frame construction, callback
`0x778B30`, reference release `0x778BD0` and cleanup execute.

The yielded-current getter is supplied and mirrors the actual retained
WaitForSeconds object into the engine harness. Native type dispatch
`0x779370` executes its WaitForSeconds branch. Duration zero comes from the
result iterator; speech duration preserves the exact float value
`0.4000000059604645`. Native production uses engine clock `+0x60`, full signed
frame counter `+0xC8` plus one, and current queue generation. Native consumption
uses the separately supplied clock `+0x90`, retained entry-time frame sample,
phase mask and generation. Clock/frame/phase availability is supplied rather
than a live engine chronology certificate.

For the standard clock one/frame seven, immediate result startup inserts a
zero wait with deadline one/frame eight. Its native callback resumes the
retained result iterator. Inside that resume, ShowInfoDelayed starts
synchronously and executes speech state zero: savedAct and text are established
before native registration of the `0.4f` wait. The speech deadline is exactly
`1.4000000059604645`, with frame threshold nine. The later native callback
completes speech and releases all retained native records.

Every queue snapshot contains **all** ordered surviving waits. Native red-black
insert/erase/drain bodies execute, and the structural validator checks complete
record bytes and links. Logical fixture IDs live in an external node map;
native zero repeat-duration/flag bytes are preserved. Stable equal-deadline
ordering is compared against explicit expectations. No guessed ready-only list
is passed to the consumer.

## Exit tests and limits

The successful corpus has 20 single-click cases, six clock/generation cases,
four retained two-request cases, one two-request generation case and three
owner-removal cases: 34 compositions and 173 native drains. The first 31
compositions complete publication; the owner-removal cases intentionally do not
resume the queued result. Two selected truthful/bluff baselines yield 254 exact
stopped managed-service prefixes.

Derived chronological native/service counts are 75 record creations and wait
insertions, 147 actual managed bridge resumes, 185 queue visits, 75 owner
lookups/erasures, 73 wait callbacks and 75 GC-handle/payload cleanup pairs.
The corpus executes 1,621 managed and 690 engine native/service addresses.
It retains 20 managed and 21 engine body fingerprints, 54 inherited Hunter
operand assertions, 16 selected click/bridge assertions and 60 cross-image
bridge checks. These are checks/fingerprints, not claimed disjoint method or
instruction counts.

The finite family compares exact generated sentences, ordered native reference
IDs (including duplicated opposite-seat references), info/closure/iterator
identity, retained prior/history records, use count, savedAct and shown text.
Nonactor character storage remains invariant. A retained second request
replaces the role's installed delegate while the earlier result iterator still
publishes its own captured info. Equal-zero and mixed speech/result deadlines
exercise that distinction for truthful and bluff Hunter producers.

The clock family exercises negative and above-u32 signed frames and generation
rollover. Suppression covers future deadlines, phase mismatch, frame threshold
and current-generation waits. The generation fixture keeps an original
successor: native visits newly inserted due speech waits but suppresses them
until the next generation. A single original entry can finish without revisiting
new insertion because the drain retains its saved successor; survival alone
would not prove the generation gate.

Supplied missing/null owners remove waits and release records without managed
resume. A supplied mismatched owner exercises the native callback diagnostic
and release path. Valid completions verify zero reference counts and empty
owner lists. Stopped managed-service fixtures verify exact chronological event
prefixes and their retained snapshots; stopped native record and queue states
are recorded. These are stops before authored service effects, not CLR
exception or failed-engine-API modeling.

The report carries per-body fingerprints, selected operands, source hashes and
unchanged original-report hashes. It contains no proprietary instruction bytes.
General multicast/reentrant cancellation, arbitrary yield types, exception
branches, actual record allocation/runtime invocation, selectable generation,
live phase availability and rendered public capture remain outside this domain.

`audit()` returns the complete expanded report after all native and prefix
checks. The CLI uses the existing
[audit_report_snapshots.py](../../scripts/audit_report_snapshots.py) lossless
full-snapshot codec only when its serialized output is smaller. It asserts both
direct expansion and JSON-roundtrip expansion equal that full audit return.
The output advertises `snapshot_encoding=sha256-full-authored-snapshot-v1` and
retains every snapshot field in `snapshot_blobs`; the codec source hash is
included. Expand with `expand_snapshots()` before reading pooled snapshot
fields. `metadata_counts` derives aggregate counters from the final case,
drain and chronological event arrays for allowlisted checkpoint diagnostics.
The successful output has 1,493 interned snapshots and 16,345,290 physical
bytes on Windows. The CLI measured 35,652,587 expanded UTF-8 bytes versus
15,732,370 pooled UTF-8 bytes. Removing only the three new snapshot fields and
new diagnostic metadata exactly reproduces the previous expanded corpus;
native contexts, chronological events, existing snapshots and stopped prefixes
are unchanged.

Reproduction in PowerShell (syntax gate first):

```powershell
python -m py_compile reverse_engineering/scripts/audit_hunter_scheduled_publication.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
$env:PYTHONPATH='B:/CodexTools/DemonBluffReverseEngineering/python-emulation'
python reverse_engineering/scripts/audit_hunter_scheduled_publication.py 'B:/SteamLibrary/steamapps/common/Demon Bluff Playtest' 'B:/CodexTools/DemonBluffReverseEngineering/artifacts/f530404b0f3f_807de4a83df4/il2cppdumper-v6.7.46' --output 'reverse_engineering/reports/f530404b0f3f_807de4a83df4_hunter_scheduled_publication.json'
```
