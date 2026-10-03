# Original first-village generation through retained initialization

This bounded dependency audit asks whether the original N5 profile witness can
reach five initialized physical actors without replacing the generated roster,
the pool graph, or the native Manage caller. It stops before publication. The
solver decision blocker is the later acquisition/action chronology: primitive
`Character.Init` is not a role's `Act(Init)` and does not establish a public clue.

The source is [audit_first_village_initialization.py](../../scripts/audit_first_village_initialization.py).
The assigned report is
`reports/f530404b0f3f_807de4a83df4_first_village_initialization.json`.
Build identity is `f530404b0f3f_807de4a83df4`; original instructions remain private.

## Admitted witness and supplied runtime

The source consumes the original asset reports and row zero from
[first-village bluff generation](first_village_bluff_generation.md). It executes
the same original profile setup, actual roster selection and pool construction
in the primary `BluffJoin` emulator. Generation indices are all zero; sort keys
are `0, .125, .25, .375, .5`. The returned assets are Minion 21596, Confessor
21614, Lover 21626, Hunter 21621 and Enlightened 21618. The retained physical
actors receive descending IDs `5,4,3,2,1`. No Minion bluff selection is executed.

Original serialized asset fields supply the runtime hydration bindings. Their
source managed classes are `Minion`, `Confessor`, `Empath`, `Tracker`, and
`Shugenja`. Scene actor objects, distinct status components and their empty
active/resistance lists, number/text controls, empty picked arrays and UI
providers are supplied. Each actor's raw initial state 20 and previous state 10
are authored scene values, not constructor defaults. Previous gameplay phase is
supplied zero. Native Character constructors execute on those same physical
actors before generation and produce fresh distinct acted/hover lists, one use,
empty saved text and the act flag. CLR list construction and MonoBehaviour base
construction are supplied gateways, with a poisoned void-constructor return.

Inherited collection, LINQ, RNG, profile-copy and positioning gateways remain
explicit providers. Role cloning supplies a distinct copy of the selected
asset's source-role bytes and preserves its supplied class pointer. Text/logging,
Unity object/activity checks and WaitForSeconds construction are supplied.
`StartCoroutine` synchronously enters the actual managed first MoveNext using
a separate lower stack frame, then restores the original outer CPU context.
This closes the managed first-step composition under the supplied adapter; it
does not execute an engine dispatcher or create a native queue record.

## Native retained exit

One continuous `Characters.ManageCharacters` invocation executes the five actual
`Character.Init` calls at `0x36CFDA`, checking the callee return at `0x36CFDF`.
The original caller stack and nonvolatile register values remain retained. The
stop is the instruction at `0x36D01E`, after the fifth return and before board
publication. Actual Hidden refresh and `RefreshView` execute; the selected
folded `ret 0` at `0x33ED50` executes natively and preserves the allocated
iterator in RAX at `0x365D2F`.

Every Init return checks all eight integer and ten XMM nonvolatile registers.
The publication stop checks the live Manage stack pointer against the first
Init caller frame and checks the original root return sentinel. Timeout or
instruction-budget pauses resume the same live Unicorn context; they never
reseed the caller or count as completion.

Each actual initialization clears its fresh acted-history list and active
statuses, incrementing their versions to one. It retains the hover-list
identity/version zero, constructor-produced saved empty text and act flag. It
sets one runtime use, Hidden raw state 5, previous raw state 20, exact generated
currentData/alignment and the descending ID; bluff/register/trailer/runtime
pointers, revealed and start flags are cleared. Asset abilityUsage zero does
not manufacture a runtime zero-use initialization result.

Actual DelayReveal state zero changes to -1, clones the same retained selected
source role, allocates its wait and returns state one with that current wait.
The native wait literal is float32 bits `0x3E99999A`, promoted exactly when a
later engine schedule is constructed. Each actor, iterator, wait and clone is
distinct; later initialization must preserve every earlier actor's object,
history/hover/status backing storage, iterator, wait and clone bytes. The
retained original profile, current script/count graph, source/cache lists and
pool identities/contents are recorded separately from actor projections.
Selected asset objects, source-role storage and source-class storage are also
checked byte-for-byte unchanged. Status snapshots expose both backing
identities, versions and actual values; actor snapshots include an authored
storage fingerprint. With the inherited inline CLR provider, each retained
List range of `0xC20` bytes covers its object and all 256 backing entries.

## Prefix and provenance checks

Every reached new initialization/first-yield service in the Manage composition
is stopped before its supplied effect. The replay must match the successful
chronological service argument prefix and exact pre-service snapshot. Earlier
completed native actors and continuations retain their bytes in these stopped
cases too. Constructor failure factors and older pool failure factors are not
recounted in this bounded corpus; their existing audits retain their domains.

Consumed Python sources and prior reports are hashed and checked unchanged at
exit. Build and Dumper inputs are pinned by the inherited manifest checks.
All semantic/prefix assertions precede lossless snapshot pooling. JSON output
must expand exactly to the audit return value, including a JSON round trip.

The successful producer derives 5 constructors, 5 native Init completions and
5 actual first yields, 583 Manage services including 115 new services, and 115
exact new-service stopped prefixes. It records 2,071 visited native instruction
addresses, 13 selected operand assertions, 41 method-body fingerprints and 9
Python source hashes. CLI pooling retains 344 full authored snapshot blobs.
These counts describe this one witness and its prefixes, not an exhaustive
cross-product of generation or scene state.

Reproduce with the pinned private Python-emulation dependencies on `PYTHONPATH`:

```powershell
python reverse_engineering/scripts/audit_first_village_initialization.py --game-root 'B:/SteamLibrary/steamapps/common/Demon Bluff Playtest' --dumper-root 'B:/CodexTools/DemonBluffReverseEngineering/artifacts/f530404b0f3f_807de4a83df4/il2cppdumper-v6.7.46' --output '<distinct-private-report-path>'
```

## Exclusions and next boundary

This witness establishes an original profile's selected generation and retained
initialization prefix under declared runtime providers. It does not certify all
generation factors, the scene's complete SetupDelay caller, engine owner/queue
readiness, DelayReveal state-one completion, board publication, Start dispatch,
role Init/AfterRoundStart, player click legality, rendered pixels, or a complete
initial-Day observation history. No hidden runtime value enters a solver or
live-game decision.

The next honest join must carry these five actors and first waits into actual
acquisition callbacks on their correct owners. After actual publication, the
Confessor `Act(Init)` path requires its concrete status writer. These boundaries
cannot be supplied as already finished actor state while claiming continuous
native acquisition.
