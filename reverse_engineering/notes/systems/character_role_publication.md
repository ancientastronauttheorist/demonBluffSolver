# Installed role callback through history and speech publication

Pinned build `f530404b0f3f_807de4a83df4`. This separate composition extends the
[first-yield callback audit](character_role_callback.md) without changing its
report. It executes actual `RoleAct` and the installed captured callback, both
`ShowActedDelayed.MoveNext` resumes, the `String.IsNullOrEmpty` leaf, the native
`List<ActedInfo>.Add` fast path, explicit `ShowInfoDelayed.MoveNext` resumes,
`Character.GetCharacterBluffIfAble`, and `Acted.Act(string)`. The report contains
158 cases, four alias/order sequences, 196 exact stopped-service prefixes,
33 instruction assertions and 557 executed instruction/service addresses.
A separate process reproduced the report exactly. Python compilation and
`git diff --check` passed.

The existing [immediate Acted](acted_surface.md) and [delayed Acted](acted_delayed.md)
audits establish their own UI callers. This composition adds the actual Character
history, Day-use accounting, speech iterator registration and saved-speech writes
across those boundaries. It supplies ActedVersion.Show and layout effects instead
of reconstructing their animation bodies again.

## Explicit ordering

The role gateway invokes its newly installed callback zero, one or two times.
Its registration service explicitly performs the initial result-iterator yield,
as in the earlier report. Later result resumes are invoked directly in the
authored forward or reverse order. Speech registration queues its newly created
iterator without resuming it. Only after the outer result resumes finish does the
harness explicitly resume those speech iterators, in registration order. These
are fixture contracts; they establish neither scheduler readiness nor elapsed
wait time, cancellation or real Unity ownership.

## Result resume, history and Day uses

The second result resume writes iterator state `-1` before checking its captured
actor. `act == false` returns false without checking info. Otherwise a null info
fails, while null or empty `info.desc` returns false before history or accounting.
The native string leaf checks the actual supplied managed string length.

A nonnull `onAboutToAct` delegate runs before the history-list reference is read.
The supplied callback can replace description/reference pointers or remove the
history field. There is no description gate repeated after that callback: an
initially nonempty record changed to null or empty still appends and can publish
that null/empty speech. Callback implementation and its mutations remain explicit
services; no native Rambler callback is claimed here.

Native List.Add increments version before checking its backing array. On its
capacity fast path it increments size, writes the same ActedInfo pointer, and
tail-calls the write barrier. Consequently a stopped barrier retains the new
size and element. A null backing array preserves the version increment while
failing before append. Capacity growth is a named supplied service, reached only
after the native version increment and native generic-context resolution.

Only trigger DWORD `30` decrements `pickableUses`. The actual instruction is
`dec dword ptr [actor+0xDC]`; fixtures retain exact 32-bit wrapping, including
`0x80000000 -> 0x7FFFFFFF`. The GameplayEvents.OnCharacterInfoRevealed delegate
runs after the append and decrement. A supplied event mutation therefore changes
the already appended object and the description subsequently captured for speech.
The native result coroutine then allocates and constructs the speech iterator,
publishes actor/state and description around separate barriers, and registers it.
Only after successful registration does it hide `pickable` when the current uses
DWORD is exactly zero. Negative uses do not enter that equality branch.

## Speech and UI publication

The actual speech iterator first writes state `-1`. If GameData.TrailerCharacters
is enabled, it requests TrailerCharacters.GetCharacterInfoOfId through an explicit
lookup service. A nonempty override is looked up again and replaces the iterator's
captured string before its barrier. Null/empty override text keeps the original.
This override changes displayed/saved speech while the historical ActedInfo's
description remains original in the fixture.

The native first speech step obtains the Acted component's GameObject/name,
concatenates the exact `Character: ` prefix, obtains the GameObject again, logs,
and invokes the blankText setter through verified function/context slots. Only
after that setter returns does it copy the iterator's captured description into
`savedAct+0x198`, before its write barrier. A setter failure therefore retains old
saved speech; a saved-speech-barrier failure retains the new pointer. History and
the Day decrement already persist at both boundaries.

Native `GetCharacterBluffIfAble` selects dataRef for states `20`/`30` or a revealed
actor. Otherwise it invokes supplied Unity null/equality on the current bluff and
rereads the bluff field for a live result. The consumed CharacterData.picking flag
or state `20` bypasses the speech wait. Other tested states publish a WaitForSeconds
with the verified literal bits `0x3ECCCCCD` (`0.4f`), then state `1`. An explicit
later resume activates the speech GameObject and executes actual Acted.Act(string),
which calls the supplied ActedVersion.Show and traverses the captured layout array.
Duplicate layout references produce duplicate calls. Null UI fields and failed
services preserve all preceding history, accounting, text and saved-pointer writes.
Completion and another explicit resume return false without further effects.

## Pointer chronology and partial failures

The history stores object pointers, not frozen copies of descriptions. Repeated
callbacks with the same ActedInfo append that same identity twice. Separate
retained Init/Day callbacks with distinct info pointers prove the authored forward
and reverse result order changes pointer chronology and the final displayed/saved
speech. A supplied post-publication fixture mutation updates the history's aliased
record description while leaving the previously captured saved string unchanged.

Three cold baselines stop at every reached service occurrence, including second
callback work, capacity growth and trailer override work. Each attempted event
and visible managed-state snapshot equals its baseline prefix exactly. Native
null/bounds guards stop at their runtime boundary; no rollback or exception
unwinding is synthesized. Successful outer calls verify stack balance, all eight
general nonvolatile registers and all 128 bits of XMM6. Actor bytes outside the
declared mutable uses/history/saved-speech fields remain unchanged.

GameAssembly and Dumper inputs remain hash-pinned. Exact declaring types/fields,
managed method rows, complete unwind families, both leaf returns and immediate
gateway identities are verified. Reached code must be a decoded native instruction
or a named service; unknown native execution is rejected. Native bodies/bytes stay
private, and reports contain authored fixtures and evidence facts only.

Metadata/class initialization, zeroed GC allocation, barriers, actual delegate
construction/invocation, role results, preappend/event implementations, list growth,
trailer lookup, Unity lifetime/object/string/log/text/Wait services, ActedVersion.Show,
layout effects and coroutine registration/resume adapters remain supplied. Real
scheduler interleaving, UI animation, delegate internals, object destruction and
native exception unwinding remain open. The native audit itself does not alter live-game behavior.

```powershell
$env:PYTHONPATH='B:/CodexTools/DemonBluffReverseEngineering/python-emulation'
python reverse_engineering/scripts/audit_character_role_publication.py GAME_ROOT DUMPER_ROOT --output REPORT
```

Report: `f530404b0f3f_807de4a83df4_character_role_publication.json`.

## Guarded Rust publication replay

The separate `bluff::character_role_publication` replay compares 78 supported
normal native fixtures/baselines and two distinct-record forward/reverse result
sequences in five focused tests. It starts from independently verified suspended
results after first yield, allowing at most two result resumes before contiguous
speech completion in registration order. Readiness and cross-kind interleaving
remain outside this version.

Complete Actor state, physical ActedInfo/reference List/string records and history
identities are retained. Strings carry UTF-16 units, including NUL and unpaired
surrogates. Shared reference Lists and repeated layout occurrences are preserved;
shared nonnull backing arrays are excluded. Typed identities and callback/layout
collisions, fresh allocation aliases, capacity and aggregate retained work are
validated before cloning. Runtime/captures, Unity liveness, trailer lookup, inert
callbacks/UI, no-growth storage and supplied resume order require explicit verified
provenance. Mutating callbacks, service failures and unsupported schedules reject
atomically. All 822 Rust library tests and the release build pass.
