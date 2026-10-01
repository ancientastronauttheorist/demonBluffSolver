# Character field accessors and mutators

The pinned build executes thirteen complete Character leaves in 146 fixtures,
five exact stopped-barrier cases and four retained setter/getter sequences.
All 31 decoded instructions are asserted. These entries lack containing unwind
records; decoding begins at their exact Dumper/ledger entries and ends at the
verified return or tailjump, excluding alignment and the neighboring method.
GameAssembly and Dumper hashes, exact signatures and consumed Character field
declarations are checked.

The onClick and onReveal accessors load/store the complete reference at offsets
`0x100` and `0x108`. Setters store before tailcalling the supplied GC barrier.
They do not invoke, combine or remove delegates. A stopped barrier retains the
stored pointer. Null, self and arbitrary authored pointer values are preserved
as raw values without Unity lifetime checks.

`CreateRuntimeData` only stores its supplied reference at `0x70` and requests
the barrier; it does not allocate runtime data. `GetRuntimeData` reads that
reference. `UpdateRegisterAsRole` and `UpdateTrailerInfo` similarly store only
their supplied references at `0x60` and `0x68`, followed by the same barrier.
No copied role, status, speech, action or UI update is performed by these leaves.

`GetRealAlignment` and `GetState` read the stored DWORDs at `0xF8` and `0xE4`.
The EAX write zero-extends their raw 32-bit enum representation. Alignment is
the current stored value; neither getter derives an answer from statuses or
register-as data. `ChangeAlignment` writes only the low DWORD of its argument
to `0xF8`. `Uninteractable` writes the byte at `0x178` to one; `Interactable`
writes it to zero. Neither method calls a UI or callback service.

Each fixture checks every other byte in a patterned 512-byte actor buffer.
Normal returns restore the stack, all eight integer nonvolatiles and all ten
XMM nonvolatiles. Input values test upper pointer bits, low-DWORD argument and
return width, zero references and unchanged neighboring bytes. The retained
sequences execute setter/getter pairs repeatedly against the same actor.

The sole service is the GC barrier. It verifies the field address and already
stored value, then supplies a deliberately poisoned void return. Controlled
stops assert the exact attempted event and actor snapshot; they do not model
native exception unwinding. Receivers and storage are authored valid mapped
objects. Invalid receivers, actual GC, delegate internals, Unity lifetime and
live game interaction remain outside scope.

Independent repository/private reports must match as JSON values. Reports
contain authored field values, actor digests, normalized metadata and evidence
facts; copied executable bytes and native bodies stay private.

```powershell
$env:PYTHONPATH='B:/CodexTools/DemonBluffReverseEngineering/python-emulation'
python reverse_engineering/scripts/audit_character_fields.py GAME_ROOT DUMPER_ROOT --output REPORT
```

Report: `f530404b0f3f_807de4a83df4_character_fields.json`.
