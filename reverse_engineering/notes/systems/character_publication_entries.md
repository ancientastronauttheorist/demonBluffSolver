# Character factory and direct speech entry callers

Pinned build `f530404b0f3f_807de4a83df4`. The offline fixture executes six
complete native Character callers in 112 cases, four retained two-call
sequences, one supplied UI callback probe and 25 exact stopped prefixes.
It reaches 143 of 145 decoded entry instructions; the two unreached instructions
are trap bytes after native null guards. Fourteen entry assertions and 33
dependency assertions pin the fields, ABI and native publication callees.

| Method | Method ID | Entry |
| --- | --- | --- |
| DelayReveal factory | tdi5487.m0033 | 0x364A20 |
| DelayedDemonKill factory | tdi5487.m0049 | 0x364A90 |
| ShowActed | tdi5487.m0068 | 0x368B00 |
| ShowInfoDelayed factory | tdi5487.m0073 | 0x369350 |
| ShowTrailerAct | tdi5487.m0074 | 0x3693E0 |
| HideActed | tdi5487.m0075 | 0x365220 |

The three factories allocate the exact metadata iterator type, execute the
verified folded empty base body, publish state zero and the physical Character
receiver before its barrier, then publish an optional evil Character or speech
string before a second barrier. They retain null receivers/arguments without
dereferencing them. Current remains the zero from the explicitly supplied
zeroed allocation. Repeated calls allocate distinct iterators and retain earlier
objects. This does not establish scene order, role cloning or generator execution.

ShowActed moves the low DWORD trigger from R8 to R9, the ActedInfo from RDX to R8,
and the delay from XMM3 to XMM1 before executing actual ShowActedDelayed. Its
captured trigger ignores poisoned upper bits; float32 bits preserve positive
and negative zero, 0.3f, 0.4f, infinity and a NaN payload. The factory publishes
the actor, description object, delay and trigger in native order before the
caller tail-forwards to supplied StartCoroutine. Registration records the
state-zero iterator only. No first yield or readiness is inferred here.

ShowTrailerAct immediately obtains the Acted GameObject and activates it. It
then reloads Character.acteds before tail-forwarding to actual Acted.Act(string).
The supplied ActedVersion.Show receives the original string, including null or
empty, and duplicate layout references cause duplicate rebuild requests. This
path does not append history, decrement uses, publish savedAct, check a trailer
mode flag, or construct a wait. HideActed obtains the same GameObject and
tail-forwards SetActive(false); version/layout fields are not consumed.

An authored SetActive callback clears Character.acteds after activation. The
native reload reaches its null guard with the active GameObject retained and
without a Show call or savedAct write. This is an explicit callback effect,
not evidence for a Unity callback implementation. Normal inert cases retain
every actor byte. Successful calls verify stack balance, all eight integer
nonvolatiles and XMM6–15. Stopped services compare exact event/state prefixes;
native exception unwinding and rollback are not synthesized.

Metadata, zeroed allocation, barriers, coroutine registration, Unity
GameObject/active effects, ActedVersion.Show and layout effects remain supplied
services. Actual generator MoveNext, scheduling, text animation and object
lifetime are outside this audit. Private bytes and bodies remain off-repository.

Independent repo/private reports are byte-identical: 1,387,197 bytes, SHA-256
`d4b25ce1eb8db33c318e65081af2f659931ddbdb5893b6f293a8fc06c7c2f843`.

```powershell
$env:PYTHONPATH='B:/CodexTools/DemonBluffReverseEngineering/python-emulation'
python reverse_engineering/scripts/audit_character_publication_entries.py GAME_ROOT DUMPER_ROOT --output REPORT
```

Report: `f530404b0f3f_807de4a83df4_character_publication_entries.json`.
