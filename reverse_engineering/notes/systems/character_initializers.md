# Character initializer physical-state audit

Status: frozen after two independent successful corrected native producers.
This family preserves the earlier
`audit_character_init.py`, initialization note, and constructor/init/reward
fixtures. It executes exactly the two initializer bodies against offline,
authored storage; it does not read a game process.

| Declaration | Symbol key | Entry | Complete end, exclusive | Next managed entry |
| --- | --- | --- | --- | --- |
| `public void Init(CharacterData character, int id = -100)` | `tdi5487.m0025` | `365A20` | `365D73` | `365D80` |
| `public void InitWithNoReset(CharacterData character, int id = -100)` | `tdi5487.m0026` | `365720` | `365A16` | `365A20` |

Each method has one complete verified unwind range. The producer checks the
containing section's raw extent, reads every byte through the next managed entry,
decodes the complete range, and distinguishes its terminal guard traps from
alignment padding. The four terminal `int3` instructions are `365A0F`, `365A15`,
`365D6C`, and `365D72`; none is executed. All other 367 instructions execute in
the bounded corpus. The tracked report retains 64 authored selected operand
assertions, exact per-body fingerprints/lengths/instruction counts, and every
gateway call and tail-call site. Complete native disassembly remains private.
Coverage means this bounded native surface, not
completeness of the game or of supplied callees.

The producer freshly verifies the SHA-256 and size of both `GameAssembly.dll`
and `global-metadata.dat`, and of Dumper `dump.cs`, `script.json`, and `il2cpp.h`.
Exact declarations bind Character, CharacterData, CharacterStatuses, ActedInfo,
the actual `<DelayReveal>d__84` iterator, Action/Delegate/MulticastDelegate, and
the relevant enums. Generated generic-list and reference/integer-array header
records bind `_items`, signed `_size`, `_version`, and element widths. The header
layout calculation independently derives class `cctor_finished` at `E0`, TMP
slot 66's function at `558` and MethodInfo at `560`, and the exact TMP_Text
`set_text` declaration. The literal slots resolve to `INIT: ` and `# {0}`.

Both methods deactivate the Acted component's GameObject, increment acted-info
list version, set its count to zero, and call whole supplied Array.Clear only
when the captured old count is positive as a signed DWORD. Zero and negative
diagnostic counts skip array clearing. The version increment wraps at 32 bits.
The array's cleared reference prefix and retained suffix are checked separately.

Both reset bluff, revealed, uses, killedByDemon, and characterStartActed. They
store the captured incoming data, then reload the owner's current data reference
to obtain its name for the supplied concat/log path. Ordinary Init also clears
trailer before the first Acted call, runtime after acted-info clearing, and
registerAs after logging. It obtains startingAlignment from the original captured
incoming data pointer; a callback replacement of the owner's current data does
not redirect that alignment read. InitWithNoReset preserves trailer, runtime,
registration, alignment, status lists/backing integers, and the other unconsumed
owner fields.

A nonzero Unity-destroyed createdDeadPrefab pointer remains stored when the
supplied inequality returns false in AL. A true AL reaches RIP deactivation,
reloads the current created object and Object class, calls supplied Destroy, and
then clears the owner's created pointer with a barrier. Callback fixtures replace
the created pointer, reset the class initialization word, and replace the shared
Object metadata slot with another physical class. They verify the captured
inequality input and the later reloaded Destroy receiver independently.

The sentinel comparison uses the captured incoming ID's low DWORD. `FFFFFF9C`
skips boxing, text, and ID storage regardless of upper R8 bits. Other IDs capture
the TMP receiver before box/format services, place the original DWORD in the
external caller's stack at entry-SP plus eight, and load the captured receiver's
current class after those services. A boxing callback replacing the owner's
number pointer does not replace the already captured receiver. A callback
replacing that receiver's class changes the actual TMP class in R9 and its MI in
R8. The native owner ID store uses the original captured ID even after a callback
changes the stack boxing slot. Nullable name, formatter result, and log context
are explicit supplied profiles; they are not inferred native callee behavior.

The previous-state copy, callback capture, and Hidden=5 store precede delegate
invocation. The callback sees the original active-status list. Ordinary Init then
reloads the current statuses reference and its current active list, increments
that list's version, and clears its count. It does not clear its backing integer
array. InitWithNoReset never reads or clears statuses. Fixtures replace both the
outer statuses reference and the inner active list, set callback state, and test
the resulting null guards. Info/status-list and GameObject aliases use one
physical byte graph.

RefreshCharacter and RefreshView are whole supplied services. The initializer
then requests its iterator class as needed, calls supplied allocation and the
shared `System.Object..ctor` gateway at `33ED50`, publishes native iterator state
zero and owner fields, and tail-calls whole supplied StartCoroutine. The exact
shared gateway is `ret 0`; all 3582 folded declarations are counted and no alias
is promoted. The separately declared iterator constructor is `357700` and is
not entered. No MoveNext, first yield, role clone, WaitForSeconds, renderer, or
implicit scheduler is executed or claimed by this family.

Every row retains complete bytes for the authored owner/data/status/list/array/
delegate/class/text/GameObject/iterator records, metadata slots/flags, and native
stack window. Engine activity/destruction and TMP text are explicitly authored
diagnostics within supplied fixture records, not alleged Unity runtime field
offsets. Every service entry records all four raw arguments, seven volatile GPRs,
six volatile XMM values, all remaining register values, caller address, SP,
native site, and pre-effect full storage/history. Completed services retain
explicit supplied writes/returns and normal callee Win64 preservation evidence;
normal outer returns also verify eight integer nonvolatiles, XMM6–15, and SP.

The verifier uses a separate byte-addressed Python model of the pinned x64
instruction subset. It independently evaluates effective addresses, operand
widths, upper-byte retention, DWORD zero extension, signed branches, call-stack
writes, and native stores from the initial complete bytes and entry registers.
It consumes no observed native trace, register, write, or final value. Supplied
contracts are authored fixture inputs shared by both executors; this comparison
does not prove their unexecuted implementation. Additional semantic assertions
check reset/preservation, capture/reload, callback ordering, reference-array
prefixes, nullable guarded paths, and ID behavior.

The write hook records successfully mapped graph writes and rejects mapped
writes outside the retained graph. The independent exact model then compares
every reached native write, all event snapshots/ABI, and final state. This is not
a direct per-instruction write allowlist, and no changed-byte allowance copies a
native final value into an expected state. Unmapped reads/writes retain their
exact native instruction, address, size, fault type, partial bytes, and registers.
Guard services and injected stops retain their full pre-effect states. A null
Array.Clear backing-array fixture stops at the explicitly supplied boundary;
the callee's native exception behavior is outside scope.

Retained calls preserve all game records and chronological supplied/native-write/
callback history. Each new invocation authors a fresh external native-stack call
frame and register inputs. Adjacent entire snapshots therefore do not claim
stack equality; continuity checks compare every other physical record and all
retained histories. Service ordinals reset per invocation. Qualified callbacks
require kind, that invocation's ordinal, and the verified native caller site;
a wrong-site plan is tested as inert. Recovery starts a new call after a stopped
callback without rollback or effects from the stopped supplied service.

The complete report uses six lossless codecs. Encoding applies register/ABI-map
interning, raw-memory interning, complete-history interning, complete memory-map
interning, state-map interning, then snapshot interning. The exported
`expand_report` expands the prior five layers before register maps.
`expand_memory` is the integration adapter for input already passed through
`expand_snapshots`; it expands the remaining prior four layers, then register
maps. Ordered lists, nullable values, full u64/128-bit register values, and every
raw byte are preserved. Producer-local tests check full round trips, adapter
equivalence, deep-copy independence, ordering, malformed/missing references, and
hash corruption across all six layers. The existing 36 RE infrastructure tests
are separate and do not imply coverage of this family-local codec.

Final evidence contains 211 cases (181 normal returns and 30 guard/fault/supplied
stops), seven retained sequences, eight baselines, and 248 full stopped prefixes.
All 367 non-trap instructions execute; every one of 484 complete rows is
independently modeled over 74 retained physical records. Both independent final
processes were preceded by successful Python syntax compilation and produced
identical bytes. A fresh serialized replay independently re-evaluated all 484
rows, every prefix and stopped register frame, each physical-window size,
retained non-stack/history continuity, and the complete decoder/adapter. It also
verified that every corpus/evidence blob matches the superseded private report;
the correction changed only the tracked instruction evidence scope.
The 36 RE infrastructure tests also passed, as did diff and owned-text checks.

The report is `reports/f530404b0f3f_807de4a83df4_character_initializers.json`, schema
`character_initializers_native_v1`, 50,781,511 bytes. Its SHA-256 is
`096123de55dfa556c015d4f114b19a472c0f47a8ddaed6eb4acaea5b9b06b2f0`.
The frozen source SHA-256 is
`b4b4dd63ce49288d4a3b3e942a5680fda69dbf7b794873412a4be6beff0bc82c`.
The private independent producer files are
`character_initializers.corrected.first.private.json` and
`character_initializers.corrected.peer.private.json` beneath the pinned build
artifact directory. Reproduce with Unicorn 2.1.4 and the private emulation PYTHONPATH:

```powershell
python -m py_compile reverse_engineering/scripts/audit_character_initializers.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python reverse_engineering/scripts/audit_character_initializers.py GAME_ROOT DUMPER_ROOT --output REPORT_PATH
```

Arbitrary callbacks/reentrancy, concurrent list races, exception unwinding, and
all whole supplied implementation bodies remain outside this bounded evidence.
