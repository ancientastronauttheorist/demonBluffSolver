# Character history and register-as type entry callers

Pinned build `f530404b0f3f_807de4a83df4`. Four complete native Character
callers execute in 116 cases, two retained five-call sequences, one supplied
liveness callback probe and 12 exact stopped prefixes. Sixteen instruction
assertions pin their fields, widths and helper entries. The corpus reaches
118 of 124 caller instructions; the six remaining instructions are trap bytes
after native guards. Actual List.get_Item and last-element List.RemoveAt slices
reach 49 helper instructions. Native bytes/bodies remain private.

| Caller | Method ID | Entry |
| --- | --- | --- |
| AddOnHoverInfo | tdi5487.m0006 | 0x3647D0 |
| GetCharacterType | tdi5487.m0011 | 0x364DE0 |
| GetCurrentActedInfo | tdi5487.m0057 | 0x364E60 |
| ClearRecentMemory | tdi5487.m0070 | 0x3649A0 |

Exact ScriptMethod signatures and List<ActedInfo> MethodInfo slots are checked.
No-info callers receive null MethodInfo in RDX; AddOnHoverInfo receives its
nullable ActedInfo argument there. Successful calls verify stack balance, all
eight integer nonvolatiles and XMM6–15. The supplied service returns do not
poison volatile registers, and no broader runtime ABI behavior is inferred.

AddOnHoverInfo consumes onHoverInfo at Character+0x150. It increments the
List version before checking backing storage or deciding whether growth is
needed. The inline branch publishes count and the same nullable info pointer
before its barrier. Growth remains a supplied bounded service; this fixture's
authored arena capacity update is not evidence for CLR array resizing.

ClearRecentMemory consumes actedInfos at +0x148. A signed count <=0 returns
without removal or a version change. Positive count tail-forwards exactly
count-1 to actual List.RemoveAt at 0xB59CE0. This reached helper slice decrements
count, clears the last reference slot, requests its barrier and only then
increments version. A stopped barrier therefore retains reduced count and
cleared slot with the old version. Non-last removal and Array.Copy are not
exercised or supplied as successful paths.

GetCurrentActedInfo always forwards count-1 to actual List.get_Item at
0xB22150. An empty List reaches the native unsigned index guard; it does not
return a default info object. A valid last slot may contain null, in which
case null is returned. Null storage and bounds failures remain stopped runtime
gateways without rollback or managed exception unwinding.

The history and hover fields can reference the same physical List. Both
five-call sequences retain the objects, versions and all eight authored
reference slots. Each normal call checks a complete before/after storage
oracle, including unused values. The eight-slot memory projection includes
authored diagnostic tail storage beyond capacity in capacity-two cases; it
does not turn those tail bytes into valid managed Array elements.

GetCharacterType obtains registerAs at +0x60, supplies it to Unity null/equality,
and chooses its type DWORD at CharacterData+0x130 when live. Absent or destroyed
registerAs falls back to dataRef+0x50. It rereads the selected Character field
after the check and returns a zero-extended DWORD, including high-bit enum
patterns. A live registered role works even when real data is null. An authored
liveness callback clearing live registerAs reaches the later null guard; no
fallback is fabricated after that mutation.

Metadata/class initialization, liveness and its callback effect, GC barriers,
growth and exception gateways remain supplied services. Inert normal cases
retain every actor byte; the one callback modifies only registerAs. Stopped
service snapshots and event prefixes exactly match successful baselines.
No real object lifetime, live-game change or managed exception unwind is claimed.

Independent repo/private reports are byte-identical: 828,432 bytes, SHA-256
`13d1a20739e1b8d39180cb874cbb86c1c5548c803890eebef97d4d1ba5dc91ad`.

```powershell
$env:PYTHONPATH='B:/CodexTools/DemonBluffReverseEngineering/python-emulation'
python reverse_engineering/scripts/audit_character_history_entries.py GAME_ROOT DUMPER_ROOT --output REPORT
```

Report: `f530404b0f3f_807de4a83df4_character_history_entries.json`.
