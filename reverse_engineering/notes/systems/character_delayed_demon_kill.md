# Captured delayed Demon kill: native factory and controlled resumes

Pinned build `f530404b0f3f_807de4a83df4`. The offline audit executes the actual
`Character.DelayedDemonKill` factory at `0x364A90` and complete
`Character.<DelayedDemonKill>d__103.MoveNext`, method `tdi5483.m0002`, at
`0x3757F0`. Calls are manually ordered against the same produced physical
iterator. This establishes the consumer's behavior after an explicit resume;
it does not establish when Unity schedules it.

## Native producer and first yield

The native factory allocates the exact iterator type, runs the folded empty
base body, captures its Character receiver at +0x20, sets state zero at +0x10,
performs the owner barrier, then captures the optional evil Character at +0x28
and performs its barrier. Null owners and null/self evil arguments are legal
captures. The first resume does not dereference the owner.

State zero first becomes -1. MoveNext allocates a physical WaitForSeconds and
executes its actual constructor at `0x1C961F0`. That constructor preserves
XMM6 around the verified folded empty base call and stores the original float
at `m_Seconds` +0x10. The exact constant is float32 bits `0x3EE66666` (0.45f).
MoveNext publishes the wait as current before its barrier, sets iterator state
one, and returns AL=1. The constructor's void return is not treated as an
object pointer.

## Second resume and death ordering

State one first becomes -1. A null owner reaches the native null gateway; a
Dead (20) Character returns AL=0 without querying its role. Otherwise the code
loads the current `dataRef.role`, calls its virtual CheckIfCanBeKilled, and
tests AL. The joined base predicate at `0x3B24C0` is exactly `mov al,1; ret`.
Other predicates in the fixture are explicit supplied services, including a
false result with poisoned upper return bits.

A successful predicate causes these effects in order:

1. Reload the current Character state into prevState; write killedByDemon=true
   and state=Dead; invoke optional onStateChange.
2. Reload Character.statuses for AddStatus(MessedUpByEvil=50), then reload it
   again for AddStatus(KilledByEvil=55). Each call rereads the iterator's evilRef.
3. Invoke optional UIEvents.OnUIUpdate.
4. Execute actual Character.Act(ETriggerPhase.OnDied=50), including its trigger
   logging, optional trigger callback, native CheckLying and RoleAct dispatch.
5. Reload current dataRef.role and call its virtual ActOnDied. The joined base
   body is the verified folded empty return at `0x33ED50`.
6. Invoke optional GameplayEvents.OnCharacterKilled with the Character.
7. Execute actual UpdateUI at `0x369470`, copying the current Gameplay.CurrentReveal
   DWORD to Character.order. Return AL=0.

Current is retained after completion. Later resumes with state -1 return AL=0
without clearing the wait or repeating effects. Other supplied iterator states
also return AL=0 unchanged. An explicit state-one fixture can run the death
path while retaining an existing current object, without allocating a wait.

## Status source ABI and stored target

The source and target arguments are distinct. At both native AddStatus entries,
R8 contains the captured evilRef, R9 is zero, and the stack MethodInfo argument
is zero. Public events record those arguments independently of the resulting
status snapshot. `CharacterStatuses.AddStatus` at `0x363AA0` does not consume
sourceRef in this body. It checks the resistance List, checks the active List,
appends only when absent, then writes its **targetRef (null)** into the shared
targetCharacter slot before its barrier. An accepted duplicate also writes
null. A resisted status does not write that slot; resisting both retains the
old target. The evil capture therefore is not a stored target in this path.

List membership and bounded append are declared services over explicit List
objects/backing storage; the AddStatus control flow and target store/barrier
are native. Version increments wrap as a DWORD. Physical aliasing of the active
and resistance Lists is exercised without replacing their identities.

## Native action join and callback boundaries

The native Act, CheckLying and RoleAct bodies execute. HealthyBluff (30) clears
lying before Corrupted (10) can restore it; supplied Unity liveness and runtime
alignment also participate. A lying runtime-Evil Character with a copied role
uses its real Act route and the copy's BluffAct route. Other lying characters
use BluffAct for both; truthful characters use Act for both. If the copy is the
same physical Role, native RoleAct allocates and publishes two distinct
callback closures/delegates in order, replacing that Role's onActed twice.
Concrete role actions and multicast invocation behavior are not implemented.

Authored callbacks mutate state, status containers, dataRef, the evil capture,
and CurrentReveal at named service points. Tests verify native reloads and
partial effects, including an append service replacing/clearing the status
container between the two AddStatus calls. The first AddStatus retains its
captured container for its target write, while the second call loads the new
container. A callback that clears a required pointer stops at the corresponding
native guard; prior death flags, accepted statuses and barriers remain visible.

## Verification and exclusions

The normal corpus contains 1,728 combinations of initial state, warm/cold
metadata, null/self/distinct evil captures, base/supplied kill predicate,
copied-role identity, active statuses and resistances. Another 53 policy,
physical List alias, raw nonzero metadata-byte and DWORD-boundary profiles,
six retained/alternate iterator-state profiles, twelve callback mutation
profiles and seven null-guard profiles exercise the boundaries. Three baselines
inject 149 stops at every reached phase-qualified native entry or supplied service.
Each stop compares the exact event/snapshot prefix and all 0x1B8 Character bytes.
Normal paths independently verify the full Actor projection and untouched bytes,
factory captures, first-yield current/wait identity, List version/storage,
role routing and terminal retention.

Every successful factory/resume checks stack balance, all eight integer
nonvolatile registers and XMM6–15. Authored services poison integer and XMM
caller-saved registers, and byte returns carry nonzero upper bits. Twenty-three
entry instruction relationships, the two folded leaves, fourteen inherited
factory relationships, exact signatures and field/type/enum declarations bind
the fixture to this build. Only the OnDied paths of the joined Act/CheckLying/
RoleAct helpers are exercised here; their unrelated caller coverage is not
broadened by this audit.

The joined target bodies execute 347 of 380 decoded instructions, with 466
native instruction addresses reached across the factory and helper joins.
The remaining helper branches and null-guard trap bytes are not promoted to
coverage by the aggregate instruction count.

Metadata/class initialization, zeroed allocation, GC barriers, delegate
construction, List membership/append, Unity liveness, trigger box/format/log,
concrete role policy and callback/subscriber effects remain explicit authored
services. CheckLying's call to CharacterStatuses.Contains at `0x363C40` is
supplied as the whole wrapper, rather than executing that wrapper's body in
this audit; its actual native wrapper is audited separately in the tutorial
death generators. There is no actual scheduler readiness, HP/score/dead-roster policy,
live game/process access, real Unity lifetime, rollback or managed exception
unwinding. The caller does not itself perform Kill, RevealAllReal, onReveal or
HP updates; subscribers or role overrides can add effects outside this audit.
Private bytes and bodies remain off-repository.

Independent native report producers succeeded. Their separately parsed outputs
were re-emitted with compact normal-case rows; complete parsed schema and
projections are unchanged and both final files are byte-identical: 33,558,291
bytes, SHA-256
`249b22f3f5b4d18f468ea360555c4264b82e381105f05e8e847a85a0ff8104c8`.
Python compilation, all 32 reverse-engineering tests and `git diff --check`
pass.

```powershell
$env:PYTHONPATH='B:/CodexTools/DemonBluffReverseEngineering/python-emulation'
python reverse_engineering/scripts/audit_character_delayed_demon_kill.py GAME_ROOT DUMPER_ROOT --output REPORT
```

Report: `f530404b0f3f_807de4a83df4_character_delayed_demon_kill.json`.
