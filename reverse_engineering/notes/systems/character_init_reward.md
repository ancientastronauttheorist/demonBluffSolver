# Character.InitReward native caller

Build: `f530404b0f3f_807de4a83df4`.

Producer: `reverse_engineering/scripts/audit_character_init_reward.py`.
Report: `reverse_engineering/reports/f530404b0f3f_807de4a83df4_character_init_reward.json`.

The audit executes the complete actual Character.InitReward body with explicit
whole Unity, Action callback, reference-barrier and Character.RevealReal services.
It owns an independent diagnostic graph and reuses the frozen DeckCharacter
surface machine's immutable input pins, complete decoder, mapped runtime and
volatile return poisoning. No Character data, art or view worker files change.

The compact JSON report uses `sha256-full-authored-snapshot-v1` from
`reverse_engineering/scripts/audit_report_snapshots.py`. Every full initial,
final and service-entry snapshot is interned by SHA-256. `expand_snapshots`
checks each blob hash and reconstructs independent full values; the producer
asserts exact expanded-value equality after all execution, retention and prefix
checks. The `audit()` function returns the full expanded report. No snapshots,
fields, diagnostic bytes or prefixes are dropped.

## Exact entry, storage and enum pins

`tdi5487.m0027` is exact `Character::public void InitReward(CharacterData character)`
at `365640..365712` (exclusive end): 210 bytes and 50 decoded instructions.
It has one matching unwind range and the next verified managed entry is
`365720`. Full native raw backing, decoder consumption and trailing `CC`
alignment are asserted. The Dumper signature is:

```text
void Character__InitReward (Character_o* __this, CharacterData_o* character, const MethodInfo* method);
```

Character is TypeDefIndex 5487. Every consumed offset is pinned to its exact
field declaration:

| Offset | Declaration | Native width |
| --- | --- | --- |
| `50` | `CharacterData dataRef` | pointer |
| `58` | `CharacterData bluff` | pointer |
| `60` | `CharacterData registerAs` | pointer |
| `A8` | `Acted acteds` | pointer |
| `DC` | private `int pickableUses` | DWORD |
| `E0` | `ECharacterState prevState` | DWORD |
| `E4` | `ECharacterState state` | DWORD |
| `F8` | `EAlignment alignment` | DWORD |
| `180` | `Action onStateChange` | pointer |

CharacterData is 5845, with `EAlignment startingAlignment +134` consumed as a
DWORD. Exact ECharacterState 5489 constants are None 0, Hidden 5, Alive 10,
Dead 20, Revealed 30. EAlignment 5492 constants are None 0, Good 10, Evil 20.
Additional high-bit DWORD fixtures are storage/operand-width probes and make
no legal managed enum admission claim.

The callback uses the exact Delegate 419 `invoke_impl +18`, IntPtr `method +28`,
and IntPtr `method_code +40` declarations already pinned by the frozen surface
machine. Its unused managed `m_target +20` has a separate token. `method_code`
is not decoded as that managed target. Class/header and unused-field buffers
are diagnostic; they do not establish complete object sizes or CLR admission.

## Actual chronology and consumed services

InitReward captures its input CharacterData into native RDI at `36564D`, then
loads owner.acteds once at `365650`. A null Acted reaches the native null
gateway immediately. Supplied whole `UnityEngine.Component.get_gameObject`
at `1C79FD0` receives the captured Acted in RCX and exact zero MethodInfo in RDX.
Its returned GameObject is guarded for null and passed directly to supplied
whole `UnityEngine.GameObject.SetActive` at `1C7D810`. There is no later Acted
reload within this invocation. SetActive receives full-width RDX zero (Boolean
false) and full-width R8 zero (MethodInfo); R9 is only diagnostic register bits.

The supplied getter's authored mapping returns GameObject0 for Acted0 and
GameObject1 for Acted1, with independent alternate or null outputs. Getter
mutations can replace or clear current owner.acteds without changing the
already captured receiver or returned GameObject. SetActive authors a separate
active-state log; it does not infer an undocumented Unity native-memory flag.

After successful SetActive, native code clears bluff, stores the original
captured CharacterData into dataRef, and clears registerAs. Each actual pointer
store is followed by a supplied reference barrier, in that exact order. It
then writes pickableUses = 1. A null input CharacterData reaches the native
null gateway only now, retaining all three pointer stores, their completed
barriers, the deactivation request and pickableUses = 1. Alignment, previous
state, current state and the state callback remain untouched on this branch.

For nonnull input, native code reads its current startingAlignment DWORD and
copies it to owner.alignment. The input identity remains the one captured at
entry; it does not reload owner.dataRef. A third-barrier mutation of that
captured data's alignment demonstrates the late field read, including when
input and original owner.dataRef alias.

It next reads current owner.state into prevState, loads current onStateChange,
and writes state = Hidden (5). A third-barrier state mutation therefore becomes
the new prevState. Callback replacement or removal at that same barrier is
observed by the later channel load.

When the callback is nonnull, its actual indirect invocation has
`RCX=Action.method_code` and `RDX=Action.method`. R8 and R9 remain explicit
volatile poison, not extra callback arguments. The complete callback-entry
snapshot already has copied alignment, copied previous state, Hidden current
state and pickableUses = 1. Independent load-phase oracles verify the original
source DWORDs at their native instruction addresses rather than reconstructing
them from a callback's later changes.

Finally the body restores Win64 nonvolatile registers and tail calls whole
supplied `Character.RevealReal` at `3682A0` with the owner in RCX and exact zero
MethodInfo in RDX. The Dumper entry/signature is verified; its native body is
not executed or promoted here. Callback mutations of dataRef, alignment,
previous/current state or pickableUses remain visible at this tail boundary.
No downstream behavior is inferred from those mutations.

Every supplied return poisons volatile integer and XMM registers. All pointer,
MethodInfo and Boolean inputs are asserted at their consumed full widths. Each
supplied-entry event, including a stopped entry, records full raw RCX/RDX/R8/R9,
its cumulative service ordinal, and the full caller return address with its
decoded native instruction (or the authored original caller stop for a tail
entry). These raw values are independently checked for consumed and untouched
registers, including all three different null-guard phases. The unused input
MethodInfo token, unused diagnostic headers and null method_code callback probe
are not scene or CLR admission claims.

## Corpus, retention and exact stopped prefixes

The corpus has 72 individual cases: 52 normal returns and 20 reached native
null gateways. It includes null Acted, null returned GameObject, null input
CharacterData, nullable state callback, alternate returned GameObject,
all declared state/alignment constants, high-bit DWORDs, preexisting nullable
dataRef/bluff/registerAs, aliased input data and null callback method_code.

Authored mutations replace or clear Acted, dataRef and the state callback;
modify the captured input alignment; and modify current/previous state,
alignment or pickableUses. Timing covers getter, SetActive, each barrier,
callback and supplied RevealReal boundaries. They verify early pointer capture
against late alignment/state/channel loads and full callback-entry state.

An independent ordered semantic model starts solely from each authored initial
snapshot, options and prior service counts. It does not read current native
memory or use the native phase-oracle values. It models captured input/Acted,
getter outputs, explicit service effects/mutations, native pointer stores before
barriers, late alignment/current-state/channel reads, callback mutations and
all partial guards/stops. Exact byte patches preserve the complete initial
diagnostic graph outside these effects. Every service-entry event must equal
the modeled kind, ordinal, arguments, full ABI and entire snapshot; the final
snapshot and return disposition must also match. Every case, retained call,
baseline and stopped prefix carries `independent_ordered_model_verified` only
after this complete comparison succeeds. The native load-phase oracles remain
as a separate check of the actual instructions.

Each reached native store grants only its exact eight-byte pointer or four-byte
DWORD field range. Reached authored mutations grant only their declared field
ranges. All remaining owner, data, Acted, GameObject, Action, class, method-token
and managed-target diagnostic bytes remain identical. DWORD stores preserve
the adjacent sentinel bytes. Completed supplied SetActive, callback and RevealReal
logs are checked independently against completed service events; failed entries
cannot author those effects.

Three retained three-call sequences keep one graph across invocations. Two
cover distinct/aliased input identities; the third replaces owner.acteds during
the first getter. The first call still deactivates the captured GameObject0,
while later invocations load current Acted1 and deactivate GameObject1. Original
input identity, field updates, active state, callback logs and native phase
records persist. Each next initial snapshot equals the previous final snapshot.

Four successful baselines generate 28 stopped prefixes, including Acted
replacement, late state replacement and callback alignment mutation. Each stop
occurs before the selected supplied effect or mutation. Its complete event list
equals the baseline prefix, and its full final snapshot equals that service's
baseline entry snapshot. Native stores reached before a barrier and all earlier
completed effects persist; no managed unwind or later continuation is invented.

There are 20 selected exact instruction assertions plus complete call-site
counts. All 49 nontrap body instructions execute, across 55 total native/service
addresses. The one remaining decoded body instruction is terminal `int3` at
`365711`, after the stopped native null gateway; its exact identity is asserted.
Normal returns verify the stack, eight integer nonvolatile registers and ten
nonvolatile XMM registers.

## Remaining scope

Whole Component.get_gameObject, GameObject.SetActive, onStateChange callback,
Character.RevealReal and GC barrier implementations remain supplied. This
caller audit does not prove Unity or CLR object admission, rendering, real event
dispatch, scheduling, runtime initialization or managed exception behavior.
It does not promote a constructor, shared alias or any downstream callee.
